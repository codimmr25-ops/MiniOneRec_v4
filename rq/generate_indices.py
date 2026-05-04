import argparse
import collections
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from torch.utils.data import DataLoader

from datasets import EmbDataset
from models.rqvae import RQVAE

def check_collision(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    return tot_item==tot_indice

def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count

def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)

    collision_item_groups = []

    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])

    return collision_item_groups

def parse_args():
    parser = argparse.ArgumentParser(description="Generate MiniOneRec SID index from an RQ-VAE checkpoint.")
    parser.add_argument("--ckpt_path", required=True, help="Path to best_collision_model.pth or another RQ-VAE checkpoint.")
    parser.add_argument("--data_path", default=None, help="Embedding .npy path. Defaults to the path stored in checkpoint args.")
    parser.add_argument("--output_path", required=True, help="Destination .index.json path.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--max_dedup_rounds", type=int, default=20)
    return parser.parse_args()


def build_model(args, data_dim):
    return RQVAE(
        in_dim=data_dim,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        layers=args.layers,
        dropout_prob=args.dropout_prob,
        bn=args.bn,
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        sk_epsilons=args.sk_epsilons,
        sk_iters=args.sk_iters,
    )


def main():
    cli_args = parse_args()
    device = torch.device(cli_args.device)

    ckpt = torch.load(
        cli_args.ckpt_path,
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    train_args = ckpt["args"]
    state_dict = ckpt["state_dict"]
    if cli_args.data_path:
        train_args.data_path = cli_args.data_path

    data = EmbDataset(train_args.data_path)
    model = build_model(train_args, data.dim)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(model)

    num_workers = train_args.num_workers if cli_args.num_workers is None else cli_args.num_workers
    data_loader = DataLoader(
        data,
        num_workers=num_workers,
        batch_size=cli_args.batch_size,
        shuffle=False,
        pin_memory=True,
    )

    all_indices = []
    all_indices_str = []
    prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]

    for d in tqdm(data_loader):
        d = d.to(device)
        indices = model.get_indices(d, use_sk=False)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for index in indices:
            code = []
            for i, ind in enumerate(index):
                code.append(prefix[i].format(int(ind)))

            all_indices.append(code)
            all_indices_str.append(str(code))

    all_indices = np.array(all_indices)
    all_indices_str = np.array(all_indices_str)

    for vq in model.rq.vq_layers[:-1]:
        vq.sk_epsilon = 0.0
    if model.rq.vq_layers[-1].sk_epsilon == 0.0:
        model.rq.vq_layers[-1].sk_epsilon = 0.003

    # There are often duplicate items in the dataset; retry only the collided groups.
    tt = 0
    while True:
        if tt >= cli_args.max_dedup_rounds or check_collision(all_indices_str):
            break

        collision_item_groups = get_collision_item(all_indices_str)
        print("Collision groups:", len(collision_item_groups))
        for collision_items in collision_item_groups:
            d = data[collision_items].to(device)

            indices = model.get_indices(d, use_sk=True)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for item, index in zip(collision_items, indices):
                code = []
                for i, ind in enumerate(index):
                    code.append(prefix[i].format(int(ind)))

                all_indices[item] = code
                all_indices_str[item] = str(code)
        tt += 1

    print("All indices number:", len(all_indices))
    print("Max number of conflicts:", max(get_indices_count(all_indices_str).values()))

    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    print("Collision Rate", (tot_item - tot_indice) / tot_item)

    all_indices_dict = {}
    for item, indices in enumerate(all_indices.tolist()):
        all_indices_dict[str(item)] = list(indices)

    os.makedirs(os.path.dirname(os.path.abspath(cli_args.output_path)), exist_ok=True)
    with open(cli_args.output_path, "w", encoding="utf-8") as fp:
        json.dump(all_indices_dict, fp, indent=2)
    print("Saved SID index:", cli_args.output_path)


if __name__ == "__main__":
    main()
