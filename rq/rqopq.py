#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FAISS OPQ + ResidualQuantizer SID construction for MiniOneRec.

The output index JSON is compatible with the current MiniOneRec data converter:
{
  "0": ["<a_12>", "<b_34>", "<c_56>"],
  ...
}
"""

import argparse
import json
import os
from typing import Iterable, List, Union

import numpy as np


TOKEN_TEMPLATES = [
    "<a_{}>",
    "<b_{}>",
    "<c_{}>",
    "<d_{}>",
    "<e_{}>",
    "<f_{}>",
    "<g_{}>",
    "<h_{}>",
]


def require_faiss():
    try:
        import faiss  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "rqopq.py requires faiss. Install faiss-gpu or faiss-cpu in the server environment."
        ) from exc
    return faiss


def validate_codebook_size(codebook_size: int) -> int:
    if codebook_size <= 1 or codebook_size & (codebook_size - 1):
        raise ValueError(f"codebook_size must be a power of two, got {codebook_size}")
    return int(np.log2(codebook_size))


def resolve_opq_m(dim: int, opq_m: Union[str, int]) -> int:
    if str(opq_m).lower() != "auto":
        value = int(opq_m)
        if value <= 0 or dim % value != 0:
            raise ValueError(f"opq_m={value} must be a positive divisor of embedding dim {dim}")
        return value

    preferred = [32, 64, 16, 8, 4, 2, 1]
    for value in preferred:
        if dim % value == 0:
            return value
    return 1


def format_sid_tokens(code: Iterable[int]) -> List[str]:
    code = list(code)
    if len(code) > len(TOKEN_TEMPLATES):
        raise ValueError(f"SID has {len(code)} levels but only {len(TOKEN_TEMPLATES)} token prefixes are defined")
    return [TOKEN_TEMPLATES[level].format(int(value)) for level, value in enumerate(code)]


def analyze_codes(codes: np.ndarray, title: str = "", verbose: bool = True) -> dict:
    if codes.ndim != 2:
        raise ValueError(f"codes must be a 2D array, got shape={codes.shape}")

    total, num_levels = codes.shape
    unique_per_level = [int(len(np.unique(codes[:, level]))) for level in range(num_levels)]
    unique_full_paths = int(len(set(map(tuple, codes.tolist()))))
    collision_rate = 0.0 if total == 0 else 1.0 - unique_full_paths / total
    stats = {
        "total": int(total),
        "num_levels": int(num_levels),
        "unique_per_level": unique_per_level,
        "unique_full_paths": unique_full_paths,
        "collision_rate": float(collision_rate),
    }

    if verbose:
        if title:
            print(title)
        print(f"  total={stats['total']}")
        for level, unique in enumerate(unique_per_level):
            print(f"  L{level + 1}: unique={unique}")
        print(f"  unique full-paths={unique_full_paths}  collision_rate={collision_rate:.6f}")
    return stats


def unpack_rq_codes(codes_packed: np.ndarray, nbits: int, num_levels: int) -> np.ndarray:
    if codes_packed.ndim == 1:
        n_bytes = (num_levels * nbits + 7) // 8
        codes_packed = codes_packed.reshape(-1, n_bytes)

    total = codes_packed.shape[0]
    packed_ints = np.zeros(total, dtype=np.int64)
    for byte_idx in range(codes_packed.shape[1]):
        packed_ints |= codes_packed[:, byte_idx].astype(np.int64) << (8 * byte_idx)

    mask = (1 << nbits) - 1
    codes = np.zeros((total, num_levels), dtype=np.int32)
    for level in range(num_levels):
        codes[:, level] = (packed_ints >> (level * nbits)) & mask
    return codes


def encode_with_rq(rq, data: np.ndarray, codebook_size: int, verbose: bool = True) -> np.ndarray:
    nbits = validate_codebook_size(codebook_size)
    data = np.ascontiguousarray(data.astype(np.float32))
    if verbose:
        print(f"Encoding {data.shape[0]} vectors with ResidualQuantizer ...")
    codes_packed = rq.compute_codes(data)
    if nbits % 8 == 0:
        codes = codes_packed.astype(np.int32)
    else:
        codes = unpack_rq_codes(codes_packed, nbits, rq.M)
    codes = codes.reshape(data.shape[0], rq.M).astype(np.int32)
    if verbose:
        print(f"  done, codes.shape={codes.shape}")
    return codes


def save_indices_json(codes: np.ndarray, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    indices = {str(item_id): format_sid_tokens(code) for item_id, code in enumerate(codes)}
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(indices, fh, indent=2)
    print("Saved SID index:", output_path)


def save_stats(stats: dict, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    print("Saved stats:", output_path)


def maybe_sample_training_data(data: np.ndarray, train_sample: int, seed: int) -> np.ndarray:
    if train_sample <= 0 or train_sample >= data.shape[0]:
        return data
    rng = np.random.RandomState(seed)
    indices = rng.choice(data.shape[0], size=train_sample, replace=False)
    return data[indices]


def train_opq(data: np.ndarray, opq_m: int, niter: int, niter_pq: int, verbose: bool):
    faiss = require_faiss()
    dim = data.shape[1]
    opq = faiss.OPQMatrix(dim, opq_m)
    opq.niter = niter
    opq.niter_pq = niter_pq
    opq.verbose = verbose
    if verbose:
        print("Training OPQ")
        print(f"  dim={dim}  opq_m={opq_m}  niter={niter}  niter_pq={niter_pq}")
    opq.train(np.ascontiguousarray(data.astype(np.float32)))
    return opq


def train_residual_quantizer(data: np.ndarray, num_levels: int, codebook_size: int, max_beam_size: int, verbose: bool):
    faiss = require_faiss()
    dim = data.shape[1]
    nbits = validate_codebook_size(codebook_size)
    rq = faiss.ResidualQuantizer(dim, num_levels, nbits)
    rq.train_type = faiss.ResidualQuantizer.Train_default
    rq.max_beam_size = max_beam_size
    if verbose:
        print("Training ResidualQuantizer on OPQ-transformed vectors")
        print(f"  dim={dim}  levels={num_levels}  codebook_size={codebook_size}  nbits={nbits}")
    rq.train(np.ascontiguousarray(data.astype(np.float32)))
    return rq


def write_faiss_artifacts(opq, rq, opq_path: str, rq_path: str) -> None:
    faiss = require_faiss()
    try:
        faiss.write_VectorTransform(opq, opq_path)
        print("Saved OPQ transform:", opq_path)
    except Exception as exc:
        print("Saving OPQ transform failed:", exc)

    try:
        nbits = get_first_nbits(rq)
        index = faiss.IndexResidualQuantizer(rq.d, rq.M, nbits)
        index.rq = rq
        index.is_trained = True
        faiss.write_index(index, rq_path)
        print("Saved RQ quantizer:", rq_path)
    except Exception as exc:
        print("Saving RQ quantizer failed:", exc)


def get_first_nbits(rq) -> int:
    faiss = require_faiss()
    if isinstance(rq.nbits, int):
        return int(rq.nbits)
    return int(faiss.vector_to_array(rq.nbits).ravel()[0])


def default_data_path(dataset: str) -> str:
    return os.path.join("data", "Amazon", "index", f"{dataset}.emb-qwen-td.npy")


def build_rqopq_sids(args) -> dict:
    np.random.seed(args.seed)

    data_path = args.data_path or default_data_path(args.dataset)
    output_dir = args.output_dir or os.path.dirname(data_path)
    os.makedirs(output_dir, exist_ok=True)

    output_prefix = args.output_prefix or os.path.join(output_dir, f"{args.dataset}.rqopq")
    out_json = f"{output_prefix}.index.json"
    out_stats = f"{output_prefix}.stats.json"
    out_opq = f"{output_prefix}.opq"
    out_rq = f"{output_prefix}.faiss"

    print("Loading embeddings:", data_path)
    data = np.load(data_path).astype(np.float32)
    if data.ndim != 2:
        raise ValueError(f"embedding file must contain a 2D array, got shape={data.shape}")
    data = np.ascontiguousarray(data)
    print("Embedding shape:", data.shape)

    nbits = validate_codebook_size(args.codebook_size)
    opq_m = resolve_opq_m(data.shape[1], args.opq_m)
    train_data = maybe_sample_training_data(data, args.train_sample, args.seed)
    print("Training vectors:", train_data.shape)

    opq = train_opq(
        train_data,
        opq_m=opq_m,
        niter=args.opq_niter,
        niter_pq=args.opq_niter_pq,
        verbose=not args.quiet,
    )
    transformed = opq.apply_py(data)
    transformed = np.ascontiguousarray(transformed.astype(np.float32))
    train_transformed = opq.apply_py(train_data)
    train_transformed = np.ascontiguousarray(train_transformed.astype(np.float32))

    rq = train_residual_quantizer(
        train_transformed,
        num_levels=args.num_levels,
        codebook_size=args.codebook_size,
        max_beam_size=args.max_beam_size,
        verbose=not args.quiet,
    )
    codes = encode_with_rq(rq, transformed, args.codebook_size, verbose=not args.quiet)
    code_stats = analyze_codes(codes, "RQ-OPQ SID statistics:", verbose=True)

    save_indices_json(codes, out_json)
    write_faiss_artifacts(opq, rq, out_opq, out_rq)

    stats = {
        "method": "rqopq",
        "dataset": args.dataset,
        "data_path": data_path,
        "data_shape": list(data.shape),
        "train_shape": list(train_data.shape),
        "num_levels": args.num_levels,
        "codebook_size": args.codebook_size,
        "nbits": nbits,
        "opq_m": opq_m,
        "opq_niter": args.opq_niter,
        "opq_niter_pq": args.opq_niter_pq,
        "max_beam_size": args.max_beam_size,
        "seed": args.seed,
        "output_index": out_json,
        "output_opq": out_opq,
        "output_rq": out_rq,
        "codes": code_stats,
    }
    save_stats(stats, out_stats)
    return stats


def parse_args():
    parser = argparse.ArgumentParser(description="Build MiniOneRec SIDs with FAISS OPQ + ResidualQuantizer.")
    parser.add_argument("--dataset", default="Industrial_and_Scientific")
    parser.add_argument("--data_path", default=None, help="Path to item embedding .npy file.")
    parser.add_argument("--output_dir", default=None, help="Directory for RQ-OPQ artifacts. Defaults to data_path dir.")
    parser.add_argument("--output_prefix", default=None, help="Full artifact prefix, without suffix.")
    parser.add_argument("--num_levels", type=int, default=3)
    parser.add_argument("--codebook_size", type=int, default=256)
    parser.add_argument("--opq_m", default="auto", help="'auto' or a positive divisor of embedding dim.")
    parser.add_argument("--opq_niter", type=int, default=50)
    parser.add_argument("--opq_niter_pq", type=int, default=4)
    parser.add_argument("--max_beam_size", type=int, default=1)
    parser.add_argument("--train_sample", type=int, default=-1, help="Optional OPQ/RQ training sample size; <=0 uses all.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    build_rqopq_sids(args)


if __name__ == "__main__":
    main()
