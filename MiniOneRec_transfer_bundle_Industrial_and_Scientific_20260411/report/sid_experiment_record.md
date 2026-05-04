# MiniOneRec SID Final Experiment Record

## 1. Experiment Scope

- Repository: `MiniOneRec`
- Dataset: `Industrial_and_Scientific`
- Goal: compare 4 SID construction methods under the same item set and prepare a minimal transfer bundle for follow-up `SFT / RL / Eval` on another machine.
- Bundle policy: keep final SID outputs and converted downstream data; exclude large embeddings, raw codebooks, FAISS binary index, and training checkpoints.

## 2. Final Comparison

### 2.1 Unified Metrics

All metrics below were recomputed from the final `index.json` files with the same logic.

| Method | num_items | unique_paths | collided_items | final collision | first3 collision | collided_groups | max_group_size | token lengths | Current code can train directly |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| constrained RQ-Kmeans | 3686 | 3686 | 0 | 0.0000000000 | 0.1090613131 | 0 | 1 | `3:3003;4:683` | No |
| RQ-Kmeans+ | 3686 | 3686 | 0 | 0.0000000000 | 0.1131307651 | 0 | 1 | `3:3012;4:674` | No |
| RQ-Kmeans | 3686 | 2969 | 717 | 0.1945198047 | 0.1945198047 | 420 | 22 | `3:3686` | Yes |
| RQ-VAE | 3686 | 3673 | 13 | 0.0035268584 | 0.0035268584 | 13 | 2 | `3:3686` | Yes |

### 2.2 Interpretation

- If the comparison is constrained to fixed 3-token SIDs, `RQ-VAE` is the best method in this experiment.
- `constrained RQ-Kmeans` and `RQ-Kmeans+` both reach zero final collision only because they append a 4th token to collided items during deduplication.
- `RQ-Kmeans` is the weakest baseline here. It uses a pure 3-level FAISS residual quantizer without the extra collision-handling logic used by the other methods.
- For the current downstream codebase, only `RQ-VAE` and `RQ-Kmeans` are directly compatible, because multiple places in `data.py` hardcode SID usage as `sids[0] + sids[1] + sids[2]`.

## 3. How Each Method Was Trained

### 3.1 constrained RQ-Kmeans

- Entry: `rq/rqkmeans_constrained.py`
- Core idea:
  - Run 4-level residual K-means.
  - Each level uses `KMeansConstrained` to force nearly balanced cluster sizes.
  - After clustering, the script uses `polars` deduplication logic to append a 4th SID token for collided paths.
- Effective setting used in this experiment:
  - dataset: `Industrial_and_Scientific`
  - levels: `4`
  - clusters per level: `256`
  - embedding source: `Industrial_and_Scientific.emb-qwen-td.npy`
- Observed effect:
  - Raw 3-token quality is decent: `first3 collision = 0.1090613131`
  - Final zero-collision output is achieved by variable-length deduplication, not by pure 3-token uniqueness
- Final outputs used for comparison:
  - `Industrial_and_Scientific.index.json`
  - `Industrial_and_Scientific.codebooks_constrained.npz`
  - `Industrial_and_Scientific.codes_constrained.npy`

### 3.2 RQ-Kmeans+

- Entry: `rq/rqkmeans_plus.py`
- Generation: `rq/generate_indices_plus.py`
- Core idea:
  - Start from constrained RQ-Kmeans codebooks as warm-start.
  - Replace encoder with residual form `Z = X + MLP(X)`.
  - Zero-initialize the encoder’s last linear layer so the model begins close to identity mapping.
  - Keep a 3-level quantizer during training, then run `polars` deduplication during SID generation to append a 4th token where needed.
- Effective setting used in this experiment:
  - pretrained codebook: constrained RQ-Kmeans codebooks
  - `num_emb_list = [256, 256, 256]`
  - `e_dim = 2560`
  - `lr = 1e-4`
  - `epochs = 10000`
  - `batch_size = 2048`
  - device: `cuda:0`
- Training result:
  - `RQ-KMeans+ Final Result -> Loss: 1.6991870999336243`
  - `RQ-KMeans+ Final Result -> Collision Rate: 0.11313076505697232`
- Generation-stage note:
  - Under the current environment, `generate_indices_plus.py` required `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` because of the PyTorch 2.6 `weights_only=True` default.
- Observed effect:
  - Pure 3-token collision is slightly worse than constrained RQ-Kmeans.
  - Final zero-collision output again depends on appending a 4th token during deduplication.

### 3.3 RQ-Kmeans

- Entry: `rq/rqkmeans_faiss.py`
- Core idea:
  - Use FAISS `ResidualQuantizer` with 3 levels and codebook size 256.
  - This experiment used the plain baseline path and did **not** enable `--uniform`.
- Effective setting used in this experiment:
  - `num_levels = 3`
  - `codebook_size = 256`
  - `--uniform` not used
- Observed effect:
  - `unique full-paths = 2969`
  - `collision_rate = 0.1945`
  - Highest collision among the four methods
- Final outputs used for comparison:
  - `Industrial_and_Scientific.faiss-rq.index.json`
  - `Industrial_and_Scientific.faiss-rq.index.faiss`

### 3.4 RQ-VAE

- Entry: `rq/rqvae.py`
- Generation: `rq/generate_indices.py`
- Core idea:
  - Train a 3-level residual VQ-VAE over item embeddings.
  - Model uses an MLP encoder/decoder and a 3-layer residual vector quantizer.
  - After training, `generate_indices.py` first encodes items with `use_sk=False`, then repeatedly revisits collided groups with `use_sk=True` to reduce conflicts.
- Effective setting used in this experiment:
  - `num_emb_list = [256, 256, 256]`
  - `e_dim = 32`
  - `lr = 1e-3`
  - `epochs = 10000`
  - `batch_size = 20480`
  - device: `cuda:0`
- Training result:
  - best checkpoint: `epoch = 9949`
  - raw best collision during training: `0.088171459576777`
- Generation result:
  - `All indices number: 3686`
  - `Max number of conflicts: 2`
  - final `Collision Rate: 0.00352685838307108`
- Observed effect:
  - Best fixed-3-token method in this experiment
  - Final collision is much lower than the raw training collision because the generation script performs extra collision resolution passes

## 4. Downstream Compatibility Analysis

### 4.1 Why Only Two Methods Are Directly Trainable Right Now

The current downstream codebase assumes 3-part SIDs in multiple places. In `data.py`, several datasets explicitly construct semantic IDs as:

`sids[0] + sids[1] + sids[2]`

This means:

- `RQ-VAE` and `RQ-Kmeans` are safe because all 3686 items use exactly 3 tokens.
- `constrained RQ-Kmeans` and `RQ-Kmeans+` are not safe under the current code, because their zero-collision result depends on 4-token items.

### 4.2 4-token Risk for the Two Variable-Length Methods

#### constrained RQ-Kmeans

| Split | rows | target uses 4 tokens | history contains any 4-token item |
|---|---:|---:|---:|
| train | 36259 | 8878 (`24.48%`) | 17621 (`48.60%`) |
| valid | 4532 | 1072 (`23.65%`) | 2617 (`57.74%`) |
| test | 4533 | 1160 (`25.59%`) | 2801 (`61.79%`) |

#### RQ-Kmeans+

| Split | rows | target uses 4 tokens | history contains any 4-token item |
|---|---:|---:|---:|
| train | 36259 | 8884 (`24.50%`) | 17877 (`49.30%`) |
| valid | 4532 | 1204 (`26.57%`) | 2686 (`59.27%`) |
| test | 4533 | 1384 (`30.53%`) | 3045 (`67.17%`) |

These ratios are too high to ignore. Treating these methods as plain 3-token SID inputs would silently discard the deduplication signal that made them zero-collision in the first place.

## 5. Transfer Bundle Construction

### 5.1 Bundle Root

- Bundle directory: `MiniOneRec_transfer_bundle_Industrial_and_Scientific_20260411`
- Intended use: move this directory, or its generated `tar.gz`, to another machine and run downstream training with explicit file paths

### 5.2 Included Files

- `shared/Industrial_and_Scientific.item.json`
- `sid_variants/*/Industrial_and_Scientific.index.json`
- `converted/*/{train,valid,test,info}/Industrial_and_Scientific_5_2016-10-2018-11.*`
- `report/sid_experiment_record.md`
- `report/sid_metrics.csv`
- `report/bundle_manifest.tsv`
- `report/SHA256SUMS`
- `runbook/README_transfer.md`

### 5.2.1 How The Converted Data Was Produced

The current repository snapshot does not contain the original `Industrial_and_Scientific.{train,valid,test}.inter` files required by `convert_dataset.py`.

To keep the transfer bundle usable, the per-method downstream files were generated by:

- reading the repository’s current `train / valid / test` CSV splits
- preserving `user_id`, item titles, and item-id sequences
- remapping `item_sid` and `history_item_sid` from `item_id / history_item_id` using each method’s final `index.json`
- rebuilding each method’s `info` file from the shared item metadata plus that method’s final SID mapping

This preserves row counts and item-id structure exactly while making each converted dataset fully aligned to its own SID variant.

### 5.3 Explicitly Excluded Files

- `Industrial_and_Scientific.emb-qwen-td.npy`
- `RQ-VAE` checkpoint directory
- `RQ-Kmeans+` checkpoint directory
- constrained codebooks and raw code arrays
- FAISS binary index
- any original experiment symlink

## 6. Verification Results

### 6.1 Default Data Alignment Check

The original default training files under `data/Amazon/{train,valid,test,info}` align only with the repository’s default `data/Amazon/index/Industrial_and_Scientific.index.json`.

- Default train target-SID alignment against default index: `36259 / 36259`
- Default train target-SID alignment against each experimental index:
  - `constrained_rqkmeans`: `0 / 36259`
  - `rqkmeans_plus`: `0 / 36259`
  - `rqkmeans_faiss`: `0 / 36259`
  - `rqvae`: `0 / 36259`
- Sampled history-SID alignment on the first 2000 default train rows:
  - default index: `2000 / 2000`
  - each experimental index: `0 / 2000`

Conclusion: the repository’s existing CSV and info files cannot be reused directly for these 4 SID variants.

### 6.2 Converted Bundle Data Check

For all four methods, the preconverted bundle data passed the following checks:

- train rows: `36259`
- valid rows: `4532`
- test rows: `4533`
- info rows: `3686`
- sampled `item_sid == ''.join(index[item_id])`
- sampled `history_item_sid` entries match the corresponding `history_item_id`
- no symbolic links exist inside the bundle

## 7. Operational Recommendation

- For immediate follow-up training on the next machine, use `rqvae` as the primary method.
- Keep `rqkmeans_faiss` as a direct-train 3-token baseline.
- Keep `constrained_rqkmeans` and `rqkmeans_plus` in the bundle for SID comparison and possible future code adaptation, but do not feed them into the current downstream pipeline without first making the data layer support variable-length SIDs.
