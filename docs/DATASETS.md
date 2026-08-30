# Dataset acquisition and provenance

The final study requires function-level labels and Devign-compatible program graphs.
Raw source-code tables alone cannot be passed directly to the model.

## Authoritative sources

| Dataset | Raw source | Expected role |
|---|---|---|
| Devign | https://sites.google.com/view/devign | Near-balanced reference |
| ReVeal/Verum | https://github.com/VulDetProject/ReVeal | Moderately imbalanced |
| Big-Vul | https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset | Severely imbalanced |

The `WIP2022/DataSampling4DLVD` replication package also publishes processed-model
artifacts at DOI `10.5281/zenodo.7057996`. Before using them, record the exact Zenodo
record version, file checksum, preprocessing implementation, feature width, graph
relations, duplicate-removal policy, and split definition.

This project uses the replication record's `reveal_model_*` graph archives:

| Dataset | Archive | Download size | Published MD5 |
|---|---|---:|---|
| Devign | `reveal_model_devign_data.zip` | 568,983,924 bytes | `86c97a7231f19491d918690b4fd987ae` |
| ReVeal | `reveal_model_reveal_data.zip` | 429,996,388 bytes | `fd6ac33a8298f07ee81d01d7bc5de277` |
| Big-Vul | `reveal_model_bigvul_dataset.zip` | 1,661,913,897 bytes | `7a4d98b0ef13ca91b180aeff998e0b52` |

Each archive has source `train_GGNNinput.json` and `test_GGNNinput.json`. The preparation
script preserves source test unchanged and deterministically selects 10% of source train
for validation. It drops token fields that the model does not use and builds indexed
JSONL files so Big-Vul is not loaded wholly into memory.

Run one dataset at a time:

```bash
python scripts/prepare_public_datasets.py --dataset devign
python scripts/prepare_public_datasets.py --dataset reveal
python scripts/prepare_public_datasets.py --dataset bigvul
```

Allow at least 12 GiB of free disk for all three downloads and converted files. Downloads
resume after interruption and are accepted only after size and MD5 verification.

## Required final representation

Every function must have:

- node feature matrix with one consistent width across all splits;
- directed edges encoded as `[source, edge_type, destination]`;
- a binary function label where `1` means vulnerable;
- a stable sample ID and grouping fields needed to prevent leakage;
- a fixed edge vocabulary shared by train, validation, and test.

Recommended edge vocabulary: AST, CFG, DFG/REACHING_DEF, and NCS, with reverse
relations added consistently if the preprocessing implementation uses them.

## Split policy

Create the split once per dataset. Group pre-fix/post-fix variants, duplicate or
near-duplicate functions, and functions from the same fixing commit before splitting.
Store the selected IDs in a split manifest. All five imbalance strategies must reuse
that manifest. Resampling is permitted only after the training IDs are selected.

The selected processed replication artifacts do not retain project, commit, or clone-group
metadata. Therefore, group-aware re-splitting cannot be reconstructed from these files.
This limitation is written to every `provenance.json` and must be disclosed in the report.

## Acceptance checks

The preparation stage must fail if it finds an invalid node index, inconsistent feature
width, unknown edge type, non-binary label, duplicate ID across splits, missing class in
a split, or a provenance file that identifies an AST-only smoke dataset as final data.
