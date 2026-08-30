"""Download and stream-convert public Devign/ReVeal/Big-Vul graph archives.

Source record: DataSampling4DLVD, Zenodo DOI 10.5281/zenodo.7057996.
The source archives contain train/test JSON arrays. This script preserves the
test set and deterministically assigns part of source train to validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

import ijson
import requests


RECORD_ID = 7057996
BASE_URL = f"https://zenodo.org/api/records/{RECORD_ID}/files"
ARCHIVES = {
    "devign": {
        "file": "reveal_model_devign_data.zip",
        "size": 568_983_924,
        "md5": "86c97a7231f19491d918690b4fd987ae",
    },
    "reveal": {
        "file": "reveal_model_reveal_data.zip",
        "size": 429_996_388,
        "md5": "fd6ac33a8298f07ee81d01d7bc5de277",
    },
    "bigvul": {
        "file": "reveal_model_bigvul_dataset.zip",
        "size": 1_661_913_897,
        "md5": "7a4d98b0ef13ca91b180aeff998e0b52",
    },
}
FEATURE_DIM = 100
NUM_EDGE_TYPES = 4


def file_md5(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()  # noqa: S324 - integrity checksum published by Zenodo
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download(dataset: str, download_dir: Path) -> Path:
    spec = ARCHIVES[dataset]
    download_dir.mkdir(parents=True, exist_ok=True)
    destination = download_dir / spec["file"]
    existing = destination.stat().st_size if destination.exists() else 0
    if existing == spec["size"] and file_md5(destination) == spec["md5"]:
        print(f"[{dataset}] verified existing archive: {destination}")
        return destination
    if existing > spec["size"]:
        raise ValueError(f"{destination} is larger than the published archive")
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    url = f"{BASE_URL}/{spec['file']}/content"
    print(f"[{dataset}] downloading {url} (resume offset {existing:,})")
    with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        append = existing > 0 and response.status_code == 206
        mode = "ab" if append else "wb"
        if existing and not append:
            print(f"[{dataset}] server did not resume; restarting archive download")
        written = existing if append else 0
        with destination.open(mode) as output:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    output.write(chunk)
                    written += len(chunk)
                    print(f"\r[{dataset}] {written / spec['size']:.1%}", end="", flush=True)
    print()
    if destination.stat().st_size != spec["size"]:
        raise IOError(f"downloaded size does not match Zenodo metadata: {destination}")
    actual = file_md5(destination)
    if actual != spec["md5"]:
        raise IOError(f"MD5 mismatch for {destination}: {actual}")
    return destination


def label_of(record: dict) -> int:
    target = record.get("targets", record.get("target", record.get("label")))
    while isinstance(target, list):
        target = target[0]
    label = int(target)
    if label not in {0, 1}:
        raise ValueError(f"non-binary label: {target}")
    return label


def validation_member(index: int, label: int, seed: int, fraction: float) -> bool:
    value = f"{seed}:{label}:{index}".encode()
    number = int.from_bytes(hashlib.blake2b(value, digest_size=8).digest(), "big")
    return number / (2**64 - 1) < fraction


class SplitWriter:
    def __init__(self, root: Path, split: str, dataset: str) -> None:
        self.split = split
        self.dataset = dataset
        self.path = root / f"{split}.jsonl"
        self.stream = self.path.open("wb")
        self.offsets: list[int] = []
        self.labels: list[int] = []

    def add(self, record: dict, sample_id: str, label: int) -> None:
        features = record["node_features"]
        edges = record["graph"]
        if not features or len(features[0]) != FEATURE_DIM:
            raise ValueError(f"{sample_id}: unexpected node feature width")
        node_count = len(features)
        for source, edge_type, target in edges:
            if not (0 <= int(source) < node_count and 0 <= int(target) < node_count):
                raise ValueError(f"{sample_id}: edge references invalid node")
            if not 0 <= int(edge_type) < NUM_EDGE_TYPES:
                raise ValueError(f"{sample_id}: edge type outside 0..{NUM_EDGE_TYPES - 1}")
        normalized = {
            "id": sample_id,
            "node_features": features,
            "graph": edges,
            "target": label,
        }
        self.offsets.append(self.stream.tell())
        self.labels.append(label)
        self.stream.write(json.dumps(normalized, separators=(",", ":")).encode("utf-8") + b"\n")

    def close(self) -> dict:
        self.stream.close()
        index = {
            "offsets": self.offsets,
            "labels": self.labels,
            "id_prefix": f"{self.dataset}-{self.split}",
            "feature_dim": FEATURE_DIM,
            "num_edge_types": NUM_EDGE_TYPES,
        }
        self.path.with_suffix(".index.json").write_text(json.dumps(index), encoding="utf-8")
        positives = sum(self.labels)
        return {
            "total": len(self.labels),
            "vulnerable": positives,
            "non_vulnerable": len(self.labels) - positives,
            "positive_rate": positives / len(self.labels) if self.labels else 0.0,
        }


def prepare(dataset: str, archive: Path, output_root: Path, seed: int,
            validation_fraction: float, max_records: int | None) -> None:
    destination = output_root / dataset
    destination.mkdir(parents=True, exist_ok=True)
    writers = {name: SplitWriter(destination, name, dataset) for name in ("train", "valid", "test")}
    source_counts = {"train": 0, "test": 0}
    with zipfile.ZipFile(archive) as bundle:
        for source_split, member in (("train", "train_GGNNinput.json"),
                                     ("test", "test_GGNNinput.json")):
            with bundle.open(member) as stream:
                for index, record in enumerate(ijson.items(stream, "item", use_float=True)):
                    if max_records is not None and index >= max_records:
                        break
                    label = label_of(record)
                    if source_split == "train" and validation_member(index, label, seed, validation_fraction):
                        target_split = "valid"
                    else:
                        target_split = source_split
                    sample_id = f"{dataset}-source-{source_split}-{index}"
                    writers[target_split].add(record, sample_id, label)
                    source_counts[source_split] += 1
                    if source_counts[source_split] % 5_000 == 0:
                        print(f"[{dataset}] converted {source_counts[source_split]:,} {source_split} records")
    distributions = {name: writer.close() for name, writer in writers.items()}
    if any(stats["vulnerable"] == 0 or stats["non_vulnerable"] == 0
           for stats in distributions.values()):
        raise ValueError("each generated split must contain both classes")
    manifest = {
        "dataset": dataset,
        "purpose": "research",
        "source_record": f"https://zenodo.org/records/{RECORD_ID}",
        "source_doi": "10.5281/zenodo.7057996",
        "archive": archive.name,
        "archive_md5": ARCHIVES[dataset]["md5"],
        "feature_dim": FEATURE_DIM,
        "num_edge_types": NUM_EDGE_TYPES,
        "validation": {
            "source": "source training split only",
            "fraction": validation_fraction,
            "seed": seed,
            "method": "deterministic class-aware BLAKE2b assignment",
        },
        "test_policy": "source test split preserved unchanged",
        "distributions": distributions,
        "limitations": [
            "Processed records do not retain commit/project grouping metadata.",
            "Edge-type numeric semantics follow the source preprocessing artifact.",
        ],
    }
    (destination / "provenance.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=[*ARCHIVES, "all"], default="all")
    parser.add_argument("--download-dir", type=Path, default=Path("data/downloads"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--max-records", type=int, default=None,
                        help="QA only: cap records read from each source split")
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 0.5:
        parser.error("--validation-fraction must be between 0 and 0.5")
    datasets = list(ARCHIVES) if args.dataset == "all" else [args.dataset]
    free = shutil.disk_usage(args.output_root.parent if args.output_root.parent.exists() else Path.cwd()).free
    if free < 12 * 1024**3 and args.max_records is None:
        print("Warning: less than 12 GiB free; full conversion may run out of disk space.", file=sys.stderr)
    for dataset in datasets:
        archive = download(dataset, args.download_dir)
        prepare(dataset, archive, args.output_root, args.seed, args.validation_fraction, args.max_records)


if __name__ == "__main__":
    main()
