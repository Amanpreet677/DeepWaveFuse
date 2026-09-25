from __future__ import annotations

import csv
import statistics
from pathlib import Path

from .metrics import evaluate_triplet


ROOT = Path(__file__).resolve().parent.parent

DATASETS = {
    "Lytro": {
        "input": ROOT / "datasets" / "Lytro",
        "fused": ROOT / "results" / "fused_v3" / "Lytro",
    },
    "MFI-WHU": {
        "input": ROOT / "datasets" / "MFI-WHU",
        "fused": ROOT / "results" / "fused_v3" / "MFI-WHU",
    },
    "COCO": {
        "input": ROOT / "datasets" / "COCO",
        "fused": ROOT / "results" / "fused_v3" / "COCO",
    },
    "VOC2012": {
        "input": ROOT / "datasets" / "VOC 2012",
        "fused": ROOT / "results" / "fused_v3" / "VOC2012",
    },
}

OUTPUT = ROOT / "results" / "metrics_v3"


def get_pairs(dataset_name, root):

    pairs = []

    if dataset_name == "Lytro":

        for i in range(1, 21):
            tag = f"{i:02d}"

            a = root / f"lytro-{tag}-A.jpg"
            b = root / f"lytro-{tag}-B.jpg"

            if a.exists() and b.exists():
                pairs.append(
                    (f"lytro-{tag}", a, b, f"lytro-{tag}-F.jpg")
                )

    elif dataset_name == "MFI-WHU":

        for i in range(1, 16):
            a = root / f"{i} (A).jpg"
            b = root / f"{i} (B).jpg"

            if a.exists() and b.exists():
                pairs.append(
                    (str(i), a, b, f"{i}-F.jpg")
                )

    elif dataset_name == "COCO":

        for i in range(1, 11):
            a = root / f"{i} (A).jpg"
            b = root / f"{i} (B).jpg"

            if a.exists() and b.exists():
                pairs.append(
                    (str(i), a, b, f"{i}-F.jpg")
                )

    elif dataset_name == "VOC2012":

        for i in range(1, 11):
            a = root / f"{i}.jpg"

            b2 = root / f"{i} (2).jpg"
            b1 = root / f"{i} (1).jpg"

            if b2.exists():
                b = b2
            elif b1.exists():
                b = b1
            else:
                continue

            if a.exists():
                pairs.append(
                    (str(i), a, b, f"{i}-F.jpg")
                )

    return pairs


def evaluate_dataset(dataset_name, config):

    input_root = config["input"]
    fused_root = config["fused"]

    print()
    print("=" * 72)
    print(dataset_name)
    print("=" * 72)

    if not input_root.exists():
        print("Dataset not found:", input_root)
        return None

    if not fused_root.exists():
        print("Fused directory not found:", fused_root)
        return None

    pairs = get_pairs(dataset_name, input_root)

    print("Input :", input_root)
    print("Fused :", fused_root)
    print("Pairs :", len(pairs))

    if not pairs:
        return None

    rows = []

    for index, (pair_id, a_path, b_path, fused_name) in enumerate(
        pairs, start=1
    ):

        fused_path = fused_root / fused_name

        if not fused_path.exists():
            print(
                f"[{index:02d}/{len(pairs):02d}] "
                f"{pair_id} | MISSING FUSED"
            )
            continue

        values = evaluate_triplet(
            fused_path,
            a_path,
            b_path,
            None,
        )

        row = {
            "Dataset": dataset_name,
            "Pair": pair_id,
            "Image": fused_name,
            "QABF": values["QABF"],
            "MI": values["MI"],
            "VIF": values["VIF"],
            "SD": values["SD"],
            "EN": values["EN"],
            "SF": values["SF"],
            "AG": values["AG"],
        }

        rows.append(row)

        print(
            f"[{index:02d}/{len(pairs):02d}] "
            f"{pair_id:>8s} | "
            f"QAB/F={row['QABF']:.4f} | "
            f"MI={row['MI']:.4f} | "
            f"VIF={row['VIF']:.4f} | "
            f"SD={row['SD']:.2f} | "
            f"EN={row['EN']:.4f}"
        )

    if not rows:
        return None

    OUTPUT.mkdir(parents=True, exist_ok=True)

    metrics = [
        "QABF",
        "MI",
        "VIF",
        "SD",
        "EN",
        "SF",
        "AG",
    ]

    summary = {
        "Dataset": dataset_name,
        "Pairs": len(rows),
    }

    print()
    print(f"{dataset_name} SUMMARY")
    print("-" * 72)

    for metric in metrics:

        values = [float(row[metric]) for row in rows]

        mean_value = statistics.mean(values)

        if len(values) > 1:
            sd_value = statistics.stdev(values)
        else:
            sd_value = 0.0

        summary[f"{metric}_Mean"] = mean_value
        summary[f"{metric}_SD"] = sd_value

        print(
            f"{metric:6s}: "
            f"{mean_value:.6f} ± {sd_value:.6f}"
        )

    output_file = OUTPUT / f"{dataset_name}_metrics.csv"

    fields = [
        "Dataset",
        "Pair",
        "Image",
        "QABF",
        "MI",
        "VIF",
        "SD",
        "EN",
        "SF",
        "AG",
    ]

    with output_file.open("w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)

    print("Saved:", output_file)

    return summary


def main():

    print("=" * 72)
    print("DeepWaveFuse V3 - Multi-Dataset Evaluation")
    print("=" * 72)

    summaries = []

    for dataset_name, config in DATASETS.items():

        result = evaluate_dataset(
            dataset_name,
            config,
        )

        if result is not None:
            summaries.append(result)

    if not summaries:
        raise RuntimeError(
            "No datasets were successfully evaluated."
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    summary_file = OUTPUT / "summary_tables.csv"

    fields = [
        "Dataset",
        "Pairs",
        "QABF_Mean",
        "QABF_SD",
        "MI_Mean",
        "MI_SD",
        "VIF_Mean",
        "VIF_SD",
        "SD_Mean",
        "SD_SD",
        "EN_Mean",
        "EN_SD",
        "SF_Mean",
        "SF_SD",
        "AG_Mean",
        "AG_SD",
    ]

    with summary_file.open("w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(summaries)

    print()
    print("=" * 72)
    print("FINAL DATASET SUMMARY")
    print("=" * 72)

    print(
        f"{'Dataset':12s} "
        f"{'N':>4s} "
        f"{'QAB/F':>16s} "
        f"{'MI':>16s} "
        f"{'VIF':>16s} "
        f"{'SD':>16s} "
        f"{'EN':>16s}"
    )

    print("-" * 82)

    for result in summaries:

        print(
            f"{result['Dataset']:12s} "
            f"{result['Pairs']:4d} "
            f"{result['QABF_Mean']:.4f}±{result['QABF_SD']:.4f} "
            f"{result['MI_Mean']:.4f}±{result['MI_SD']:.4f} "
            f"{result['VIF_Mean']:.4f}±{result['VIF_SD']:.4f} "
            f"{result['SD_Mean']:.2f}±{result['SD_SD']:.2f} "
            f"{result['EN_Mean']:.4f}±{result['EN_SD']:.4f}"
        )

    print()
    print("Summary saved:", summary_file)


if __name__ == "__main__":
    main()
