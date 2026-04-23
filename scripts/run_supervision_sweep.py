import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


INFERENCE_PY = ROOT / "inference.py"
METRIC_KEYS = ("PK", "WD", "F1", "Precision", "Recall", "Score")
HIGHER_IS_BETTER = {"F1", "Precision", "Recall", "Score"}
LOWER_IS_BETTER = {"PK", "WD"}


def resolve_dataset_path(dataset_name_or_path: str) -> str:
    if dataset_name_or_path.lower().endswith(".json"):
        return dataset_name_or_path
    mapping = {
        "vfh": "./data/dataset/vhf.json",
        "vhf": "./data/dataset/vhf.json",
        "dialseg711": "./data/dataset/dialseg711.json",
        "doc2dial": "./data/dataset/doc2dial.json",
        "tiage": "./data/dataset/tiage.json",
        "superseg": "./data/dataset/superseg.json",
    }
    key = dataset_name_or_path.lower()
    return mapping.get(key, mapping["vhf"])


def _parse_ratio_list(text: str) -> list[float]:
    ratios: list[float] = []
    for raw in text.split(","):
        token = raw.strip()
        if not token:
            continue
        value = float(token)
        if value > 1:
            value = value / 100.0
        if value <= 0 or value > 1:
            raise ValueError(f"Invalid ratio: {raw!r}")
        ratios.append(value)
    if not ratios:
        raise ValueError("At least one ratio is required.")
    return ratios


def _load_split_counts(dataset: str) -> tuple[str, Counter]:
    ds_path = resolve_dataset_path(dataset)
    with open(ds_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    counts: Counter = Counter()
    for item in data:
        split = str(item.get("set", "train")).strip().lower()
        if split in {"valid", "val", "dev"}:
            counts["val"] += 1
        elif split == "test":
            counts["test"] += 1
        else:
            counts["train"] += 1
    return ds_path, counts


def _subset_count_from_train_ratio(train_count: int, train_ratio: float) -> int:
    target_count = int(round(train_count * train_ratio))
    target_count = max(1, target_count)
    return min(train_count, target_count)


def _results_path(dataset: str, exp_name: str) -> Path:
    ds_path = resolve_dataset_path(dataset)
    ds_stem = Path(ds_path).stem
    return ROOT / "checkpoints" / ds_stem / exp_name / "results.json"


def _load_metrics_from_results(results_path: Path, split: str) -> dict:
    data = json.loads(results_path.read_text(encoding="utf-8"))
    if split == "test":
        return data.get("metrics_test") or {}
    if split == "val":
        return data.get("metrics_val") or data.get("metrics_dev") or {}
    raise ValueError(f"Unsupported split: {split}")


def _metric_delta(metric_name: str, current_value: float, baseline_value: float) -> float:
    if metric_name in HIGHER_IS_BETTER:
        return current_value - baseline_value
    if metric_name in LOWER_IS_BETTER:
        return baseline_value - current_value
    raise ValueError(f"Unsupported metric for comparison: {metric_name}")


def _reached_baseline(metric_name: str, current_value: float, baseline_value: float) -> bool:
    if metric_name in HIGHER_IS_BETTER:
        return current_value >= baseline_value
    if metric_name in LOWER_IS_BETTER:
        return current_value <= baseline_value
    raise ValueError(f"Unsupported metric for comparison: {metric_name}")


def _flatten_metrics(row: dict, prefix: str, metrics: dict | None) -> None:
    metrics = metrics or {}
    for key in METRIC_KEYS:
        row[f"{prefix}_{key}"] = metrics.get(key, "")


def _validate_passthrough_args(extra_args: list[str]) -> None:
    reserved = {
        "--dataset",
        "--exp_name",
        "--seed",
        "--train_subset_count",
        "--train_subset_seed",
    }
    conflicts = [arg for arg in extra_args if arg in reserved]
    if conflicts:
        joined = ", ".join(conflicts)
        raise SystemExit(
            "These arguments are controlled by the sweep script and should not be "
            f"passed through again: {joined}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a supervised-data sweep over post-split train-subset ratios for DTS models."
    )
    parser.add_argument("--dataset", required=True, help="Dataset alias or JSON path.")
    parser.add_argument(
        "--ratios",
        default="0.01,0.03,0.05,0.1,0.25,0.5,0.75",
        help="Comma-separated target train-subset ratios after split, e.g. 0.01,0.03,... or 1,3,...",
    )
    parser.add_argument("--exp_prefix", default="supervision_sweep")
    parser.add_argument("--seed", type=int, default=42, help="Training seed passed to inference.py.")
    parser.add_argument(
        "--subset_seed",
        type=int,
        default=None,
        help="Sampling seed for train subset selection. Defaults to --seed.",
    )
    parser.add_argument(
        "--compare_split",
        choices=("val", "test"),
        default="test",
        help="Which split to use when deciding whether supervised reaches the baseline.",
    )
    parser.add_argument(
        "--compare_metric",
        choices=tuple(sorted(HIGHER_IS_BETTER | LOWER_IS_BETTER)),
        default="Score",
        help="Metric used for the baseline comparison decision.",
    )
    parser.add_argument(
        "--baseline_json",
        default="",
        help="Optional unsupervised results.json to compare against.",
    )
    parser.add_argument(
        "--output_csv",
        default="",
        help="Optional output CSV path. Defaults to data/experiments/<dataset>_<exp_prefix>.csv",
    )
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args, extra_args = parser.parse_known_args()

    _validate_passthrough_args(extra_args)

    ratios = _parse_ratio_list(args.ratios)
    subset_seed = args.subset_seed if args.subset_seed is not None else args.seed
    dataset_path, counts = _load_split_counts(args.dataset)
    ds_stem = Path(dataset_path).stem
    train_count = int(counts["train"])
    val_count = int(counts["val"])
    test_count = int(counts["test"])
    if train_count <= 0:
        raise SystemExit("No train split found in dataset.")

    baseline_metrics = None
    baseline_value = ""
    if args.baseline_json:
        baseline_path = Path(args.baseline_json)
        if not baseline_path.is_absolute():
            baseline_path = ROOT / baseline_path
        if not baseline_path.exists():
            raise SystemExit(f"Baseline results not found: {baseline_path}")
        baseline_metrics = _load_metrics_from_results(baseline_path, args.compare_split)
        if args.compare_metric not in baseline_metrics:
            raise SystemExit(
                f"Metric {args.compare_metric} not found in baseline: {baseline_path}"
            )
        baseline_value = baseline_metrics[args.compare_metric]

    if args.output_csv:
        output_csv = Path(args.output_csv)
        if not output_csv.is_absolute():
            output_csv = ROOT / output_csv
    else:
        output_csv = ROOT / "data" / "experiments" / f"{ds_stem}_{args.exp_prefix}.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {args.dataset} -> {dataset_path}")
    print(f"Splits  train={train_count}  val={val_count}  test={test_count}")
    print(f"Ratios (train split): {', '.join(f'{r:.0%}' for r in ratios)}")
    print(f"Training seed={args.seed}  subset seed={subset_seed}")
    if baseline_metrics is not None:
        print(
            f"Baseline: {args.baseline_json}  compare={args.compare_split}.{args.compare_metric}="
            f"{baseline_value:.6f}"
        )

    rows: list[dict] = []
    first_reached_ratio = None

    for ratio in ratios:
        subset_count = _subset_count_from_train_ratio(train_count, ratio)
        actual_train_ratio = subset_count / train_count
        ratio_tag = int(round(ratio * 100))
        exp_name = f"{args.exp_prefix}_p{ratio_tag:02d}"
        result_path = _results_path(args.dataset, exp_name)

        row = {
            "dataset": args.dataset,
            "dataset_path": dataset_path,
            "exp_name": exp_name,
            "target_train_ratio": ratio,
            "target_train_pct": ratio * 100.0,
            "actual_train_ratio": actual_train_ratio,
            "actual_train_pct": actual_train_ratio * 100.0,
            "train_subset_count": subset_count,
            "train_count": train_count,
            "val_count": val_count,
            "test_count": test_count,
            "seed": args.seed,
            "subset_seed": subset_seed,
            "result_json": str(result_path.relative_to(ROOT)),
            "status": "planned",
            "compare_split": args.compare_split,
            "compare_metric": args.compare_metric,
            "baseline_json": args.baseline_json,
            "baseline_value": baseline_value,
            "compare_value": "",
            "delta_to_baseline": "",
            "reached_baseline": "",
        }

        cmd = [
            args.python_bin,
            str(INFERENCE_PY),
            "--dataset",
            args.dataset,
            "--exp_name",
            exp_name,
            "--seed",
            str(args.seed),
            "--train_subset_count",
            str(subset_count),
            "--train_subset_seed",
            str(subset_seed),
            *extra_args,
        ]

        print(
            f"\n[{exp_name}] target={ratio:.0%} of train split  "
            f"-> train_subset_count={subset_count} ({actual_train_ratio:.2%} of train split)"
        )

        if args.dry_run:
            print("DRY RUN:", " ".join(cmd))
            row["status"] = "dry_run"
            rows.append(row)
            continue

        if args.skip_existing and result_path.exists():
            print(f"Skip existing run: {result_path}")
        else:
            completed = subprocess.run(cmd, cwd=ROOT)
            if completed.returncode != 0:
                print(f"Run failed with exit code {completed.returncode}")
                row["status"] = f"failed:{completed.returncode}"
                rows.append(row)
                continue

        if not result_path.exists():
            print(f"Missing results after run: {result_path}")
            row["status"] = "missing_results"
            rows.append(row)
            continue

        data = json.loads(result_path.read_text(encoding="utf-8"))
        metrics_val = data.get("metrics_val", {})
        metrics_test = data.get("metrics_test", {})
        compare_metrics = metrics_test if args.compare_split == "test" else metrics_val
        compare_value = compare_metrics.get(args.compare_metric, "")

        row["status"] = "ok"
        row["compare_value"] = compare_value
        _flatten_metrics(row, "val", metrics_val)
        _flatten_metrics(row, "test", metrics_test)

        if baseline_metrics is not None and compare_value != "":
            delta = _metric_delta(args.compare_metric, compare_value, baseline_value)
            reached = _reached_baseline(args.compare_metric, compare_value, baseline_value)
            row["delta_to_baseline"] = delta
            row["reached_baseline"] = int(reached)
            if reached and first_reached_ratio is None:
                first_reached_ratio = ratio

        rows.append(row)

    fieldnames = [
        "dataset",
        "dataset_path",
        "exp_name",
        "target_train_ratio",
        "target_train_pct",
        "actual_train_ratio",
        "actual_train_pct",
        "train_subset_count",
        "train_count",
        "val_count",
        "test_count",
        "seed",
        "subset_seed",
        "compare_split",
        "compare_metric",
        "baseline_json",
        "baseline_value",
        "compare_value",
        "delta_to_baseline",
        "reached_baseline",
        "val_PK",
        "val_WD",
        "val_F1",
        "val_Precision",
        "val_Recall",
        "val_Score",
        "test_PK",
        "test_WD",
        "test_F1",
        "test_Precision",
        "test_Recall",
        "test_Score",
        "result_json",
        "status",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved sweep summary to {output_csv}")
    if baseline_metrics is not None:
        if first_reached_ratio is None:
            print(
                f"No supervised ratio reached baseline on "
                f"{args.compare_split}.{args.compare_metric}."
            )
        else:
            print(
                f"First ratio reaching baseline on {args.compare_split}.{args.compare_metric}: "
                f"{first_reached_ratio:.0%}"
            )


if __name__ == "__main__":
    main()
