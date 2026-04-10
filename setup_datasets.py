#!/usr/bin/env python3
"""Download and set up all datasets required by medriskeval.

Usage:
    python setup_datasets.py
    python setup_datasets.py --skip-hf    # skip HuggingFace datasets (auto-downloaded at runtime)
    python setup_datasets.py --facts-csv path/to/FACTS_examples.csv
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# Root of medriskeval project
PROJECT_ROOT = Path(__file__).resolve().parent


def run(cmd: list[str], cwd: Path | None = None) -> bool:
    """Run a command and return True on success."""
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [FAIL] {result.stderr.strip()}")
        return False
    return True


def setup_msb() -> bool:
    """Clone the MedSafetyBench repository if not already present."""
    print("\n[MSB] MedSafetyBench — GitHub clone")
    repo_dir = PROJECT_ROOT / "med-safety-bench"
    data_dir = repo_dir / "datasets" / "test" / "gpt4"

    if data_dir.exists():
        csv_count = len(list(data_dir.glob("med_safety_demonstrations_category_*.csv")))
        if csv_count == 9:
            print(f"  Already present ({csv_count} category files). Skipping.")
            return True
        print(f"  Directory exists but only {csv_count}/9 files found. Re-cloning.")
        shutil.rmtree(repo_dir, ignore_errors=True)

    url = "https://github.com/AI4LIFE-GROUP/med-safety-bench.git"
    if not run(["git", "clone", "--depth", "1", url], cwd=PROJECT_ROOT):
        print("  [FAIL] Could not clone. Make sure git is installed and you have internet access.")
        return False

    if not data_dir.exists():
        print(f"  [FAIL] Expected data path not found: {data_dir}")
        return False

    csv_count = len(list(data_dir.glob("med_safety_demonstrations_category_*.csv")))
    print(f"  [OK] {csv_count} category CSV files ready.")
    return True


def setup_facts(csv_source: str | None = None) -> bool:
    """Place FACTS_examples.csv in the project root.

    Tries (in order):
    1. Explicit --facts-csv path
    2. Common local locations (Downloads, home, cache)
    3. Kaggle download via kagglehub
    """
    print("\n[FACTS_MED] FACTS Medical — local CSV")
    dest = PROJECT_ROOT / "FACTS_examples.csv"

    if dest.exists():
        print(f"  Already present at {dest}. Skipping.")
        return True

    # Try explicit source path
    if csv_source:
        src = Path(csv_source).expanduser().resolve()
        if src.exists():
            shutil.copy2(src, dest)
            print(f"  [OK] Copied from {src}")
            return True
        else:
            print(f"  [FAIL] Provided path not found: {src}")
            return False

    # Search common locations
    candidates = [
        Path.home() / "Downloads" / "FACTS_examples.csv",
        Path.home() / "FACTS_examples.csv",
        Path.home() / ".cache" / "medriskeval" / "facts" / "FACTS_examples.csv",
    ]

    for candidate in candidates:
        if candidate.exists():
            shutil.copy2(candidate, dest)
            print(f"  [OK] Found and copied from {candidate}")
            return True

    # Try downloading from Kaggle
    print("  Not found locally. Attempting Kaggle download...")
    if _download_facts_from_kaggle(dest):
        return True

    print("  [FAIL] FACTS_examples.csv not found.")
    print("  Options:")
    print("    1. pip install kagglehub  (then re-run this script)")
    print("    2. Download manually from: https://www.kaggle.com/datasets/deepmind/facts-grounding-examples")
    print(f"    3. python setup_datasets.py --facts-csv /path/to/FACTS_examples.csv")
    print(f"    4. Place it directly at: {dest}")
    return False


def _download_facts_from_kaggle(dest: Path) -> bool:
    """Download FACTS_examples.csv from Kaggle using kagglehub."""
    try:
        import kagglehub
    except ImportError:
        print("  kagglehub not installed. Install with: pip install kagglehub")
        return False

    try:
        path = kagglehub.dataset_download("deepmind/facts-grounding-examples")
        download_dir = Path(path)
        print(f"  Downloaded to: {download_dir}")

        # Find the CSV in the downloaded files
        csv_files = list(download_dir.rglob("*.csv"))
        if not csv_files:
            print(f"  [FAIL] No CSV files found in {download_dir}")
            return False

        # Prefer FACTS_examples.csv, otherwise take the first CSV
        src = None
        for f in csv_files:
            if "FACTS" in f.name or "facts" in f.name or "examples" in f.name.lower():
                src = f
                break
        if src is None:
            src = csv_files[0]

        shutil.copy2(src, dest)
        print(f"  [OK] Downloaded and copied from Kaggle: {src.name}")
        return True
    except Exception as e:
        print(f"  [FAIL] Kaggle download failed: {e}")
        return False


def setup_hf_datasets() -> bool:
    """Pre-download HuggingFace datasets (PSB, JBB, XSTest)."""
    print("\n[HF] Pre-downloading HuggingFace datasets...")

    try:
        from datasets import load_dataset
    except ImportError:
        print("  [FAIL] 'datasets' package not installed. Run: pip install datasets")
        return False

    hf_datasets = [
        ("PSB", "microsoft/PatientSafetyBench", None, "train"),
        ("JBB", "JailbreakBench/JBB-Behaviors", "behaviors", "harmful"),
        ("XSTest", "walledai/XSTest", None, "test"),
    ]

    all_ok = True
    for name, dataset_id, config, split in hf_datasets:
        print(f"\n  [{name}] {dataset_id} (split={split})")
        try:
            ds = load_dataset(dataset_id, config, split=split, trust_remote_code=True)
            print(f"  [OK] {len(ds)} examples loaded and cached.")
        except Exception as e:
            print(f"  [FAIL] {e}")
            all_ok = False

    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Set up datasets for medriskeval benchmarks."
    )
    parser.add_argument(
        "--skip-hf",
        action="store_true",
        help="Skip HuggingFace dataset downloads (PSB, JBB, XSTest auto-download at runtime).",
    )
    parser.add_argument(
        "--facts-csv",
        type=str,
        default=None,
        help="Path to FACTS_examples.csv file.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("medriskeval — Dataset Setup")
    print(f"Project root: {PROJECT_ROOT}")
    print("=" * 60)

    results = {}

    # 1. MSB (requires git clone)
    results["MSB"] = setup_msb()

    # 2. FACTS_med (requires local CSV)
    results["FACTS_MED"] = setup_facts(args.facts_csv)

    # 3. HF datasets (PSB, JBB, XSTest)
    if not args.skip_hf:
        results["HF (PSB/JBB/XSTest)"] = setup_hf_datasets()
    else:
        print("\n[HF] Skipped (--skip-hf). PSB, JBB, XSTest will download on first run.")
        results["HF (PSB/JBB/XSTest)"] = True

    # Summary
    print("\n" + "=" * 60)
    print("Setup Summary")
    print("=" * 60)
    for name, ok in results.items():
        status = "OK" if ok else "FAIL"
        print(f"  [{status:>4}] {name}")

    failed = sum(1 for ok in results.values() if not ok)
    if failed:
        print(f"\n{failed} dataset(s) need attention. See messages above.")
        sys.exit(1)
    else:
        print("\nAll datasets ready. You can now run:")
        print("  python -m medriskeval.cli.main run-config configs/full_eval.yaml")


if __name__ == "__main__":
    main()
