from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retentioniq.data import generate_demo_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate clearly labeled synthetic churn-demo data.")
    parser.add_argument("--rows", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "synthetic_customers.csv")
    args = parser.parse_args()

    data = generate_demo_data(args.rows, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False)
    print(f"Wrote {len(data):,} synthetic records to {args.output}")


if __name__ == "__main__":
    main()