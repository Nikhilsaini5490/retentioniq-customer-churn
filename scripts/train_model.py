from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retentioniq.data import generate_demo_data
from retentioniq.service import DEFAULT_MODEL_PATH, train_and_save


def main() -> None:
    data = generate_demo_data()
    artifact = train_and_save(data)
    print(
        json.dumps(
            {
                "artifact": str(DEFAULT_MODEL_PATH),
                "model": artifact["model_name"],
                "training_rows": artifact["training_rows"],
                "holdout_roc_auc": artifact["metrics"]["roc_auc"],
                "trained_at_utc": artifact["trained_at_utc"],
                "data_notice": "Metrics describe a synthetic dataset; they are not real-world evidence.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()