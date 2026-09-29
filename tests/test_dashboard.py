import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from streamlit.testing.v1 import AppTest


def test_dashboard_renders_demo_workspace_without_exceptions():
    app = AppTest.from_file(ROOT / "app.py", default_timeout=45).run()
    assert not app.exception
    assert any(metric.label == "Customers analyzed" for metric in app.metric)
    assert any(metric.label == "Holdout ROC-AUC" for metric in app.metric)
    review_count = next(metric.value for metric in app.metric if metric.label == "High-risk review")
    assert int(review_count.replace(",", "")) > 0