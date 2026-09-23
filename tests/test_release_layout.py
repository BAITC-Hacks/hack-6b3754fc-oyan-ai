"""The old release path delegates to the one canonical, current implementation."""

import json
from pathlib import Path
import subprocess
import sys

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_compatibility_demo_uses_root_snapshots_from_another_cwd(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "ready_project" / "run_demo.py"), "--verify"],
        cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", timeout=30,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "OK: 5 saved responses" in result.stdout


def test_compatibility_ui_runs_current_core_and_demo(monkeypatch):
    monkeypatch.delenv("CONTRACTORS_DATA", raising=False)
    app = AppTest.from_file(str(ROOT / "ready_project" / "app.py"), default_timeout=20).run()
    assert not app.exception and not app.error
    app.selectbox(key="demo_choice").select("dense").run()
    app.button[0].click().run()
    assert not app.exception and not app.error
    manifest = json.loads((ROOT / "demo_queries.json").read_text(encoding="utf-8"))
    case = next(case for case in manifest["cases"] if case["name"] == "dense")
    assert app.session_state["last_result"][1] == case["response"]


def test_no_second_implementation_or_dataset_in_release_directory():
    release = ROOT / "ready_project"
    assert {p.name for p in release.glob("*.py")} == {"app.py", "run_demo.py"}
    assert not list((release / "tests").glob("*.py"))
    assert not list(release.rglob("*.csv"))
    assert not list(release.rglob("*.json"))
    assert (release / "requirements.txt").read_text().strip() == "-r ../requirements.txt"
    assert (release / "requirements.lock").read_text().strip() == "-r ../requirements.lock"
