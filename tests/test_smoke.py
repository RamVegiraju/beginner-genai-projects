"""Fast checks that do not need workspace credentials or call a model."""

import ast
import importlib.util
from pathlib import Path

ROOT = Path(__file__).parents[1]
SAMPLES = [
    ROOT / "01-first-llm-call" / "invoke.py",
    ROOT / "02-streamlit-chatbot" / "app.py",
    ROOT / "03-langgraph-agent" / "weather_tool.py",
    ROOT / "04-agent-memory" / "app.py",
    ROOT / "05-fastapi-server" / "server.py",
    ROOT / "06-mlflow-evals" / "app.py",
]


def load_module(name: str, path: Path):
    """Import one file by path without turning the samples into packages."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def test_every_sample_defaults_to_haiku():
    for path in SAMPLES:
        tree = ast.parse(path.read_text())
        assignments = {
            node.targets[0].id: node.value
            for node in tree.body
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        }
        model = assignments["MODEL"]
        assert isinstance(model, ast.Call)
        assert len(model.args) == 2
        assert model.args[1].value == "databricks-claude-haiku-4-5"


def test_every_sample_allows_ambient_authentication():
    for path in SAMPLES:
        tree = ast.parse(path.read_text())
        profile = next(
            node.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "PROFILE"
        )
        assert isinstance(profile, ast.Call)
        assert len(profile.args) == 1  # no hard-coded profile default


def test_distillation_replaces_the_profile():
    distill = load_module("sample_distill", ROOT / "04-agent-memory" / "distill.py")

    class Store:
        def __init__(self):
            self.values = {"old": {"fact": "Old fact"}}

        def search(self, _namespace, limit):
            assert limit == 100
            return [type("Item", (), {"key": key}) for key in self.values]

        def delete(self, _namespace, key):
            del self.values[key]

        def put(self, _namespace, key, value):
            self.values[key] = value

    store = Store()
    distill.remember(store, ("user", "profile"), ["New fact", "Another fact"])
    assert store.values == {
        "fact-0": {"fact": "New fact"},
        "fact-1": {"fact": "Another fact"},
    }


def test_mlflow_sample_declares_autologging_dependencies():
    requirements = (ROOT / "06-mlflow-evals" / "requirements.txt").read_text().splitlines()
    assert any(line.startswith("langchain>=") for line in requirements)
    assert any(line.startswith("mlflow[databricks]>=3.15.1") for line in requirements)
