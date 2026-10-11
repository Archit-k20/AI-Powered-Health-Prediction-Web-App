"""Compare regenerated examples, allowing only floating-point roundoff.

The model artifact and all other metadata are still checked byte-for-byte in CI.
SHAP's native kernels differ by roughly 1e-13 between macOS ARM and Linux x86.
"""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def compare(expected, actual, path="samples"):
    if isinstance(expected, float):
        assert isinstance(actual, (int, float)) and abs(expected - actual) <= 1e-10, path
    elif isinstance(expected, dict):
        assert isinstance(actual, dict) and expected.keys() == actual.keys(), path
        for key in expected:
            compare(expected[key], actual[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(expected) == len(actual), path
        for index, (left, right) in enumerate(zip(expected, actual)):
            compare(left, right, f"{path}[{index}]")
    else:
        assert type(expected) is type(actual) and expected == actual, path


if __name__ == "__main__":
    expected = json.loads(subprocess.check_output(
        ["git", "show", "HEAD:shared/samples.json"], cwd=ROOT
    ))
    actual = json.loads((ROOT / "shared/samples.json").read_text())
    compare(expected, actual)
    print("Sample structure, ranks and sources match; numeric difference <= 1e-10.")
