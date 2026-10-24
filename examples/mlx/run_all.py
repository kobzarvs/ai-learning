"""Run every standalone MLX example on the explicitly requested device."""
from contextlib import redirect_stdout
from pathlib import Path
import io
import runpy
import re
import sys

failures = []
for path in sorted(Path(__file__).parent.glob("*.py")):
    if path.name == "run_all.py":
        continue
    output = io.StringIO()
    try:
        with redirect_stdout(output):
            runpy.run_path(str(path), run_name="__main__")
        if re.search(r"\b(?:nan|inf)\b", output.getvalue(), re.IGNORECASE):
            raise ValueError("Non-finite result")
        print("PASS", path.stem, output.getvalue().strip().replace("\n", " | ")[:320], flush=True)
    except Exception as error:
        failures.append((path.stem, repr(error)))
        print("FAIL", path.stem, repr(error), flush=True)
print("FAILED:", failures)
sys.exit(bool(failures))
