#!/usr/bin/env python
import subprocess
import sys

# Run mypy with the same configuration as CI
cmd = ["python", "-m", "mypy", "src/swebench_runner"]
result = subprocess.run(cmd, capture_output=True, text=True)

print("STDOUT:")
print(result.stdout)
print("\nSTDERR:")
print(result.stderr)
print(f"\nReturn code: {result.returncode}")

sys.exit(result.returncode)