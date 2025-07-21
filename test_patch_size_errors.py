#!/usr/bin/env python3
"""Test patch size error messages."""

import sys
import json
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from swebench_runner.docker_run import run_evaluation
from swebench_runner.models import EvaluationResult

def create_patch_of_size(size_kb: int) -> str:
    """Create a patch of approximately the given size in KB."""
    # Create a patch header
    patch = "diff --git a/test.py b/test.py\n"
    patch += "index 0000000..1111111 100644\n"
    patch += "--- a/test.py\n"
    patch += "+++ b/test.py\n"
    patch += "@@ -1,1 +1,1 @@\n"
    patch += "-old line\n"
    
    # Add content to reach desired size
    # Each character is 1 byte, so we need size_kb * 1024 characters
    remaining_bytes = (size_kb * 1024) - len(patch.encode('utf-8'))
    
    # Add lines of ~80 characters each
    line_content = "+" + "x" * 78 + "\n"  # 80 bytes per line
    num_lines = remaining_bytes // 80
    
    for _ in range(num_lines):
        patch += line_content
    
    # Add remaining bytes
    remaining = remaining_bytes % 80
    if remaining > 1:
        patch += "+" + "x" * (remaining - 2) + "\n"
    
    return patch

def test_patch_sizes():
    print("=== Testing Patch Size Error Messages ===\n")
    
    # Test scenarios: (size_kb, max_patch_mb, description)
    test_cases = [
        (100, 5, "Small patch - should pass"),
        (600, 5, "600KB patch - exceeds Docker limit but within general limit"),
        (6000, 5, "6MB patch - exceeds both limits"),
        (600, 1, "600KB patch with 1MB general limit - exceeds Docker limit only"),
        (1500, 1, "1.5MB patch with 1MB general limit - exceeds both"),
    ]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        for size_kb, max_mb, description in test_cases:
            print(f"Test: {description}")
            print(f"  Patch size: {size_kb}KB, Max limit: {max_mb}MB")
            
            # Create patch file
            patch_data = {
                "instance_id": f"test-{size_kb}kb",
                "patch": create_patch_of_size(size_kb)
            }
            
            patch_file = temp_path / f"patch_{size_kb}kb.jsonl"
            with open(patch_file, 'w') as f:
                json.dump(patch_data, f)
            
            # Mock the docker and resource checks
            import unittest.mock
            with unittest.mock.patch('swebench_runner.docker_run.check_docker_running'):
                with unittest.mock.patch('swebench_runner.docker_run.check_resources'):
                    # Run evaluation
                    result = run_evaluation(
                        str(patch_file),
                        no_input=True,
                        max_patch_size_mb=max_mb
                    )
            
            if result.passed:
                print("  ✅ Passed (no size limit exceeded)")
            else:
                if result.error:
                    # Check which error message we got
                    if "❌ Patch Too Large" in result.error:
                        print("  ❌ Exceeded general limit")
                        # Extract key info from error
                        lines = result.error.split('\n')
                        for line in lines:
                            if "exceeds the general limit" in line:
                                print(f"     {line.strip()}")
                                break
                    elif "⚠️  Patch Size Limitation" in result.error:
                        print("  ⚠️  Exceeded Docker env limit only")
                        # Show the size limits section
                        in_limits_section = False
                        for line in result.error.split('\n'):
                            if "📊 Size Limits Explained:" in line:
                                in_limits_section = True
                            elif in_limits_section and line.strip() == "":
                                break
                            elif in_limits_section:
                                print(f"     {line}")
                    else:
                        print(f"  ❓ Unexpected error: {result.error[:100]}...")
                else:
                    print("  ❓ Failed with no error message")
            
            print()

if __name__ == "__main__":
    test_patch_sizes()