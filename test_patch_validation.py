#!/usr/bin/env python3
"""Test patch size validation messages."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from swebench_runner.models import Patch

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

def test_patch_validation():
    print("=== Testing Patch Validation Error Messages ===\n")
    
    # Test scenarios: (size_kb, max_patch_mb, description)
    test_cases = [
        (100, 5, "Small patch - should pass"),
        (600, 5, "600KB patch - exceeds Docker limit but within general limit"),
        (6000, 5, "6MB patch - exceeds both limits"),
        (600, 0.4, "600KB patch with 0.4MB general limit - exceeds both"),
        (1500, 1, "1.5MB patch with 1MB general limit - exceeds both"),
    ]
    
    for size_kb, max_mb, description in test_cases:
        print(f"Test: {description}")
        print(f"  Patch size: {size_kb}KB, Max limit: {max_mb}MB")
        
        try:
            patch = Patch(
                instance_id=f"test-{size_kb}kb",
                patch=create_patch_of_size(size_kb)
            )
            patch.validate(max_size_mb=max_mb)
            print("  ✅ Passed validation")
        except ValueError as e:
            error_msg = str(e)
            if error_msg.startswith("PATCH_TOO_LARGE:"):
                print("  ❌ Exceeds general limit")
                print(f"     {error_msg.replace('PATCH_TOO_LARGE: ', '')}")
            elif error_msg.startswith("DOCKER_ENV_LIMIT:"):
                print("  ⚠️  Exceeds Docker env limit only")
                print(f"     {error_msg.replace('DOCKER_ENV_LIMIT: ', '')}")
            else:
                print(f"  ❓ Other error: {error_msg}")
        
        print()

if __name__ == "__main__":
    test_patch_validation()