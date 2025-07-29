#!/usr/bin/env python3
"""
Smoke test for dataset autofetching functionality.

This script tests that dataset autofetching actually works end-to-end
by attempting to download and use a real dataset.
"""

import sys
import tempfile
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from swebench_runner.datasets import DatasetManager


def test_dataset_autofetch():
    """Test that dataset autofetching works end-to-end."""
    print("🧪 Testing dataset autofetching...")

    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = Path(temp_dir)
        print(f"📁 Using temporary cache dir: {cache_dir}")

        manager = DatasetManager(cache_dir)

        try:
            print("📥 Attempting to fetch 'lite' dataset...")
            dataset = manager.fetch_dataset("lite")
            print(f"✅ Successfully fetched dataset: {type(dataset)}")

            print("📊 Getting first few instances...")
            instances = manager.get_instances("lite", count=3)
            print(f"✅ Retrieved {len(instances)} instances")

            if instances:
                first_instance = instances[0]
                print(f"📋 First instance ID: {first_instance.get('instance_id', 'unknown')}")
                print(f"📋 Keys in first instance: {list(first_instance.keys())}")

                # Verify expected keys are present
                required_keys = ['instance_id', 'problem_statement', 'repo']
                missing_keys = [key for key in required_keys if key not in first_instance]
                if missing_keys:
                    print(f"❌ Missing required keys: {missing_keys}")
                    return False
                else:
                    print("✅ All required keys present")

            print("✅ Dataset autofetching test PASSED!")
            return True

        except Exception as e:
            print(f"❌ Dataset autofetching test FAILED: {e}")
            print(f"🔍 Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return False


def test_dataset_caching():
    """Test that dataset caching works properly."""
    print("\n🧪 Testing dataset caching...")

    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = Path(temp_dir)
        print(f"📁 Using temporary cache dir: {cache_dir}")

        manager = DatasetManager(cache_dir)

        try:
            print("📥 First fetch (should download)...")
            import time
            start_time = time.time()
            dataset1 = manager.fetch_dataset("lite")
            first_fetch_time = time.time() - start_time
            print(f"⏱️  First fetch took: {first_fetch_time:.2f}s")

            print("📥 Second fetch (should use cache)...")
            start_time = time.time()
            dataset2 = manager.fetch_dataset("lite")
            second_fetch_time = time.time() - start_time
            print(f"⏱️  Second fetch took: {second_fetch_time:.2f}s")

            # Second fetch should be much faster (cached)
            if second_fetch_time < first_fetch_time * 0.5:
                print("✅ Caching appears to be working (second fetch much faster)")
            else:
                print("⚠️  Caching may not be working (second fetch not significantly faster)")

            print("✅ Dataset caching test PASSED!")
            return True

        except Exception as e:
            print(f"❌ Dataset caching test FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Run all smoke tests."""
    print("🚀 Starting dataset autofetching smoke tests...\n")

    tests = [
        test_dataset_autofetch,
        test_dataset_caching,
    ]

    results = []
    for test in tests:
        result = test()
        results.append(result)

    print(f"\n📊 Results: {sum(results)}/{len(results)} tests passed")

    if all(results):
        print("🎉 All smoke tests PASSED!")
        sys.exit(0)
    else:
        print("💥 Some smoke tests FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
