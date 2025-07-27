"""Demo script showing BatchProcessor usage."""

import asyncio
import logging
from pathlib import Path

from swebench_runner.generation import BatchProcessor, PatchGenerator
from swebench_runner.providers import OpenAIProvider, ModelConfig


# Configure logging to see progress
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Demonstrate BatchProcessor capabilities."""

    # Sample SWE-bench instances
    sample_instances = [
        {
            "instance_id": "django__django-123",
            "repo": "django/django",
            "problem_statement": """
The Django QuerySet.update() method doesn't properly handle F() expressions
when used with foreign key relationships. This causes incorrect SQL generation
and can lead to data corruption.

Steps to reproduce:
1. Create models with foreign key relationships
2. Use QuerySet.update() with F() expressions referencing related fields
3. Observe incorrect SQL generation

Expected: Proper JOIN handling in UPDATE statements
Actual: Missing JOINs, causing incorrect updates
            """,
            "hints_text": "Look at the update() method in QuerySet class",
        },
        {
            "instance_id": "requests__requests-456",
            "repo": "psf/requests",
            "problem_statement": """
The requests library doesn't properly handle cookies with domain attributes
that contain leading dots. This affects subdomain matching and can cause
security issues.

Steps to reproduce:
1. Set a cookie with domain=".example.com"
2. Make request to subdomain.example.com
3. Cookie is not sent properly

Expected: Cookie should be sent to all subdomains
Actual: Cookie handling fails for dotted domains
            """,
        },
        {
            "instance_id": "scikit-learn__scikit-learn-789",
            "repo": "scikit-learn/scikit-learn",
            "problem_statement": """
The StandardScaler fit_transform method produces inconsistent results when
called multiple times on the same data due to floating point precision issues.

Steps to reproduce:
1. Create StandardScaler instance
2. Call fit_transform on same data twice
3. Compare results

Expected: Identical results on same input
Actual: Small numerical differences due to precision
            """
        }
    ]

    print("🚀 BatchProcessor Demo")
    print("=" * 50)

    # Create temporary directories for this demo
    demo_dir = Path("./batch_demo")
    checkpoint_dir = demo_dir / "checkpoints"
    results_dir = demo_dir / "results"

    demo_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    try:
        # Initialize model provider (using mock for demo)
        # In practice, you'd use a real provider like OpenAI
        print("📡 Setting up model provider...")

        # For demo purposes, we'll use a mock provider
        # Replace with real provider for actual usage:
        # config = ModelConfig(
        #     model="gpt-4",
        #     api_key="your-api-key",
        #     max_tokens=4000
        # )
        # provider = OpenAIProvider(config)

        from swebench_runner.providers import MockProvider
        config = ModelConfig(model="mock-gpt-4", max_tokens=4000)
        provider = MockProvider(config)

        # Create PatchGenerator
        generator = PatchGenerator(
            provider=provider,
            max_retries=2,
            initial_temperature=0.0
        )

        # Create BatchProcessor with checkpointing
        print("⚙️ Setting up BatchProcessor...")
        processor = BatchProcessor(
            generator=generator,
            checkpoint_dir=checkpoint_dir,
            max_concurrent=2,  # Process 2 instances concurrently
            retry_failed=True,
            max_retries=1,
            progress_bar=True,
            save_intermediate=True,
            checkpoint_interval=2  # Save checkpoint every 2 completions
        )

        # Demonstrate time and cost estimation
        print("\n📊 Batch Estimates:")
        estimated_time = processor.estimate_batch_time(len(sample_instances))
        estimated_cost = processor.estimate_batch_cost(sample_instances)
        print(f"   Estimated time: {estimated_time:.1f} seconds")
        print(f"   Estimated cost: ${estimated_cost:.4f}")

        # Process the batch
        print(f"\n🔄 Processing {len(sample_instances)} instances...")
        print("   (This is a demo with mock responses)")

        result = await processor.process_batch(
            sample_instances,
            resume_from_checkpoint=True,
            save_final_checkpoint=True
        )

        # Display results
        print("\n✅ Batch Processing Complete!")
        print(f"   Successful: {len(result.successful)}")
        print(f"   Failed: {len(result.failed)}")
        print(f"   Skipped: {len(result.skipped)}")
        print(f"   Total cost: ${result.stats.total_cost:.4f}")
        print(f"   Success rate: {result.stats.success_rate:.1%}")
        print(f"   Total time: {result.stats.total_time:.1f} seconds")

        # Show some successful patches
        if result.successful:
            print(f"\n📄 Sample Generated Patches:")
            for i, gen_result in enumerate(result.successful[:2]):  # Show first 2
                print(f"\n   Instance: {gen_result.instance_id}")
                print(f"   Model: {gen_result.model}")
                print(f"   Attempts: {gen_result.attempts}")
                print(f"   Cost: ${gen_result.cost:.4f}")
                print(f"   Patch preview: {gen_result.patch[:200]}...")

        # Show failed instances
        if result.failed:
            print(f"\n❌ Failed Instances:")
            for failure in result.failed:
                instance_id = failure["instance"]["instance_id"]
                error = failure["error"]
                print(f"   {instance_id}: {error}")

        # Generate detailed report
        print("\n📋 Detailed Report:")
        print("-" * 50)
        report = processor.generate_report(result)
        print(report)

        # Save results to file
        results_file = results_dir / "batch_results.json"
        processor.save_results(result, results_file)
        print(f"\n💾 Results saved to: {results_file}")

        # Demonstrate checkpoint functionality
        print(f"\n🔄 Checkpoint Info:")
        checkpoint = processor.load_checkpoint()
        if checkpoint:
            print(f"   Checkpoint exists: {checkpoint.timestamp}")
            print(f"   Completed instances: {len(checkpoint.completed)}")
            print(f"   Failed instances: {len(checkpoint.failed)}")
        else:
            print("   No checkpoint found")

        # Demonstrate resume functionality
        print(f"\n🔄 Simulating Resume from Checkpoint...")
        print("   (Processing same batch again - should skip completed instances)")

        # Process again - should skip all instances
        resume_result = await processor.process_batch(
            sample_instances,
            resume_from_checkpoint=True
        )

        print(f"   Skipped (already completed): {len(resume_result.skipped)}")
        print(f"   Newly processed: {len(resume_result.successful)}")

        # Clean up checkpoint for demo
        print(f"\n🧹 Cleaning up checkpoint...")
        processor.clear_checkpoint()

        print(f"\n🎉 Demo Complete!")
        print(f"   Check {results_dir} for saved results")
        print(f"   Check {checkpoint_dir} for checkpoint files (now cleared)")

    except Exception as e:
        logger.exception(f"Demo failed: {e}")
        print(f"\n❌ Demo failed: {e}")

    finally:
        # Optional: Clean up demo directories
        # import shutil
        # shutil.rmtree(demo_dir, ignore_errors=True)
        pass


def progress_callback(instance_id: str, current: int, total: int):
    """Custom progress callback function."""
    percent = (current / total * 100) if total > 0 else 0
    print(f"🔄 Progress: {current}/{total} ({percent:.1f}%) - Processing {instance_id}")


async def demo_with_custom_progress():
    """Demonstrate BatchProcessor with custom progress tracking."""
    print("\n" + "=" * 50)
    print("🚀 Custom Progress Demo")
    print("=" * 50)

    # Use the PatchGenerator's built-in batch method with custom progress
    from swebench_runner.providers import MockProvider
    config = ModelConfig(model="mock-gpt-4", max_tokens=4000)
    provider = MockProvider(config)

    generator = PatchGenerator(provider=provider)

    # Sample instances
    instances = [
        {"instance_id": f"demo-{i}", "problem_statement": f"Problem {i}"}
        for i in range(3)
    ]

    print("🔄 Processing with custom progress callback...")
    results = await generator.generate_batch(
        instances,
        concurrency=2,
        progress_callback=progress_callback
    )

    successful = sum(1 for r in results if r.success)
    print(f"\n✅ Batch complete: {successful}/{len(results)} successful")


if __name__ == "__main__":
    print("🎯 Running BatchProcessor Demo")

    # Run main demo
    asyncio.run(main())

    # Run progress demo
    asyncio.run(demo_with_custom_progress())

    print("\n🏁 All demos complete!")
