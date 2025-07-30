#!/usr/bin/env python3
"""Example of using the rate limiting coordinator.

This example demonstrates how to set up rate limiting for different providers
and coordinate them through the RateLimitCoordinator.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for example
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from swebench_runner.providers.rate_limiters import (
    AcquisitionRequest,
    RateLimitConfig,
    RateLimitCoordinator,
    create_anthropic_limiter,
    create_coordinator_from_config,
    create_ollama_limiter,
    create_openai_limiter,
)


async def basic_example():
    """Basic example of rate limiting usage."""
    print("=== Basic Rate Limiting Example ===")

    # Create rate limiters for different providers
    openai_limiter = create_openai_limiter(requests_per_minute=100, tokens_per_minute=5000)
    anthropic_limiter = create_anthropic_limiter(tokens_per_minute=3000)
    ollama_limiter = create_ollama_limiter(concurrent_requests=2)

    # Set up coordinator
    coordinator = RateLimitCoordinator()
    coordinator.add_provider_limiter("openai", openai_limiter)
    coordinator.add_provider_limiter("anthropic", anthropic_limiter)
    coordinator.add_provider_limiter("ollama", ollama_limiter)

    # Make some requests
    providers = ["openai", "anthropic", "ollama"]
    requests = [
        AcquisitionRequest(estimated_tokens=100),
        AcquisitionRequest(estimated_tokens=200),
        AcquisitionRequest(estimated_tokens=50),
    ]

    for provider, request in zip(providers, requests):
        result = await coordinator.acquire(provider, request)
        if result.acquired:
            print(f"✓ {provider}: Acquired {request.estimated_tokens} tokens")
            # Simulate work
            await asyncio.sleep(0.1)
            # Release resources
            coordinator.release(provider, request.estimated_tokens)
        else:
            print(f"✗ {provider}: Rate limited - {result.limited_by}")

    # Get status
    status = coordinator.get_status()
    print(f"\nFinal status: {status}")


async def config_based_example():
    """Example using configuration-based setup."""
    print("\n=== Configuration-Based Example ===")

    # Define rate limits via configuration
    config = {
        "openai": RateLimitConfig(
            requests_per_minute=50,
            tokens_per_minute=2000
        ),
        "anthropic": RateLimitConfig(
            tokens_per_minute=1500
        ),
        "custom_provider": RateLimitConfig(
            requests_per_minute=20,
            concurrent_requests=1
        ),
    }

    # Create coordinator from config
    coordinator = create_coordinator_from_config(config)

    # Test each provider
    for provider in config.keys():
        request = AcquisitionRequest(estimated_tokens=50)
        result = await coordinator.acquire(provider, request)

        print(f"{provider}: {'✓ Acquired' if result.acquired else '✗ Rate limited'}")

        if result.acquired:
            coordinator.release(provider, request.estimated_tokens)


async def burst_handling_example():
    """Example demonstrating burst handling."""
    print("\n=== Burst Handling Example ===")

    # Create a token bucket with burst allowance
    coordinator = RateLimitCoordinator()

    # Anthropic limiter has 10% burst allowance by default
    coordinator.add_provider_limiter("anthropic", create_anthropic_limiter(1000))

    # Make several requests quickly (simulating burst)
    burst_requests = [100, 150, 200, 250, 300]  # Total: 1000 tokens

    print("Making burst requests...")
    for i, tokens in enumerate(burst_requests):
        request = AcquisitionRequest(estimated_tokens=tokens)
        result = await coordinator.acquire("anthropic", request)

        if result.acquired:
            print(f"  Request {i+1}: ✓ {tokens} tokens acquired")
            coordinator.release("anthropic", tokens)
        else:
            print(f"  Request {i+1}: ✗ {tokens} tokens rate limited")
            print(f"    Wait time: {result.wait_time:.2f}s")


async def global_limiter_example():
    """Example with global rate limiting."""
    print("\n=== Global Rate Limiting Example ===")

    coordinator = RateLimitCoordinator()

    # Set a global limit that applies to all providers
    global_limiter = create_openai_limiter(requests_per_minute=10, tokens_per_minute=500)
    coordinator.set_global_limiter(global_limiter)

    # Add provider-specific limiters
    coordinator.add_provider_limiter("openai", create_openai_limiter(100, 5000))
    coordinator.add_provider_limiter("anthropic", create_anthropic_limiter(3000))

    # Make requests - they must pass both global AND provider limits
    providers = ["openai", "anthropic", "unknown_provider"]

    for provider in providers:
        request = AcquisitionRequest(estimated_tokens=100)
        result = await coordinator.acquire(provider, request)

        if result.acquired:
            limiters = result.metadata.get("limiters_acquired", [])
            print(f"✓ {provider}: Passed limiters: {limiters}")
            coordinator.release(provider, request.estimated_tokens)
        else:
            failed_limiter = result.metadata.get("failed_limiter", "unknown")
            print(f"✗ {provider}: Blocked by {failed_limiter} limiter")


async def main():
    """Run all examples."""
    await basic_example()
    await config_based_example()
    await burst_handling_example()
    await global_limiter_example()


if __name__ == "__main__":
    asyncio.run(main())
