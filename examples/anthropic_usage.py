#!/usr/bin/env python3
"""Example usage of the Anthropic provider with unified interface.

This example demonstrates how to use the new Anthropic provider
with both the unified interface and legacy compatibility.

Requirements:
- Python 3.10+
- ANTHROPIC_API_KEY environment variable set
"""

import asyncio
import os
from swebench_runner.providers import AnthropicProvider, ProviderConfig
from swebench_runner.providers.unified_models import UnifiedRequest


async def main():
    """Demonstrate Anthropic provider usage."""
    
    print("Anthropic Provider Usage Example")
    print("=" * 40)
    
    # Check for API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("⚠️  ANTHROPIC_API_KEY not set - this example will show setup only")
        print("\nTo use this example:")
        print("1. Set your API key: export ANTHROPIC_API_KEY='your-key-here'")
        print("2. Run this script again")
        show_setup_only()
        return
    
    # Create provider configuration
    config = ProviderConfig(
        name="anthropic",
        api_key=api_key,
        model="claude-sonnet-4-20250514",
        temperature=0.7,
        max_tokens=1000,
        timeout=120
    )
    
    # Initialize provider
    provider = AnthropicProvider(config)
    print(f"✓ Initialized {provider.name} provider")
    print(f"  Model: {provider.config.model}")
    print(f"  Supports streaming: {provider.supports_streaming}")
    print(f"  Token limit: {provider.get_token_limit():,}")
    
    # Test connection
    print("\n1. Testing connection...")
    try:
        is_connected = await provider.validate_connection()
        if is_connected:
            print("✓ Connection successful")
        else:
            print("✗ Connection failed")
            return
    except Exception as e:
        print(f"✗ Connection error: {e}")
        return
    
    # Example 1: Basic unified interface
    print("\n2. Basic generation (unified interface)...")
    try:
        request = UnifiedRequest(
            prompt="Explain what SWE-bench is in one sentence.",
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            temperature=0.7
        )
        
        response = await provider.generate_unified(request)
        print(f"✓ Response: {response.content}")
        print(f"  Tokens: {response.usage.prompt_tokens} → {response.usage.completion_tokens}")
        print(f"  Cost: ${response.cost:.6f}")
        print(f"  Latency: {response.latency_ms}ms")
    except Exception as e:
        print(f"✗ Generation error: {e}")
    
    # Example 2: Legacy interface compatibility
    print("\n3. Legacy interface compatibility...")
    try:
        response = await provider.generate(
            "What is the capital of France?",
            max_tokens=50,
            temperature=0.3
        )
        
        print(f"✓ Response: {response.content}")
        print(f"  Usage: {response.usage}")
        print(f"  Cost: ${response.cost:.6f}")
    except Exception as e:
        print(f"✗ Legacy generation error: {e}")
    
    # Example 3: System message usage
    print("\n4. System message example...")
    try:
        request = UnifiedRequest(
            prompt="How do I write a good commit message?",
            system_message="You are a senior software engineer helping a junior developer.",
            model="claude-sonnet-4-20250514",
            max_tokens=150,
            temperature=0.5
        )
        
        response = await provider.generate_unified(request)
        print(f"✓ Response: {response.content}")
        print(f"  Tokens used: {response.usage.total_tokens}")
    except Exception as e:
        print(f"✗ System message error: {e}")
    
    # Example 4: Streaming response
    print("\n5. Streaming example...")
    try:
        stream_request = UnifiedRequest(
            prompt="Count from 1 to 5, with each number on a new line.",
            model="claude-sonnet-4-20250514",
            max_tokens=50,
            stream=True
        )
        
        print("Streaming response: ", end="", flush=True)
        async for chunk in provider.generate_stream(stream_request):
            if chunk.delta:
                print(chunk.delta, end="", flush=True)
            if chunk.done:
                print(f"\n✓ Streaming complete")
                break
    except Exception as e:
        print(f"✗ Streaming error: {e}")
    
    # Example 5: Cost estimation
    print("\n6. Cost estimation...")
    try:
        test_request = UnifiedRequest(
            prompt="Write a comprehensive guide to Python testing with pytest.",
            model="claude-opus-4-20250514",  # More expensive model
            max_tokens=2000
        )
        
        estimated_cost = await provider.estimate_cost_unified(test_request)
        print(f"✓ Estimated cost for long generation: ${estimated_cost:.6f}")
        
        # Compare models
        for model in ["claude-haiku-3-5-20241022", "claude-sonnet-4-20250514", "claude-opus-4-20250514"]:
            test_request.model = model
            cost = await provider.estimate_cost_unified(test_request)
            print(f"  {model}: ${cost:.6f}")
    except Exception as e:
        print(f"✗ Cost estimation error: {e}")
    
    # Example 6: Rate limit information
    print("\n7. Rate limit status...")
    try:
        rate_info = provider.get_rate_limit_info()
        print(f"✓ Rate limit info: {rate_info['unified_status']}")
    except Exception as e:
        print(f"✗ Rate limit info error: {e}")
    
    print("\n✓ All examples completed!")


def show_setup_only():
    """Show provider setup without making API calls."""
    
    print("\nProvider Setup Example (no API calls)")
    print("-" * 40)
    
    # Show configuration options
    config = ProviderConfig(
        name="anthropic",
        api_key="your-api-key-here",
        model="claude-sonnet-4-20250514",
        temperature=0.7,
        max_tokens=1000,
        timeout=120,
        extra_params={
            "anthropic_beta": "message-batches-2024-09-24"  # Optional beta features
        }
    )
    
    print("Configuration options:")
    print(f"  name: {config.name}")
    print(f"  model: {config.model}")
    print(f"  temperature: {config.temperature}")
    print(f"  max_tokens: {config.max_tokens}")
    print(f"  timeout: {config.timeout}")
    print(f"  extra_params: {config.extra_params}")
    
    print("\nSupported models:")
    models = [
        "claude-opus-4-20250514",      # $15 input, $75 output per 1M tokens
        "claude-sonnet-4-20250514",    # $3 input, $15 output per 1M tokens  
        "claude-haiku-3-5-20241022",   # $0.8 input, $4 output per 1M tokens
    ]
    
    for model in models:
        print(f"  - {model}")
    
    print("\nKey features:")
    features = [
        "Unified interface with UnifiedRequest/UnifiedResponse",
        "Legacy ModelProvider interface compatibility", 
        "Streaming support with SSE parsing",
        "API-based token counting",
        "Rate limiting with token bucket",
        "Circuit breaker for fault tolerance",
        "Cost estimation with 2025 pricing",
        "System message separation (Anthropic format)",
        "Required max_tokens handling"
    ]
    
    for feature in features:
        print(f"  ✓ {feature}")


if __name__ == "__main__":
    asyncio.run(main())