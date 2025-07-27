#!/usr/bin/env python3
"""Demo script showing PatchGenerator usage with the mock provider."""

import asyncio
import logging
from typing import Dict, Any
from swebench_runner.generation import PatchGenerator
from swebench_runner.providers import ModelProvider, ProviderConfig, ModelResponse, ProviderCapabilities

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class DemoMockProvider(ModelProvider):
    """Mock provider for demonstration."""

    name = "demo-mock"
    description = "Mock provider for demos"
    requires_api_key = False

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.generate_responses = []
        self.generate_call_count = 0

    def _init_capabilities(self):
        return ProviderCapabilities(max_context_length=4096)

    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Return predefined responses."""
        if self.generate_call_count < len(self.generate_responses):
            response = self.generate_responses[self.generate_call_count]
            self.generate_call_count += 1
            return response
        else:
            return ModelResponse(
                content="No more responses configured",
                model="demo-mock",
                cost=0.001,
            )

    def estimate_cost(self, prompt_tokens: int, max_tokens: int) -> float:
        return 0.001

    @classmethod
    def _config_from_env(cls, env_vars: Dict[str, str], model: str = None) -> ProviderConfig:
        return ProviderConfig(name="demo-mock", model=model or "demo-model")


async def main():
    """Demonstrate patch generation with mock provider."""

    # Create a mock provider
    config = ProviderConfig(name="demo-mock", model="mock-gpt4")
    provider = DemoMockProvider(config)

    # Configure mock responses
    provider.generate_responses = [
        ModelResponse(
            content="""I'll fix this issue by updating the validation logic.

```diff
--- a/src/validators.py
+++ b/src/validators.py
@@ -42,7 +42,8 @@ class EmailValidator:
     def validate(self, email):
         if not email:
             raise ValueError("Email cannot be empty")
-        if "@" not in email:
+        # Fix: Check for valid email format with regex
+        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
             raise ValueError("Invalid email format")
         return True
```

This adds proper email validation using a regex pattern.""",
            model="mock-gpt4",
            cost=0.003,
            finish_reason="stop",
        )
    ]

    # Create patch generator
    generator = PatchGenerator(provider)

    # Sample SWE-bench instance
    instance = {
        "instance_id": "django-12345",
        "repo": "django/django",
        "problem_statement": """
The EmailValidator in django.core.validators accepts invalid email addresses.
Currently it only checks for the presence of '@' symbol, but doesn't validate
the format properly. Emails like '@invalid' or 'test@' are incorrectly accepted.

The validator should ensure:
1. There's text before the @ symbol
2. There's a domain after the @ symbol
3. The domain has at least one dot followed by a TLD
""",
        "hints_text": "Look at the EmailValidator class in django/core/validators.py",
    }

    # Generate patch
    print("Generating patch for django-12345...")
    result = await generator.generate_patch(instance)

    # Display results
    print("\n" + "="*60)
    print("GENERATION RESULT")
    print("="*60)
    print(f"Instance ID: {result.instance_id}")
    print(f"Model: {result.model}")
    print(f"Success: {result.success}")
    print(f"Attempts: {result.attempts}")
    print(f"Cost: ${result.cost:.4f}")
    print(f"Truncated: {result.truncated}")

    if result.success and result.patch:
        print(f"\nGenerated Patch:\n{'-'*40}")
        print(result.patch)
        print('-'*40)
    else:
        print(f"\nError: {result.error}")

    # Demo batch generation
    print("\n\nDemonstrating batch generation...")
    instances = [
        {"instance_id": f"test-{i}", "problem_statement": f"Fix bug {i}"}
        for i in range(3)
    ]

    # Configure more responses for batch
    provider.generate_responses.extend([
        ModelResponse(
            content=f"--- a/file{i}.py\n+++ b/file{i}.py\n@@ -1 +1 @@\n-bug{i}\n+fixed{i}",
            model="mock-gpt4",
            cost=0.001,
        )
        for i in range(3)
    ])

    # Progress callback
    def progress_callback(instance_id: str, current: int, total: int):
        print(f"  Progress: {instance_id} - {current}/{total}")

    # Generate batch
    results = await generator.generate_batch(
        instances,
        concurrency=2,
        progress_callback=progress_callback
    )

    print(f"\nBatch Results: {len([r for r in results if r.success])}/{len(results)} successful")
    print(f"Total cost: ${sum(r.cost for r in results):.4f}")


if __name__ == "__main__":
    asyncio.run(main())
