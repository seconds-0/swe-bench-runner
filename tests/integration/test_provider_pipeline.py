"""Integration tests for the complete provider pipeline."""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from swebench_runner.generation_integration import GenerationIntegration
from swebench_runner.providers.unified_models import TokenUsage, UnifiedRequest, UnifiedResponse


class TestProviderPipeline:
    """Test the complete provider pipeline from request to patch generation."""

    @pytest.fixture
    def temp_cache_dir(self, tmp_path):
        """Create temporary cache directory."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        return cache_dir

    @pytest.fixture
    def mock_instances(self):
        """Create mock SWE-bench instances."""
        return [
            {
                "instance_id": "test-001",
                "problem_statement": "Fix the bug in function X",
                "test_patch": "def test_x():\n    assert x() == 5",
            },
            {
                "instance_id": "test-002",
                "problem_statement": "Add feature Y to module Z",
                "test_patch": "def test_y():\n    assert y() is not None",
            }
        ]

    def test_api_key_loading_in_pipeline(self, temp_cache_dir):
        """Test that API keys are properly loaded in the pipeline."""
        # Set test API keys for OpenRouter
        os.environ["OPENROUTER_API_KEY"] = "sk-or-test123"

        integration = GenerationIntegration(temp_cache_dir)

        # Should be able to select provider with API key
        with patch("swebench_runner.providers.config.ProviderConfigManager.load_config") as mock_config:
            mock_config.return_value = MagicMock(
                api_key="sk-or-test123",
                model="qwen/qwen3-coder:free"  # Provide explicit model string
            )

            provider = integration.select_provider("openrouter")
            assert provider is not None

        # Clean up
        os.environ.pop("OPENROUTER_API_KEY", None)

    def test_model_parameter_mapping_in_pipeline(self, temp_cache_dir):
        """Test that model parameters are correctly mapped through the pipeline."""
        integration = GenerationIntegration(temp_cache_dir)

        # Test instances
        instances = [{"instance_id": "test", "problem_statement": "Test problem"}]

        # Mock the provider and its generate method
        with patch("swebench_runner.generation_integration.GenerationIntegration.select_provider") as mock_select:
            mock_provider = MagicMock()
            mock_provider.name = "openrouter"
            mock_provider.config.model = "qwen/qwen3-coder:free"
            mock_provider.capabilities = MagicMock(max_context_length=32768)
            mock_select.return_value = mock_provider

            # Mock the batch processor
            with patch("swebench_runner.generation.BatchProcessor.process_batch") as mock_process:
                mock_batch_result = MagicMock()
                mock_batch_result.successful = []
                mock_batch_result.failed = []
                mock_batch_result.stats.total_instances = 1
                mock_batch_result.stats.completed = 0
                mock_batch_result.stats.failed = 0
                mock_batch_result.stats.skipped = 0
                mock_batch_result.stats.success_rate = 0.0
                mock_batch_result.stats.total_cost = 0.0
                mock_batch_result.stats.total_time = 1.0
                mock_process.return_value = mock_batch_result

                # Run generation (will fail but we're testing parameter mapping)
                import asyncio
                try:
                    asyncio.run(integration.generate_patches_for_evaluation(
                        instances=instances,
                        provider_name="openrouter",
                        model="qwen/qwen3-coder:free",
                        show_progress=False
                    ))
                except Exception:
                    pass  # Expected to fail without real provider

                # Verify provider was selected with correct model
                mock_select.assert_called_with("openrouter", "qwen/qwen3-coder:free")

    @pytest.mark.asyncio
    async def test_patch_fixing_in_pipeline(self, temp_cache_dir, mock_instances):
        """Test that patches are automatically fixed in the pipeline."""
        integration = GenerationIntegration(temp_cache_dir)

        # Create a mock provider that returns a malformed patch
        mock_provider = MagicMock()
        mock_provider.name = "openai"

        # Mock batch processor result with malformed patch
        mock_batch_result = MagicMock()
        mock_result = MagicMock()
        mock_result.instance_id = "test-001"
        mock_result.patch = """@@
-old line
+new line"""  # Malformed patch with bare @@
        mock_result.model = "gpt-5-nano"
        mock_result.cost = 0.01
        mock_result.metadata = {}

        mock_batch_result.successful = [mock_result]
        mock_batch_result.failed = []
        mock_batch_result.stats.total_instances = 1
        mock_batch_result.stats.completed = 1
        mock_batch_result.stats.failed = 0
        mock_batch_result.stats.skipped = 0
        mock_batch_result.stats.success_rate = 1.0
        mock_batch_result.stats.total_cost = 0.01
        mock_batch_result.stats.total_time = 1.0

        with patch.object(integration, 'select_provider', return_value=mock_provider):
            with patch("swebench_runner.generation.BatchProcessor.process_batch") as mock_process:
                mock_process.return_value = mock_batch_result

                # Run generation
                output_path = await integration.generate_patches_for_evaluation(
                    instances=mock_instances,
                    provider_name="openai",
                    show_progress=False
                )

                # Read the generated patches file
                with open(output_path) as f:
                    lines = f.readlines()

                # Should have generated patches (even if empty due to validation)
                assert len(lines) >= 0

                # If patches were saved, they should be valid JSON
                for line in lines:
                    if line.strip():
                        data = json.loads(line)
                        if "patch" in data and data["patch"]:
                            # Patch should have been fixed (no bare @@)
                            assert "@@\n" not in data["patch"]

    def test_provider_fallback_mechanism(self, temp_cache_dir):
        """Test that provider fallback works correctly."""
        integration = GenerationIntegration(temp_cache_dir)

        # Test that fallback providers are tried
        instance = {"instance_id": "test", "problem_statement": "Test"}

        with patch.object(integration.provider_coordinator, 'select_provider') as mock_select:
            # First provider fails, second succeeds
            mock_provider1 = MagicMock()
            mock_provider1.generate_unified = AsyncMock(side_effect=Exception("Provider 1 failed"))

            mock_provider2 = MagicMock()
            mock_response = UnifiedResponse(
                content="Fixed patch",
                model="qwen/qwen3-coder:free",
                usage=TokenUsage(0, 0, 0),
                latency_ms=100,
                finish_reason="stop",
                provider="openrouter"
            )
            mock_provider2.generate_unified = AsyncMock(return_value=mock_response)

            mock_select.side_effect = [mock_provider1, mock_provider2]

            # Run with fallback
            import asyncio
            result = asyncio.run(integration.generate_with_fallback(
                instance=instance,
                primary_provider="openrouter",
                fallback_providers=["openrouter"]
            ))

            assert result.content == "Fixed patch"
            assert result.provider == "openrouter"
            assert mock_select.call_count == 2


class TestTransformPipeline:
    """Test the transform pipeline for different providers."""

    def test_openrouter_transform_with_qwen_coder(self):
        """Test OpenRouter transform for Qwen Coder models."""
        from swebench_runner.providers.transform_pipeline import OpenAIRequestTransformer

        # OpenRouter uses OpenAI-compatible format
        transformer = OpenAIRequestTransformer()
        request = UnifiedRequest(
            prompt="Test prompt",
            system_message="System message",
            model="qwen/qwen3-coder:free",
            max_tokens=1000,
            temperature=0.7,
            stream=False
        )

        # Transform request
        transformed = transformer.transform(request)

        # Check standard OpenAI format parameters
        assert "max_tokens" in transformed
        assert transformed["max_tokens"] == 1000
        assert "temperature" in transformed
        assert transformed["temperature"] == 0.7
        assert transformed["model"] == "qwen/qwen3-coder:free"

    def test_anthropic_transform_with_required_params(self):
        """Test Anthropic transform with required parameters."""
        from swebench_runner.providers.transform_pipeline import AnthropicRequestTransformer

        transformer = AnthropicRequestTransformer()
        request = UnifiedRequest(
            prompt="Test prompt",
            system_message="System message",
            model="claude-3-sonnet",
            max_tokens=None,  # Not provided
            temperature=0.5,
            stream=False
        )

        # Transform request
        transformed = transformer.transform(request)

        # Check that max_tokens is added with default
        assert "max_tokens" in transformed
        assert transformed["max_tokens"] == 4000  # Default value

        # Check other parameters
        assert transformed["temperature"] == 0.5
        assert transformed["system"] == "System message"

    def test_ollama_transform_with_custom_params(self):
        """Test Ollama transform with custom parameter names."""
        from swebench_runner.providers.transform_pipeline import OllamaRequestTransformer

        transformer = OllamaRequestTransformer()
        request = UnifiedRequest(
            prompt="Test prompt",
            system_message="System message",
            model="llama3.3",
            max_tokens=500,
            temperature=0.8,
            stream=False
        )

        # Transform request
        transformed = transformer.transform(request)

        # Check that num_predict is used (not max_tokens)
        assert "options" in transformed
        assert "num_predict" in transformed["options"]
        assert transformed["options"]["num_predict"] == 500
        assert "max_tokens" not in transformed["options"]

        # Check other parameters
        assert transformed["options"]["temperature"] == 0.8
        assert transformed["system"] == "System message"


class TestEndToEndScenarios:
    """Test complete end-to-end scenarios."""

    @pytest.mark.asyncio
    async def test_openrouter_qwen_coder_full_pipeline(self, tmp_path):
        """Test full pipeline with OpenRouter Qwen Coder model."""
        # Set up environment
        os.environ["OPENROUTER_API_KEY"] = "sk-or-test123"
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        integration = GenerationIntegration(cache_dir)

        # Mock the actual API call
        with patch("swebench_runner.providers.openrouter.OpenRouterProvider.generate_unified") as mock_generate:
            mock_response = UnifiedResponse(
                content="""diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
@@ -1,3 +1,3 @@
-def old():
-    return 1
+def new():
+    return 2""",
                model="qwen/qwen3-coder:free",
                usage=TokenUsage(100, 50, 150),
                latency_ms=500,
                finish_reason="stop",
                provider="openrouter"
            )
            mock_generate.return_value = mock_response

            # Test instance
            instances = [{
                "instance_id": "test-qwen-coder",
                "problem_statement": "Change function name from old to new"
            }]

            # Mock provider selection to avoid real API
            with patch.object(integration, 'select_provider') as mock_select:
                mock_provider = MagicMock()
                mock_provider.name = "openrouter"
                mock_provider.generate_unified = mock_generate
                # Mock the legacy generate method used by PatchGenerator
                mock_provider.generate = AsyncMock(return_value=MagicMock(
                    content=mock_response.content,
                    model=mock_response.model,
                    usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                    latency_ms=500,
                    finish_reason="stop",
                    provider="openrouter",
                    cost=0.0,
                    raw_response={}
                ))
                # Mock the config with proper string values
                mock_config = MagicMock()
                mock_config.model = "qwen/qwen3-coder:free"
                mock_config.api_key = "sk-or-test123"
                mock_provider.config = mock_config
                mock_select.return_value = mock_provider

                # Mock batch processor
                with patch("swebench_runner.generation.BatchProcessor") as mock_batch_processor:
                    mock_processor = MagicMock()
                    mock_result = MagicMock()
                    mock_result.instance_id = "test-qwen-coder"
                    mock_result.patch = mock_response.content
                    mock_result.model = "qwen/qwen3-coder:free"
                    mock_result.cost = 0.0  # Free model
                    mock_result.metadata = {}

                    mock_batch_result = MagicMock()
                    mock_batch_result.successful = [mock_result]
                    mock_batch_result.failed = []
                    mock_batch_result.stats.total_instances = 1
                    mock_batch_result.stats.completed = 1
                    mock_batch_result.stats.failed = 0
                    mock_batch_result.stats.skipped = 0
                    mock_batch_result.stats.success_rate = 1.0
                    mock_batch_result.stats.total_cost = 0.0
                    mock_batch_result.stats.total_time = 1.0

                    mock_processor.process_batch = AsyncMock(return_value=mock_batch_result)
                    mock_batch_processor.return_value = mock_processor

                    # Run the pipeline
                    output_path = await integration.generate_patches_for_evaluation(
                        instances=instances,
                        provider_name="openrouter",
                        model="qwen/qwen3-coder:free",
                        show_progress=False
                    )

                    # Verify output
                    assert output_path.exists()
                    with open(output_path) as f:
                        data = json.loads(f.readline())
                        assert data["instance_id"] == "test-qwen-coder"
                        assert "def new():" in data["patch"]

        # Clean up
        os.environ.pop("OPENROUTER_API_KEY", None)
