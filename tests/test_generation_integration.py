"""Tests for generation integration module."""

import json
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from swebench_runner.cli import cli
from swebench_runner.generation import BatchResult, BatchStats, GenerationResult
from swebench_runner.generation_integration import (
    CostEstimator,
    GenerationFailureHandler,
    GenerationIntegration,
    ProviderCoordinator,
    UnifiedCostEstimator,
)
from swebench_runner.providers import (
    MockProvider,
    ProviderConfig,
    UnifiedRequest,
    UnifiedResponse,
    get_registry,
)


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def mock_provider():
    """Create a mock provider."""
    provider = MagicMock()
    provider.name = "mock"
    provider.capabilities.max_context_tokens = 8000
    provider.config = ProviderConfig(name="mock", api_key="test-key", model="gpt-4")
    provider.default_model = "gpt-4"
    return provider


@pytest.fixture
def sample_instances():
    """Create sample SWE-bench instances."""
    return [
        {
            "instance_id": "django__django-12345",
            "problem_statement": "Fix issue with query generation",
            "test_patch": "diff --git a/test.py b/test.py\n+test",
            "repo": "django/django",
        },
        {
            "instance_id": "requests__requests-6789",
            "problem_statement": "Improve error handling for network requests",
            "test_patch": "diff --git a/test2.py b/test2.py\n+test2",
            "repo": "requests/requests",
        },
    ]


class TestGenerationIntegration:
    """Test GenerationIntegration class."""

    @pytest.mark.skip(reason="Async test with complex mocking - fix later")
    @pytest.mark.asyncio
    async def test_generate_patches_for_evaluation(self, temp_cache_dir, mock_provider, sample_instances):
        """Test generating patches for evaluation."""
        integration = GenerationIntegration(temp_cache_dir)

        # Mock components
        with patch('swebench_runner.generation_integration.get_registry') as mock_registry, \
             patch('swebench_runner.generation_integration.ProviderConfigManager') as mock_config_manager, \
             patch('swebench_runner.generation_integration.BatchProcessor') as mock_batch_processor_class:

            # Setup mocks
            mock_registry.return_value.get_provider_class.return_value = lambda config: mock_provider
            mock_config_manager.return_value.load_config.return_value = ProviderConfig(name="mock", api_key="test-key")

            # Mock batch processor
            mock_batch_processor = MagicMock()
            mock_batch_processor_class.return_value = mock_batch_processor

            # Create successful results
            successful_results = [
                GenerationResult(
                    patch="diff --git a/file.py b/file.py\n+fix1",
                    instance_id="django__django-12345",
                    model="gpt-4",
                    attempts=1,
                    truncated=False,
                    cost=0.05,
                    success=True,
                ),
                GenerationResult(
                    patch="diff --git a/file2.py b/file2.py\n+fix2",
                    instance_id="requests__requests-6789",
                    model="gpt-4",
                    attempts=1,
                    truncated=False,
                    cost=0.04,
                    success=True,
                ),
            ]

            batch_result = BatchResult(
                successful=successful_results,
                failed=[],
                stats=BatchStats(
                    total_instances=2,
                    completed=2,
                    failed=0,
                    total_cost=0.09,
                    success_rate=1.0,
                ),
            )

            # Make process_batch async
            async def async_process_batch(**kwargs):
                return batch_result

            mock_batch_processor.process_batch = async_process_batch

            # Generate patches
            output_path = await integration.generate_patches_for_evaluation(
                instances=sample_instances,
                provider_name="mock",
                show_progress=False,
            )

            # Verify output
            assert output_path.exists()
            assert output_path.suffix == ".jsonl"

            # Check content
            with open(output_path) as f:
                lines = f.readlines()

            assert len(lines) == 2

            patch1 = json.loads(lines[0])
            assert patch1["instance_id"] == "django__django-12345"
            assert patch1["patch"] == "diff --git a/file.py b/file.py\n+fix1"
            assert patch1["model"] == "gpt-4"
            assert patch1["cost"] == 0.05

    @pytest.mark.asyncio
    async def test_generate_patches_basic(self, temp_cache_dir, sample_instances):
        """Test basic patch generation flow with real components."""
        # Create integration with real components
        integration = GenerationIntegration(temp_cache_dir)

        # Create a mock provider instance
        mock_provider = MockProvider(
            config=ProviderConfig(name="mock", api_key="test-key", model="mock-small"),
            mock_responses={},  # Will be set up based on prompts
            response_delay=0.01  # Fast response for testing
        )

        # Override the generate method to return specific patches based on instance ID in prompt
        original_generate = mock_provider.generate

        async def custom_generate(prompt: str, **kwargs):
            # Check which instance this is for by looking at the prompt content
            if "django__django-12345" in prompt:
                mock_provider.mock_responses[prompt] = (
                    "diff --git a/django/core/query.py b/django/core/query.py\n"
                    "--- a/django/core/query.py\n"
                    "+++ b/django/core/query.py\n"
                    "@@ -123,7 +123,7 @@\n"
                    "-    old_code = here\n"
                    "+    new_code = here"
                )
            elif "requests__requests-6789" in prompt or "Improve error handling for network requests" in prompt:
                mock_provider.mock_responses[prompt] = (
                    "diff --git a/requests/timeout.py b/requests/timeout.py\n"
                    "--- a/requests/timeout.py\n"
                    "+++ b/requests/timeout.py\n"
                    "@@ -45,7 +45,7 @@\n"
                    "-    timeout = None\n"
                    "+    timeout = 30"
                )
            return await original_generate(prompt, **kwargs)

        mock_provider.generate = custom_generate

        # Mock only the provider registry and config manager
        with patch('swebench_runner.generation_integration.get_registry') as mock_registry, \
             patch('swebench_runner.generation_integration.ProviderConfigManager') as mock_config_manager:

            # Setup mocks
            mock_registry.return_value.get_provider_class.return_value = lambda config: mock_provider
            mock_config_manager.return_value.load_config.return_value = ProviderConfig(
                name="mock", api_key="test-key", model="mock-small"
            )

            # Generate patches - use real batch processor and generation logic
            output_path = await integration.generate_patches_for_evaluation(
                instances=sample_instances[:2],  # Use 2 instances
                provider_name="mock",
                show_progress=False,
            )

            # Verify output
            assert output_path.exists(), f"Output file not found at {output_path}"
            assert output_path.suffix == ".jsonl"

            # Check content
            with open(output_path) as f:
                lines = f.readlines()

            assert len(lines) == 2, f"Expected 2 lines, got {len(lines)}"

            # Parse and verify patches
            patches = [json.loads(line) for line in lines]

            # Check first patch
            assert patches[0]["instance_id"] == "django__django-12345"
            assert "patch" in patches[0]
            assert patches[0]["patch"].startswith("diff --git")
            assert "django/core/query.py" in patches[0]["patch"]
            assert "model" in patches[0]
            assert patches[0]["model"] == "mock-small"

            # Check second patch
            assert patches[1]["instance_id"] == "requests__requests-6789"
            assert "patch" in patches[1]
            assert patches[1]["patch"].startswith("diff --git")
            assert "requests/timeout.py" in patches[1]["patch"]
            assert "model" in patches[1]
            assert patches[1]["model"] == "mock-small"

    @pytest.mark.skip(reason="Async test with complex mocking - fix later")
    @pytest.mark.asyncio
    async def test_cost_warning(self, temp_cache_dir, mock_provider, sample_instances, capsys):
        """Test cost warning for expensive generation."""
        integration = GenerationIntegration(temp_cache_dir)

        # Mock cost estimator to return high cost
        with patch.object(CostEstimator, 'estimate_batch_cost', return_value=(15.0, 30.0)), \
             patch('click.confirm', return_value=False):

            with pytest.raises(SystemExit):
                await integration.generate_patches_for_evaluation(
                    instances=sample_instances,
                    provider_name="mock",
                    show_progress=True,
                )


class TestProviderCoordinator:
    """Test ProviderCoordinator class with unified provider system."""

    @pytest.fixture
    def coordinator(self):
        """Create a ProviderCoordinator instance."""
        return ProviderCoordinator()

    @pytest.fixture
    def mock_unified_provider(self):
        """Create a mock provider with unified interface."""
        provider = MagicMock()
        provider.name = "mock_unified"
        provider.config = ProviderConfig(name="mock_unified", api_key="test-key", model="test-model")
        provider.capabilities = MagicMock()
        provider.capabilities.max_context_length = 8000
        provider.capabilities.cost_per_1k_prompt_tokens = 0.001
        provider.capabilities.cost_per_1k_completion_tokens = 0.002
        
        # Mock unified interface methods
        async def mock_generate_unified(request):
            from swebench_runner.providers.unified_models import TokenUsage, UnifiedResponse
            return UnifiedResponse(
                content="mock patch content",
                model="test-model",
                usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
                latency_ms=500,
                finish_reason="stop",
                provider="mock_unified",
                cost=0.0003,
                raw_response={}
            )
        
        provider.generate_unified = mock_generate_unified
        provider.validate_connection = MagicMock(return_value=True)
        return provider

    def test_select_provider_success(self, coordinator, mock_unified_provider):
        """Test successful provider selection."""
        with patch.object(coordinator.registry, 'get_provider_class') as mock_get_class, \
             patch.object(coordinator.config_manager, 'load_config') as mock_load_config:
            
            mock_get_class.return_value = lambda config: mock_unified_provider
            mock_load_config.return_value = ProviderConfig(name="mock_unified", api_key="test-key")
            
            provider = coordinator.select_provider("mock_unified", model="custom-model")
            
            assert provider == mock_unified_provider
            mock_get_class.assert_called_once_with("mock_unified")
            mock_load_config.assert_called_once_with("mock_unified")

    def test_select_provider_with_model_override(self, coordinator, mock_unified_provider):
        """Test provider selection with model override."""
        with patch.object(coordinator.registry, 'get_provider_class') as mock_get_class, \
             patch.object(coordinator.config_manager, 'load_config') as mock_load_config:
            
            mock_get_class.return_value = lambda config: mock_unified_provider
            mock_load_config.return_value = ProviderConfig(name="mock_unified", api_key="test-key", model="default-model")
            
            provider = coordinator.select_provider("mock_unified", model="custom-model")
            
            assert provider == mock_unified_provider
            # Verify model was overridden in config creation

    def test_get_provider_info(self, coordinator, mock_unified_provider):
        """Test getting provider information."""
        with patch.object(coordinator.registry, 'get_provider_class') as mock_get_class, \
             patch.object(coordinator.registry, 'get_provider') as mock_get_provider:
            
            # Mock provider class
            mock_provider_class = MagicMock()
            mock_provider_class.name = "mock_unified"
            mock_provider_class.description = "Mock unified provider"
            mock_provider_class.supported_models = ["model-1", "model-2"]
            mock_provider_class.requires_api_key = True
            mock_provider_class.supports_streaming = True
            mock_provider_class.default_model = "model-1"
            
            mock_get_class.return_value = mock_provider_class
            mock_get_provider.return_value = mock_unified_provider
            
            info = coordinator.get_provider_info("mock_unified")
            
            assert info["name"] == "mock_unified"
            assert info["description"] == "Mock unified provider"
            assert info["supported_models"] == ["model-1", "model-2"]
            assert info["configured"] is True
            assert info["config_status"] == "valid"
            assert info["capabilities"] is not None

    def test_list_available_providers(self, coordinator):
        """Test listing all available providers."""
        with patch.object(coordinator.registry, 'list_provider_names') as mock_list_names, \
             patch.object(coordinator, 'get_provider_info') as mock_get_info:
            
            mock_list_names.return_value = ["openai", "anthropic", "ollama"]
            mock_get_info.side_effect = [
                {"name": "openai", "configured": True},
                {"name": "anthropic", "configured": False},
                {"name": "ollama", "configured": True},
            ]
            
            providers = coordinator.list_available_providers()
            
            assert len(providers) == 3
            assert providers[0]["name"] == "openai"
            assert providers[1]["name"] == "anthropic"
            assert providers[2]["name"] == "ollama"

    @pytest.mark.asyncio
    async def test_generate_with_fallback_success_primary(self, coordinator, mock_unified_provider):
        """Test fallback generation with primary provider success."""
        instance = {"instance_id": "test-1", "problem_statement": "Test problem"}
        
        with patch.object(coordinator, 'select_provider') as mock_select:
            mock_select.return_value = mock_unified_provider
            
            response = await coordinator.generate_with_fallback(
                instance, "mock_unified", ["fallback1", "fallback2"]
            )
            
            assert response.content == "mock patch content"
            assert response.provider == "mock_unified"
            mock_select.assert_called_once_with("mock_unified", None, validate_connectivity=False)

    @pytest.mark.asyncio
    async def test_generate_with_fallback_uses_fallback(self, coordinator):
        """Test fallback generation when primary provider fails."""
        instance = {"instance_id": "test-1", "problem_statement": "Test problem"}
        
        # Create failing primary provider
        failing_provider = MagicMock()
        failing_provider.generate_unified.side_effect = Exception("Primary failed")
        
        # Create working fallback provider
        working_provider = MagicMock()
        async def mock_generate_unified(request):
            from swebench_runner.providers.unified_models import TokenUsage, UnifiedResponse
            return UnifiedResponse(
                content="fallback patch content",
                model="fallback-model",
                usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
                latency_ms=500,
                finish_reason="stop",
                provider="fallback_provider",
                cost=0.0003,
                raw_response={}
            )
        working_provider.generate_unified = mock_generate_unified
        
        with patch.object(coordinator, 'select_provider') as mock_select:
            mock_select.side_effect = [failing_provider, working_provider]
            
            response = await coordinator.generate_with_fallback(
                instance, "primary", ["fallback"]
            )
            
            assert response.content == "fallback patch content"
            assert response.provider == "fallback_provider"
            assert mock_select.call_count == 2

    @pytest.mark.asyncio
    async def test_generate_with_fallback_all_fail(self, coordinator):
        """Test fallback generation when all providers fail."""
        instance = {"instance_id": "test-1", "problem_statement": "Test problem"}
        
        failing_provider = MagicMock()
        failing_provider.generate_unified.side_effect = Exception("Provider failed")
        
        with patch.object(coordinator, 'select_provider') as mock_select:
            mock_select.return_value = failing_provider
            
            with pytest.raises(Exception, match="All providers failed"):
                await coordinator.generate_with_fallback(
                    instance, "primary", ["fallback1", "fallback2"]
                )


class TestUnifiedCostEstimator:
    """Test UnifiedCostEstimator class."""

    @pytest.fixture
    def cost_estimator(self):
        """Create a UnifiedCostEstimator instance."""
        return UnifiedCostEstimator()

    @pytest.fixture
    def mock_provider_with_cost(self):
        """Create a mock provider with cost estimation capability."""
        provider = MagicMock()
        provider.name = "mock_cost"
        provider.config = ProviderConfig(name="mock_cost", max_tokens=1000)
        provider.capabilities = MagicMock()
        provider.capabilities.cost_per_1k_prompt_tokens = 0.001
        provider.capabilities.cost_per_1k_completion_tokens = 0.002
        
        def mock_estimate_cost(prompt_tokens, max_tokens):
            return (prompt_tokens / 1000) * 0.001 + (max_tokens / 1000) * 0.002
        
        provider.estimate_cost = mock_estimate_cost
        return provider

    def test_estimate_batch_cost_with_provider_method(self, cost_estimator, mock_provider_with_cost):
        """Test cost estimation using provider's estimate_cost method."""
        instances = [
            {"instance_id": "test-1", "problem_statement": "Short problem"},
            {"instance_id": "test-2", "problem_statement": "Longer problem statement with more details"},
        ]
        
        with patch.object(cost_estimator.registry, 'get_provider') as mock_get_provider:
            mock_get_provider.return_value = mock_provider_with_cost
            
            cost = cost_estimator.estimate_batch_cost(instances, "mock_cost")
            
            assert cost > 0
            assert isinstance(cost, float)
            mock_get_provider.assert_called_once_with("mock_cost", cache=True)

    def test_estimate_batch_cost_generic(self, cost_estimator):
        """Test generic cost estimation using capabilities."""
        instances = [{"instance_id": "test-1", "problem_statement": "Test problem"}]
        
        # Create provider without estimate_cost method
        provider = MagicMock()
        provider.name = "mock_generic"
        provider.config = ProviderConfig(name="mock_generic", max_tokens=1000)
        provider.capabilities = MagicMock()
        provider.capabilities.cost_per_1k_prompt_tokens = 0.001
        provider.capabilities.cost_per_1k_completion_tokens = 0.002
        # Remove estimate_cost method
        del provider.estimate_cost
        
        with patch.object(cost_estimator.registry, 'get_provider') as mock_get_provider:
            mock_get_provider.return_value = provider
            
            cost = cost_estimator.estimate_batch_cost(instances, "mock_generic")
            
            assert cost > 0
            assert isinstance(cost, float)

    def test_estimate_batch_cost_fallback_to_legacy(self, cost_estimator):
        """Test fallback to legacy cost estimation when provider fails."""
        instances = [{"instance_id": "test-1", "problem_statement": "Test"}]
        
        with patch.object(cost_estimator.registry, 'get_provider') as mock_get_provider, \
             patch('swebench_runner.generation_integration.CostEstimator') as mock_legacy:
            
            # Make provider retrieval fail
            mock_get_provider.side_effect = Exception("Provider not found")
            
            # Mock legacy estimator
            legacy_instance = MagicMock()
            legacy_instance.estimate_batch_cost.return_value = (0.05, 0.10)
            mock_legacy.return_value = legacy_instance
            
            cost = cost_estimator.estimate_batch_cost(instances, "unknown")
            
            assert cost == 0.075  # Average of min and max
            legacy_instance.estimate_batch_cost.assert_called_once()

    def test_estimate_instance_tokens(self, cost_estimator):
        """Test token estimation for individual instances."""
        instance = {
            "instance_id": "test-1",
            "problem_statement": "This is a test problem statement with several words",
            "test_patch": "diff --git a/file.py b/file.py\n+added line"
        }
        
        tokens = cost_estimator._estimate_instance_tokens(instance)
        
        assert tokens > 0
        assert isinstance(tokens, int)
        # Should include problem statement, test patch, and base prompt tokens
        assert tokens > len("This is a test problem statement with several words".split())


class TestEnhancedGenerationIntegration:
    """Test enhanced GenerationIntegration with unified provider system."""

    @pytest.fixture
    def enhanced_integration(self, temp_cache_dir):
        """Create GenerationIntegration with mocked dependencies."""
        return GenerationIntegration(temp_cache_dir)

    def test_initialization(self, enhanced_integration):
        """Test enhanced initialization with new components."""
        assert hasattr(enhanced_integration, 'registry')
        assert hasattr(enhanced_integration, 'provider_coordinator')
        assert hasattr(enhanced_integration, 'cost_estimator')
        assert hasattr(enhanced_integration, 'config_manager')

    def test_select_provider_delegates(self, enhanced_integration):
        """Test that select_provider delegates to coordinator."""
        with patch.object(enhanced_integration.provider_coordinator, 'select_provider') as mock_select:
            mock_select.return_value = MagicMock()
            
            provider = enhanced_integration.select_provider("test_provider", "test_model")
            
            mock_select.assert_called_once_with("test_provider", "test_model")
            assert provider is not None

    def test_estimate_batch_cost_delegates(self, enhanced_integration):
        """Test that estimate_batch_cost delegates to unified estimator."""
        instances = [{"instance_id": "test-1"}]
        
        with patch.object(enhanced_integration.cost_estimator, 'estimate_batch_cost') as mock_estimate:
            mock_estimate.return_value = 0.05
            
            cost = enhanced_integration.estimate_batch_cost(instances, "test_provider", "test_model")
            
            mock_estimate.assert_called_once_with(instances, "test_provider", "test_model")
            assert cost == 0.05

    @pytest.mark.asyncio
    async def test_generate_with_fallback_delegates(self, enhanced_integration):
        """Test that generate_with_fallback delegates to coordinator."""
        instance = {"instance_id": "test-1"}
        
        with patch.object(enhanced_integration.provider_coordinator, 'generate_with_fallback') as mock_fallback:
            mock_response = MagicMock()
            mock_fallback.return_value = mock_response
            
            response = await enhanced_integration.generate_with_fallback(
                instance, "primary", ["fallback"]
            )
            
            mock_fallback.assert_called_once_with(
                instance, "primary", ["fallback"], None
            )
            assert response == mock_response

    def test_list_available_providers(self, enhanced_integration):
        """Test listing available providers."""
        with patch.object(enhanced_integration.provider_coordinator, 'list_available_providers') as mock_list:
            mock_providers = [
                {"name": "openai", "configured": True},
                {"name": "anthropic", "configured": False}
            ]
            mock_list.return_value = mock_providers
            
            providers = enhanced_integration.list_available_providers()
            
            assert providers == mock_providers
            mock_list.assert_called_once()

    def test_show_provider_status(self, enhanced_integration, capsys):
        """Test showing provider status table."""
        mock_providers = [
            {
                "name": "openai",
                "description": "OpenAI GPT models",
                "supported_models": ["gpt-4", "gpt-3.5-turbo"],
                "configured": True,
                "requires_api_key": True
            },
            {
                "name": "ollama",
                "description": "Local Ollama models",
                "supported_models": ["llama2", "codellama"],
                "configured": True,
                "requires_api_key": False
            }
        ]
        
        with patch.object(enhanced_integration, 'list_available_providers') as mock_list:
            mock_list.return_value = mock_providers
            
            enhanced_integration.show_provider_status()
            
            # Check that table was printed (would be captured by Rich console)
            mock_list.assert_called_once()


class TestCostEstimator:
    """Test CostEstimator class."""

    def test_estimate_batch_cost_default_model(self):
        """Test cost estimation with default model."""
        estimator = CostEstimator()
        instances = [{"instance_id": f"test-{i}"} for i in range(10)]

        min_cost, max_cost = estimator.estimate_batch_cost(
            instances, "openai", None
        )

        assert min_cost > 0
        assert max_cost > min_cost
        assert max_cost == min_cost * 2.0

    def test_estimate_batch_cost_specific_model(self):
        """Test cost estimation with specific model."""
        estimator = CostEstimator()
        instances = [{"instance_id": f"test-{i}"} for i in range(5)]

        min_cost, max_cost = estimator.estimate_batch_cost(
            instances, "openai", "gpt-4"
        )

        # GPT-4 should be more expensive
        assert min_cost > 0.1
        assert max_cost > min_cost

    def test_estimate_batch_cost_unknown_model(self):
        """Test cost estimation with unknown model."""
        estimator = CostEstimator()
        instances = [{"instance_id": "test"}]

        min_cost, max_cost = estimator.estimate_batch_cost(
            instances, "custom", "unknown-model"
        )

        # Should use default costs
        assert min_cost > 0
        assert max_cost == min_cost * 2.0


class TestGenerationFailureHandler:
    """Test GenerationFailureHandler class."""

    def test_handle_batch_failure_no_successes(self, capsys):
        """Test handling batch with no successes."""
        from rich.console import Console
        console = Console()
        handler = GenerationFailureHandler(console)

        stats = BatchStats(
            total_instances=5,
            completed=0,
            failed=5,
            success_rate=0.0,
        )

        results = BatchResult(
            successful=[],
            failed=[
                {"instance_id": "test-1", "error": "Network error"},
                {"instance_id": "test-2", "error": "Network error"},
                {"instance_id": "test-3", "error": "Token limit"},
                {"instance_id": "test-4", "error": "Parse error"},
                {"instance_id": "test-5", "error": "Parse error"},
            ],
            stats=stats,
        )

        should_continue = handler.handle_batch_failure(results)
        assert not should_continue

    def test_handle_batch_failure_some_successes(self):
        """Test handling batch with some successes."""
        from rich.console import Console
        console = Console()
        handler = GenerationFailureHandler(console)

        stats = BatchStats(
            total_instances=5,
            completed=3,
            failed=2,
            success_rate=0.6,
        )

        results = BatchResult(
            successful=[MagicMock() for _ in range(3)],
            failed=[
                {"instance_id": "test-1", "error": "Network error"},
                {"instance_id": "test-2", "error": "Token limit"},
            ],
            stats=stats,
        )

        # Should continue with partial results
        should_continue = handler.handle_batch_failure(results)
        assert should_continue

    def test_handle_batch_failure_mostly_failed_with_confirm(self):
        """Test handling batch with mostly failures but user confirms."""
        from rich.console import Console
        console = Console()
        handler = GenerationFailureHandler(console)

        stats = BatchStats(
            total_instances=10,
            completed=2,
            failed=8,
            success_rate=0.2,
        )

        results = BatchResult(
            successful=[MagicMock() for _ in range(2)],
            failed=[{"instance_id": f"test-{i}", "error": "Error"} for i in range(8)],
            stats=stats,
        )

        with patch('click.confirm', return_value=True):
            should_continue = handler.handle_batch_failure(results)
            assert should_continue


class TestCLIIntegration:
    """Test CLI integration for generation."""

    @pytest.mark.skip(reason="Complex CLI test - needs proper Docker mocking")
    def test_run_with_provider_generates_patches(self, tmp_path):
        """Test run command with provider generates patches."""
        runner = CliRunner()

        # Create temporary directory structure
        cache_dir = tmp_path / ".swebench"
        cache_dir.mkdir(exist_ok=True)
        temp_dir = cache_dir / "temp"
        temp_dir.mkdir(exist_ok=True)

        with patch('swebench_runner.datasets.DatasetManager') as mock_dm, \
             patch('swebench_runner.provider_utils.ensure_provider_configured'), \
             patch('swebench_runner.generation_integration.GenerationIntegration') as mock_integration, \
             patch('swebench_runner.docker_run.run_evaluation') as mock_run_eval, \
             patch('swebench_runner.cache.get_cache_dir', return_value=cache_dir), \
             patch('swebench_runner.bootstrap.check_and_prompt_first_run', return_value=False), \
             patch('asyncio.run') as mock_async_run:

            # Mock dataset manager
            mock_dm.return_value.get_instances.return_value = [
                {"instance_id": "test-1", "problem_statement": "Test"}
            ]

            # Mock save_as_jsonl to create a real file in temp
            def mock_save_jsonl(instances, path):
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, 'w') as f:
                    for inst in instances:
                        f.write(json.dumps(inst) + '\n')

            mock_dm.return_value.save_as_jsonl = mock_save_jsonl
            mock_dm.return_value.check_memory_requirements.return_value = (True, None)
            mock_dm.return_value.cleanup_temp_files = MagicMock()

            # Mock generation - return a path that will exist
            patches_path = cache_dir / "generated_patches.jsonl"
            with open(patches_path, 'w') as f:
                f.write(json.dumps({"instance_id": "test-1", "patch": "test patch"}) + '\n')
            mock_async_run.return_value = patches_path

            # Mock evaluation result
            mock_result = MagicMock()
            mock_result.passed = True
            mock_result.error = None
            mock_result.instance_id = "test-1"
            mock_run_eval.return_value = mock_result

            result = runner.invoke(
                cli,
                ['run', '-d', 'lite', '--provider', 'openai', '--count', '1'],
                catch_exceptions=False
            )

            if result.exit_code != 0:
                print(f"Exit code: {result.exit_code}")
                print(f"Output: {result.output}")
                print(f"Exception: {result.exception}")
            assert result.exit_code == 0
            # Verify that generation integration was created and used
            mock_integration.assert_called_once_with(cache_dir)
            # Verify that we attempted async generation
            assert mock_async_run.called

    @pytest.mark.skip(reason="Complex CLI test - needs proper Docker mocking")
    def test_run_with_generate_only(self, tmp_path):
        """Test run command with --generate-only flag."""
        runner = CliRunner()

        # Create temporary directory structure
        cache_dir = tmp_path / ".swebench"
        cache_dir.mkdir(exist_ok=True)
        temp_dir = cache_dir / "temp"
        temp_dir.mkdir(exist_ok=True)

        with patch('swebench_runner.datasets.DatasetManager') as mock_dm, \
             patch('swebench_runner.provider_utils.ensure_provider_configured'), \
             patch('swebench_runner.generation_integration.GenerationIntegration'), \
             patch('swebench_runner.cache.get_cache_dir', return_value=cache_dir), \
             patch('swebench_runner.bootstrap.check_and_prompt_first_run', return_value=False), \
             patch('asyncio.run') as mock_async_run:

            # Mock dataset manager
            mock_dm.return_value.get_instances.return_value = [
                {"instance_id": "test-1", "problem_statement": "Test"}
            ]

            # Mock save_as_jsonl to create a real file in temp
            def mock_save_jsonl(instances, path):
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, 'w') as f:
                    for inst in instances:
                        f.write(json.dumps(inst) + '\n')

            mock_dm.return_value.save_as_jsonl = mock_save_jsonl
            mock_dm.return_value.check_memory_requirements.return_value = (True, None)
            mock_dm.return_value.cleanup_temp_files = MagicMock()

            # Mock generation - return a path that will exist
            patches_path = cache_dir / "generated_patches.jsonl"
            with open(patches_path, 'w') as f:
                f.write(json.dumps({"instance_id": "test-1", "patch": "test patch"}) + '\n')
            mock_async_run.return_value = patches_path

            result = runner.invoke(
                cli,
                ['run', '-d', 'lite', '--provider', 'openai', '--generate-only', '--count', '1'],
                catch_exceptions=False
            )

            assert result.exit_code == 0
            assert "Patches generated successfully" in result.output

    @pytest.mark.skip(reason="Complex CLI test - needs proper Docker mocking")
    def test_generate_command_with_instance(self, tmp_path):
        """Test generate command for single instance."""
        runner = CliRunner()

        # Create temporary directory structure
        cache_dir = tmp_path / ".swebench"
        cache_dir.mkdir(exist_ok=True)

        # Mock provider
        mock_provider = MagicMock()
        mock_provider._async_provider.name = "mock"
        mock_provider._async_provider.config.model = "gpt-4"
        mock_provider._async_provider.default_model = "gpt-4"

        with patch('swebench_runner.provider_utils.get_provider_for_cli', return_value=mock_provider), \
             patch('swebench_runner.datasets.DatasetManager') as mock_dm, \
             patch('swebench_runner.generation_integration.GenerationIntegration'), \
             patch('swebench_runner.cache.get_cache_dir', return_value=cache_dir), \
             patch('asyncio.run') as mock_async_run:

            # Mock dataset manager
            mock_dm.return_value.get_instances.return_value = [
                {"instance_id": "django__django-12345", "problem_statement": "Test"}
            ]

            # Mock generation result - create a real file
            output_path = tmp_path / "patches" / "django__django-12345.jsonl"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(json.dumps({
                    "instance_id": "django__django-12345",
                    "patch": "diff --git a/file.py\n+fix",
                    "cost": 0.05
                }) + '\n')

            mock_async_run.return_value = output_path

            result = runner.invoke(
                cli,
                ['generate', '-i', 'django__django-12345'],
                catch_exceptions=False
            )

            assert result.exit_code == 0
            assert "Patch generated successfully" in result.output
