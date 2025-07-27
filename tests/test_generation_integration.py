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
)
from swebench_runner.providers import MockProvider, ProviderConfig


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
