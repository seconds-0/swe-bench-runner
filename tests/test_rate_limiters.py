"""Tests for rate limiting coordinator."""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch

from swebench_runner.providers.rate_limiters import (
    AcquisitionRequest,
    AcquisitionResult,
    CompositeLimiter,
    LimiterType,
    RateLimitConfig,
    RateLimitCoordinator,
    SemaphoreLimiter,
    SlidingWindowLimiter,
    TokenBucketLimiter,
    create_anthropic_limiter,
    create_coordinator_from_config,
    create_ollama_limiter,
    create_openai_limiter,
)


class TestRateLimitConfig:
    """Test rate limit configuration."""

    def test_valid_config(self):
        """Test valid configuration."""
        config = RateLimitConfig(requests_per_minute=100)
        assert config.requests_per_minute == 100

    def test_invalid_config(self):
        """Test invalid configuration raises error."""
        with pytest.raises(ValueError, match="At least one rate limit must be specified"):
            RateLimitConfig()


class TestTokenBucketLimiter:
    """Test token bucket rate limiter."""

    @pytest.fixture
    def limiter(self):
        """Create a token bucket limiter."""
        return TokenBucketLimiter(capacity=10, refill_rate=1.0)

    @pytest.mark.asyncio
    async def test_acquire_success(self, limiter):
        """Test successful token acquisition."""
        request = AcquisitionRequest(estimated_tokens=5)
        result = await limiter.acquire(request)
        
        assert result.acquired is True
        assert result.metadata["tokens_remaining"] == 5.0
        assert result.metadata["capacity"] == 10

    @pytest.mark.asyncio
    async def test_acquire_insufficient_tokens(self, limiter):
        """Test acquisition when insufficient tokens."""
        request = AcquisitionRequest(estimated_tokens=15)
        result = await limiter.acquire(request)
        
        assert result.acquired is False
        assert result.limited_by == "token_bucket"
        assert result.wait_time > 0
        assert result.metadata["tokens_needed"] == 15

    @pytest.mark.asyncio
    async def test_token_refill(self):
        """Test token refill over time."""
        limiter = TokenBucketLimiter(capacity=10, refill_rate=10.0)  # 10 tokens/sec
        
        # Consume all tokens
        request = AcquisitionRequest(estimated_tokens=10)
        result = await limiter.acquire(request)
        assert result.acquired is True
        
        # Should not have tokens immediately
        result = await limiter.acquire(request)
        assert result.acquired is False
        
        # Wait for refill (mock time to avoid actual waiting)
        with patch('time.time') as mock_time:
            mock_time.side_effect = [
                limiter._last_refill,  # Initial call
                limiter._last_refill + 1.0  # 1 second later
            ]
            
            # Should have 10 tokens after 1 second
            result = await limiter.acquire(request)
            assert result.acquired is True

    def test_get_status(self, limiter):
        """Test status reporting."""
        status = limiter.get_status()
        
        assert status["type"] == "token_bucket"
        assert status["tokens_available"] == 10.0
        assert status["capacity"] == 10
        assert status["refill_rate"] == 1.0
        assert 0 <= status["utilization"] <= 1

    def test_limiter_type(self, limiter):
        """Test limiter type property."""
        assert limiter.limiter_type == LimiterType.TOKEN_BUCKET

    def test_release_noop(self, limiter):
        """Test release is no-op for token bucket."""
        # Should not raise an error
        limiter.release(5)

    def test_burst_allowance(self):
        """Test burst allowance functionality."""
        limiter = TokenBucketLimiter(capacity=10, refill_rate=1.0, burst_allowance=5)
        
        # Mock time to allow accumulation beyond capacity
        with patch('time.time') as mock_time:
            mock_time.side_effect = [
                0,  # Initial time
                20.0  # 20 seconds later
            ]
            
            status = limiter.get_status()
            # Should cap at capacity + burst_allowance = 15
            assert status["tokens_available"] == 15.0


class TestSlidingWindowLimiter:
    """Test sliding window rate limiter."""

    @pytest.fixture
    def limiter(self):
        """Create a sliding window limiter."""
        return SlidingWindowLimiter(limit=5, window_seconds=60)

    @pytest.mark.asyncio
    async def test_acquire_success(self, limiter):
        """Test successful request acquisition."""
        request = AcquisitionRequest()
        result = await limiter.acquire(request)
        
        assert result.acquired is True
        assert result.metadata["requests_in_window"] == 1
        assert result.metadata["limit"] == 5

    @pytest.mark.asyncio
    async def test_acquire_at_limit(self, limiter):
        """Test acquisition when at limit."""
        request = AcquisitionRequest()
        
        # Fill up to limit
        for _ in range(5):
            result = await limiter.acquire(request)
            assert result.acquired is True
        
        # Next request should be denied
        result = await limiter.acquire(request)
        assert result.acquired is False
        assert result.limited_by == "sliding_window"
        assert result.wait_time >= 0

    @pytest.mark.asyncio
    async def test_cleanup_old_requests(self):
        """Test cleanup of old requests."""
        limiter = SlidingWindowLimiter(limit=2, window_seconds=1)
        request = AcquisitionRequest()
        
        # Add requests
        await limiter.acquire(request)
        await limiter.acquire(request)
        
        # Should be at limit
        result = await limiter.acquire(request)
        assert result.acquired is False
        
        # Mock time to make requests expire
        with patch('time.time') as mock_time:
            mock_time.return_value = time.time() + 2  # 2 seconds later
            
            # Should be able to acquire again
            result = await limiter.acquire(request)
            assert result.acquired is True

    def test_get_status(self, limiter):
        """Test status reporting."""
        status = limiter.get_status()
        
        assert status["type"] == "sliding_window"
        assert status["requests_in_window"] == 0
        assert status["limit"] == 5
        assert status["window_seconds"] == 60
        assert status["utilization"] == 0.0

    def test_limiter_type(self, limiter):
        """Test limiter type property."""
        assert limiter.limiter_type == LimiterType.SLIDING_WINDOW

    def test_release_noop(self, limiter):
        """Test release is no-op for sliding window."""
        # Should not raise an error
        limiter.release()


class TestSemaphoreLimiter:
    """Test semaphore rate limiter."""

    @pytest.fixture
    def limiter(self):
        """Create a semaphore limiter."""
        return SemaphoreLimiter(concurrent_limit=3)

    @pytest.mark.asyncio
    async def test_acquire_success(self, limiter):
        """Test successful semaphore acquisition."""
        request = AcquisitionRequest()
        result = await limiter.acquire(request)
        
        assert result.acquired is True
        assert result.metadata["active_requests"] == 1
        assert result.metadata["concurrent_limit"] == 3

    @pytest.mark.asyncio
    async def test_acquire_at_limit(self, limiter):
        """Test acquisition when at concurrent limit."""
        request = AcquisitionRequest()
        
        # Acquire up to limit
        for _ in range(3):
            result = await limiter.acquire(request)
            assert result.acquired is True
        
        # Next acquisition should timeout
        request_with_timeout = AcquisitionRequest(timeout=0.1)
        result = await limiter.acquire(request_with_timeout)
        assert result.acquired is False
        assert result.limited_by == "semaphore"
        assert result.metadata["timeout_exceeded"] is True

    @pytest.mark.asyncio
    async def test_release(self, limiter):
        """Test semaphore release."""
        request = AcquisitionRequest()
        
        # Acquire all permits
        for _ in range(3):
            await limiter.acquire(request)
        
        # Release one
        limiter.release()
        
        # Should be able to acquire again
        result = await limiter.acquire(request)
        assert result.acquired is True

    def test_get_status(self, limiter):
        """Test status reporting."""
        status = limiter.get_status()
        
        assert status["type"] == "semaphore"
        assert status["active_requests"] == 0
        assert status["concurrent_limit"] == 3
        assert status["available"] == 3
        assert status["utilization"] == 0.0

    def test_limiter_type(self, limiter):
        """Test limiter type property."""
        assert limiter.limiter_type == LimiterType.SEMAPHORE


class TestCompositeLimiter:
    """Test composite rate limiter."""

    @pytest.fixture
    def composite_limiter(self):
        """Create a composite limiter."""
        token_bucket = TokenBucketLimiter(capacity=10, refill_rate=1.0)
        sliding_window = SlidingWindowLimiter(limit=5, window_seconds=60)
        return CompositeLimiter([token_bucket, sliding_window])

    def test_empty_limiters_raises_error(self):
        """Test that empty limiters list raises error."""
        with pytest.raises(ValueError, match="At least one limiter required"):
            CompositeLimiter([])

    @pytest.mark.asyncio
    async def test_acquire_success_all_limiters(self, composite_limiter):
        """Test successful acquisition from all limiters."""
        request = AcquisitionRequest(estimated_tokens=3)
        result = await composite_limiter.acquire(request)
        
        assert result.acquired is True
        assert len(result.metadata["limiters_used"]) == 2
        assert "token_bucket" in result.metadata["limiters_used"]
        assert "sliding_window" in result.metadata["limiters_used"]

    @pytest.mark.asyncio
    async def test_acquire_failure_rollback(self):
        """Test rollback when one limiter fails."""
        # Create a limiter that will fail
        token_bucket = TokenBucketLimiter(capacity=2, refill_rate=1.0)
        sliding_window = SlidingWindowLimiter(limit=5, window_seconds=60)
        composite = CompositeLimiter([token_bucket, sliding_window])
        
        # Request more tokens than available
        request = AcquisitionRequest(estimated_tokens=5)
        result = await composite.acquire(request)
        
        assert result.acquired is False
        assert result.limited_by == "token_bucket"
        assert result.metadata["failed_limiter"] == "token_bucket"
        
        # Verify sliding window didn't consume a request (rollback worked)
        window_status = sliding_window.get_status()
        assert window_status["requests_in_window"] == 0

    @pytest.mark.asyncio
    async def test_exception_during_acquire_rollback(self, composite_limiter):
        """Test rollback when exception occurs during acquisition."""
        # Mock one limiter to raise an exception
        with patch.object(composite_limiter.limiters[0], 'acquire', side_effect=Exception("Test error")):
            request = AcquisitionRequest(estimated_tokens=1)
            
            with pytest.raises(Exception, match="Test error"):
                await composite_limiter.acquire(request)
            
            # Verify no resources were left acquired
            status = composite_limiter.get_status()
            assert status["limiters"][1]["requests_in_window"] == 0

    def test_release_all_limiters(self, composite_limiter):
        """Test release calls all limiters."""
        # Mock release methods
        for limiter in composite_limiter.limiters:
            limiter.release = Mock()
        
        composite_limiter.release(5)
        
        # Verify all limiters were called
        for limiter in composite_limiter.limiters:
            limiter.release.assert_called_once_with(5)

    def test_get_status(self, composite_limiter):
        """Test status reporting."""
        status = composite_limiter.get_status()
        
        assert status["type"] == "composite"
        assert len(status["limiters"]) == 2
        assert status["limiters"][0]["type"] == "token_bucket"
        assert status["limiters"][1]["type"] == "sliding_window"

    def test_limiter_type(self, composite_limiter):
        """Test limiter type property."""
        assert composite_limiter.limiter_type == LimiterType.COMPOSITE


class TestRateLimitCoordinator:
    """Test rate limit coordinator."""

    @pytest.fixture
    def coordinator(self):
        """Create a rate limit coordinator."""
        return RateLimitCoordinator()

    def test_add_provider_limiter(self, coordinator):
        """Test adding provider-specific limiters."""
        limiter = TokenBucketLimiter(capacity=10, refill_rate=1.0)
        coordinator.add_provider_limiter("openai", limiter)
        
        assert "openai" in coordinator._provider_limiters
        assert coordinator._provider_limiters["openai"] is limiter

    def test_set_global_limiter(self, coordinator):
        """Test setting global limiter."""
        limiter = SlidingWindowLimiter(limit=100, window_seconds=60)
        coordinator.set_global_limiter(limiter)
        
        assert coordinator._global_limiter is limiter

    @pytest.mark.asyncio
    async def test_acquire_no_limiters(self, coordinator):
        """Test acquisition with no limiters configured."""
        request = AcquisitionRequest()
        result = await coordinator.acquire("openai", request)
        
        assert result.acquired is True

    @pytest.mark.asyncio
    async def test_acquire_provider_limiter_only(self, coordinator):
        """Test acquisition with only provider limiter."""
        limiter = TokenBucketLimiter(capacity=10, refill_rate=1.0)
        coordinator.add_provider_limiter("openai", limiter)
        
        request = AcquisitionRequest(estimated_tokens=5)
        result = await coordinator.acquire("openai", request)
        
        assert result.acquired is True
        assert "openai" in result.metadata["limiters_acquired"]

    @pytest.mark.asyncio
    async def test_acquire_global_limiter_only(self, coordinator):
        """Test acquisition with only global limiter."""
        limiter = SlidingWindowLimiter(limit=5, window_seconds=60)
        coordinator.set_global_limiter(limiter)
        
        request = AcquisitionRequest()
        result = await coordinator.acquire("anthropic", request)
        
        assert result.acquired is True
        assert "global" in result.metadata["limiters_acquired"]

    @pytest.mark.asyncio
    async def test_acquire_both_limiters(self, coordinator):
        """Test acquisition with both global and provider limiters."""
        global_limiter = SlidingWindowLimiter(limit=10, window_seconds=60)
        provider_limiter = TokenBucketLimiter(capacity=20, refill_rate=1.0)
        
        coordinator.set_global_limiter(global_limiter)
        coordinator.add_provider_limiter("openai", provider_limiter)
        
        request = AcquisitionRequest(estimated_tokens=5)
        result = await coordinator.acquire("openai", request)
        
        assert result.acquired is True
        assert "global" in result.metadata["limiters_acquired"]
        assert "openai" in result.metadata["limiters_acquired"]

    @pytest.mark.asyncio
    async def test_acquire_failure_rollback(self, coordinator):
        """Test rollback when one limiter fails in coordinator."""
        # Global limiter allows, provider limiter denies
        global_limiter = SlidingWindowLimiter(limit=10, window_seconds=60)
        provider_limiter = TokenBucketLimiter(capacity=2, refill_rate=1.0)
        
        coordinator.set_global_limiter(global_limiter)
        coordinator.add_provider_limiter("openai", provider_limiter)
        
        request = AcquisitionRequest(estimated_tokens=5)
        result = await coordinator.acquire("openai", request)
        
        assert result.acquired is False
        assert result.metadata["failed_limiter"] == "openai"
        
        # Verify global limiter was rolled back
        global_status = global_limiter.get_status()
        assert global_status["requests_in_window"] == 0

    def test_release_provider_only(self, coordinator):
        """Test release with only provider limiter."""
        limiter = Mock()
        coordinator.add_provider_limiter("openai", limiter)
        
        coordinator.release("openai", 5)
        
        limiter.release.assert_called_once_with(5)

    def test_release_global_only(self, coordinator):
        """Test release with only global limiter."""
        limiter = Mock()
        coordinator.set_global_limiter(limiter)
        
        coordinator.release("anthropic", 3)
        
        limiter.release.assert_called_once_with(3)

    def test_release_both_limiters(self, coordinator):
        """Test release with both limiters."""
        global_limiter = Mock()
        provider_limiter = Mock()
        
        coordinator.set_global_limiter(global_limiter)
        coordinator.add_provider_limiter("openai", provider_limiter)
        
        coordinator.release("openai", 7)
        
        global_limiter.release.assert_called_once_with(7)
        provider_limiter.release.assert_called_once_with(7)

    def test_get_status_no_provider(self, coordinator):
        """Test status without specific provider."""
        global_limiter = TokenBucketLimiter(capacity=10, refill_rate=1.0)
        provider_limiter = SlidingWindowLimiter(limit=5, window_seconds=60)
        
        coordinator.set_global_limiter(global_limiter)
        coordinator.add_provider_limiter("openai", provider_limiter)
        
        status = coordinator.get_status()
        
        assert status["global"]["type"] == "token_bucket"
        assert "openai" in status["providers"]
        assert status["providers"]["openai"]["type"] == "sliding_window"

    def test_get_status_specific_provider(self, coordinator):
        """Test status for specific provider."""
        provider_limiter = TokenBucketLimiter(capacity=10, refill_rate=1.0)
        coordinator.add_provider_limiter("anthropic", provider_limiter)
        
        status = coordinator.get_status("anthropic")
        
        assert status["global"] is None
        assert "anthropic" in status["providers"]
        assert status["providers"]["anthropic"]["type"] == "token_bucket"


class TestFactoryFunctions:
    """Test factory functions for creating limiters."""

    def test_create_openai_limiter(self):
        """Test OpenAI limiter factory."""
        limiter = create_openai_limiter(requests_per_minute=500, tokens_per_minute=20000)
        
        assert isinstance(limiter, CompositeLimiter)
        assert len(limiter.limiters) == 2
        assert limiter.limiters[0].limiter_type == LimiterType.SLIDING_WINDOW
        assert limiter.limiters[1].limiter_type == LimiterType.TOKEN_BUCKET

    def test_create_anthropic_limiter(self):
        """Test Anthropic limiter factory."""
        limiter = create_anthropic_limiter(tokens_per_minute=30000)
        
        assert isinstance(limiter, TokenBucketLimiter)
        assert limiter.capacity == 30000
        assert limiter.refill_rate == 500.0  # 30000 / 60
        assert limiter.burst_allowance == 3000  # 10% of capacity

    def test_create_ollama_limiter(self):
        """Test Ollama limiter factory."""
        limiter = create_ollama_limiter(concurrent_requests=2)
        
        assert isinstance(limiter, SemaphoreLimiter)
        assert limiter.concurrent_limit == 2

    def test_create_coordinator_from_config(self):
        """Test creating coordinator from configuration."""
        config = {
            "openai": RateLimitConfig(requests_per_minute=100, tokens_per_minute=10000),
            "anthropic": RateLimitConfig(tokens_per_minute=20000),
            "ollama": RateLimitConfig(concurrent_requests=2),
            "custom": RateLimitConfig(requests_per_minute=50, concurrent_requests=1),
        }
        
        coordinator = create_coordinator_from_config(config)
        
        # Check that all providers were added
        assert "openai" in coordinator._provider_limiters
        assert "anthropic" in coordinator._provider_limiters
        assert "ollama" in coordinator._provider_limiters
        assert "custom" in coordinator._provider_limiters
        
        # Check OpenAI has composite limiter
        openai_limiter = coordinator._provider_limiters["openai"]
        assert isinstance(openai_limiter, CompositeLimiter)
        
        # Check Anthropic has token bucket
        anthropic_limiter = coordinator._provider_limiters["anthropic"]
        assert isinstance(anthropic_limiter, TokenBucketLimiter)
        
        # Check Ollama has semaphore
        ollama_limiter = coordinator._provider_limiters["ollama"]
        assert isinstance(ollama_limiter, SemaphoreLimiter)
        
        # Check custom has composite (RPM + concurrent)
        custom_limiter = coordinator._provider_limiters["custom"]
        assert isinstance(custom_limiter, CompositeLimiter)

    def test_create_coordinator_from_config_empty_provider(self):
        """Test creating coordinator with provider that has no valid limits."""
        config = {
            "empty": RateLimitConfig(requests_per_minute=100),
            "invalid": RateLimitConfig(),  # This should be caught by __post_init__
        }
        
        # Should not raise error, just skip invalid provider
        with pytest.raises(ValueError):
            create_coordinator_from_config(config)

    def test_create_coordinator_from_config_single_limit_types(self):
        """Test coordinator creation with single limit types."""
        config = {
            "rpm_only": RateLimitConfig(requests_per_minute=100),
            "tpm_only": RateLimitConfig(tokens_per_minute=5000),
            "concurrent_only": RateLimitConfig(concurrent_requests=3),
        }
        
        coordinator = create_coordinator_from_config(config)
        
        # RPM only should create sliding window
        rpm_limiter = coordinator._provider_limiters["rpm_only"]
        assert isinstance(rpm_limiter, SlidingWindowLimiter)
        
        # TPM only should create token bucket
        tpm_limiter = coordinator._provider_limiters["tpm_only"]
        assert isinstance(tpm_limiter, TokenBucketLimiter)
        
        # Concurrent only should create semaphore
        concurrent_limiter = coordinator._provider_limiters["concurrent_only"]
        assert isinstance(concurrent_limiter, SemaphoreLimiter)


class TestIntegration:
    """Integration tests for rate limiting system."""

    @pytest.mark.asyncio
    async def test_realistic_openai_scenario(self):
        """Test realistic OpenAI rate limiting scenario."""
        # Create OpenAI-style limiter: 1000 RPM, 40000 TPM
        limiter = create_openai_limiter(requests_per_minute=5, tokens_per_minute=100)
        
        # Make several requests within limits
        requests = [
            AcquisitionRequest(estimated_tokens=10),
            AcquisitionRequest(estimated_tokens=15),
            AcquisitionRequest(estimated_tokens=20),
        ]
        
        for request in requests:
            result = await limiter.acquire(request)
            assert result.acquired is True
        
        # Should hit token limit
        big_request = AcquisitionRequest(estimated_tokens=60)
        result = await limiter.acquire(big_request)
        assert result.acquired is False
        assert result.limited_by == "token_bucket"

    @pytest.mark.asyncio
    async def test_coordinator_with_multiple_providers(self):
        """Test coordinator managing multiple providers."""
        coordinator = RateLimitCoordinator()
        
        # Set up different limiters for different providers
        coordinator.add_provider_limiter("openai", create_openai_limiter(10, 100))
        coordinator.add_provider_limiter("anthropic", create_anthropic_limiter(50))
        coordinator.add_provider_limiter("ollama", create_ollama_limiter(2))
        
        # Test each provider independently
        openai_request = AcquisitionRequest(estimated_tokens=10)
        result = await coordinator.acquire("openai", openai_request)
        assert result.acquired is True
        
        anthropic_request = AcquisitionRequest(estimated_tokens=20)
        result = await coordinator.acquire("anthropic", anthropic_request)
        assert result.acquired is True
        
        ollama_request = AcquisitionRequest()
        result = await coordinator.acquire("ollama", ollama_request)
        assert result.acquired is True
        
        # Get status for all providers
        status = coordinator.get_status()
        assert "openai" in status["providers"]
        assert "anthropic" in status["providers"]
        assert "ollama" in status["providers"]

    @pytest.mark.asyncio
    async def test_concurrent_access_safety(self):
        """Test thread safety with concurrent access."""
        limiter = SemaphoreLimiter(concurrent_limit=2)
        
        async def make_request():
            request = AcquisitionRequest()
            result = await limiter.acquire(request)
            if result.acquired:
                # Simulate some work
                await asyncio.sleep(0.1)
                limiter.release()
            return result.acquired
        
        # Launch several concurrent requests
        tasks = [make_request() for _ in range(5)]
        results = await asyncio.gather(*tasks)
        
        # At least some should succeed (exactly how many depends on timing)
        assert any(results)
        
        # Final status should show no active requests
        await asyncio.sleep(0.2)  # Wait for all releases
        status = limiter.get_status()
        assert status["active_requests"] == 0