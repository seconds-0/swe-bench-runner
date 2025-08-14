"""Pytest configuration and fixtures for E2E tests."""

from typing import Any

import pytest

from tests.e2e.test_doubles import (
    DockerClientDouble,
    FileSystemDouble,
    HuggingFaceDouble,
    NetworkDouble,
    ProviderDouble,
    TestDoubleFactory,
)


@pytest.fixture
def docker_double():
    """Provide a Docker client test double."""
    def _create(scenario: str = "success") -> DockerClientDouble:
        return TestDoubleFactory.create_docker_double(scenario)
    return _create


@pytest.fixture
def huggingface_double():
    """Provide a HuggingFace test double."""
    def _create(scenario: str = "success") -> HuggingFaceDouble:
        return TestDoubleFactory.create_huggingface_double(scenario)
    return _create


@pytest.fixture
def provider_double():
    """Provide a provider test double."""
    def _create(provider: str = "openai", scenario: str = "success") -> ProviderDouble:
        return TestDoubleFactory.create_provider_double(provider, scenario)
    return _create


@pytest.fixture
def network_double():
    """Provide a network test double."""
    def _create(scenario: str = "success") -> NetworkDouble:
        return TestDoubleFactory.create_network_double(scenario)
    return _create


@pytest.fixture
def filesystem_double():
    """Provide a file system test double."""
    def _create(scenario: str = "success") -> FileSystemDouble:
        return TestDoubleFactory.create_filesystem_double(scenario)
    return _create


@pytest.fixture
def all_doubles():
    """Provide all test doubles with the same scenario."""
    def _create(scenario: str = "success") -> dict[str, Any]:
        return TestDoubleFactory.create_all_doubles(scenario)
    return _create


@pytest.fixture
def test_doubles_injector(monkeypatch):
    """Fixture to inject test doubles into production code.

    This fixture provides methods to replace production dependencies
    with test doubles at specific injection points.
    """
    class Injector:
        def __init__(self, monkeypatch):
            self.monkeypatch = monkeypatch
            self.injected = []

        def inject_docker(self, double: DockerClientDouble):
            """Inject Docker client double."""
            # Replace docker.from_env() calls
            self.monkeypatch.setattr(
                "swebench_runner.docker_run.docker.from_env",
                lambda: double
            )
            self.monkeypatch.setattr(
                "swebench_runner.cli.docker.from_env",
                lambda: double
            )
            self.injected.append("docker")

        def inject_huggingface(self, double: HuggingFaceDouble):
            """Inject HuggingFace double."""
            # This would replace actual dataset loading
            # Implementation depends on how HF is used in the code
            self.injected.append("huggingface")

        def inject_provider(self, provider: str, double: ProviderDouble):
            """Inject provider double."""
            # This would replace provider instantiation
            # Implementation depends on provider registry
            self.injected.append(f"provider:{provider}")

        def inject_network(self, double: NetworkDouble):
            """Inject network double."""
            # Replace requests or urllib calls
            self.injected.append("network")

        def inject_filesystem(self, double: FileSystemDouble):
            """Inject file system double."""
            # Replace Path operations if needed
            self.injected.append("filesystem")

        def cleanup(self):
            """Clean up all injections."""
            # Monkeypatch automatically cleans up
            self.injected.clear()

    return Injector(monkeypatch)
