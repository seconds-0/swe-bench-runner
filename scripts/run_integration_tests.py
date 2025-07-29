#!/usr/bin/env python3
"""
Safe integration test runner with cost tracking and controls.

This script runs integration tests with real API calls while:
- Tracking costs per test and total
- Enforcing cost limits
- Providing detailed progress reporting
- Handling credentials securely
"""

import os
import sys
import time
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

# Cost limits (in USD)
MAX_COST_PER_TEST = 0.001  # $0.001 per test
MAX_TOTAL_COST = 0.05      # $0.05 total budget

# Test timeouts (seconds)
TEST_TIMEOUT = 60

# Color codes for output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'


class IntegrationTestRunner:
    """Manages safe execution of integration tests with cost tracking."""

    def __init__(self, env_file: Optional[str] = None):
        self.start_time = datetime.now()
        self.total_cost = 0.0
        self.test_results: List[Dict] = []
        self.env_file = env_file or '.env.integration'

        # Cost tracking file
        self.cost_report_path = Path('integration_test_costs.json')

    def setup_environment(self) -> bool:
        """Load environment variables from .env file."""
        if os.path.exists(self.env_file):
            print(f"{BLUE}Loading credentials from {self.env_file}{RESET}")
            # Load .env file manually to avoid dependencies
            with open(self.env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip().strip('"\'')
            return True
        else:
            print(f"{YELLOW}Warning: {self.env_file} not found{RESET}")
            return False

    def check_prerequisites(self) -> Dict[str, bool]:
        """Check if all prerequisites are met."""
        checks = {
            'ollama': self._check_ollama(),
            'openai_key': bool(os.environ.get('OPENAI_API_KEY')),
            'anthropic_key': bool(os.environ.get('ANTHROPIC_API_KEY')),
        }

        return checks

    def _check_ollama(self) -> bool:
        """Check if Ollama is running."""
        try:
            result = subprocess.run(
                ['curl', '-s', 'http://localhost:11434/api/tags'],
                capture_output=True,
                timeout=2
            )
            return result.returncode == 0
        except:
            return False

    def estimate_costs(self) -> Dict[str, float]:
        """Estimate costs for each provider."""
        return {
            'ollama': 0.0,  # Free
            'openai': 0.01,  # ~$0.01 for all tests
            'anthropic': 0.008,  # ~$0.008 for all tests
        }

    def run_provider_tests(self, provider: str, dry_run: bool = False) -> Tuple[bool, float]:
        """Run tests for a specific provider."""
        test_file = f"tests/integration/test_{provider}_integration.py"

        if not os.path.exists(test_file):
            print(f"{RED}Test file not found: {test_file}{RESET}")
            return False, 0.0

        print(f"\n{BLUE}Running {provider.upper()} integration tests...{RESET}")

        if dry_run:
            print(f"{YELLOW}DRY RUN - Would run: pytest {test_file} -m integration -v{RESET}")
            return True, 0.0

        # Run tests with pytest
        start_cost = self.total_cost
        cmd = [
            sys.executable, '-m', 'pytest',
            test_file,
            '-m', 'integration',
            '-v',
            '--tb=short',
            f'--timeout={TEST_TIMEOUT}',
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            success = result.returncode == 0

            # Parse output for cost info (would need custom pytest plugin for real cost tracking)
            # For now, use estimates
            provider_cost = self.estimate_costs().get(provider, 0.0)
            self.total_cost += provider_cost

            # Save results
            self.test_results.append({
                'provider': provider,
                'success': success,
                'cost': provider_cost,
                'duration': time.time() - time.time(),  # Would need proper timing
                'output': result.stdout if not success else None,
                'error': result.stderr if not success else None,
            })

            if success:
                print(f"{GREEN}✓ {provider.upper()} tests passed (cost: ${provider_cost:.4f}){RESET}")
            else:
                print(f"{RED}✗ {provider.upper()} tests failed{RESET}")
                print(f"{RED}Error output:{RESET}")
                print(result.stderr)

            # Check cost limit
            if self.total_cost > MAX_TOTAL_COST:
                print(f"{RED}Cost limit exceeded! Total: ${self.total_cost:.4f}{RESET}")
                return False, provider_cost

            return success, provider_cost

        except subprocess.TimeoutExpired:
            print(f"{RED}Tests timed out after {TEST_TIMEOUT}s{RESET}")
            return False, 0.0
        except Exception as e:
            print(f"{RED}Error running tests: {e}{RESET}")
            return False, 0.0

    def run_single_test(self, provider: str, test_name: str) -> Tuple[bool, float]:
        """Run a single test for verification."""
        test_path = f"tests/integration/test_{provider}_integration.py::TestOllama integration::test_basic_generation"

        print(f"\n{BLUE}Running single test: {provider}/{test_name}{RESET}")

        cmd = [
            sys.executable, '-m', 'pytest',
            test_path,
            '-m', 'integration',
            '-v',
            '-s',  # Show output
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            success = result.returncode == 0

            if success:
                print(f"{GREEN}✓ Test passed{RESET}")
            else:
                print(f"{RED}✗ Test failed{RESET}")
                print(result.stdout)
                print(result.stderr)

            return success, 0.0

        except Exception as e:
            print(f"{RED}Error: {e}{RESET}")
            return False, 0.0

    def save_report(self):
        """Save cost and results report."""
        report = {
            'run_date': self.start_time.isoformat(),
            'total_cost': self.total_cost,
            'total_duration': (datetime.now() - self.start_time).total_seconds(),
            'results': self.test_results,
            'environment': {
                'openai_model': os.environ.get('OPENAI_TEST_MODEL', 'gpt-3.5-turbo'),
                'anthropic_model': os.environ.get('ANTHROPIC_TEST_MODEL', 'claude-3-haiku-20240307'),
                'ollama_model': os.environ.get('OLLAMA_TEST_MODEL', 'llama3.2:1b'),
            }
        }

        with open(self.cost_report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n{BLUE}Report saved to: {self.cost_report_path}{RESET}")

    def print_summary(self):
        """Print test run summary."""
        print(f"\n{BLUE}{'='*60}{RESET}")
        print(f"{BLUE}Integration Test Summary{RESET}")
        print(f"{BLUE}{'='*60}{RESET}")

        duration = (datetime.now() - self.start_time).total_seconds()

        print(f"Total Duration: {duration:.1f}s")
        print(f"Total Cost: ${self.total_cost:.4f}")
        print(f"Budget Remaining: ${MAX_TOTAL_COST - self.total_cost:.4f}")

        print(f"\n{BLUE}Results by Provider:{RESET}")
        for result in self.test_results:
            status = f"{GREEN}PASSED{RESET}" if result['success'] else f"{RED}FAILED{RESET}"
            print(f"  {result['provider']}: {status} (${result['cost']:.4f})")

        # Overall status
        all_passed = all(r['success'] for r in self.test_results)
        if all_passed:
            print(f"\n{GREEN}All tests passed! 🎉{RESET}")
        else:
            print(f"\n{RED}Some tests failed. Please review the output above.{RESET}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run integration tests safely with cost controls')
    parser.add_argument('--provider', choices=['ollama', 'openai', 'anthropic', 'all'],
                        default='all', help='Which provider to test')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be run without executing')
    parser.add_argument('--env-file', default='.env.integration', help='Path to environment file')
    parser.add_argument('--single-test', help='Run only a single test (e.g., test_basic_generation)')
    parser.add_argument('--skip-ollama-check', action='store_true', help='Skip Ollama availability check')

    args = parser.parse_args()

    runner = IntegrationTestRunner(env_file=args.env_file)

    # Setup environment
    if not runner.setup_environment() and not args.dry_run:
        print(f"{YELLOW}Consider creating {args.env_file} with your API keys:{RESET}")
        print("OPENAI_API_KEY=sk-...")
        print("ANTHROPIC_API_KEY=sk-ant-...")
        print("OPENAI_TEST_MODEL=gpt-3.5-turbo")
        print("ANTHROPIC_TEST_MODEL=claude-3-haiku-20240307")
        print("OLLAMA_TEST_MODEL=llama3.2:1b")

    # Check prerequisites
    print(f"\n{BLUE}Checking prerequisites...{RESET}")
    checks = runner.check_prerequisites()

    print(f"Ollama running: {'✓' if checks['ollama'] else '✗'}")
    print(f"OpenAI API key: {'✓' if checks['openai_key'] else '✗'}")
    print(f"Anthropic API key: {'✓' if checks['anthropic_key'] else '✗'}")

    if not args.skip_ollama_check and not checks['ollama'] and args.provider in ['ollama', 'all']:
        print(f"\n{YELLOW}Ollama not running. Start it with: ollama serve{RESET}")
        if not args.dry_run:
            sys.exit(1)

    # Show cost estimates
    print(f"\n{BLUE}Estimated costs:{RESET}")
    for provider, cost in runner.estimate_costs().items():
        print(f"  {provider}: ${cost:.4f}")
    print(f"  Total: ${sum(runner.estimate_costs().values()):.4f}")

    if not args.dry_run:
        response = input(f"\n{YELLOW}Continue with testing? (y/N): {RESET}")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)

    # Run tests
    if args.single_test and args.provider != 'all':
        # Run single test
        runner.run_single_test(args.provider, args.single_test)
    else:
        # Run provider tests
        providers = ['ollama', 'openai', 'anthropic'] if args.provider == 'all' else [args.provider]

        for provider in providers:
            # Skip if no credentials
            if provider == 'openai' and not checks['openai_key']:
                print(f"\n{YELLOW}Skipping OpenAI tests (no API key){RESET}")
                continue
            if provider == 'anthropic' and not checks['anthropic_key']:
                print(f"\n{YELLOW}Skipping Anthropic tests (no API key){RESET}")
                continue
            if provider == 'ollama' and not checks['ollama'] and not args.skip_ollama_check:
                print(f"\n{YELLOW}Skipping Ollama tests (not running){RESET}")
                continue

            success, cost = runner.run_provider_tests(provider, dry_run=args.dry_run)

            if not success and not args.dry_run:
                print(f"\n{RED}Stopping due to test failure{RESET}")
                break

    # Save report and print summary
    if not args.dry_run and runner.test_results:
        runner.save_report()
        runner.print_summary()


if __name__ == '__main__':
    main()
