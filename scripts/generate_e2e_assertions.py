#!/usr/bin/env python3
"""Generate E2E test assertions from UX_Plan.md specifications.

This script parses the UX_Plan.md document and generates proper assertion
code for each error condition defined in Section 6: Expected Error Messages.

Usage:
    python scripts/generate_e2e_assertions.py

Output:
    - Assertion code snippets for each error code
    - Test validation report
    - Coverage metrics
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple


# Mapping of UX error codes to exit codes based on error_utils.py categories
ERROR_CODE_MAP = {
    2: 2,   # Docker not running -> DOCKER_NOT_FOUND
    3: 3,   # Network failure -> NETWORK_ERROR
    4: 4,   # Disk space -> RESOURCE_ERROR
    5: 1,   # Invalid patch -> GENERAL_ERROR
    6: 1,   # Patch too large -> GENERAL_ERROR
    7: 1,   # Unsupported arch -> GENERAL_ERROR (warning)
    10: 2,  # Docker permission -> DOCKER_NOT_FOUND
    11: 2,  # Docker Desktop -> DOCKER_NOT_FOUND
    13: 4,  # OOM -> RESOURCE_ERROR
    14: 1,  # Patch conflict -> GENERAL_ERROR
    15: 3,  # GHCR blocked -> NETWORK_ERROR
    16: 3,  # Git rate limit -> NETWORK_ERROR
    17: 1,  # Corrupted cache -> GENERAL_ERROR
    18: 1,  # Stale image -> GENERAL_ERROR (warning)
    19: 4,  # Docker storage -> RESOURCE_ERROR
    20: 1,  # Invalid Python -> GENERAL_ERROR
    21: 1,  # Container timeout -> GENERAL_ERROR
    22: 1,  # Invalid instance -> GENERAL_ERROR
    23: 1,  # Encoding error -> GENERAL_ERROR
    24: 1,  # Patch too large -> GENERAL_ERROR
    25: 1,  # Binary patch -> GENERAL_ERROR
    26: 1,  # Patch failed -> GENERAL_ERROR
    27: 3,  # HF rate limit -> NETWORK_ERROR
    28: 1,  # Instance timeout -> GENERAL_ERROR
    29: 1,  # Flaky test -> GENERAL_ERROR (warning)
    30: 1,  # Container limit -> GENERAL_ERROR (warning)
}


def parse_ux_plan_errors(ux_plan_path: Path) -> List[Dict]:
    """Parse UX_Plan.md to extract error definitions.

    Returns:
        List of error definitions with code, trigger, message, and action
    """
    errors = []

    with open(ux_plan_path) as f:
        content = f.read()

    # Find the error messages section
    error_section = re.search(
        r'## 6\. Expected Error Messages.*?\n\n(.*?)(?=\n##|\Z)',
        content,
        re.DOTALL
    )

    if not error_section:
        print("Warning: Could not find error messages section in UX_Plan.md")
        return errors

    # Parse the error table
    lines = error_section.group(1).split('\n')
    for line in lines:
        if '|' in line and not line.startswith('|---'):
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 5 and parts[1].isdigit():
                errors.append({
                    'code': int(parts[1]),
                    'trigger': parts[2],
                    'message': parts[3],
                    'action': parts[4]
                })

    return errors


def generate_assertion_code(error: Dict) -> str:
    """Generate assertion code for a specific error.

    Args:
        error: Error definition from UX_Plan

    Returns:
        Python test code with proper assertions
    """
    ux_code = error['code']
    exit_code = ERROR_CODE_MAP.get(ux_code, 1)

    # Determine error category for helper function
    if exit_code == 2:
        category = "docker"
    elif exit_code == 3:
        category = "network"
    elif exit_code == 4:
        category = "resource"
    else:
        category = "general"

    # Extract key terms from the error message
    message_lower = error['message'].lower()
    key_terms = []

    if "docker" in message_lower:
        if "permission" in message_lower:
            key_terms.append('assert_docker_error(combined, "permission")')
        elif "desktop" in message_lower:
            key_terms.append('assert_docker_error(combined, "desktop_stopped")')
        else:
            key_terms.append('assert_docker_error(combined, "not_running")')

    if "network" in message_lower or "rate limit" in message_lower:
        if "ghcr" in message_lower:
            key_terms.append('assert_network_error(combined, "ghcr_blocked")')
        elif "huggingface" in message_lower or "hf" in message_lower:
            key_terms.append('assert_network_error(combined, "hf_rate_limit")')
        elif "git" in message_lower:
            key_terms.append('assert_network_error(combined, "git_rate_limit")')
        else:
            key_terms.append('assert_network_error(combined, "general")')

    if "disk" in message_lower or "space" in message_lower:
        key_terms.append('assert_resource_error(combined, "disk_space")')
    elif "memory" in message_lower or "oom" in message_lower:
        key_terms.append('assert_resource_error(combined, "memory")')

    if "patch" in message_lower:
        if "conflict" in message_lower:
            key_terms.append('assert_patch_error(combined, "conflict")')
        elif "large" in message_lower or "size" in message_lower:
            key_terms.append('assert_patch_error(combined, "too_large")')
        elif "encoding" in message_lower:
            key_terms.append('assert_patch_error(combined, "encoding")')
        elif "binary" in message_lower:
            key_terms.append('assert_patch_error(combined, "binary")')

    if "timeout" in message_lower:
        key_terms.append('assert_timeout_error(combined)')

    if "instance" in message_lower and "invalid" in message_lower:
        key_terms.append('assert_validation_error(combined, "instance_id")')

    # Generate the test code
    code = f'''def test_error_{ux_code}_{error["trigger"].lower().replace(" ", "_")[:20]}(self):
    """Test {error['trigger']} (UX error {ux_code}, exit code {exit_code})."""
    with SWEBenchTestHarness() as harness:
        patch_file = harness.create_minimal_patch()

        # Set up mock environment for this error
        env = {{
            "SWEBENCH_MOCK_ERROR_{ux_code}": "true",
            "SWEBENCH_MOCK_NO_DOCKER": "true"
        }}

        returncode, stdout, stderr = harness.run_cli(
            ["run", "--patches", str(patch_file)],
            env=env
        )
        combined = stdout + stderr

        # Verify exit code from UX_Plan
        assert_exit_code(returncode, {exit_code}, "{error['trigger']}")
        '''

    # Add specific assertions
    for term in key_terms:
        code += f'''
        # Verify error messages
        {term}'''

    # Always add suggestion check
    code += '''

        # Must provide actionable suggestion
        assert_contains_suggestion(combined)'''

    return code


def validate_existing_tests(test_file: Path, ux_errors: List[Dict]) -> Dict:
    """Validate that existing tests cover all UX_Plan errors.

    Returns:
        Dictionary with validation results
    """
    with open(test_file) as f:
        content = f.read()

    results = {
        'total_errors': len(ux_errors),
        'covered': [],
        'missing': [],
        'has_assertions': [],
        'needs_assertions': []
    }

    for error in ux_errors:
        test_name = f"test_error_{error['code']}_"

        if test_name in content:
            results['covered'].append(error['code'])

            # Check if test has assertions (not just _ = stdout + stderr)
            test_match = re.search(
                f'{test_name}.*?(?=def test_|class |\\Z)',
                content,
                re.DOTALL
            )

            if test_match:
                test_body = test_match.group(0)
                if 'assert_' in test_body or 'assert ' in test_body:
                    results['has_assertions'].append(error['code'])
                else:
                    results['needs_assertions'].append(error['code'])
        else:
            results['missing'].append(error['code'])

    return results


def generate_report(results: Dict) -> str:
    """Generate a validation report.

    Returns:
        Formatted report string
    """
    coverage = len(results['covered']) / results['total_errors'] * 100
    assertion_rate = len(results['has_assertions']) / len(results['covered']) * 100 if results['covered'] else 0

    report = f"""
E2E Test Validation Report
=========================

Coverage Summary:
- Total UX_Plan errors: {results['total_errors']}
- Tests implemented: {len(results['covered'])} ({coverage:.1f}%)
- Tests with assertions: {len(results['has_assertions'])} ({assertion_rate:.1f}% of implemented)
- Tests needing assertions: {len(results['needs_assertions'])}
- Missing tests: {len(results['missing'])}

Tests Needing Assertions:
{', '.join(map(str, sorted(results['needs_assertions'])))}

Missing Tests:
{', '.join(map(str, sorted(results['missing'])))}

Recommendation:
"""

    if results['needs_assertions']:
        report += f"- Add assertions to {len(results['needs_assertions'])} existing tests\n"

    if results['missing']:
        report += f"- Create {len(results['missing'])} new tests for missing error codes\n"

    if assertion_rate < 100:
        report += "- Use assertion helper functions for consistent validation\n"

    return report


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent

    # Parse UX_Plan
    ux_plan_path = project_root / "Documentation" / "UX_Plan.md"
    if not ux_plan_path.exists():
        print(f"Error: {ux_plan_path} not found")
        sys.exit(1)

    ux_errors = parse_ux_plan_errors(ux_plan_path)
    print(f"Parsed {len(ux_errors)} error definitions from UX_Plan.md")

    # Validate existing tests
    test_file = project_root / "tests" / "e2e" / "test_error_handling.py"
    if test_file.exists():
        results = validate_existing_tests(test_file, ux_errors)
        print(generate_report(results))

        # Generate code for missing assertions
        if results['needs_assertions']:
            print("\n\nGenerated Assertion Code for Existing Tests:")
            print("=" * 50)

            for error in ux_errors:
                if error['code'] in results['needs_assertions']:
                    print(f"\n# Error {error['code']}: {error['trigger']}")
                    print(generate_assertion_code(error))

    # Generate fixture data
    fixture_dir = project_root / "tests" / "e2e" / "fixtures" / "expected" / "error_messages"
    fixture_dir.mkdir(parents=True, exist_ok=True)

    for error in ux_errors:
        fixture_file = fixture_dir / f"error_{error['code']}.json"
        fixture_data = {
            "code": error['code'],
            "exit_code": ERROR_CODE_MAP.get(error['code'], 1),
            "trigger": error['trigger'],
            "expected_message": error['message'],
            "expected_action": error['action'],
            "key_terms": extract_key_terms(error['message'])
        }

        with open(fixture_file, 'w') as f:
            json.dump(fixture_data, f, indent=2)

    print(f"\nGenerated {len(ux_errors)} fixture files in {fixture_dir}")


def extract_key_terms(message: str) -> List[str]:
    """Extract key terms that must appear in error output."""
    terms = []

    # Extract commands (things in backticks)
    commands = re.findall(r'`([^`]+)`', message)
    terms.extend(commands)

    # Extract key phrases
    if "Docker" in message:
        terms.append("Docker")
    if "permission" in message.lower():
        terms.append("permission")
    if "network" in message.lower():
        terms.append("network")
    if "disk" in message.lower() or "space" in message.lower():
        terms.append("space")

    return terms


if __name__ == "__main__":
    main()

