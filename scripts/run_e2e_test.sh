#!/bin/bash
#
# Comprehensive E2E test runner with safety controls
#
# This script runs the full end-to-end test suite with real API calls.
# It includes safety checks, cost controls, and detailed logging.
#
# Usage:
#   ./scripts/run_e2e_test.sh                    # Run full test suite
#   ./scripts/run_e2e_test.sh --minimal          # Run minimal test only
#   ./scripts/run_e2e_test.sh --dry-run          # Check setup without running
#   ./scripts/run_e2e_test.sh --scenario 1       # Run specific scenario
#   ./scripts/run_e2e_test.sh --no-confirm       # Skip confirmation prompt
#
# Environment variables:
#   OPENAI_API_KEY          - OpenAI API key
#   ANTHROPIC_API_KEY       - Anthropic API key
#   SWEBENCH_E2E_MAX_COST   - Maximum cost limit (default: $0.10)
#   SWEBENCH_E2E_LOG_DIR    - Directory for logs (default: ./e2e_logs)

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Configuration
MAX_COST="${SWEBENCH_E2E_MAX_COST:-0.10}"
LOG_DIR="${SWEBENCH_E2E_LOG_DIR:-./e2e_logs}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/e2e_test_${TIMESTAMP}.log"
PYTHON_CMD="${PYTHON_CMD:-python3}"

# Parse arguments
MINIMAL=false
DRY_RUN=false
NO_CONFIRM=false
SCENARIO=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --minimal)
            MINIMAL=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --no-confirm)
            NO_CONFIRM=true
            shift
            ;;
        --scenario)
            SCENARIO="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--minimal] [--dry-run] [--no-confirm] [--scenario N]"
            exit 1
            ;;
    esac
done

# Create log directory
mkdir -p "$LOG_DIR"

# Logging functions
log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")

    case $level in
        INFO)
            echo -e "${BLUE}[${timestamp}] → ${message}${NC}"
            ;;
        SUCCESS)
            echo -e "${GREEN}[${timestamp}] ✓ ${message}${NC}"
            ;;
        WARNING)
            echo -e "${YELLOW}[${timestamp}] ⚠ ${message}${NC}"
            ;;
        ERROR)
            echo -e "${RED}[${timestamp}] ✗ ${message}${NC}"
            ;;
    esac

    echo "[${timestamp}] [${level}] ${message}" >> "$LOG_FILE"
}

# Header
echo -e "${BOLD}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}          SWE-Bench Runner - End-to-End Test Suite${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════════════════${NC}"
echo ""

log INFO "Test configuration:"
log INFO "  Max cost limit: \$${MAX_COST}"
log INFO "  Log directory: ${LOG_DIR}"
log INFO "  Log file: ${LOG_FILE}"
log INFO "  Python: ${PYTHON_CMD}"

# Pre-flight checks
log INFO "Running pre-flight checks..."

# Check Python version
python_version=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
log INFO "Python version: ${python_version}"

if [[ "${python_version}" < "3.11" ]]; then
    log WARNING "Python ${python_version} detected. Project requires 3.11+"
fi

# Check Docker
if docker --version >/dev/null 2>&1; then
    docker_version=$(docker --version | cut -d' ' -f3 | tr -d ',')
    log SUCCESS "Docker available: ${docker_version}"

    # Check if Docker daemon is running
    if docker info >/dev/null 2>&1; then
        log SUCCESS "Docker daemon is running"
    else
        log ERROR "Docker daemon is not running. Please start Docker."
        exit 1
    fi
else
    log ERROR "Docker not found. Please install Docker."
    exit 1
fi

# Check disk space (cross-platform)
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    free_space=$(df -g . | awk 'NR==2 {print $4}')
else
    # Linux
    free_space=$(df -BG . | awk 'NR==2 {print $4}' | tr -d 'G')
fi
log INFO "Free disk space: ${free_space}GB"

if [[ ${free_space} -lt 10 ]]; then
    log WARNING "Low disk space. Recommend at least 10GB free."
fi

# Check API keys
api_keys_found=0

if [[ -n "${OPENAI_API_KEY:-}" ]]; then
    masked_key="${OPENAI_API_KEY:0:7}...${OPENAI_API_KEY: -4}"
    log SUCCESS "OpenAI API key found: ${masked_key}"
    ((api_keys_found++))
else
    log WARNING "OpenAI API key not found (OPENAI_API_KEY)"
fi

if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
    masked_key="${ANTHROPIC_API_KEY:0:10}...${ANTHROPIC_API_KEY: -4}"
    log SUCCESS "Anthropic API key found: ${masked_key}"
    ((api_keys_found++))
else
    log WARNING "Anthropic API key not found (ANTHROPIC_API_KEY)"
fi

if [[ ${api_keys_found} -eq 0 ]]; then
    log ERROR "No API keys found. At least one is required:"
    log ERROR "  export OPENAI_API_KEY=sk-..."
    log ERROR "  export ANTHROPIC_API_KEY=sk-ant-..."
    exit 1
fi

# Check if CLI is installed
if command -v swebench >/dev/null 2>&1; then
    swebench_version=$(swebench --version 2>&1 || echo "unknown")
    log SUCCESS "SWE-bench CLI found: ${swebench_version}"
else
    log ERROR "SWE-bench CLI not found. Please install:"
    log ERROR "  pip install -e ."
    exit 1
fi

# Dry run mode
if [[ "$DRY_RUN" == true ]]; then
    log INFO "DRY RUN MODE - Not executing tests"
    log SUCCESS "All pre-flight checks passed!"
    log INFO "Would run tests with:"
    log INFO "  - ${api_keys_found} API key(s)"
    log INFO "  - Max cost: \$${MAX_COST}"
    log INFO "  - Scenarios: ${SCENARIO:-all}"
    exit 0
fi

# Cost warning
echo ""
echo -e "${YELLOW}${BOLD}⚠️  WARNING: Real API Costs ⚠️${NC}"
echo -e "${YELLOW}This test will make real API calls and incur costs:${NC}"
echo -e "${YELLOW}  - Estimated cost: \$0.01 - \$0.05${NC}"
echo -e "${YELLOW}  - Maximum budget: \$${MAX_COST}${NC}"
echo -e "${YELLOW}  - Providers: ${api_keys_found} available${NC}"
echo ""

# Confirmation prompt
if [[ "$NO_CONFIRM" != true ]]; then
    read -p "Do you want to continue? (yes/no): " -r
    echo
    if [[ ! $REPLY =~ ^[Yy]es$ ]]; then
        log INFO "Test cancelled by user"
        exit 0
    fi
fi

# Enable E2E tests
export SWEBENCH_E2E_ENABLED=true
export SWEBENCH_E2E_MAX_COST="${MAX_COST}"
export SWEBENCH_NO_INPUT=1

# Run tests
log INFO "Starting E2E test execution..."
echo ""

if [[ "$MINIMAL" == true ]]; then
    # Run minimal test
    log INFO "Running minimal test only..."
    $PYTHON_CMD tests/e2e/test_real_api_workflow.py --minimal
    exit_code=$?
else
    # Run full test suite or specific scenario
    if [[ -n "$SCENARIO" ]]; then
        log INFO "Running scenario ${SCENARIO}..."
        test_name="tests/e2e/test_real_api_workflow.py::TestRealAPIWorkflow::test_scenario_${SCENARIO}_*"
    else
        log INFO "Running full test suite..."
        test_name="tests/e2e/test_real_api_workflow.py"
    fi

    # Run with pytest
    $PYTHON_CMD -m pytest "${test_name}" \
        -v \
        -s \
        --tb=short \
        --color=yes \
        2>&1 | tee -a "$LOG_FILE"

    exit_code=${PIPESTATUS[0]}
fi

echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════════════════${NC}"

# Summary
if [[ $exit_code -eq 0 ]]; then
    log SUCCESS "All E2E tests passed!"

    # Parse costs from log if available
    if grep -q "Total cost:" "$LOG_FILE"; then
        total_cost=$(grep "Total cost:" "$LOG_FILE" | tail -1 | grep -oE '\$[0-9]+\.[0-9]+' | tail -1)
        log INFO "Total API cost: ${total_cost:-unknown}"
    fi
else
    log ERROR "E2E tests failed with exit code: ${exit_code}"
fi

log INFO "Full log saved to: ${LOG_FILE}"

# Look for test report JSON
report_file=$(find . -name "e2e_test_*.json" -mmin -5 2>/dev/null | head -1)
if [[ -n "$report_file" ]]; then
    log INFO "Test report saved to: ${report_file}"

    # Show summary from report
    if command -v jq >/dev/null 2>&1; then
        duration=$(jq -r '.test_run.duration' "$report_file" 2>/dev/null)
        total_cost=$(jq -r '.test_run.total_cost' "$report_file" 2>/dev/null)
        steps=$(jq -r '.test_run.steps' "$report_file" 2>/dev/null)

        echo ""
        echo -e "${BOLD}Test Summary:${NC}"
        echo "  Duration: ${duration}s"
        echo "  Total cost: \$${total_cost}"
        echo "  Steps executed: ${steps}"
    fi
fi

echo -e "${BOLD}═══════════════════════════════════════════════════════════════${NC}"

exit $exit_code
