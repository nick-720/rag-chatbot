#!/bin/bash

# Development script for code quality checks

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

usage() {
    echo "Usage: $0 [check|fix|all]"
    echo ""
    echo "Commands:"
    echo "  check   Check code formatting (default)"
    echo "  fix     Auto-fix formatting issues"
    echo "  all     Run all quality checks and tests"
    echo ""
    exit 1
}

check_format() {
    echo -e "${YELLOW}Checking code formatting with black...${NC}"
    if uv run black --check .; then
        echo -e "${GREEN}All files are properly formatted!${NC}"
        return 0
    else
        echo -e "${RED}Some files need formatting. Run './quality.sh fix' to auto-fix.${NC}"
        return 1
    fi
}

fix_format() {
    echo -e "${YELLOW}Formatting code with black...${NC}"
    uv run black .
    echo -e "${GREEN}Code formatting complete!${NC}"
}

run_tests() {
    echo -e "${YELLOW}Running tests...${NC}"
    uv run pytest backend/tests -v
    echo -e "${GREEN}Tests complete!${NC}"
}

run_all() {
    check_format
    run_tests
    echo -e "${GREEN}All quality checks passed!${NC}"
}

COMMAND=${1:-check}

case $COMMAND in
    check)
        check_format
        ;;
    fix)
        fix_format
        ;;
    all)
        run_all
        ;;
    *)
        usage
        ;;
esac
