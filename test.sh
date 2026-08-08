#!/bin/bash
cd "$(dirname "$0")"

PASSED=0
FAILED=0
FAILURES=""

run_test() {
    local test_file="$1"
    local label="${test_file#tests/}"
    printf "%-55s " "$label"
    output=$(python3 "$test_file" 2>&1)
    if [ $? -eq 0 ]; then
        echo "PASS"
        PASSED=$((PASSED + 1))
    else
        echo "FAIL"
        FAILED=$((FAILED + 1))
        FAILURES="$FAILURES\n  $label"
        if [ -n "$VERBOSE" ]; then
            echo "$output" | tail -20
            echo ""
        fi
    fi
}

echo "======================================================================"
echo "  RecPulse Test Suite"
echo "======================================================================"
echo ""

echo "--- Unit Tests ---"
for f in tests/unit/test_*.py; do
    run_test "$f"
done

echo ""
echo "--- Autograd Tests ---"
for f in tests/autograd/test_*.py; do
    run_test "$f"
done

echo ""
echo "--- Integration Tests ---"
for f in tests/integration/test_*.py; do
    run_test "$f"
done

echo ""
echo "======================================================================"
TOTAL=$((PASSED + FAILED))
echo "  Results: $PASSED/$TOTAL passed, $FAILED failed"
if [ $FAILED -gt 0 ]; then
    echo -e "  Failed:$FAILURES"
    echo "======================================================================"
    exit 1
else
    echo "======================================================================"
    exit 0
fi
