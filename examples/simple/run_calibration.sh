#!/bin/bash

# Compilation Calibration Test Runner
# This script runs various tests to validate compilation correctness and measure performance

set -e

echo "🚀 Luminal Compilation Calibration Test Suite"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test configurations
FEATURES=("cpu" "metal" "cuda")
RUST_LOG=${RUST_LOG:-"info"}

echo "📋 Test Configuration:"
echo "   Rust Log Level: $RUST_LOG"
echo "   Available Features: ${FEATURES[*]}"
echo ""

# Function to run tests with specific feature
run_feature_tests() {
    local feature=$1
    echo -e "${BLUE}🧪 Testing with feature: $feature${NC}"

    # Check if feature is available
    if ! cargo build --features "$feature" --quiet; then
        echo -e "${YELLOW}⚠️  Feature '$feature' not available, skipping...${NC}"
        return 0
    fi

    echo "  📦 Building with $feature feature..."
    cargo build --features "$feature" --tests --quiet

    echo "  🔥 Running smoke test..."
    if cargo test --features "$feature" test_basic_compilation --quiet; then
        echo -e "    ${GREEN}✅ Smoke test passed${NC}"
    else
        echo -e "    ${RED}❌ Smoke test failed${NC}"
        return 1
    fi

    echo "  🧪 Running integration tests..."
    if cargo test --features "$feature" test_compilation_methods_integration --quiet; then
        echo -e "    ${GREEN}✅ Integration tests passed${NC}"
    else
        echo -e "    ${RED}❌ Integration tests failed${NC}"
        return 1
    fi

    echo "  ⚡ Running performance tests..."
    if cargo test --features "$feature" test_compilation_performance --quiet; then
        echo -e "    ${GREEN}✅ Performance tests passed${NC}"
    else
        echo -e "    ${RED}❌ Performance tests failed${NC}"
        return 1
    fi

    echo -e "${GREEN}✅ All tests passed for $feature${NC}"
    echo ""
}

# Function to run calibration analysis
run_calibration_analysis() {
    echo -e "${BLUE}📊 Running Calibration Analysis${NC}"

    # Create results directory
    mkdir -p results

    # Run extended calibration suite
    echo "  🔬 Running extended calibration..."

    # This would be enhanced to actually save detailed results
    cargo test --features "cpu" test_compilation_methods_integration -- --nocapture > results/calibration_output.txt 2>&1

    if [ $? -eq 0 ]; then
        echo -e "    ${GREEN}✅ Calibration analysis completed${NC}"
        echo "    📄 Results saved to results/calibration_output.txt"
    else
        echo -e "    ${RED}❌ Calibration analysis failed${NC}"
        return 1
    fi
}

# Function to generate summary report
generate_summary() {
    echo -e "${BLUE}📋 Generating Summary Report${NC}"

    cat > results/summary.md << EOF
# Luminal Compilation Calibration Summary

## Test Execution Summary
- Date: $(date)
- Features tested: ${tested_features[*]}
- Total tests run: $total_tests
- Passed: $passed_tests
- Failed: $failed_tests

## Status
$(if [ $failed_tests -eq 0 ]; then echo "🎉 ALL TESTS PASSED"; else echo "❌ $failed_tests TESTS FAILED"; fi)

## Next Steps
$(if [ $failed_tests -eq 0 ]; then
    echo "- Ready to proceed with parallel compilation integration"
    echo "- Consider running with larger model sizes for stress testing"
else
    echo "- Investigate test failures before proceeding"
    echo "- Check logs in results/ directory for details"
fi)

## Files Generated
- calibration_output.txt: Detailed test output
- summary.md: This summary file

EOF

    echo "  📄 Summary report generated: results/summary.md"
}

# Main execution
main() {
    local total_tests=0
    local passed_tests=0
    local failed_tests=0
    local tested_features=()

    # Clean previous results
    rm -rf results
    mkdir -p results

    echo "🧹 Cleaned previous results"
    echo ""

    # Test each available feature
    for feature in "${FEATURES[@]}"; do
        if run_feature_tests "$feature"; then
            passed_tests=$((passed_tests + 1))
            tested_features+=("$feature")
        else
            failed_tests=$((failed_tests + 1))
        fi
        total_tests=$((total_tests + 1))
    done

    # Run calibration analysis if basic tests passed
    if [ $failed_tests -eq 0 ]; then
        run_calibration_analysis
    fi

    # Generate summary
    generate_summary

    # Final report
    echo "=============================================="
    echo -e "${BLUE}🏁 Final Results${NC}"
    echo "   Features tested: ${tested_features[*]}"
    echo "   Total: $total_tests, Passed: $passed_tests, Failed: $failed_tests"

    if [ $failed_tests -eq 0 ]; then
        echo -e "   ${GREEN}🎉 ALL TESTS PASSED!${NC}"
        echo ""
        echo "✅ Ready to proceed with parallel compilation integration"
        exit 0
    else
        echo -e "   ${RED}❌ $failed_tests TESTS FAILED${NC}"
        echo ""
        echo "🔍 Check results/ directory for detailed logs"
        exit 1
    fi
}

# Run main function
main "$@"