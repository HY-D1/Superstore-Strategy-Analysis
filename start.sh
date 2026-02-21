#!/bin/bash
# =============================================================================
# Superstore Strategy Analysis - One-Click Runner
# =============================================================================
# This script automates the entire analysis pipeline:
#   Day 1: Data Cleaning + QA
#   Day 2: Insights (tables + charts)
#   Day 3: Forecast + RFM Segmentation
#   Day 4: BI Export (Power BI ready)
#   Day 5: Executive Summary + Slide Outline
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
VENV_DIR=".venv"
RAW_DATA="data_raw/train.csv"
CLEAN_DATA="data_clean/Superstore_Cleaned.xlsx"
OUTPUT_DIR="outputs"

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# =============================================================================
# Step 0: Environment Setup
# =============================================================================

setup_environment() {
    print_header "STEP 0: Environment Setup"
    
    # Check Python version
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3.9 or higher."
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    print_info "Python version: $PYTHON_VERSION"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "$VENV_DIR" ]; then
        print_info "Creating virtual environment..."
        python3 -m venv "$VENV_DIR"
        print_success "Virtual environment created"
    else
        print_info "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    print_success "Virtual environment activated"
    
    # Upgrade pip
    pip install --quiet --upgrade pip
    
    # Install requirements
    print_info "Installing dependencies..."
    pip install --quiet -r requirements.txt
    print_success "Dependencies installed"
}

# =============================================================================
# Step 1: Data Cleaning
# =============================================================================

day1_clean() {
    print_header "STEP 1: Data Cleaning + QA"
    
    # Check for raw data
    if [ ! -f "$RAW_DATA" ]; then
        # Try sample data
        if [ -f "data_raw/sample_train.csv" ]; then
            print_warning "Using sample_train.csv as input"
            RAW_DATA="data_raw/sample_train.csv"
        else
            print_error "No data file found in data_raw/"
            exit 1
        fi
    fi
    
    print_info "Input file: $RAW_DATA"
    
    # Create output directories
    mkdir -p data_clean outputs
    
    # Run cleaning script
    python src/day1_clean.py \
        --input "$RAW_DATA" \
        --output "$CLEAN_DATA"
    
    if [ -f "$CLEAN_DATA" ]; then
        print_success "Cleaned data saved to: $CLEAN_DATA"
        print_info "QA report saved to: outputs/day1_qa_report.md"
    else
        print_error "Cleaning failed - output file not created"
        exit 1
    fi
}

# =============================================================================
# Step 2: Insights Generation
# =============================================================================

day2_insights() {
    print_header "STEP 2: Generate Insights"
    
    if [ ! -f "$CLEAN_DATA" ]; then
        print_error "Cleaned data not found. Run Day 1 first."
        exit 1
    fi
    
    python src/day2_insights.py \
        --input "$CLEAN_DATA"
    
    if [ -f "outputs/day2_insights.md" ]; then
        print_success "Insights report generated: outputs/day2_insights.md"
        print_info "Charts saved to: outputs/day2_charts/"
        print_info "Tables saved to: outputs/day2_tables/"
    else
        print_error "Insights generation failed"
        exit 1
    fi
}

# =============================================================================
# Step 3: Forecast + RFM Segmentation
# =============================================================================

day3_forecast_rfm() {
    print_header "STEP 3: Forecast + RFM Segmentation"
    
    if [ ! -f "$CLEAN_DATA" ]; then
        print_error "Cleaned data not found. Run Day 1 first."
        exit 1
    fi
    
    python src/day3_forecast_rfm.py \
        --input "$CLEAN_DATA" \
        --horizon 12
    
    if [ -f "outputs/day3_forecast.csv" ]; then
        print_success "Forecast generated: outputs/day3_forecast.csv"
        print_success "RFM segments saved: outputs/day3_rfm_segments.csv"
        print_info "Forecast chart: outputs/day3_charts/monthly_sales_forecast.png"
    else
        print_error "Forecast generation failed"
        exit 1
    fi
}

# =============================================================================
# Step 4: BI Export
# =============================================================================

day4_bi_export() {
    print_header "STEP 4: Power BI Export"
    
    python src/day4_export_bi.py \
        --master_xlsx "$CLEAN_DATA" \
        --rfm_csv outputs/day3_rfm_segments.csv \
        --out_csv outputs/bi/superstore_bi.csv
    
    if [ -f "outputs/bi/superstore_bi.csv" ]; then
        print_success "BI-ready data exported: outputs/bi/superstore_bi.csv"
    else
        print_error "BI export failed"
        exit 1
    fi
}

# =============================================================================
# Step 5: Executive Summary
# =============================================================================

day5_story_pack() {
    print_header "STEP 5: Executive Summary + Story Pack"
    
    python src/day5_story_pack.py \
        --out_exec outputs/day5_executive_summary.md \
        --out_slides docs/day5_slide_outline.md \
        --out_talk docs/day5_talk_track.md
    
    if [ -f "outputs/day5_executive_summary.md" ]; then
        print_success "Executive summary: outputs/day5_executive_summary.md"
        print_success "Slide outline: docs/day5_slide_outline.md"
        print_success "Talk track: docs/day5_talk_track.md"
    else
        print_error "Story pack generation failed"
        exit 1
    fi
}

# =============================================================================
# Summary Report
# =============================================================================

print_summary() {
    print_header "ANALYSIS COMPLETE - SUMMARY"
    
    echo ""
    echo -e "${GREEN}Generated Files:${NC}"
    echo ""
    
    echo -e "${CYAN}📊 Data:${NC}"
    ls -lh "$CLEAN_DATA" 2>/dev/null || echo "  (not found)"
    
    echo ""
    echo -e "${CYAN}📈 Reports:${NC}"
    for file in outputs/*.md; do
        [ -f "$file" ] && echo "  ✓ $(basename $file)"
    done
    
    echo ""
    echo -e "${CYAN}📉 Charts:${NC}"
    find outputs -name "*.png" -exec basename {} \; | sed 's/^/  ✓ /'
    
    echo ""
    echo -e "${CYAN}📋 Tables:${NC}"
    find outputs -name "*.csv" -exec basename {} \; | head -10 | sed 's/^/  ✓ /'
    [ $(find outputs -name "*.csv" | wc -l) -gt 10 ] && echo "  ... and more"
    
    echo ""
    echo -e "${CYAN}🎯 Presentations:${NC}"
    [ -f "docs/day5_slide_outline.md" ] && echo "  ✓ Slide outline"
    [ -f "docs/day5_talk_track.md" ] && echo "  ✓ Talk track"
    
    echo ""
    echo -e "${GREEN}Next Steps:${NC}"
    echo "  1. Review outputs/day5_executive_summary.md for key findings"
    echo "  2. Check outputs/day2_charts/ for visualizations"
    echo "  3. Import outputs/bi/superstore_bi.csv into Power BI"
    echo "  4. Use docs/day5_slide_outline.md for your presentation"
    echo "  5. Run './start.sh dashboard' to explore interactively"
    echo ""
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║     Superstore Strategy Analysis Pipeline                    ║${NC}"
    echo -e "${CYAN}║     Clean → Insights → Forecast → RFM → BI → Dashboard       ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    
    START_TIME=$(date +%s)
    
    # Run all steps
    setup_environment
    day1_clean
    day2_insights
    day3_forecast_rfm
    day4_bi_export
    day5_story_pack
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    print_summary
    
    echo ""
    print_success "Total execution time: ${DURATION}s"
    echo ""
}

# =============================================================================
# Streamlit Dashboard
# =============================================================================

launch_dashboard() {
    print_header "🚀 Launching Streamlit Dashboard"
    
    source "$VENV_DIR/bin/activate" 2>/dev/null || true
    
    # Check if outputs exist
    if [ ! -f "outputs/day3_forecast.csv" ]; then
        print_warning "Pipeline outputs not found. Running full pipeline first..."
        main
    fi
    
    print_info "Starting Streamlit dashboard at http://localhost:8501"
    print_info "Press Ctrl+C to stop the server"
    echo ""
    
    streamlit run src/dashboard.py
}

# =============================================================================
# Handle command line arguments
# =============================================================================

case "${1:-}" in
    --help|-h)
        echo "Superstore Strategy Analysis - One-Click Runner"
        echo ""
        echo "Usage: ./start.sh [option]"
        echo ""
        echo "Options:"
        echo "  (no args)       Run full pipeline"
        echo "  day1            Run only data cleaning"
        echo "  day2            Run only insights generation"
        echo "  day3            Run only forecast + RFM"
        echo "  day4            Run only BI export"
        echo "  day5            Run only story pack generation"
        echo "  dashboard       Launch Streamlit dashboard"
        echo "  full+dashboard  Run pipeline then launch dashboard"
        echo "  --help          Show this help message"
        echo ""
        exit 0
        ;;
    day1)
        setup_environment
        day1_clean
        ;;
    day2)
        source "$VENV_DIR/bin/activate" 2>/dev/null || true
        day2_insights
        ;;
    day3)
        source "$VENV_DIR/bin/activate" 2>/dev/null || true
        day3_forecast_rfm
        ;;
    day4)
        source "$VENV_DIR/bin/activate" 2>/dev/null || true
        day4_bi_export
        ;;
    day5)
        source "$VENV_DIR/bin/activate" 2>/dev/null || true
        day5_story_pack
        ;;
    dashboard)
        launch_dashboard
        ;;
    full+dashboard)
        main
        launch_dashboard
        ;;
    *)
        main
        ;;
esac
