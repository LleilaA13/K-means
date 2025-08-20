#!/bin/bash
# K-means Master Script Runner
# Provides easy access to all scripts from the project root

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

show_usage() {
    echo "K-means Script Runner"
    echo "Usage: $0 <category> <script> [args...]"
    echo ""
    echo "Categories:"
    echo "  local     - Local execution scripts"
    echo "  slurm     - SLURM cluster scripts"
    echo "  analysis  - Analysis and visualization tools"
    echo "  utils     - Utility scripts"
    echo ""
    echo "Available scripts:"
    echo ""
    echo "LOCAL SCRIPTS:"
    echo "  run_kmeans     - General-purpose K-means runner"
    echo "  cluster_run    - Legacy cluster runner (deprecated)"
    echo ""
    echo "SLURM SCRIPTS:"
    echo "  omp_perf       - OpenMP performance test (slurm_omp_performance.sh)"
    echo "  comprehensive  - Multi-version comparison (slurm_comprehensive.sh)"
    echo "  cuda          - GPU performance test (slurm_cuda.sh)"
    echo "  python        - Python analysis runner (slurm_python.sh)"
    echo ""
    echo "ANALYSIS SCRIPTS:"
    echo "  log           - Performance analysis (log_module.py)"
    echo "  compare       - Results comparison (compare_module.py)"
    echo "  plot          - 2D visualization (plot2d_module.py)"
    echo "  quick         - Quick testing (quick_test.py)"
    echo ""
    echo "Examples:"
    echo "  $0 local run_kmeans"
    echo "  $0 slurm omp_perf"
    echo "  $0 analysis log"
    echo "  $0 slurm comprehensive --help"
}

# Check if we're in the right directory
if [ ! -f "$PROJECT_ROOT/Makefile" ]; then
    print_error "Please run this script from the K-means project directory"
    exit 1
fi

# Change to project root
cd "$PROJECT_ROOT"

# Parse arguments
if [ $# -lt 2 ]; then
    show_usage
    exit 1
fi

CATEGORY="$1"
SCRIPT="$2"
shift 2  # Remove first two arguments, leave the rest for the script

case "$CATEGORY" in
    "local")
        case "$SCRIPT" in
            "run_kmeans")
                print_header "Running local K-means script"
                exec scripts/local/run_kmeans.sh "$@"
                ;;
            "cluster_run")
                print_warning "cluster_run.sh is deprecated, use run_kmeans.sh instead"
                exec scripts/local/cluster_run.sh "$@"
                ;;
            *)
                print_error "Unknown local script: $SCRIPT"
                show_usage
                exit 1
                ;;
        esac
        ;;
    
    "slurm")
        case "$SCRIPT" in
            "omp_perf"|"omp_performance")
                print_header "Submitting OpenMP performance test to SLURM"
                exec sbatch scripts/slurm/slurm_omp_performance.sh
                ;;
            "comprehensive"|"comp")
                print_header "Submitting comprehensive analysis to SLURM"
                exec sbatch scripts/slurm/slurm_comprehensive.sh
                ;;
            "cuda"|"gpu")
                print_header "Submitting CUDA performance test to SLURM"
                exec sbatch scripts/slurm/slurm_cuda.sh
                ;;
            "python"|"py")
                print_header "Submitting Python analysis to SLURM"
                exec sbatch scripts/slurm/slurm_python.sh
                ;;
            *)
                print_error "Unknown SLURM script: $SCRIPT"
                show_usage
                exit 1
                ;;
        esac
        ;;
    
    "analysis")
        case "$SCRIPT" in
            "log")
                print_header "Running performance analysis"
                exec python3 scripts/analysis/log_module.py "$@"
                ;;
            "compare")
                print_header "Running results comparison"
                exec python3 scripts/analysis/compare_module.py "$@"
                ;;
            "plot"|"plot2d")
                print_header "Running 2D visualization"
                exec python3 scripts/analysis/plot2d_module.py "$@"
                ;;
            "quick")
                print_header "Running quick test"
                exec python3 scripts/analysis/quick_test.py "$@"
                ;;
            *)
                print_error "Unknown analysis script: $SCRIPT"
                show_usage
                exit 1
                ;;
        esac
        ;;
    
    "utils")
        print_warning "No utility scripts available yet"
        exit 1
        ;;
    
    "help"|"--help"|"-h")
        show_usage
        exit 0
        ;;
    
    *)
        print_error "Unknown category: $CATEGORY"
        show_usage
        exit 1
        ;;
esac
