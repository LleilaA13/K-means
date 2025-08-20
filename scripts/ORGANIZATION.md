# Scripts Directory - Organized Structure

## 📁 Directory Overview

```
scripts/
├── README.md                    # Main documentation (this file)
├── run.sh                      # Master script runner - USE THIS!
├── local/                      # 🖥️ Local execution scripts
│   ├── run_kmeans.sh          # Universal K-means runner
│   └── cluster_run.sh         # Legacy script (deprecated)
├── slurm/                      # ⚡ SLURM cluster scripts
│   ├── README.md              # SLURM-specific documentation
│   ├── slurm_omp_performance.sh    # OpenMP scaling test
│   ├── slurm_comprehensive.sh      # Multi-version comparison
│   ├── slurm_cuda.sh              # GPU performance test
│   ├── slurm_python.sh            # Python analysis runner
│   └── cluster_analysis.py        # Advanced cluster analysis tool
├── analysis/                   # 📊 Analysis and visualization
│   ├── log_module.py          # Performance analysis & logging
│   ├── compare_module.py      # Results comparison tool
│   ├── plot2d_module.py       # 2D visualization
│   └── quick_test.py          # Fast testing utility
└── utils/                      # 🔧 Utility scripts (future use)
```

## 🚀 Quick Start (New Improved Workflow)

### Using the Master Runner (Recommended)

```bash
# From project root directory:
./scripts/run.sh <category> <script>

# Examples:
./scripts/run.sh local run_kmeans         # Run local K-means
./scripts/run.sh slurm omp_perf          # Submit OpenMP test to cluster
./scripts/run.sh analysis log            # Run performance analysis
./scripts/run.sh slurm python            # Submit Python analysis to cluster
```

### Direct Script Access (Traditional)

```bash
# Local execution
./scripts/local/run_kmeans.sh

# SLURM cluster
sbatch scripts/slurm/slurm_omp_performance.sh

# Analysis tools
python3 scripts/analysis/log_module.py
```

## 📋 Script Categories Explained

### 🖥️ Local Scripts (`local/`)

**Purpose**: Development and local testing

- **Environment**: macOS/Linux workstations
- **Use case**: Development, debugging, small-scale testing

### ⚡ SLURM Scripts (`slurm/`)

**Purpose**: Production HPC cluster execution

- **Environment**: Sapienza University HPC cluster
- **Use case**: Large-scale performance analysis, research

### 📊 Analysis Scripts (`analysis/`)

**Purpose**: Data processing and visualization

- **Environment**: Any Python 3.6+ environment
- **Use case**: Performance evaluation, result validation

### 🔧 Utils Scripts (`utils/`)

**Purpose**: Helper utilities (empty for now)

- **Environment**: Various
- **Use case**: Build automation, data preprocessing

## 🎯 Workflow Recommendations

### Development Workflow

1. `./scripts/run.sh analysis quick` - Quick correctness test
2. `./scripts/run.sh local run_kmeans` - Local performance test
3. `./scripts/run.sh analysis log` - Analyze local results
4. `./scripts/run.sh analysis compare` - Validate correctness

### Production Workflow

1. `./scripts/run.sh slurm omp_perf` - Quick cluster test
2. `./scripts/run.sh slurm python` - Comprehensive cluster analysis
3. Review `logs/cluster_performance_report.txt`

### Research Workflow

1. `./scripts/run.sh slurm comprehensive` - Full multi-version test
2. `./scripts/run.sh slurm cuda` - GPU evaluation (if needed)
3. `./scripts/run.sh analysis plot` - Generate publication plots

## 🔄 Migration from Old Structure

**Before** (flat structure):

```bash
./log_module.py
./slurm_omp_performance.sh
./compare_module.py
```

**After** (organized structure):

```bash
./scripts/run.sh analysis log
./scripts/run.sh slurm omp_perf
./scripts/run.sh analysis compare
```

## 📝 Key Improvements

### ✅ Benefits of New Organization

1. **Clear Separation**: Scripts grouped by purpose and environment
2. **Easy Discovery**: Logical directory structure
3. **Path Safety**: All paths automatically adjusted for subdirectories
4. **Master Runner**: Single entry point with help system
5. **Documentation**: Comprehensive guides in each directory
6. **Future-Proof**: Room for growth in utils/ directory

### 🔧 Automatic Path Handling

All scripts now use correct relative paths:

- `data/` → `../../data/`
- `logs/` → `../../logs/`
- `results/` → `../../results/`
- `src/` → `../../src/`
- `build/` → `../../build/`

### 🎮 Enhanced User Experience

- **Color-coded output** in master runner
- **Help system** with `./scripts/run.sh help`
- **Error checking** and validation
- **Consistent interface** across all script types

## 🏆 Best Practices

1. **Always use the master runner**: `./scripts/run.sh category script`
2. **Check documentation**: Each subdirectory has specific README
3. **Run from project root**: Scripts expect to be run from main directory
4. **Test locally first**: Use local scripts before cluster submission
5. **Monitor cluster jobs**: Use `squeue -u $USER` for SLURM jobs

## 📞 Need Help?

- **General usage**: `./scripts/run.sh help`
- **SLURM cluster**: See `scripts/slurm/README.md`
- **Analysis tools**: See individual script headers
- **Path issues**: Ensure you're in project root directory

---

This organization provides a professional, scalable structure for the K-means project while maintaining full backward compatibility through the master runner script.
