#!/usr/bin/env python3
"""
K-means Performance Analysis Script

This script analyzes the timing results from comprehensive K-means testing,
computing performance metrics and generating visualization plots.

Author: Automated analysis script
Date: 2025-08-30
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class KmeansPerformanceAnalyzer:
    def __init__(self, base_path=".", output_dir="analysis_results"):
        """Initialize the analyzer with paths."""
        self.base_path = Path(base_path)
        self.output_dir = Path(output_dir)
        self.timing_logs_dir = self.base_path / "logs" / "timing_logs"
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "metrics").mkdir(exist_ok=True)
        
        # Initialize data containers
        self.data = {}
        self.sequential_baseline = {}
        
    def load_timing_data(self):
        """Load timing data from all log files."""
        print("Loading timing data...")
        
        # Load sequential data
        seq_file = self.timing_logs_dir / "sequential_times.log"
        if seq_file.exists():
            self.data['sequential'] = pd.read_csv(seq_file, comment='#')
            print(f"  Loaded {len(self.data['sequential'])} sequential timings")
        
        # Load OpenMP data
        omp_file = self.timing_logs_dir / "openmp_times.log"
        if omp_file.exists():
            self.data['openmp'] = pd.read_csv(omp_file, comment='#')
            print(f"  Loaded {len(self.data['openmp'])} OpenMP timings")
        
        # Load MPI data
        mpi_file = self.timing_logs_dir / "mpi_times.log"
        if mpi_file.exists():
            self.data['mpi'] = pd.read_csv(mpi_file, comment='#')
            print(f"  Loaded {len(self.data['mpi'])} MPI timings")
        
        # Load MPI+OpenMP data
        mpi_omp_file = self.timing_logs_dir / "mpi_openmp_times.log"
        if mpi_omp_file.exists():
            self.data['mpi_openmp'] = pd.read_csv(mpi_omp_file, comment='#')
            print(f"  Loaded {len(self.data['mpi_openmp'])} MPI+OpenMP timings")
        
        if not self.data:
            print("ERROR: No timing data found!")
            sys.exit(1)
            
    def compute_sequential_baseline(self):
        """Compute sequential baseline times for each dataset."""
        if 'sequential' not in self.data:
            print("WARNING: No sequential data found for baseline computation")
            return
            
        print("Computing sequential baselines...")
        seq_data = self.data['sequential']
        
        for dataset in seq_data['dataset'].unique():
            dataset_times = seq_data[seq_data['dataset'] == dataset]['computation_time']
            self.sequential_baseline[dataset] = {
                'mean': dataset_times.mean(),
                'std': dataset_times.std(),
                'min': dataset_times.min(),
                'max': dataset_times.max()
            }
            print(f"  {dataset}: {self.sequential_baseline[dataset]['mean']:.3f}s ± {self.sequential_baseline[dataset]['std']:.3f}s")
    
    def compute_performance_metrics(self):
        """Compute comprehensive performance metrics for all implementations."""
        print("Computing performance metrics...")
        
        metrics = {}
        
        for impl_name, data in self.data.items():
            if impl_name == 'sequential':
                continue
                
            print(f"  Processing {impl_name}...")
            impl_metrics = []
            
            for dataset in data['dataset'].unique():
                dataset_data = data[data['dataset'] == dataset]
                
                if impl_name == 'openmp':
                    group_cols = ['dataset', 'threads']
                elif impl_name == 'mpi':
                    group_cols = ['dataset', 'processes']
                elif impl_name == 'mpi_openmp':
                    group_cols = ['dataset', 'processes', 'threads', 'total_cores']
                
                grouped = dataset_data.groupby(group_cols[1:])
                
                for config, group in grouped:
                    times = group['computation_time']
                    
                    # Basic statistics
                    mean_time = times.mean()
                    std_time = times.std()
                    min_time = times.min()
                    max_time = times.max()
                    
                    # Performance metrics vs sequential
                    if dataset in self.sequential_baseline:
                        seq_time = self.sequential_baseline[dataset]['mean']
                        speedup = seq_time / mean_time
                        
                        if impl_name == 'openmp':
                            cores = config  # threads
                            efficiency = speedup / cores * 100
                        elif impl_name == 'mpi':
                            cores = config  # processes
                            efficiency = speedup / cores * 100
                        elif impl_name == 'mpi_openmp':
                            if isinstance(config, tuple) and len(config) >= 3:
                                cores = config[2]  # total_cores
                            else:
                                cores = config
                            efficiency = speedup / cores * 100
                    else:
                        speedup = np.nan
                        efficiency = np.nan
                        cores = np.nan
                    
                    # Create metric record
                    metric_record = {
                        'dataset': dataset,
                        'mean_time': mean_time,
                        'std_time': std_time,
                        'min_time': min_time,
                        'max_time': max_time,
                        'speedup': speedup,
                        'efficiency': efficiency,
                        'num_runs': len(times)
                    }
                    
                    # Add implementation-specific fields
                    if impl_name == 'openmp':
                        metric_record['threads'] = config
                        metric_record['cores'] = config
                    elif impl_name == 'mpi':
                        metric_record['processes'] = config
                        metric_record['cores'] = config
                    elif impl_name == 'mpi_openmp':
                        if isinstance(config, tuple):
                            metric_record['processes'] = config[0]
                            metric_record['threads'] = config[1]
                            metric_record['cores'] = config[2] if len(config) > 2 else config[0] * config[1]
                        else:
                            metric_record['processes'] = 1
                            metric_record['threads'] = config
                            metric_record['cores'] = config
                    
                    impl_metrics.append(metric_record)
            
            metrics[impl_name] = pd.DataFrame(impl_metrics)
        
        self.metrics = metrics
        return metrics
    
    def save_metrics_to_csv(self):
        """Save computed metrics to CSV files."""
        print("Saving metrics to CSV files...")
        
        for impl_name, metrics_df in self.metrics.items():
            output_file = self.output_dir / "metrics" / f"{impl_name}_metrics.csv"
            metrics_df.to_csv(output_file, index=False)
            print(f"  Saved {impl_name} metrics to {output_file}")
    
    def plot_speedup_analysis(self):
        """Generate speedup analysis plots."""
        print("Generating speedup analysis plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('K-means Speedup Analysis', fontsize=16, fontweight='bold')
        
        datasets = list(self.sequential_baseline.keys())
        
        for idx, impl_name in enumerate(['openmp', 'mpi']):
            if impl_name not in self.metrics:
                continue
                
            ax = axes[idx, 0]
            metrics_df = self.metrics[impl_name]
            
            for dataset in datasets:
                dataset_metrics = metrics_df[metrics_df['dataset'] == dataset]
                if len(dataset_metrics) == 0:
                    continue
                    
                cores = dataset_metrics['cores'].values
                speedup = dataset_metrics['speedup'].values
                
                ax.plot(cores, speedup, 'o-', label=dataset, linewidth=2, markersize=6)
            
            # Add ideal speedup line
            max_cores = max([df['cores'].max() for df in self.metrics.values() if len(df) > 0])
            ideal_cores = np.linspace(1, max_cores, 100)
            ax.plot(ideal_cores, ideal_cores, 'k--', alpha=0.5, label='Ideal Speedup')
            
            ax.set_xlabel('Number of Cores')
            ax.set_ylabel('Speedup')
            ax.set_title(f'{impl_name.upper()} Speedup vs Cores')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_xscale('log', base=2)
            ax.set_yscale('log', base=2)
            
            # Efficiency plot
            ax_eff = axes[idx, 1]
            for dataset in datasets:
                dataset_metrics = metrics_df[metrics_df['dataset'] == dataset]
                if len(dataset_metrics) == 0:
                    continue
                    
                cores = dataset_metrics['cores'].values
                efficiency = dataset_metrics['efficiency'].values
                
                ax_eff.plot(cores, efficiency, 'o-', label=dataset, linewidth=2, markersize=6)
            
            ax_eff.axhline(y=100, color='k', linestyle='--', alpha=0.5, label='Ideal Efficiency')
            ax_eff.set_xlabel('Number of Cores')
            ax_eff.set_ylabel('Efficiency (%)')
            ax_eff.set_title(f'{impl_name.upper()} Efficiency vs Cores')
            ax_eff.grid(True, alpha=0.3)
            ax_eff.legend()
            ax_eff.set_xscale('log', base=2)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "speedup_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_weak_scaling_analysis(self):
        """Generate weak scaling analysis plots."""
        print("Generating weak scaling analysis plots...")
        
        # Sort datasets by size (assuming larger datasets have larger numbers in filename)
        datasets = sorted(self.sequential_baseline.keys(), 
                         key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else 0)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('K-means Weak Scaling Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Execution time vs dataset size for different core counts
        ax1 = axes[0, 0]
        for impl_name in ['openmp', 'mpi']:
            if impl_name not in self.metrics:
                continue
                
            metrics_df = self.metrics[impl_name]
            
            # Get common core counts across datasets
            common_cores = [1, 2, 4, 8, 16, 32]
            
            for cores in common_cores:
                times = []
                valid_datasets = []
                
                for dataset in datasets:
                    dataset_data = metrics_df[(metrics_df['dataset'] == dataset) & 
                                            (metrics_df['cores'] == cores)]
                    if len(dataset_data) > 0:
                        times.append(dataset_data['mean_time'].iloc[0])
                        valid_datasets.append(dataset)
                
                if len(times) > 1:
                    ax1.plot(range(len(valid_datasets)), times, 'o-', 
                            label=f'{impl_name} {cores} cores', linewidth=2, markersize=6)
        
        ax1.set_xlabel('Dataset Size (relative)')
        ax1.set_ylabel('Execution Time (s)')
        ax1.set_title('Execution Time vs Dataset Size')
        ax1.set_xticks(range(len(datasets)))
        ax1.set_xticklabels([d.replace('.inp', '').replace('data/', '') for d in datasets], rotation=45)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_yscale('log')
        
        # Plot 2: Efficiency vs dataset size
        ax2 = axes[0, 1]
        for impl_name in ['openmp', 'mpi']:
            if impl_name not in self.metrics:
                continue
                
            metrics_df = self.metrics[impl_name]
            
            for cores in [8, 16, 32]:
                efficiencies = []
                valid_datasets = []
                
                for dataset in datasets:
                    dataset_data = metrics_df[(metrics_df['dataset'] == dataset) & 
                                            (metrics_df['cores'] == cores)]
                    if len(dataset_data) > 0:
                        efficiencies.append(dataset_data['efficiency'].iloc[0])
                        valid_datasets.append(dataset)
                
                if len(efficiencies) > 1:
                    ax2.plot(range(len(valid_datasets)), efficiencies, 'o-', 
                            label=f'{impl_name} {cores} cores', linewidth=2, markersize=6)
        
        ax2.set_xlabel('Dataset Size (relative)')
        ax2.set_ylabel('Efficiency (%)')
        ax2.set_title('Efficiency vs Dataset Size')
        ax2.set_xticks(range(len(datasets)))
        ax2.set_xticklabels([d.replace('.inp', '').replace('data/', '') for d in datasets], rotation=45)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: MPI+OpenMP comparison
        if 'mpi_openmp' in self.metrics:
            ax3 = axes[1, 0]
            metrics_df = self.metrics['mpi_openmp']
            
            for dataset in datasets[:2]:  # Show for first two datasets to avoid clutter
                dataset_data = metrics_df[metrics_df['dataset'] == dataset]
                if len(dataset_data) == 0:
                    continue
                    
                cores = dataset_data['cores'].values
                speedup = dataset_data['speedup'].values
                
                ax3.plot(cores, speedup, 'o-', label=f'{dataset}', linewidth=2, markersize=6)
            
            ax3.set_xlabel('Total Cores')
            ax3.set_ylabel('Speedup')
            ax3.set_title('MPI+OpenMP Hybrid Speedup')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            ax3.set_xscale('log', base=2)
            ax3.set_yscale('log', base=2)
        
        # Plot 4: Standard deviation analysis
        ax4 = axes[1, 1]
        for impl_name in ['openmp', 'mpi']:
            if impl_name not in self.metrics:
                continue
                
            metrics_df = self.metrics[impl_name]
            
            cores = metrics_df['cores'].values
            std_times = metrics_df['std_time'].values
            mean_times = metrics_df['mean_time'].values
            cv = (std_times / mean_times) * 100  # Coefficient of variation
            
            ax4.scatter(cores, cv, alpha=0.6, label=f'{impl_name}', s=50)
        
        ax4.set_xlabel('Number of Cores')
        ax4.set_ylabel('Coefficient of Variation (%)')
        ax4.set_title('Timing Variability vs Cores')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.set_xscale('log', base=2)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "weak_scaling_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_hybrid_comparison(self):
        """Generate MPI+OpenMP hybrid comparison plots."""
        if 'mpi_openmp' not in self.metrics:
            print("No MPI+OpenMP data available for hybrid comparison")
            return
            
        print("Generating hybrid comparison plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('MPI+OpenMP Hybrid Performance Analysis', fontsize=16, fontweight='bold')
        
        metrics_df = self.metrics['mpi_openmp']
        datasets = metrics_df['dataset'].unique()
        
        # Plot 1: Speedup heatmap for different process/thread combinations
        ax1 = axes[0, 0]
        
        # Choose one dataset for detailed analysis
        main_dataset = datasets[0] if len(datasets) > 0 else None
        if main_dataset:
            dataset_data = metrics_df[metrics_df['dataset'] == main_dataset]
            
            # Create pivot table for heatmap
            pivot_data = dataset_data.pivot_table(values='speedup', 
                                                 index='processes', 
                                                 columns='threads', 
                                                 fill_value=0)
            
            sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='viridis', ax=ax1)
            ax1.set_title(f'Speedup Heatmap - {main_dataset}')
            ax1.set_xlabel('Threads per Process')
            ax1.set_ylabel('Number of Processes')
        
        # Plot 2: Efficiency comparison
        ax2 = axes[0, 1]
        for dataset in datasets[:2]:  # Limit to avoid clutter
            dataset_data = metrics_df[metrics_df['dataset'] == dataset]
            cores = dataset_data['cores'].values
            efficiency = dataset_data['efficiency'].values
            
            ax2.scatter(cores, efficiency, alpha=0.7, label=dataset, s=60)
        
        ax2.axhline(y=100, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Total Cores')
        ax2.set_ylabel('Efficiency (%)')
        ax2.set_title('MPI+OpenMP Efficiency')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: Compare pure MPI vs MPI+OpenMP
        ax3 = axes[1, 0]
        if 'mpi' in self.metrics:
            for dataset in datasets[:2]:
                # MPI data
                mpi_data = self.metrics['mpi'][self.metrics['mpi']['dataset'] == dataset]
                if len(mpi_data) > 0:
                    ax3.plot(mpi_data['cores'], mpi_data['speedup'], 
                           'o-', label=f'Pure MPI - {dataset}', linewidth=2)
                
                # MPI+OpenMP data  
                hybrid_data = metrics_df[metrics_df['dataset'] == dataset]
                if len(hybrid_data) > 0:
                    ax3.plot(hybrid_data['cores'], hybrid_data['speedup'], 
                           's--', label=f'MPI+OpenMP - {dataset}', linewidth=2)
        
        ax3.set_xlabel('Total Cores')
        ax3.set_ylabel('Speedup')
        ax3.set_title('Pure MPI vs MPI+OpenMP Speedup')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.set_xscale('log', base=2)
        ax3.set_yscale('log', base=2)
        
        # Plot 4: Process vs Thread scaling
        ax4 = axes[1, 1]
        if main_dataset:
            dataset_data = metrics_df[metrics_df['dataset'] == main_dataset]
            
            # Group by number of processes
            for processes in sorted(dataset_data['processes'].unique()):
                proc_data = dataset_data[dataset_data['processes'] == processes]
                threads = proc_data['threads'].values
                speedup = proc_data['speedup'].values
                
                ax4.plot(threads, speedup, 'o-', label=f'{processes} processes', 
                        linewidth=2, markersize=6)
        
        ax4.set_xlabel('Threads per Process')
        ax4.set_ylabel('Speedup')
        ax4.set_title(f'Thread Scaling by Process Count - {main_dataset}')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.set_xscale('log', base=2)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "hybrid_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        print("Generating summary report...")
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("K-MEANS COMPREHENSIVE PERFORMANCE ANALYSIS REPORT")
        report_lines.append("="*80)
        report_lines.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Sequential baseline summary
        report_lines.append("SEQUENTIAL BASELINE PERFORMANCE:")
        report_lines.append("-" * 40)
        for dataset, baseline in self.sequential_baseline.items():
            report_lines.append(f"{dataset:20s}: {baseline['mean']:8.3f}s ± {baseline['std']:6.3f}s")
        report_lines.append("")
        
        # Best performance summary for each implementation
        for impl_name, metrics_df in self.metrics.items():
            if len(metrics_df) == 0:
                continue
                
            report_lines.append(f"{impl_name.upper()} BEST PERFORMANCE:")
            report_lines.append("-" * 40)
            
            for dataset in metrics_df['dataset'].unique():
                dataset_data = metrics_df[metrics_df['dataset'] == dataset]
                best_speedup = dataset_data.loc[dataset_data['speedup'].idxmax()]
                
                if impl_name == 'openmp':
                    config_str = f"{best_speedup['threads']} threads"
                elif impl_name == 'mpi':
                    config_str = f"{best_speedup['processes']} processes"
                elif impl_name == 'mpi_openmp':
                    config_str = f"{best_speedup['processes']}p×{best_speedup['threads']}t"
                
                report_lines.append(f"{dataset:15s}: {best_speedup['speedup']:6.2f}x speedup "
                                  f"({best_speedup['efficiency']:5.1f}% eff.) with {config_str}")
            report_lines.append("")
        
        # Scaling analysis
        report_lines.append("SCALING ANALYSIS:")
        report_lines.append("-" * 40)
        for impl_name, metrics_df in self.metrics.items():
            if len(metrics_df) == 0:
                continue
                
            # Find best scaling dataset
            best_scaling = {}
            for dataset in metrics_df['dataset'].unique():
                dataset_data = metrics_df[metrics_df['dataset'] == dataset]
                max_speedup = dataset_data['speedup'].max()
                max_cores = dataset_data.loc[dataset_data['speedup'].idxmax(), 'cores']
                scaling_efficiency = max_speedup / max_cores * 100
                best_scaling[dataset] = scaling_efficiency
            
            best_dataset = max(best_scaling, key=best_scaling.get)
            report_lines.append(f"{impl_name:12s}: Best scaling on {best_dataset} "
                              f"({best_scaling[best_dataset]:.1f}% scaling efficiency)")
        
        # Save report
        report_file = self.output_dir / "performance_report.txt"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Summary report saved to {report_file}")
        
        # Also print to console
        print("\n" + '\n'.join(report_lines))
    
    def run_complete_analysis(self):
        """Run the complete performance analysis."""
        print("Starting comprehensive K-means performance analysis...")
        print("="*60)
        
        # Load and process data
        self.load_timing_data()
        self.compute_sequential_baseline()
        self.compute_performance_metrics()
        
        # Save metrics
        self.save_metrics_to_csv()
        
        # Generate plots
        self.plot_speedup_analysis()
        self.plot_weak_scaling_analysis()
        self.plot_hybrid_comparison()
        
        # Generate summary
        self.generate_summary_report()
        
        print("="*60)
        print("Analysis complete! Results saved in:", self.output_dir.absolute())
        print("  - Metrics (CSV): analysis_results/metrics/")
        print("  - Plots (PNG): analysis_results/plots/")
        print("  - Report (TXT): analysis_results/performance_report.txt")


def main():
    """Main function to run the analysis."""
    # Check if we're running from the correct directory
    if not Path("logs/timing_logs").exists():
        print("ERROR: timing_logs directory not found!")
        print("Please run this script from the K-means project root directory.")
        sys.exit(1)
    
    # Run analysis
    analyzer = KmeansPerformanceAnalyzer()
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()
