#!/usr/bin/env python3
"""
Comprehensive K-means Performance Analysis Script

This script analyzes timing data from various K-means implementations:
- Sequential (seq)
- OpenMP (omp) 
- MPI (mpi)
- MPI+OpenMP hybrid (mpi_omp)

It creates structured tables showing:
- Dataset name
- Configuration (threads/processes)
- Average computation time
- Standard deviation
- Speedup (relative to sequential)
- Efficiency (speedup / number of cores)

Results are saved as CSV tables and performance plots.
"""

import os
import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime

class KMeansPerformanceAnalyzer:
    def __init__(self, results_dir='results', analysis_dir='analysis_results'):
        self.results_dir = results_dir
        self.analysis_dir = analysis_dir
        self.implementations = ['seq', 'omp', 'mpi', 'mpi_omp', 'cuda']
        
        # Create analysis directory if it doesn't exist
        if not os.path.exists(self.analysis_dir):
            os.makedirs(self.analysis_dir)
        
        # Store all timing data
        self.timing_data = []
        
        # Sequential baseline times for speedup calculation
        self.sequential_baselines = {}
        
    def parse_filename(self, filename):
        """Parse filename to extract implementation type, dataset, configuration, and run number."""
        # Handle different filename patterns for different implementations
        if filename.startswith('seq_'):
            parts = filename.replace('seq_', '').replace('.out.timing', '').split('_')
            return {
                'implementation': 'seq',
                'dataset': '_'.join(parts[:-1]),
                'run': int(parts[-1].replace('run', ''))
            }
        elif filename.startswith('omp_'):
            parts = filename.replace('omp_', '').replace('.out.timing', '').split('_')
            threads = int(parts[-2].replace('t', ''))
            return {
                'implementation': 'omp',
                'dataset': '_'.join(parts[:-2]),
                'threads': threads,
                'run': int(parts[-1].replace('run', ''))
            }
        elif filename.startswith('mpi_omp_'):
            parts = filename.replace('mpi_omp_', '').replace('.out.timing', '').split('_')
            processes = int(parts[-3].replace('p', ''))
            threads = int(parts[-2].replace('t', ''))
            return {
                'implementation': 'mpi_omp',
                'dataset': '_'.join(parts[:-3]),
                'processes': processes,
                'threads': threads,
                'run': int(parts[-1].replace('run', ''))
            }
        elif filename.startswith('mpi_'):
            parts = filename.replace('mpi_', '').replace('.out.timing', '').split('_')
            processes = int(parts[-2].replace('p', ''))
            return {
                'implementation': 'mpi',
                'dataset': '_'.join(parts[:-2]),
                'processes': processes,
                'run': int(parts[-1].replace('run', ''))
            }
        elif filename.startswith('cuda_'):
            parts = filename.replace('cuda_', '').replace('.out.timing', '').split('_')
            return {
                'implementation': 'cuda',
                'dataset': '_'.join(parts[:-1]),
                'run': int(parts[-1].replace('run', ''))
            }
        else:
            return None
    
    def read_timing_file(self, filepath):
        """Read computation time from a .timing file."""
        try:
            with open(filepath, 'r') as f:
                content = f.read().strip()
                # Extract time from "computation_time: X.XXXXXX"
                match = re.search(r'computation_time:\s*([0-9.]+)', content)
                if match:
                    return float(match.group(1))
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
        return None
    
    def collect_timing_data(self):
        """Collect all timing data from .timing files."""
        timing_files = list(self.results_dir.glob("*.timing"))
        
        print(f"Found {len(timing_files)} timing files")
        
        for timing_file in timing_files:
            filename = timing_file.name
            parsed = self.parse_filename(filename)
            
            if parsed is None:
                print(f"Could not parse filename: {filename}")
                continue
                
            computation_time = self.read_timing_file(timing_file)
            if computation_time is None:
                print(f"Could not read timing from: {filename}")
                continue
            
            # Add timing data
            data_point = parsed.copy()
            data_point['computation_time'] = computation_time
            data_point['filename'] = filename
            
            self.timing_data.append(data_point)
        
        print(f"Successfully parsed {len(self.timing_data)} timing files")
        
        # Add configuration strings and total cores
        self.add_config_info()
    
    def add_config_info(self):
        """Add configuration strings and total core counts to timing data."""
        for data_point in self.timing_data:
            impl = data_point['implementation']
            
            if impl == 'seq':
                data_point['config'] = 'sequential'
                data_point['total_cores'] = 1
            elif impl == 'omp':
                threads = data_point['threads']
                data_point['config'] = f'{threads}t'
                data_point['total_cores'] = threads
            elif impl == 'mpi':
                processes = data_point['processes']
                data_point['config'] = f'{processes}p'
                data_point['total_cores'] = processes
            elif impl == 'mpi_omp':
                processes = data_point['processes']
                threads = data_point['threads']
                data_point['config'] = f'{processes}p_{threads}t'
                data_point['total_cores'] = processes * threads
            elif impl == 'cuda':
                data_point['config'] = 'cuda'
                data_point['total_cores'] = 1  # GPU counts as one unit for comparison
    
    def calculate_baselines(self):
        """Calculate sequential baselines for each dataset."""
        df = pd.DataFrame(self.timing_data)
        
        # Get sequential results
        seq_data = df[df['implementation'] == 'seq']
        
        for dataset in seq_data['dataset'].unique():
            dataset_seq = seq_data[seq_data['dataset'] == dataset]
            if len(dataset_seq) > 0:
                avg_time = dataset_seq['computation_time'].mean()
                self.sequential_baselines[dataset] = avg_time
                print(f"Sequential baseline for {dataset}: {avg_time:.3f}s")
    
    def create_summary_tables(self):
        """Create summary tables for each implementation."""
        df = pd.DataFrame(self.timing_data)
        
        # Group by implementation, dataset, and configuration
        grouped = df.groupby(['implementation', 'dataset', 'config', 'total_cores'])
        
        summary_data = []
        
        for (impl, dataset, config, total_cores), group in grouped:
            times = group['computation_time']
            
            avg_time = times.mean()
            std_time = times.std()
            min_time = times.min()
            max_time = times.max()
            count = len(times)
            
            # Calculate speedup vs sequential
            baseline = self.sequential_baselines.get(dataset, None)
            speedup = baseline / avg_time if baseline and avg_time > 0 else None
            efficiency = speedup / total_cores if speedup and total_cores > 0 else None
            
            summary_data.append({
                'Implementation': impl,
                'Dataset': dataset,
                'Configuration': config,
                'Total_Cores': total_cores,
                'Avg_Time_s': avg_time,
                'Std_Time_s': std_time,
                'Min_Time_s': min_time,
                'Max_Time_s': max_time,
                'Runs': count,
                'Speedup': speedup,
                'Efficiency': efficiency,
                'Sequential_Baseline_s': baseline
            })
        
        return pd.DataFrame(summary_data)
    
    def save_implementation_tables(self, summary_df):
        """Save separate tables for each implementation."""
        implementations = summary_df['Implementation'].unique()
        
        for impl in implementations:
            impl_df = summary_df[summary_df['Implementation'] == impl].copy()
            
            # Sort by dataset and total cores
            impl_df = impl_df.sort_values(['Dataset', 'Total_Cores'])
            
            # Format numeric columns
            for col in ['Avg_Time_s', 'Std_Time_s', 'Min_Time_s', 'Max_Time_s', 'Sequential_Baseline_s']:
                if col in impl_df.columns:
                    impl_df[col] = impl_df[col].round(3)
            
            for col in ['Speedup', 'Efficiency']:
                if col in impl_df.columns:
                    impl_df[col] = impl_df[col].round(2)
            
            # Save to CSV
            filename = self.output_dir / f"{impl}_performance_table.csv"
            impl_df.to_csv(filename, index=False)
            print(f"Saved {impl} performance table: {filename}")
            
            # Also save a formatted version for display
            filename_formatted = self.output_dir / f"{impl}_performance_formatted.txt"
            with open(filename_formatted, 'w') as f:
                f.write(f"{impl.upper()} K-MEANS PERFORMANCE ANALYSIS\n")
                f.write("=" * 50 + "\n\n")
                f.write(impl_df.to_string(index=False))
                f.write("\n\nColumns:\n")
                f.write("- Dataset: Input dataset name\n")
                f.write("- Configuration: Thread/process configuration\n")
                f.write("- Total_Cores: Total computational units used\n")
                f.write("- Avg_Time_s: Average computation time (seconds)\n")
                f.write("- Std_Time_s: Standard deviation of times\n")
                f.write("- Speedup: Performance relative to sequential\n")
                f.write("- Efficiency: Speedup per core used\n")
            
            print(f"Saved {impl} formatted table: {filename_formatted}")
    
    def create_speedup_plots(self, summary_df):
        """Create speedup and efficiency plots."""
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        datasets = summary_df['Dataset'].unique()
        
        for dataset in datasets:
            dataset_df = summary_df[summary_df['Dataset'] == dataset].copy()
            
            if len(dataset_df) == 0:
                continue
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Determine power of 2 ticks based on data range
            all_cores = dataset_df['Total_Cores'].unique()
            max_cores = max(all_cores)
            
            # Generate power of 2 ticks: 1, 2, 4, 8, 16, 32, 64, ...
            power_of_2_ticks = []
            power = 0
            while 2**power <= max_cores * 1.5:  # Go slightly beyond max for better visualization
                if 2**power >= 1:
                    power_of_2_ticks.append(2**power)
                power += 1
            
            # Speedup plot
            implementations = dataset_df['Implementation'].unique()
            for impl in implementations:
                impl_data = dataset_df[dataset_df['Implementation'] == impl]
                if len(impl_data) > 0:
                    ax1.plot(impl_data['Total_Cores'], impl_data['Speedup'], 
                            marker='o', linewidth=2, markersize=8, label=impl)
            
            # Ideal speedup line
            ax1.plot([1, max_cores], [1, max_cores], '--', color='gray', alpha=0.7, label='Ideal')
            
            ax1.set_xlabel('Number of Cores')
            ax1.set_ylabel('Speedup')
            ax1.set_title(f'Speedup vs Cores - {dataset}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(left=0.5)
            ax1.set_ylim(bottom=0)
            ax1.set_xticks(power_of_2_ticks)
            ax1.set_xscale('log', base=2)
            
            # Efficiency plot
            for impl in implementations:
                impl_data = dataset_df[dataset_df['Implementation'] == impl]
                if len(impl_data) > 0:
                    ax2.plot(impl_data['Total_Cores'], impl_data['Efficiency'], 
                            marker='s', linewidth=2, markersize=8, label=impl)
            
            ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Perfect Efficiency')
            ax2.set_xlabel('Number of Cores')
            ax2.set_ylabel('Efficiency')
            ax2.set_title(f'Efficiency vs Cores - {dataset}')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(left=0.5)
            ax2.set_ylim(bottom=0)
            ax2.set_xticks(power_of_2_ticks)
            ax2.set_xscale('log', base=2)
            
            plt.tight_layout()
            
            # Save plot
            plot_filename = self.output_dir / f"performance_plots_{dataset}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Saved performance plot: {plot_filename}")
            plt.close()
    
    def create_comprehensive_comparison(self, summary_df):
        """Create a comprehensive comparison table across all implementations."""
        # Pivot table showing speedup for each implementation
        pivot_df = summary_df.pivot_table(
            index=['Dataset', 'Total_Cores'], 
            columns='Implementation', 
            values='Speedup',
            aggfunc='mean'
        ).round(2)
        
        # Save comparison table
        filename = self.output_dir / "comprehensive_speedup_comparison.csv"
        pivot_df.to_csv(filename)
        print(f"Saved comprehensive comparison: {filename}")
        
        # Create efficiency comparison
        efficiency_pivot = summary_df.pivot_table(
            index=['Dataset', 'Total_Cores'], 
            columns='Implementation', 
            values='Efficiency',
            aggfunc='mean'
        ).round(3)
        
        filename = self.output_dir / "comprehensive_efficiency_comparison.csv"
        efficiency_pivot.to_csv(filename)
        print(f"Saved efficiency comparison: {filename}")
        
        return pivot_df, efficiency_pivot
    
    def generate_summary_report(self, summary_df):
        """Generate a comprehensive summary report."""
        report_file = self.output_dir / "performance_analysis_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("K-MEANS PERFORMANCE ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Dataset summary
            f.write("DATASETS ANALYZED:\n")
            f.write("-" * 20 + "\n")
            for dataset in sorted(summary_df['Dataset'].unique()):
                baseline = self.sequential_baselines.get(dataset, 'N/A')
                f.write(f"- {dataset}: Sequential baseline = {baseline:.3f}s\n")
            f.write("\n")
            
            # Implementation summary
            f.write("IMPLEMENTATIONS TESTED:\n")
            f.write("-" * 25 + "\n")
            for impl in sorted(summary_df['Implementation'].unique()):
                impl_data = summary_df[summary_df['Implementation'] == impl]
                max_speedup = impl_data['Speedup'].max()
                best_config = impl_data.loc[impl_data['Speedup'].idxmax(), 'Configuration']
                f.write(f"- {impl}: Best speedup = {max_speedup:.2f}x ({best_config})\n")
            f.write("\n")
            
            # Best performances per dataset
            f.write("BEST PERFORMANCE PER DATASET:\n")
            f.write("-" * 30 + "\n")
            for dataset in sorted(summary_df['Dataset'].unique()):
                dataset_data = summary_df[summary_df['Dataset'] == dataset]
                if len(dataset_data) > 0:
                    best_idx = dataset_data['Speedup'].idxmax()
                    best_row = dataset_data.loc[best_idx]
                    f.write(f"{dataset}:\n")
                    f.write(f"  Best: {best_row['Implementation']} - {best_row['Configuration']}\n")
                    f.write(f"  Speedup: {best_row['Speedup']:.2f}x\n")
                    f.write(f"  Efficiency: {best_row['Efficiency']:.3f}\n")
                    f.write(f"  Time: {best_row['Avg_Time_s']:.3f}s\n\n")
        
        print(f"Saved comprehensive report: {report_file}")
    
    def run_analysis(self):
        """Run the complete performance analysis."""
        print("Starting K-means Performance Analysis...")
        print("=" * 50)
        
        # Step 1: Collect timing data
        print("\n1. Collecting timing data...")
        self.collect_timing_data()
        
        if not self.timing_data:
            print("No timing data found! Please check the results directory.")
            return
        
        # Step 2: Calculate baselines
        print("\n2. Calculating sequential baselines...")
        self.calculate_baselines()
        
        # Step 3: Create summary tables
        print("\n3. Creating summary tables...")
        summary_df = self.create_summary_tables()
        
        # Step 4: Save implementation-specific tables
        print("\n4. Saving implementation tables...")
        self.save_implementation_tables(summary_df)
        
        # Step 5: Create plots
        print("\n5. Creating performance plots...")
        self.create_speedup_plots(summary_df)
        
        # Step 6: Create comprehensive comparison
        print("\n6. Creating comprehensive comparisons...")
        self.create_comprehensive_comparison(summary_df)
        
        # Step 7: Generate summary report
        print("\n7. Generating summary report...")
        self.generate_summary_report(summary_df)
        
        print(f"\nAnalysis complete! Results saved in: {self.output_dir}")
        print("\nGenerated files:")
        for file in sorted(self.output_dir.glob("*")):
            print(f"  - {file.name}")

def main():
    parser = argparse.ArgumentParser(description='Analyze K-means performance data')
    parser.add_argument('--results-dir', default='results', 
                       help='Directory containing .timing files (default: results)')
    parser.add_argument('--output-dir', default='analysis_results',
                       help='Output directory for analysis results (default: analysis_results)')
    
    args = parser.parse_args()
    
    analyzer = KMeansPerformanceAnalyzer(args.results_dir, args.output_dir)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
