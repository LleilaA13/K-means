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
from matplotlib.patches import Patch
from matplotlib.ticker import LogFormatter, FuncFormatter

class KMeansPerformanceAnalyzer:
    def __init__(self, results_dir='results', analysis_dir='analysis_results'):
        self.results_dir = Path(results_dir)
        self.analysis_dir = Path(analysis_dir)
        self.output_dir = self.analysis_dir
        
        # Initialize data storage
        self.timing_data = []
        self.sequential_baselines = {}
        
        # Define dataset size ordering
        self.dataset_size_order = {
            'input100D': 0,
            'input100D2': 1, 
            '200k_100': 2,
            '400k_100': 3,
            '800k_100': 4
        }
    
    def elegant_time_formatter(self, x, pos):
        """Custom formatter for time values to avoid scientific notation."""
        if x >= 1:
            return f'{x:.1f}'
        elif x >= 0.1:
            return f'{x:.2f}'
        elif x >= 0.01:
            return f'{x:.3f}'
        else:
            return f'{x:.4f}'
    
    def sort_datasets_by_size(self, datasets):
        """Sort datasets by their size order."""
        def get_sort_key(dataset):
            # Remove file extension if present
            clean_name = dataset.replace('.inp', '')
            return self.dataset_size_order.get(clean_name, 999)  # Unknown datasets go to end
        
        return sorted(datasets, key=get_sort_key)  # Add output_dir alias
        self.implementations = ['seq', 'omp', 'mpi', 'mpi_omp', 'cuda']
        
        # Create analysis directory if it doesn't exist
        if not self.analysis_dir.exists():
            self.analysis_dir.mkdir(parents=True, exist_ok=True)
        
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
            
            # Sort by dataset size order, then by total cores
            impl_df['Dataset_Order'] = impl_df['Dataset'].map(
                lambda x: self.dataset_size_order.get(x.replace('.inp', ''), 999)
            )
            impl_df = impl_df.sort_values(['Dataset_Order', 'Total_Cores'])
            impl_df = impl_df.drop('Dataset_Order', axis=1)  # Remove the helper column
            
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
        """Create comprehensive speedup and efficiency plots with better visualization."""
        # Set up improved plotting style
        plt.style.use('default')
        sns.set_palette("Set1")
        
        # Define colors for consistent implementation visualization
        impl_colors = {
            'seq': '#1f77b4',      # Blue
            'omp': '#d62728',      # Red  
            'mpi': '#2ca02c',      # Green
            'mpi_omp': '#d62728',  # Red
            'mpi_omp_2p': '#1f77b4',  # Blue for 2 processes
            'mpi_omp_4p': '#ff7f0e',  # Orange for 4 processes
            'cuda': '#9467bd'      # Purple
        }
        
        # Define markers for better distinction
        impl_markers = {
            'seq': 'o',
            'omp': 's', 
            'mpi': '^',
            'mpi_omp': 'D',
            'mpi_omp_2p': 'D',     # Diamond for 2 processes
            'mpi_omp_4p': 'v',     # Triangle down for 4 processes
            'cuda': '*'
        }
        
        # Define line styles for MPI+OpenMP variants
        impl_linestyles = {
            'seq': '-',
            'omp': '-', 
            'mpi': '-',
            'mpi_omp': '-',
            'mpi_omp_2p': '-',     # Solid for 2 processes
            'mpi_omp_4p': '--',    # Dashed for 4 processes
            'cuda': '-'
        }
        
        datasets = summary_df['Dataset'].unique()
        # Sort datasets by size
        datasets = self.sort_datasets_by_size(datasets)
        
        for dataset in datasets:
            dataset_df = summary_df[summary_df['Dataset'] == dataset].copy()
            
            if len(dataset_df) == 0:
                continue
            
            # Create figure with 3 subplots: speedup, efficiency, and execution time
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'K-means Performance Analysis - {dataset}', fontsize=16, fontweight='bold')
            
            # Determine core range for better visualization
            all_cores = sorted(dataset_df['Total_Cores'].unique())
            max_cores = max(all_cores)
            
            # Generate power of 2 ticks for x-axis with proper labels
            power_of_2_ticks = []
            power = 0
            while 2**power <= max_cores * 1.5:
                if 2**power >= 1:
                    power_of_2_ticks.append(2**power)
                power += 1
            
            # 1. SPEEDUP PLOT
            implementations = dataset_df['Implementation'].unique()
            
            # Create separate entries for mpi_omp by process count
            plot_data = []
            for impl in implementations:
                impl_data = dataset_df[dataset_df['Implementation'] == impl].copy()
                
                if impl == 'mpi_omp':
                    # Separate by process count
                    for _, row in impl_data.iterrows():
                        # Extract process count from configuration
                        config = row['Configuration']
                        if 'p' in config:
                            # Parse configuration like "2p_4t" or "4p_2t"
                            parts = config.split('_')
                            for part in parts:
                                if part.endswith('p'):
                                    process_count = int(part[:-1])
                                    impl_key = f'mpi_omp_{process_count}p'
                                    plot_data.append({
                                        'impl_key': impl_key,
                                        'impl_label': f'MPI+OMP ({process_count}p)',
                                        'data': row
                                    })
                                    break
                else:
                    # Regular implementation
                    for _, row in impl_data.iterrows():
                        plot_data.append({
                            'impl_key': impl,
                            'impl_label': impl.upper(),
                            'data': row
                        })
            
            # Group plot data by implementation
            grouped_data = {}
            for item in plot_data:
                key = item['impl_key']
                if key not in grouped_data:
                    grouped_data[key] = {'label': item['impl_label'], 'rows': []}
                grouped_data[key]['rows'].append(item['data'])
            
            # Plot each implementation group
            cuda_speedup = None
            for impl_key, group_info in grouped_data.items():
                if impl_key == 'seq':  # Skip sequential in speedup plot
                    continue
                    
                # Convert rows to DataFrame for easier handling
                group_df = pd.DataFrame(group_info['rows']).sort_values('Total_Cores')
                
                if len(group_df) > 0:
                    color = impl_colors.get(impl_key, impl_colors.get('mpi_omp', 'gray'))
                    marker = impl_markers.get(impl_key, impl_markers.get('mpi_omp', 'o'))
                    linestyle = impl_linestyles.get(impl_key, '-')
                    
                    if impl_key == 'cuda':
                        # For CUDA, use horizontal line at the speedup value
                        cuda_speedup = group_df['Speedup'].iloc[0]  # CUDA should have consistent speedup
                        ax1.axhline(y=cuda_speedup, color=color, linestyle='-', linewidth=2, 
                                   alpha=0.8, label=group_info['label'])
                        # Add annotation for CUDA speedup
                        ax1.annotate(f'{cuda_speedup:.1f}x', 
                                   (max_cores*0.8, cuda_speedup),
                                   textcoords="offset points", xytext=(0,10), 
                                   ha='center', fontsize=10, alpha=0.8, fontweight='bold')
                    else:
                        ax1.plot(group_df['Total_Cores'], group_df['Speedup'], 
                                marker=marker, linewidth=2, markersize=8, linestyle=linestyle,
                                label=group_info['label'], color=color, alpha=0.8)
                        
                        # Add data point annotations for key values
                        for _, row in group_df.iterrows():
                            if row['Total_Cores'] in [1, 2, 4, 8, 16, 32, 64]:
                                ax1.annotate(f'{row["Speedup"]:.1f}x', 
                                           (row['Total_Cores'], row['Speedup']),
                                           textcoords="offset points", xytext=(0,10), 
                                           ha='center', fontsize=9, alpha=0.7)
            
            # Ideal speedup line
            ideal_cores = [c for c in power_of_2_ticks if c <= max_cores]
            ax1.plot(ideal_cores, ideal_cores, '--', color='black', alpha=0.5, 
                    linewidth=2, label='Ideal Linear Speedup')
            
            ax1.set_xlabel('Number of Cores', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Speedup', fontsize=12, fontweight='bold')
            ax1.set_title('Speedup vs Number of Cores', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
            ax1.grid(True, alpha=0.3, linestyle=':')
            ax1.set_xlim(left=0.5, right=max_cores*1.1)
            
            # Calculate maximum speedup including CUDA
            max_speedup = dataset_df['Speedup'].max()
            if cuda_speedup:
                max_speedup = max(max_speedup, cuda_speedup)
            ax1.set_ylim(bottom=0, top=max_speedup*1.1)
            
            ax1.set_xticks(power_of_2_ticks)
            ax1.set_xticklabels([str(tick) for tick in power_of_2_ticks])
            # Use linear scale instead of log scale to show actual numbers
            ax1.set_xlim(left=0.5, right=max_cores*1.1)
            
            # 2. EFFICIENCY PLOT
            for impl_key, group_info in grouped_data.items():
                if impl_key == 'seq' or impl_key == 'cuda':  # Skip sequential and CUDA in efficiency plot
                    continue
                    
                group_df = pd.DataFrame(group_info['rows']).sort_values('Total_Cores')
                
                if len(group_df) > 0:
                    color = impl_colors.get(impl_key, impl_colors.get('mpi_omp', 'gray'))
                    marker = impl_markers.get(impl_key, impl_markers.get('mpi_omp', 'o'))
                    linestyle = impl_linestyles.get(impl_key, '-')
                    
                    ax2.plot(group_df['Total_Cores'], group_df['Efficiency'], 
                            marker=marker, linewidth=2, markersize=8, linestyle=linestyle,
                            label=group_info['label'], color=color, alpha=0.8)
            
            # Perfect efficiency line
            ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, 
                       linewidth=2, label='Perfect Efficiency (100%)')
            
            ax2.set_xlabel('Number of Cores', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Efficiency', fontsize=12, fontweight='bold')
            ax2.set_title('Parallel Efficiency vs Number of Cores', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
            ax2.grid(True, alpha=0.3, linestyle=':')
            ax2.set_xlim(left=0.5, right=max_cores*1.1)
            ax2.set_ylim(bottom=0, top=1.2)
            ax2.set_xticks(power_of_2_ticks)
            ax2.set_xticklabels([str(tick) for tick in power_of_2_ticks])
            # Use linear scale instead of log scale to show actual numbers
            
            # 3. EXECUTION TIME COMPARISON
            for impl_key, group_info in grouped_data.items():
                group_df = pd.DataFrame(group_info['rows']).sort_values('Total_Cores')
                
                if len(group_df) > 0:
                    color = impl_colors.get(impl_key, impl_colors.get('mpi_omp', 'gray'))
                    marker = impl_markers.get(impl_key, impl_markers.get('mpi_omp', 'o'))
                    linestyle = impl_linestyles.get(impl_key, '-')
                    
                    ax3.plot(group_df['Total_Cores'], group_df['Avg_Time_s'], 
                            marker=marker, linewidth=2, markersize=8, linestyle=linestyle,
                            label=group_info['label'], color=color, alpha=0.8)
                    
                    # Add error bars for standard deviation
                    ax3.errorbar(group_df['Total_Cores'], group_df['Avg_Time_s'],
                               yerr=group_df['Std_Time_s'], color=color, alpha=0.3,
                               capsize=5, capthick=2)
            
            ax3.set_xlabel('Number of Cores', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
            ax3.set_title('Execution Time vs Number of Cores', fontsize=14, fontweight='bold')
            ax3.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
            ax3.grid(True, alpha=0.3, linestyle=':')
            ax3.set_xlim(left=0.5, right=max_cores*1.1)
            ax3.set_yscale('log')
            # Use elegant formatting instead of scientific notation
            ax3.yaxis.set_major_formatter(FuncFormatter(self.elegant_time_formatter))
            ax3.grid(True, which='minor', alpha=0.2, linestyle=':')
            if len(power_of_2_ticks) > 1:
                ax3.set_xticks(power_of_2_ticks)
                ax3.set_xticklabels([str(tick) for tick in power_of_2_ticks])
                # Use linear scale instead of log scale to show actual numbers
                ax3.set_xlim(left=0.5, right=max_cores*1.1)
            
            # 4. PERFORMANCE SUMMARY BAR CHART
            # Show best performance for each implementation
            best_performance = []
            for impl_key, group_info in grouped_data.items():
                group_df = pd.DataFrame(group_info['rows'])
                
                if len(group_df) > 0:
                    if impl_key == 'seq':
                        best_perf = group_df.iloc[0]
                        best_speedup = 1.0
                    else:
                        best_perf = group_df.loc[group_df['Speedup'].idxmax()]
                        best_speedup = best_perf['Speedup']
                    
                    best_performance.append({
                        'Implementation': group_info['label'],
                        'Best_Speedup': best_speedup,
                        'Best_Time': best_perf['Avg_Time_s'],
                        'Best_Cores': best_perf['Total_Cores'],
                        'impl_key': impl_key
                    })
            
            if best_performance:
                best_df = pd.DataFrame(best_performance)
                bars = ax4.bar(best_df['Implementation'], best_df['Best_Speedup'], 
                              color=[impl_colors.get(row['impl_key'], 'gray') for _, row in best_df.iterrows()],
                              alpha=0.8, edgecolor='black', linewidth=1.5)
                
                # Add value labels on bars
                for i, (bar, speedup, cores, time, impl_key) in enumerate(zip(bars, best_df['Best_Speedup'], 
                                                   best_df['Best_Cores'], best_df['Best_Time'], best_df['impl_key'])):
                    height = bar.get_height()
                    if impl_key == 'cuda':
                        # For CUDA, don't show cores, just speedup and time
                        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                               f'{speedup:.1f}x\n{time:.2f}s',
                               ha='center', va='bottom', fontsize=10, fontweight='bold')
                    else:
                        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                               f'{speedup:.1f}x\n({cores} cores)\n{time:.2f}s',
                               ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax4.set_ylabel('Best Speedup Achieved', fontsize=12, fontweight='bold')
            ax4.set_title('Best Performance Summary', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3, linestyle=':', axis='y')
            
            # Use the actual maximum speedup for better scaling
            if len(best_df) > 0:
                ax4.set_ylim(bottom=0, top=best_df['Best_Speedup'].max()*1.3)
            else:
                ax4.set_ylim(bottom=0, top=10)
            
            # Use subplots_adjust instead of tight_layout to avoid size issues
            plt.subplots_adjust(left=0.08, bottom=0.08, right=0.95, top=0.92, wspace=0.25, hspace=0.35)
            
            # Save the plot
            plot_filename = self.output_dir / f'performance_plots_{dataset}.png'
            plt.savefig(plot_filename, dpi=200, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            print(f"Saved performance plots: {plot_filename}")
        
        # Create a comprehensive comparison plot across all datasets (optional)
        try:
            self.create_cross_dataset_comparison(summary_df, impl_colors, impl_markers)
        except Exception as e:
            print(f"Warning: Could not create cross-dataset comparison: {e}")
            print("Continuing with analysis...")
        
        # Create implementation-specific analysis plots
        try:
            self.create_implementation_specific_plots(summary_df, impl_colors, impl_markers, impl_linestyles)
        except Exception as e:
            print(f"Warning: Could not create implementation-specific plots: {e}")
            print("Continuing with analysis...")
    
    def create_cross_dataset_comparison(self, summary_df, impl_colors, impl_markers):
        """Create cross-dataset comparison plots."""
        datasets = summary_df['Dataset'].unique()
        # Sort datasets by size
        datasets = self.sort_datasets_by_size(datasets)
        
        if len(datasets) < 2:
            print("Skipping cross-dataset comparison - only one dataset found")
            return  # Skip if only one dataset
        
        try:
            # Create a simpler comparison figure with fixed size
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle('K-means Performance Comparison Across Datasets', fontsize=14, fontweight='bold')
            
            # 1. Best speedup comparison across datasets
            best_speedups = []
            for dataset in datasets:
                dataset_data = summary_df[summary_df['Dataset'] == dataset]
                
                for impl in dataset_data['Implementation'].unique():
                    if impl == 'seq':
                        continue
                    
                    impl_data = dataset_data[dataset_data['Implementation'] == impl]
                    
                    if impl == 'mpi_omp':
                        # For MPI+OMP, find the best speedup across all configurations
                        # and separate by process count for visualization
                        process_groups = {}
                        for _, row in impl_data.iterrows():
                            config = row['Configuration']
                            if 'p' in config:
                                parts = config.split('_')
                                for part in parts:
                                    if part.endswith('p'):
                                        process_count = int(part[:-1])
                                        if process_count not in process_groups:
                                            process_groups[process_count] = []
                                        process_groups[process_count].append(row)
                                        break
                        
                        # Add best speedup for each process configuration
                        for process_count, rows in process_groups.items():
                            if rows:
                                best_row = max(rows, key=lambda x: x['Speedup'])
                                impl_label = f'MPI+OMP ({process_count}p)'
                                best_speedups.append({
                                    'Dataset': dataset,
                                    'Implementation': impl_label,
                                    'Best_Speedup': best_row['Speedup'],
                                    'impl_key': f'mpi_omp_{process_count}p'
                                })
                    else:
                        # Regular implementation - find best speedup
                        if len(impl_data) > 0:
                            best_speedup = impl_data['Speedup'].max()
                            best_speedups.append({
                                'Dataset': dataset,
                                'Implementation': impl.upper(),
                                'Best_Speedup': best_speedup,
                                'impl_key': impl
                            })
            
            if best_speedups:
                speedup_df = pd.DataFrame(best_speedups)
                
                # Create grouped bar chart instead of heatmap
                implementations = speedup_df['Implementation'].unique()
                x = np.arange(len(datasets))
                width = 0.8 / len(implementations)
                
                for i, impl in enumerate(implementations):
                    impl_data = speedup_df[speedup_df['Implementation'] == impl]
                    values = []
                    impl_key = None
                    for dataset in datasets:
                        dataset_value = impl_data[impl_data['Dataset'] == dataset]['Best_Speedup']
                        if len(dataset_value) > 0:
                            values.append(dataset_value.iloc[0])
                            if impl_key is None:
                                impl_key = impl_data[impl_data['Dataset'] == dataset]['impl_key'].iloc[0]
                        else:
                            values.append(0)
                    
                    color = impl_colors.get(impl_key, 'gray')
                    ax1.bar(x + i * width, values, width, label=impl, color=color, alpha=0.8)
                
                ax1.set_xlabel('Dataset', fontsize=10, fontweight='bold')
                ax1.set_ylabel('Best Speedup', fontsize=10, fontweight='bold')
                ax1.set_title('Best Speedup by Implementation', fontsize=12, fontweight='bold')
                ax1.set_xticks(x + width * (len(implementations) - 1) / 2)
                ax1.set_xticklabels(datasets, fontsize=9)
                ax1.legend(fontsize=9)
                ax1.grid(True, alpha=0.3, axis='y')
            
            # 2. Sequential execution time comparison
            seq_baselines = []
            for dataset in datasets:
                baseline = self.sequential_baselines.get(dataset)
                if baseline:
                    seq_baselines.append({'Dataset': dataset, 'Sequential_Time': baseline})
            
            if seq_baselines:
                baseline_df = pd.DataFrame(seq_baselines)
                bars = ax2.bar(baseline_df['Dataset'], baseline_df['Sequential_Time'], 
                              color='lightblue', alpha=0.8, edgecolor='black')
                
                # Add value labels on bars
                for bar, time in zip(bars, baseline_df['Sequential_Time']):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                           f'{time:.2f}s', ha='center', va='bottom', fontweight='bold', fontsize=9)
                
                ax2.set_ylabel('Sequential Time (seconds)', fontsize=10, fontweight='bold')
                ax2.set_title('Dataset Complexity Comparison', fontsize=12, fontweight='bold')
                ax2.grid(True, alpha=0.3, linestyle=':', axis='y')
                ax2.set_yscale('log')
                # Use elegant formatting instead of scientific notation
                ax2.yaxis.set_major_formatter(FuncFormatter(self.elegant_time_formatter))
                ax2.grid(True, which='minor', alpha=0.2, linestyle=':')
                ax2.tick_params(axis='x', labelsize=9)
            
            # Adjust layout with safe margins
            plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.85, wspace=0.3)
            
            # Save with conservative settings
            comparison_filename = self.output_dir / 'comprehensive_performance_comparison.png'
            plt.savefig(comparison_filename, dpi=150, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            print(f"Saved comprehensive comparison plot: {comparison_filename}")
            
        except Exception as e:
            print(f"Warning: Could not create comprehensive comparison plot: {e}")
            print("Continuing with analysis...")
            if 'fig' in locals():
                plt.close(fig)
    
    def create_implementation_specific_plots(self, summary_df, impl_colors, impl_markers, impl_linestyles):
        """Create detailed plots for each implementation showing performance across datasets and configurations."""
        
        implementations = summary_df['Implementation'].unique()
        datasets = self.sort_datasets_by_size(summary_df['Dataset'].unique())
        
        # Define distinct colors for different datasets in implementation-specific plots - bright palette
        dataset_colors = {
            'input100D': '#2E86C1',    # Bright blue
            'input100D2': '#E74C3C',   # Bright red
            '200k_100': '#28B463',     # Bright green
            '400k_100': "#E4971A",     # Bright orange
            '800k_100': "#AA6AC6"      # Bright purple
        }
        
        # Define distinct markers for datasets
        dataset_markers = {
            'input100D': 'o',     # Circle
            'input100D2': 's',    # Square
            '200k_100': '^',      # Triangle up
            '400k_100': 'D',      # Diamond
            '800k_100': 'v'       # Triangle down
        }
        
        # Define distinct line styles for datasets
        dataset_linestyles = {
            'input100D': '-',     # Solid
            'input100D2': '-',    # Solid 
            '200k_100': '-',      # Solid
            '400k_100': '-',      # Solid
            '800k_100': '-'       # Solid
        }
        
        for impl in implementations:
            impl_data = summary_df[summary_df['Implementation'] == impl].copy()
            
            if len(impl_data) == 0:
                continue
            
            # Create figure for this implementation
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'{impl.upper()} Implementation - Detailed Performance Analysis', 
                        fontsize=16, fontweight='bold')
            
            # For MPI+OMP, separate by process count for better visualization
            if impl == 'mpi_omp':
                process_groups = {}
                for _, row in impl_data.iterrows():
                    config = row['Configuration']
                    if 'p' in config:
                        parts = config.split('_')
                        for part in parts:
                            if part.endswith('p'):
                                process_count = int(part[:-1])
                                if process_count not in process_groups:
                                    process_groups[process_count] = []
                                process_groups[process_count].append(row)
                                break
                
                # Plot each process group separately with distinct colors
                process_colors = {2: '#1f77b4', 4: '#ff7f0e'}  # Blue for 2p, Orange for 4p
                process_markers = {2: 'D', 4: 'v'}
                process_linestyles = {2: '-', 4: '-'}  # Both use solid lines
                
                # 1. SPEEDUP ACROSS DATASETS for MPI+OMP
                for process_count, rows in process_groups.items():
                    process_df = pd.DataFrame(rows)
                    for dataset in datasets:
                        dataset_data = process_df[process_df['Dataset'] == dataset]
                        if len(dataset_data) > 0:
                            dataset_data = dataset_data.sort_values('Total_Cores')
                            
                            color = process_colors.get(process_count, '#d62728')
                            marker = dataset_markers.get(dataset, 'o')  # Use dataset-specific markers
                            linestyle = process_linestyles.get(process_count, '-')
                            
                            ax1.plot(dataset_data['Total_Cores'], dataset_data['Speedup'], 
                                    marker=marker, linewidth=2, markersize=8, linestyle=linestyle,
                                    color=color, alpha=0.8, 
                                    label=f'{dataset} ({process_count}p)')
                
                ax1.set_xlabel('Number of Cores', fontsize=12, fontweight='bold')
                ax1.set_ylabel('Speedup', fontsize=12, fontweight='bold')
                ax1.set_title(f'{impl.upper()} - Speedup Across Datasets', fontsize=14, fontweight='bold')
                ax1.legend(fontsize=9, frameon=True, ncol=2)
                ax1.grid(True, alpha=0.3, linestyle=':')
                ax1.set_xscale('log', base=2)
                # Set custom tick labels to show actual numbers
                if len(datasets) > 0:
                    all_cores = []
                    for dataset in datasets:
                        dataset_data = impl_data[impl_data['Dataset'] == dataset]
                        if len(dataset_data) > 0:
                            all_cores.extend(dataset_data['Total_Cores'].tolist())
                    if all_cores:
                        max_cores = max(all_cores)
                        # Use actual core values instead of powers of 2
                        ticks = sorted(list(set(all_cores)))
                        ax1.set_xticks(ticks)
                        ax1.set_xticklabels([str(tick) for tick in ticks])
                ax1.set_ylim(bottom=0)
                
                # 2. EFFICIENCY ACROSS DATASETS for MPI+OMP
                for process_count, rows in process_groups.items():
                    process_df = pd.DataFrame(rows)
                    for dataset in datasets:
                        dataset_data = process_df[process_df['Dataset'] == dataset]
                        if len(dataset_data) > 0:
                            dataset_data = dataset_data.sort_values('Total_Cores')
                            
                            color = process_colors.get(process_count, '#d62728')
                            marker = dataset_markers.get(dataset, 'o')  # Use dataset-specific markers
                            linestyle = process_linestyles.get(process_count, '-')
                            
                            ax2.plot(dataset_data['Total_Cores'], dataset_data['Efficiency'], 
                                    marker=marker, linewidth=2, markersize=8, linestyle=linestyle,
                                    color=color, alpha=0.8, 
                                    label=f'{dataset} ({process_count}p)')
                
                ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, linewidth=2, 
                           label='Perfect Efficiency')
                ax2.set_xlabel('Number of Cores', fontsize=12, fontweight='bold')
                ax2.set_ylabel('Efficiency', fontsize=12, fontweight='bold')
                ax2.set_title(f'{impl.upper()} - Efficiency Across Datasets', fontsize=14, fontweight='bold')
                ax2.legend(fontsize=9, frameon=True, ncol=2, loc='lower left')
                ax2.grid(True, alpha=0.3, linestyle=':')
                ax2.set_xscale('log', base=2)
                # Set custom tick labels to show actual numbers  
                if len(datasets) > 0:
                    all_cores = []
                    for dataset in datasets:
                        dataset_data = impl_data[impl_data['Dataset'] == dataset]
                        if len(dataset_data) > 0:
                            all_cores.extend(dataset_data['Total_Cores'].tolist())
                    if all_cores:
                        max_cores = max(all_cores)
                        # Use actual core values instead of powers of 2
                        ticks = sorted(list(set(all_cores)))
                        ax2.set_xticks(ticks)
                        ax2.set_xticklabels([str(tick) for tick in ticks])
                ax2.set_ylim(bottom=0, top=1.2)
                
                # 3. EXECUTION TIME ACROSS DATASETS for MPI+OMP
                for process_count, rows in process_groups.items():
                    process_df = pd.DataFrame(rows)
                    for dataset in datasets:
                        dataset_data = process_df[process_df['Dataset'] == dataset]
                        if len(dataset_data) > 0:
                            dataset_data = dataset_data.sort_values('Total_Cores')
                            
                            color = process_colors.get(process_count, '#d62728')
                            marker = dataset_markers.get(dataset, 'o')  # Use dataset-specific markers
                            linestyle = process_linestyles.get(process_count, '-')
                            
                            ax3.plot(dataset_data['Total_Cores'], dataset_data['Avg_Time_s'], 
                                    marker=marker, linewidth=2, markersize=8, linestyle=linestyle,
                                    color=color, alpha=0.8, 
                                    label=f'{dataset} ({process_count}p)')
                            
                            # Add error bars
                            ax3.errorbar(dataset_data['Total_Cores'], dataset_data['Avg_Time_s'],
                                       yerr=dataset_data['Std_Time_s'], color=color, alpha=0.3,
                                       capsize=3, capthick=1)
                
                ax3.set_xlabel('Number of Cores', fontsize=12, fontweight='bold')
                ax3.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
                ax3.set_title(f'{impl.upper()} - Execution Time Across Datasets', fontsize=14, fontweight='bold')
                ax3.legend(fontsize=9, frameon=True, ncol=2)
                ax3.grid(True, alpha=0.3, linestyle=':')
                ax3.set_yscale('log')
                # Use elegant formatting instead of scientific notation
                ax3.yaxis.set_major_formatter(FuncFormatter(self.elegant_time_formatter))
                ax3.grid(True, which='minor', alpha=0.2, linestyle=':')
                ax3.set_xscale('log', base=2)
                # Set custom tick labels to show actual numbers
                if len(datasets) > 0:
                    all_cores = []
                    for dataset in datasets:
                        dataset_data = impl_data[impl_data['Dataset'] == dataset]
                        if len(dataset_data) > 0:
                            all_cores.extend(dataset_data['Total_Cores'].tolist())
                    if all_cores:
                        max_cores = max(all_cores)
                        # Use actual core values instead of powers of 2
                        ticks = sorted(list(set(all_cores)))
                        ax3.set_xticks(ticks)
                        ax3.set_xticklabels([str(tick) for tick in ticks])
                
            else:
                # For other implementations, use distinct colors for datasets
                
                # 1. SPEEDUP ACROSS DATASETS
                if impl != 'seq':  # Sequential doesn't have speedup > 1
                    for dataset in datasets:
                        dataset_data = impl_data[impl_data['Dataset'] == dataset]
                        if len(dataset_data) > 0:
                            dataset_data = dataset_data.sort_values('Total_Cores')
                            
                            # Use distinct colors for each dataset
                            color = dataset_colors.get(dataset, '#666666')
                            marker = dataset_markers.get(dataset, 'o')
                            linestyle = dataset_linestyles.get(dataset, '-')
                            
                            if impl == 'cuda':
                                # For CUDA, show as horizontal line
                                cuda_speedup = dataset_data['Speedup'].iloc[0]
                                ax1.axhline(y=cuda_speedup, color=color, linestyle=linestyle, 
                                           linewidth=2, alpha=0.8, label=f'{dataset}')
                            else:
                                ax1.plot(dataset_data['Total_Cores'], dataset_data['Speedup'], 
                                        marker=marker, linewidth=2, markersize=8, linestyle=linestyle,
                                        color=color, alpha=0.8, label=f'{dataset}')
                    
                    ax1.set_xlabel('Number of Cores', fontsize=12, fontweight='bold')
                    ax1.set_ylabel('Speedup', fontsize=12, fontweight='bold')
                    ax1.set_title(f'{impl.upper()} - Speedup Across Datasets', fontsize=14, fontweight='bold')
                    ax1.legend(fontsize=10, frameon=True)
                    ax1.grid(True, alpha=0.3, linestyle=':')
                    if impl != 'cuda':
                        ax1.set_xscale('log', base=2)
                        # Set custom tick labels to show actual numbers
                        all_cores = []
                        for dataset in datasets:
                            dataset_data = impl_data[impl_data['Dataset'] == dataset]
                            if len(dataset_data) > 0:
                                all_cores.extend(dataset_data['Total_Cores'].tolist())
                        if all_cores:
                            max_cores = max(all_cores)
                            # Use actual core values instead of powers of 2
                            ticks = sorted(list(set(all_cores)))
                            ax1.set_xticks(ticks)
                            ax1.set_xticklabels([str(tick) for tick in ticks])
                    ax1.set_ylim(bottom=0)
                else:
                    ax1.text(0.5, 0.5, 'Sequential Implementation\n(Baseline for Speedup)', 
                            ha='center', va='center', transform=ax1.transAxes, 
                            fontsize=14, fontweight='bold')
                    ax1.set_title(f'{impl.upper()} - Baseline Implementation', fontsize=14, fontweight='bold')
                
                # 2. EFFICIENCY ACROSS DATASETS (skip for sequential and CUDA)
                if impl != 'seq' and impl != 'cuda':
                    for dataset in datasets:
                        dataset_data = impl_data[impl_data['Dataset'] == dataset]
                        if len(dataset_data) > 0:
                            dataset_data = dataset_data.sort_values('Total_Cores')
                            
                            color = dataset_colors.get(dataset, '#666666')
                            marker = dataset_markers.get(dataset, 'o')
                            linestyle = dataset_linestyles.get(dataset, '-')
                            
                            ax2.plot(dataset_data['Total_Cores'], dataset_data['Efficiency'], 
                                    marker=marker, linewidth=2, markersize=8, linestyle=linestyle,
                                    color=color, alpha=0.8, label=f'{dataset}')
                    
                    ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, linewidth=2, 
                               label='Perfect Efficiency')
                    ax2.set_xlabel('Number of Cores', fontsize=12, fontweight='bold')
                    ax2.set_ylabel('Efficiency', fontsize=12, fontweight='bold')
                    ax2.set_title(f'{impl.upper()} - Efficiency Across Datasets', fontsize=14, fontweight='bold')
                    ax2.legend(fontsize=10, frameon=True)
                    ax2.grid(True, alpha=0.3, linestyle=':')
                    ax2.set_xscale('log', base=2)
                    # Set custom tick labels to show actual numbers
                    all_cores = []
                    for dataset in datasets:
                        dataset_data = impl_data[impl_data['Dataset'] == dataset]
                        if len(dataset_data) > 0:
                            all_cores.extend(dataset_data['Total_Cores'].tolist())
                    if all_cores:
                        max_cores = max(all_cores)
                        # Use actual core values instead of powers of 2
                        ticks = sorted(list(set(all_cores)))
                        ax2.set_xticks(ticks)
                        ax2.set_xticklabels([str(tick) for tick in ticks])
                    ax2.set_ylim(bottom=0, top=1.2)
                else:
                    if impl == 'cuda':
                        ax2.text(0.5, 0.5, 'CUDA Implementation\n(Efficiency not applicable)', 
                                ha='center', va='center', transform=ax2.transAxes, 
                                fontsize=14, fontweight='bold')
                    else:
                        ax2.text(0.5, 0.5, 'Sequential Implementation\n(Efficiency = 1.0)', 
                                ha='center', va='center', transform=ax2.transAxes, 
                                fontsize=14, fontweight='bold')
                    ax2.set_title(f'{impl.upper()} - Efficiency Analysis', fontsize=14, fontweight='bold')
                
                # 3. EXECUTION TIME ACROSS DATASETS
                for dataset in datasets:
                    dataset_data = impl_data[impl_data['Dataset'] == dataset]
                    if len(dataset_data) > 0:
                        dataset_data = dataset_data.sort_values('Total_Cores')
                        
                        color = dataset_colors.get(dataset, '#666666')
                        marker = dataset_markers.get(dataset, 'o')
                        linestyle = dataset_linestyles.get(dataset, '-')
                        
                        if impl == 'cuda':
                            # For CUDA, show as single point since it doesn't scale with cores
                            ax3.scatter([1], [dataset_data['Avg_Time_s'].iloc[0]], 
                                       s=100, marker=marker, color=color, alpha=0.8, label=f'{dataset}')
                        else:
                            ax3.plot(dataset_data['Total_Cores'], dataset_data['Avg_Time_s'], 
                                    marker=marker, linewidth=2, markersize=8, linestyle=linestyle,
                                    color=color, alpha=0.8, label=f'{dataset}')
                            
                            # Add error bars
                            ax3.errorbar(dataset_data['Total_Cores'], dataset_data['Avg_Time_s'],
                                       yerr=dataset_data['Std_Time_s'], color=color, alpha=0.3,
                                       capsize=3, capthick=1)
                
                ax3.set_xlabel('Number of Cores' if impl != 'cuda' else 'Implementation', fontsize=12, fontweight='bold')
                ax3.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
                ax3.set_title(f'{impl.upper()} - Execution Time Across Datasets', fontsize=14, fontweight='bold')
                ax3.legend(fontsize=10, frameon=True)
                ax3.grid(True, alpha=0.3, linestyle=':')
                ax3.set_yscale('log')
                # Use elegant formatting instead of scientific notation
                ax3.yaxis.set_major_formatter(FuncFormatter(self.elegant_time_formatter))
                ax3.grid(True, which='minor', alpha=0.2, linestyle=':')
                if impl != 'cuda':
                    ax3.set_xscale('log', base=2)
                    # Set custom tick labels to show actual numbers
                    all_cores = []
                    for dataset in datasets:
                        dataset_data = impl_data[impl_data['Dataset'] == dataset]
                        if len(dataset_data) > 0:
                            all_cores.extend(dataset_data['Total_Cores'].tolist())
                    if all_cores:
                        max_cores = max(all_cores)
                        # Use actual core values instead of powers of 2
                        ticks = sorted(list(set(all_cores)))
                        ax3.set_xticks(ticks)
                        ax3.set_xticklabels([str(tick) for tick in ticks])
            
            # 4. CONFIGURATION ANALYSIS (for MPI+OpenMP show process/thread breakdown)
            if impl == 'mpi_omp':
                # Create a performance comparison showing best speedup by configuration across datasets
                configs = impl_data['Configuration'].unique()
                config_performance = {}
                
                for config in configs:
                    if 'p' in config and 't' in config:
                        parts = config.split('_')
                        process_count = None
                        thread_count = None
                        for part in parts:
                            if part.endswith('p'):
                                process_count = int(part[:-1])
                            elif part.endswith('t'):
                                thread_count = int(part[:-1])
                        
                        if process_count and thread_count:
                            config_data = impl_data[impl_data['Configuration'] == config]
                            for dataset in datasets:
                                dataset_config = config_data[config_data['Dataset'] == dataset]
                                if len(dataset_config) > 0:
                                    best_speedup = dataset_config['Speedup'].max()
                                    key = f'{process_count}p_{thread_count}t'
                                    if key not in config_performance:
                                        config_performance[key] = {}
                                    config_performance[key][dataset] = best_speedup
                
                # Plot best speedup by configuration for each dataset
                if config_performance:
                    x_pos = 0
                    bar_width = 0.15
                    
                    for i, dataset in enumerate(datasets):
                        dataset_configs = []
                        dataset_speedups = []
                        
                        for config_key in sorted(config_performance.keys()):
                            if dataset in config_performance[config_key]:
                                dataset_configs.append(config_key)
                                dataset_speedups.append(config_performance[config_key][dataset])
                        
                        if dataset_configs:
                            x_positions = [x_pos + j * bar_width for j in range(len(dataset_configs))]
                            color = dataset_colors.get(dataset, '#666666')
                            bars = ax4.bar(x_positions, dataset_speedups, bar_width, 
                                         label=dataset, color=color, alpha=0.8, edgecolor='black')
                            
                            # Add value labels on bars
                            for bar, speedup in zip(bars, dataset_speedups):
                                height = bar.get_height()
                                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                                       f'{speedup:.1f}x', ha='center', va='bottom', 
                                       fontsize=8, fontweight='bold')
                            
                            x_pos += len(dataset_configs) * bar_width + 0.1
                    
                    # Set x-axis labels
                    all_configs = sorted(set(config for configs in config_performance.values() for config in configs))
                    ax4.set_xlabel('MPI+OpenMP Configuration', fontsize=12, fontweight='bold')
                    ax4.set_ylabel('Best Speedup Achieved', fontsize=12, fontweight='bold')
                    ax4.set_title(f'{impl.upper()} - Best Performance by Configuration', fontsize=14, fontweight='bold')
                    ax4.legend(fontsize=9, frameon=True)
                    ax4.grid(True, alpha=0.3, linestyle=':', axis='y')
                    
                    # Create custom x-tick labels showing configurations
                    config_labels = []
                    config_positions = []
                    current_pos = 0
                    for i, dataset in enumerate(datasets):
                        dataset_configs = []
                        for config_key in sorted(config_performance.keys()):
                            if dataset in config_performance[config_key]:
                                dataset_configs.append(config_key)
                        
                        if dataset_configs:
                            mid_pos = current_pos + (len(dataset_configs) * bar_width) / 2
                            config_positions.extend([current_pos + j * bar_width for j in range(len(dataset_configs))])
                            config_labels.extend(dataset_configs)
                            current_pos += len(dataset_configs) * bar_width + 0.1
                    
                    if config_labels:
                        ax4.set_xticks(config_positions)
                        ax4.set_xticklabels(config_labels, rotation=90, ha='center', fontsize=8)
            
            elif impl in ['omp', 'mpi']:
                # Show scaling efficiency with distinct colors for datasets
                scaling_data = []
                for dataset in datasets:
                    dataset_data = impl_data[impl_data['Dataset'] == dataset]
                    if len(dataset_data) > 1:
                        dataset_data = dataset_data.sort_values('Total_Cores')
                        cores = dataset_data['Total_Cores'].values
                        speedups = dataset_data['Speedup'].values
                        
                        # Calculate scaling slope (log-log)
                        log_cores = np.log2(cores)
                        log_speedup = np.log2(speedups)
                        if len(log_cores) > 1:
                            slope = np.polyfit(log_cores, log_speedup, 1)[0]
                            scaling_data.append({'Dataset': dataset, 'Scaling_Slope': slope})
                
                if scaling_data:
                    scaling_df = pd.DataFrame(scaling_data)
                    # Use distinct colors for each dataset
                    colors = [dataset_colors.get(dataset, '#666666') for dataset in scaling_df['Dataset']]
                    bars = ax4.bar(scaling_df['Dataset'], scaling_df['Scaling_Slope'], 
                                  color=colors, alpha=0.8, edgecolor='black')
                    
                    # Add value labels
                    for bar, slope in zip(bars, scaling_df['Scaling_Slope']):
                        height = bar.get_height()
                        quality = "Excellent" if slope > 0.9 else "Good" if slope > 0.7 else "Moderate" if slope > 0.5 else "Poor"
                        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                               f'{slope:.3f}\n({quality})', ha='center', va='bottom', 
                               fontsize=9, fontweight='bold')
                    
                    ax4.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Perfect Scaling')
                    ax4.set_ylabel('Scaling Slope (log-log)', fontsize=12, fontweight='bold')
                    ax4.set_title(f'{impl.upper()} - Scaling Quality by Dataset', fontsize=14, fontweight='bold')
                    ax4.legend()
                    ax4.grid(True, alpha=0.3, linestyle=':', axis='y')
            
            else:
                # For sequential and CUDA, show dataset complexity comparison with distinct colors
                dataset_times = []
                for dataset in datasets:
                    dataset_data = impl_data[impl_data['Dataset'] == dataset]
                    if len(dataset_data) > 0:
                        avg_time = dataset_data['Avg_Time_s'].mean()
                        dataset_times.append({'Dataset': dataset, 'Avg_Time': avg_time})
                
                if dataset_times:
                    time_df = pd.DataFrame(dataset_times)
                    # Use distinct colors for each dataset
                    colors = [dataset_colors.get(dataset, '#666666') for dataset in time_df['Dataset']]
                    bars = ax4.bar(time_df['Dataset'], time_df['Avg_Time'], 
                                  color=colors, alpha=0.8, edgecolor='black')
                    
                    # Add value labels
                    for bar, time in zip(bars, time_df['Avg_Time']):
                        height = bar.get_height()
                        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                               f'{time:.2f}s', ha='center', va='bottom', 
                               fontsize=10, fontweight='bold')
                    
                    ax4.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
                    ax4.set_title(f'{impl.upper()} - Dataset Complexity', fontsize=14, fontweight='bold')
                    ax4.set_yscale('log')
                    # Use elegant formatting instead of scientific notation
                    ax4.yaxis.set_major_formatter(FuncFormatter(self.elegant_time_formatter))
                    ax4.grid(True, which='minor', alpha=0.2, linestyle=':')
                    ax4.grid(True, alpha=0.3, linestyle=':', axis='y')
            
            # Adjust layout
            plt.subplots_adjust(left=0.08, bottom=0.15, right=0.95, top=0.92, wspace=0.25, hspace=0.35)
            
            # Save the implementation-specific plot
            impl_plot_filename = self.output_dir / f'implementation_analysis_{impl}.png'
            plt.savefig(impl_plot_filename, dpi=200, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            print(f"Saved {impl} implementation analysis: {impl_plot_filename}")
    
    def analyze_scaling_behavior(self, summary_df):
        """Analyze and report scaling behavior patterns."""
        scaling_report = self.output_dir / "scaling_analysis_report.txt"
        
        with open(scaling_report, 'w') as f:
            f.write("K-MEANS SCALING BEHAVIOR ANALYSIS\n")
            f.write("=" * 50 + "\n\n")
            
            for impl in summary_df['Implementation'].unique():
                if impl == 'seq':
                    continue
                    
                f.write(f"{impl.upper()} IMPLEMENTATION ANALYSIS:\n")
                f.write("-" * 30 + "\n")
                
                impl_data = summary_df[summary_df['Implementation'] == impl]
                # Sort datasets by size
                datasets = self.sort_datasets_by_size(impl_data['Dataset'].unique())
                
                for dataset in datasets:
                    dataset_data = impl_data[impl_data['Dataset'] == dataset].sort_values('Total_Cores')
                    
                    if len(dataset_data) < 2:
                        continue
                    
                    f.write(f"\nDataset: {dataset}\n")
                    
                    # Calculate scaling metrics
                    cores = dataset_data['Total_Cores'].values
                    speedups = dataset_data['Speedup'].values
                    efficiencies = dataset_data['Efficiency'].values
                    
                    # Find optimal core count (best efficiency)
                    best_eff_idx = np.argmax(efficiencies)
                    optimal_cores = cores[best_eff_idx]
                    optimal_efficiency = efficiencies[best_eff_idx]
                    
                    # Calculate scaling slope (log-log)
                    if len(cores) > 1:
                        log_cores = np.log2(cores)
                        log_speedup = np.log2(speedups)
                        slope = np.polyfit(log_cores, log_speedup, 1)[0]
                        
                        f.write(f"  Scaling slope (log-log): {slope:.3f}\n")
                        f.write(f"  Optimal cores: {optimal_cores} (efficiency: {optimal_efficiency:.3f})\n")
                        
                        # Classify scaling behavior
                        if slope > 0.9:
                            scaling_type = "Excellent (near-linear)"
                        elif slope > 0.7:
                            scaling_type = "Good"
                        elif slope > 0.5:
                            scaling_type = "Moderate"
                        else:
                            scaling_type = "Poor"
                        
                        f.write(f"  Scaling quality: {scaling_type}\n")
                        
                        # Efficiency degradation analysis
                        if len(efficiencies) > 1:
                            eff_degradation = (efficiencies[0] - efficiencies[-1]) / efficiencies[0] * 100
                            f.write(f"  Efficiency degradation: {eff_degradation:.1f}%\n")
                    
                    f.write("\n")
                
                f.write("\n")
        
        print(f"Saved scaling analysis: {scaling_report}")
    
    def create_performance_summary_table(self, summary_df):
        """Create a comprehensive performance summary table."""
        # Create summary statistics
        summary_stats = []
        
        for impl in summary_df['Implementation'].unique():
            impl_data = summary_df[summary_df['Implementation'] == impl]
            # Sort datasets by size
            datasets = self.sort_datasets_by_size(impl_data['Dataset'].unique())
            
            for dataset in datasets:
                dataset_data = impl_data[impl_data['Dataset'] == dataset]
                
                if impl == 'seq':
                    summary_stats.append({
                        'Implementation': impl.upper(),
                        'Dataset': dataset,
                        'Best_Time_s': dataset_data['Avg_Time_s'].iloc[0],
                        'Best_Speedup': 1.0,
                        'Best_Efficiency': 1.0,
                        'Optimal_Cores': 1,
                        'Worst_Time_s': dataset_data['Avg_Time_s'].iloc[0],
                        'Avg_Efficiency': 1.0
                    })
                else:
                    best_speedup_idx = dataset_data['Speedup'].idxmax()
                    best_eff_idx = dataset_data['Efficiency'].idxmax()
                    
                    summary_stats.append({
                        'Implementation': impl.upper(),
                        'Dataset': dataset,
                        'Best_Time_s': dataset_data['Avg_Time_s'].min(),
                        'Best_Speedup': dataset_data['Speedup'].max(),
                        'Best_Efficiency': dataset_data['Efficiency'].max(),
                        'Optimal_Cores': dataset_data.loc[best_eff_idx, 'Total_Cores'],
                        'Worst_Time_s': dataset_data['Avg_Time_s'].max(),
                        'Avg_Efficiency': dataset_data['Efficiency'].mean()
                    })
        
        summary_table = pd.DataFrame(summary_stats)
        
        # Sort by implementation and dataset size order
        summary_table['Dataset_Order'] = summary_table['Dataset'].map(
            lambda x: self.dataset_size_order.get(x.replace('.inp', ''), 999)
        )
        summary_table = summary_table.sort_values(['Implementation', 'Dataset_Order'])
        summary_table = summary_table.drop('Dataset_Order', axis=1)  # Remove helper column
        
        # Round numerical columns
        numerical_cols = ['Best_Time_s', 'Best_Speedup', 'Best_Efficiency', 'Worst_Time_s', 'Avg_Efficiency']
        for col in numerical_cols:
            summary_table[col] = summary_table[col].round(3)
        
        # Save summary table
        summary_filename = self.output_dir / "performance_summary_table.csv"
        summary_table.to_csv(summary_filename, index=False)
        
        # Create formatted version
        summary_formatted = self.output_dir / "performance_summary_formatted.txt"
        with open(summary_formatted, 'w') as f:
            f.write("K-MEANS PERFORMANCE SUMMARY TABLE\n")
            f.write("=" * 60 + "\n\n")
            f.write(summary_table.to_string(index=False))
            f.write("\n\nColumn Descriptions:\n")
            f.write("- Best_Time_s: Fastest execution time achieved\n")
            f.write("- Best_Speedup: Maximum speedup vs sequential\n")
            f.write("- Best_Efficiency: Maximum parallel efficiency\n")
            f.write("- Optimal_Cores: Core count achieving best efficiency\n")
            f.write("- Avg_Efficiency: Average efficiency across all core counts\n")
        
        print(f"Saved performance summary: {summary_filename}")
        print(f"Saved formatted summary: {summary_formatted}")
        
        return summary_table
    
    def create_comprehensive_comparison(self, summary_df):
        """Create a comprehensive comparison table across all implementations."""
        # Get datasets and sort by size
        datasets = self.sort_datasets_by_size(summary_df['Dataset'].unique())
        
        # Pivot table showing speedup for each implementation
        pivot_df = summary_df.pivot_table(
            index=['Dataset', 'Total_Cores'], 
            columns='Implementation', 
            values='Speedup',
            aggfunc='mean'
        ).round(2)
        
        # Reorder the index to match dataset size order
        pivot_df = pivot_df.reindex([
            (dataset, cores) for dataset in datasets 
            for cores in sorted(pivot_df.index.get_level_values(1).unique())
            if (dataset, cores) in pivot_df.index
        ])
        
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
        
        # Reorder the index to match dataset size order
        efficiency_pivot = efficiency_pivot.reindex([
            (dataset, cores) for dataset in datasets 
            for cores in sorted(efficiency_pivot.index.get_level_values(1).unique())
            if (dataset, cores) in efficiency_pivot.index
        ])
        
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
            # Sort datasets by size for consistent reporting
            datasets = self.sort_datasets_by_size(summary_df['Dataset'].unique())
            for dataset in datasets:
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
        
        # Step 7: Analyze scaling behavior
        print("\n7. Analyzing scaling behavior...")
        self.analyze_scaling_behavior(summary_df)
        
        # Step 8: Create performance summary table
        print("\n8. Creating performance summary table...")
        self.create_performance_summary_table(summary_df)
        
        # Step 9: Generate summary report
        print("\n9. Generating summary report...")
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
