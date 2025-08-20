#!/bin/bash
# Quick SLURM job status checker

echo "=== Your Current Jobs ==="
squeue -u $USER -o "%.10i %.15P %.20j %.8T %.10M %.20R"

echo ""
echo "=== Job Summary ==="
running=$(squeue -u $USER -t R 2>/dev/null | tail -n +2 | wc -l)
pending=$(squeue -u $USER -t PD 2>/dev/null | tail -n +2 | wc -l)
total=$(squeue -u $USER 2>/dev/null | tail -n +2 | wc -l)

echo "Running: $running, Pending: $pending, Total: $total"

if [ $total -eq 0 ]; then
    echo "No active jobs found."
    exit 0
fi

echo ""
echo "=== Recent Job Output ==="
# Find most recent SLURM output file
recent_log=$(ls -t logs/slurm_*.out 2>/dev/null | head -1)
if [ ! -z "$recent_log" ]; then
    echo "Latest log: $recent_log"
    echo "Last 5 lines:"
    tail -5 "$recent_log"
else
    echo "No job output files found in logs/"
fi

echo ""
echo "=== Quick Commands ==="
echo "Follow job output: tail -f logs/slurm_*.out"
echo "Watch queue:      watch -n 30 squeue -u \$USER"
echo "Cancel job:       scancel <job_id>"
