#!/bin/bash
# Quick local build test for macOS/Linux compatibility

echo "============================================"
echo "K-means Build Compatibility Test"
echo "============================================"

# Show system info
echo "System: $(uname -s)"
echo "Hostname: $(hostname)"
echo ""

# Test compiler detection
echo "Testing Makefile compiler detection..."
make compiler-info
echo ""

# Test basic build
echo "Testing basic sequential build..."
make clean-build
if make KMEANS_seq; then
    echo "✅ Sequential build successful"
else
    echo "❌ Sequential build failed"
    exit 1
fi

echo ""
echo "Testing OpenMP build..."
if make KMEANS_omp; then
    echo "✅ OpenMP build successful"
else
    echo "❌ OpenMP build failed - this is expected if OpenMP is not available"
fi

echo ""
echo "Testing MPI build..."
if make KMEANS_mpi; then
    echo "✅ MPI build successful"
else
    echo "❌ MPI build failed - this is expected if MPI is not available"
fi

echo ""
echo "Testing MPI+OpenMP hybrid build..."
if make KMEANS_mpi_omp; then
    echo "✅ MPI+OpenMP hybrid build successful"
else
    echo "❌ MPI+OpenMP hybrid build failed - this is expected if MPI or OpenMP is not available"
fi

echo ""
echo "============================================"
echo "Build test complete!"
echo ""
echo "Available executables:"
ls -la build/ 2>/dev/null || echo "No build directory found"
echo "============================================"
