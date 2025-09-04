#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <float.h>
#include <cuda.h>
#include <sys/time.h>

// ================================================================================================
// CUDA ERROR CHECKING UTILITIES
// ================================================================================================

#define CHECK_CUDA_CALL(a)                                                                      \
  {                                                                                             \
    cudaError_t ok = a;                                                                         \
    if (ok != cudaSuccess)                                                                      \
      fprintf(stderr, "-- Error CUDA call in line %d: %s\n", __LINE__, cudaGetErrorString(ok)); \
  }

// ================================================================================================
// WARP-LEVEL REDUCTION PRIMITIVES
// ================================================================================================

/**
 * Ultra-fast warp-level reduction using register-level communication
 * Reduces 32 values to 1 using shuffle operations - bypasses shared memory entirely
 * Time complexity: O(log₂(32)) = 5 operations per warp
 */
__device__ __forceinline__ float warp_reduce(float val)
{
  // Participation mask: all 32 threads in warp participate (0xffffffff = 32 bits set)
  const unsigned int FULL_MASK = 0xffffffff;

  // Binary tree reduction: 32→16→8→4→2→1
#pragma unroll
  for (unsigned int i = 16; i > 0; i /= 2)
  {
    // Exchange values directly through registers (fastest possible communication)
    val = max(val, __shfl_down_sync(FULL_MASK, val, i));
  }
  return val;
}

// ================================================================================================
// BLOCK-LEVEL REDUCTION KERNEL
// ================================================================================================

/**
 * High-performance parallel reduction kernel
 *
 * REQUIREMENTS:
 * - input_size must be padded to next power of 2
 * - Pad with FLT_MIN for MAX operation, 0.0f for SUM operation
 *
 * ALGORITHM:
 * 1. Each thread loads data and performs initial reduction within warps
 * 2. Warp leaders store partial results in shared memory
 * 3. Final warp reduces partial results to single value per block
 *
 * PERFORMANCE:
 * - Time complexity: O(log₂(input_size))
 * - Uses register-level communication (fastest possible)
 * - Minimal shared memory usage (32 elements max)
 */
__global__ void reduce(float *inputs, unsigned int input_size, float *outputs)
{
  // Initialize with neutral element for MAX operation
  float sum = FLT_MIN;

  // ===== PHASE 1: DATA LOADING WITH GRID-STRIDE LOOP =====
  // Handle cases where we have fewer threads than data elements
  for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < input_size;
       i += blockDim.x * gridDim.x)
    sum = max(sum, inputs[i]);

  // ===== PHASE 2: SHARED MEMORY SETUP =====
  // One slot per warp (max 32 warps per block = 1024/32)
  __shared__ float shared[32];

  // Thread positioning within block structure
  unsigned int lane = threadIdx.x % warpSize; // Position within warp (0-31)
  unsigned int wid = threadIdx.x / warpSize;  // Warp ID within block

  // ===== PHASE 3: INTRA-WARP REDUCTION =====
  // Each warp reduces its 32 values to 1
  sum = warp_reduce(sum);

  // Only thread 0 of each warp writes to shared memory
  if (lane == 0)
    shared[wid] = sum;

  // Wait for all warps to complete their reductions
  __syncthreads();

  // ===== PHASE 4: INTER-WARP REDUCTION =====
  // Create virtual warp from warp leaders for final reduction
  sum = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

  // Only warp 0 performs the final reduction
  if (wid == 0)
    sum = warp_reduce(sum);

  // ===== PHASE 5: OUTPUT =====
  // Thread 0 of block writes final result
  if (threadIdx.x == 0)
    outputs[blockIdx.x] = sum;
}

// ================================================================================================
// TEST MAIN FUNCTION
// ================================================================================================

int main(int argc, char **argv)
{
  // ===== TEST DATA SETUP =====
  // Create test array of 64 elements (power of 2 requirement)
  float h_array[64] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

  float *d_array;
  float *d_res;
  float h_res = 0.0f;

  // ===== CPU REFERENCE CALCULATION =====
  float cpu_result = 0.0f;
  for (unsigned int i = 0; i < 64; ++i)
  {
    cpu_result = max(cpu_result, h_array[i]); // Should be 1.0 for this test
  }
  printf("CPU result: %.6f\n", cpu_result);

  // ===== GPU MEMORY ALLOCATION =====
  CHECK_CUDA_CALL(cudaMalloc(&d_array, 64 * sizeof(float)));
  CHECK_CUDA_CALL(cudaMalloc(&d_res, sizeof(float)));
  CHECK_CUDA_CALL(cudaMemcpy(d_array, h_array, 64 * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA_CALL(cudaMemset(d_res, 0, sizeof(float)));

  // ===== GPU REDUCTION EXECUTION =====
  /*
   * MULTI-BLOCK REDUCTION PATTERN:
   * When using multiple blocks, reduction requires TWO kernel calls:
   *
   * Call 1: reduce<<<numBlocks, blockSize>>>(input, inputSize, partialResults)
   *         - Produces one result per block
   *
   * Call 2: reduce<<<1, min(numBlocks, 32)>>>(partialResults, numBlocks, finalResult)
   *         - Combines partial results into final answer
   *         - Use single block with up to 32 threads (one warp)
   *
   * For this 64-element example:
   * - Option 1: Single block of 64 threads (direct result)
   * - Option 2: Two blocks of 32 threads each (requires two calls)
   */

  // Single block reduction (direct approach for small data)
  reduce<<<1, 64>>>(d_array, 64, d_res);

  // Alternative multi-block approach (uncomment to test):
  // reduce<<<2, 32>>>(d_array, 64, d_res);        // 2 blocks → 2 partial results
  // reduce<<<1, 2>>>(d_res, 2, d_res);            // 1 block → 1 final result

  CHECK_CUDA_CALL(cudaDeviceSynchronize());

  // ===== RESULT VERIFICATION =====
  CHECK_CUDA_CALL(cudaMemcpy(&h_res, d_res, sizeof(float), cudaMemcpyDeviceToHost));
  printf("GPU result: %.6f\n", h_res);

  // Verify correctness
  if (abs(h_res - cpu_result) < 1e-6)
  {
    printf("✓ Test PASSED: Results match\n");
  }
  else
  {
    printf("✗ Test FAILED: Results differ\n");
  }

  // ===== CLEANUP =====
  CHECK_CUDA_CALL(cudaFree(d_array));
  CHECK_CUDA_CALL(cudaFree(d_res));

  return EXIT_SUCCESS;
}
