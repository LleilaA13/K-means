#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <float.h>
#include <cuda.h>
#include <sys/time.h>


#define CHECK_CUDA_CALL(a)                                                                            \
	{                                                                                                 \
		cudaError_t ok = a;                                                                           \
		if (ok != cudaSuccess)                                                                        \
			fprintf(stderr, "-- Error CUDA call in line %d: %s\n", __LINE__, cudaGetErrorString(ok)); \
	}


__device__ __forceinline__ float warp_reduce(float val)
{
	FULL_MASK = 0xffffffff;
# pragma unroll
	for (unsigned int i = 16; i > 0; i /= 2)
	{
		val = max(val, __shfl_down_sync(FULL_MASK, val, i));
	}
	return val;
}

__global__ void reduce(float* inputs, unsigned int input_size, float* outputs)
{
	/* Eccoci qui all'interno della reduce più veloce del west. Questa implementazione è presa da questo blog:
	 * https://ashvardanian.com/posts/cuda-parallel-reductions/
	 * Praticamente questa implementazione sfrutta delle operazioni che vengono eseguite a livello dei warp. Se vi ricordate, in cuda
	 * i warp sono il più basso livello logico in cui le istruzioni vengono eseguite.
	 * ATTENZIONE: per fare si che questo algoritmo funzioni, input_size DEVE essere una potenza di 2, quindi dovete paddare il vostro array finché non ha
	 * la grandezza desiderata. Questo non influisce sulla correttezza del vostro algoritmo, vi dovete solo ricordare di paddare con un valore neutro per
	 * la vostra operazione (nel caso del MAX il valore è -FLT_MAX oppure semplicemente FLT_MIN)
	 */
    float sum = FLT_MIN;
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
            i < input_size;
            i += blockDim.x * gridDim.x)
        sum = max(sum, inputs[i]); // Questo for serve in caso non abbiate abbastanza thread per parallelizzare, e quindi ogni thread deve gestire più elementi. Per fortuna non è il vostro caso, quindi questo for in realtà di riduce semplicemente a sum += inputs[i] (fate la prova togliendolo per vedere che effettivamente l'algoritmo funziona lo stesso)

    __shared__ float shared[32]; // Qui la shared mem è fissa a 32 perché in un blocco possono esserci al massimo 1024 thread, e siccome la riduzione è a livello warp (32 thread) ogni blocco potrà eseguire al massimo 32 riduzioni

    // queste sono variabili che servono per l'algoritmo. Bonus tips se capite a cosa servono ;)
    unsigned int lane = threadIdx.x % warpSize;
    unsigned int wid = threadIdx.x / warpSize;

    sum = warp_reduce(sum); // prima chiamata della riduzione
    if (lane == 0)
        shared[wid] = sum; // se sono il thread 0 all'interno del warp scrivo in shared memory

    // Wait for all partial reductions
    __syncthreads(); // synchtreads() d'obbligo

    sum = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0; // bonus tip se capisci a cosa serve questa ;)
    if (wid == 0)
        sum = warp_reduce(sum); // Seconda chiamata alla reduce per fare in modo che la riduzione venga fatta all'interno del blocco (ricorda che warp_reduce riduce all'interno di ogni warp)

    if (threadIdx.x == 0)
        outputs[blockIdx.x] = sum; // Il thread 0 di ogni blocco scrive in global memory il risultato della riduzione di questo blocco
}


int main(int argc, char** argv)
{
  // Qui il main per testare che effettivamente funzioni! leggi i commenti bene per capire come impostare questa operazione
  // ATTENZIONE: è cruciale che sia chiaro con quanti blocchi/thread chiamare questa operazione, se sbagli quello non funziona niente. Consiglio
  // di leggere il pdf che ho linkato nel file KMEANS_cuda.cu per capire bene la logica dietro questa implementazione
  float h_array = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                   1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                   1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                   1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                   1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                   1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                   1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                   1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  float* d_array;
  float* d_res;
  float res = 0.0f;
  float*
  for (unsigned int i = 0; i < 64; ++i)
  {
    res += h_array[i];
  }
  std::cout << "cpu result: " << res << std::endl;

  CHECK_CUDA_CALL(cudaMalloc(&d_array, 64 * sizeof(float)))
  CHECK_CUDA_CALL(cudaMalloc(&d_res, sizeof(float)))
  CHECK_CUDA_CALL(cudaMemcpy(d_array, &array, 64 * sizeof(float), cudaMemcpyHostToDevice))
  CHECK_CUDA_CALL(cudaMemset(d_res, 0.0f, sizeof(float)))

  /*
   * ATTENZIONE: Nel caso vi trovaste nella necessità di lanciare più di un blocco, vi dovete ricordare di effettuare l'operazione DUE VOLTE,
   * perché il primo risultato è una riduzione parziale. Quindi, se lanciate una griglia di 4 blocchi, d_res deve essere un array di 4 elementi
   * e la chiamata sarà una roba del genere:
   * reduce<<<4, 64>>>(d_array, 64, d_res);
   * reduce<<<1, 32>>>(d_res, 4, d_res);
   * fate attenzione che la seconda chiamata è un solo blocco di grandezza dimGrid SE E SOLO SE dimGrid è una potenza di 2, altrimenti chiamatela con
   * un blocco di grandezza 32 che è la dimensione di un warp!
   */
  reduce<<<riempi qui ;), riempi qui ;)>>>(d_array, 64, d_res);
  reduce<<<riempi qui ;), riempi qui ;)>>>(d_res, che valore ci metto qui?, d_res);

  CHECK_CUDA_CALL(cudaDeviceSynchronize())

  float* h_res;

  CHECK_CUDA_CALL(cudaMemcpy(h_res, d_res, sizeof(float), cudaMemcyDeviceToHost))

  std::cout << "gpu result: " << h_res << std::endl;

  return EXIT_SUCCESS;
}
