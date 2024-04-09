#include "stdint.h"

#include "data.h"
#include "memref.h"
#include "snax_rt.h"

/*
 * These libraries are included from github.com/KULeuven-MICAS/snitch_cluster
 * Interested users, might want to look at:
 *
 * /sw/snRuntime/api
 * /target/snitch_cluster/sw/runtime/rtl/src
 * /target/snitch_cluster/sw/runtime/common
 * */
#include <snrt.h>

/* These libraries are included from github.com/KULeuven-MICAS/snitch_cluster
 * Interested users, might want to look at:
 *
 * /target/snitch_cluster/sw/snax/gemm/include"
 * /target/snitch_cluster/sw/snax/mac/include"
 *
 * */
#include "snax-gemm-lib.h"
#include "snax-gemm-params.h"

uint8_t Batch = 1;
// meshRow, tileSize and meshCol are defined in snax-gemm-params.h
uint8_t M_param = M_size / meshRow;
uint8_t K_param = K_size / tileSize;
uint8_t N_param = N_size / meshCol;
// Extracted from datagen.py in snitch_cluster repo
uint32_t strideInnermostA = 256;
uint32_t strideInnermostB = 256;
uint32_t strideInnermostC = 256;
uint32_t ldA = 512;
uint32_t ldB = 512;
uint32_t ldC = 512;
uint32_t strideA = 0;
uint32_t strideB = 0;
uint32_t strideC = 0;

// Kernel provided via external definition
void _mlir_ciface_simple_matmul(TwoDMemrefI8_t *a, TwoDMemrefI8_t *b,
                                TwoDMemrefI32_t *c);

void _mlir_ciface_snax_qgemm(TwoDMemrefI8_t *a, TwoDMemrefI8_t *b, int32_t zpa,
                             int32_t zpb, TwoDMemrefI32_t *c) {

  int8_t *a_ptr = a->aligned_data;
  int8_t *b_ptr = b->aligned_data;
  int32_t *c_ptr = c->aligned_data;
  printf("Executing snax_qgemm with a=%p, b=%p, c=%p \n", a_ptr, b_ptr, c_ptr);

  uint32_t size_setting = gen_size_config(Batch, M_param, K_param, N_param);

  set_batch_gemm(size_setting, a_ptr, b_ptr, 0, c_ptr, strideInnermostA,
                 strideInnermostB, strideInnermostC, ldA, ldB, ldC, strideA,
                 strideB, strideC);

  start_batch_gemm();

  wait_batch_gemm();
}

void cCodeEquivalent(TwoDMemrefI8_t *x, TwoDMemrefI8_t *y, TwoDMemrefI32_t *z);

int main() {

  printf("PAMPLEMOUSSE VOLCANO: Setting up the data.\n");

  // Create memref objects for data stored in L3
  TwoDMemrefI8_t memrefA;
  memrefA.data = (int8_t *)&A;
  memrefA.aligned_data = memrefA.data;
  memrefA.offset = 0;

  TwoDMemrefI8_t memrefB;
  memrefB.data = (int8_t *)&B;
  memrefB.aligned_data = memrefB.data;
  memrefB.offset = 0;

  TwoDMemrefI32_t memrefC;
  memrefC.data = (int32_t *)&C;
  memrefC.aligned_data = memrefC.data;
  memrefC.offset = 0;

  printf("before linalg call, A's first elt is %d\n",memrefA.aligned_data[0]);

  (void)snrt_mcycle();

  printf("PAMPLEMOUSSE VOLCANO: calling _mlir_ciface_simple_matmu with\n "
         "&memrefA=%p, &memrefB=%p, &memrefC=%p \n",
         &memrefA, &memrefB, &memrefC);

  _mlir_ciface_simple_matmul(&memrefA, &memrefB, &memrefC);

  snrt_cluster_hw_barrier();

  (void)snrt_mcycle();

  // Correctness check -
  // from this point on only core 0 is required to be alive.
  int thiscore = snrt_cluster_core_idx();
  if (thiscore != 0)
    return 0;

  int nerr = 0;
  for (int i = 0; i < M_size * N_size; i++) {
    int32_t error = memrefC.aligned_data[i] - C_golden[i];
    if (error != 0)
      nerr += 1;
  }

  if (nerr != 0) {
    printf("C does not match the golden value!\n");
    return nerr;
  }
  printf("after linalg call, A's first elt is %d\n",memrefA.aligned_data[0]);
  // second correctness check - is the c code really equivalent??
  TwoDMemrefI32_t z;
  z.data = (int32_t *)&C;
  z.aligned_data = z.data;
  z.offset = 0;
  cCodeEquivalent(&memrefA, &memrefB, &z);
  nerr = 0;
  for (int i = 0; i < M_size * N_size; i++) {
    int32_t error = z.aligned_data[i] - C_golden[i];
    if (error != 0)
      nerr += 1;
  }
  if (nerr != 0) {
    printf("Z does not match the golden value!\n");
  }
  return nerr;
}

// void printTwoDMemrefI8_t(TwoDMemrefI8_t *x);
// void printTwoDMemrefI32_t(TwoDMemrefI8_t *y);

void cCodeEquivalent(TwoDMemrefI8_t *x, TwoDMemrefI8_t *y, TwoDMemrefI32_t *z) {
  printf("M_size is %d and N_size is %d\n",M_size, N_size);
  for (int i = 0; i < M_size * N_size; i++) {
    z->aligned_data[i] = x->aligned_data[i] * y->aligned_data[i];
   // z->aligned_data[i] = C_golden[i];
  }
}

/*
#define N_size 16
#define K_size 16
#define M_size 16

extern const int8_t A[256];
extern const int8_t B[256];
extern const int32_t C_golden[256];
extern const int32_t C[256];
uint8_t Batch = 1;
// meshRow, tileSize and meshCol are defined in snax-gemm-params.h
uint8_t M_param = M_size / meshRow;
uint8_t K_param = K_size / tileSize;
uint8_t N_param = N_size / meshCol;

// Extracted from datagen.py in snitch_cluster repo
uint32_t strideInnermostA = 256;
uint32_t strideInnermostB = 256;
uint32_t strideInnermostC = 256;
uint32_t ldA = 512;
uint32_t ldB = 512;
uint32_t ldC = 512;
uint32_t strideA = 0;
uint32_t strideB = 0;
uint32_t strideC = 0;

#define tileSize 8
#define meshRow 8
#define meshCol 8

struct TwoDMemrefI32 {
  int32_t *data; // allocated pointer: Pointer to data buffer as allocated,
                 // only used for deallocating the memref
  int32_t *aligned_data; // aligned pointer: Pointer to properly aligned data
                         // that memref indexes
  uint32_t offset;
  uint32_t shape[2];
  uint32_t stride[2];
};
*/

// void printTwoDMemrefI8_t(TwoDMemrefI8_t *x){
//   // we assume 16 x 16 shape right now
//   for(int i = 0; i < x->shape[0]; i++){
//     for(int j = 0; j < x->shape[1]; j++){
//       print()

//     }
//   }
//   printf(x->aligned_data[])

// }
// void printTwoDMemrefI32_t(TwoDMemrefI8_t *y){
//   // we assume 16 x 16 shape right now
// }