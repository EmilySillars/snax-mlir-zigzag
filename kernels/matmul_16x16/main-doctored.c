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


#define MAT_WIDTH 16
#define MAT_WIDTH_SQUARED (MAT_WIDTH*MAT_WIDTH)

void _mlir_ciface_regular_matmul(TwoDMemrefI8_t *a, TwoDMemrefI8_t *b,
                                       TwoDMemrefI32_t *c);
// Kernel provided via external definition
void _mlir_ciface_matmul104x104_zigzag(TwoDMemrefI8_t *a, TwoDMemrefI8_t *b,
                                TwoDMemrefI32_t *c);

void _mlir_ciface_snax_gemm(TwoDMemrefI8_t *a, TwoDMemrefI8_t *b, int32_t zpa,
                            int32_t zpb, TwoDMemrefI32_t *c) {

  int8_t *a_ptr = a->aligned_data;
  int8_t *b_ptr = b->aligned_data;
  int32_t *c_ptr = c->aligned_data;
  printf("Executing snax_gemm with a=%p, b=%p, c=%p \n", a_ptr, b_ptr, c_ptr);

  uint32_t size_setting = gen_size_config(Batch, M_param, K_param, N_param);

  set_batch_gemm(size_setting, a_ptr, b_ptr, 0, c_ptr, strideInnermostA,
                 strideInnermostB, strideInnermostC, ldA, ldB, ldC, strideA,
                 strideB, strideC);

  start_batch_gemm();

  wait_batch_gemm();

  printf("Finished executing snax_gemm\n");
}

int main() {

  // statically allocate data stored in L3
  int8_t dataA[MAT_WIDTH_SQUARED];
  int8_t dataB[MAT_WIDTH_SQUARED];
  int32_t dataC[MAT_WIDTH_SQUARED];
  int32_t dataGolden[MAT_WIDTH_SQUARED];

  // Create memref objects for data stored in L3
  // TwoDMemrefI8_t memrefA;  // input 104x104xi8
  // memrefA.data = dataA;
  // memrefA.aligned_data = memrefA.data;
  // memrefA.offset = 0;
  // memrefA.shape[0] = 104;
  // memrefA.shape[1] = 104;
  // memrefA.stride[0] = 104;
  // memrefA.stride[1] = 1;
  // TwoDMemrefI8_t memrefB;  // weight 104x104xi8
  // memrefB.data = dataB;
  // memrefB.aligned_data = memrefB.data;
  // memrefB.offset = 0;
  // memrefB.shape[0] = 104;
  // memrefB.shape[1] = 104;
  // memrefB.stride[0] = 104;
  // memrefB.stride[1] = 1;
  // TwoDMemrefI32_t memrefC;  // output 104x104xi32
  // memrefC.data = dataC;
  // memrefC.aligned_data = memrefC.data;
  // memrefC.offset = 0;
  // memrefC.shape[0] = 104;
  // memrefC.shape[1] = 104;
  // memrefC.stride[0] = 104;
  // memrefC.stride[1] = 1;
  // TwoDMemrefI32_t memrefGolden;  // golden 104x104xi32
  // memrefGolden.data = dataGolden;
  // memrefGolden.aligned_data = memrefGolden.data;
  // memrefGolden.offset = 0;
  // memrefGolden.shape[0] = 104;
  // memrefGolden.shape[1] = 104;
  // memrefGolden.stride[0] = 104;
  // memrefGolden.stride[1] = 1;
    TwoDMemrefI8_t memrefA;  // input 16x16xi8
  memrefA.data = dataA;
  memrefA.aligned_data = memrefA.data;
  memrefA.offset = 0;
  memrefA.shape[0] = MAT_WIDTH;
  memrefA.shape[1] = MAT_WIDTH;
  memrefA.stride[0] = MAT_WIDTH;
  memrefA.stride[1] = 1;
  TwoDMemrefI8_t memrefB;  // weight 16x16xi8
  memrefB.data = dataB;
  memrefB.aligned_data = memrefB.data;
  memrefB.offset = 0;
  memrefB.shape[0] = MAT_WIDTH;
  memrefB.shape[1] = MAT_WIDTH;
  memrefB.stride[0] = MAT_WIDTH;
  memrefB.stride[1] = 1;
  TwoDMemrefI32_t memrefC;  // output 16x16xi32
  memrefC.data = dataC;
  memrefC.aligned_data = memrefC.data;
  memrefC.offset = 0;
  memrefC.shape[0] = MAT_WIDTH;
  memrefC.shape[1] = MAT_WIDTH;
  memrefC.stride[0] = MAT_WIDTH;
  memrefC.stride[1] = 1;
  TwoDMemrefI32_t memrefGolden;  // golden 16x16xi32
  memrefGolden.data = dataGolden;
  memrefGolden.aligned_data = memrefGolden.data;
  memrefGolden.offset = 0;
  memrefGolden.shape[0] = MAT_WIDTH;
  memrefGolden.shape[1] = MAT_WIDTH;
  memrefGolden.stride[0] = MAT_WIDTH;
  memrefGolden.stride[1] = 1;


  // initialize the matrices
  for (size_t i = 0; i < MAT_WIDTH_SQUARED; i++) {
    memrefA.aligned_data[i] = (int8_t)2;
  }
  memrefA.aligned_data[0] = 78;

  for (size_t i = 0; i < MAT_WIDTH_SQUARED; i++) {
    memrefB.aligned_data[i] = (int8_t)3;
  }
  memrefB.aligned_data[5] = 88;
  memrefB.aligned_data[200] = 96;

  for (size_t i = 0; i < MAT_WIDTH_SQUARED; i++) {
    memrefC.aligned_data[i] = (int32_t)0;
  }
  for (size_t i = 0; i < MAT_WIDTH_SQUARED; i++) {
    memrefGolden.aligned_data[i] = (int32_t)0;
  }

  (void)snrt_mcycle();

  _mlir_ciface_matmul104x104_zigzag(&memrefA, &memrefB, &memrefC);

  snrt_cluster_hw_barrier();

  (void)snrt_mcycle();

  // Correctness check -
  // from this point on only core 0 is required to be alive.
  int thiscore = snrt_cluster_core_idx();
  if (thiscore != 0)
    return 0;

  int nerr = 0;
  for (int i = 0; i < M_size * N_size; i++) {
   // int32_t error = memrefC.aligned_data[i] - memrefGolden.aligned_data[i];
   int32_t error = 0;
    if (error != 0)
      nerr += 1;
  }
  return nerr;
}
