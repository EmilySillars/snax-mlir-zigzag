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
void _mlir_ciface_tiled_matmul(TwoDMemrefI8_t *a, TwoDMemrefI8_t *b,
                               TwoDMemrefI32_t *c);
void _mlir_ciface_tester(TwoDMemrefI8_t *a, TwoDMemrefI8_t *b,
                                TwoDMemrefI32_t *c);
void pineapple(TwoDMemrefI8_t *a, TwoDMemrefI8_t *b,
                                TwoDMemrefI32_t *c);
// void _mlir_ciface_hoodle(TwoDMemrefI8_t *a, TwoDMemrefI8_t *b,
//                                 TwoDMemrefI32_t *c);

void hoodle(TwoDMemrefI8_t *x){
  printf("hoodle\n");
}

void _mlir_ciface_mlirFunc(TwoDMemrefI8_t *a, TwoDMemrefI8_t *b, TwoDMemrefI32_t *c);
//void _mlir_ciface_cFunc(TwoDMemrefI8_t *a);


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

void cCodeEquivalentThreeLoops(TwoDMemrefI8_t *x, TwoDMemrefI8_t *y,
                               TwoDMemrefI32_t *z);
void cCodeEquivalent(TwoDMemrefI8_t *x, TwoDMemrefI8_t *y, TwoDMemrefI32_t *z);
void print2DMemRefI8_t(TwoDMemrefI8_t *x, int32_t width);
void print2DMemRefI32_t(TwoDMemrefI32_t *x, int32_t width);

// ADDING EVEN SMALLER MATRICES TO TEST!
const int8_t little_A[256] = {
    1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 1,  2,  3,
    4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 1,  2,  3,  4,  5,  6,
    7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 1,  2,  3,  4,  5,  6,  7,  8,  9,
    10, 11, 12, 13, 14, 15, 16, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
    13, 14, 15, 16, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
    16, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 1,  2,
    3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 1,  2,  3,  4,  5,
    6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 1,  2,  3,  4,  5,  6,  7,  8,
    9,  10, 11, 12, 13, 14, 15, 16, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
    12, 13, 14, 15, 16, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
    15, 16, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 1,
    2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 1,  2,  3,  4,
    5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 1,  2,  3,  4,  5,  6,  7,
    8,  9,  10, 11, 12, 13, 14, 15, 16};
const int8_t little_B[256] = {
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
const int32_t little_golden[256] = {
    408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408,
    408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408,
    408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408,
    408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408,
    408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408,
    408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408,
    408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408,
    408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408,
    408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408,
    408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408,
    408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408,
    408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408,
    408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408,
    408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408,
    408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408,
    408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408,
    408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408, 408,
    408};

// LET'S TRY TO SPLICE IN THE SCF TILE LOGIC FOR LOOPS!
// void for_each_tile(TwoDMemrefI8_t *a, TwoDMemrefI8_t *b, TwoDMemrefI32_t *c) {


// }

void process_tile(TwoDMemrefI8_t *a, TwoDMemrefI8_t *b, TwoDMemrefI32_t *c) {
  (void)snrt_mcycle();
  _mlir_ciface_simple_matmul(a, b, c);
  snrt_cluster_hw_barrier();
  (void)snrt_mcycle();
}

void process_matrix(TwoDMemrefI8_t *a, TwoDMemrefI8_t *b, TwoDMemrefI32_t *c) {
  (void)snrt_mcycle();

  _mlir_ciface_simple_matmul(a, b, c);

  snrt_cluster_hw_barrier();

  (void)snrt_mcycle();
}

int main() {

  printf("PAMPLEMOUSSE VOLCANO: Setting up the data.\n");

  // Create memref objects for data stored in L3
  TwoDMemrefI8_t memrefA;
  memrefA.data = (int8_t *)&little_A;
  memrefA.aligned_data = memrefA.data;
  memrefA.offset = 0;

  TwoDMemrefI8_t memrefB;
  memrefB.data = (int8_t *)&little_B;
  memrefB.aligned_data = memrefB.data;
  memrefB.offset = 0;

  TwoDMemrefI32_t memrefC;
  memrefC.data = (int32_t *)&C;
  memrefC.aligned_data = memrefC.data;
  memrefC.offset = 0;

  // -------------------------------------------------- V
  // I want a C function to call an MLIR function
  _mlir_ciface_mlirFunc(&memrefA, &memrefB, &memrefC);

  // I want that MLIR function to call a C function
  // -------------------------------------------------- ^

  // print2DMemRefI8_t(&memrefA, M_size); // PAMPLEMOUSSE
  // print2DMemRefI8_t(&memrefB, M_size); // PAMPLEMOUSSE
  // print2DMemRefI32_t(&memrefC, M_size); // PAMPLEMOUSSE

  //process_matrix(&memrefA, &memrefB, &memrefC);
  // _mlir_ciface_simple_matmul(&memrefA, &memrefB, &memrefC);
  //(void)snrt_mcycle();
  process_matrix(&memrefA, &memrefB, &memrefC);
  // snrt_cluster_hw_barrier();
  // (void)snrt_mcycle();
  // pineapple(&memrefA, &memrefB, &memrefC);

  printf("PAMPLEMOUSSE VOLCANO: Performed MLIR kernel.\n");

  // Correctness check -
  // from this point on only core 0 is required to be alive.
  int thiscore = snrt_cluster_core_idx();
  if (thiscore != 0)
    return 0;

  int nerr = 0;
  for (int i = 0; i < M_size * N_size; i++) {
    int32_t error = memrefC.aligned_data[i] - little_golden[i]; // C_golden[i];
    if (error != 0)
      nerr += 1;
  }

  // print2DMemRefI32_t(&memrefC, M_size);

  if (nerr != 0) {
    printf("Output does not match the golden value!\n");
    return nerr;
  }
  else{
    printf("correct!\n");
  }

  // print2DMemRefI32_t(&memrefC, M_size);

  // second correctness check - is the c code really equivalent??
  // TwoDMemrefI32_t z;
  // z.data = (int32_t *)&C;
  // z.aligned_data = z.data;
  // z.offset = 0;

  // cCodeEquivalent(&memrefA, &memrefB, &z); // PAMPLEMOUSSE

  // nerr = 0;
  // for (int i = 0; i < M_size * N_size; i++) {
  //   int32_t error = z.aligned_data[i] - C_golden[i];
  //   if (error != 0)
  //     nerr += 1;
  // }
  // if (nerr != 0) {
  //   printf("Z does not match the golden value!\n");
  //   print2DMemRefI32_t(&z, M_size); // PAMPLEMOUSSE
  // }

  // third correctness check - is THIS c code really equivalent???
  // TwoDMemrefI32_t w;
  // w.data = (int32_t *)&C;
  // w.aligned_data = w.data;
  // w.offset = 0;

  // cCodeEquivalentThreeLoops(&memrefA, &memrefB, &w); // PAMPLEMOUSSE

  // nerr = 0;
  // for (int i = 0; i < M_size * N_size; i++) {
  //   int32_t error = w.aligned_data[i] - C_golden[i];
  //   if (error != 0)
  //     nerr += 1;
  // }
  // if (nerr != 0) {
  //   printf("w does not match the golden value!\n");
  //   print2DMemRefI32_t(&w, M_size); // PAMPLEMOUSSE
  // }
  return nerr;
}

// helper funcs below
void print2DMemRefI8_t(TwoDMemrefI8_t *x, int32_t width) {
  printf("[\n");
  // we ASSUME a square 2D array
  int32_t col = 0;
  for (int i = 0; i < width * width; i++) {
    if (col == width) {
      col = 0;
      printf("\n %d ", x->aligned_data[i]);

    } else {
      printf(" %d ", x->aligned_data[i]);
    }
    col++;
  }
  printf("]\n");
}

void print2DMemRefI32_t(TwoDMemrefI32_t *x, int32_t width) {
  printf("[\n");
  // we ASSUME a square 2D array
  int32_t col = 0;
  for (int i = 0; i < width * width; i++) {
    if (col == width) {
      col = 0;
      printf("\n %d ", x->aligned_data[i]);

    } else {
      printf(" %d ", x->aligned_data[i]);
    }
    col++;
  }
  printf("]\n");
}

void cCodeEquivalent(TwoDMemrefI8_t *x, TwoDMemrefI8_t *y, TwoDMemrefI32_t *z) {
  printf("M_size is %d and N_size is %d\n", M_size, N_size);
  for (int i = 0; i < M_size * N_size; i++) {
    z->aligned_data[i] =
        (int32_t)x->aligned_data[i] * (int32_t)y->aligned_data[i];
  }
}

void cCodeEquivalentThreeLoops(TwoDMemrefI8_t *x, TwoDMemrefI8_t *y,
                               TwoDMemrefI32_t *z) {
  // printf("M_size is %d and N_size is %d\n",M_size, N_size);
  // for (int i = 0; i < M_size * N_size; i++) {
  //   z->aligned_data[i] = x->aligned_data[i] * y->aligned_data[i];
  // }
  int z_index, x_index, y_index = 0;
  for (int d0 = 0; d0 < M_size; d0++) {
    for (int d1 = 0; d1 < M_size; d1++) {
      for (int d2 = 0; d2 < M_size; d2++) {
        // arg7[d0][d1] += arg3[d0][d2] * arg4[d2][d1]; // and this is a MAC!
        z_index = (d0 * M_size) + d1;
        x_index = (d0 * M_size) + d2;
        y_index = (d2 * M_size) + d1;
        z->aligned_data[z_index] +=
            x->aligned_data[x_index] * y->aligned_data[y_index];
      }
    }
  }
}

/*
for d0; d0 < 16; d0++:
for d1; d1 < 16; d1++;
for d2; d2 < 16; d2++;
  arg7[d0][d1] += arg3[d0][d2] * arg4[d2][d1]; // and this is a MAC!
*/

/*
PLAN:
1) rewrite the single loop C code version as a two-loop code version?
2) Create equivalent python workload
3) transform based on this ouput
*/

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