#include "data-modified.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define MATRIX_LEN 16
#define CACHE_LINE_SIZE_BYTES 64
#define CACHE_LINE_COUNT 750 // cache size is 48 KB

/*
rm test.o out/result out/groundTruth;clear; \
gcc -o test.o data-modified.c double_checking_matmul_tiling.c &&
./test.o; diff out/result out/groundTruth
*/

// book keeping structs/funcs
typedef struct squareMatrix {
  uint64_t **mat;
  uint64_t len;
} squareMat;
void createSquareMat(squareMat *m, uint64_t len);
void destroySquareMat(squareMat *m);
void printSquareMat(squareMat *m, FILE *out);
void fillSquareMat(squareMat *a, uint64_t num);
void fillSquareMatFrom1D(squareMat *a, uint64_t num, const uint64_t *flat);
void transposeSquareMat(squareMat *t, squareMat *m);

// interesting functions
void mlir_qmat(squareMat *a, squareMat *b, squareMat *c, squareMat *dummy);
void mlir_qmat_transformed(squareMat *a, squareMat *b, squareMat *c,
                           squareMat *dummy);
void multmat_elt_wise(squareMat *a, squareMat *b, squareMat *c);
void multmat(squareMat *a, squareMat *b, squareMat *c);
void multmatTiled(squareMat *a, squareMat *b, squareMat *c);
void multmatTiledGeneral(squareMat *a, squareMat *b, squareMat *c,
                         uint64_t block_size);

int main() {
  printf("yodelayheehooooo~~~~! %ld\n\n", A[0]);
  squareMat x, y, z, u, dummy;
  createSquareMat(&x, MATRIX_LEN);
  createSquareMat(&y, MATRIX_LEN);
  createSquareMat(&z, MATRIX_LEN);
  createSquareMat(&u, MATRIX_LEN);
  createSquareMat(&dummy, MATRIX_LEN);

  // fill x and y with unique values from header file data-modified.c
  fillSquareMatFrom1D(&x, 3, (const uint64_t *)&A);        // input 1
  fillSquareMatFrom1D(&y, 2, (const uint64_t *)&B);        // input 2
  fillSquareMatFrom1D(&u, 2, (const uint64_t *)&C_golden); // ground truth
  fillSquareMat(&z, 0);                                    // result
  fillSquareMat(&dummy, 0);                                // dummy

  // file handling
  FILE *result = fopen("out/result", "w");
  FILE *groundTruth = fopen("out/groundTruth", "w");

  // perform equivalent of matmul MLIR code on matrices x and y
  // mlir_qmat(&x, &y, &z, &dummy);

  // perform transformed C code version of matmul MLIR
  mlir_qmat_transformed(&x, &y, &z, &dummy);

  // print result and ground truth
  printSquareMat(&z, result);
  printSquareMat(&u, groundTruth);

  // close files and clean up
  fclose(result);
  fclose(groundTruth);
  destroySquareMat(&x);
  // destroySquareMat(&dummy);
  destroySquareMat(&y);
  destroySquareMat(&z);
  destroySquareMat(&u);
  destroySquareMat(&dummy);
  return 0;
}

// equivalent to linalg.quantized_matmul in matmul.mlir
void mlir_qmat(squareMat *a, squareMat *b, squareMat *c, squareMat *dummy) {
  transposeSquareMat(b, dummy);
  // only square matrices allowed
  multmat(a, b, c);
}

// equivalent to numpy.matmul
void multmat(squareMat *a, squareMat *b, squareMat *c) {
  if (!((a->len == b->len) && (b->len == c->len))) {
    return;
  }                                       // only square matrices allowed
  for (size_t i = 0; i < a->len; i++) {   // for each row
    for (size_t j = 0; j < a->len; j++) { // for each col
      for (size_t k = 0; k < a->len;
           k++) { // sum (each elt in row * each elt in col)
        c->mat[i][j] += a->mat[i][k] * b->mat[k][j];
      }
    }
  }
}

// equivalent to numpy.multiply
void multmat_elt_wise(squareMat *a, squareMat *b, squareMat *c) {
  if (!((a->len == b->len) && (b->len == c->len))) {
    return;
  } // only square matrices allowed
  uint64_t cells = 0;
  for (size_t i = 0; i < a->len; i++) {   // for each row
    for (size_t j = 0; j < a->len; j++) { // for each col

      c->mat[i][j] = a->mat[i][j] * b->mat[j][i];
    }
  }
}

void multmatTiledGeneral(squareMat *a, squareMat *b, squareMat *c,
                         uint64_t bsize) {

  int i, j, k, kk, jj;
  double sum;
  uint64_t n = c->len;
  int en = bsize * (n / bsize); // Amount that fits evenly into blocks
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      c->mat[i][j] = 0.0;

  for (kk = 0; kk < en; kk += bsize) {
    for (jj = 0; jj < en; jj += bsize) {
      for (i = 0; i < n; i++) {
        for (j = jj; j < jj + bsize; j++) {
          sum = c->mat[i][j];
          for (k = kk; k < kk + bsize; k++) {
            sum += a->mat[i][k] * b->mat[k][j];
          }
          c->mat[i][j] = sum;
        }
      }
    }
  }
}

/*
for d0; d0 < 16; d0++:
for d1; d1 < 16; d1++;
for d2; d2 < 16; d2++;
  arg7[d0][d1] += arg3[d0][d2] * arg4[d2][d1]; // and this is a MAC!

TRANSFORMED INTO...

==========================================================================
Temporal Loops                     O            W            I
==========================================================================
for D0 in [0, 4):                  l1           l1           l1
--------------------------------------------------------------------------
  for D0 in [0, 4):                l1           l1           l1
--------------------------------------------------------------------------
    for D1 in [0, 2):              l1           l1           l1
--------------------------------------------------------------------------
      for D1 in [0, 2):            l1           l1           l1
--------------------------------------------------------------------------
        for D1 in [0, 4):          l1           l1           l1
--------------------------------------------------------------------------
          for D2 in [0, 2):        rf_32b_O     l1           l1
--------------------------------------------------------------------------
==========================================================================
Spatial Loops
==========================================================================
            parfor D2 in [0, 8):
--------------------------------------------------------------------------



*/
void mlir_qmat_transformed(squareMat *a, squareMat *b, squareMat *c,
                           squareMat *dummy) {
  transposeSquareMat(b, dummy);
  // only square matrices allowed
  size_t d0_1_bk_sz = a->len / 4;
  size_t d1_1_bk_sz = a->len / 2;
  size_t d1_2_bk_sz = d1_1_bk_sz / 2;
  size_t d2_1_bk_sz = a->len / 2;

  for (size_t d0_1 = 0; d0_1 < 4; d0_1++) {
    for (size_t d0_2 = 0; d0_2 < 4; d0_2++) {
      for (size_t d1_1 = 0; d1_1 < 2; d1_1++) {
        for (size_t d1_2 = 0; d1_2 < 2; d1_2++) {
          for (size_t d1_3 = 0; d1_3 < 4; d1_3++) {
            for (size_t d2_1 = 0; d2_1 < 2; d2_1++) {
              for (size_t d2_2 = 0; d2_2 < 8;
                   d2_2++) { // technically spacially unrolled, but won't show
                             // that here
                size_t d0 = d0_1 * d0_1_bk_sz + d0_2;
                size_t d1 = d1_1 * d1_1_bk_sz + d1_2 * d1_2_bk_sz + d1_3;
                size_t d2 = d2_1 * d2_1_bk_sz + d2_2;
                c->mat[d0][d1] += a->mat[d0][d2] * b->mat[d2][d1];
              }
            }
          }
        }
      }
    }
  }
}

// void mlir_qmat_transformed(squareMat *a, squareMat *b, squareMat *c,
//                            squareMat *dummy) {
//   transposeSquareMat(b, dummy);
//   // only square matrices allowed
//   for (size_t d0 = 0; d0 < a->len; d0++) {   // for each row
//     for (size_t d1 = 0; d1 < a->len; d1++) { // for each col
//       for (size_t d2 = 0; d2 < a->len;
//            d2++) {
//         c->mat[d0][d1] += a->mat[d0][d2] * b->mat[d2][d1];
//       }
//     }
//   }
// }

// void mlir_qmat_transformed(squareMat *a, squareMat *b, squareMat *c,
//                            squareMat *dummy) {
//   transposeSquareMat(b, dummy);
//   // only square matrices allowed
//   for (size_t i = 0; i < a->len; i++) {   // for each row
//     for (size_t j = 0; j < a->len; j++) { // for each col
//       for (size_t k = 0; k < a->len;
//            k++) { // sum (each elt in row * each elt in col)
//         c->mat[i][j] += a->mat[i][k] * b->mat[k][j];
//       }
//     }
//   }
// }

// printing helper function
void printSquareMat(squareMat *m, FILE *out) {
  fprintf(out, "[\n");
  for (size_t i = 0; i < m->len; i++) {   // for each row
    for (size_t j = 0; j < m->len; j++) { // print each elt of row
      fprintf(out, " %d", (int)m->mat[i][j]);
    }
    fprintf(out, "\n");
  }
  fprintf(out, "]\n");
}

// matrix helpers
void createSquareMat(squareMat *m, uint64_t len) {
  m->mat = malloc(sizeof(uint64_t *) * len);
  m->len = len;
  for (size_t i = 0; i < m->len; i++) { // for each row
    m->mat[i] = malloc(sizeof(uint64_t) * len);
  }
}

void destroySquareMat(squareMat *m) {
  for (size_t i = 0; i < m->len; i++) { // for each row
    free(m->mat[i]);
  }
  free(m->mat);
  m->len = 0;
}

// set every elt of matrix a to the value num
void fillSquareMat(squareMat *a, uint64_t num) {
  for (size_t i = 0; i < a->len; i++) {   // for each row
    for (size_t j = 0; j < a->len; j++) { // for each col
      a->mat[i][j] = num;
    }
  }
}

// take a flat representation of a matrix, and turn it into a 2D array
void fillSquareMatFrom1D(squareMat *a, uint64_t num, const uint64_t *flat) {
  // assume matrix a is fully allocated
  for (size_t i = 0; i < a->len; i++) {   // for each row
    for (size_t j = 0; j < a->len; j++) { // for each col
      a->mat[i][j] = flat[(i * a->len) + j];
    }
  }
}

void transposeSquareMat(squareMat *t, squareMat *dummy) {
  // assume t and dummy are both fully allocated + initialized matrices
  uint64_t len = t->len;
  for (size_t i = 0; i < len; i++) {   // for each row
    for (size_t j = 0; j < len; j++) { // for each col
      dummy->mat[i][j] = t->mat[i][j];
    }
  }
  for (size_t i = 0; i < len; i++) {   // for each row
    for (size_t j = 0; j < len; j++) { // for each col
      t->mat[i][j] = dummy->mat[j][i];
    }
  }
}

/*
"builtin.module"() ({
  "func.func"() <{function_type = (memref<16x16xi8>, memref<16x16xi8,
strided<[1, 16]>>, memref<16x16xi32>) -> (), sym_name = "simple_matmul"}> ({
  ^bb0(%arg0: memref<16x16xi8>, %arg1: memref<16x16xi8, strided<[1, 16]>>,
%arg2: memref<16x16xi32>): %0 = "arith.constant"() <{value = 0 : i32}> : () ->
i32 "linalg.generic"(%arg0, %arg1, %0, %0, %arg2) <{indexing_maps =
[affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>,
affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0,
d1, d2) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>,
#linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
operandSegmentSizes = array<i32: 4, 1>}> ({ ^bb0(%arg3: i8, %arg4: i8, %arg5:
i32, %arg6: i32, %arg7: i32): %1 = "arith.extsi"(%arg3) : (i8) -> i32 %2 =
"arith.subi"(%1, %arg5)  : (i32, i32) -> i32 %3 = "arith.extsi"(%arg4) : (i8) ->
i32 %4 = "arith.subi"(%3, %arg6)  : (i32, i32) -> i32 %5 = "arith.muli"(%2, %4)
: (i32, i32) -> i32 %6 = "arith.addi"(%arg7, %5) : (i32, i32) -> i32
      "linalg.yield"(%6) : (i32) -> ()
    }) : (memref<16x16xi8>, memref<16x16xi8, strided<[1, 16]>>, i32, i32,
memref<16x16xi32>) -> () "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
*/

/*
code/mem/matmult/bmm.c
 void bijk(array A, array B, array C, int n, int bsize)
 {
 int i, j, k, kk, jj;
 double sum;
 int en = bsize * (n/bsize); /* Amount that fits evenly into blocks

 for (i = 0; i < n; i++)
 for (j = 0; j < n; j++)
 C[i][j] = 0.0;

 for (kk = 0; kk < en; kk += bsize) {
 for (jj = 0; jj < en; jj += bsize) {
 for (i = 0; i < n; i++) {
 for (j = jj; j < jj + bsize; j++) {
 sum = C[i][j];
 for (k = kk; k < kk + bsize; k++) {
 sum += A[i][k]*B[k][j];
 }
 C[i][j] = sum;
 }
 }
 }
 }
 }
*/