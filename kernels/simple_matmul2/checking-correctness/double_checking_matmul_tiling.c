#include "data.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define MATRIX_LEN 16
#define CACHE_LINE_SIZE_BYTES 64
#define CACHE_LINE_COUNT 750 // cache size is 48 KB

/*
rm test.o out/regular_output out/tiled_general_output;clear; gcc -o test data.c
double_checking_matmul_tiling.c ;./test; diff out/regular_output
out/tiled_general_output

the compiler to stop after the compilation phase, without linking. Then, you can
link your two object files as so:

gcc -o myprog main.o module.o

*/

/*
This file contains an example of loop blocking
"blocks" in this case are of size 64 Bytes, the size of a single cache line
on my laptop

command to run example:
rm test.o out/regular_output out/tiled_general_output;clear;
gcc double_checking_matmul_tiling.c -o test.o;
./test.o; diff out/regular_output out/tiled_general_output

reference:
http://www.nic.uoregon.edu/~khuck/ts/acumem-report/manual_html/ch03s02.html
https://stackoverflow.com/questions/5141960/get-the-current-time-in-c
*/

/* results from my laptop...
$ lscpu | grep "cache"
L1d cache:                          448 KiB (12 instances)
L1i cache:                          640 KiB (12 instances)
L2 cache:                           9 MiB (6 instances)
L3 cache:                           18 MiB (1 instance)

$ cd /sys/devices/system/cpu/cpu0/cache/
$ cd index0
$ cat level type coherency_line_size size
1
Data
64
48K

(48 kilobytes) / (64 bytes) = 750
*/

/*Sample Output:
A x B = C where all matrices are 2048 x 2048
Block size is 64 / 8 = 8

Comparing tiled matrix multiplication to non-tiled...

-------------------------------------------------------------------
Time to execute multmatTiledGeneral: 21.000000
Time to execute multmat: 48.000000

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
void transposeSquareMat(squareMat *t, squareMat *m) ;

// interesting functions
void multmat_elt_wise(squareMat *a, squareMat *b, squareMat *c);
void multmat(squareMat *a, squareMat *b, squareMat *c);
void multmatTiled(squareMat *a, squareMat *b, squareMat *c);
void multmatTiledGeneral(squareMat *a, squareMat *b, squareMat *c,
                         uint64_t block_size);

int main() {
  printf("yodelayheehooooo~~~~! %ld\n\n", A[0]);
  // create and initialize three matrices x, y, and z.
  squareMat x, y, z, w, u;
  createSquareMat(&x, MATRIX_LEN); // dimensions are 2048 x 2048
  createSquareMat(&y, MATRIX_LEN);
  createSquareMat(&z, MATRIX_LEN);
  // createSquareMat(&w, MATRIX_LEN);
  createSquareMat(&u, MATRIX_LEN);
  // printf("trying trans\n");
  // transposeSquareMat(&y,&x);

  // // fill x and y with unique values from header file included
  fillSquareMatFrom1D(&x, 3, (const uint64_t *)&A);        // input 1
  fillSquareMatFrom1D(&y, 2, (const uint64_t *)&B);        // input 2
  fillSquareMatFrom1D(&u, 2, (const uint64_t *)&C_golden); // ground truth
  fillSquareMat(&z, 0);                                    // result 1
  // fillSquareMat(&w, 0);                                    // result 2
  // uint64_t block_size = CACHE_LINE_SIZE_BYTES / sizeof(uint64_t);

  // file handling
  FILE *reg_output = fopen("out/regular_output", "w");
  // FILE *tiled_general_output = fopen("out/tiled_general_output", "w");

  // // perform qmat equiv of MLIR on matrices x and y
  printf("before trans\n");
  printSquareMat(&x, reg_output);
  printSquareMat(&y, reg_output);
  transposeSquareMat(&y,&u);
  printf("after trans\n");
  printSquareMat(&y, reg_output);
  multmat(&x, &y, &z);

  // printSquareMat(&x, reg_output);
  // printSquareMat(&y, reg_output);
  printSquareMat(&z, reg_output);

  // close files and clean up
  fclose(reg_output);
  // fclose(tiled_general_output);
  destroySquareMat(&x);
  destroySquareMat(&y);
  destroySquareMat(&z);
  // destroySquareMat(&w);
  destroySquareMat(&u);
  printf("HOODLE!!!!!!\n");
  return 0;
}

void multmat(squareMat *a, squareMat *b, squareMat *c) {
  if (!((a->len == b->len) && (b->len == c->len))) {
    return;
  } // only square matrices allowed
  uint64_t cells = 0;
  for (size_t i = 0; i < a->len; i++) {   // for each row
    for (size_t j = 0; j < a->len; j++) { // for each col
      for (size_t k = 0; k < a->len;
           k++) { // sum (each elt in row * each elt in col)
        c->mat[i][j] += a->mat[i][k] * b->mat[k][j];
      }
      cells++;
    }
  }
}

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
  // printf("en is %d \n",en);
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
} // end of func

// printing helper function
void printSquareMat(squareMat *m, FILE *out) {
  fprintf(out, "\n{");
  for (size_t i = 0; i < m->len; i++) { // for each row
    fprintf(out, "{ ");
    for (size_t j = 0; j < m->len; j++) { // print each elt of row
      fprintf(out, "%d ", (int)m->mat[i][j]);
    }
    fprintf(out, " }\n");
  }
  fprintf(out, "}\n");
}

// matrix helpers
void createSquareMat(squareMat *m, uint64_t len) {
  printf("start of function call: howdy\n");
  m->mat = malloc(sizeof(uint64_t *) * len);
  printf("howdy2\n");
  m->len = len;
  printf("howdy3\n");
  for (size_t i = 0; i < m->len; i++) { // for each row
    printf("i is %ld\n",i);
    m->mat[i] = malloc(sizeof(uint64_t) * len);
  }
  printf("howdy4\n");
}

void destroySquareMat(squareMat *m) {
  for (size_t i = 0; i < m->len; i++) { // for each row
    free(m->mat[i]);
  }
  free(m->mat);
  m->len = 0;
}

void fillSquareMat(squareMat *a, uint64_t num) {
  for (size_t i = 0; i < a->len; i++) {   // for each row
    for (size_t j = 0; j < a->len; j++) { // for each col
      a->mat[i][j] = num;
    }
  }
}

void fillSquareMatFrom1D(squareMat *a, uint64_t num, const uint64_t *flat) {
  for (size_t i = 0; i < a->len; i++) {   // for each row
    for (size_t j = 0; j < a->len; j++) { // for each col
      a->mat[i][j] = flat[(i * a->len) + j];
    }
  }
}

void transposeSquareMat(squareMat *t, squareMat *m) {
  printf("hello\n");
  printf("t->len is %ld\n", t->len);
  uint64_t len = t->len;
  printf("len is %ld\n", len);
  // squareMat *m;
  // createSquareMat(m, MATRIX_LEN);
  printf("hello 2\n");

  printf("hello 3\n");
  for (size_t i = 0; i < len; i++) {   // for each row
    for (size_t j = 0; j < len; j++) { // for each col
      m->mat[i][j] = t->mat[i][j];
    }
  }
  printf("hello 4\n");
  for (size_t i = 0; i < len; i++) {   // for each row
    for (size_t j = 0; j < len; j++) { // for each col
      t->mat[i][j] = m->mat[j][i];
    }
  }
}

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