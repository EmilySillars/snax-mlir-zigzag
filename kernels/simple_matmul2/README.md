# QMatMul MLIR ----> ZigZag ----> Manually Transformed MLIR

- Currently, SNAX runs matrix multiplication by embedding a call to an MLIR function into C code.

[main.c](main.c) calls `_mlir_ciface_simple_matmul(&memrefA, &memrefB, &memrefC);` which is fed to the accelerator.

- [I have been practicing writing simple MLIR functions here.](https://github.com/EmilySillars/llvm-project-pistachio/tree/learn-llvm/EMILY-NOTES/learning-mlir#quick-examples-w-mlir-cpu-runner)

## Running matmul2.mlir on snax

```
sudo chmod 666 /var/run/docker.sock
docker run -itv `pwd`:/repo:z ghcr.io/kuleuven-micas/snax-mlir:main
pip3 install -e /repo
cd repo/kernels/simple_matmul2/
make clean; make allrun
```

## input MLIR

```
func.func @simple_matmul(%arg0: memref<16x16xi8>, %arg1: memref<16x16xi8, strided<[1, 16], offset:0>>, %arg2: memref<16x16xi32>) {
    %c0_i32 = arith.constant 0 : i32
    linalg.quantized_matmul ins(%arg0, %arg1, %c0_i32, %c0_i32 : memref<16x16xi8>, memref<16x16xi8, strided<[1, 16], offset:0>>, i32, i32) outs(%arg2 : memref<16x16xi32>)
    return
}
```

which lowers to the following using gobolt.org MLIR opt (trunk) `--linalg-generalize-named-ops --mlir-print-local-scope --mlir-print-op-generic`

```
"builtin.module"() ({
  "func.func"() <{function_type = (memref<16x16xi8>, memref<16x16xi8, strided<[1, 16]>>, memref<16x16xi32>) -> (), sym_name = "simple_matmul"}> ({
  ^bb0(%arg0: memref<16x16xi8>, %arg1: memref<16x16xi8, strided<[1, 16]>>, %arg2: memref<16x16xi32>):
    %0 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "linalg.generic"(%arg0, %arg1, %0, %0, %arg2) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 4, 1>}> ({
    ^bb0(%arg3: i8, %arg4: i8, %arg5: i32, %arg6: i32, %arg7: i32):
      %1 = "arith.extsi"(%arg3) : (i8) -> i32
      %2 = "arith.subi"(%1, %arg5)  : (i32, i32) -> i32
      %3 = "arith.extsi"(%arg4) : (i8) -> i32
      %4 = "arith.subi"(%3, %arg6)  : (i32, i32) -> i32
      %5 = "arith.muli"(%2, %4)  : (i32, i32) -> i32
      %6 = "arith.addi"(%arg7, %5) : (i32, i32) -> i32
      "linalg.yield"(%6) : (i32) -> ()
    }) : (memref<16x16xi8>, memref<16x16xi8, strided<[1, 16]>>, i32, i32, memref<16x16xi32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
```

## equivalent python workload object

C-ish pseudocode (ignoring sign extension and subtracting 0 instructions)

```
for d0; d0 < 16; d0++:
for d1; d1 < 16; d1++;
for d2; d2 < 16; d2++;
  arg7[d0][d1] += arg3[d0][d2] * arg4[d2][d1]; // and this is a MAC!
```

equivalent workload object (TENTATIVE!!)

```
workload = {
    0: {
        "operator_type": "default",
        "equation": "O[d0][d1] += I[d0][d2] * W[d2][d1]",
        "dimension_relations": [],
        "loop_dim_size": {"D0": 16, "D1": 16, "D2": 16},
        "operand_precision": {"O": 32, "O_final": 32, "W": 8, "I": 8},
        "operand_source": {"W": [], "I": []},
        "constant_operands": ["I", "W"],
        "padding": {},
    }
}
```

## output from ZigZag

```
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
SpatialMapping({'O': [[('D2', 8.0)], [], [], []], 'W': [[('D2', 8.0)], [], []], 'I': [[('D2', 8.0)], [], []]})
```

## transformed C-code

```
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
```

## transformation in MLIR WIP

- [Full WIP file here](https://github.com/EmilySillars/llvm-project-pistachio/blob/learn-llvm/EMILY-NOTES/learning-mlir/print-memrefs-qmat-tiled.mlir)

- Instead of complex zigzag tiling, I started with simple 16x2 and 2x16 tiles...

```
"func.func"() <{function_type = (memref<16x16xi8>, memref<16x16xi8, strided<[1, 16]>>, memref<16x16xi32, strided<[16,1]>>) -> (), sym_name = "simple_matmul_tiled"}> ({
  ^bb0(%arg0: memref<16x16xi8>, %arg1: memref<16x16xi8, strided<[1,16]>>, %arg2: memref<16x16xi32, strided<[16,1]>>):
    %0 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %zero = arith.constant 0 : index
    %one = arith.constant 1: index
    %sixteen = arith.constant 16 : index
    %two = arith.constant 2 : index
    %alloc = memref.alloc() {alignment = 1 : i64} : memref<2x2xi32> // fake result value
    linalg.fill ins(%0 : i32) outs(%alloc : memref<2x2xi32>)
    %arg1_t = memref.transpose %arg1 (i, j) -> (j, i): memref<16x16xi8, strided<[1,16]>> to  memref<16x16xi8, strided<[16,1]>>
    %fake_arg2 = memref.alloc() {alignment = 1 : i64} : memref<16x16xi32> // fake result value
  
  // enter FOR LOOP
    scf.for %i = %zero to %sixteen step %two iter_args() -> () {
  
  // pull out left tile
    %leftTile = memref.subview %arg0[%i,%zero][2,16][1,1] : memref<16x16xi8> to memref<2x16xi8, strided<[16, 1], offset: ?>>
    %leftTileCasted = memref.cast %leftTile : memref<2x16xi8, strided<[16, 1], offset: ?>> to memref<2x16xi8>
   
   // pull out right tile
    %rightTile = memref.subview %arg1_t[%zero,%i][16,2][1,1] : memref<16x16xi8, strided<[16,1]>> to memref<16x2xi8, strided<[16,1], offset: ?>>
    %rightTileCasted = memref.cast %rightTile : memref<16x2xi8, strided<[16,1], offset: ?>> to memref<16x2xi8, strided<[16,1]>>
    
    // pull out output tile
    %outputTile = memref.subview %arg2[%i,%i][2,2][1,1] : memref<16x16xi32, strided<[16,1]>> to memref<2x2xi32, strided<[16,1], offset: ?>>
    %outputTileCasted = memref.cast %outputTile : memref<2x2xi32, strided<[16,1], offset: ?>> to memref<2x2xi32, strided<[16,1]>>
   //%outputTileCasted2 = memref.cast %outputTileCasted : memref<2x2xi32, strided<[16,1]>> to memref<2x2xi32>
  
  //feed computation to linalg generic (accelerator workload)
    "linalg.generic"(%leftTileCasted, %rightTileCasted, %0, %0, %outputTileCasted) <{
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d2)>, 
        affine_map<(d0, d1, d2) -> (d2, d1)>, 
        affine_map<(d0, d1, d2) -> ()>, 
        affine_map<(d0, d1, d2) -> ()>, 
        affine_map<(d0, d1, d2) -> (d0, d1)>], 
        iterator_types = [
          #linalg.iterator_type<parallel>, 
          #linalg.iterator_type<parallel>, 
          #linalg.iterator_type<reduction>], 
          operandSegmentSizes = array<i32: 4, 1>}> ({
    ^bb0(%arg3: i8, %arg4: i8, %arg5: i32, %arg6: i32, %arg7: i32):
      %1 = "arith.extsi"(%arg3) : (i8) -> i32
      %2 = "arith.subi"(%1, %arg5) : (i32, i32) -> i32
      %3 = "arith.extsi"(%arg4) : (i8) -> i32
      %4 = "arith.subi"(%3, %arg6) : (i32, i32) -> i32
      %5 = "arith.muli"(%2, %4) : (i32, i32) -> i32
      %6 = "arith.addi"(%arg7, %5) : (i32, i32) -> i32
      func.call @myPrintI32(%6) : (i32 )-> ()
      "linalg.yield"(%6) : (i32) -> ()
    }) : (memref<2x16xi8>, memref<16x2xi8, strided<[16,1]>>, i32, i32, memref<2x2xi32, strided<[16,1]>>) -> ()
    %alloc3 = memref.cast %alloc :  memref<2x2xi32> to memref<*xi32>
    func.call @printMemrefI32(%alloc3) : (memref<*xi32>) -> ()
    }
    
    %alloc2 = memref.cast %alloc :  memref<2x2xi32> to memref<*xi32>
    func.call @printMemrefI32(%alloc2) : (memref<*xi32>) -> ()
    memref.dealloc %alloc : memref<2x2xi32>
    memref.dealloc %fake_arg2 : memref<16x16xi32>
    "func.return"() : () -> ()
  }) : () -> ()
```

Input matrix A:

```
memref.global "private" constant @__constant_16x16f32 : memref<16x16xi8> = 
dense<[
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]> 
```

Input matrix B:

```
// a 16x16 matrix filled with the value 3
```

Current output:

```
Untiled:
Unranked Memref base@ = 0xc46e540 rank = 2 offset = 0 sizes = [16, 16] strides = [16, 1] data = 
[[408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408], 
 [408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408], 
 [408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408], 
 [408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408], 
 [408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408], 
 [408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408], 
 [408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408], 
 [408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408], 
 [408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408], 
 [408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408], 
 [408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408], 
 [408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408], 
 [408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408], 
 [408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408], 
 [408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408], 
 [408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408]]

"Tiled"
Unranked Memref base@ = 0xc481e40 rank = 2 offset = 0 sizes = [16, 16] strides = [16, 1] data = 
[[408,   408,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0], 
 [408,   408,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0], 
 [0,   0,   408,   408,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0], 
 [0,   0,   408,   408,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0], 
 [0,   0,   0,   0,   408,   408,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0], 
 [0,   0,   0,   0,   408,   408,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0], 
 [0,   0,   0,   0,   0,   0,   408,   408,   0,   0,   0,   0,   0,   0,   0,   0], 
 [0,   0,   0,   0,   0,   0,   408,   408,   0,   0,   0,   0,   0,   0,   0,   0], 
 [0,   0,   0,   0,   0,   0,   0,   0,   408,   408,   0,   0,   0,   0,   0,   0], 
 [0,   0,   0,   0,   0,   0,   0,   0,   408,   408,   0,   0,   0,   0,   0,   0], 
 [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   408,   408,   0,   0,   0,   0], 
 [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   408,   408,   0,   0,   0,   0], 
 [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   408,   408,   0,   0], 
 [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   408,   408,   0,   0], 
 [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   408,   408], 
 [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   408,   408]]

```



## ignore notes below this line -----------------------------------------



## transformed MLIR WIP (make it suboptimal just to try doing tiling on snax?)

plan:

tile size: tile size of 4 x 4???? or is it 4 vertical slices with width 4 and depth 16?

loop order?

```
for d0; d0 < 16; d0++:
for d1; d1 < 16; d1++;
for d2; d2 < 16; d2++;
  arg7[d0][d1] += arg3[d0][d2] * arg4[d2][d1]; // and this is a MAC!
```

maybe this is tiled a little. does it give same output?

```
for d0_1; d0_1 < 16; d0_1++:
for d0_2; d0_2 < 16; d0_2++:
for d1; d1 < 16; d1++:
for d2; d2 < 16; d2++:
  d0 = d0_1 + d0_2*4
  arg7[d0][d1] += arg3[d0][d2] * arg4[d2][d1]; // and this is a MAC!
```





change this to have 4 tiles?

~~~~for d0; d0 < 16; d0++:~~
~~for d1; d1 < 4; d0++:~~
~~for d2; d2 < 4; d2++:~~
 arg7[d0][d1] += arg3[d0][d2] * arg4[d2][d1];~~
~~~~



hoodle

```
"builtin.module"() ({
  "func.func"() <{function_type = (memref<16x16xi8>, memref<16x16xi8, strided<[1, 16]>>, memref<16x16xi32>) -> (), sym_name = "simple_matmul"}> ({
  ^bb0(%arg0: memref<16x16xi8>, %arg1: memref<16x16xi8, strided<[1, 16]>>, %arg2: memref<16x16xi32>):
    %0 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "linalg.generic"(%arg0, %arg1, %0, %0, %arg2) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 4, 1>}> ({
    ^bb0(%arg3: i8, %arg4: i8, %arg5: i32, %arg6: i32, %arg7: i32):
      %1 = "arith.extsi"(%arg3) : (i8) -> i32
      %2 = "arith.subi"(%1, %arg5)  : (i32, i32) -> i32
      %3 = "arith.extsi"(%arg4) : (i8) -> i32
      %4 = "arith.subi"(%3, %arg6)  : (i32, i32) -> i32
      %5 = "arith.muli"(%2, %4)  : (i32, i32) -> i32
      %6 = "arith.addi"(%arg7, %5) : (i32, i32) -> i32
      "linalg.yield"(%6) : (i32) -> ()
    }) : (memref<16x16xi8>, memref<16x16xi8, strided<[1, 16]>>, i32, i32, memref<16x16xi32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
```

I don't think I need to meet yet. Anything you'd like to check in with me about?

## transformed MLIR

