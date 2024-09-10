# Tiling 16x16 Matrix Multiplication on SNAX + Gemm

[back to all tests](../../zigzag-fork/README.md#Examples)

## I. Input MLIR

### linalg on memrefs (someday tensors!)

```
func.func @matmul_16x16(%arg0: memref<16x16xi8>, %arg1: memref<16x16xi8, strided<[1, 16], offset:0>>, %arg2: memref<16x16xi32>) attributes {llvm.emit_c_interface = true} {
    %c0_i32 = arith.constant 0 : i32
    linalg.quantized_matmul ins(%arg0, %arg1, %c0_i32, %c0_i32 : memref<16x16xi8>, memref<16x16xi8, strided<[1, 16], offset:0>>, i32, i32) outs(%arg2 : memref<16x16xi32>)
    return
}
```

Lower the specific linalg operation to a linalg generic operation with `--linalg-generalize-named-ops` and print in generic MLIR syntax with `--mlir-print-op-generic`. Print the affine maps inside the function with `--mlir-print-local-scope`.

```
"builtin.module"() ({
  "func.func"() <{function_type = (memref<16x16xi8>, memref<16x16xi8, strided<[1, 16]>>, memref<16x16xi32>) -> (), sym_name = "matmul_16x16"}> ({
  ^bb0(%arg0: memref<16x16xi8>, %arg1: memref<16x16xi8, strided<[1, 16]>>, %arg2: memref<16x16xi32>):
    %0 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "linalg.generic"(%arg0, %arg1, %0, %0, %arg2) 
    <{indexing_maps = [
    affine_map<(d0, d1, d2) -> (d0, d2)>, 
    affine_map<(d0, d1, d2) -> (d2, d1)>, 
    affine_map<(d0, d1, d2) -> ()>, 
    affine_map<(d0, d1, d2) -> ()>, 
    affine_map<(d0, d1, d2) -> (d0, d1)>], 
    iterator_types = [
    #linalg.iterator_type<parallel>, 
    #linalg.iterator_type<parallel>, 
    #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 4, 1>}> ({
    ^bb0(%arg3: i8, %arg4: i8, %arg5: i32, %arg6: i32, %arg7: i32):
      %1 = "arith.extsi"(%arg3) : (i8) -> i32
      %2 = "arith.subi"(%1, %arg5) : (i32, i32) -> i32
      %3 = "arith.extsi"(%arg4) : (i8) -> i32
      %4 = "arith.subi"(%3, %arg6) : (i32, i32) -> i32
      %5 = "arith.muli"(%2, %4) : (i32, i32) -> i32
      %6 = "arith.addi"(%arg7, %5) : (i32, i32) -> i32
      "linalg.yield"(%6) : (i32) -> ()
    }) : (memref<16x16xi8>, memref<16x16xi8, strided<[1, 16]>>, i32, i32, memref<16x16xi32>) -> ()
    "func.return"() : () -> ()
  }) {llvm.emit_c_interface = true} : () -> ()
}) : () -> ()
```

## II. ZigZag Tiling Scheme

As a workload object, the input 16x16 matmul workload looks like

```
- id: 0 
  name: matmul_16_x_16  # name can be used to specify mapping
  operator_type: MatMul  # operator_type can be used to specify mapping
  equation: O[a][b]+=I[a][c]*W[c][b]
  dimension_relations: []
  loop_dims: [A,B,C]
  loop_sizes: [16, 16, 16]
  operand_precision:
    W: 8
    I: 8
    O: 32
    O_final: 32
  operand_source:
    I: 0
    W: 0
```

Given this workload, an empty mapping file, and snax-gemm hardware description, 

ZigZag recommends the following tiling scheme:

``` 
Loop ordering for matmul_16_x_16
==============================================================
Temporal Loops                  O         W         I         
==============================================================
for A in [0, 2):                l1        l1        l1        
--------------------------------------------------------------
  for B in [0, 2):              l1        l1        l1        
--------------------------------------------------------------
    for C in [0, 2):            reg_O     l1        l1        
--------------------------------------------------------------
==============================================================
Spatial Loops                                                 
==============================================================
      parfor B in [0, 8):                                     
--------------------------------------------------------------
      parfor A in [0, 8):                                     
--------------------------------------------------------------
      parfor C in [0, 8):                                     
--------------------------------------------------------------
```

For a more detailed explanation of using ZigZag, go [here](https://github.com/EmilySillars/zigzag/blob/manual-examples/modeling-gemm-with-zigzag.md#i-matmul-16-x-16).

## III. Output MLIR

Recall: `O[a][b]+=I[a][c]*W[c][b]`

Plans for next steps:

1. Write out the C-ish pseudocode for this zigzag tiling scheme

```
copyFromL3toL1(weight[0][0], shape[16][16])
copyFromL3toL1(input[0][0], shape[16][16])
copyFromL3toL1(output[0][0], shape[16][16])

a0_bk_sz = 8;
b0_bk_sz = 8;
c0_bk_sz = 8;

for (a0 = 0; a0 < 2; a0++){
for (b0 = 0; b0 < 2; b0++){
for (c0 = 0; c0 < 2; c0++){
// following inner loops should execute in parallel
for (a1 = 0; a1 < 8; a1++){
for (b1 = 0; b1 < 8; b1++){
for (c1 = 0; c1 < 8; c1++){
a = a0*a0_bk_sz + a1;
b = b0*b0_bk_sz + b1;
c = c0*c0_bk_sz + c1;
output[a][b] = input[a][c]*weight[c][b]
}
}
}		
}
}
}
```

1. Write out the C-pseudocode for this zigzag tilingscheme when spatial loops are left for the accelerator to handle

```
copyFromL3toL1(weight[0][0], shape[16][16])
copyFromL3toL1(input[0][0], shape[16][16])
copyFromL3toL1(output[0][0], shape[16][16])

a0_bk_sz = 8;
b0_bk_sz = 8;
c0_bk_sz = 8;

for (a0 = 0; a0 < 2; a0++){
for (b0 = 0; b0 < 2; b0++){
for (c0 = 0; c0 < 2; c0++){
// copyFromL3toL1(weight[c2*c2_bk_sz][0], shape[c2_bk_sz][104])
// %slice_W_L3 = memref.subview %weight[%slice_W_L3_offset, %zero][26,104][1,1]
weight_submatrix = memref.subview %weight_L1[0,0]
// following inner loops should execute in parallel
for (a1 = 0; a1 < 8; a1++){
for (b1 = 0; b1 < 8; b1++){
for (c1 = 0; c1 < 8; c1++){
a = a0*a0_bk_sz + a1;
b = b0*b0_bk_sz + b1;
c = c0*c0_bk_sz + c1;
O[a][b] = I[a][c]*W[c][b]
}
}
}		
}
}
}
```



1. Try to write the execution of this tiling scheme using base_gemm, source code here: https://github.com/KULeuven-MICAS/snax_cluster/blob/e0c51a37b0e9048f3667d720bb982d3bfc98b3c0/target/snitch_cluster/sw/snax/gemm/src/snax-gemm-lib.c#L26

   ```
   void base_gemm(uint8_t m, uint8_t k, uint8_t n, int8_t* A, int8_t* B,
                  int8_t subtraction_a, int8_t subtraction_b, int32_t* C_cpu,
                  bool clear) {
       for (int i = 0; i < m; i++) {
           for (int j = 0; j < n; j++) {
               // clear memory first before start matrix multiplication
               // to accumulate in K dimension
               if (clear == true) {
                   C_cpu[i * n + j] = 0;
               }
               for (int s = 0; s < k; s++) {
                   C_cpu[i * n + j] =
                       C_cpu[i * n + j] +
                       ((int32_t)A[i * k + s] - (int32_t)subtraction_a) *
                           ((int32_t)B[s + j * k] - (int32_t)subtraction_b);
               }
           }
       }
   };
   ```

   - Look at ZigZag-generated diagram, then try drawing my own diagram of gemm accelerator if it doesn't make sense

   - Ask Arne about gemm hardware description - can he confirm my diagram is accurate?

   - Can I draw tiling diagram after hardware description?

   - Can I verify that output needs to be in weird order to be correct when using gemm accelerator?

   - How do I know that the output is register is used how ZigZag recommends??


4. a. take notes on base_gemm

   b. take notes on block_gemm (i did this)

   c. take notes on batch_gemm (i asked questions about this because I'm confused)

5. I think we really only need to use the base_gemm function call, right? 'cause ZigZag ensures all the tiles fit?

Understand every part of the function calls

```
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

...

uint32_t size_setting = gen_size_config(Batch, M_param, K_param, N_param);

set_batch_gemm(size_setting, a_ptr, b_ptr, 0, c_ptr, strideInnermostA,
               strideInnermostB, strideInnermostC, ldA, ldB, ldC, strideA,
               strideB, strideC);

start_batch_gemm();
```

What kind of matmul does this accelerator perform? It seems like the most it can perform is 512 macs in parallel, right? But we only have 256-element matrices!

For ZigZag scheme, it seems like we cut matrices A, B, and C in half, and perform one matmul on half of A, and then another matmul on the other half of A?? And the entire peice of B and C are used in both of these two matmuls????

Question: What kind of tiling scheme does batch_gemm perform? Is it the same as ZigZag (I don't think so...??)

## IV. Run on SNAX + gemm

### Use built-in SNAX-MLIR flow past this point
