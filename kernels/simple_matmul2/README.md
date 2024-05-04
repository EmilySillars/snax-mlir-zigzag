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
"builtin.module"() ({
"func.func"() <{function_type = (memref<16x16xi8>, memref<16x16xi8, strided<[1, 16]>>, memref<16x16xi32, strided<[16,1]>>) -> (), sym_name = "simple_matmul"}> ({
  ^bb0(%arg0: memref<16x16xi8>, %arg1: memref<16x16xi8, strided<[1,16]>>, %arg2: memref<16x16xi32, strided<[16,1]>>):
    %0 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %zero = arith.constant 0 : index
    %one = arith.constant 1: index
    %sixteen = arith.constant 16 : index
    %two = arith.constant 2 : index

    // enter scf nested FOR LOOP
    scf.for %k = %zero to %sixteen step %two iter_args() -> () {
    scf.for %j = %zero to %one step %one iter_args() -> () {    
    scf.for %i = %zero to %sixteen step %two iter_args() -> () {    
    // pull out left tile
    %leftTile = memref.subview %arg0[%i,%j][2,16][1,1] : memref<16x16xi8> to memref<2x16xi8, strided<[16, 1], offset: ?>>
    %leftTileCasted = memref.cast %leftTile : memref<2x16xi8, strided<[16, 1], offset: ?>> to memref<2x16xi8>    
    // pull out right tile
    %rightTile = memref.subview %arg1[%j,%k][16,2][1,1] : memref<16x16xi8, strided<[1,16]>> to memref<16x2xi8, strided<[1,16], offset: ?>>
    %rightTileCasted = memref.cast %rightTile : memref<16x2xi8, strided<[1,16], offset: ?>> to memref<16x2xi8, strided<[1,16]>>
    // pull out output tile
    %outputTile = memref.subview %arg2[%i,%k][2,2][1,1] : memref<16x16xi32, strided<[16,1]>> to memref<2x2xi32, strided<[16,1], offset: ?>>
    %outputTileCasted = memref.cast %outputTile : memref<2x2xi32, strided<[16,1], offset: ?>> to memref<2x2xi32, strided<[16,1]>>  
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
      "linalg.yield"(%6) : (i32) -> ()
    }) : (memref<2x16xi8>, memref<16x2xi8, strided<[1,16]>>, i32, i32, memref<2x2xi32, strided<[16,1]>>) -> ()
    } // end of i for
    } // end of j for
    } // end of k for
    "func.return"() : () -> ()
  }) : () -> ()

}) : () -> ()
```

## Running the MLIR

### mlir-cpu-runner

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
// a 16x16 matrix filled with the value 3, except for elt at (13,13), which is set to 87
```

Current output C with mlir-cpu-runner:

```
untiled:
Unranked Memref base@ = 0xd1143c0 rank = 2 offset = 0 sizes = [16, 16] strides = [16, 1] data = 
[[408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   1584,   408,   408], 
 [408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   1584,   408,   408], 
 [408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   1584,   408,   408], 
 [408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   1584,   408,   408], 
 [408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   1584,   408,   408], 
 [660,   660,   660,   660,   660,   660,   660,   660,   660,   660,   660,   660,   660,   1836,   660,   660], 
 [408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   1584,   408,   408], 
 [408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   1584,   408,   408], 
 [642,   642,   642,   642,   642,   642,   642,   642,   642,   642,   642,   642,   642,   1818,   642,   642], 
 [408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   1584,   408,   408], 
 [408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   1584,   408,   408], 
 [408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   1584,   408,   408], 
 [408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   1584,   408,   408], 
 [408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   1584,   408,   408], 
 [408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   1584,   408,   408], 
 [408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   1584,   408,   408]]
 
 tiled:
Unranked Memref base@ = 0xd001cc0 rank = 2 offset = 0 sizes = [16, 16] strides = [16, 1] data = 
[[408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   1584,   408,   408], 
 [408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   1584,   408,   408], 
 [408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   1584,   408,   408], 
 [408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   1584,   408,   408], 
 [408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   1584,   408,   408], 
 [660,   660,   660,   660,   660,   660,   660,   660,   660,   660,   660,   660,   660,   1836,   660,   660], 
 [408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   1584,   408,   408], 
 [408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   1584,   408,   408], 
 [642,   642,   642,   642,   642,   642,   642,   642,   642,   642,   642,   642,   642,   1818,   642,   642], 
 [408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   1584,   408,   408], 
 [408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   1584,   408,   408], 
 [408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   1584,   408,   408], 
 [408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   1584,   408,   408], 
 [408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   1584,   408,   408], 
 [408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   1584,   408,   408], 
 [408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   408,   1584,   408,   408]]
```

### running into errors trying to run on snax...

```
sh run_simple_matmul.sh matmul-transformed.mlir
```

Execution Hangs :(











sed -i 's/-9223372036854775808/0/g'

```
sed -i 's/old-word/new-word/g' *.txt
```

1. xDSL parse error

```
...
  File "/opt/python3.11/lib/python3.11/site-packages/xdsl/parser/attribute_parser.py", line 1298, in _parse_optional_builtin_type
    return self._parse_optional_builtin_parametrized_type()
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/python3.11/lib/python3.11/site-packages/xdsl/parser/attribute_parser.py", line 401, in _parse_optional_builtin_parametrized_type
    res = builtin_parsers.get(name, unimplemented)()
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/python3.11/lib/python3.11/site-packages/xdsl/parser/attribute_parser.py", line 540, in _parse_memref_attrs
    self.raise_error(
  File "/opt/python3.11/lib/python3.11/site-packages/xdsl/parser/base_parser.py", line 98, in raise_error
    raise ParseError(at_position, msg)
xdsl.utils.exceptions.ParseError: matmul.hoodle.mlir:14:119
          %8 = memref.alloc() {"alignment" = 64 : i64} : memref<2x16xi8, #tsl.tsl<[?, 8] -> (?, 8), [?, 8] -> (256, 1)>>
                                                                                                                       ^
                                                                                                                       Cannot decide if the given attribute is a layout or a memory space!


The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/repo/runtime//../compiler/snax-opt", line 8, in <module>
    sys.exit(main())
             ^^^^^^
  File "/repo/compiler/tools/snax_
```

**Solution:** [Joren says](https://xdsl.zulipchat.com/#narrow/stream/368602-Convolve/topic/ZigZag.2FStream.20Integration/near/434308178) to modify the snax-mlir-opt `--set-memory-space` pass to annotate subviews with correct memory space, [which I did here](https://github.com/EmilySillars/snax-mlir-zigzag/blob/d4853d11fe75a1de4d21f562331866e83de59898/compiler/transforms/set_memory_space.py#L115).

2. Even when memory space is annotated, `mlir-opt` does not like the large negative offsets in the `memref.subview`s:

   ```
   matmul.postproc.mlir:11:304: error: expected a 64-bit signed integer or '?'
             %5 = "memref.subview"(%arg0) <{"static_offsets" = array<i64: -9223372036854775808, -9223372036854775808>, "static_sizes" = array<i64: 2, 16>, "static_strides" = array<i64: 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<16x16xi8>) -> memref<2x16xi8, strided<[16, 1], offset: -156797324626531188736>>
   ```

   **Solution:** change all static offsets to question marks like so

   ```
             %5 = "memref.subview"(%arg0) <{"static_offsets" = array<i64: ?, ?>, "static_sizes" = array<i64: 2, 16>, "static_strides" = array<i64: 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<16x16xi8>) -> memref<2x16xi8, strided<[16, 1], offset: -156797324626531188736>>
   ```

   

3. After changing the static offsets to question marks, `mlir-opt` does not like the question marks:

   ```
   matmul.postproc.mlir:11:72: error: expected integer literal
             %5 = "memref.subview"(%arg0) <{"static_offsets" = array<i64: ?, ?>, "static_sizes" = array<i64: 2, 16>, "static_strides" = array<i64: 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<16x16xi8>) -> memref<2x16xi8, strided<[16, 1], offset: -156797324626531188736>>
   ```

   **Possible solution:** use some of the compilation flags from the `mlir-cpu-runner` compiler pass to get rid of the `memref.subview` operations before the other lowering pass can complain about their static offsets.

   **New Error:** xdsl parse error on llvm dialect operation

   ```
   File "/opt/python3.11/lib/python3.11/site-packages/xdsl/parser/base_parser.py", line 98, in raise_error
       raise ParseError(at_position, msg)
   xdsl.utils.exceptions.ParseError: matmul.preprocfinal.mlir:15:21
       %10 = "llvm.icmp"(%9, %1) <{predicate = 2 : i64}> : (i64, i64) -> i1
                        ^
                        unregistered operation llvm.icmp!
   ```

   **Question I have: Should I edit the xDSL parser, or will it balloon into too much work and I should look for another solution to this static_offsets problem?**

4. [After changing the compilation script to use more intermediate steps](https://github.com/EmilySillars/snax-mlir-zigzag/blob/zigzag-to-snax/kernels/simple_matmul2/run_simple_matmul2.sh#L23-L48), the invalid static_offsets are no longer generated. BUT I get a new error:

   ```
     File "/opt/python3.11/lib/python3.11/site-packages/xdsl/tools/command_line_tool.py", line 688, in parse_chunk
       raise Exception("Failed to parse:\n" + e.with_context()) from e
   Exception: Failed to parse:
   out/matmul.preprocfinal.mlir:11:36
             %subview = memref.subview %arg0[%arg5, %arg4] [2, 16] [1, 1] : memref<16x16xi8> to memref<2x16xi8, strided<[16, 1], offset: ?>>
                                       ^^^^^
                                       Operation memref.subview does not have a custom format.
   ```

   Solution???

   - I need to edit [this file](https://github.com/xdslproject/xdsl/blob/main/xdsl/dialects/memref.py#L531) to add dynamic subviews to the xDSL parser!
   - OR I need to add the [ShapedType::kDynamic](https://mlir.llvm.org/docs/Dialects/MemRef/#memrefsubview-memrefsubviewop) attribute to the memref's or memref.subview's meref type? Relevant discourse question [here](https://discourse.llvm.org/t/how-can-i-create-memory-alloc-by-memref-alloc-for-dynamic-dimensions-using-c/69318).
   - OR I need to modify the xDSL parser to it can take in subviews with affine_maps??

   Q: Also, which pass was it that was causing the insertion of invalid static_offsets?

   A: This one, but then executing more passes one by one gets rid of these invalid static offsets :O

   ```
   mlir-opt-17 --pass-pipeline='builtin.module(func.func(tosa-to-linalg-named, tosa-to-tensor, tosa-to-scf, tosa-to-linalg))' \
   --mlir-print-op-generic --mlir-print-local-scope -o out/matmul.preproc1.mlir matmul-transformed.mlir
   ```

   **Solution:** 

   1) Add support for subviews w/o static offsets to xDSL's parser, and to snax-opt's `--set-memory-space` pass.
   2) Running `sh run_simple_matmul_on_doctored.sh` compiles the tiled matmul, but then execution on SNAX hangs. Maybe this is because the GEMM accerator cannot handle more than a simple matmul, and the tiling is inside the matmul function?
   3) Implement tiling logic for host in C, the in MLIR, and run with tiled MLIR code trimmed down to just the kernel getting fed to accelerator.

## errors

```
matmul.postproc.mlir:11:72: error: expected integer literal
          %5 = "memref.subview"(%arg0) <{"static_offsets" = array<i64: ?, ?>, "static_sizes" = array<i64: 2, 16>, "static_strides" = array<i64: 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<16x16xi8>) -> memref<2x16xi8, strided<[16, 1], offset: ?>>
                                                                       ^
```

caused by parser MLIR parser file:

```
ParseResult DenseArrayElementParser::parseIntegerElement(Parser &p) {
  bool isNegative = p.consumeIf(Token::minus);

  // Parse an integer literal as an APInt.
  std::optional<APInt> value;
  StringRef spelling = p.getToken().getSpelling();
  if (p.getToken().isAny(Token::kw_true, Token::kw_false)) {
    if (!type.isInteger(1))
      return p.emitError("expected i1 type for 'true' or 'false' values");
    value = APInt(/*numBits=*/8, p.getToken().is(Token::kw_true),
                  !type.isUnsignedInteger());
    p.consumeToken();
  } else if (p.consumeIf(Token::integer)) {
    value = buildAttributeAPInt(type, isNegative, spelling);
    if (!value)
      return p.emitError("integer constant out of range");
  } else {
    return p.emitError("expected integer literal");
  }
  append(*value);
  return success();
}
```

- But if i had a dynamic subview, I wouldn't have to parse dense arrays, right? because i wouldn't have a static offset field??

If i get rid of the static_offsets field, I get the following error:

```
matmul.postproc.mlir:11:16: error: invalid properties {operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_sizes = array<i64: 2, 16>, static_strides = array<i64: 1, 1>} for op memref.subview: expected key entry for static_offsets in DictionaryAttr to set Properties.
          %5 = "memref.subview"(%arg0) <{ "static_sizes" = array<i64: 2, 16>, "static_strides" = array<i64: 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<16x16xi8>) -> memref<2x16xi8, strided<[16, 1], offset: ?>>
               ^
```

possible file throwing the error: `/home/hoppip/llvm-project-pistachio/mlir/tools/mlir-tblgen/OpDefinitionsGen.cpp`

- but static offsets are first introduced by an xDSL pass, RIGHT???? If i never put them in, will i get MLIR errors?

/home/hoppip/snax-mlir-zigzag/

../../kernels/simple_matmul2/out/matmul.preprocfinal.mlir

```
python3 snax_opt_main.py ../../kernels/simple_matmul2/out/matmul.preprocfinal.mlir

```

