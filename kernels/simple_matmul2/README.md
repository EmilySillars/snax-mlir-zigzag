# QMatMul MLIR ----> ZigZag ----> Manually Transformed MLIR

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

