matmul.postproc.mlir:49:7: error: failed to legalize operation 'builtin.unrealized_conversion_cast' that was explicitly marked illegal
      "memref.copy"(%arg1, %24) : (memref<16x16xi8, strided<[1, 16]>>, memref<16x16xi8, strided<[1, 16]>>) -> ()
      ^
matmul.postproc.mlir:49:7: note: see current operation: %23 = "builtin.unrealized_conversion_cast"(%22) : (memref<16x16xi8, strided<[1, 16]>>) -> !llvm.struct<(ptr, ptr, i32, array<2 x i32>, array<2 x i32>)>


// func.func @simple_matmul(%arg0: memref<16x16xi8>, %arg1: memref<16x16xi8, strided<[1, 16], offset:0>>, %arg2: memref<16x16xi32>) {
//     %c0_i32 = arith.constant 0 : i32
//     linalg.quantized_matmul ins(%arg0, %arg1, %c0_i32, %c0_i32 : memref<16x16xi8>, memref<16x16xi8, strided<[1, 16], offset:0>>, i32, i32) outs(%arg2 : memref<16x16xi32>)
//     return
// }

// "builtin.module"() ({
//   %0:3 = "test.op"() : () -> (memref<16x16xf32>, memref<16x16xf32>, memref<16x16xf32>)
//   "linalg.generic"(%0#0, %0#1, %0#2) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 2, 1>}> ({
//   ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
//     %1 = "arith.mulf"(%arg0, %arg1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
//     %2 = "arith.addf"(%arg2, %1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
//     "linalg.yield"(%2) : (f32) -> ()
//   }) : (memref<16x16xf32>, memref<16x16xf32>, memref<16x16xf32>) -> ()
// }) : () -> ()

// func.func @simple_matmul(%arg10: memref<16x16xi8>, %arg11: memref<16x16xi8, strided<[1, 16], offset:0>>, %arg12: memref<16x16xi32>) {
//     %c0_i32 = arith.constant 0 : i32
//     %0:3 = "test.op"() : () -> (memref<16x16xf32>, memref<16x16xf32>, memref<16x16xf32>)
//   "linalg.generic"(%0#0, %0#1, %0#2) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 2, 1>}> ({
//   ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
//     %1 = "arith.mulf"(%arg0, %arg1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
//     %2 = "arith.addf"(%arg2, %1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
//     "linalg.yield"(%2) : (f32) -> ()
//   }) : (memref<16x16xf32>, memref<16x16xf32>, memref<16x16xf32>) -> ()
//   return
// }

// https://mlir.llvm.org/docs/Tutorials/transform/Ch0/#tiling-and-loop-materialization <-- needed for tiling with scf loops!!
// https://mlir.llvm.org/docs/Dialects/MemRef/#memrefsubview-memrefsubviewop <-- needed to get tiling slices!!!

"builtin.module"() ({
  "func.func"() <{function_type = 
  (memref<16x16xi8>, memref<16x16xi8, strided<[1, 16]>>, memref<16x16xi32>) -> (), sym_name = "simple_matmul"}> ({
  ^bb0(%arg0: memref<16x16xi8>, %arg1: memref<16x16xi8, strided<[1, 16]>>, %arg2: memref<16x16xi32>):
    %0 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "linalg.generic"(%arg0, %arg1, %0, %0, %arg2) 
    <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, 
                       affine_map<(d0, d1, d2) -> (d2, d1)>, 
                       affine_map<(d0, d1, d2) -> ()>, 
                       affine_map<(d0, d1, d2) -> ()>, 
                       affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 4, 1>}> ({
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

------- messing with it again -------
"builtin.module"() ({
  "func.func"() <{function_type = 
  (memref<16x16xi8>, memref<16x16xi8, strided<[1, 16]>>, memref<16x16xi32>) -> (), sym_name = "simple_matmul"}> ({
  ^bb0(%arg0: memref<16x16xi8>, %arg1: memref<16x16xi8, strided<[1, 16]>>, %arg2: memref<16x16xi32>):
    %0 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "linalg.generic"(%arg0, %arg1, %0, %0, %arg2) 

    <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, // memref<16x16xi8>
                       affine_map<(d0, d1, d2) -> (d2, d1)>, // memref<16x16xi8
                       affine_map<(d0, d1, d2) -> ()>,       //                    ====> d0 = 16, d1 = 16, and d2 = 16 (max)
                       affine_map<(d0, d1, d2) -> ()>, 
                       affine_map<(d0, d1, d2) -> (d0, d1)>], //memref<16x16xi32>
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], 
      operandSegmentSizes = array<i32: 4, 1>}> ({
   //    %arg0      %arg1                              %arg2
    ^bb0(%arg3: i8, %arg4: i8, %arg5: i32, %arg6: i32, %arg7: i32):
      %1 = "arith.extsi"(%arg3) : (i8) -> i32             // extend arg3 to 32 bits
      %2 = "arith.subi"(%1, %arg5)  : (i32, i32) -> i32   // %2 = arg3 - 0
      %3 = "arith.extsi"(%arg4) : (i8) -> i32             // extend arg4 to 32 bits
      %4 = "arith.subi"(%3, %arg6)  : (i32, i32) -> i32   // %4 = arg4 - 0
      %5 = "arith.muli"(%2, %4)  : (i32, i32) -> i32      // %5 = arg3[d0][d2] * arg4[d2][d1]
      %6 = "arith.addi"(%arg7, %5) : (i32, i32) -> i32    // %6 = %5 + arg7[d0][d1]
      "linalg.yield"(%6) : (i32) -> ()                    // arg7[d0][d1] = %6
    }) : (memref<16x16xi8>, memref<16x16xi8, strided<[1, 16]>>, i32, i32, memref<16x16xi32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
 -- so i think what we really have is...
for d0; d0 < 16; d0++:
for d1; d1 < 16; d1++;
for d2; d2 < 16; d2++;
  arg7[d0][d1] += arg3[d0][d2] * arg4[d2][d1]; // and this is a MAC!

NOTICE: it's these last 3 lines that make the body of the loop a MAC:
      %5 = "arith.muli"(%2, %4)  : (i32, i32) -> i32      // %5 = arg3[d0][d2] * arg4[d2][d1]
      %6 = "arith.addi"(%arg7, %5) : (i32, i32) -> i32    // %6 = %5 + arg7[d0][d1]
      "linalg.yield"(%6) : (i32) -> ()                    // arg7[d0][d1] = %6
EDGE CASE: what if there are more macs above these three lines? Then zigzag can't handle it, right? 
Maybe zigzag can only recogizze sign extension and subtract by zero pattern?

what would the zigzag python object look like? 
tentatively, like this:
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

If i tile one dimension into 4 chunks, what would the MLIR look like?

for d0_1; d0_1 < 16; d0_1++:
for d0_2; d0_2 < 16; d0_2++:
for d1; d1 < 16; d1++:
for d2; d2 < 16; d2++:
  d0 = d0_1 + d0_2*4
  arg7[d0][d1] += arg3[d0][d2] * arg4[d2][d1]; // and this is a MAC!

"builtin.module"() ({
  "func.func"() <{function_type = (memref<16x16xi8>, memref<16x16xi8, strided<[1, 16]>>, memref<16x16xi32>) -> (), sym_name = "simple_matmul"}> ({
  ^bb0(%arg0: memref<16x16xi8>, %arg1: memref<16x16xi8, strided<[1, 16]>>, %arg2: memref<16x16xi32>):
    %0 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    // need scf loops
    %0 = scf.forall (%i, %j) in (4, 4)
     shared_outs(%shared = %init) -> (tensor<8x16xf32>) {

     }
    // split into tiles
    // feed linalg.generic to accelerator
    //instead od %arg0, %arg1, and %arg2, we need three NEW operands that get updated every scf loop!!!
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

--------------------------------------------------------- vvvvvvvvvv
// A special "multi-for" loop that supports tensor-insertion semantics 
// as opposed to implicit updates. The resulting 8x16 tensor will be produced
// by this loop.
// The trip count of iterators is computed dividing the original tensor size,
// 8x16, by the tile size, 2x8, to obtain 4x2.
// When tensor sizes are dynamic, the trip count computation is emitted as IR
// and is being computed at runtime.
%0 = scf.forall (%i, %j) in (4, 2)
     shared_outs(%shared = %init) -> (tensor<8x16xf32>) {

  // Scale the loop induction variables by the tile sizes.
  %3 = affine.apply affine_map<(d0) -> (d0 * 2)>(%i)
  %4 = affine.apply affine_map<(d0) -> (d0 * 8)>(%j)

  // Take slices of inputs and outputs. Only the "i" and "j" dimensions are sliced.
  %lhs_slice = tensor.extract_slice %lhs[%3, 0] [2, 10] [1, 1]
             : tensor<8x10xf32> to tensor<2x10xf32>
  %rhs_slice = tensor.extract_slice %rhs[0, %4] [10, 8] [1, 1] 
             : tensor<10x16xf32> to tensor<10x8xf32>
  %result_slice = tensor.extract_slice %shared[%3, %4] [2, 8] [1, 1] 
                : tensor<8x16xf32> to tensor<2x8xf32>

  // This is exactly the same operation as before, but now operating on smaller
  // slices of data.
  %partial =  linalg.generic {
  indexing_maps = [affine_map<(i, j, k) -> (i, k)>,
                   affine_map<(i, j, k) -> (k, j)>,
                   affine_map<(i, j, k) -> (i, j)>],
  iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%lhs_slice, %rhs_slice : tensor<2x10xf32>, tensor<10x8xf32>) 
    outs(%result_slice : tensor<2x8xf32>) -> tensor<2x8xf32> {
  ^bb0(%lhs_one: f32, %rhs_one: f32, %init_one: f32):
    %0 = arith.mulf %lhs_one, %rhs_one : f32
    %1 = arith.addf %init_one, %0 : f32
    linalg.yield %1 : f32
  } : tensor<2x8xf32>

  // Terminator for the loop with tensor-insertion semantics. Inserts a slice
  // into a larger tensor, potentially in parallel.
  scf.forall.in_parallel {
    tensor.parallel_insert_slice %partial into %shared[%3, %4] [2, 8] [1, 1]
        : tensor<2x8xf32> into tensor<8x16xf32>
  }
}
--------------------------------------------------------- ^^^^^^^^^^


------------------------------ simpler example first -------------------------------- I can do this!
"builtin.module"() ({
  %0:3 = "test.op"() : () -> (memref<5x7xf32>, memref<7x6xf32>, memref<5x6xf32>)
  "linalg.generic"(%0#0, %0#1, %0#2) 
  <{indexing_maps = [
    affine_map<(d0, d1, d2) -> (d0, d2)>,  // memref<5x7xf32>,
    affine_map<(d0, d1, d2) -> (d2, d1)>,  // memref<7x6xf32>, ====> d0 = 5, d1 = 6, d2 = 7 (max)
    affine_map<(d0, d1, d2) -> (d0, d1)>], // memref<5x6xf32>

    iterator_types = [#linalg.iterator_type<parallel>, 
                      #linalg.iterator_type<parallel>, 
                      #linalg.iterator_type<reduction>],
                      operandSegmentSizes = array<i32: 2, 1>}> 
  
  ({ // here is the body of the linalg operation
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %1 = "arith.mulf"(%arg0, %arg1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    %2 = "arith.addf"(%arg2, %1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "linalg.yield"(%2) : (f32) -> ()
  }) : (memref<5x7xf32>, memref<7x6xf32>, memref<5x6xf32>) -> ()
}) : () -> ()

C = A * B, where C::<5x6xf32>, A::memref<5x7xf32>, and B::memref<7x6xf32>
arg0 = A, arg1 = B, and arg2 = C 

 %1 = "arith.mulf"(%arg0, %arg1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
 translates to
 %1 = arg0[d0][d2] * arg1[d2][d1]

 then, 
  %2 = "arith.addf"(%arg2, %1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
  translates to 
  %2 = arg2[d0][d1] + %1

  then,
  "linalg.yield"(%2) : (f32) -> ()
  translates to
  arg2[d0][d1] = %2 somehow - I think the last argument to the linalg is always the output value??? NO. check the ins and outs params!!!
--- so what we really have is:
for d0; d0 < ?; d0++:
for d1; d1 < ?; d1++;
for d2; d2 < ?; d2++;
  arg2[d0][d1] += arg0[d0][d2] * arg1[d2][d1];
  C[d0][d1] += A[d0][d2] * B[d2][d1];
---------------------------------
for d0; d0 < 5; d0++:
for d1; d1 < 6; d1++;
for d2; d2 < 7; d2++;
  C[d0][d1] += A[d0][d2] * B[d2][d1];

Answer check:
workload = {
    0: {
        "operator_type": "default",
        "equation": "O[d0][d1] += I[d0][d2] * W[d2][d1]",
        "dimension_relations": [],
        "loop_dim_size": {"D0": 5, "D1": 7, "D2": 7},
        "operand_precision": {"O": 32, "O_final": 32, "W": 32, "I": 32},
        "operand_source": {"W": [], "I": []},
        "constant_operands": ["I", "W"],
        "padding": {},
    }
}
