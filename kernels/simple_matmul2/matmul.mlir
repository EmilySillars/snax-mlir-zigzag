func.func @simple_matmul(%arg0: memref<16x16xi8>, %arg1: memref<16x16xi8, strided<[1, 16], offset:0>>, %arg2: memref<16x16xi32>) {
    %c0_i32 = arith.constant 0 : i32
    linalg.quantized_matmul ins(%arg0, %arg1, %c0_i32, %c0_i32 : memref<16x16xi8>, memref<16x16xi8, strided<[1, 16], offset:0>>, i32, i32) outs(%arg2 : memref<16x16xi32>)
    return
}

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

// "builtin.module"() ({
//   "func.func"() <{function_type = (memref<16x16xi8>, memref<16x16xi8, strided<[1, 16]>>, memref<16x16xi32>) -> (), sym_name = "simple_matmul"}> ({
//   ^bb0(%arg0: memref<16x16xi8>, %arg1: memref<16x16xi8, strided<[1, 16]>>, %arg2: memref<16x16xi32>):
//     %0 = "arith.constant"() <{value = 0 : i32}> : () -> i32
//     "linalg.generic"(%arg0, %arg1, %0, %0, %arg2) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 4, 1>}> ({
//     ^bb0(%arg3: i8, %arg4: i8, %arg5: i32, %arg6: i32, %arg7: i32):
//       %1 = "arith.extsi"(%arg3) : (i8) -> i32
//       %2 = "arith.subi"(%1, %arg5)  : (i32, i32) -> i32
//       %3 = "arith.extsi"(%arg4) : (i8) -> i32
//       %4 = "arith.subi"(%3, %arg6)  : (i32, i32) -> i32
//       %5 = "arith.muli"(%2, %4)  : (i32, i32) -> i32
//       %6 = "arith.addi"(%arg7, %5) : (i32, i32) -> i32
//       "linalg.yield"(%6) : (i32) -> ()
//     }) : (memref<16x16xi8>, memref<16x16xi8, strided<[1, 16]>>, i32, i32, memref<16x16xi32>) -> ()
//     "func.return"() : () -> ()
//   }) : () -> ()
// }) : () -> ()
