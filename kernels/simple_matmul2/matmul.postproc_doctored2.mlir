builtin.module {
  func.func @simple_matmul(%arg0 : memref<16x16xi8>, %arg1 : memref<16x16xi8, strided<[1, 16]>>, %arg2 : memref<16x16xi32, strided<[16, 1]>>) {
    %0 = arith.constant 0 : i32
    %1 = arith.constant 0 : index
    %2 = arith.constant 1 : index
    %3 = arith.constant 16 : index
    %4 = arith.constant 2 : index
    scf.for %arg3 = %1 to %3 step %4 {
      scf.for %arg4 = %1 to %2 step %2 {
        scf.for %arg5 = %1 to %3 step %4 {
          // %5 = "memref.subview"(%arg0) <{"operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<16x16xi8>) -> memref<2x16xi8, strided<[16, 1], offset: ?>>
          // %6 = "memref.subview"(%arg1) <{ "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<16x16xi8, strided<[1, 16]>>) -> memref<16x2xi8, strided<[1, 16], offset: ?>>
          // %7 = "memref.subview"(%arg2) <{"operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<16x16xi32, strided<[16, 1]>>) -> memref<2x2xi32, strided<[16, 1], offset: ?>>
          
          // these subview instructions come from matmul.preprocfinal.mlir 
          // when running run_simple_matmul2.sh on matmul-transformed.mlir
          %5 = memref.subview %arg0[%arg5, %arg4] [2, 16] [1, 1] : memref<16x16xi8> to memref<2x16xi8, strided<[16, 1], offset: ?>>
          %6 = memref.subview %arg1[%arg4, %arg3] [16, 2] [1, 1] : memref<16x16xi8, strided<[1, 16]>> to memref<16x2xi8, strided<[1, 16], offset: ?>>
          %7 = memref.subview %arg2[%arg5, %arg3] [2, 2] [1, 1] : memref<16x16xi32, strided<[16, 1]>> to memref<2x2xi32, strided<[16, 1], offset: ?>>
         
         
          // %5 = "memref.subview"(%arg0) <{"static_offsets" = array<i64: ?, ?>, "static_sizes" = array<i64: 2, 16>, "static_strides" = array<i64: 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<16x16xi8>) -> memref<2x16xi8, strided<[16, 1], offset: ?>>
          // %6 = "memref.subview"(%arg1) <{"static_offsets" = array<i64: ?, ?>, "static_offsets" = array<i64: ?, ?>, "static_sizes" = array<i64: 16, 2>, "static_strides" = array<i64: 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<16x16xi8, strided<[1, 16]>>) -> memref<16x2xi8, strided<[1, 16], offset: ?>>
          // %7 = "memref.subview"(%arg2) <{"static_offsets" = array<i64: ?, ?>, "static_sizes" = array<i64: 2, 2>, "static_strides" = array<i64: 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<16x16xi32, strided<[16, 1]>>) -> memref<2x2xi32, strided<[16, 1], offset: ?>>
          %8 = "memref.memory_space_cast"(%5) : (memref<2x16xi8, strided<[16, 1], offset: ?>>) -> memref<2x16xi8, strided<[16, 1], offset: ?>, 1 : i32>
          %9 = "memref.memory_space_cast"(%6) : (memref<16x2xi8, strided<[1, 16], offset: ?>>) -> memref<16x2xi8, strided<[1, 16], offset: ?>, 1 : i32>
          %10 = "memref.memory_space_cast"(%7) : (memref<2x2xi32, strided<[16, 1], offset: ?>>) -> memref<2x2xi32, strided<[16, 1], offset: ?>, 1 : i32>
          %11 = arith.constant 2 : index
          %12 = arith.constant 16 : index
          %13 = arith.constant 8 : index
          %14 = arith.divui %11, %13 : index
          %15 = arith.constant 8 : index
          %16 = arith.constant 8 : index
          %17 = arith.divui %12, %16 : index
          %18 = arith.constant 8 : index
          %19 = arith.constant 256 : index
          %20 = arith.constant 1 : index
          %21 = arith.constant 256 : index
          %22 = arith.constant 8 : index
          %23 = arith.muli %17, %19 : index
          %24 = arith.constant 1 : index
          %25 = arith.constant 0 : index
          %26 = arith.subi %14, %24 : index
          %27 = arith.muli %26, %23 : index
          %28 = arith.addi %25, %27 : index
          %29 = arith.subi %15, %24 : index
          %30 = arith.muli %29, %22 : index
          %31 = arith.addi %28, %30 : index
          %32 = arith.subi %17, %24 : index
          %33 = arith.muli %32, %21 : index
          %34 = arith.addi %31, %33 : index
          %35 = arith.subi %18, %24 : index
          %36 = arith.muli %35, %20 : index
          %37 = arith.addi %34, %36 : index
          %38 = arith.constant 1 : index
          %39 = arith.addi %37, %38 : index
          %40 = arith.muli %24, %39 : index
          %41 = arith.constant 64 : index
          %42 = func.call @snax_alloc_l1(%40, %41) : (index, index) -> !llvm.ptr
          %43 = "llvm.load"(%42) : (!llvm.ptr) -> !llvm.struct<(!llvm.ptr, !llvm.ptr)>
          %44 = "llvm.extractvalue"(%43) <{"position" = array<i64: 0>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr)>) -> !llvm.ptr
          %45 = "llvm.extractvalue"(%43) <{"position" = array<i64: 1>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr)>) -> !llvm.ptr
          %46 = "llvm.mlir.undef"() : () -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
          %47 = "llvm.insertvalue"(%46, %44) <{"position" = array<i64: 0>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, !llvm.ptr) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
          %48 = "llvm.insertvalue"(%47, %45) <{"position" = array<i64: 1>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, !llvm.ptr) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
          %49 = arith.constant 0 : i32
          %50 = "llvm.insertvalue"(%48, %49) <{"position" = array<i64: 2>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
          %51 = builtin.unrealized_conversion_cast %11 : index to i32
          %52 = "llvm.insertvalue"(%50, %51) <{"position" = array<i64: 3, 0>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
          %53 = builtin.unrealized_conversion_cast %12 : index to i32
          %54 = "llvm.insertvalue"(%52, %53) <{"position" = array<i64: 3, 1>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
          %55 = builtin.unrealized_conversion_cast %54 : !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)> to memref<2x16xi8>
          %56 = arith.constant 16 : index
          %57 = arith.constant 2 : index
          %58 = arith.constant 8 : index
          %59 = arith.divui %56, %58 : index
          %60 = arith.constant 8 : index
          %61 = arith.constant 8 : index
          %62 = arith.divui %57, %61 : index
          %63 = arith.constant 8 : index
          %64 = arith.constant 256 : index
          %65 = arith.constant 8 : index
          %66 = arith.muli %59, %64 : index
          %67 = arith.constant 1 : index
          %68 = arith.constant 256 : index
          %69 = arith.constant 1 : index
          %70 = arith.constant 0 : index
          %71 = arith.subi %59, %69 : index
          %72 = arith.muli %71, %68 : index
          %73 = arith.addi %70, %72 : index
          %74 = arith.subi %60, %69 : index
          %75 = arith.muli %74, %67 : index
          %76 = arith.addi %73, %75 : index
          %77 = arith.subi %62, %69 : index
          %78 = arith.muli %77, %66 : index
          %79 = arith.addi %76, %78 : index
          %80 = arith.subi %63, %69 : index
          %81 = arith.muli %80, %65 : index
          %82 = arith.addi %79, %81 : index
          %83 = arith.constant 1 : index
          %84 = arith.addi %82, %83 : index
          %85 = arith.muli %69, %84 : index
          %86 = arith.constant 64 : index
          %87 = func.call @snax_alloc_l1(%85, %86) : (index, index) -> !llvm.ptr
          %88 = "llvm.load"(%87) : (!llvm.ptr) -> !llvm.struct<(!llvm.ptr, !llvm.ptr)>
          %89 = "llvm.extractvalue"(%88) <{"position" = array<i64: 0>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr)>) -> !llvm.ptr
          %90 = "llvm.extractvalue"(%88) <{"position" = array<i64: 1>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr)>) -> !llvm.ptr
          %91 = "llvm.mlir.undef"() : () -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
          %92 = "llvm.insertvalue"(%91, %89) <{"position" = array<i64: 0>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, !llvm.ptr) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
          %93 = "llvm.insertvalue"(%92, %90) <{"position" = array<i64: 1>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, !llvm.ptr) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
          %94 = arith.constant 0 : i32
          %95 = "llvm.insertvalue"(%93, %94) <{"position" = array<i64: 2>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
          %96 = builtin.unrealized_conversion_cast %56 : index to i32
          %97 = "llvm.insertvalue"(%95, %96) <{"position" = array<i64: 3, 0>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
          %98 = builtin.unrealized_conversion_cast %57 : index to i32
          %99 = "llvm.insertvalue"(%97, %98) <{"position" = array<i64: 3, 1>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
          %100 = builtin.unrealized_conversion_cast %99 : !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)> to memref<16x2xi8>
          %101 = arith.constant 2 : index
          %102 = arith.constant 2 : index
          %103 = arith.constant 8 : index
          %104 = arith.divui %101, %103 : index
          %105 = arith.constant 8 : index
          %106 = arith.constant 8 : index
          %107 = arith.divui %102, %106 : index
          %108 = arith.constant 8 : index
          %109 = arith.constant 256 : index
          %110 = arith.constant 4 : index
          %111 = arith.constant 256 : index
          %112 = arith.constant 32 : index
          %113 = arith.muli %107, %109 : index
          %114 = arith.constant 1 : index
          %115 = arith.constant 0 : index
          %116 = arith.subi %104, %114 : index
          %117 = arith.muli %116, %113 : index
          %118 = arith.addi %115, %117 : index
          %119 = arith.subi %105, %114 : index
          %120 = arith.muli %119, %112 : index
          %121 = arith.addi %118, %120 : index
          %122 = arith.subi %107, %114 : index
          %123 = arith.muli %122, %111 : index
          %124 = arith.addi %121, %123 : index
          %125 = arith.subi %108, %114 : index
          %126 = arith.muli %125, %110 : index
          %127 = arith.addi %124, %126 : index
          %128 = arith.constant 4 : index
          %129 = arith.addi %127, %128 : index
          %130 = arith.muli %114, %129 : index
          %131 = arith.constant 64 : index
          %132 = func.call @snax_alloc_l1(%130, %131) : (index, index) -> !llvm.ptr
          %133 = "llvm.load"(%132) : (!llvm.ptr) -> !llvm.struct<(!llvm.ptr, !llvm.ptr)>
          %134 = "llvm.extractvalue"(%133) <{"position" = array<i64: 0>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr)>) -> !llvm.ptr
          %135 = "llvm.extractvalue"(%133) <{"position" = array<i64: 1>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr)>) -> !llvm.ptr
          %136 = "llvm.mlir.undef"() : () -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
          %137 = "llvm.insertvalue"(%136, %134) <{"position" = array<i64: 0>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, !llvm.ptr) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
          %138 = "llvm.insertvalue"(%137, %135) <{"position" = array<i64: 1>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, !llvm.ptr) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
          %139 = arith.constant 0 : i32
          %140 = "llvm.insertvalue"(%138, %139) <{"position" = array<i64: 2>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
          %141 = builtin.unrealized_conversion_cast %101 : index to i32
          %142 = "llvm.insertvalue"(%140, %141) <{"position" = array<i64: 3, 0>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
          %143 = builtin.unrealized_conversion_cast %102 : index to i32
          %144 = "llvm.insertvalue"(%142, %143) <{"position" = array<i64: 3, 1>}> : (!llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>, i32) -> !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)>
          %145 = builtin.unrealized_conversion_cast %144 : !llvm.struct<(!llvm.ptr, !llvm.ptr, i32, !llvm.array<2 x i32>, !llvm.array<2 x i32>)> to memref<2x2xi32>
          %146 = "memref.extract_aligned_pointer_as_index"(%6) : (memref<16x2xi8, strided<[1, 16], offset: ?>>) -> index
          %147 = "memref.extract_aligned_pointer_as_index"(%100) : (memref<16x2xi8>) -> index
          %148 = arith.constant 0 : index
          %149 = "memref.dim"(%6, %148) : (memref<16x2xi8, strided<[1, 16], offset: ?>>, index) -> index
          %150 = arith.constant 1 : index
          %151 = "memref.dim"(%6, %150) : (memref<16x2xi8, strided<[1, 16], offset: ?>>, index) -> index
          %152 = arith.constant 8 : index
          %153 = arith.divui %149, %152 : index
          %154 = arith.constant 8 : index
          %155 = arith.constant 8 : index
          %156 = arith.divui %151, %155 : index
          %157 = arith.constant 8 : index
          %158 = arith.constant 128 : index
          %159 = arith.constant 16 : index
          %160 = arith.constant 128 : index
          %161 = arith.constant 1 : index
          %162 = arith.constant 8 : index
          %163 = arith.constant 256 : index
          %164 = arith.constant 8 : index
          %165 = arith.muli %153, %163 : index
          %166 = arith.constant 1 : index
          %167 = arith.constant 256 : index
          %168 = arith.constant 8 : index
          %169 = arith.constant 0 : index
          %170 = arith.constant 1 : index
          scf.for %171 = %169 to %153 step %170 {
            %172 = arith.muli %171, %162 : index
            %173 = arith.addi %146, %172 : index
            %174 = arith.muli %171, %167 : index
            %175 = arith.addi %147, %174 : index
            scf.for %176 = %169 to %156 step %170 {
              %177 = arith.muli %176, %160 : index
              %178 = arith.addi %173, %177 : index
              %179 = arith.muli %176, %165 : index
              %180 = arith.addi %175, %179 : index
              func.call @snax_dma_2d_transfer(%178, %180, %168, %159, %164, %157) : (index, index, index, index, index, index) -> ()
            }
          }
          %181 = "memref.extract_aligned_pointer_as_index"(%5) : (memref<2x16xi8, strided<[16, 1], offset: ?>>) -> index
          %182 = "memref.extract_aligned_pointer_as_index"(%55) : (memref<2x16xi8>) -> index
          %183 = arith.constant 0 : index
          %184 = "memref.dim"(%5, %183) : (memref<2x16xi8, strided<[16, 1], offset: ?>>, index) -> index
          %185 = arith.constant 1 : index
          %186 = "memref.dim"(%5, %185) : (memref<2x16xi8, strided<[16, 1], offset: ?>>, index) -> index
          %187 = arith.constant 8 : index
          %188 = arith.divui %184, %187 : index
          %189 = arith.constant 8 : index
          %190 = arith.constant 8 : index
          %191 = arith.divui %186, %190 : index
          %192 = arith.constant 8 : index
          %193 = arith.constant 128 : index
          %194 = arith.constant 1 : index
          %195 = arith.constant 8 : index
          %196 = arith.constant 16 : index
          %197 = arith.constant 128 : index
          %198 = arith.constant 256 : index
          %199 = arith.constant 1 : index
          %200 = arith.constant 256 : index
          %201 = arith.constant 8 : index
          %202 = arith.muli %191, %198 : index
          %203 = arith.constant 8 : index
          %204 = arith.constant 0 : index
          %205 = arith.constant 1 : index
          scf.for %206 = %204 to %188 step %205 {
            %207 = arith.muli %206, %197 : index
            %208 = arith.addi %181, %207 : index
            %209 = arith.muli %206, %202 : index
            %210 = arith.addi %182, %209 : index
            scf.for %211 = %204 to %191 step %205 {
              %212 = arith.muli %211, %195 : index
              %213 = arith.addi %208, %212 : index
              %214 = arith.muli %211, %200 : index
              %215 = arith.addi %210, %214 : index
              func.call @snax_dma_2d_transfer(%213, %215, %203, %196, %201, %189) : (index, index, index, index, index, index) -> ()
            }
          }
          func.call @snax_cluster_hw_barrier() : () -> ()
          %216 = "memref.cast"(%55) : (memref<2x16xi8>) -> memref<?x?xi8>
          %217 = "memref.cast"(%100) : (memref<16x2xi8>) -> memref<?x?xi8>
          %218 = "memref.cast"(%145) : (memref<2x2xi32>) -> memref<?x?xi32>
          func.call @snax_qgemm(%216, %217, %0, %0, %218) : (memref<?x?xi8>, memref<?x?xi8>, i32, i32, memref<?x?xi32>) -> ()
          func.call @snax_cluster_hw_barrier() : () -> ()
          %219 = "memref.extract_aligned_pointer_as_index"(%145) : (memref<2x2xi32>) -> index
          %220 = "memref.extract_aligned_pointer_as_index"(%7) : (memref<2x2xi32, strided<[16, 1], offset: ?>>) -> index
          %221 = arith.constant 0 : index
          %222 = "memref.dim"(%145, %221) : (memref<2x2xi32>, index) -> index
          %223 = arith.constant 1 : index
          %224 = "memref.dim"(%145, %223) : (memref<2x2xi32>, index) -> index
          %225 = arith.constant 8 : index
          %226 = arith.divui %222, %225 : index
          %227 = arith.constant 8 : index
          %228 = arith.constant 8 : index
          %229 = arith.divui %224, %228 : index
          %230 = arith.constant 8 : index
          %231 = arith.constant 256 : index
          %232 = arith.constant 4 : index
          %233 = arith.constant 256 : index
          %234 = arith.constant 32 : index
          %235 = arith.muli %229, %231 : index
          %236 = arith.constant 128 : index
          %237 = arith.constant 1 : index
          %238 = arith.constant 8 : index
          %239 = arith.constant 16 : index
          %240 = arith.constant 128 : index
          %241 = arith.constant 1 : index
          %242 = arith.constant 0 : index
          %243 = arith.constant 1 : index
          scf.for %244 = %242 to %230 step %243 {
            %245 = arith.muli %244, %232 : index
            %246 = arith.addi %219, %245 : index
            %247 = arith.muli %244, %237 : index
            %248 = arith.addi %220, %247 : index
            scf.for %249 = %242 to %226 step %243 {
              %250 = arith.muli %249, %235 : index
              %251 = arith.addi %246, %250 : index
              %252 = arith.muli %249, %240 : index
              %253 = arith.addi %248, %252 : index
              scf.for %254 = %242 to %229 step %243 {
                %255 = arith.muli %254, %233 : index
                %256 = arith.addi %251, %255 : index
                %257 = arith.muli %254, %238 : index
                %258 = arith.addi %253, %257 : index
                func.call @snax_dma_2d_transfer(%256, %258, %241, %234, %239, %227) : (index, index, index, index, index, index) -> ()
              }
            }
          }
        }
      }
    }
    func.return
  }
  func.func private @snax_qgemm(memref<?x?xi8>, memref<?x?xi8>, i32, i32, memref<?x?xi32>) -> ()
  func.func private @snax_dma_2d_transfer(index, index, index, index, index, index) -> ()
  func.func private @snax_cluster_hw_barrier() -> ()
  func.func private @snax_alloc_l1(index, index) -> !llvm.ptr
}

