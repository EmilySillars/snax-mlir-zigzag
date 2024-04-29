python3 gendata.py

mlir-opt-17 --pass-pipeline='builtin.module(func.func(tosa-to-linalg-named, tosa-to-tensor, tosa-to-scf, tosa-to-linalg))' \
--mlir-print-op-generic --mlir-print-local-scope -o out/matmul.preproc1.mlir matmul.mlir

mlir-opt-17 --tosa-to-arith="include-apply-rescale" --empty-tensor-to-alloc-tensor -o out/matmul.preproc2.0.mlir out/matmul.preproc1.mlir

# crazy interlude vvv
# mlir-opt-17 -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf out/matmul.preproc1.mlir > out/crazy1.mlir

# mlir-opt-17 -convert-vector-to-llvm --convert-cf-to-llvm out/crazy1.mlir > out/crazy2.mlir

# mlir-opt-17 -expand-strided-metadata -lower-affine -convert-arith-to-llvm \
# --memref-expand -finalize-memref-to-llvm out/crazy2.mlir > out/crazy3.mlir
# crazy interlude ^^^

# continue processing crazy3.mlir
# mlir-opt-17 --test-linalg-transform-patterns="test-generalize-pad-tensor" --linalg-generalize-named-ops \
# --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries allow-return-allocs \
# function-boundary-type-conversion=identity-layout-map" --mlir-print-op-generic --mlir-print-local-scope -o matmul.preproc3.mlir out/crazy3.mlir 
# cat matmul.preproc3.mlir | sed 's/arith.maxf/arith.maximumf/g' | sed 's/arith.minf/arith.minimumf/g' > matmul.preprocfinal.mlir

# preproc2 to preproc3 with intermediate steps vvv
mlir-opt-17 --test-linalg-transform-patterns="test-generalize-pad-tensor" \
-o out/matmul.preproc2.1.mlir out/matmul.preproc2.0.mlir

mlir-opt-17 --linalg-generalize-named-ops \
-o out/matmul.preproc2.2.mlir out/matmul.preproc2.1.mlir

mlir-opt-17 --empty-tensor-to-alloc-tensor \
-o out/matmul.preproc2.3.mlir out/matmul.preproc2.2.mlir

mlir-opt-17 --one-shot-bufferize="bufferize-function-boundaries allow-return-allocs \
function-boundary-type-conversion=identity-layout-map" \
-o out/matmul.preproc2.4.mlir out/matmul.preproc2.3.mlir

mlir-opt-17 --mlir-print-op-generic \
-o out/matmul.preproc2.5.mlir out/matmul.preproc2.4.mlir

mlir-opt-17 --mlir-print-local-scope \
-o out/matmul.preproc3.mlir out/matmul.preproc2.5.mlir
# preproc2 to preproc3 with intermediate steps ^^^

# between preproc2 and preproc3
# mlir-opt-17 --test-linalg-transform-patterns="test-generalize-pad-tensor" --linalg-generalize-named-ops \
# --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries allow-return-allocs \
# function-boundary-type-conversion=identity-layout-map" --mlir-print-op-generic --mlir-print-local-scope \
# -o out/matmul.preproc3.mlir out/matmul.preproc2.0.mlir

cat out/matmul.preproc3.mlir | sed 's/arith.maxf/arith.maximumf/g' | sed 's/arith.minf/arith.minimumf/g' > out/matmul.preprocfinal.mlir

# separating memory attotated IR from non-memory annotated IR
/repo/runtime//../compiler/snax-opt -p dispatch-kernels,set-memory-space -o out/matmul.afterMemAnns.mlir out/matmul.preprocfinal.mlir
/repo/runtime//../compiler/snax-opt -p set-memory-layout,realize-memref-casts -o out/matmul.afterMemAnns2.mlir out/matmul.afterMemAnns.mlir
/repo/runtime//../compiler/snax-opt -p \
insert-sync-barrier,dispatch-regions,linalg-to-library-call,snax-copy-to-dma,memref-to-snax,snax-to-func,clear-memory-space \
-o out/matmul.snax-opt.mlir out/matmul.afterMemAnns2.mlir

cat out/matmul.snax-opt.mlir | sed 's/arith.maximumf/arith.maxf/g' | sed 's/arith.minimumf/arith.minf/g' > out/matmul.postproc.mlir

mlir-opt-17  --convert-linalg-to-loops --convert-scf-to-cf --lower-affine --canonicalize \
--cse --convert-math-to-llvm --llvm-request-c-wrappers --expand-strided-metadata \
--convert-index-to-llvm=index-bitwidth=32 --convert-cf-to-llvm=index-bitwidth=32 \
--convert-arith-to-llvm=index-bitwidth=32 --convert-func-to-llvm='index-bitwidth=32' \
--finalize-memref-to-llvm='use-generic-functions index-bitwidth=32' --canonicalize \
--reconcile-unrealized-casts -o out/matmul.ll.mlir out/matmul.postproc.mlir

mlir-translate-17 --mlir-to-llvmir -o out/matmul.ll out/matmul.ll.mlir
/repo/runtime//tollvm12.py < out/matmul.ll > out/matmul.ll12
/usr/bin/clang-17 -I/opt/snax-gemm/target/snitch_cluster/sw/snax/gemm/include -Wno-unused-command-line-argument -I/opt/snax-gemm/target/snitch_cluster/sw/runtime/rtl-generic/src -I/opt/snax-gemm/target/snitch_cluster/sw/runtime/common -I/opt/snax-gemm/sw/snRuntime/api -I/opt/snax-gemm/sw/snRuntime/src -I/opt/snax-gemm/sw/snRuntime/src/omp/ -I/opt/snax-gemm/sw/snRuntime/api/omp/ -I/opt/snax-gemm/sw/math/arch/riscv64/bits/ -I/opt/snax-gemm/sw/math/arch/generic -I/opt/snax-gemm/sw/math/src/include -I/opt/snax-gemm/sw/math/src/internal -I/opt/snax-gemm/sw/math/include/bits -I/opt/snax-gemm/sw/math/include -I/repo/runtime/include -D__DEFINED_uint64_t --target=riscv32-unknown-elf -mcpu=generic-rv32 -march=rv32imafdzfh -mabi=ilp32d -mcmodel=medany -ftls-model=local-exec -ffast-math -fno-builtin-printf -fno-common -O3 -std=gnu11 -Wall -Wextra -x ir -c out/matmul.ll12 -o matmul.o
/usr/bin/clang-17 -I/opt/snax-gemm/target/snitch_cluster/sw/snax/gemm/include -Wno-unused-command-line-argument -I/opt/snax-gemm/target/snitch_cluster/sw/runtime/rtl-generic/src -I/opt/snax-gemm/target/snitch_cluster/sw/runtime/common -I/opt/snax-gemm/sw/snRuntime/api -I/opt/snax-gemm/sw/snRuntime/src -I/opt/snax-gemm/sw/snRuntime/src/omp/ -I/opt/snax-gemm/sw/snRuntime/api/omp/ -I/opt/snax-gemm/sw/math/arch/riscv64/bits/ -I/opt/snax-gemm/sw/math/arch/generic -I/opt/snax-gemm/sw/math/src/include -I/opt/snax-gemm/sw/math/src/internal -I/opt/snax-gemm/sw/math/include/bits -I/opt/snax-gemm/sw/math/include -I/repo/runtime/include -D__DEFINED_uint64_t --target=riscv32-unknown-elf -mcpu=generic-rv32 -march=rv32imafdzfh -mabi=ilp32d -mcmodel=medany -ftls-model=local-exec -ffast-math -fno-builtin-printf -fno-common -O3 -std=gnu11 -Wall -Wextra -c main.c -o main.o
/usr/bin/clang-17 -I/opt/snax-gemm/target/snitch_cluster/sw/snax/gemm/include -Wno-unused-command-line-argument -I/opt/snax-gemm/target/snitch_cluster/sw/runtime/rtl-generic/src -I/opt/snax-gemm/target/snitch_cluster/sw/runtime/common -I/opt/snax-gemm/sw/snRuntime/api -I/opt/snax-gemm/sw/snRuntime/src -I/opt/snax-gemm/sw/snRuntime/src/omp/ -I/opt/snax-gemm/sw/snRuntime/api/omp/ -I/opt/snax-gemm/sw/math/arch/riscv64/bits/ -I/opt/snax-gemm/sw/math/arch/generic -I/opt/snax-gemm/sw/math/src/include -I/opt/snax-gemm/sw/math/src/internal -I/opt/snax-gemm/sw/math/include/bits -I/opt/snax-gemm/sw/math/include -I/repo/runtime/include -D__DEFINED_uint64_t --target=riscv32-unknown-elf -mcpu=generic-rv32 -march=rv32imafdzfh -mabi=ilp32d -mcmodel=medany -ftls-model=local-exec -ffast-math -fno-builtin-printf -fno-common -O3 -std=gnu11 -Wall -Wextra -c data.c -o data.o
/usr/bin/clang-17 /opt/snax-gemm/target/snitch_cluster/sw/snax/gemm/build/snax-gemm-lib.o -fuse-ld=/usr/bin/ld.lld-17 -L/opt/snitch-llvm/lib/clang/12.0.1/lib/ -L/opt/snitch-llvm/riscv32-unknown-elf/lib/ --target=riscv32-unknown-elf -mcpu=generic-rv32 -march=rv32imafdzfh -mabi=ilp32d -mcmodel=medany -T/opt/snax-gemm/sw/snRuntime/base.ld -L/opt/snax-gemm/target/snitch_cluster/sw/runtime/rtl-generic -L/opt/snax-gemm/target/snitch_cluster/sw/runtime/rtl-generic/build -nostdlib -lclang_rt.builtins-riscv32 -lc -lsnRuntime matmul.o main.o data.o -o matmul.x
rm -fr ./logs/
/opt/snax-gemm-rtl/bin/snitch_cluster.vlt matmul.x


