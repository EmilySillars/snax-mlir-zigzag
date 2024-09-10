# $1: the name of the mlir file to compile for snax-gemm, without its .mlir file extension.
# $2: the directory where $1 is located
IN="$2"
OUT="$2/out"
OUTACC="$2/out-acc"
mkdir -p $OUT $OUTACC
rm $OUT/* $OUTACC/*

echo "IN is $IN"
echo "OUT is $OUT"
echo "OUTACC is $OUTACC"
echo "$IN/$1.mlir"
ls "$IN/$1.mlir"



mlir-opt-17 --pass-pipeline='builtin.module(func.func(tosa-to-linalg-named, tosa-to-tensor, tosa-to-scf, tosa-to-linalg))' --mlir-print-op-generic --mlir-print-local-scope -o "$OUT/$1.preproc1.mlir" "$IN/$1.mlir"
mlir-opt-17 --tosa-to-arith="include-apply-rescale" --empty-tensor-to-alloc-tensor -o "$OUT/$1.preproc2.mlir" "$OUT/$1.preproc1.mlir"
mlir-opt-17 --test-linalg-transform-patterns="test-generalize-pad-tensor" --linalg-generalize-named-ops --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" --mlir-print-op-generic --mlir-print-local-scope -o "$OUT/$1.preproc3.mlir" "$OUT/$1.preproc2.mlir"
cat "$OUT/$1.preproc3.mlir" | sed 's/arith.maxf/arith.maximumf/g' | sed 's/arith.minf/arith.minimumf/g' > "$OUT/$1.preprocfinal.mlir"
/repo/runtime//../compiler/snax-opt -p dispatch-kernels,set-memory-space,set-memory-layout,realize-memref-casts,reuse-memref-allocs,insert-sync-barrier,dispatch-regions,linalg-to-library-call,snax-copy-to-dma,memref-to-snax,snax-to-func,clear-memory-space -o "$OUT/$1.snax-opt.mlir" "$OUT/$1.preprocfinal.mlir"
cat "$OUT/$1.snax-opt.mlir" | sed 's/arith.maximumf/arith.maxf/g' | sed 's/arith.minimumf/arith.minf/g' > "$OUT/$1.postproc.mlir"
mlir-opt-17 --convert-linalg-to-loops --convert-scf-to-cf --lower-affine --canonicalize --cse --convert-math-to-llvm --llvm-request-c-wrappers --expand-strided-metadata --lower-affine --convert-index-to-llvm=index-bitwidth=32 --convert-cf-to-llvm=index-bitwidth=32 --convert-arith-to-llvm=index-bitwidth=32 --convert-func-to-llvm='index-bitwidth=32' --finalize-memref-to-llvm='use-generic-functions index-bitwidth=32' --canonicalize --reconcile-unrealized-casts -o "$OUT/$1.ll.mlir" "$OUT/$1.postproc.mlir"
mlir-translate-17 --mlir-to-llvmir -o "$OUT/$1.ll" "$OUT/$1.ll.mlir"
/repo/runtime//tollvm12.py < "$OUT/$1.ll" > "$OUT/$1.ll12"
/usr/bin/clang-17 -I/opt/snax-gemm/target/snitch_cluster/sw/snax/gemm/include -Wno-unused-command-line-argument -I/opt/snax-gemm/target/snitch_cluster/sw/runtime/rtl-generic/src -I/opt/snax-gemm/target/snitch_cluster/sw/runtime/common -I/opt/snax-gemm/sw/snRuntime/api -I/opt/snax-gemm/sw/snRuntime/src -I/opt/snax-gemm/sw/snRuntime/src/omp/ -I/opt/snax-gemm/sw/snRuntime/api/omp/ -I/opt/snax-gemm/sw/math/arch/riscv64/bits/ -I/opt/snax-gemm/sw/math/arch/generic -I/opt/snax-gemm/sw/math/src/include -I/opt/snax-gemm/sw/math/src/internal -I/opt/snax-gemm/sw/math/include/bits -I/opt/snax-gemm/sw/math/include -I/repo/runtime/include -D__DEFINED_uint64_t --target=riscv32-unknown-elf -mcpu=generic-rv32 -march=rv32imafdzfh -mabi=ilp32d -mcmodel=medany -ftls-model=local-exec -ffast-math -fno-builtin-printf -fno-builtin-memset -fno-common -O3 -std=gnu11 -Wall -Wextra -x ir -c "$OUT/$1.ll12" -o "$OUT/$1.o"
/usr/bin/clang-17 -I/opt/snax-gemm/target/snitch_cluster/sw/snax/gemm/include -Wno-unused-command-line-argument -I/opt/snax-gemm/target/snitch_cluster/sw/runtime/rtl-generic/src -I/opt/snax-gemm/target/snitch_cluster/sw/runtime/common -I/opt/snax-gemm/sw/snRuntime/api -I/opt/snax-gemm/sw/snRuntime/src -I/opt/snax-gemm/sw/snRuntime/src/omp/ -I/opt/snax-gemm/sw/snRuntime/api/omp/ -I/opt/snax-gemm/sw/math/arch/riscv64/bits/ -I/opt/snax-gemm/sw/math/arch/generic -I/opt/snax-gemm/sw/math/src/include -I/opt/snax-gemm/sw/math/src/internal -I/opt/snax-gemm/sw/math/include/bits -I/opt/snax-gemm/sw/math/include -I/repo/runtime/include -D__DEFINED_uint64_t --target=riscv32-unknown-elf -mcpu=generic-rv32 -march=rv32imafdzfh -mabi=ilp32d -mcmodel=medany -ftls-model=local-exec -ffast-math -fno-builtin-printf -fno-builtin-memset -fno-common -O3 -std=gnu11 -Wall -Wextra -c main.c -o "$OUT/main.o"
# In file included from main.c:5:
# In file included from /repo/runtime/include/snax_rt.h:3:
# In file included from /opt/snax-gemm/target/snitch_cluster/sw/runtime/rtl-generic/src/snrt.h:27:
# /opt/snax-gemm/sw/snRuntime/src/dma.h:204:21: warning: unused variable 'memset_txid' [-Wunused-variable]
#   204 |     snrt_dma_txid_t memset_txid =
#       |                     ^~~~~~~~~~~
# In file included from main.c:5:
# In file included from /repo/runtime/include/snax_rt.h:3:
# In file included from /opt/snax-gemm/target/snitch_cluster/sw/runtime/rtl-generic/src/snrt.h:29:
# /opt/snax-gemm/sw/snRuntime/src/omp/eu.h:198:14: warning: unused variable 'scratch' [-Wunused-variable]
#   198 |     uint32_t scratch;
#       |              ^~~~~~~
# /opt/snax-gemm/sw/snRuntime/src/omp/eu.h:199:14: warning: variable 'nthds' set but not used [-Wunused-but-set-variable]
#   199 |     uint32_t nthds;
#       |              ^
# /opt/snax-gemm/sw/snRuntime/src/omp/eu.h:273:14: warning: unused variable 'nfini' [-Wunused-variable]
#   273 |     unsigned nfini, scratch;
#       |              ^~~~~
# In file included from main.c:5:
# In file included from /repo/runtime/include/snax_rt.h:3:
# In file included from /opt/snax-gemm/target/snitch_cluster/sw/runtime/rtl-generic/src/snrt.h:37:
# /opt/snax-gemm/sw/snRuntime/src/sync.h:139:44: warning: comparison of integers of different signs: 'unsigned int' and 'int' [-Wsign-compare]
#   139 |         for (unsigned int level = 0; level < num_levels; level++) {
#       |                                      ~~~~~ ^ ~~~~~~~~~~
# main.c:47:75: warning: unused parameter 'zpa' [-Wunused-parameter]
#    47 | void _mlir_ciface_snax_gemm(TwoDMemrefI8_t *a, TwoDMemrefI8_t *b, int32_t zpa,
#       |                                                                           ^
# main.c:48:37: warning: unused parameter 'zpb' [-Wunused-parameter]
#    48 |                             int32_t zpb, TwoDMemrefI32_t *c) {
#       |                                     ^
# main.c:72:16: warning: incompatible pointer types assigning to 'int8_t *' (aka 'signed char *') from 'const int8_t (*)[256]' (aka 'const signed char (*)[256]') [-Wincompatible-pointer-types]
#    72 |   memrefA.data = &A;
#       |                ^ ~~
# main.c:77:16: warning: incompatible pointer types assigning to 'int8_t *' (aka 'signed char *') from 'const int8_t (*)[256]' (aka 'const signed char (*)[256]') [-Wincompatible-pointer-types]
#    77 |   memrefB.data = &B;
#       |                ^ ~~
# main.c:82:16: warning: incompatible pointer types assigning to 'int32_t *' (aka 'int *') from 'const int32_t (*)[256]' (aka 'const int (*)[256]') [-Wincompatible-pointer-types]
#    82 |   memrefC.data = &C;
#       |                ^ ~~
# 10 warnings generated.
/usr/bin/clang-17 -I/opt/snax-gemm/target/snitch_cluster/sw/snax/gemm/include -Wno-unused-command-line-argument -I/opt/snax-gemm/target/snitch_cluster/sw/runtime/rtl-generic/src -I/opt/snax-gemm/target/snitch_cluster/sw/runtime/common -I/opt/snax-gemm/sw/snRuntime/api -I/opt/snax-gemm/sw/snRuntime/src -I/opt/snax-gemm/sw/snRuntime/src/omp/ -I/opt/snax-gemm/sw/snRuntime/api/omp/ -I/opt/snax-gemm/sw/math/arch/riscv64/bits/ -I/opt/snax-gemm/sw/math/arch/generic -I/opt/snax-gemm/sw/math/src/include -I/opt/snax-gemm/sw/math/src/internal -I/opt/snax-gemm/sw/math/include/bits -I/opt/snax-gemm/sw/math/include -I/repo/runtime/include -D__DEFINED_uint64_t --target=riscv32-unknown-elf -mcpu=generic-rv32 -march=rv32imafdzfh -mabi=ilp32d -mcmodel=medany -ftls-model=local-exec -ffast-math -fno-builtin-printf -fno-builtin-memset -fno-common -O3 -std=gnu11 -Wall -Wextra -c data.c -o "$OUT/data.o"
/usr/bin/clang-17 /opt/snax-gemm/target/snitch_cluster/sw/snax/gemm/build/snax-gemm-lib.o -fuse-ld=/usr/bin/ld.lld-17 --target=riscv32-unknown-elf -mcpu=generic-rv32 -march=rv32imafdzfh -mabi=ilp32d -mcmodel=medany -T/opt/snax-gemm/sw/snRuntime/base.ld -L/opt/snax-gemm/target/snitch_cluster/sw/runtime/rtl-generic -L/opt/snax-gemm/target/snitch_cluster/sw/runtime/rtl-generic/build -nostdlib -lsnRuntime -fno-builtin-memset "$OUT/$1.o" "$OUT/main.o" "$OUT/data.o" -o "$OUT/$1.x"
rm -fr ./logs/
/opt/snax-gemm-rtl/bin/snitch_cluster.vlt "$OUT/$1.x"
# Warning: Failed to write binary name to logs/.rtlbinary
# [fesvr] Wrote 36 bytes of bootrom to 0x1000
# [fesvr] Wrote entry point 0x80000000 to bootloader slot 0x1020
# [fesvr] Wrote 56 bytes of bootdata to 0x1024
# [Tracer] Logging Hart          0 to logs/trace_hart_00000.dasm
# [Tracer] Logging Hart          1 to logs/trace_hart_00001.dasm
# Executing snax_gemm with a=10000040, b=100003C0, c=10000740 
# Finished executing snax_gemm
mv logs $1.x.logs
/repo/runtime//../compiler/snax-opt -p dispatch-kernels,set-memory-space,set-memory-layout,realize-memref-casts,insert-sync-barrier,dispatch-regions,insert-accfg-op{accelerator=snax_gemm},convert-linalg-to-accfg,convert-accfg-to-csr,snax-copy-to-dma,memref-to-snax,snax-to-func,clear-memory-space -o "$OUTACC/$1.acc_dialect.snax-opt.mlir" "$OUT/$1.preprocfinal.mlir"
cat "$OUTACC/$1.acc_dialect.snax-opt.mlir" | sed 's/arith.maximumf/arith.maxf/g' | sed 's/arith.minimumf/arith.minf/g' > "$OUTACC/$1.acc_dialect.postproc.mlir"
mlir-opt-17 --convert-linalg-to-loops --convert-scf-to-cf --lower-affine --canonicalize --cse --convert-math-to-llvm --llvm-request-c-wrappers --expand-strided-metadata --lower-affine --convert-index-to-llvm=index-bitwidth=32 --convert-cf-to-llvm=index-bitwidth=32 --convert-arith-to-llvm=index-bitwidth=32 --convert-func-to-llvm='index-bitwidth=32' --finalize-memref-to-llvm='use-generic-functions index-bitwidth=32' --canonicalize --reconcile-unrealized-casts -o "$OUTACC/$1.acc_dialect.ll.mlir" "$OUTACC/$1.acc_dialect.postproc.mlir"
mlir-translate-17 --mlir-to-llvmir -o "$OUTACC/$1.acc_dialect.ll" "$OUTACC/$1.acc_dialect.ll.mlir"
/repo/runtime//tollvm12.py < "$OUTACC/$1.acc_dialect.ll" > "$OUTACC/$1.acc_dialect.ll12"
/usr/bin/clang-17 -I/opt/snax-gemm/target/snitch_cluster/sw/snax/gemm/include -Wno-unused-command-line-argument -I/opt/snax-gemm/target/snitch_cluster/sw/runtime/rtl-generic/src -I/opt/snax-gemm/target/snitch_cluster/sw/runtime/common -I/opt/snax-gemm/sw/snRuntime/api -I/opt/snax-gemm/sw/snRuntime/src -I/opt/snax-gemm/sw/snRuntime/src/omp/ -I/opt/snax-gemm/sw/snRuntime/api/omp/ -I/opt/snax-gemm/sw/math/arch/riscv64/bits/ -I/opt/snax-gemm/sw/math/arch/generic -I/opt/snax-gemm/sw/math/src/include -I/opt/snax-gemm/sw/math/src/internal -I/opt/snax-gemm/sw/math/include/bits -I/opt/snax-gemm/sw/math/include -I/repo/runtime/include -D__DEFINED_uint64_t --target=riscv32-unknown-elf -mcpu=generic-rv32 -march=rv32imafdzfh -mabi=ilp32d -mcmodel=medany -ftls-model=local-exec -ffast-math -fno-builtin-printf -fno-builtin-memset -fno-common -O3 -std=gnu11 -Wall -Wextra -x ir -c "$OUTACC/$1.acc_dialect.ll12" -o "$OUTACC/$1.acc_dialect.o"
/usr/bin/clang-17 /opt/snax-gemm/target/snitch_cluster/sw/snax/gemm/build/snax-gemm-lib.o -fuse-ld=/usr/bin/ld.lld-17 --target=riscv32-unknown-elf -mcpu=generic-rv32 -march=rv32imafdzfh -mabi=ilp32d -mcmodel=medany -T/opt/snax-gemm/sw/snRuntime/base.ld -L/opt/snax-gemm/target/snitch_cluster/sw/runtime/rtl-generic -L/opt/snax-gemm/target/snitch_cluster/sw/runtime/rtl-generic/build -nostdlib -lsnRuntime -fno-builtin-memset "$OUTACC/$1.acc_dialect.o" "$OUT/main.o" "$OUT/data.o" -o "$OUTACC/$1.acc_dialect.x"
rm -fr ./logs/
/opt/snax-gemm-rtl/bin/snitch_cluster.vlt "$OUTACC/$1.acc_dialect.x"
