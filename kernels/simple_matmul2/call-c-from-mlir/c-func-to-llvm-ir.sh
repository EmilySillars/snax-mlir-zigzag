basename=`basename $1 | sed 's/[.][^.]*$//'`

# emit llvm IR instead of an object file
/usr/bin/clang-17 -S -emit-llvm -I/opt/snax-gemm/target/snitch_cluster/sw/snax/gemm/include \
-Wno-unused-command-line-argument -I/opt/snax-gemm/target/snitch_cluster/sw/runtime/rtl-generic/src \
-I/opt/snax-gemm/target/snitch_cluster/sw/runtime/common -I/opt/snax-gemm/sw/snRuntime/api \
-I/opt/snax-gemm/sw/snRuntime/src -I/opt/snax-gemm/sw/snRuntime/src/omp/ \
-I/opt/snax-gemm/sw/snRuntime/api/omp/ -I/opt/snax-gemm/sw/math/arch/riscv64/bits/ \
-I/opt/snax-gemm/sw/math/arch/generic -I/opt/snax-gemm/sw/math/src/include \
-I/opt/snax-gemm/sw/math/src/internal -I/opt/snax-gemm/sw/math/include/bits \
-I/opt/snax-gemm/sw/math/include -I/repo/runtime/include -D__DEFINED_uint64_t \
--target=riscv32-unknown-elf -mcpu=generic-rv32 -march=rv32imafdzfh -mabi=ilp32d \
-mcmodel=medany -ftls-model=local-exec -ffast-math -fno-builtin-printf -fno-common -O3 \
-std=gnu11 -Wall -Wextra -c $basename.c -o $basename.ll

