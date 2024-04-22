#!/bin/sh
readonly YODEL="MLIR to LLVM"
echo $YODEL
basename=`basename $1 | sed 's/[.][^.]*$//'`
funcname=`basename $2 | sed 's/[.][^.]*$//'`

echo "Converting $basename.mlir to LLVM DIALECT..."

mlir-opt-17 "$basename.mlir" \
-convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -convert-vector-to-llvm --convert-cf-to-llvm -expand-strided-metadata \
-lower-affine -convert-arith-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts > out/$basename-in-llvm-dialect.mlir

echo "Converting LLVM DIALECT to LLVM IR..."
echo "mlir-translate-17 --mlir-to-llvmir out/$basename-in-llvm-dialect.mlir > out/$basename.ll"
mlir-translate-17 --mlir-to-llvmir out/$basename-in-llvm-dialect.mlir > out/$basename.ll
echo ""

echo "Compile and then run LLVM IR..."
#echo "clang out/$basename-frankenstein.ll -o out/$basename-frankenstein.o"
#clang out/$basename.ll -o out/$basename.o libmlir_c_runner_utils.so libmlir_runner_utils.so -Wl -rpath,/home/hoppip/llvm-project-pistachio/build-riscv/lib/libmlir_c_runner_utils.so,/home/hoppip/llvm-project-pistachio/build-riscv/lib/libmlir_runner_utils.so

#clang out/$basename.ll -o out/$basename.o /home/hoppip/llvm-project-pistachio/build-riscv/lib/libmlir_c_runner_utils.so /home/hoppip/llvm-project-pistachio/build-riscv/lib/libmlir_runner_utils.so
#clang out/$basename.ll -o out/$basename.o -L/home/hoppip/llvm-project-pistachio/build-riscv/lib -lmlir_c_runner_utils -lmlir_runner_utils
#clang out/$basename.ll -o out/$basename.o -L/home/hoppip/llvm-project-pistachio/build-riscv/lib 

#clang out/$basename.ll -o out/$basename.o -L/home/hoppip/llvm-project-pistachio/build-riscv/lib -lmlir_c_runner_utils -lmlir_runner_utils

clang out/$basename.ll -o out/$basename.o \
/home/hoppip/llvm-project-pistachio/build-riscv/lib/libmlir_c_runner_utils.so \
/home/hoppip/llvm-project-pistachio/build-riscv/lib/libmlir_c_runner_utils.so.18git \
/home/hoppip/llvm-project-pistachio/build-riscv/lib/libmlir_runner_utils.so \
/home/hoppip/llvm-project-pistachio/build-riscv/lib/libmlir_runner_utils.so.18git

out/$basename.o

# -test-linalg-transform-patterns=test-linalg-to-vector-patterns \
# -empty-tensor-to-alloc-tensor -linalg-bufferize -arith-bufferize \
# -bufferization-bufferize -tensor-bufferize -func-bufferize \
# -finalizing-bufferize -buffer-deallocation-pipeline -convert-bufferization-to-memref \