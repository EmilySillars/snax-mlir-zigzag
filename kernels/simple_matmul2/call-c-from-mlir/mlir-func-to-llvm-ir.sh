basename=`basename $1 | sed 's/[.][^.]*$//'`

python3 gendata.py

mlir-opt-17 --pass-pipeline='builtin.module(func.func(tosa-to-linalg-named, tosa-to-tensor, tosa-to-scf, tosa-to-linalg))' \
--mlir-print-op-generic --mlir-print-local-scope -o out/$basename.preproc1.mlir $basename.mlir

mlir-opt-17 --tosa-to-arith="include-apply-rescale" --empty-tensor-to-alloc-tensor -o out/$basename.preproc2.0.mlir out/$basename.preproc1.mlir

# preproc2 to preproc3 with intermediate steps vvv
mlir-opt-17 --test-linalg-transform-patterns="test-generalize-pad-tensor" \
-o out/$basename.preproc2.1.mlir out/$basename.preproc2.0.mlir

mlir-opt-17 --linalg-generalize-named-ops \
-o out/$basename.preproc2.2.mlir out/$basename.preproc2.1.mlir

mlir-opt-17 --empty-tensor-to-alloc-tensor \
-o out/$basename.preproc2.3.mlir out/$basename.preproc2.2.mlir

mlir-opt-17 --one-shot-bufferize="bufferize-function-boundaries allow-return-allocs \
function-boundary-type-conversion=identity-layout-map" \
-o out/$basename.preproc2.4.mlir out/$basename.preproc2.3.mlir

mlir-opt-17 --mlir-print-op-generic \
-o out/$basename.preproc2.5.mlir out/$basename.preproc2.4.mlir

mlir-opt-17 --mlir-print-local-scope \
-o out/$basename.preproc3.mlir out/$basename.preproc2.5.mlir
# preproc2 to preproc3 with intermediate steps ^^^

cat out/$basename.preproc3.mlir | sed 's/arith.maxf/arith.maximumf/g' | sed 's/arith.minf/arith.minimumf/g' > out/$basename.preprocfinal.mlir

mlir-opt-17 --mlir-print-op-generic \
-o out/$basename.preprocfinal.generic.mlir out/$basename.preprocfinal.mlir

# separating memory attotated IR from non-memory annotated IR
/repo/runtime//../compiler/snax-opt -p dispatch-kernels,set-memory-space -o out/$basename.afterMemAnns.mlir out/$basename.preprocfinal.generic.mlir
/repo/runtime//../compiler/snax-opt -p set-memory-layout,realize-memref-casts -o out/$basename.afterMemAnns2.mlir out/$basename.afterMemAnns.mlir
/repo/runtime//../compiler/snax-opt -p \
insert-sync-barrier,dispatch-regions,linalg-to-library-call,snax-copy-to-dma,memref-to-snax,snax-to-func,clear-memory-space \
-o out/$basename.snax-opt.mlir out/$basename.afterMemAnns2.mlir

cat out/$basename.snax-opt.mlir | sed 's/arith.maximumf/arith.maxf/g' | sed 's/arith.minimumf/arith.minf/g' > out/$basename.postproc.mlir

# TODO: Get rid of hard coding!
# THIS IS BAD HARDCODING THAT SHOULD BE REPLACED WITH AN XDSL PASS VVVVVVVVVVVVVVVV
# fill static_offset fields with zero values
# awk '/-9223372036854775808/ && ++count==2{sub(/-9223372036854775808/,"0")} 1' \
sed 's/-9223372036854775808/0/g' out/$basename.postproc.mlir > out/$basename.postproc.cleared.static.offsets.mlir
# fill offset fields with zero values
# awk '/offset: -156797324626531188736/ && ++count==2{sub(/offset: -156797324626531188736/,"offset: 0")} 1' \
sed 's/-156797324626531188736/0/g' out/$basename.postproc.cleared.static.offsets.mlir > out/$basename.postproc.cleared.offset.mlir
# THIS IS BAD HARDCODING THAT SHOULD BE REPLACED WITH AN XDSL PASS ^^^^^^^^^^^^^^^^

mlir-opt-17  --convert-linalg-to-loops --convert-scf-to-cf --lower-affine --canonicalize \
--cse --convert-math-to-llvm --llvm-request-c-wrappers --expand-strided-metadata \
--convert-index-to-llvm=index-bitwidth=32 --convert-cf-to-llvm=index-bitwidth=32 \
--convert-arith-to-llvm=index-bitwidth=32 --convert-func-to-llvm='index-bitwidth=32' \
--finalize-memref-to-llvm='use-generic-functions index-bitwidth=32' --canonicalize \
--reconcile-unrealized-casts -o out/$basename.ll.mlir out/$basename.postproc.cleared.offset.mlir

mlir-translate-17 --mlir-to-llvmir -o out/$basename.ll out/$basename.ll.mlir
