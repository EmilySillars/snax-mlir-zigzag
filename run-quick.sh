if [[ $1 == "-z" ]]; then
    # set up
    INPUT="/repo/kernels/$2/$2.mlir"
    OUT="/repo/kernels/$2/out-my-lowering"
    mkdir -p $OUT
    rm $OUT/*
    # remove old logs    
    rm -fr /repo/kernels/$2/*.logs/
    
    # run input through my lowering passes
    mlir-opt $INPUT --mlir-print-op-generic > "$OUT/$2_zigzag2.mlir"
    mlir-opt "$OUT/$2_zigzag2.mlir" --linalg-generalize-named-ops --mlir-print-op-generic --mlir-print-local-scope >"$OUT/$2_zigzag3.mlir"
    # mlir-opt "$OUT/$2_zigzag3.mlir" --one-shot-bufferize='bufferize-function-boundaries' > "$OUT/$2_zigzag4.mlir"
    # save processed input as next input to built in snax-mlir flow
    cat "$OUT/$2_zigzag3.mlir" > "/repo/kernels/$2/$2_zigzag.mlir"
    # build and run example
    cd /repo/kernels/$2
    sh /repo/zigzag-fork/snax-gemm-step-by-step.sh "$2_zigzag" "/repo/kernels/$2"
    cd /repo
    
else
    cd /repo/kernels/$1
    rm -fr ./*.logs/
    make allrun
    cd /repo
fi