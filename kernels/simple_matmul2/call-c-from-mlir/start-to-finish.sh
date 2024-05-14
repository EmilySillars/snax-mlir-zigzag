basename=`basename $1 | sed 's/[.][^.]*$//'`
cfile=`basename $2 | sed 's/[.][^.]*$//'`

echo "$basename $cfile"
echo "compiling $cfile.c to llvm-ir..."
sh c-func-to-llvm-ir.sh "$cfile.c"

echo "compiling $basename.mlir to llvm-ir..."
sh mlir-func-to-llvm-ir.sh "$basename.mlir"

echo "running the code..."
sh compile-together-then-run.sh "$basename.ll" "$cfile.ll"

# run with:
# sh start-to-finish.sh matmul-transformed-two-pieces.mlir main.c
