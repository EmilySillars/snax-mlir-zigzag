 %result = linalg.generic {
    // This {...} section is the "attributes" - some compile-time settings for this op.
    indexing_maps=[
      // Indexing maps for the parameters listed in `ins(...)`
      #map_1d_identity,
      #map_1d_identity,
      // Indexing maps for the parameters listed in `outs(...)`
      #map_1d_identity
    ],
    // There is one tensor dimension, and it's a parallel-iteration dimension,
    // meaning that it occurs also as a result tensor dimension. The alternative
    // would be "reduction", for dimensions that do not occur in the result tensor.
    iterator_types=["parallel"]
  } // End of the attributes for this linalg.generic. Next come the parameters:
    // `ins` is where we pass regular input-parameters
    ins(%lhs, %rhs : tensor<10xf32>, tensor<10xf32>)
    // `outs` is where we pass the "outputs", but that term has a subtle meaning
    // in linalg. Here we are passing a tensor.empty, meaning just a placeholder
    // for the output with no preexisting element values. In other examples with
    // an accumulator, this is where the accumulator would be passed.
    outs(%result_empty : tensor<10xf32>)
    // End of parameters. The next {...} part is the "code block".   
  {
    // bb0 is a code block taking one scalar from each input tensor as argument, and
    // computing and "yielding" (ie returning) the corresponding output tensor element.
    ^bb0(%lhs_entry : f32, %rhs_entry : f32, %unused_result_entry : f32):
      %add = arith.addf %lhs_entry, %rhs_entry : f32
      linalg.yield %add : f32
  } // End of the basic block. Finally, we describe the return type.
  -> tensor<10xf32>
