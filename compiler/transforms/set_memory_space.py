from xdsl.dialects import builtin, func, linalg, memref
from xdsl.ir import MLContext, SSAValue, BlockArgument
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
#from xdsl.ir import Attribute, BlockArgument, OpResult

class InitFuncMemorySpace(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter):
        """Add a default (0 : i32) memory space to memrefs used in the function
        that do not have a memory space specified yet"""

        # Function must be public
        if op.sym_visibility is not None and op.sym_visibility.data != "public":
            return

        # Function must have memref arguments with an undefined memory space
        if not any(
            [
                isinstance(x, memref.MemRefType)
                and isinstance(x.memory_space, builtin.NoneAttr)
                for x in [*op.function_type.inputs, *op.function_type.outputs]
            ]
        ):
            return
        
        # Mapping function to assign default memory space (0 : i32)
        def change_to_memory_space(t):
            if isinstance(t, memref.MemRefType):
                if isinstance(t.memory_space, builtin.NoneAttr):
                    return memref.MemRefType(
                        t.element_type,
                        t.get_shape(),
                        t.layout,
                        builtin.IntegerAttr(1989, builtin.i32), # CHANGE THIS BACK TO ZERO BEFORE FINILIZING EDITS!!!! PINEAPPLE
                    )
            return t

        # Define new function type with updated inputs and outputs
        # mapped to a default memory space
        new_function_type = builtin.FunctionType.from_lists(
            map(change_to_memory_space, op.function_type.inputs),
            map(change_to_memory_space, op.function_type.outputs),
        )

        # Change region of function to use new argument types
        for arg in op.args:
            arg.type = change_to_memory_space(arg.type)

        # Define new function op with new type and copy region contents
        new_op = func.FuncOp(
            op.sym_name.data,
            new_function_type,
            region=rewriter.move_region_contents_to_new_regions(op.regions[0]),
            visibility=op.sym_visibility,
        )

        # Replace function op
        rewriter.replace_matched_op(new_op)


class InitMemRefGlobalMemorySpace(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.GetGlobal, rewriter: PatternRewriter):
        # global variables should go in memory space 0 (L3)
        memspace = op.memref.type.memory_space

        # If memory space is already 0, don't do anything
        if isinstance(memspace, builtin.IntegerAttr) and memspace.value.data == 0:
            return

        # otherwise, create new memref type with correct memory space
        new_memref_type = memref.MemRefType(
            op.memref.type.element_type,
            op.memref.type.get_shape(),
            op.memref.type.layout,
            builtin.IntegerAttr(0, builtin.i32),
        )

        # create new get_global op
        new_op = memref.GetGlobal.get(op.name_.root_reference.data, new_memref_type)

        # replace op
        rewriter.replace_matched_op(new_op)


class InitMemRefAllocMemorySpace(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.Alloc, rewriter: PatternRewriter):
        # allocs should go in memory space 1 (L1)
        memspace = op.memref.type.memory_space

        if isinstance(memspace, builtin.IntegerAttr) and memspace.value.data == 1:
            # good, nothing left to do
            return

        # create new alloc op
        new_op = memref.Alloc.get(
            op.memref.type.element_type,
            op.alignment,
            op.memref.type.get_shape(),
            dynamic_sizes=op.dynamic_sizes,
            layout=op.memref.type.layout,
            memory_space=builtin.IntegerAttr(1, builtin.i32),
        )

        # replace op
        rewriter.replace_matched_op(new_op, new_results=[new_op.memref])

# @irdl_op_definition
# class Subview(IRDLOperation):
#     name = "memref.subview"

#     source: Operand = operand_def(MemRefType)
#     offsets: VarOperand = var_operand_def(IndexType)
#     sizes: VarOperand = var_operand_def(IndexType)
#     strides: VarOperand = var_operand_def(IndexType)
#     static_offsets: DenseArrayBase = prop_def(DenseArrayBase)
#     static_sizes: DenseArrayBase = prop_def(DenseArrayBase)
#     static_strides: DenseArrayBase = prop_def(DenseArrayBase)
#     result: OpResult = result_def(MemRefType)

    # def from_static_parameters(
    #     source: SSAValue | Operation,
    #     source_type: MemRefType[Attribute],
    #     offsets: Sequence[int],
    #     sizes: Sequence[int],
    #     strides: Sequence[int],
    #     reduce_rank: bool = False,
    # ) -> Subview:
    #     source = SSAValue.get(source)

    #     source_shape = source_type.get_shape()
    #     source_offset = 0
    #     source_strides = [1]
    #     for input_size in reversed(source_shape[1:]):
    #         source_strides.insert(0, source_strides[0] * input_size)
    #     if isinstance(source_type.layout, StridedLayoutAttr):
    #         if isinstance(source_type.layout.offset, IntAttr):
    #             source_offset = source_type.layout.offset.data
    #         if isa(source_type.layout.strides, ArrayAttr[IntAttr]):
    #             source_strides = [s.data for s in source_type.layout.strides]

    #     layout_strides = [a * b for (a, b) in zip(strides, source_strides)]

    #     layout_offset = (
    #         sum(stride * offset for stride, offset in zip(source_strides, offsets))
    #         + source_offset
    #     )

    #     if reduce_rank:
    #         composed_strides = layout_strides
    #         layout_strides: list[int] = []
    #         result_sizes: list[int] = []

    #         for stride, size in zip(composed_strides, sizes):
    #             if size == 1:
    #                 continue
    #             layout_strides.append(stride)
    #             result_sizes.append(size)

    #     else:
    #         result_sizes = list(sizes)

    #     layout = StridedLayoutAttr(layout_strides, layout_offset)

    #     return_type = MemRefType(
    #         source_type.element_type,
    #         result_sizes,
    #         layout,
    #         source_type.memory_space,
    #     )

    #     return Subview.build(
    #         operands=[source, [], [], []],
    #         result_types=[return_type],
    #         properties={
    #             "static_offsets": DenseArrayBase.from_list(i64, offsets),
    #             "static_sizes": DenseArrayBase.from_list(i64, sizes),
    #             "static_strides": DenseArrayBase.from_list(i64, strides),
    #         },
    #     )


class InitMemRefSubviewMemorySpace(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.Subview, rewriter: PatternRewriter):
        assert len(op.operands) != 0
        inputType = op.operands[0].type
        if ( op.result.type.memory_space == inputType.memory_space):
            return
        if (not inputType.memory_space):
            inputType.memory_space = builtin.IntegerAttr(0, builtin.i32) # by default
        
        memspace = inputType.memory_space
        # subview's result should have the same memory space as the memref the subview is viewing

        print(f'I care about\n{op.operands[0].type} and')
        print(f'{op.result.type}\nfrom the subview.')
    # @staticmethod
    # def from_static_parameters(
    #     source: SSAValue | Operation,
    #     source_type: MemRefType[Attribute],
    #     offsets: Sequence[int],
    #     sizes: Sequence[int],
    #     strides: Sequence[int],
    #     reduce_rank: bool = False,
    # def test_memref_subview_constant_parameters():
    # alloc = Alloc.get(i32, 8, [10, 10, 10])
    # assert isa(alloc.memref.type, MemRefType[Attribute])

    # subview = Subview.from_static_parameters(
    #     alloc, alloc.memref.type, [2, 2, 2], [2, 2, 2], [3, 3, 3]
    # )

            # for store in stores:
            # field = store.field
            # assert isa(field.type, FieldType[Attribute])
            # assert isa(field.type.bounds, StencilBoundsAttr)
            # temp = store.temp
            # assert isa(temp.type, TempType[Attribute])
            # offsets = [i for i in -field.type.bounds.lb]
            # sizes = [i for i in temp.type.get_shape()]
            # subview = memref.Subview.from_static_parameters(
            #     field,
            #     StencilToMemRefType(field.type),
            #     offsets,
            #     sizes,
            #     [1] * len(sizes),
            # )
            # name = None
            # if subview.source.name_hint:
            #     name = subview.source.name_hint + "_storeview"
            # subview.result.name_hint = name
            # if isinstance(field.owner, Operation):
            #     rewriter.insert_op_after(subview, field.owner)
            # else:
            #     rewriter.insert_op_at_start(subview, field.owner)

            # rewriter.erase_op(store)


    # assert isinstance(subview, Subview)
    # assert isinstance(subview.result.type, MemRefType)
    # assert isinstance(subview.result.type.layout, StridedLayoutAttr)
    # assert isa(subview.result.type.layout.strides, ArrayAttr[IntAttr])
    # out_strides = [i.data for i in subview.result.type.layout.strides.data]
    # assert out_strides == [300, 30, 3]
    # assert isinstance(subview.result.type.layout.offset, IntAttr)
    # assert subview.result.type.layout.offset.data == 222
        print('creating a new op...\n')
        print(f'old op is {str(op)}\n')
    # old op is %0 = "memref.subview"(%arg2, %arg5, %arg3) <{
    #     "operandSegmentSizes" = array<i32: 1, 2, 0, 0>, 
    #     "static_offsets" = array<i64: -9223372036854775808, -9223372036854775808>, 
    #     "static_sizes" = array<i64: 2, 2>, "static_strides" = array<i64: 1, 1>}> : 
    # (memref<16x16xi32, strided<[16, 1]>, 1989 : i32>, index, index) -> memref<2x2xi32, strided<[16, 1], offset: ?>>


        print(f'op.source: {str(type(op.source))} {op.source}\n')
        print(f'inputType: {str(type(inputType))} {inputType}\n')

        # print(f'static_offsets: {str(type(op.static_offsets))} {op.offsets}\n')
        print(f'{str(op.static_offsets.as_tuple())}\n')
        for off in op.static_offsets.as_tuple():
            print(f'\t{str(off)}\n')
            # print(f'offset elt: type: {type(off)} value: {off} type field: {str(off.type)}\n')
            # print(f'offset elt: type: {type(off)} value: {off} index into value:{off.ops[off.index]}\n')
        print(f'static_sizes: {str(type(op.static_sizes))} {str(op.static_sizes)}\n')
        print(f'static_sizes: {str(type(op.static_sizes))} {str(op.static_sizes.as_tuple())}\n')

        print(f'static_strides: {str(type(op.static_strides))} {str(op.static_strides.as_tuple())} \n')

        # %5 = "memref.subview"(%arg0, %arg5, %arg4) <{
        #     operandSegmentSizes = array<i32: 1, 2, 0, 0>,

        #     static_offsets = array<i64: -9223372036854775808, -9223372036854775808>, 
        #     static_sizes = array<i64: 2, 16>, 
        # static_strides = array<i64: 1, 1>}> : 

        # (memref<16x16xi8>, index, index) -> memref<2x16xi8, strided<[16, 1], offset: ?>>
        # new_op =  memref.Subview.from_static_parameters(
        # op.source,
        # inputType,
        # op.offsets,
        # op.sizes,
        # op.strides,)
   #     op.reduce_rank)
        new_op =  memref.Subview.from_static_parameters(
        op.source,
        inputType,
        op.static_offsets.as_tuple(),
        op.static_sizes.as_tuple(),
        op.static_strides.as_tuple(),)
        # [2,2,2], # offsets
        # [3,3,3], # sizes
        # [3,3,3],) # strides
        # print("new op is:")
        # print(str(new_op))
        

        #new_op.result.type.memory_space = memspace

    

        rewriter.replace_matched_op(new_op)#,new_results=[new_op.memref])


###############################################################################
#         p = Printer()
# p.print_op(subview_module)

        # replace op
        # rewriter.replace_matched_op(new_op, new_results=[new_op.memref])
       #isinstance(memspace, builtin.IntegerAttr) and memspace.value.data == 1:
        # if not isinstance(memspace, builtin.IntegerAttr):
        #     # good, nothing left to do
        #     raise ValueError("YODEL")
        # else:
        #     raise ValueError("AYHEEHOOOOOOOOOO~~~~~~~~!")

        # create new alloc op
        # new_op = memref.subview.get(
        #     op.memref.type.element_type,
        #     op.alignment,
        #     op.memref.type.get_shape(),
        #     dynamic_sizes=op.dynamic_sizes,
        #     layout=op.memref.type.layout,
        #     memory_space=memspace,
        # )
###############################################################################
class InitLinalgMemorySpace(RewritePattern):
    """Walk through dispatchable operations (just linalg.Generic for now)
    and change them to use only memrefs in memory space (1 : i32). If they
    currently use a memref in a different memory adress space, insert a
    memref.memory_space_cast operation to convert the two"""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.Generic, rewriter: PatternRewriter):
        # Op must have memref arguments with memory space not equal to 1
        if not any(
            [
                isinstance(x.type, memref.MemRefType)
                and isinstance(x.type.memory_space, builtin.IntegerAttr)
                and x.type.memory_space.value.data != 1
                for x in op.inputs
            ]
        ):
            return

        # Function to find/create casting operand if it is necessary
        def get_cast_op(operand) -> None | memref.MemorySpaceCast:
            # check if cast is required: must be a memref in wrong memory space
            if not isinstance(operand, SSAValue):
                return None
            if not isinstance(operand.type, memref.MemRefType):
                return None
            if (
                isinstance(operand.type.memory_space, builtin.IntegerAttr)
                and operand.type.memory_space.value.data == 1
            ):
                return None

            # cast required: find previous cast or create new one
            cast_op = None
            for use in operand.uses:
                if (
                    isinstance(use.operation, memref.MemorySpaceCast)
                    and isinstance(use.operation.dest.type, memref.MemRefType)
                    and use.operation.dest.type.memory_space
                    == builtin.IntegerAttr(1, builtin.i32)
                ):
                    cast_op = use.operation
                    break
            # If cast op not found, create and insert new one
            if cast_op is None:
                cast_op = memref.MemorySpaceCast.from_type_and_target_space(
                    operand, operand.type, builtin.IntegerAttr(1, builtin.i32)
                )
                rewriter.insert_op_before_matched_op(cast_op)

            return cast_op

        # cast all inputs and outputs to correct memory space
        new_inputs = [
            inp if get_cast_op(inp) is None else get_cast_op(inp).dest
            for inp in op.inputs
        ]
        new_outputs = [
            out if get_cast_op(out) is None else get_cast_op(out).dest
            for out in op.outputs
        ]

        # new linalg op with new inputs & outputs
        linalg_op = linalg.Generic(
            new_inputs,
            new_outputs,
            rewriter.move_region_contents_to_new_regions(op.regions[0]),
            op.indexing_maps,
            op.iterator_types,
            [],
            op.doc,
            op.library_call,
        )

        # replace op
        rewriter.replace_matched_op(linalg_op)


class HandleFuncReturns(RewritePattern):
    """Function returns which return a memref object must be replaced
    such that the memref object is returned in the memory space specified
    in the function handle"""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.Return, rewriter: PatternRewriter):
        # get function op
        func_op: func.FuncOp = op.parent_op()

        outputs = [*func_op.function_type.outputs]

        new_arguments = []
        changes_made = False

        # all outputs must be in the correct memory space
        for i in range(len(outputs)):
            func_op_output = outputs[i]
            func_return_output = op.arguments[i]

            if not isinstance(func_op_output, memref.MemRefType):
                new_arguments.append(func_return_output)
                continue
            if not isinstance(func_return_output.type, memref.MemRefType):
                new_arguments.append(func_return_output)
                continue

            if func_op_output.memory_space != func_return_output.type.memory_space:
                # create cast op
                cast_op = memref.MemorySpaceCast.from_type_and_target_space(
                    func_return_output,
                    func_return_output.type,
                    func_op_output.memory_space,
                )
                rewriter.insert_op_before_matched_op(cast_op)

                # replace return value with cast
                new_arguments.append(cast_op)
                changes_made = True

        if not changes_made:
            return

        # create new return op
        new_op = func.Return(*new_arguments)

        # replace op
        rewriter.replace_matched_op(new_op)


class SetMemorySpace(ModulePass):
    name = "set-memory-space"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(InitFuncMemorySpace()).rewrite_module(op)
        PatternRewriteWalker(InitMemRefGlobalMemorySpace()).rewrite_module(op)
        PatternRewriteWalker(InitMemRefAllocMemorySpace()).rewrite_module(op)
        PatternRewriteWalker(InitMemRefSubviewMemorySpace()).rewrite_module(op)
        PatternRewriteWalker(InitLinalgMemorySpace()).rewrite_module(op)
        PatternRewriteWalker(HandleFuncReturns()).rewrite_module(op)
