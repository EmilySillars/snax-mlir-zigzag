from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from xdsl.dialects.builtin import (
    AnyIntegerAttr,
    IndexType,
    IntegerAttr,
    IntegerType,
    MemRefType,
    NoneAttr,
    UnrankedMemrefType,
    i32,
)
from xdsl.dialects.llvm import LLVMStructType
from xdsl.ir import Attribute, Data, Dialect, Operation, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    Operand,
    OpResult,
    VarOperand,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    result_def,
    var_operand_def,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException

from compiler.accelerators.streamers import StreamerConfiguration
from compiler.accelerators.streamers.streamers import Streamer, StreamerType
from compiler.util.memref_descriptor import LLVMMemrefDescriptor


@irdl_op_definition
class ClusterSyncOp(IRDLOperation):
    """Cluster sync operation for a snax cluster. This
    translates directly to the C function snrt_cluster_hw_barrier()"""

    name = "snax.cluster_sync_op"


@irdl_op_definition
class MCycleOp(IRDLOperation):
    """Utility operation that translates to risc-v mcycle instruction
    for trace annotation."""

    name = "snax.mcycle"


@irdl_op_definition
class LayoutCast(IRDLOperation):
    """LayoutCast operation for memrefs in a snax cluster. This
    operation is used to change the layout of the memref data"""

    name = "snax.layout_cast"

    source = operand_def(MemRefType[Attribute] | UnrankedMemrefType[Attribute])
    dest = result_def(MemRefType[Attribute] | UnrankedMemrefType[Attribute])

    def __init__(
        self,
        source: SSAValue | Operation,
        dest: MemRefType[Attribute] | UnrankedMemrefType[Attribute],
    ):
        super().__init__(operands=[source], result_types=[dest])

    @staticmethod
    def from_type_and_target_layout(
        source: SSAValue | Operation,
        layout: Attribute,
    ) -> LayoutCast:
        assert isinstance(source.type, MemRefType)
        dest = MemRefType(
            source.type.get_element_type(),
            shape=source.type.get_shape(),
            layout=layout,
            memory_space=source.type.memory_space,
        )
        return LayoutCast(source, dest)

    def verify_(self) -> None:
        source = cast(MemRefType[Attribute], self.source.type)
        dest = cast(MemRefType[Attribute], self.dest.type)
        if source.get_shape() != dest.get_shape():
            raise VerifyException(
                "Expected source and destination to have the same shape."
            )
        if source.get_element_type() != dest.get_element_type():
            raise VerifyException(
                "Expected source and destination to have the same element type."
            )
        if source.memory_space != dest.memory_space:
            raise VerifyException(
                "Expected source and destination to have the same memory space."
            )


@irdl_op_definition
class Alloc(IRDLOperation):
    """Alloc operation in a snax cluster.

    Contrary to a memref.alloc, this operation does not generate
    a memref. Instead, it returns an llvm struct memref descriptor.
    When other operations get lowered to llvm, the llvm structs will
    match and the conversion casts can be realized.
    """

    name = "snax.alloc"

    size: Operand = operand_def(IntegerType | IndexType)
    shapes: VarOperand = var_operand_def(IntegerType | IndexType)
    result: OpResult = result_def(LLVMStructType)
    memory_space: Attribute | None = opt_prop_def(Attribute)
    alignment: AnyIntegerAttr | None = opt_prop_def(AnyIntegerAttr)

    def __init__(
        self,
        rank: int,
        size: SSAValue | Operation,
        shapes: list[SSAValue | Operation],
        memory_space: Attribute = NoneAttr(),
        alignment: AnyIntegerAttr = None,
        integer_type: IntegerType = i32,
    ):
        # output type is llvm struct memref descriptor
        descriptor = LLVMMemrefDescriptor.from_rank_and_integer_type(rank, integer_type)

        if not alignment:
            alignment = IntegerAttr(1, IntegerType(64))

        super().__init__(
            operands=[size, shapes],
            result_types=[descriptor.descriptor],
            properties={"memory_space": memory_space, "alignment": alignment},
        )

    def verify_(self) -> None:
        # check for a correct result type
        if not isinstance(self.result.type, LLVMStructType):
            raise VerifyException("Expected result type to be LLVMStructType")

        descriptor = LLVMMemrefDescriptor(self.result.type)
        descriptor.verify()


@irdl_attr_definition
class StreamerConfigurationAttr(Data[StreamerConfiguration]):
    name = "snax.streamer_config"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> StreamerConfiguration:
        # parse a streamer config in the following format:
        # for a streamer with 2 readers, 1 writer, each with
        # a temporal and spatial dimension:
        # snax.streamer_config<r[5, 4], r[2, 5], w[1, 1]>

        with parser.in_angle_brackets():
            streamers: Sequence[Streamer] = []

            while True:
                # Determine streamer type
                streamer_type: StreamerType = parser.parse_str_enum(StreamerType)

                # Determine temporal and spatial dimensions
                dimensions = parser.parse_comma_separated_list(
                    parser.Delimiter.SQUARE, parser.parse_integer
                )
                assert len(dimensions) == 2
                streamers.append(Streamer(streamer_type, *dimensions))

                if not parser.parse_optional_punctuation(","):
                    break

            return StreamerConfiguration(streamers)

    def print_parameter(self, printer: Printer) -> None:
        # print a streamer config in the following format:
        # for a streamer with 2 readers, 1 writer, each with
        # a temporal and spatial dimension:
        # snax.streamer_config<r[5, 4], r[2, 5], w[1, 1]>

        streamer_strings = [
            f"{streamer.type.value}[{streamer.temporal_dim}, {streamer.spatial_dim}]"
            for streamer in self.data.streamers
        ]
        printer.print_string(f"<{', '.join(streamer_strings)}>")


Snax = Dialect(
    "snax", [ClusterSyncOp, MCycleOp, LayoutCast, Alloc], [StreamerConfigurationAttr]
)
