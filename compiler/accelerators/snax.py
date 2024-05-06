from abc import ABC
from collections.abc import Sequence

from xdsl.dialects import arith, builtin, llvm
from xdsl.dialects.builtin import i32
from xdsl.dialects.scf import Condition, While, Yield
from xdsl.ir import Operation

from compiler.accelerators.accelerator import Accelerator
from compiler.dialects import acc


class SNAXAccelerator(Accelerator, ABC):
    """
    Abstract base class for extending AcceleratorInterfaces
    with common SNAX lowerings.
    """

    @staticmethod
    def lower_acc_await(acc_op: acc.AcceleratorOp) -> Sequence[Operation]:
        return [
            While(
                [],
                [],
                [
                    barrier := arith.Constant(acc_op.barrier),
                    zero := arith.Constant(
                        builtin.IntegerAttr.from_int_and_width(0, 32)
                    ),
                    status := llvm.InlineAsmOp(
                        "csrr $0, $1",
                        # I = any 12 bit immediate
                        # =r = store result in A 32- or 64-bit
                        # general-purpose register (depending on the platform XLEN)
                        "=r, I",
                        [barrier],
                        [i32],
                        has_side_effects=True,
                    ),
                    # check if not equal to zero
                    comparison := arith.Cmpi(status, zero, "ne"),
                    Condition(comparison.results[0]),
                ],
                [
                    Yield(),
                ],
            ),
            addr_val := arith.Constant(builtin.IntegerAttr(965, 12)),  # 0x3c5 = 965
            zero := arith.Constant(builtin.IntegerAttr.from_int_and_width(0, 5)),
            llvm.InlineAsmOp(
                "csrw $0, $1",
                "I, K",
                [addr_val, zero],
                has_side_effects=True,
            ),
            # Three nops for random but important reasons
            llvm.InlineAsmOp("nop", "", [], [], has_side_effects=True),
            llvm.InlineAsmOp("nop", "", [], [], has_side_effects=True),
            llvm.InlineAsmOp("nop", "", [], [], has_side_effects=True),
        ]

    @staticmethod
    def lower_acc_launch(
        launch_op: acc.LaunchOp, acc_op: acc.AcceleratorOp
    ) -> Sequence[Operation]:
        # Get the launch address, for SNAX should be only one
        launch_address = None
        for field, val in acc_op.launch_field_items():
            assert field == "launch"
            launch_address = val
        assert launch_address is not None
        # Get the launch value,
        # For SNAX there should only be one value here
        launch_value = None
        for field, val in launch_op.iter_params():
            assert field == "launch"
            launch_value = val
        assert launch_value is not None
        return [
            addr_val := arith.Constant(launch_address),
            llvm.InlineAsmOp(
                "csrw $0, $1",
                # I = any 12 bit immediate, K = any 5 bit immediate
                # The K allows LLVM to emit an `csrrwi` instruction,
                # which has room for one 5 bit immediate only.
                "I, K",
                [addr_val, launch_value],
                has_side_effects=True,
            ),
        ]

    @staticmethod
    def lower_acc_setup(
        setup_op: acc.SetupOp, acc_op: acc.AcceleratorOp
    ) -> Sequence[Operation]:
        field_to_csr = dict(acc_op.field_items())
        ops: Sequence[Operation] = []
        for field, val in setup_op.iter_params():
            addr = field_to_csr[field]
            ops.extend(
                [
                    addr_val := arith.Constant(addr),
                    llvm.InlineAsmOp(
                        "csrw $0, $1",
                        "I, rK",
                        [addr_val, val],
                        has_side_effects=True,
                    ),
                ]
            )
        return ops
