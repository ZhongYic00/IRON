# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import ClassVar, Dict

from iron.common import (
    MLIROperator,
    AIERuntimeArgSpec,
    KernelObjectArtifact,
    SourceArtifact,
    PythonGeneratedMLIRArtifact,
    DesignGenerator,
)
import aie.utils as aie_utils


@dataclass
class GEMVf32(MLIROperator):
    """f32 GEMV operator — same as GEMV but uses float instead of bfloat16.

    For precision-sensitive operations like lm_head where bf16 argmax drifts.
    """

    M: int
    K: int
    num_aie_columns: int = 1
    tile_size_input: int = 2
    tile_size_output: int | None = None
    kernel_vector_size: int = field(default=32, repr=False)  # f32 vec size
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {
        **MLIROperator._name_aliases,
        "num_aie_columns": "col",
        "tile_size_input": "tsi",
        "tile_size_output": "tso",
    }

    def __post_init__(self):
        if self.tile_size_output is None:
            self.tile_size_output = self.tile_size_input
        if not (
            self.tile_size_output % self.tile_size_input == 0
            and self.tile_size_output >= self.tile_size_input
        ):
            raise ValueError("tile_size_output must be a multiple of tile_size_input")
        if not (
            self.K >= self.kernel_vector_size and self.K % self.kernel_vector_size == 0
        ):
            raise ValueError("K must be multiple of kernel_vector_size")
        MLIROperator.__init__(self, context=self.context)

    @property
    def _kernel_link_file(self):
        return f"gemv_f32_{self.K}k_{self.kernel_vector_size}vs.o"

    def get_mlir_artifact(self):
        mlir_verbose = getattr(self.context, "mlir_verbose", False)
        # Reuse GEMV design.py — it's generic enough, just different kernel/dtype
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "my_matvec",
                (
                    aie_utils.get_current_device(),
                    self.num_aie_columns,
                    self.M,
                    self.K,
                    self.tile_size_input,
                    self.tile_size_output,
                    1,  # num_batches
                ),
                {
                    "verbose": mlir_verbose,
                    "kernel_object": self._kernel_link_file,
                    "epilogue": "none",
                },
            ),
        )

    def get_kernel_artifacts(self):
        return [
            KernelObjectArtifact(
                self._kernel_link_file,
                dependencies=[
                    SourceArtifact(
                        self.context.base_dir / "aie_kernels" / "generic" / "mv_f32.cc"
                    )
                ],
                extra_flags=[
                    f"-DDIM_K={self.K}",
                    f"-DVEC_SIZE={self.kernel_vector_size}",
                ],
            ),
        ]

    def get_arg_spec(self):
        return [
            AIERuntimeArgSpec("in", (self.M, self.K)),   # matrix (f32)
            AIERuntimeArgSpec("in", (self.K,)),           # vector (f32)
            AIERuntimeArgSpec("out", (self.M,)),          # output (f32)
        ]
