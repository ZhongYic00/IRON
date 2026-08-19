# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import ClassVar, Dict
from ml_dtypes import bfloat16

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
class GEMVArgmaxBF16(MLIROperator):
    """bf16 GEMV with fused argmax — f32 accumulation, VEC_SIZE=64.

    Inputs are bf16 (fast), argmax uses f32 accumulation (precise).
    Output: 8 × (val, idx) = 16 floats. CPU does 8-way reduction.
    """

    M: int
    K: int
    num_aie_columns: int = 1
    tile_size_input: int = 4
    tile_size_output: int | None = None
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
        MLIROperator.__init__(self, context=self.context)

    @property
    def _kernel_link_file(self):
        return f"gemv_argmax_bf16_{self.K}k.o"

    def get_mlir_artifact(self):
        mlir_verbose = getattr(self.context, "mlir_verbose", False)
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "my_matvec_argmax",
                (
                    aie_utils.get_current_device(),
                    self.num_aie_columns,
                    self.M,
                    self.K,
                    self.tile_size_input,
                    self.tile_size_output,
                ),
                {
                    "verbose": mlir_verbose,
                    "kernel_object": self._kernel_link_file,
                },
            ),
        )

    def get_kernel_artifacts(self):
        return [
            KernelObjectArtifact(
                self._kernel_link_file,
                dependencies=[
                    SourceArtifact(
                        self.context.base_dir / "aie_kernels" / "generic" / "mv_argmax_bf16.cc"
                    )
                ],
                extra_flags=[
                    f"-DDIM_K={self.K}",
                    f"-DVEC_SIZE=64",  # bf16 vec size
                ],
            ),
        ]

    def get_arg_spec(self):
        cols = self.num_aie_columns
        return [
            AIERuntimeArgSpec("in", (self.M, self.K)),       # matrix (bf16)
            AIERuntimeArgSpec("in", (self.K,)),              # vector (bf16)
            AIERuntimeArgSpec("out", (cols * 2,)),           # 8×(val, idx) (f32)
        ]
