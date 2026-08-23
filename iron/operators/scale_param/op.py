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
class ScaleByParam(MLIROperator):
    """Scale a vector by a runtime scratchpad parameter (int32 broadcast to bf16).

    Used to verify the ScratchpadParameter end-to-end path inside an
    OperatorSequence: host writes ``param_name`` via ParameterScratchpad,
    kernel reads it and computes out = in * factor.
    """

    size: int
    tile_size: int
    param_name: str = "scale_param"
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {
        **MLIROperator._name_aliases,
        "param_name": "p",
    }

    def __post_init__(self):
        MLIROperator.__init__(self, context=self.context)

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "scale_param_design",
                (),
                {
                    "dev": aie_utils.get_current_device(),
                    "size": self.size,
                    "tile_size": self.tile_size,
                    "trace_size": 0,
                    "param_name": self.param_name,
                    "kernel_obj_file": "scale_param.o",
                },
            ),
        )

    def get_kernel_artifacts(self):
        return [
            KernelObjectArtifact(
                "scale_param.o",
                dependencies=[
                    SourceArtifact(
                        self.context.base_dir / "aie_kernels" / "aie2p" / "scale_param.cc"
                    )
                ],
            )
        ]

    def get_arg_spec(self):
        return [
            AIERuntimeArgSpec("in", (self.size,)),
            AIERuntimeArgSpec("out", (self.size,)),
        ]
