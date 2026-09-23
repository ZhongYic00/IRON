# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from .op import SwiGLUMLPDataParallel
from .op_ours import SwiGLUMLPDataParallelOurs

__all__ = ["SwiGLUMLPDataParallel", "SwiGLUMLPDataParallelOurs"]
