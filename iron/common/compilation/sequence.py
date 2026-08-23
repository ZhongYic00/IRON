# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Temporal fusion of multiple MLIR modules into one module with multiple devices and a main runtime sequence that calls into them.
"""

from __future__ import annotations

import numpy as np
import importlib.util
from functools import partial
from pathlib import Path
from aie import ir
from aie.dialects import aie, aiex, memref
from aie.extras.context import mlir_mod_ctx
import ml_dtypes

from typing import Any

from . import (
    CompilationArtifactGraph,
    CompilationRule,
    CompilationCommand,
    PythonCallbackCompilationCommand,
    PythonGeneratedMLIRArtifact,
    MLIRArtifact,
)

# ---------------------------------------------------------------------------
# mlir-aie 1.4.0 compatibility shim
#
# In 1.4.0 the Runtime class API changed: Runtime(seq_fn, fn_args) where
# seq_fn is a plain callback and fn_args entries that are types become
# RuntimeData.  The old API (Runtime() with no args, then rt.sequence(),
# rt.start(), rt.fill(), rt.drain(), rt.task_group(), etc.) no longer works.
#
# Many IRON design modules (binary_elementwise_design, channeled_unary_design,
# softmax, gemv, etc.) still use the old API.  This shim patches Runtime so
# the old-style calls are accepted: operations are recorded during the
# rt.sequence() context-manager body and replayed inside the real seq_fn
# callback when Program.resolve_program() calls rt.resolve().
#
# The shim also patches ShellCompilationCommand.run to strip aiecc flags that
# 1.4.0's aiecc does not recognise (--no-compile-host, --aie-generate-xclbin,
# --aie-generate-npu-insts, --no-compile) and adds --get-xclbin / --get-npu-insts
# shorthands.  This is the same patch already applied by mha/design.py and
# gemm/design.py; the guard prevents double-patching.
# ---------------------------------------------------------------------------

import contextlib
import itertools
import os

_1_4_0_patch_done = False


def _patch_for_1_4_0():
    global _1_4_0_patch_done
    if _1_4_0_patch_done:
        return
    _1_4_0_patch_done = True

    from typing import get_origin
    import numpy as _np

    from aie.iron.runtime.runtime import Runtime as _Runtime
    from aie.iron.program import Program as _Program
    from aie.iron.runtime.data import RuntimeData
    from aie.iron.runtime.taskgroup import TaskGroup as _TaskGroup
    from aie.iron.runtime.endpoint import RuntimeEndpoint
    from aie.iron.runtime.runtime import sync_parameters as _sync_parameters

    _orig_runtime_init = _Runtime.__init__
    _orig_resolve = _Runtime.resolve
    _orig_program_init = _Program.__init__

    # -- Placeholder / recording helpers -----------------------------------

    class _Placeholder:
        """Stand-in for a runtime input, replaced by the real SSA value
        during ``resolve()`` replay."""
        __slots__ = ("idx", "type_", "real")

        def __init__(self, idx, type_):
            self.idx = idx
            self.type_ = type_
            self.real = None

    class _RecordedTaskGroup:
        """Recording-phase stand-in for a TaskGroup; swapped for the real
        TaskGroup during replay."""
        pass

    def _convert_kwargs(kwargs, tg_map):
        """Rename old-style ``task_group=`` to 1.4.0 ``group=``."""
        out = {}
        for k, v in kwargs.items():
            if k == "task_group":
                out["group"] = tg_map.get(v, v)
            else:
                out[k] = v
        return out

    # -- Runtime.__init__ shim ---------------------------------------------

    def _compat_init(self, seq_fn=None, fn_args=None, *, strict_task_groups=True):
        if seq_fn is not None:
            # New 1.4.0 API — delegate to the original constructor.
            _orig_runtime_init(self, seq_fn, fn_args, strict_task_groups=strict_task_groups)
            return

        # Old API: Runtime() with no arguments.
        # Fields are populated lazily by sequence() / fill() / drain() …
        # and finalised in _compat_resolve().
        self._compat_types = None
        self._compat_workers = []
        self._compat_trace = None
        self._compat_recorded = []
        self._compat_placeholders = []

        # Initialise the fields that Program.resolve_program() inspects so
        # they exist even before _compat_resolve() populates the real ones.
        self._seq_fn = None
        self._fn_args = []
        self._const_inputs = []
        self._rt_data = []
        self._fifos = set()
        self._flows = []
        self._locks = []
        self._tile_dmas = []
        self._scratchpad_parameters = []
        self._strict_task_groups = strict_task_groups
        self._task_group_index = itertools.count()

    # -- Old-style Runtime methods (record for later replay) ---------------

    def _sequence(self, *types):
        self._compat_types = types
        self._compat_placeholders = [_Placeholder(i, t) for i, t in enumerate(types)]

        @contextlib.contextmanager
        def _ctx():
            yield tuple(self._compat_placeholders)

        return _ctx()

    def _start(self, *workers):
        self._compat_workers = list(workers)

    def _fill(self, handle, source, *args, **kwargs):
        self._compat_recorded.append(("fill", handle, source, args, kwargs))
        if hasattr(handle, "_object_fifo"):
            if handle.endpoint is None:
                handle.endpoint = RuntimeEndpoint(handle._shim_tile)
            self._fifos.add(handle)

    def _drain(self, handle, dest, *args, **kwargs):
        self._compat_recorded.append(("drain", handle, dest, args, kwargs))
        if hasattr(handle, "_object_fifo"):
            if handle.endpoint is None:
                handle.endpoint = RuntimeEndpoint(handle._shim_tile)
            self._fifos.add(handle)

    def _task_group(self):
        tg = _RecordedTaskGroup()
        self._compat_recorded.append(("task_group", tg))
        return tg

    def _finish_task_group(self, tg):
        self._compat_recorded.append(("finish_task_group", tg))

    def _record_sync_parameters(self):
        self._compat_recorded.append(("sync_parameters",))

    def _inline_ops(self, fn, *args):
        self._compat_recorded.append(("inline_ops", fn, args))

    def _set_barrier(self, barrier, value):
        self._compat_recorded.append(("set_barrier", barrier, value))

    def _enable_trace(self, trace_size, workers=None, **kwargs):
        self._compat_trace = (trace_size, workers, kwargs)

    # -- Runtime.resolve shim ----------------------------------------------
    # When the Runtime was created with the old API (no seq_fn), build a
    # seq_fn callback from the recorded operations, populate the fields the
    # original resolve() expects, then delegate to the original resolve().

    def _compat_resolve(
        self,
        loc=None,
        ip=None,
        *,
        trace_size=None,
        reuse_output_buffer=False,
        egress_shim_col=0,
        load_pdi_device_ref=None,
    ):
        if not hasattr(self, "_compat_types"):
            # New API — call original resolve directly.
            _orig_resolve(
                self,
                loc=loc,
                ip=ip,
                trace_size=trace_size,
                reuse_output_buffer=reuse_output_buffer,
                egress_shim_col=egress_shim_col,
                load_pdi_device_ref=load_pdi_device_ref,
            )
            return

        types = self._compat_types or []
        placeholders = self._compat_placeholders
        recorded = self._compat_recorded
        tg_map = {}

        def seq_fn(*real_args):
            # Bind placeholders to the real SSA / RuntimeData values.
            for ph, real in zip(placeholders, real_args):
                ph.real = real

            for record in recorded:
                op = record[0]
                if op == "fill":
                    _, handle, source, args, kwargs = record
                    if isinstance(source, _Placeholder):
                        source = source.real
                    handle.fill(source, *args, **_convert_kwargs(kwargs, tg_map))
                elif op == "drain":
                    _, handle, dest, args, kwargs = record
                    if isinstance(dest, _Placeholder):
                        dest = dest.real
                    handle.drain(dest, *args, **_convert_kwargs(kwargs, tg_map))
                elif op == "task_group":
                    _, tg_ph = record
                    tg_map[tg_ph] = _TaskGroup()
                elif op == "finish_task_group":
                    _, tg_ph = record
                    tg_map[tg_ph].finish()
                elif op == "sync_parameters":
                    _sync_parameters()
                elif op == "inline_ops":
                    _, fn, args = record
                    fn(*args)
                elif op == "set_barrier":
                    _, barrier, value = record
                    barrier.set(value)

        # Populate the fields that original __init__ would have set.
        self._seq_fn = seq_fn
        self._fn_args = list(types)
        self._const_inputs = [
            v if isinstance(v, (int, _np.integer)) and not isinstance(v, bool) else None
            for v in self._fn_args
        ]
        self._rt_data = [
            RuntimeData(arg)
            if c is None
            and (isinstance(arg, type) or get_origin(arg) is _np.ndarray)
            else None
            for c, arg in zip(self._const_inputs, self._fn_args)
        ]
        self._register_fn_args()

        _orig_resolve(
            self,
            loc=loc,
            ip=ip,
            trace_size=trace_size,
            reuse_output_buffer=reuse_output_buffer,
            egress_shim_col=egress_shim_col,
            load_pdi_device_ref=load_pdi_device_ref,
        )

    # Apply Runtime patches.
    _Runtime.__init__ = _compat_init
    _Runtime.sequence = _sequence
    _Runtime.start = _start
    _Runtime.fill = _fill
    _Runtime.drain = _drain
    _Runtime.task_group = _task_group
    _Runtime.finish_task_group = _finish_task_group
    _Runtime.sync_parameters = _record_sync_parameters
    _Runtime.inline_ops = _inline_ops
    _Runtime.set_barrier = _set_barrier
    _Runtime.enable_trace = _enable_trace
    _Runtime.resolve = _compat_resolve

    # -- Program.__init__ shim ---------------------------------------------
    # Extract workers and trace config from a compat Runtime so that
    # Program(dev, rt) works without explicit workers=.

    def _compat_program_init(self, device, rt, workers=None):
        if workers is None and hasattr(rt, "_compat_workers"):
            workers = rt._compat_workers
        _orig_program_init(self, device, rt, workers)
        if hasattr(rt, "_compat_trace") and rt._compat_trace is not None:
            trace_size, trace_workers, kwargs = rt._compat_trace
            self.enable_trace(trace_size=trace_size, workers=trace_workers, **kwargs)

    _Program.__init__ = _compat_program_init

    # -- aiecc flag patching ------------------------------------------------

    import iron.common.compilation.base as _base

    _REMOVE_FLAGS = {
        "--no-compile-host",
        "--aie-generate-xclbin",
        "--aie-generate-npu-insts",
        "--no-compile",
    }

    def _fix_cmd(cmd_list):
        """Fix aiecc command for 1.4.0 compatibility:
        - Replace --generate-full-elf with --get-full-elf (1.4.0 syntax)
        - Convert --full-elf-name <path> to --full-elf-name=<path>
        - Remove flags not supported by 1.4.0 aiecc
        - Ensure /opt/xilinx/xrt/bin is on PATH for aiebu-asm
        """
        fixed = [c for c in cmd_list if c not in _REMOVE_FLAGS]
        # Replace --generate-full-elf with --get-full-elf
        fixed = ["--get-full-elf" if c == "--generate-full-elf" else c for c in fixed]
        # Convert "--full-elf-name <path>" to "--full-elf-name=<path>"
        for i, c in enumerate(fixed):
            if c == "--full-elf-name" and i + 1 < len(fixed):
                fixed[i] = f"--full-elf-name={fixed[i + 1]}"
                fixed[i + 1] = None  # remove next arg (already consumed)
        fixed = [c for c in fixed if c is not None]
        has_xclbin = any(c.startswith("--xclbin-name=") for c in fixed)
        has_insts = any(c.startswith("--npu-insts-name=") for c in fixed)
        if has_xclbin and "--get-xclbin" not in fixed:
            idx = next(
                i for i, c in enumerate(fixed) if c.startswith("--xclbin-name=")
            )
            fixed.insert(idx, "--get-xclbin")
        if has_insts and "--get-npu-insts" not in fixed:
            idx = next(
                i for i, c in enumerate(fixed) if c.startswith("--npu-insts-name=")
            )
            fixed.insert(idx, "--get-npu-insts")
        return fixed

    if not hasattr(_base.ShellCompilationCommand, "_aiecc_patched"):
        _orig_run = _base.ShellCompilationCommand.run

        def _patched_run(self):
            if hasattr(self, "command") and isinstance(self.command, list):
                if self.command and "aiecc" in self.command[0]:
                    self.command = _fix_cmd(self.command)
                    xrt_bin = "/opt/xilinx/xrt/bin"
                    if hasattr(self, "env") and isinstance(self.env, dict):
                        path_val = self.env.get("PATH", "")
                        if xrt_bin not in path_val:
                            self.env["PATH"] = xrt_bin + ":" + path_val
            return _orig_run(self)

        _base.ShellCompilationCommand.run = _patched_run
        _base.ShellCompilationCommand._aiecc_patched = True


_patch_for_1_4_0()

RESET_DEVICE = "reset_device"


# Compilation Artifacts
# ##########################################################################


class SequenceMLIRArtifact(MLIRArtifact):
    def __init__(
        self,
        filename: str,
        operator_mlir_map: dict[str, PythonGeneratedMLIRArtifact],
        runlist: list[tuple[str, ...]],
        subbuffer_layout: dict[str, tuple[str, int, int]],
        buffer_sizes: tuple[int, int, int],
        slice_info: dict[str, tuple[str, int, int]] | None = None,
        independent_order: list[str] | None = None,
        independent_sizes: list[int] | None = None,
    ) -> None:
        dependencies = list(operator_mlir_map.values())
        super().__init__(filename, dependencies)
        self.operator_mlir_map = operator_mlir_map
        self.runlist = runlist
        self.subbuffer_layout = subbuffer_layout
        self.buffer_sizes = buffer_sizes
        self.slice_info = slice_info or {}
        self.independent_order = independent_order or []
        self.independent_sizes = independent_sizes or []


# Helper Functions
# ##########################################################################


def extract_runtime_sequence_arg_types(dev_op: Any) -> list[Any]:
    """MLIR helper: Extract argument types from a device operation's runtime sequence."""
    for nested_op in dev_op.body_region.blocks[0].operations:
        op_name = nested_op.operation.name
        if op_name == "aie.runtime_sequence":
            if hasattr(nested_op, "body") and hasattr(nested_op.body, "blocks"):
                if len(nested_op.body.blocks) > 0:
                    entry_block = nested_op.body.blocks[0]
                    arg_types = [
                        entry_block.arguments[i].type
                        for i in range(len(entry_block.arguments))
                    ]
                    return arg_types
    raise RuntimeError("Could not find runtime sequence in device operation")


def get_child_mlir_module(mlir_artifact: PythonGeneratedMLIRArtifact) -> Any:
    """Extract MLIR module from a PythonGeneratedMLIRArtifact.

    Uses the artifact's DesignGenerator to dynamically import the design
    module and call the callback, returning the raw (non-stringified) MLIR
    module object for further inspection by the fusion pass.
    """
    if not isinstance(mlir_artifact, PythonGeneratedMLIRArtifact):
        raise TypeError(
            f"Expected PythonGeneratedMLIRArtifact, got {type(mlir_artifact).__name__}"
        )
    gen = mlir_artifact.generator
    spec = importlib.util.spec_from_file_location(gen.source_path.name, gen.source_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    callback_function = getattr(module, gen.fn_name)
    return callback_function(*gen.args, **gen.kwargs)


def needs_additional_reset(runlist: list[Any]) -> bool:
    """Whether the sequence must configure one more device than the runlist asks for.

    ``aiecc --expand-load-pdis`` marks each configure point by loading one of two
    otherwise empty PDIs, alternating between them from a fixed start. A load of the
    PDI already loaded has no effect, so a sequence with an odd number of configure
    points ends on the one the next dispatch starts with, and that dispatch
    reconfigures over the state the last design left. Configuring one more device
    makes the count even. Consecutive entries running the same operator share a
    configure point.
    """
    points = 0
    previous = None
    for op_name, *_ in runlist:
        if op_name != previous:
            points += 1
            previous = op_name
    return points % 2 == 1


def fuse_mlir(artifact: SequenceMLIRArtifact) -> None:
    """Fuse multiple MLIR modules by inlining their device operations and adding a new main device and runtime sequence that call into sequence of operations based on a runlist."""

    input_buffer_size, output_buffer_size, scratch_buffer_size = artifact.buffer_sizes

    # Extract device operations and module-level parameter decls from each
    # operator's MLIR artifact.  Note: in the current MLIR-AIE pipeline,
    # ``aiex.scratchpad_parameter`` ops are emitted at *module* scope (above the
    # ``aie.device``), because the scratchpad is a single hardware resource
    # shared across all PDIs in a runlist and the verifier on
    # ``aiex.read_scratchpad_parameter`` requires the decl to be visible at module
    # scope.  We collect those module-level decls per-operator so we can
    # re-declare them once at the top of the fused module.
    device_mlir_strings = {}
    operator_param_decls: dict[str, dict[str, ir.Type]] = {}
    device_ty = None
    sequence_arg_types = {}
    for op_name, mlir_artifact in artifact.operator_mlir_map.items():
        mlir_module = get_child_mlir_module(mlir_artifact)
        device_ops = []
        params_here: dict[str, ir.Type] = {}
        for op in mlir_module.body.operations:
            if isinstance(op, aie.DeviceOp):
                device_ops.append(op)
            elif op.operation.name == "aiex.scratchpad_parameter":
                sym_name = ir.StringAttr(op.operation.attributes["sym_name"]).value
                param_type = ir.TypeAttr(op.operation.attributes["type"]).value
                params_here[sym_name] = param_type
        if len(device_ops) != 1:
            raise ValueError(
                f"Expected exactly one device operation in MLIR artifact for operator '{op_name}', "
                f"got {len(device_ops)}"
            )
        device_op = device_ops[0]
        if device_ty is None:
            device_ty = device_op.device
        device_mlir_strings[op_name] = str(device_op)
        operator_param_decls[op_name] = params_here
        sequence_arg_types[op_name] = extract_runtime_sequence_arg_types(device_op)

    # Deduplicate parameter decls across operators (same name must have the
    # same type; otherwise indices would collide in the global state table).
    hoisted_params: dict[str, ir.Type] = {}
    for op_name, params_here in operator_param_decls.items():
        for sym_name, param_type in params_here.items():
            existing = hoisted_params.get(sym_name)
            if existing is not None and str(existing) != str(param_type):
                raise ValueError(
                    f"ScratchpadParameter '{sym_name}' is declared with conflicting "
                    f"types across operators: {existing} vs {param_type}"
                )
            hoisted_params[sym_name] = param_type

    # Build fused MLIR module
    with mlir_mod_ctx() as ctx:

        # Emit hoisted parameters first.
        with ir.InsertionPoint.at_block_begin(ctx.module.body):
            for sym_name, param_type in hoisted_params.items():
                aiex.scratchpad_parameter(sym_name, param_type)

        # Concatenate aie.device ops.
        params_preamble = "\n".join(
            f"  aiex.scratchpad_parameter @{name} : {param_type}"
            for name, param_type in hoisted_params.items()
        )
        for op_name, device_str in device_mlir_strings.items():
            wrapped = f"module {{\n{params_preamble}\n{device_str}\n}}"
            wrapper_module = ir.Module.parse(wrapped)
            # Find the (sole) DeviceOp in the wrapper module.
            dev_op = None
            for op in wrapper_module.body.operations:
                if isinstance(op, aie.DeviceOp):
                    dev_op = op
                    break
            assert (
                dev_op is not None
            ), f"DeviceOp missing after re-parse for operator '{op_name}'"
            dev_op.sym_name = ir.StringAttr.get(op_name)
            ctx.module.body.append(dev_op)

        needs_reset = needs_additional_reset(artifact.runlist)
        if needs_reset:

            @aie.device(device_ty)
            def reset():
                @aiex.runtime_sequence()
                def sequence():
                    pass

            reset.operation.attributes["sym_name"] = ir.StringAttr.get(RESET_DEVICE)

        # Create the main device -- this contains the runtime sequence calling into the other devices
        @aie.device(device_ty)
        def main():
            buf_dtype = np.dtype[
                ml_dtypes.bfloat16
            ]  # TODO: support for other data types
            itemsize = np.dtype(ml_dtypes.bfloat16).itemsize

            # RuntimeSequenceOp
            seq_type_args = [
                np.ndarray[(input_buffer_size // itemsize,), buf_dtype],
                np.ndarray[(output_buffer_size // itemsize,), buf_dtype],
                np.ndarray[(scratch_buffer_size // itemsize,), buf_dtype],
            ]
            for name, size in zip(
                artifact.independent_order, artifact.independent_sizes
            ):
                seq_type_args.append(np.ndarray[(size // itemsize,), buf_dtype])

            @aiex.runtime_sequence(*seq_type_args)
            def sequence(*bufs):
                input_buf, output_buf, scratch_buf = bufs[0], bufs[1], bufs[2]
                consolidated_buffers = {
                    "input": input_buf,
                    "output": output_buf,
                    "scratch": scratch_buf,
                }
                for name, buf in zip(artifact.independent_order, bufs[3:]):
                    consolidated_buffers[name] = buf

                # Execute operations in runlist order
                configure_op = None
                last_op_name = None
                for op_name, *buffer_names in artifact.runlist:
                    expected_arg_types = sequence_arg_types[op_name]

                    # Avoid reconfiguring altogether if the same op is called multiple times consecutively
                    if configure_op is None or op_name != last_op_name:
                        # Configure Op
                        configure_sym_ref_attr = ir.FlatSymbolRefAttr.get(op_name)
                        configure_op = aiex.ConfigureOp(
                            configure_sym_ref_attr
                        )  # TODO: optimization -- if previous op was in the same device, skip reconfiguration
                        configure_body = configure_op.body.blocks.append()
                        last_op_name = op_name

                    with ir.InsertionPoint(configure_body):

                        # For each buffer, add subview and reinterpret_cast ops
                        buffer_ssa_values = []
                        for idx, buf_name in enumerate(buffer_names):
                            # Check if this is a sliced buffer
                            if buf_name in artifact.slice_info:
                                base_name, start, end = artifact.slice_info[buf_name]
                                # Get parent buffer info
                                buf_type, parent_offset, parent_length = (
                                    artifact.subbuffer_layout[base_name]
                                )
                                # Calculate actual offset and length for slice
                                offset = parent_offset + start
                                length = end - start
                            else:
                                # Regular buffer
                                buf_type, offset, length = artifact.subbuffer_layout[
                                    buf_name
                                ]

                            # Subview Op
                            consolidated_buf = consolidated_buffers[buf_type]
                            offset_elements = offset // itemsize
                            size_elements = length // itemsize
                            subview = memref.subview(
                                consolidated_buf,
                                [offset_elements],
                                [size_elements],
                                [1],
                            )

                            # Reinterpret_cast Op
                            target_type = expected_arg_types[idx]
                            expected_memref = ir.MemRefType(target_type)
                            target_shape = [
                                expected_memref.shape[i]
                                for i in range(expected_memref.rank)
                            ]
                            expected_size = np.prod(target_shape)
                            assert (
                                expected_size == size_elements
                            ), f"Size mismatch for buffer '{buf_name}': MLIR runtime sequence expected {expected_size}, Python fused operator provided {size_elements}"
                            strides = []
                            stride = 1
                            for dim in reversed(target_shape):
                                strides.insert(0, stride)
                                stride *= dim
                            result_type = ir.MemRefType.get(
                                target_shape, ir.BF16Type.get()
                            )
                            reinterpreted = memref.reinterpret_cast(
                                result=result_type,
                                source=subview,
                                offsets=[],
                                sizes=[],
                                strides=[],
                                static_offsets=[0],
                                static_sizes=target_shape,
                                static_strides=strides,
                            )
                            buffer_ssa_values.append(reinterpreted)

                        # Run Op
                        sequence_sym_ref_attr = ir.FlatSymbolRefAttr.get("sequence")
                        run_op = aiex.RunOp(sequence_sym_ref_attr, buffer_ssa_values)

                if needs_reset:
                    reset_op = aiex.ConfigureOp(ir.FlatSymbolRefAttr.get(RESET_DEVICE))
                    reset_op.body.blocks.append()

        # Write the fused MLIR to file
        with open(artifact.filename, "w") as f:
            f.write(str(ctx.module))


# Compilation Rules
# ##########################################################################


class FusePythonGeneratedMLIRCompilationRule(CompilationRule):
    """Compilation rule that fuses multiple MLIR modules into one."""

    def matches(self, graph: CompilationArtifactGraph) -> bool:
        return any(graph.get_worklist(SequenceMLIRArtifact))

    def compile(self, graph: CompilationArtifactGraph) -> list[CompilationCommand]:
        commands: list[CompilationCommand] = []
        worklist = graph.get_worklist(SequenceMLIRArtifact)
        for artifact in worklist:
            callback = partial(fuse_mlir, artifact)
            commands.append(PythonCallbackCompilationCommand(callback))
            artifact.available = True
        return commands
