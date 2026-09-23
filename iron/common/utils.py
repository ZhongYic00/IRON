# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from aie.dialects.aie import get_target_model, WireBundle
from aie.utils.hostruntime.tensor_class import NpuTensor
from aie.utils.hostruntime.tensor_class import NpuTensor
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor


def get_shim_dma_limit(dev) -> int:
    """Return the total number of ShimDMA output channels available on the device.

    Each shim tile exposes a fixed number of DMA source connections; summing
    across all shim tiles gives the device-wide ShimDMA budget.
    """
    tm = get_target_model(dev.resolve())
    return sum(
        tm.get_num_source_shim_mux_connections(col, row, WireBundle.DMA)
        for col in range(tm.columns())
        for row in range(tm.rows())
        if tm.is_shim_noc_or_pl_tile(col, row)
    )


def float_to_name(v: float) -> str:
    """Convert a float to a filesystem-safe string for use in operator names.

    Uses repr() for the shortest exact round-trip representation, then sanitizes
    characters that are problematic in filenames or shell scripts:
      '.' -> 'p'  (decimal point)
      '-' -> 'n'  (negative sign / negative exponent)
      '+' -> ''   (positive exponent, redundant)

    Examples:
      3.0   -> '3p0'
      0.01  -> '0p01'
      -0.5  -> 'n0p5'
      1e-10 -> '1en10'
    """
    return repr(v).replace(".", "p").replace("-", "n").replace("+", "")


class XRTSubBuffer(XRTTensor):
    """
    A view into a sub-region of an XRTTensor's underlying storage.

    Inherits from XRTTensor so that isinstance checks in the runtime pass.
    Implemented on the wheel's native view contract: the sub-view SHARES the
    parent's Storage (bytes + coherence map), so residency is tracked per byte
    range on one allocation — a host write through this view's torch_view/data
    marks its own range host-dirty, the parent's ``to("npu")`` syncs exactly the
    dirty ranges (no whole-parent clobbering), and a device-side rewrite is
    pulled back range-wise by ``to("cpu")``.

    The parent XRTTensor must remain alive as long as this sub-buffer is in use
    (held via ``_parent``).
    """

    def __init__(self, parent_bo, offset_bytes, size_bytes, shape, dtype, parent=None):
        """
        Args:
            parent_bo: The parent pyxrt.bo object (unused; kept for call-site
                compatibility — the region is derived from the parent tensor's
                shared storage, which is what the coherence model requires).
            offset_bytes: Byte offset into the parent buffer.
            size_bytes: Byte size of this sub-region (must match shape/dtype).
            shape: Logical shape of this sub-buffer.
            dtype: numpy dtype for interpreting the buffer contents.
            parent: The parent XRTTensor this sub-buffer views into (required:
                the coherence model tracks residency on the shared allocation,
                which only the parent tensor carries).
        """
        if parent is None:
            raise ValueError(
                "XRTSubBuffer requires the parent XRTTensor (residency is "
                "tracked per range on the shared allocation, which a raw bo "
                "handle cannot provide)")
        dtype = np.dtype(dtype)
        shape = tuple(shape)
        nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if nbytes != size_bytes:
            raise ValueError(
                f"XRTSubBuffer: size_bytes={size_bytes} does not match "
                f"shape={shape} x {dtype} ({nbytes} bytes)")
        # NpuTensor contract fields, mirroring the wheel's XRTTensor._subview:
        # share the parent's storage and byte range instead of allocating.
        NpuTensor.__init__(self, shape, dtype=dtype, device=parent.device)
        self._parent = parent
        self._shape = shape
        self._storage = parent.storage
        self._offset_bytes = parent.storage_offset + offset_bytes
        self._bo = parent.storage.binding_handle(self._offset_bytes, nbytes)
        self._data = (
            parent.storage.host_bytes[
                self._offset_bytes : self._offset_bytes + nbytes
            ]
            .view(self.dtype)
            .reshape(self._shape)
        )

    @classmethod
    def from_parent(cls, parent, shape, offset_elements, length_elements, dtype):
        """Create an XRTSubBuffer into a sub-region of a parent XRTTensor.

        Accepts element-count offsets/lengths and converts to bytes internally.
        """
        itemsize = np.dtype(dtype).itemsize
        return cls(
            parent_bo=parent.buffer_object(),
            offset_bytes=offset_elements * itemsize,
            size_bytes=length_elements * itemsize,
            shape=shape,
            dtype=dtype,
            parent=parent,
        )
