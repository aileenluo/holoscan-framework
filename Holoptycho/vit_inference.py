"""
PtychoViT inference operators for Holoscan pipeline.

Runs PtychoViT neural network inference in parallel with the iterative
PtychoRecon solver using Holoscan's native InferenceOp.

Pipeline topology (three DAG nodes):

    source (diff_amp, image_indices)
        │
        ▼
    VitPreprocessOp ──[preprocessed]──► InferenceOp ──[transmitter]──► VitSaveOp
             │                                                              ▲
             └──────────────────[metadata]──────────────────────────────────┘

VitPreprocessOp  — fftshift + padding + tensor wrapping
InferenceOp      — Holoscan native TRT inference (wired in ptycho_holo.py)
VitSaveOp        — strips padding, reshapes, saves per-batch .npy files

Usage:
    See ptycho_holo.py for wiring into PtychoApp / PtychoSimulApp.
"""

import collections
import logging
import os

import numpy as np

from holoscan.core import Operator, OperatorSpec, ConditionType, IOSpec


class VitPreprocessOp(Operator):
    """Preprocesses diffraction batches for Holoscan InferenceOp.

    Inputs:
        diff_amp:      [B, H, W] float32 — preprocessed diffraction amplitude
        image_indices: [B] int32 — frame indices

    Outputs:
        preprocessed: dict {"input": [B, 1, H_engine, W_engine] float32}
                      → InferenceOp "receivers" port
        metadata:     (indices, B_actual, spatial_pad)
                      → VitSaveOp "metadata" port

    Parameters:
        batch_size:       Engine batch size (default 64)
        model_h:          Engine input height (default 256)
        model_w:          Engine input width  (default 256)
        data_is_shifted:  If True, undo fftshift before feeding model
                          (live mode: ImagePreprocessorOp shifts data;
                           simulate mode: data is already unshifted)
    """

    def __init__(
        self,
        fragment,
        *args,
        batch_size: int = 64,
        model_h: int = 256,
        model_w: int = 256,
        data_is_shifted: bool = False,
        **kwargs,
    ):
        super().__init__(fragment, *args, **kwargs)
        self._logger = logging.getLogger("VitPreprocessOp")
        self._B_engine = batch_size
        self._H_engine = model_h
        self._W_engine = model_w
        self._data_is_shifted = data_is_shifted

    def setup(self, spec: OperatorSpec):
        spec.input("diff_amp").connector(
            IOSpec.ConnectorType.DOUBLE_BUFFER, capacity=32
        )
        spec.input("image_indices").connector(
            IOSpec.ConnectorType.DOUBLE_BUFFER, capacity=32
        )
        spec.output("preprocessed")  # → InferenceOp receivers
        spec.output("metadata").condition(ConditionType.NONE)  # → VitSaveOp

    def compute(self, op_input, op_output, context):
        diff_amp = op_input.receive("diff_amp")
        indices = op_input.receive("image_indices")

        if diff_amp is None:
            return

        # Undo fftshift when data arrives pre-shifted (live mode).
        # Model was trained on unshifted data (DC at corners).
        # For even-sized arrays fftshift == ifftshift, so applying it again
        # restores the unshifted layout.
        if self._data_is_shifted:
            diff_amp = np.fft.fftshift(diff_amp, axes=(1, 2))

        B_actual = diff_amp.shape[0]
        H_data = diff_amp.shape[1]
        W_data = diff_amp.shape[2]
        B_engine = self._B_engine
        H_engine = self._H_engine
        W_engine = self._W_engine

        model_input = diff_amp[:, np.newaxis, :, :]  # [B, 1, H, W]

        # Spatial padding: center-pad if data is smaller than engine input.
        spatial_pad = None
        if H_data != H_engine or W_data != W_engine:
            pad_h = H_engine - H_data
            pad_w = W_engine - W_data
            if pad_h < 0 or pad_w < 0:
                self._logger.error(
                    "Data spatial dims (%d,%d) larger than engine (%d,%d). "
                    "Skipping batch.",
                    H_data, W_data, H_engine, W_engine,
                )
                return
            top = pad_h // 2
            left = pad_w // 2
            spatial_pad = (top, top + H_data, left, left + W_data)
            padded = np.zeros(
                (B_actual, 1, H_engine, W_engine), dtype=np.float32
            )
            padded[:, :, top:top + H_data, left:left + W_data] = model_input
            model_input = padded

        # Batch padding: pad final (smaller) batch to engine batch size.
        if B_actual < B_engine:
            pad = np.zeros(
                (B_engine - B_actual, 1, H_engine, W_engine), dtype=np.float32
            )
            model_input = np.concatenate([model_input, pad], axis=0)
        elif B_actual > B_engine:
            self._logger.error(
                "Batch too large: input %d vs engine %d. "
                "Check that ONNX batch size matches ImageBatchOp batchsize.",
                B_actual, B_engine,
            )
            return

        model_input = np.ascontiguousarray(model_input, dtype=np.float32)

        op_output.emit({"input": model_input}, "preprocessed")
        op_output.emit((indices, B_actual, spatial_pad), "metadata")


class VitSaveOp(Operator):
    """Postprocessor and saver for VIT inference output.

    Receives:
        infer_output: TensorMap from InferenceOp "transmitter" port
        metadata:     (indices, B_actual, spatial_pad) from VitPreprocessOp

    Metadata and inference output arrive on separate paths; they are
    correlated via an in-order queue (one metadata entry per batch,
    matched FIFO with the corresponding inference result).

    Writes (O(1) per batch, no full-scan accumulation):
        vit_batch_000000_pred.npy    — predictions [B, 2, H, W] or [B, H, W]
        vit_batch_000000_indices.npy — frame indices [B]
        vit_pred_latest.npy          — most recent batch (atomic write)

    Detects new scans by watching for frame indices that reset toward 0,
    then clears old batch files.

    Parameters:
        save_dir: Directory for saving predictions (default /data/users/Holoscan)
    """

    def __init__(self, fragment, *args, save_dir="/data/users/Holoscan", **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self._logger = logging.getLogger("VitSaveOp")
        self.save_dir = save_dir
        self._meta_queue = collections.deque()
        self.batch_num = 0
        self.max_index_seen = -1

    def setup(self, spec: OperatorSpec):
        spec.input("infer_output").connector(
            IOSpec.ConnectorType.DOUBLE_BUFFER, capacity=32
        )
        spec.input("metadata").connector(
            IOSpec.ConnectorType.DOUBLE_BUFFER, capacity=32
        )

    def _clear_old_batches(self):
        import glob as globmod
        for pattern in ["vit_batch_*_pred.npy", "vit_batch_*_indices.npy"]:
            for f in globmod.glob(os.path.join(self.save_dir, pattern)):
                try:
                    os.remove(f)
                except OSError:
                    pass

    def compute(self, op_input, op_output, context):
        try:
            self._compute_inner(op_input, op_output, context)
        except Exception:
            self._logger.exception("VitSaveOp failed (pipeline continues)")

    def _compute_inner(self, op_input, op_output, context):
        meta = op_input.receive("metadata")
        infer_out = op_input.receive("infer_output")

        if meta is not None:
            self._meta_queue.append(meta)

        if infer_out is None or not self._meta_queue:
            return

        indices, B_actual, spatial_pad = self._meta_queue.popleft()

        # Extract output tensor from InferenceOp TensorMap.
        # InferenceOp emits a dict-like TensorMap; access by tensor name.
        raw = infer_out["output"]
        pred = np.asarray(raw)  # handles both numpy arrays and holoscan tensors

        # Strip batch padding
        pred = pred[:B_actual]

        # Strip spatial padding
        if spatial_pad is not None:
            top, bot, left, right = spatial_pad
            if pred.ndim == 4:  # [B, 2, H, W]
                pred = pred[:, :, top:bot, left:right]
            else:               # [B, H, W]
                pred = pred[:, top:bot, left:right]

        os.makedirs(self.save_dir, exist_ok=True)

        # Detect new scan: smallest index in batch below previously seen max
        min_idx = int(indices.min())
        if min_idx < self.max_index_seen and self.batch_num > 0:
            self._clear_old_batches()
            self.batch_num = 0
            self.max_index_seen = -1

        self.max_index_seen = max(self.max_index_seen, int(indices.max()))

        np.save(
            os.path.join(self.save_dir, f"vit_batch_{self.batch_num:06d}_pred.npy"),
            pred,
        )
        np.save(
            os.path.join(self.save_dir, f"vit_batch_{self.batch_num:06d}_indices.npy"),
            indices,
        )

        # Atomic write of latest batch for quick inspection
        tmp = os.path.join(self.save_dir, "_vit_pred_latest.tmp.npy")
        np.save(tmp, pred)
        os.replace(tmp, os.path.join(self.save_dir, "vit_pred_latest.npy"))

        self.batch_num += 1
