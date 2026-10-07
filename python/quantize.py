"""Quantize the FP32 classifier to INT8 (dynamic quantization) with ONNX Runtime
and compare model size, latency and accuracy against the FP32 version.

Usage: python python/quantize.py [--models models]
"""

import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic


def bench(session: ort.InferenceSession, x: np.ndarray, iters: int = 200) -> float:
    name = session.get_inputs()[0].name
    for _ in range(10):
        session.run(None, {name: x})
    t0 = time.perf_counter()
    for _ in range(iters):
        session.run(None, {name: x})
    return (time.perf_counter() - t0) * 1000 / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=str(Path(__file__).resolve().parent.parent / "models"))
    args = parser.parse_args()
    models = Path(args.models)
    fp32 = models / "classifier_fp32.onnx"
    int8 = models / "classifier_int8.onnx"

    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    s32 = ort.InferenceSession(str(fp32), opts, providers=["CPUExecutionProvider"])
    s8 = ort.InferenceSession(str(int8), opts, providers=["CPUExecutionProvider"])

    x = np.random.default_rng(0).normal(size=(256, 16)).astype(np.float32)
    y32 = s32.run(None, {"input": x})[0]
    y8 = s8.run(None, {"input": x})[0]

    size32, size8 = fp32.stat().st_size / 1024, int8.stat().st_size / 1024
    agreement = float((y32.argmax(1) == y8.argmax(1)).mean())
    max_err = float(np.abs(y32 - y8).max())
    t32, t8 = bench(s32, x), bench(s8, x)

    print(f"{'':12}{'FP32':>12}{'INT8':>12}")
    print(f"{'size (KB)':12}{size32:12.1f}{size8:12.1f}   ({size32 / size8:.1f}x smaller)")
    print(f"{'latency ms':12}{t32:12.3f}{t8:12.3f}   (batch 256, 1 thread)")
    print(f"top-1 agreement: {agreement * 100:.1f}%   max |logit diff|: {max_err:.4f}")
    print(f"wrote {int8}")


if __name__ == "__main__":
    main()
