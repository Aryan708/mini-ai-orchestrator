"""Generate the small ONNX models used by the demo pipeline.

normalize.onnx        : (x - mean) / std over 16 features
classifier_fp32.onnx  : 16 -> 256 -> 256 -> 4 MLP with ReLU (fixed random weights)

Usage: python python/make_models.py [--out models]
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

FEATURES = 16
HIDDEN = 256
CLASSES = 4
OPSET = 17
IR_VERSION = 9  # keeps models loadable by older ONNX Runtime releases


def make_normalize(rng: np.random.Generator) -> onnx.ModelProto:
    mean = rng.normal(0.0, 0.1, FEATURES).astype(np.float32)
    std = rng.uniform(0.8, 1.2, FEATURES).astype(np.float32)
    nodes = [
        helper.make_node("Sub", ["input", "mean"], ["centered"]),
        helper.make_node("Div", ["centered", "std"], ["output"]),
    ]
    graph = helper.make_graph(
        nodes,
        "normalize",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", FEATURES])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", FEATURES])],
        initializer=[numpy_helper.from_array(mean, "mean"), numpy_helper.from_array(std, "std")],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)], ir_version=IR_VERSION)


def make_classifier(rng: np.random.Generator) -> onnx.ModelProto:
    dims = [FEATURES, HIDDEN, HIDDEN, CLASSES]
    initializers, nodes = [], []
    prev = "input"
    for i in range(len(dims) - 1):
        w = (rng.normal(0.0, 1.0, (dims[i], dims[i + 1])) / np.sqrt(dims[i])).astype(np.float32)
        b = rng.normal(0.0, 0.01, dims[i + 1]).astype(np.float32)
        initializers += [numpy_helper.from_array(w, f"W{i}"), numpy_helper.from_array(b, f"B{i}")]
        last = i == len(dims) - 2
        nodes.append(helper.make_node("MatMul", [prev, f"W{i}"], [f"mm{i}"]))
        nodes.append(helper.make_node("Add", [f"mm{i}", f"B{i}"], ["output" if last else f"lin{i}"]))
        if not last:
            nodes.append(helper.make_node("Relu", [f"lin{i}"], [f"act{i}"]))
            prev = f"act{i}"
    graph = helper.make_graph(
        nodes,
        "mlp_classifier",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", FEATURES])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", CLASSES])],
        initializer=initializers,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)], ir_version=IR_VERSION)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "models"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(7)
    for name, model in [("normalize.onnx", make_normalize(rng)), ("classifier_fp32.onnx", make_classifier(rng))]:
        onnx.checker.check_model(model)
        onnx.save(model, out / name)
        print(f"wrote {out / name} ({(out / name).stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
