# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from segment_anything import sam_model_registry
from segment_anything.utils.onnx_freespace import SamVit4RoadSegOnnx, SegHeadOnnx, SegHead

import argparse
import warnings

try:
    import onnxruntime  # type: ignore

    onnxruntime_exists = True
except ImportError:
    onnxruntime_exists = False

parser = argparse.ArgumentParser(
    description="Export the SAM prompt encoder and mask decoder to an ONNX model."
)

parser.add_argument(
    "--sam-ckpt", type=str, required=True, help="The path to the SAM model checkpoint."
)

parser.add_argument(
    "--seghead-ckpt", type=str, required=True, help="The path to the seghead model checkpoint."
)

parser.add_argument(
    "--output", type=str, required=True, help="The filename to save the ONNX model to."
)

parser.add_argument(
    "--model-type",
    type=str,
    required=True,
    help="In ['default', 'vit_h', 'vit_l', 'vit_b']. Which type of SAM model to export.",
)

parser.add_argument(
    "--opset",
    type=int,
    default=17,
    help="The ONNX opset version to use. Must be >=11",
)


def run_export(
    model_type: str,
    sam_ckpt: str,
    seghead_ckpt: str,
    output: str,
    opset: int,
):
    print("Loading model...")
    sam = sam_model_registry[model_type](checkpoint=sam_ckpt)
    vit_encoder = sam.image_encoder
    vit_model = SamVit4RoadSegOnnx(vit_model=vit_encoder)
    seg_decoder = SegHead()
    seg_decoder.load_state_dict(torch.load(seghead_ckpt, map_location='cpu'))
    seg_decoder.eval()
    seghead_model = SegHeadOnnx(model=seg_decoder)
    dummy_inputs = {
        "image": torch.randn(1, 3, 1024, 1024, dtype=torch.float)
    }
    dummy_inputs2 = {
        "image_embeddings": torch.randn(1, 256, 64, 64, dtype=torch.float),
        "inner_state": torch.randn(32, 1, 64, 64, 1280, dtype=torch.float),
        "ori_size": torch.tensor([1080, 1920], dtype=torch.int32)
    }
    output_names1 = ["vit_embedding", "inner_state"]
    output_names2 = ["masks"]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        seghead_output = output+'_decoder.onnx'
        with open(seghead_output, "wb") as f:
            print(f"Exporting onnx model to {seghead_output}...")
            torch.onnx.export(
                seghead_model,
                tuple(dummy_inputs2.values()),
                f,
                export_params=True,
                verbose=False,
                opset_version=opset,
                do_constant_folding=True,
                input_names=list(dummy_inputs2.keys()),
                output_names=output_names2,
                # dynamic_axes=dynamic_axes,
            )
        vit_output = output+'_encoder.onnx'
        with open(vit_output, "wb") as f:
            print(f"Exporting onnx model to {vit_output}...")
            torch.onnx.export(
                vit_model,
                tuple(dummy_inputs.values()),
                'weights/tmp/'+vit_output,
                export_params=True,
                verbose=True,
                opset_version=opset,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys()),
                output_names=output_names1,
                # use_external_data_format=True
            )

    if onnxruntime_exists:
        ort_inputs = {k: to_numpy(v) for k, v in dummy_inputs2.items()}
        # set cpu provider default
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        ort_session = onnxruntime.InferenceSession(seghead_output, providers=providers)
        _ = ort_session.run(None, ort_inputs)
        print("Model has successfully been run with ONNXRuntime.")


def to_numpy(tensor):
    return tensor.cpu().numpy()


if __name__ == "__main__":
    args = parser.parse_args()
    run_export(
        model_type=args.model_type,
        sam_ckpt=args.sam_ckpt,
        seghead_ckpt=args.seghead_ckpt,
        output=args.output,
        opset=args.opset
    )
    print("Done!")
