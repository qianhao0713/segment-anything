# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from segment_anything import sam_model_registry
from segment_anything.utils.onnx import SamOnnxModel

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
    "--checkpoint", type=str, required=True, help="The path to the SAM model checkpoint."
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
    "--return-single-mask",
    action="store_true",
    help=(
        "If true, the exported ONNX model will only return the best mask, "
        "instead of returning multiple masks. For high resolution images "
        "this can improve runtime when upscaling masks is expensive."
    ),
)

parser.add_argument(
    "--opset",
    type=int,
    default=17,
    help="The ONNX opset version to use. Must be >=11",
)

parser.add_argument(
    "--quantize-out",
    type=str,
    default=None,
    help=(
        "If set, will quantize the model and save it with this name. "
        "Quantization is performed with quantize_dynamic from onnxruntime.quantization.quantize."
    ),
)

parser.add_argument(
    "--gelu-approximate",
    action="store_true",
    help=(
        "Replace GELU operations with approximations using tanh. Useful "
        "for some runtimes that have slow or unimplemented erf ops, used in GELU."
    ),
)

parser.add_argument(
    "--use-stability-score",
    action="store_true",
    help=(
        "Replaces the model's predicted mask quality score with the stability "
        "score calculated on the low resolution masks using an offset of 1.0. "
    ),
)

parser.add_argument(
    "--return-extra-metrics",
    action="store_true",
    help=(
        "The model will return five results: (masks, scores, stability_scores, "
        "areas, low_res_logits) instead of the usual three. This can be "
        "significantly slower for high resolution outputs."
    ),
)


def run_export(
    model_type: str,
    checkpoint: str,
    output: str,
    opset: int,
    return_single_mask: bool,
    gelu_approximate: bool = False,
    use_stability_score: bool = False,
    return_extra_metrics=False,
):
    print("Loading model...")
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    vit_encoder = sam.image_encoder

    onnx_model = SamOnnxModel(
        model=sam,
        return_single_mask=return_single_mask,
        use_stability_score=use_stability_score,
        return_extra_metrics=return_extra_metrics,
    )

    if gelu_approximate:
        for n, m in onnx_model.named_modules():
            if isinstance(m, torch.nn.GELU):
                m.approximate = "tanh"

    dynamic_axes = {
        "point_coords": {0: "num_batches", 1: "num_points"},
        "point_labels": {0: "num_batches", 1: "num_points"},
    }

    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size
    mask_input_size = [4 * x for x in embed_size]
    dummy_inputs2 = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
        "point_coords": torch.randint(low=0, high=1024, size=(1, 2, 2), dtype=torch.float),
        "point_labels": torch.randint(low=0, high=4, size=(1, 2), dtype=torch.float),
        "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
        "has_mask_input": torch.tensor([1], dtype=torch.float),
        "orig_im_size": torch.tensor([1080, 1920], dtype=torch.int32),
    }
    dummy_inputs = {
        "image": torch.randn(1, 3, 1024, 1024, dtype=torch.float)
    }
    _ = onnx_model(**dummy_inputs2)

    output_names = ["masks", "iou_predictions", "iou_token_out"]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        mask_decoder_output = output+'_mask_decoder.onnx'
        image_encoder_output = output+'_vit_encoder.onnx'
        with open(mask_decoder_output, "wb") as f:
            print(f"Exporting onnx model to {mask_decoder_output}...")
            torch.onnx.export(
                onnx_model,
                tuple(dummy_inputs2.values()),
                f,
                export_params=True,
                verbose=False,
                opset_version=opset,
                do_constant_folding=True,
                input_names=list(dummy_inputs2.keys()),
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )
        with open(image_encoder_output, "wb") as f:
            print(f"Exporting onnx model to {image_encoder_output}...")
            torch.onnx.export(
                vit_encoder,
                tuple(dummy_inputs.values()),
                f,
                export_params=True,
                verbose=True,
                opset_version=opset,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys())
                # use_external_data_format=True
            )

    if onnxruntime_exists:
        ort_inputs = {k: to_numpy(v) for k, v in dummy_inputs2.items()}
        # set cpu provider default
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        ort_session = onnxruntime.InferenceSession(mask_decoder_output, providers=providers)
        _ = ort_session.run(None, ort_inputs)
        print("Model has successfully been run with ONNXRuntime.")


def to_numpy(tensor):
    return tensor.cpu().numpy()


if __name__ == "__main__":
    args = parser.parse_args()
    run_export(
        model_type=args.model_type,
        checkpoint=args.checkpoint,
        output=args.output,
        opset=args.opset,
        return_single_mask=args.return_single_mask,
        gelu_approximate=args.gelu_approximate,
        use_stability_score=args.use_stability_score,
        return_extra_metrics=args.return_extra_metrics,
    )

    if args.quantize_out is not None:
        assert onnxruntime_exists, "onnxruntime is required to quantize the model."
        from onnxruntime.quantization import QuantType, QuantFormat  # type: ignore
        from onnxruntime.quantization.quantize import quantize_dynamic, quantize_static  # type: ignore
        from onnxruntime.quantization.calibrate import CalibrationMethod
        from segment_anything.calib.calib_dataloader import SamCalibrationDataReader

        print(f"Quantizing model and writing to {args.quantize_out}...")
        calib_loader = SamCalibrationDataReader()
        quantize_static(
            model_input='%s_mask_decoder.onnx' % args.output,
            model_output='%s_mask_decoder_int8.onnx' % args.quantize_out,
            calibration_data_reader=calib_loader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            optimize_model=False,
            use_external_data_format=False,
            calibrate_method=CalibrationMethod.MinMax,
            extra_options={
                # "CalibTensorRangeSymmetric": True,
		        "ForceQuantizeNoInputCheck": True,
                "ActivationSymmetric":True,
                "WeightSymmetric":True,
                "QuantizeBias": False
            }
        )
        # quantize_dynamic(
        #     model_input=args.output,
        #     model_output=args.quantize_out,
        #     optimize_model=True,
        #     per_channel=False,
        #     reduce_range=False,
        #     weight_type=QuantType.QUInt8,
        # )
        print("Done!")
