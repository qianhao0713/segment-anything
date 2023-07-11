import torch
import tensorrt as trt
import argparse

def set_quantize_dynamic_range(tensor_range_file, network):
    tr_map = {}
    with open(tensor_range_file) as f:
        for line in f:
            t_name, t_value = line.strip().split(' ')
            tr_map[t_name]=float(t_value)
    for i in range(network.num_inputs):
        n_input = network.get_input(i)
        if n_input.name in tr_map:
            max_value = tr_map[n_input.name]
            n_input.set_dynamic_range(-max_value, max_value)
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        for j in range(layer.num_outputs):
            l_output = layer.get_output(j)
            if l_output.name in tr_map:
                max_value = tr_map[l_output.name]
                l_output.set_dynamic_range(-max_value, max_value)

def export_engine_image_encoder(onnx_in='vit_l_embedding.onnx', trt_out='vit_l_embedding.trt', dynamic_input={}, dynamic_input_value={}):
    from pathlib import Path
    from segment_anything.calib.calib_dataloader import SAMCalibrator
    import os
    file = Path(onnx_in)
    # f = file.with_name(trt_out)  # TensorRT engine file
    onnx = file.with_suffix('.onnx')
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    profile = builder.create_optimization_profile()
    calib_profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    workspace = 20
    config.max_workspace_size = workspace * 1 << 30
    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx)):
        raise RuntimeError(f'failed to load ONNX file: {onnx}')

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    for inp in inputs:
        if inp.name in dynamic_input:
            profile.set_shape(inp.name, *dynamic_input[inp.name])
            calib_profile.set_shape(inp.name, dynamic_input[inp.name][1], dynamic_input[inp.name][1], dynamic_input[inp.name][1])
        if inp.name in dynamic_input_value:
            profile.set_shape_input(inp.name, *dynamic_input_value[inp.name])
            calib_profile.set_shape_input(inp.name, dynamic_input_value[inp.name][1], dynamic_input_value[inp.name][1], dynamic_input_value[inp.name][1])
        print(f'input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    config.add_optimization_profile(profile)
    config.set_calibration_profile(calib_profile)
    for out in outputs:
        print(f'output "{out.name}" with shape{out.shape} {out.dtype}')
    tensor_range_file = "%s/../calibration.cache" % os.path.abspath(os.path.dirname(__file__))

    half = True
    use_int8 = False
    print(f'building FP{16 if builder.platform_has_fast_fp16 and half else 32} engine')
    if builder.platform_has_fast_fp16 and half:
        config.set_flag(trt.BuilderFlag.FP16)
    if builder.platform_has_fast_int8 and use_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        int8_calibrator = SAMCalibrator(device="cuda:0")
        config.int8_calibrator = int8_calibrator
        set_quantize_dynamic_range(tensor_range_file, network)
    with builder.build_engine(network, config) as engine, open(trt_out, 'wb') as t:
        t.write(engine.serialize())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Export the onnx model to an tensor-rt model."
    )
    parser.add_argument(
        "--onnx", type=str, required=True, help="The path to the input onnx"
    )
    parser.add_argument(
        "--trt", type=str, required=True, help="The filename to save the tensorrt-model to."
    )
    args = parser.parse_args()
    with torch.no_grad():
        dynamic_input = {
            "point_coords":[(1,1,2), (64,100,2), (128,2,2)],
            "point_labels":[(1,1), (64,100), (128,2)]
        }
        dynamic_input_value = {
            "orig_im_size":[(1,1),(1080,1920),(1200,2000)]
        }
        export_engine_image_encoder(onnx_in=args.onnx, trt_out=args.trt, dynamic_input=dynamic_input, dynamic_input_value=dynamic_input_value)
