import torch
def export_engine_image_encoder(f='vit_l_embedding.onnx', dynamic_input={}, dynamic_input_value={}):
    import tensorrt as trt
    from pathlib import Path
    file = Path(f)
    f = file.with_suffix('.trt')  # TensorRT engine file
    onnx = file.with_suffix('.onnx')
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    workspace = 16
    print("workspace: ", workspace)
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
        if inp.name in dynamic_input_value:
            profile.set_shape_input(inp.name, *dynamic_input_value[inp.name])
        print(f'input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    config.add_optimization_profile(profile)
    for out in outputs:
        print(f'output "{out.name}" with shape{out.shape} {out.dtype}')

    half = True
    print(f'building FP{16 if builder.platform_has_fast_fp16 and half else 32} engine as {f}')
    if builder.platform_has_fast_fp16 and half:
        config.set_flag(trt.BuilderFlag.FP16)
    with builder.build_engine(network, config) as engine, open(f, 'wb') as t:
        t.write(engine.serialize())
with torch.no_grad():
    dynamic_input = {
        "point_coords":[(1,1,2), (64,1,2), (128,2,2)],
        "point_labels":[(1,1), (64,1), (128,2)]
    }
    dynamic_input_value = {
        "orig_im_size":[(1,1),(1200,1800),(1200,2000)]
    }
    export_engine_image_encoder('./weights/sam_single_mask_mask_decoder_fold.onnx',dynamic_input,dynamic_input_value)
    # export_engine_image_encoder('./weights/sam_image_encoder_fold.onnx')
