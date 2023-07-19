import onnx
import onnx_graphsurgeon as gs
from onnx.external_data_helper import convert_model_to_external_data
import argparse
parser = argparse.ArgumentParser(
    description="fold constant for sam_encoder."
)
parser.add_argument(
    "--onnx-in", type=str, required=True
)
parser.add_argument(
    "--onnx-out", type=str, required=True
)
args = parser.parse_args()
ext_data_name = args.onnx_in.replace('.onnx', '.data')
graph = gs.import_onnx(onnx.load(args.onnx_in))
graph.fold_constants().cleanup()
model = gs.export_onnx(graph)
convert_model_to_external_data(model, location=ext_data_name)
onnx.save(model, args.onnx_out)

