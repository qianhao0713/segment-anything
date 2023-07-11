cur_dir=$(cd $(dirname $0); pwd)
cd ${cur_dir}/..
out_name=sam_vit_l
python scripts/export_onnx_model.py --checkpoint weights/sam_vit_l_0b3195.pth --model-type vit_l --out $out_name --opset 15
mv *.onnx weights/
