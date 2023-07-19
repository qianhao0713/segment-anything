cur_dir=$(cd $(dirname $0); pwd)
cd ${cur_dir}/..
out_name=sam_freespace
vit_dir=weights/tmp
mkdir -p ${vit_dir}
# return single mask
# python scripts/export_onnx_model.py --checkpoint weights/sam_vit_l_0b3195.pth --model-type vit_l --return-single-mask --out $out_name --opset 15
#python scripts/export_onnx_model_roadseg.py --sam-ckpt weights/sam_vit_l_0b3195.pth --seghead-ckpt weights/roadseg.pth --model-type vit_l --out $out_name --opset 15
python scripts/export_onnx_model_freespace.py --sam-ckpt weights/sam_vit_h_4b8939.pth --seghead-ckpt weights/roadseg.pth --model-type vit_h --out $out_name --opset 15

./bin/polygraphy surgeon sanitize ${out_name}_decoder.onnx --fold-constants -o ${out_name}_decoder_folded.onnx
mv *.onnx weights/

cd ${vit_dir}
python ${cur_dir}/fold_vit_h.py --onnx-in=${out_name}_encoder.onnx --onnx-out=${out_name}_encoder_folded.onnx