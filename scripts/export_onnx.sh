cur_dir=$(cd $(dirname $0); pwd)
cd ${cur_dir}/..
out_name=sam_vit_l
# return single mask
# python scripts/export_onnx_model.py --checkpoint weights/sam_vit_l_0b3195.pth --model-type vit_l --return-single-mask --out $out_name --opset 15
python scripts/export_onnx_model.py --checkpoint weights/sam_vit_l_0b3195.pth --model-type vit_l --out $out_name --opset 15
./bin/polygraphy surgeon sanitize ${out_name}_vit_encoder.onnx --fold-constants -o ${out_name}_vit_encoder_folded.onnx
./bin/polygraphy surgeon sanitize ${out_name}_mask_decoder.onnx --fold-constants -o ${out_name}_mask_decoder_folded.onnx
mv *.onnx weights/
