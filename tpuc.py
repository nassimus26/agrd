#!/bin/bash

set -e

net_name=fights
input_w=224
input_h=224

# convert to mlir
model_transform.py \
--model_name ${net_name} \
--model_def ../${net_name}.onnx \
--input_shapes [[1,${input_h},${input_w},3],[1,${input_h},${input_w},3],[1,${input_h},${input_w},3],[1,${input_h},${input_w},3],[1,${input_h},${input_w},3],[1,${input_h},${input_w},3],[1,${input_h},${input_w},3],[1,${input_h},${input_w},3]] \
--mean "0,0,0" \
--scale "0,0,0" \
--keep_aspect_ratio \
--pixel_format rgb \
--channel_format nhwc \
--output_names "model_2" \
--output_names "model_2/mobilenetv2_1.00_224/out_relu/Relu6_7;model_2/mobilenetv2_1.00_224/Conv_1_bn/FusedBatchNormV3_7;model_2/mobilenetv2_1.00_224/Conv_1/Conv2D_7,model_2/mobilenetv2_1.00_224/out_relu/Relu6_6;model_2/mobilenetv2_1.00_224/Conv_1_bn/FusedBatchNormV3_6;model_2/mobilenetv2_1.00_224/Conv_1/Conv2D_7;model_2/mobilenetv2_1.00_224/Conv_1/Conv2D_6;model_2/mobilenetv2_1.00_224/Conv_1_bn/FusedBatchNormV3_7,model_2/mobilenetv2_1.00_224/out_relu/Relu6_5;model_2/mobilenetv2_1.00_224/Conv_1_bn/FusedBatchNormV3_5;model_2/mobilenetv2_1.00_224/Conv_1/Conv2D_7;model_2/mobilenetv2_1.00_224/Conv_1/Conv2D_5;model_2/mobilenetv2_1.00_224/Conv_1_bn/FusedBatchNormV3_7,model_2/mobilenetv2_1.00_224/out_relu/Relu6_4;model_2/mobilenetv2_1.00_224/Conv_1_bn/FusedBatchNormV3_4;model_2/mobilenetv2_1.00_224/Conv_1/Conv2D_7;model_2/mobilenetv2_1.00_224/Conv_1/Conv2D_4;model_2/mobilenetv2_1.00_224/Conv_1_bn/FusedBatchNormV3_7,model_2/mobilenetv2_1.00_224/out_relu/Relu6_3;model_2/mobilenetv2_1.00_224/Conv_1_bn/FusedBatchNormV3_3;model_2/mobilenetv2_1.00_224/Conv_1/Conv2D_7;model_2/mobilenetv2_1.00_224/Conv_1/Conv2D_3;model_2/mobilenetv2_1.00_224/Conv_1_bn/FusedBatchNormV3_7,model_2/mobilenetv2_1.00_224/out_relu/Relu6_1;model_2/mobilenetv2_1.00_224/Conv_1_bn/FusedBatchNormV3_1;model_2/mobilenetv2_1.00_224/Conv_1/Conv2D_7;model_2/mobilenetv2_1.00_224/Conv_1/Conv2D_1;model_2/mobilenetv2_1.00_224/Conv_1_bn/FusedBatchNormV3_7,model_2/mobilenetv2_1.00_224/out_relu/Relu6_2;model_2/mobilenetv2_1.00_224/Conv_1_bn/FusedBatchNormV3_2;model_2/mobilenetv2_1.00_224/Conv_1/Conv2D_7;model_2/mobilenetv2_1.00_224/Conv_1/Conv2D_2;model_2/mobilenetv2_1.00_224/Conv_1_bn/FusedBatchNormV3_7,model_2/mobilenetv2_1.00_224/out_relu/Relu6;model_2/mobilenetv2_1.00_224/Conv_1_bn/FusedBatchNormV3;model_2/mobilenetv2_1.00_224/Conv_1/Conv2D_7;model_2/mobilenetv2_1.00_224/Conv_1/Conv2D;model_2/mobilenetv2_1.00_224/Conv_1_bn/FusedBatchNormV3_7" \
--test_input ../dog.jpg ../dog.jpg ../dog.jpg ../dog.jpg ../dog.jpg ../dog.jpg ../dog.jpg ../dog.jpg \
--test_result ${net_name}_top_outputs.npz \
--tolerance 0.99,0.99 \
--mlir ${net_name}.mlir

# export bf16 model
#   not use --quant_input, use float32 for easy coding
model_deploy.py \
--mlir ${net_name}.mlir \
--quantize BF16 \
--processor cv181x \
--test_input ${net_name}_in_f32.npz \
--test_reference ${net_name}_top_outputs.npz \
--model ${net_name}_bf16.cvimodel

echo "calibrate for int8 model"
# export int8 model
run_calibration.py ${net_name}.mlir \
--dataset ../images \
--input_num 200 \
-o ${net_name}_cali_table

echo "convert to int8 model"
# export int8 model
#    add --quant_input, use int8 for faster processing in maix.nn.NN.forward_image
model_deploy.py \
--mlir ${net_name}.mlir \
--quantize INT8 \
--quant_input \
--calibration_table ${net_name}_cali_table \
--processor cv181x \
--test_input ${net_name}_in_f32.npz \
--test_reference ${net_name}_top_outputs.npz \
--tolerance 0.9,0.6 \
--model ${net_name}_int8.cvimodel
