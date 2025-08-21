# Quantize the ONNX model for faster inference
# This reduces model size and improves CPU inference speed

import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import os

# Paths
input_model = "./onnx_model/t5_creo.onnx"
output_model = "./onnx_model/t5_creo_quantized.onnx"

# Check if input model exists
if not os.path.exists(input_model):
    print(f"Error: {input_model} not found!")
    print("Please run export_onnx.py first")
    exit(1)

print("Quantizing model for faster inference...")
print(f"Input: {input_model}")
print(f"Output: {output_model}")

# Dynamic quantization (INT8)
# This reduces model size by ~75% and speeds up CPU inference by 2-4x
quantize_dynamic(
    model_input=input_model,
    model_output=output_model,
    weight_type=QuantType.QInt8  # Use INT8 quantization
)

# Check file sizes
original_size = os.path.getsize(input_model) / (1024*1024)  # MB
quantized_size = os.path.getsize(output_model) / (1024*1024)  # MB

print(f"\n✅ Quantization complete!")
print(f"Original model: {original_size:.1f} MB")
print(f"Quantized model: {quantized_size:.1f} MB")
print(f"Size reduction: {(1-quantized_size/original_size)*100:.1f}%")
print(f"Expected speedup: 2-4x faster on CPU")

print(f"\nTo use quantized model, update CMakeLists.txt to use:")
print(f"t5_creo_quantized.onnx instead of t5_creo.onnx")
