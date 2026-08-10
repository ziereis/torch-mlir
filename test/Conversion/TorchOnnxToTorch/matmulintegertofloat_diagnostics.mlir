// RUN: torch-mlir-opt <%s -split-input-file -verify-diagnostics -convert-torch-onnx-to-torch

// The operator allows a scale and a zero point to independently be scalars or
// 1-D tensors, but `quantized_decomposed.dequantize_per_channel` takes either no
// zero point or exactly one per channel.
func.func @test_matmulintegertofloat_mixed_granularity(%arg0: !torch.vtensor<[4,3],ui8>, %arg1: !torch.vtensor<[3,2],si8>, %arg2: !torch.vtensor<[],f32>, %arg3: !torch.vtensor<[2],f32>, %arg4: !torch.vtensor<[],ui8>, %arg5: !torch.vtensor<[],si8>) -> !torch.vtensor<[4,2],f32> attributes {torch.onnx_meta.opset_version = 21 : si64, torch.onnx_meta.opset_versions = {com.microsoft = 1 : si64}} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.MatMulIntegerToFloat"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) : (!torch.vtensor<[4,3],ui8>, !torch.vtensor<[3,2],si8>, !torch.vtensor<[],f32>, !torch.vtensor<[2],f32>, !torch.vtensor<[],ui8>, !torch.vtensor<[],si8>) -> !torch.vtensor<[4,2],f32>
  return %0 : !torch.vtensor<[4,2],f32>
}

// -----

// Quantization parameters have to be scalars or 1-D tensors.
func.func @test_matmulintegertofloat_rank_2_scale(%arg0: !torch.vtensor<[4,3],ui8>, %arg1: !torch.vtensor<[3,2],si8>, %arg2: !torch.vtensor<[],f32>, %arg3: !torch.vtensor<[1,2],f32>) -> !torch.vtensor<[4,2],f32> attributes {torch.onnx_meta.opset_version = 21 : si64, torch.onnx_meta.opset_versions = {com.microsoft = 1 : si64}} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.MatMulIntegerToFloat"(%arg0, %arg1, %arg2, %arg3) : (!torch.vtensor<[4,3],ui8>, !torch.vtensor<[3,2],si8>, !torch.vtensor<[],f32>, !torch.vtensor<[1,2],f32>) -> !torch.vtensor<[4,2],f32>
  return %0 : !torch.vtensor<[4,2],f32>
}

// -----

// The result has to be f16 or f32.
func.func @test_matmulintegertofloat_f64_result(%arg0: !torch.vtensor<[4,3],ui8>, %arg1: !torch.vtensor<[3,2],si8>, %arg2: !torch.vtensor<[],f64>, %arg3: !torch.vtensor<[],f64>) -> !torch.vtensor<[4,2],f64> attributes {torch.onnx_meta.opset_version = 21 : si64, torch.onnx_meta.opset_versions = {com.microsoft = 1 : si64}} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.MatMulIntegerToFloat"(%arg0, %arg1, %arg2, %arg3) : (!torch.vtensor<[4,3],ui8>, !torch.vtensor<[3,2],si8>, !torch.vtensor<[],f64>, !torch.vtensor<[],f64>) -> !torch.vtensor<[4,2],f64>
  return %0 : !torch.vtensor<[4,2],f64>
}
