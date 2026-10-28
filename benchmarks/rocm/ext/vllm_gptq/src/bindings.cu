// Registration shim for the vLLM exllama GPTQ kernel (q_gemm.cu) so it can be
// built standalone with torch.utils.cpp_extension for the ROCm baseline bench.
#include "torch_utils.h"
#include <torch/csrc/stable/library.h>

torch::stable::Tensor gptq_gemm(torch::stable::Tensor a,
                                torch::stable::Tensor b_q_weight,
                                torch::stable::Tensor b_gptq_qzeros,
                                torch::stable::Tensor b_gptq_scales,
                                bool use_exllama, bool use_v2_format,
                                int64_t bit);
void gptq_shuffle(torch::stable::Tensor q_weight, int64_t bit);

STABLE_TORCH_LIBRARY(vllm_gptq_baseline, ops) {
  ops.def(
      "gptq_gemm(Tensor a, Tensor b_q_weight, Tensor b_gptq_qzeros, "
      "Tensor b_gptq_scales, bool use_exllama, bool use_v2_format, int bit) "
      "-> Tensor");
  ops.def("gptq_shuffle(Tensor! q_weight, int bit) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(vllm_gptq_baseline, CUDA, ops) {
  ops.impl("gptq_gemm", TORCH_BOX(&gptq_gemm));
  ops.impl("gptq_shuffle", TORCH_BOX(&gptq_shuffle));
}
