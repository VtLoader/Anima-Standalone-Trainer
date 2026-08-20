#include <torch/extension.h>

torch::Tensor rmsnorm_forward_cuda(torch::Tensor input, torch::Tensor weight, double eps);
std::vector<torch::Tensor> rope_forward_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor freqs);
torch::Tensor noisy_input_cuda(torch::Tensor latents, torch::Tensor noise, torch::Tensor timesteps);
torch::Tensor rectified_flow_mse_loss_cuda(
    torch::Tensor model_pred,
    torch::Tensor latents,
    torch::Tensor noise,
    torch::Tensor weighting,
    torch::Tensor loss_weights);

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor rmsnorm_forward(torch::Tensor input, torch::Tensor weight, double eps) {
  CHECK_INPUT(input);
  CHECK_INPUT(weight);
  TORCH_CHECK(input.dim() >= 1, "input must have at least 1 dimension");
  TORCH_CHECK(weight.dim() == 1, "weight must be 1D");
  TORCH_CHECK(input.size(-1) == weight.size(0), "weight length must match input last dimension");
  return rmsnorm_forward_cuda(input, weight, eps);
}

std::vector<torch::Tensor> rope_forward(torch::Tensor q, torch::Tensor k, torch::Tensor freqs) {
  CHECK_INPUT(q);
  CHECK_INPUT(k);
  CHECK_INPUT(freqs);
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4, "q and k must be 4D tensors: (B, S, H, D)");
  TORCH_CHECK(q.sizes() == k.sizes(), "q and k must have identical shapes");
  TORCH_CHECK(freqs.dim() == 2, "freqs must be 2D: (S, rot_dim)");
  TORCH_CHECK(freqs.size(0) >= q.size(1), "freqs sequence length must cover q/k sequence length");
  TORCH_CHECK(freqs.size(1) <= q.size(3), "freqs rot_dim must be <= head_dim");
  TORCH_CHECK(freqs.size(1) % 2 == 0, "freqs rot_dim must be even for non-interleaved RoPE");
  return rope_forward_cuda(q, k, freqs);
}

torch::Tensor noisy_input(torch::Tensor latents, torch::Tensor noise, torch::Tensor timesteps) {
  CHECK_INPUT(latents);
  CHECK_INPUT(noise);
  CHECK_INPUT(timesteps);
  TORCH_CHECK(latents.sizes() == noise.sizes(), "latents and noise must have identical shapes");
  TORCH_CHECK(timesteps.dim() == 1 && timesteps.size(0) == latents.size(0), "timesteps must be (B,)");
  return noisy_input_cuda(latents, noise, timesteps);
}

torch::Tensor rectified_flow_mse_loss(
    torch::Tensor model_pred,
    torch::Tensor latents,
    torch::Tensor noise,
    torch::Tensor weighting,
    torch::Tensor loss_weights) {
  CHECK_INPUT(model_pred);
  CHECK_INPUT(latents);
  CHECK_INPUT(noise);
  CHECK_INPUT(weighting);
  CHECK_INPUT(loss_weights);
  TORCH_CHECK(model_pred.sizes() == latents.sizes(), "model_pred and latents must have identical shapes");
  TORCH_CHECK(model_pred.sizes() == noise.sizes(), "model_pred and noise must have identical shapes");
  TORCH_CHECK(weighting.numel() == model_pred.size(0), "weighting must have B elements");
  TORCH_CHECK(loss_weights.numel() == model_pred.size(0), "loss_weights must have B elements");
  return rectified_flow_mse_loss_cuda(model_pred, latents, noise, weighting, loss_weights);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rmsnorm_forward", &rmsnorm_forward, "Anima RMSNorm forward (CUDA)");
  m.def("rope_forward", &rope_forward, "Anima non-interleaved RoPE q/k forward (CUDA)");
  m.def("noisy_input", &noisy_input, "Anima rectified-flow noisy input (CUDA)");
  m.def("rectified_flow_mse_loss", &rectified_flow_mse_loss, "Anima rectified-flow MSE loss reduce (CUDA)");
}
