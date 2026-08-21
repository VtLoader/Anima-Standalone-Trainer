#include <torch/extension.h>

torch::Tensor rmsnorm_forward_cuda(torch::Tensor input, torch::Tensor weight, double eps);
std::pair<torch::Tensor, torch::Tensor> rmsnorm_backward_cuda(
    torch::Tensor grad_output, torch::Tensor input, torch::Tensor weight, double eps);
std::vector<torch::Tensor> rope_forward_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor freqs);
std::pair<torch::Tensor, torch::Tensor> rope_backward_cuda(
    torch::Tensor grad_q, torch::Tensor grad_k, torch::Tensor freqs);
torch::Tensor noisy_input_cuda(torch::Tensor latents, torch::Tensor noise, torch::Tensor timesteps);
torch::Tensor rectified_flow_mse_loss_cuda(
    torch::Tensor model_pred,
    torch::Tensor latents,
    torch::Tensor noise,
    torch::Tensor weighting,
    torch::Tensor loss_weights);
torch::Tensor rectified_flow_mse_loss_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor model_pred,
    torch::Tensor latents,
    torch::Tensor noise,
    torch::Tensor weighting,
    torch::Tensor loss_weights);
torch::Tensor adaln_norm_forward_cuda(torch::Tensor x, torch::Tensor scale, torch::Tensor shift, double eps);
std::vector<torch::Tensor> adaln_norm_backward_cuda(
    torch::Tensor grad_output, torch::Tensor x, torch::Tensor scale,
    int64_t bt_total, int hw, double eps);
void adamw_step_cuda(torch::Tensor grad, torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
                     double lr, double beta1, double beta2, double eps, double weight_decay,
                     double bias_correction1, double bias_correction2, double clip_scale);

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

std::vector<torch::Tensor> rmsnorm_backward(
    torch::Tensor grad_output, torch::Tensor input, torch::Tensor weight, double eps) {
  CHECK_INPUT(grad_output);
  CHECK_INPUT(input);
  CHECK_INPUT(weight);
  TORCH_CHECK(input.dim() >= 1, "input must have at least 1 dimension");
  TORCH_CHECK(weight.dim() == 1, "weight must be 1D");
  TORCH_CHECK(input.size(-1) == weight.size(0), "weight length must match input last dimension");
  auto res = rmsnorm_backward_cuda(grad_output, input, weight, eps);
  return {res.first, res.second};
}

std::vector<torch::Tensor> rope_backward(torch::Tensor grad_q, torch::Tensor grad_k, torch::Tensor freqs) {
  CHECK_INPUT(grad_q);
  CHECK_INPUT(grad_k);
  CHECK_INPUT(freqs);
  TORCH_CHECK(grad_q.dim() == 4 && grad_k.dim() == 4, "grad_q and grad_k must be 4D tensors: (B, S, H, D)");
  TORCH_CHECK(grad_q.sizes() == grad_k.sizes(), "grad_q and grad_k must have identical shapes");
  auto res = rope_backward_cuda(grad_q, grad_k, freqs);
  return {res.first, res.second};
}

torch::Tensor rectified_flow_mse_loss_backward(
    torch::Tensor grad_output,
    torch::Tensor model_pred,
    torch::Tensor latents,
    torch::Tensor noise,
    torch::Tensor weighting,
    torch::Tensor loss_weights) {
  CHECK_INPUT(grad_output);
  CHECK_INPUT(model_pred);
  CHECK_INPUT(latents);
  CHECK_INPUT(noise);
  CHECK_INPUT(weighting);
  CHECK_INPUT(loss_weights);
  TORCH_CHECK(model_pred.sizes() == latents.sizes(), "model_pred and latents must have identical shapes");
  TORCH_CHECK(model_pred.sizes() == noise.sizes(), "model_pred and noise must have identical shapes");
  return rectified_flow_mse_loss_backward_cuda(
      grad_output, model_pred, latents, noise, weighting, loss_weights);
}

torch::Tensor adaln_norm_forward(torch::Tensor x, torch::Tensor scale, torch::Tensor shift, double eps) {
  CHECK_INPUT(x);
  CHECK_INPUT(scale);
  CHECK_INPUT(shift);
  TORCH_CHECK(x.dim() == 5, "x must be 5D (B, T, H, W, D)");
  TORCH_CHECK(scale.dim() == 2 && shift.dim() == 2, "scale/shift must be 2D (bt, D)");
  TORCH_CHECK(scale.size(-1) == x.size(-1) && shift.size(-1) == x.size(-1), "scale/shift D must match x D");
  TORCH_CHECK(scale.size(0) * (x.size(2) * x.size(3)) == (x.numel() / x.size(-1)),
              "scale bt must equal B*T");
  return adaln_norm_forward_cuda(x, scale, shift, eps);
}

std::vector<torch::Tensor> adaln_norm_backward(
    torch::Tensor grad_output, torch::Tensor x, torch::Tensor scale,
    int64_t bt_total, int hw, double eps) {
  CHECK_INPUT(grad_output);
  CHECK_INPUT(x);
  CHECK_INPUT(scale);
  return adaln_norm_backward_cuda(grad_output, x, scale, bt_total, hw, eps);
}

void adamw_step(torch::Tensor grad, torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
                double lr, double beta1, double beta2, double eps, double weight_decay,
                double bias_correction1, double bias_correction2, double clip_scale) {
  CHECK_INPUT(grad);
  CHECK_INPUT(param);
  CHECK_INPUT(exp_avg);
  CHECK_INPUT(exp_avg_sq);
  TORCH_CHECK(grad.scalar_type() == torch::kFloat32, "adamw_step requires fp32 master params");
  TORCH_CHECK(param.scalar_type() == torch::kFloat32, "adamw_step requires fp32 master params");
  TORCH_CHECK(grad.numel() == param.numel() && param.numel() == exp_avg.numel() && exp_avg.numel() == exp_avg_sq.numel(),
              "grad/param/state sizes must match");
  adamw_step_cuda(grad, param, exp_avg, exp_avg_sq, lr, beta1, beta2, eps, weight_decay,
                  bias_correction1, bias_correction2, clip_scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rmsnorm_forward", &rmsnorm_forward, "Anima RMSNorm forward (CUDA)");
  m.def("rmsnorm_backward", &rmsnorm_backward, "Anima RMSNorm backward (CUDA)");
  m.def("rope_forward", &rope_forward, "Anima non-interleaved RoPE q/k forward (CUDA)");
  m.def("rope_backward", &rope_backward, "Anima non-interleaved RoPE q/k backward (CUDA)");
  m.def("noisy_input", &noisy_input, "Anima rectified-flow noisy input (CUDA)");
  m.def("rectified_flow_mse_loss", &rectified_flow_mse_loss, "Anima rectified-flow MSE loss reduce (CUDA)");
  m.def("rectified_flow_mse_loss_backward", &rectified_flow_mse_loss_backward, "Anima rectified-flow MSE loss backward (CUDA)");
  m.def("adaln_norm_forward", &adaln_norm_forward, "Anima fused LayerNorm+AdaLN forward (CUDA)");
  m.def("adaln_norm_backward", &adaln_norm_backward, "Anima fused LayerNorm+AdaLN backward (CUDA)");
  m.def("adamw_step", &adamw_step, "Anima fused AdamW step (CUDA)");
}
