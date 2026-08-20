#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/types.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>

template <typename scalar_t>
__device__ __forceinline__ float to_float(scalar_t v) {
  return static_cast<float>(v);
}

template <>
__device__ __forceinline__ float to_float<c10::Half>(c10::Half v) {
  return static_cast<float>(v);
}

template <>
__device__ __forceinline__ float to_float<c10::BFloat16>(c10::BFloat16 v) {
  return static_cast<float>(v);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t from_float(float v) {
  return static_cast<scalar_t>(v);
}

template <>
__device__ __forceinline__ c10::Half from_float<c10::Half>(float v) {
  return c10::Half(v);
}

template <>
__device__ __forceinline__ c10::BFloat16 from_float<c10::BFloat16>(float v) {
  return c10::BFloat16(v);
}

__device__ __forceinline__ float block_reduce_sum(float v) {
  __shared__ float shared[32];
  int lane = threadIdx.x & 31;
  int wid = threadIdx.x >> 5;
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffff, v, offset);
  }
  if (lane == 0) shared[wid] = v;
  __syncthreads();
  v = (threadIdx.x < (blockDim.x + 31) / 32) ? shared[lane] : 0.0f;
  if (wid == 0) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      v += __shfl_down_sync(0xffffffff, v, offset);
    }
  }
  return v;
}

template <typename scalar_t>
__global__ void rmsnorm_kernel(const scalar_t* __restrict__ input,
                               const scalar_t* __restrict__ weight,
                               scalar_t* __restrict__ output,
                               int64_t rows,
                               int cols,
                               float eps) {
  int64_t row = blockIdx.x;
  if (row >= rows) return;
  const scalar_t* x = input + row * cols;
  scalar_t* y = output + row * cols;

  float ss = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    float v = to_float(x[col]);
    ss += v * v;
  }
  ss = block_reduce_sum(ss);
  __shared__ float inv_rms;
  if (threadIdx.x == 0) inv_rms = rsqrtf(ss / static_cast<float>(cols) + eps);
  __syncthreads();

  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    float v = to_float(x[col]) * inv_rms * to_float(weight[col]);
    y[col] = from_float<scalar_t>(v);
  }
}

template <typename scalar_t>
__global__ void rope_kernel(const scalar_t* __restrict__ q,
                            const scalar_t* __restrict__ k,
                            const scalar_t* __restrict__ freqs,
                            scalar_t* __restrict__ out_q,
                            scalar_t* __restrict__ out_k,
                            int64_t total,
                            int S,
                            int H,
                            int D,
                            int rot_dim) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int half = rot_dim / 2;
  for (; idx < total; idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int d = idx % D;
    int64_t t = idx / D;
    t /= H;
    int s = t % S;

    float qv = to_float(q[idx]);
    float kv = to_float(k[idx]);
    float qo = qv;
    float ko = kv;

    if (d < rot_dim) {
      int mate_d = d < half ? d + half : d - half;
      int64_t mate_idx = idx + (mate_d - d);
      float cosv = cosf(to_float(freqs[static_cast<int64_t>(s) * rot_dim + d]));
      float sinv = sinf(to_float(freqs[static_cast<int64_t>(s) * rot_dim + d]));
      float qmate = to_float(q[mate_idx]);
      float kmate = to_float(k[mate_idx]);
      float qrot = d < half ? -qmate : qmate;
      float krot = d < half ? -kmate : kmate;
      qo = qv * cosv + qrot * sinv;
      ko = kv * cosv + krot * sinv;
    }
    out_q[idx] = from_float<scalar_t>(qo);
    out_k[idx] = from_float<scalar_t>(ko);
  }
}

template <typename scalar_t>
__global__ void noisy_input_kernel(const scalar_t* __restrict__ latents,
                                   const scalar_t* __restrict__ noise,
                                   const scalar_t* __restrict__ timesteps,
                                   scalar_t* __restrict__ output,
                                   int64_t total,
                                   int64_t elems_per_sample) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (; idx < total; idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int64_t b = idx / elems_per_sample;
    float t = to_float(timesteps[b]);
    float l = to_float(latents[idx]);
    float n = to_float(noise[idx]);
    output[idx] = from_float<scalar_t>((1.0f - t) * l + t * n);
  }
}

template <typename scalar_t>
__global__ void mse_loss_per_sample_kernel(const scalar_t* __restrict__ model_pred,
                                           const scalar_t* __restrict__ latents,
                                           const scalar_t* __restrict__ noise,
                                           const scalar_t* __restrict__ weighting,
                                           const scalar_t* __restrict__ loss_weights,
                                           float* __restrict__ partial,
                                           int B,
                                           int64_t elems_per_sample) {
  int b = blockIdx.x;
  if (b >= B) return;
  int64_t base = static_cast<int64_t>(b) * elems_per_sample;
  float sum = 0.0f;
  for (int64_t i = threadIdx.x; i < elems_per_sample; i += blockDim.x) {
    float target = to_float(noise[base + i]) - to_float(latents[base + i]);
    float diff = to_float(model_pred[base + i]) - target;
    sum += diff * diff;
  }
  sum = block_reduce_sum(sum);
  if (threadIdx.x == 0) {
    float v = sum / static_cast<float>(elems_per_sample);
    v *= to_float(weighting[b]) * to_float(loss_weights[b]);
    partial[b] = v;
  }
}

__global__ void final_mean_kernel(const float* __restrict__ partial, float* __restrict__ output, int B) {
  float sum = 0.0f;
  for (int i = threadIdx.x; i < B; i += blockDim.x) sum += partial[i];
  sum = block_reduce_sum(sum);
  if (threadIdx.x == 0) output[0] = sum / static_cast<float>(B);
}

static int blocks_for(int64_t n, int threads) {
  int64_t b = (n + threads - 1) / threads;
  if (b > 65535) b = 65535;
  return static_cast<int>(b);
}

torch::Tensor rmsnorm_forward_cuda(torch::Tensor input, torch::Tensor weight, double eps) {
  auto output = torch::empty_like(input);
  int cols = static_cast<int>(input.size(-1));
  int64_t rows = input.numel() / cols;
  const int threads = 256;
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "rmsnorm_forward_cuda", [&] {
    rmsnorm_kernel<scalar_t><<<rows, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), rows, cols, static_cast<float>(eps));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::vector<torch::Tensor> rope_forward_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor freqs) {
  auto out_q = torch::empty_like(q);
  auto out_k = torch::empty_like(k);
  int S = static_cast<int>(q.size(1));
  int H = static_cast<int>(q.size(2));
  int D = static_cast<int>(q.size(3));
  int rot_dim = static_cast<int>(freqs.size(1));
  int64_t total = q.numel();
  const int threads = 256;
  int blocks = blocks_for(total, threads);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, q.scalar_type(), "rope_forward_cuda", [&] {
    rope_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), freqs.data_ptr<scalar_t>(),
        out_q.data_ptr<scalar_t>(), out_k.data_ptr<scalar_t>(), total, S, H, D, rot_dim);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out_q, out_k};
}

torch::Tensor noisy_input_cuda(torch::Tensor latents, torch::Tensor noise, torch::Tensor timesteps) {
  auto output = torch::empty_like(latents);
  int64_t total = latents.numel();
  int64_t elems_per_sample = total / latents.size(0);
  const int threads = 256;
  int blocks = blocks_for(total, threads);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, latents.scalar_type(), "noisy_input_cuda", [&] {
    noisy_input_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        latents.data_ptr<scalar_t>(), noise.data_ptr<scalar_t>(), timesteps.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(), total, elems_per_sample);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor rectified_flow_mse_loss_cuda(
    torch::Tensor model_pred,
    torch::Tensor latents,
    torch::Tensor noise,
    torch::Tensor weighting,
    torch::Tensor loss_weights) {
  int B = static_cast<int>(model_pred.size(0));
  int64_t elems_per_sample = model_pred.numel() / B;
  auto partial = torch::empty({B}, model_pred.options().dtype(torch::kFloat32));
  auto output = torch::empty({}, model_pred.options().dtype(torch::kFloat32));
  const int threads = 256;
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, model_pred.scalar_type(), "rectified_flow_mse_loss_cuda", [&] {
    mse_loss_per_sample_kernel<scalar_t><<<B, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        model_pred.data_ptr<scalar_t>(), latents.data_ptr<scalar_t>(), noise.data_ptr<scalar_t>(),
        weighting.data_ptr<scalar_t>(), loss_weights.data_ptr<scalar_t>(), partial.data_ptr<float>(), B, elems_per_sample);
  });
  final_mean_kernel<<<1, 256, 0, at::cuda::getCurrentCUDAStream()>>>(partial.data_ptr<float>(), output.data_ptr<float>(), B);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
