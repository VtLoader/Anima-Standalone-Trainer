#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <type_traits>
#include <torch/types.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>

// ---------------------------------------------------------------------------
// Scalar conversion helpers
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Block reduction (single accumulator, warp shuffle + shared)
// ---------------------------------------------------------------------------

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

__device__ __forceinline__ void block_reduce_sum2(float& a, float& b) {
  __shared__ float sa[32];
  __shared__ float sb[32];
  int lane = threadIdx.x & 31;
  int wid = threadIdx.x >> 5;
  for (int offset = 16; offset > 0; offset >>= 1) {
    a += __shfl_down_sync(0xffffffff, a, offset);
    b += __shfl_down_sync(0xffffffff, b, offset);
  }
  if (lane == 0) { sa[wid] = a; sb[wid] = b; }
  __syncthreads();
  a = (threadIdx.x < (blockDim.x + 31) / 32) ? sa[lane] : 0.0f;
  b = (threadIdx.x < (blockDim.x + 31) / 32) ? sb[lane] : 0.0f;
  if (wid == 0) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      a += __shfl_down_sync(0xffffffff, a, offset);
      b += __shfl_down_sync(0xffffffff, b, offset);
    }
  }
}

// ---------------------------------------------------------------------------
// RMSNorm forward (vectorized: float4 for fp32, uint4 for half/bf16)
//   y[row,c] = x[row,c] * rsqrt(mean_c x^2 + eps) * weight[c]
// ---------------------------------------------------------------------------

// Number of scalar elements packed in one 16-byte vector load/store.
template <typename scalar_t>
__device__ __forceinline__ constexpr int kElemPer16() {
  return 4;  // fp32: 4 * 4B = 16B
}

template <>
__device__ __forceinline__ constexpr int kElemPer16<c10::Half>() { return 8; }

template <>
__device__ __forceinline__ constexpr int kElemPer16<c10::BFloat16>() { return 8; }

// Load a 16-byte chunk as scalar_t values into out[0..kElemPer16-1].
template <typename scalar_t>
__device__ __forceinline__ void load16(const scalar_t* __restrict__ p, float* __restrict__ out) {
  const uint4* v = reinterpret_cast<const uint4*>(p);
  if constexpr (sizeof(scalar_t) == 4) {
    out[0] = __uint_as_float(v->x);
    out[1] = __uint_as_float(v->y);
    out[2] = __uint_as_float(v->z);
    out[3] = __uint_as_float(v->w);
  } else if constexpr (std::is_same_v<scalar_t, c10::Half>) {
    float2 a = __half22float2(reinterpret_cast<const __half2&>(v->x));
    float2 b = __half22float2(reinterpret_cast<const __half2&>(v->y));
    float2 c = __half22float2(reinterpret_cast<const __half2&>(v->z));
    float2 d = __half22float2(reinterpret_cast<const __half2&>(v->w));
    out[0] = a.x; out[1] = a.y;
    out[2] = b.x; out[3] = b.y;
    out[4] = c.x; out[5] = c.y;
    out[6] = d.x; out[7] = d.y;
  } else {  // bfloat16
    float2 a = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162&>(v->x));
    float2 b = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162&>(v->y));
    float2 c = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162&>(v->z));
    float2 d = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162&>(v->w));
    out[0] = a.x; out[1] = a.y;
    out[2] = b.x; out[3] = b.y;
    out[4] = c.x; out[5] = c.y;
    out[6] = d.x; out[7] = d.y;
  }
}

// Store a 16-byte chunk from in[0..kElemPer16-1].
template <typename scalar_t>
__device__ __forceinline__ void store16(scalar_t* __restrict__ p, const float* __restrict__ in) {
  uint4* v = reinterpret_cast<uint4*>(p);
  if constexpr (sizeof(scalar_t) == 4) {
    v->x = __float_as_uint(in[0]);
    v->y = __float_as_uint(in[1]);
    v->z = __float_as_uint(in[2]);
    v->w = __float_as_uint(in[3]);
  } else if constexpr (std::is_same_v<scalar_t, c10::Half>) {
    reinterpret_cast<__half2&>(v->x) = __floats2half2_rn(in[0], in[1]);
    reinterpret_cast<__half2&>(v->y) = __floats2half2_rn(in[2], in[3]);
    reinterpret_cast<__half2&>(v->z) = __floats2half2_rn(in[4], in[5]);
    reinterpret_cast<__half2&>(v->w) = __floats2half2_rn(in[6], in[7]);
  } else {  // bfloat16
    reinterpret_cast<__nv_bfloat162&>(v->x) = __floats2bfloat162_rn(in[0], in[1]);
    reinterpret_cast<__nv_bfloat162&>(v->y) = __floats2bfloat162_rn(in[2], in[3]);
    reinterpret_cast<__nv_bfloat162&>(v->z) = __floats2bfloat162_rn(in[4], in[5]);
    reinterpret_cast<__nv_bfloat162&>(v->w) = __floats2bfloat162_rn(in[6], in[7]);
  }
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

  constexpr int V = kElemPer16<scalar_t>();
  const int nchunks = cols / V;  // full 16-byte chunks

  float ss = 0.0f;
  float buf[8];
  for (int cv = threadIdx.x; cv < nchunks; cv += blockDim.x) {
    load16<scalar_t>(x + cv * V, buf);
    for (int e = 0; e < V; ++e) { float v = buf[e]; ss += v * v; }
  }
  // Scalar tail for cols not a multiple of V.
  for (int c = nchunks * V + threadIdx.x; c < cols; c += blockDim.x) {
    float v = to_float(x[c]);
    ss += v * v;
  }
  ss = block_reduce_sum(ss);
  __shared__ float inv_rms;
  if (threadIdx.x == 0) inv_rms = rsqrtf(ss / static_cast<float>(cols) + eps);
  __syncthreads();

  const float inv = inv_rms;
  const scalar_t* w = weight;
  float xbuf[8];
  float obuf[8];
  for (int cv = threadIdx.x; cv < nchunks; cv += blockDim.x) {
    load16<scalar_t>(x + cv * V, xbuf);
    load16<scalar_t>(w + cv * V, obuf);
    for (int e = 0; e < V; ++e) obuf[e] = xbuf[e] * inv * obuf[e];
    store16<scalar_t>(y + cv * V, obuf);
  }
  for (int c = nchunks * V + threadIdx.x; c < cols; c += blockDim.x) {
    y[c] = from_float<scalar_t>(to_float(x[c]) * inv * to_float(w[c]));
  }
}

// ---------------------------------------------------------------------------
// RMSNorm backward
//   grad_x[r,c] = inv[r]*g[r,c]*w[c] - x[r,c]*(inv[r]^3 / cols)*dot[r]
//   inv[r]  = rsqrt(mean_c x^2 + eps)
//   dot[r]  = sum_c g[r,c]*w[c]*x[r,c]
// ---------------------------------------------------------------------------

// One thread per row (cols is small, e.g. head_dim). Avoids block-reduce/sync
// overhead and gives full-SM occupancy for the large row count.
//   grad_x[r,c] = inv[r]*g[r,c]*w[c] - x[r,c]*(inv[r]^3 / cols)*dot[r]
template <typename scalar_t>
__global__ void rmsnorm_backward_gradx_kernel(const scalar_t* __restrict__ input,
                                              const scalar_t* __restrict__ grad_out,
                                              const scalar_t* __restrict__ weight,
                                              scalar_t* __restrict__ grad_x,
                                              float* __restrict__ inv_out,
                                              int64_t rows,
                                              int cols,
                                              float eps) {
  int64_t row = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (; row < rows; row += stride) {
    const scalar_t* x = input + row * cols;
    const scalar_t* g = grad_out + row * cols;
    scalar_t* gx = grad_x + row * cols;

    float ss = 0.0f;
    float dot = 0.0f;
    for (int c = 0; c < cols; ++c) {
      float xv = to_float(x[c]);
      ss += xv * xv;
      dot += to_float(g[c]) * to_float(weight[c]) * xv;
    }
    float inv = rsqrtf(ss / static_cast<float>(cols) + eps);
    inv_out[row] = inv;
    float coef = (inv * inv * inv) / static_cast<float>(cols);
    for (int c = 0; c < cols; ++c) {
      gx[c] = from_float<scalar_t>(inv * to_float(g[c]) * to_float(weight[c]) - to_float(x[c]) * coef * dot);
    }
  }
}

// grad_w_fp32[c] = sum_r g[r,c]*x[r,c]*inv[r]. Block per row-tile; thread per
// column, rows read coalesced across columns, per-column atomicAdd into fp32.
template <typename scalar_t>
__global__ void rmsnorm_backward_gradw_kernel(const scalar_t* __restrict__ input,
                                              const scalar_t* __restrict__ grad_out,
                                              const float* __restrict__ inv,
                                              float* __restrict__ grad_w_fp32,
                                              int64_t rows,
                                              int cols,
                                              int tile_rows) {
  int64_t row0 = blockIdx.x * (int64_t)tile_rows;
  int64_t rowEnd = min(row0 + tile_rows, rows);
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    float col_acc = 0.0f;
    for (int64_t r = row0; r < rowEnd; ++r) {
      col_acc += to_float(grad_out[r * cols + c]) * to_float(input[r * cols + c]) * inv[r];
    }
    if (col_acc != 0.0f) atomicAdd(&grad_w_fp32[c], col_acc);
  }
}

// ---------------------------------------------------------------------------
// RoPE forward/backward (pair-based, one block per (s, bh), shared sincos)
//   Forward  low_sign = -1;  Backward low_sign = +1
//   out[d] = x[d]*cos[d] + sign*x[mate]*sin[d], sign = d<half ? low_sign : -low_sign
//   mate   = d<half ? d+half : d-half
// ---------------------------------------------------------------------------

//   Forward  low_sign = -1, use_mate_sin = 0; Backward low_sign = +1, use_mate_sin = 1.
//   For rotated output index d: self term uses cos[d]; cross term uses sin[d]
//   (forward, x[mate] rotated by sin[d]) or sin[mate] (backward, transpose).
template <typename scalar_t>
__global__ void rope_kernel(const scalar_t* __restrict__ x,
                            const scalar_t* __restrict__ x2,
                            const float* __restrict__ freqs,
                            scalar_t* __restrict__ ox,
                            scalar_t* __restrict__ ox2,
                            int S,
                            int H,
                            int D,
                            int rot_dim,
                            int low_sign,
                            int use_mate_sin,
                            int BH) {
  const int s = blockIdx.x;
  const int half = rot_dim >> 1;

  __shared__ float sh_cos[1024];
  __shared__ float sh_sin[1024];

  // Compute cos/sin once for every (s, d) column; reused across all b/h.
  // freqs is passed as fp32 (converted on the host) so the rotation angles
  // match the reference path exactly instead of being pre-rounded to bf16.
  for (int d = threadIdx.x; d < rot_dim; d += blockDim.x) {
    float f = freqs[static_cast<int64_t>(s) * rot_dim + d];
    sincosf(f, &sh_sin[d], &sh_cos[d]);
  }
  __syncthreads();

  for (int bh = 0; bh < BH; ++bh) {
    const int b = bh / H;
    const int h = bh % H;
    const int64_t base = ((static_cast<int64_t>(b) * S + s) * H + h) * D;

    for (int d = threadIdx.x; d < D; d += blockDim.x) {
      if (d < rot_dim) {
        const int mate = d < half ? d + half : d - half;
        const int sign = d < half ? low_sign : -low_sign;
        const float cosd = sh_cos[d];
        const float sind = use_mate_sin ? sh_sin[mate] : sh_sin[d];
        const float xv = to_float(x[base + d]);
        const float xm = to_float(x[base + mate]);
        const float x2v = to_float(x2[base + d]);
        const float x2m = to_float(x2[base + mate]);
        ox[base + d] = from_float<scalar_t>(xv * cosd + static_cast<float>(sign) * xm * sind);
        ox2[base + d] = from_float<scalar_t>(x2v * cosd + static_cast<float>(sign) * x2m * sind);
      } else {
        ox[base + d] = x[base + d];
        ox2[base + d] = x2[base + d];
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Noisy input (rectified-flow blend) — vectorized-free, memory bound elementwise
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// MSE loss forward (multi-block per sample) and backward
// ---------------------------------------------------------------------------

template <typename scalar_t>
__global__ void mse_loss_partial_kernel_t(
    const scalar_t* __restrict__ model_pred,
    const scalar_t* __restrict__ latents,
    const scalar_t* __restrict__ noise,
    const scalar_t* __restrict__ weighting,
    const scalar_t* __restrict__ loss_weights,
    float* __restrict__ partial,
    int B,
    int64_t elems_per_sample,
    int blocks_per_sample) {
  const int b = blockIdx.x / blocks_per_sample;
  const int bp = blockIdx.x % blocks_per_sample;
  if (b >= B) return;
  const int64_t base = static_cast<int64_t>(b) * elems_per_sample;
  const int64_t chunk = (elems_per_sample + blocks_per_sample - 1) / blocks_per_sample;
  const int64_t start = base + static_cast<int64_t>(bp) * chunk;
  const int64_t end = min(base + elems_per_sample, start + chunk);

  float sum = 0.0f;
  for (int64_t i = start + threadIdx.x; i < end; i += blockDim.x) {
    float target = to_float(noise[i]) - to_float(latents[i]);
    float diff = to_float(model_pred[i]) - target;
    sum += diff * diff;
  }
  sum = block_reduce_sum(sum);
  if (threadIdx.x == 0) {
    float v = sum / static_cast<float>(elems_per_sample);
    v *= to_float(weighting[b]) * to_float(loss_weights[b]);
    partial[b * blocks_per_sample + bp] = v;
  }
}

__global__ void final_mean_kernel(const float* __restrict__ partial, float* __restrict__ output, int count, int divisor) {
  float sum = 0.0f;
  for (int i = threadIdx.x; i < count; i += blockDim.x) sum += partial[i];
  sum = block_reduce_sum(sum);
  if (threadIdx.x == 0) output[0] = sum / static_cast<float>(divisor);
}

template <typename scalar_t>
__global__ void mse_loss_backward_kernel(const scalar_t* __restrict__ model_pred,
                                         const scalar_t* __restrict__ latents,
                                         const scalar_t* __restrict__ noise,
                                         const scalar_t* __restrict__ weighting,
                                         const scalar_t* __restrict__ loss_weights,
                                         const float* __restrict__ grad_out,
                                         scalar_t* __restrict__ grad_pred,
                                         int64_t total,
                                         int64_t elems_per_sample,
                                         float scale) {
  const float gov = to_float(grad_out[0]);
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (; idx < total; idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int64_t b = idx / elems_per_sample;
    float target = to_float(noise[idx]) - to_float(latents[idx]);
    float diff = to_float(model_pred[idx]) - target;
    float w = to_float(weighting[b]) * to_float(loss_weights[b]);
    grad_pred[idx] = from_float<scalar_t>(diff * w * scale * gov);
  }
}

// ---------------------------------------------------------------------------
// Host launchers
// ---------------------------------------------------------------------------

static int blocks_for(int64_t n, int threads) {
  int64_t b = (n + threads - 1) / threads;
  if (b > 65535) b = 65535;
  return static_cast<int>(b);
}

torch::Tensor rmsnorm_forward_cuda(torch::Tensor input, torch::Tensor weight, double eps) {
  auto output = torch::empty_like(input);
  int cols = static_cast<int>(input.size(-1));
  int64_t rows = input.numel() / cols;
  // Vectorized 16-byte loads require each row start to be 16B-aligned, i.e. the
  // row stride (cols * elem_size) must be a multiple of 16. torch tensors are
  // base-aligned, so this is the only alignment requirement.
  TORCH_CHECK((cols * static_cast<int>(input.element_size())) % 16 == 0,
              "rmsnorm kernel requires cols * element_size to be a multiple of 16");
  const int threads = 256;
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "rmsnorm_forward_cuda", [&] {
    rmsnorm_kernel<scalar_t><<<rows, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), rows, cols, static_cast<float>(eps));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::pair<torch::Tensor, torch::Tensor> rmsnorm_backward_cuda(
    torch::Tensor grad_output, torch::Tensor input, torch::Tensor weight, double eps) {
  auto grad_x = torch::empty_like(input);
  int cols = static_cast<int>(input.size(-1));
  int64_t rows = input.numel() / cols;
  TORCH_CHECK((cols * static_cast<int>(input.element_size())) % 16 == 0,
              "rmsnorm kernel requires cols * element_size to be a multiple of 16");
  const int threads = 256;
  auto inv_buf = torch::empty({rows}, input.options().dtype(torch::kFloat32));
  auto grad_w_fp32 = torch::zeros({cols}, input.options().dtype(torch::kFloat32));
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "rmsnorm_backward_cuda", [&] {
    int gx_blocks = blocks_for(rows, threads);
    rmsnorm_backward_gradx_kernel<scalar_t><<<gx_blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<scalar_t>(), grad_output.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(),
        grad_x.data_ptr<scalar_t>(), inv_buf.data_ptr<float>(), rows, cols, static_cast<float>(eps));
    const int tile_rows = 64;
    int64_t n_tiles = (rows + tile_rows - 1) / tile_rows;
    int gw_blocks = blocks_for(n_tiles, 1);
    // atomicAdd into a fresh fp32 buffer; needs zero-init (done above).
    rmsnorm_backward_gradw_kernel<scalar_t><<<gw_blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<scalar_t>(), grad_output.data_ptr<scalar_t>(), inv_buf.data_ptr<float>(),
        grad_w_fp32.data_ptr<float>(), rows, cols, tile_rows);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto grad_w = grad_w_fp32.to(weight.scalar_type());
  return {grad_x, grad_w};
}

std::vector<torch::Tensor> rope_forward_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor freqs) {
  auto out_q = torch::empty_like(q);
  auto out_k = torch::empty_like(k);
  int S = static_cast<int>(q.size(1));
  int H = static_cast<int>(q.size(2));
  int D = static_cast<int>(q.size(3));
  int rot_dim = static_cast<int>(freqs.size(1));
  TORCH_CHECK(rot_dim <= 1024,
              "rope kernel only supports rot_dim <= 1024 (shared cos/sin table)");
  int BH = static_cast<int>(q.size(0)) * H;
  const int threads = 128;
  const int grid = S;
  // Angles are immutable inputs; keep them fp32 so cos/sin match the torch
  // reference (which computes them from the fp32 freq buffer) exactly.
  auto freqs_f = freqs.to(torch::kFloat32);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, q.scalar_type(), "rope_forward_cuda", [&] {
    rope_kernel<scalar_t><<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), freqs_f.data_ptr<float>(),
        out_q.data_ptr<scalar_t>(), out_k.data_ptr<scalar_t>(), S, H, D, rot_dim, -1, 0, BH);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out_q, out_k};
}

std::pair<torch::Tensor, torch::Tensor> rope_backward_cuda(
    torch::Tensor grad_q, torch::Tensor grad_k, torch::Tensor freqs) {
  auto gq = torch::empty_like(grad_q);
  auto gk = torch::empty_like(grad_k);
  int S = static_cast<int>(grad_q.size(1));
  int H = static_cast<int>(grad_q.size(2));
  int D = static_cast<int>(grad_q.size(3));
  int rot_dim = static_cast<int>(freqs.size(1));
  TORCH_CHECK(rot_dim <= 1024,
              "rope kernel only supports rot_dim <= 1024 (shared cos/sin table)");
  int BH = static_cast<int>(grad_q.size(0)) * H;
  const int threads = 128;
  const int grid = S;
  auto freqs_f = freqs.to(torch::kFloat32);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, grad_q.scalar_type(), "rope_backward_cuda", [&] {
    rope_kernel<scalar_t><<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        grad_q.data_ptr<scalar_t>(), grad_k.data_ptr<scalar_t>(), freqs_f.data_ptr<float>(),
        gq.data_ptr<scalar_t>(), gk.data_ptr<scalar_t>(), S, H, D, rot_dim, +1, 1, BH);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {gq, gk};
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
  const int threads = 256;
  // One block per (sample, chunk); aim ~64 elems/thread for good occupancy.
  int bps = static_cast<int>((elems_per_sample + threads * 64 - 1) / (threads * 64));
  if (bps < 1) bps = 1;
  auto partial = torch::empty({B * bps}, model_pred.options().dtype(torch::kFloat32));
  auto output = torch::empty({}, model_pred.options().dtype(torch::kFloat32));
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, model_pred.scalar_type(), "rectified_flow_mse_loss_cuda", [&] {
    mse_loss_partial_kernel_t<scalar_t><<<B * bps, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        model_pred.data_ptr<scalar_t>(), latents.data_ptr<scalar_t>(), noise.data_ptr<scalar_t>(),
        weighting.data_ptr<scalar_t>(), loss_weights.data_ptr<scalar_t>(), partial.data_ptr<float>(),
        B, elems_per_sample, bps);
  });
  // final_mean divides by the number of samples B (each partial is already
  // normalized by elems_per_sample and weighted).
  final_mean_kernel<<<1, 256, 0, at::cuda::getCurrentCUDAStream()>>>(partial.data_ptr<float>(), output.data_ptr<float>(), B * bps, B);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor rectified_flow_mse_loss_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor model_pred,
    torch::Tensor latents,
    torch::Tensor noise,
    torch::Tensor weighting,
    torch::Tensor loss_weights) {
  auto grad_pred = torch::empty_like(model_pred);
  int64_t total = model_pred.numel();
  int64_t elems_per_sample = total / model_pred.size(0);
  int B = static_cast<int>(model_pred.size(0));
  float scale = 2.0f / (static_cast<float>(elems_per_sample) * static_cast<float>(B));
  const int threads = 256;
  int blocks = blocks_for(total, threads);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, model_pred.scalar_type(), "rectified_flow_mse_loss_backward_cuda", [&] {
    mse_loss_backward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        model_pred.data_ptr<scalar_t>(), latents.data_ptr<scalar_t>(), noise.data_ptr<scalar_t>(),
        weighting.data_ptr<scalar_t>(), loss_weights.data_ptr<scalar_t>(),
        grad_output.is_contiguous() ? grad_output.data_ptr<float>() : nullptr,
        grad_pred.data_ptr<scalar_t>(),
        total, elems_per_sample, scale);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_pred;
}

// ---------------------------------------------------------------------------
// Fused LayerNorm(affine=False) + AdaLN scale/shift modulation
//   x  : (B,T,H,W,D) -> rows = B*T*H*W, cols = D
//   scale/shift : (B*T, D) indexed by bt = row / hw (hw = H*W)
//   y[row,d] = ((x[row,d]-mean_row)*rstd_row) * (1 + scale[bt,d]) + shift[bt,d]
//   where mean_row/var_row are over the D (last) dim per spatial position.
// ---------------------------------------------------------------------------

template <typename scalar_t>
__global__ void adaln_norm_kernel(const scalar_t* __restrict__ x,
                                  const scalar_t* __restrict__ scale,
                                  const scalar_t* __restrict__ shift,
                                  scalar_t* __restrict__ y,
                                  int64_t rows,
                                  int cols,
                                  int hw,
                                  float eps) {
  int64_t row = blockIdx.x;
  if (row >= rows) return;
  const scalar_t* xr = x + row * cols;
  scalar_t* yr = y + row * cols;
  const int bt = static_cast<int>(row / hw);
  const scalar_t* sr = scale + static_cast<int64_t>(bt) * cols;
  const scalar_t* hr = shift + static_cast<int64_t>(bt) * cols;

  float sum = 0.0f, sq = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    float v = to_float(xr[c]);
    sum += v;
    sq += v * v;
  }
  block_reduce_sum2(sum, sq);
  __shared__ float s_mean;
  __shared__ float s_rstd;
  if (threadIdx.x == 0) {
    float mean = sum / static_cast<float>(cols);
    float var = sq / static_cast<float>(cols) - mean * mean;
    s_mean = mean;
    s_rstd = rsqrtf(var + eps);
  }
  __syncthreads();
  const float mean = s_mean;
  const float rstd = s_rstd;

  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    float v = (to_float(xr[c]) - mean) * rstd;
    yr[c] = from_float<scalar_t>(v * (1.0f + to_float(sr[c])) + to_float(hr[c]));
  }
}

// Backward gem grad_x. Also writes per-row (mean, rstd) so the scale/shift
// reduction kernel can recompute n without storing the full n tensor.
template <typename scalar_t>
__global__ void adaln_norm_backward_gradx_kernel(const scalar_t* __restrict__ gy,
                                                 const scalar_t* __restrict__ x,
                                                 const scalar_t* __restrict__ scale,
                                                 scalar_t* __restrict__ dx,
                                                 float* __restrict__ mean_rstd,
                                                 int64_t rows,
                                                 int cols,
                                                 int hw,
                                                 float eps) {
  int64_t row = blockIdx.x;
  if (row >= rows) return;
  const scalar_t* xr = x + row * cols;
  const scalar_t* gr = gy + row * cols;
  const int bt = static_cast<int>(row / hw);
  const scalar_t* sr = scale + static_cast<int64_t>(bt) * cols;
  scalar_t* dxr = dx + row * cols;

  float sum = 0.0f, sq = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    float v = to_float(xr[c]);
    sum += v;
    sq += v * v;
  }
  block_reduce_sum2(sum, sq);
  __shared__ float s_mean;
  __shared__ float s_rstd;
  if (threadIdx.x == 0) {
    float mean = sum / static_cast<float>(cols);
    float var = sq / static_cast<float>(cols) - mean * mean;
    s_mean = mean;
    s_rstd = rsqrtf(var + eps);
    mean_rstd[row * 2] = mean;
    mean_rstd[row * 2 + 1] = s_rstd;
  }
  __syncthreads();
  const float mean = s_mean;
  const float rstd = s_rstd;

  // Second pass: sum_dn and sum_dn*n where dn = gy*(1+scale).
  float sdn = 0.0f, sdnn = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    float n = (to_float(xr[c]) - mean) * rstd;
    float dn = to_float(gr[c]) * (1.0f + to_float(sr[c]));
    sdn += dn;
    sdnn += dn * n;
  }
  block_reduce_sum2(sdn, sdnn);
  __shared__ float s_dn;
  __shared__ float s_dn_n;
  if (threadIdx.x == 0) {
    s_dn = sdn;
    s_dn_n = sdnn;
  }
  __syncthreads();
  const float mean_dn = s_dn / static_cast<float>(cols);
  const float mean_dn_n = s_dn_n / static_cast<float>(cols);

  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    float n = (to_float(xr[c]) - mean) * rstd;
    float dn = to_float(gr[c]) * (1.0f + to_float(sr[c]));
    dxr[c] = from_float<scalar_t>(rstd * (dn - mean_dn - n * mean_dn_n));
  }
}

// Reduce grad scale/shift: chunked by row groups for parallelism. Each block
// handles a contiguous slice of hw rows for one bt, threads over columns, then
// atomicAdd the per-column partials. dscale/dshift must be zero-initialized.
template <typename scalar_t>
__global__ void adaln_norm_backward_scale_shift_kernel(const scalar_t* __restrict__ gy,
                                                       const scalar_t* __restrict__ x,
                                                       const float* __restrict__ mean_rstd,
                                                       float* __restrict__ dscale,
                                                       float* __restrict__ dshift,
                                                       int64_t rows,
                                                       int cols,
                                                       int hw,
                                                       int bt_total,
                                                       int rows_per_block) {
  const int nchunks = (hw + rows_per_block - 1) / rows_per_block;
  const int blk = blockIdx.x;
  const int bt = blk / nchunks;
  if (bt >= bt_total) return;
  const int chunk = blk % nchunks;
  const int64_t r0 = static_cast<int64_t>(bt) * hw + static_cast<int64_t>(chunk) * rows_per_block;
  const int64_t rEnd = min(r0 + rows_per_block, static_cast<int64_t>(bt + 1) * hw);
  float* gsc = dscale + static_cast<int64_t>(bt) * cols;
  float* gsh = dshift + static_cast<int64_t>(bt) * cols;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    float accs = 0.0f, acch = 0.0f;
    for (int64_t r = r0; r < rEnd; ++r) {
      float mean = mean_rstd[r * 2];
      float rstd = mean_rstd[r * 2 + 1];
      float n = (to_float(x[r * cols + c]) - mean) * rstd;
      float gv = to_float(gy[r * cols + c]);
      accs += gv * n;
      acch += gv;
    }
    if (accs != 0.0f) atomicAdd(&gsc[c], accs);
    if (acch != 0.0f) atomicAdd(&gsh[c], acch);
  }
}

torch::Tensor adaln_norm_forward_cuda(torch::Tensor x, torch::Tensor scale, torch::Tensor shift, double eps) {
  auto y = torch::empty_like(x);
  int cols = static_cast<int>(x.size(-1));
  int64_t rows = x.numel() / cols;
  int hw = static_cast<int>(x.size(2) * x.size(3));  // H*W for (B,T,H,W,D)
  const int threads = 256;
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "adaln_norm_forward_cuda", [&] {
    adaln_norm_kernel<scalar_t><<<rows, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<scalar_t>(), scale.data_ptr<scalar_t>(), shift.data_ptr<scalar_t>(),
        y.data_ptr<scalar_t>(), rows, cols, hw, static_cast<float>(eps));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

std::vector<torch::Tensor> adaln_norm_backward_cuda(
    torch::Tensor grad_output, torch::Tensor x, torch::Tensor scale,
    int64_t bt_total, int hw, double eps) {
  auto dx = torch::empty_like(x);
  auto dscale = torch::zeros({bt_total, x.size(-1)}, x.options().dtype(torch::kFloat32));
  auto dshift = torch::zeros({bt_total, x.size(-1)}, x.options().dtype(torch::kFloat32));
  int cols = static_cast<int>(x.size(-1));
  int64_t rows = x.numel() / cols;
  int64_t nr = std::max<int64_t>(rows, 1);
  auto mean_rstd = torch::empty({nr, 2}, x.options().dtype(torch::kFloat32));
  const int threads = 256;
  const int rows_per_block = 256;
  int nchunks = std::max(1, (hw + rows_per_block - 1) / rows_per_block);
  int ss_blocks = static_cast<int>(bt_total) * nchunks;
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "adaln_norm_backward_cuda", [&] {
    adaln_norm_backward_gradx_kernel<scalar_t><<<rows, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        grad_output.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(), scale.data_ptr<scalar_t>(),
        dx.data_ptr<scalar_t>(), mean_rstd.data_ptr<float>(), rows, cols, hw, static_cast<float>(eps));
    adaln_norm_backward_scale_shift_kernel<scalar_t><<<ss_blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        grad_output.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(), mean_rstd.data_ptr<float>(),
        dscale.data_ptr<float>(), dshift.data_ptr<float>(), rows, cols, hw, static_cast<int>(bt_total), rows_per_block);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dx, dscale, dshift};
}

// ---------------------------------------------------------------------------
// Fused AdamW step (decoupled weight decay)
//   m = m*beta1 + (1-beta1)*g
//   v = v*beta2 + (1-beta2)*g^2
//   p = p*(1 - lr*wd) - lr * (m/bc1) / (sqrt(v/bc2) + eps)
// ---------------------------------------------------------------------------

__global__ void adamw_step_kernel(const float* __restrict__ g,
                                  float* __restrict__ p,
                                  float* __restrict__ m,
                                  float* __restrict__ v,
                                  int64_t n,
                                  float lr,
                                  float beta1,
                                  float beta2,
                                  float eps,
                                  float wd,
                                  float bc1,
                                  float bc2,
                                  float clip_scale) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (; idx < n; idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    // Global-norm gradient clipping folded into the step: scale the gradient
    // before it enters the momentum/variance accumulators (matches
    // clip_grad_norm_ semantics of scaling the raw gradient). clip_scale is
    // min(1, max_grad_norm / grad_norm) computed on the host.
    float gv = g[idx] * clip_scale;
    float ms = m[idx] * beta1 + (1.0f - beta1) * gv;
    float vs = v[idx] * beta2 + (1.0f - beta2) * gv * gv;
    m[idx] = ms;
    v[idx] = vs;
    float step = lr * (ms / bc1) / (sqrtf(vs / bc2) + eps);
    p[idx] = p[idx] * (1.0f - lr * wd) - step;
  }
}

void adamw_step_cuda(torch::Tensor grad, torch::Tensor param, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
                     double lr, double beta1, double beta2, double eps, double weight_decay,
                     double bias_correction1, double bias_correction2, double clip_scale) {
  int64_t n = grad.numel();
  const int threads = 256;
  int blocks = blocks_for(n, threads);
  adamw_step_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      grad.data_ptr<float>(), param.data_ptr<float>(), exp_avg.data_ptr<float>(), exp_avg_sq.data_ptr<float>(),
      n, static_cast<float>(lr), static_cast<float>(beta1), static_cast<float>(beta2),
      static_cast<float>(eps), static_cast<float>(weight_decay),
      static_cast<float>(bias_correction1), static_cast<float>(bias_correction2),
      static_cast<float>(clip_scale));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
