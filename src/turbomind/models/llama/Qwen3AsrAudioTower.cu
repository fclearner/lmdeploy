// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/llama/Qwen3AsrAudioTower.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "src/turbomind/core/cuda_data_type.h"
#include "src/turbomind/core/copy.h"
#include "src/turbomind/kernels/norm/rms_norm.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/utils/cuda_type_utils.cuh"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {
namespace {

constexpr float kLayerNormEps = 1e-5f;

bool IsAudioProfileEnabled()
{
    const char* value = std::getenv("LMDEPLOY_QWEN3_ASR_PROFILE");
    return value != nullptr && value[0] != '\0' && !(value[0] == '0' && value[1] == '\0');
}

bool IsEnvFlagEnabled(const char* name)
{
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && !(value[0] == '0' && value[1] == '\0');
}

bool IsEnvFlagDisabled(const char* name)
{
    const char* value = std::getenv(name);
    return value != nullptr && value[0] == '0' && value[1] == '\0';
}

class AudioTowerProfile {
public:
    explicit AudioTowerProfile(cudaStream_t stream): enabled_{IsAudioProfileEnabled()}, stream_{stream}
    {
        if (!enabled_) {
            return;
        }
        check_cuda_error(cudaEventCreate(&start_));
        check_cuda_error(cudaEventCreate(&stop_));
        check_cuda_error(cudaEventRecord(start_, stream_));
    }

    ~AudioTowerProfile()
    {
        if (!enabled_) {
            return;
        }
        cudaEventDestroy(stop_);
        cudaEventDestroy(start_);
    }

    bool enabled() const noexcept
    {
        return enabled_;
    }

    void AddCpu(const char* name, float ms)
    {
        if (enabled_) {
            records_.push_back({name, ms});
        }
    }

    void Mark(const char* name)
    {
        if (!enabled_) {
            return;
        }
        check_cuda_error(cudaEventRecord(stop_, stream_));
        check_cuda_error(cudaEventSynchronize(stop_));
        float ms = 0.f;
        check_cuda_error(cudaEventElapsedTime(&ms, start_, stop_));
        records_.push_back({name, ms});
        std::swap(start_, stop_);
    }

    void Flush(int audio_count, int chunk_count, int token_count, int max_segment_len) const
    {
        if (!enabled_) {
            return;
        }

        struct Aggregate {
            std::string name;
            float       ms    = 0.f;
            int         count = 0;
        };

        float total_ms = 0.f;
        std::vector<Aggregate> aggregates;
        for (const auto& record : records_) {
            total_ms += record.ms;
            auto iter = std::find_if(aggregates.begin(), aggregates.end(), [&](const Aggregate& aggregate) {
                return aggregate.name == record.name;
            });
            if (iter == aggregates.end()) {
                aggregates.push_back({record.name, record.ms, 1});
            }
            else {
                iter->ms += record.ms;
                ++iter->count;
            }
        }

        std::ostringstream details;
        details.setf(std::ios::fixed);
        details.precision(3);
        for (const auto& aggregate : aggregates) {
            details << aggregate.name << '=' << aggregate.ms << "ms";
            if (aggregate.count > 1) {
                details << '/' << aggregate.count;
            }
            details << ' ';
        }

        TM_LOG_WARNING("qwen3_asr_audio_profile audio_count={} chunk_count={} token_count={} max_segment_len={} "
                       "total_ms={:.3f} {}",
                       audio_count,
                       chunk_count,
                       token_count,
                       max_segment_len,
                       total_ms,
                       details.str());
    }

private:
    struct Record {
        std::string name;
        float       ms;
    };

    bool                enabled_;
    cudaStream_t        stream_;
    cudaEvent_t         start_{};
    cudaEvent_t         stop_{};
    std::vector<Record> records_;
};

thread_local AudioTowerProfile* current_audio_profile = nullptr;

class ScopedAudioTowerProfile {
public:
    explicit ScopedAudioTowerProfile(AudioTowerProfile* profile): previous_{current_audio_profile}
    {
        current_audio_profile = profile;
    }

    ~ScopedAudioTowerProfile()
    {
        current_audio_profile = previous_;
    }

private:
    AudioTowerProfile* previous_;
};

AudioTowerProfile* CurrentAudioProfile()
{
    return current_audio_profile;
}

bool UseGemmConv()
{
    static const bool enabled = !IsEnvFlagDisabled("LMDEPLOY_QWEN3_ASR_CONV_GEMM");
    return enabled;
}

bool UseWarpConv()
{
    static const bool enabled = IsEnvFlagEnabled("LMDEPLOY_QWEN3_ASR_CONV_WARP");
    return enabled;
}

cublasGemmAlgo_t ConvGemmAlgo()
{
    static const bool use_tensor_op = IsEnvFlagEnabled("LMDEPLOY_QWEN3_ASR_CONV_GEMM_TENSOR_OP");
    if (use_tensor_op) {
        return CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    }
    return CUBLAS_GEMM_DEFAULT;
}

class AudioCublasHandle {
public:
    AudioCublasHandle()
    {
        check_cuda_error(cublasCreate(&handle_));
    }

    ~AudioCublasHandle()
    {
        cublasDestroy(handle_);
    }

    cublasHandle_t Get(cudaStream_t stream)
    {
        if (stream_ != stream) {
            check_cuda_error(cublasSetStream(handle_, stream));
            stream_ = stream;
        }
        return handle_;
    }

private:
    cublasHandle_t handle_{};
    cudaStream_t   stream_{};
};

AudioCublasHandle& GetAudioCublasHandle()
{
    thread_local AudioCublasHandle handle;
    return handle;
}

int ConvOutputLength(int input_length)
{
    return (input_length - 1) / 2 + 1;
}

template<class T>
__device__ float LoadAsFloat(const T* ptr, int index)
{
    return cuda_cast<float>(ptr[index]);
}

template<class T>
__device__ void StoreFromFloat(T* ptr, int index, float value)
{
    ptr[index] = cuda_cast<T>(value);
}

template<class T>
__device__ float GeluDevice(float x)
{
    return 0.5f * x * (1.f + erff(x * 0.7071067811865476f));
}

template<class T>
__global__ void PrepareChunkFeaturesKernel(T*         output,
                                           const T*   input,
                                           const int* chunk_audio_ids,
                                           const int* chunk_starts,
                                           const int* chunk_lengths,
                                           int        chunk_count,
                                           int        mel_bins,
                                           int        input_frames,
                                           int        chunk_frames)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = chunk_count * mel_bins * chunk_frames;
    if (index >= total) {
        return;
    }

    const int frame = index % chunk_frames;
    const int mel   = (index / chunk_frames) % mel_bins;
    const int chunk = index / (mel_bins * chunk_frames);

    float value = 0.f;
    if (frame < chunk_lengths[chunk]) {
        const int audio_id = chunk_audio_ids[chunk];
        const int src_pos  = chunk_starts[chunk] + frame;
        value = LoadAsFloat(input, (audio_id * mel_bins + mel) * input_frames + src_pos);
    }
    StoreFromFloat(output, index, value);
}

template<class T>
__global__ void Conv2dGeluKernel(T*       output,
                                 const T* input,
                                 const T* weight,
                                 const T* bias,
                                 int      batch,
                                 int      in_channels,
                                 int      in_freq,
                                 int      in_time,
                                 int      out_channels,
                                 int      out_freq,
                                 int      out_time)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch * out_channels * out_freq * out_time;
    if (index >= total) {
        return;
    }

    int rest = index;
    const int time = rest % out_time;
    rest /= out_time;
    const int freq = rest % out_freq;
    rest /= out_freq;
    const int out_channel = rest % out_channels;
    const int batch_id    = rest / out_channels;

    float acc = LoadAsFloat(bias, out_channel);
    for (int in_channel = 0; in_channel < in_channels; ++in_channel) {
        for (int kernel_freq = 0; kernel_freq < 3; ++kernel_freq) {
            const int src_freq = freq * 2 + kernel_freq - 1;
            if (src_freq < 0 || src_freq >= in_freq) {
                continue;
            }
            for (int kernel_time = 0; kernel_time < 3; ++kernel_time) {
                const int src_time = time * 2 + kernel_time - 1;
                if (src_time < 0 || src_time >= in_time) {
                    continue;
                }

                const int src_index =
                    ((batch_id * in_channels + in_channel) * in_freq + src_freq) * in_time + src_time;
                const int weight_index =
                    ((out_channel * in_channels + in_channel) * 3 + kernel_freq) * 3 + kernel_time;
                acc += LoadAsFloat(input, src_index) * LoadAsFloat(weight, weight_index);
            }
        }
    }

    StoreFromFloat(output, index, GeluDevice<T>(acc));
}

template<class T>
__global__ void Conv2dIm2ColKernel(T*       columns,
                                   const T* input,
                                   int      batch,
                                   int      in_channels,
                                   int      in_freq,
                                   int      in_time,
                                   int      out_freq,
                                   int      out_time)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int kernel_dim = in_channels * 3 * 3;
    const int rows = batch * out_freq * out_time;
    const int total = rows * kernel_dim;
    if (index >= total) {
        return;
    }

    const int kernel_index = index % kernel_dim;
    const int row = index / kernel_dim;

    const int kernel_time = kernel_index % 3;
    const int kernel_freq = (kernel_index / 3) % 3;
    const int channel = kernel_index / 9;

    const int time = row % out_time;
    const int freq = (row / out_time) % out_freq;
    const int batch_id = row / (out_freq * out_time);

    const int src_freq = freq * 2 + kernel_freq - 1;
    const int src_time = time * 2 + kernel_time - 1;

    float value = 0.f;
    if (src_freq >= 0 && src_freq < in_freq && src_time >= 0 && src_time < in_time) {
        const int src_index = ((batch_id * in_channels + channel) * in_freq + src_freq) * in_time + src_time;
        value = LoadAsFloat(input, src_index);
    }
    StoreFromFloat(columns, index, value);
}

template<class T>
__global__ void Conv2dGeluWarpKernel(T*       output,
                                     const T* input,
                                     const T* weight,
                                     const T* bias,
                                     int      batch,
                                     int      in_channels,
                                     int      in_freq,
                                     int      in_time,
                                     int      out_channels,
                                     int      out_freq,
                                     int      out_time)
{
    constexpr int kWarpSize = 32;
    constexpr int kWarpsPerBlock = 8;

    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    const int index = blockIdx.x * kWarpsPerBlock + warp;
    const int total = batch * out_channels * out_freq * out_time;
    if (index >= total) {
        return;
    }

    int rest = index;
    const int time = rest % out_time;
    rest /= out_time;
    const int freq = rest % out_freq;
    rest /= out_freq;
    const int out_channel = rest % out_channels;
    const int batch_id = rest / out_channels;

    const int kernel_dim = in_channels * 3 * 3;
    float acc = 0.f;
    for (int kernel_index = lane; kernel_index < kernel_dim; kernel_index += kWarpSize) {
        const int kernel_time = kernel_index % 3;
        const int kernel_freq = (kernel_index / 3) % 3;
        const int in_channel = kernel_index / 9;
        const int src_freq = freq * 2 + kernel_freq - 1;
        const int src_time = time * 2 + kernel_time - 1;
        if (src_freq < 0 || src_freq >= in_freq || src_time < 0 || src_time >= in_time) {
            continue;
        }
        const int src_index = ((batch_id * in_channels + in_channel) * in_freq + src_freq) * in_time + src_time;
        const int weight_index = out_channel * kernel_dim + kernel_index;
        acc += LoadAsFloat(input, src_index) * LoadAsFloat(weight, weight_index);
    }

    unsigned mask = 0xffffffffU;
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(mask, acc, offset);
    }
    if (lane == 0) {
        StoreFromFloat(output, index, GeluDevice<T>(acc + LoadAsFloat(bias, out_channel)));
    }
}

template<class T>
__global__ void Conv2dRowMajorToNchwGeluKernel(T*       output,
                                               const float* gemm_output,
                                               const T* bias,
                                               int      batch,
                                               int      out_channels,
                                               int      out_freq,
                                               int      out_time)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch * out_channels * out_freq * out_time;
    if (index >= total) {
        return;
    }

    int rest = index;
    const int time = rest % out_time;
    rest /= out_time;
    const int freq = rest % out_freq;
    rest /= out_freq;
    const int out_channel = rest % out_channels;
    const int batch_id = rest / out_channels;

    const int row = (batch_id * out_freq + freq) * out_time + time;
    const int col = out_channel;
    const float value = gemm_output[row * out_channels + col] + LoadAsFloat(bias, out_channel);
    StoreFromFloat(output, index, GeluDevice<T>(value));
}

template<class T>
__global__ void FlattenConvOutputKernel(T* output,
                                        const T* input,
                                        int      batch,
                                        int      channels,
                                        int      freq,
                                        int      time)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch * time * channels * freq;
    if (index >= total) {
        return;
    }

    int rest = index;
    const int freq_index = rest % freq;
    rest /= freq;
    const int channel = rest % channels;
    rest /= channels;
    const int time_index = rest % time;
    const int batch_id   = rest / time;

    const int src_index = ((batch_id * channels + channel) * freq + freq_index) * time + time_index;
    StoreFromFloat(output, index, LoadAsFloat(input, src_index));
}

template<class T>
__global__ void AddSinusoidPositionKernel(T* data, int batch, int time, int dim)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch * time * dim;
    if (index >= total) {
        return;
    }

    const int dim_index  = index % dim;
    const int time_index = (index / dim) % time;
    const int half_dim   = dim / 2;
    const int basis      = dim_index < half_dim ? dim_index : dim_index - half_dim;
    const float log_inc  = logf(10000.f) / static_cast<float>(half_dim - 1);
    const float scaled   = static_cast<float>(time_index) * expf(-log_inc * static_cast<float>(basis));
    const float pos      = dim_index < half_dim ? sinf(scaled) : cosf(scaled);
    StoreFromFloat(data, index, LoadAsFloat(data, index) + pos);
}

template<class T>
__global__ void GatherValidChunksKernel(T*         output,
                                        const T*   input,
                                        const int* valid_lens,
                                        const int* output_offsets,
                                        int        chunk_count,
                                        int        max_time,
                                        int        dim)
{
    const int dim_index = threadIdx.x + blockIdx.x * blockDim.x;
    const int time      = blockIdx.y;
    const int chunk     = blockIdx.z;
    if (chunk >= chunk_count || time >= valid_lens[chunk] || dim_index >= dim) {
        return;
    }

    const int dst_token = output_offsets[chunk] + time;
    const int src_index = (chunk * max_time + time) * dim + dim_index;
    const int dst_index = dst_token * dim + dim_index;
    StoreFromFloat(output, dst_index, LoadAsFloat(input, src_index));
}

template<class T>
__global__ void LayerNormKernel(T*       output,
                                const T* input,
                                const T* weight,
                                const T* bias,
                                int      rows,
                                int      dim,
                                float    eps)
{
    __shared__ float sums[1024];
    __shared__ float square_sums[1024];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    float sum = 0.f;
    float square_sum = 0.f;
    for (int i = tid; i < dim; i += blockDim.x) {
        const float value = LoadAsFloat(input, row * dim + i);
        sum += value;
        square_sum += value * value;
    }

    sums[tid]        = sum;
    square_sums[tid] = square_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sums[tid] += sums[tid + stride];
            square_sums[tid] += square_sums[tid + stride];
        }
        __syncthreads();
    }

    const float mean = sums[0] / static_cast<float>(dim);
    const float var  = square_sums[0] / static_cast<float>(dim) - mean * mean;
    const float inv  = rsqrtf(fmaxf(var, 0.f) + eps);

    for (int i = tid; i < dim; i += blockDim.x) {
        const float value = LoadAsFloat(input, row * dim + i);
        const float norm  = (value - mean) * inv;
        const float out   = norm * LoadAsFloat(weight, i) + LoadAsFloat(bias, i);
        StoreFromFloat(output, row * dim + i, out);
    }
}

template<class T>
__global__ void GeluInplaceKernel(T* data, int total)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total) {
        StoreFromFloat(data, index, GeluDevice<T>(LoadAsFloat(data, index)));
    }
}

template<class T>
__global__ void AddResidualKernel(T* output, const T* residual, int total, bool clamp_fp16)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) {
        return;
    }

    float value = LoadAsFloat(output, index) + LoadAsFloat(residual, index);
    if (clamp_fp16) {
        value = fminf(fmaxf(value, -64504.f), 64504.f);
    }
    StoreFromFloat(output, index, value);
}

template<class T>
__global__ void AudioAttentionKernel(T*         output,
                                     const T*   query,
                                     const T*   key,
                                     const T*   value,
                                     const int* cu_seqlens,
                                     int        head_num,
                                     int        head_dim,
                                     int        max_segment_len,
                                     float      scale)
{
    extern __shared__ float shared[];
    float* scores = shared;

    const int query_index = blockIdx.x;
    const int head        = blockIdx.y;
    const int segment     = blockIdx.z;
    const int begin       = cu_seqlens[segment];
    const int end         = cu_seqlens[segment + 1];
    const int seq_len     = end - begin;
    const int tid         = threadIdx.x;

    if (query_index >= seq_len) {
        return;
    }

    const int query_token = begin + query_index;
    const int query_base  = (query_token * head_num + head) * head_dim;

    if (tid == 0) {
        float max_score = -3.402823466e+38F;
        for (int key_index = 0; key_index < seq_len; ++key_index) {
            const int key_token = begin + key_index;
            const int key_base  = (key_token * head_num + head) * head_dim;
            float dot = 0.f;
            for (int dim = 0; dim < head_dim; ++dim) {
                dot += LoadAsFloat(query, query_base + dim) * LoadAsFloat(key, key_base + dim);
            }
            float score = dot * scale;
            if (!(score >= -3.402823466e+38F && score <= 3.402823466e+38F)) {
                score = -3.402823466e+38F;
            }
            scores[key_index] = score;
            max_score = fmaxf(max_score, score);
        }

        float prob_sum = 0.f;
        for (int key_index = 0; key_index < seq_len; ++key_index) {
            const float prob = expf(scores[key_index] - max_score);
            scores[key_index] = prob;
            prob_sum += prob;
        }
        if (prob_sum > 0.f && prob_sum <= 3.402823466e+38F) {
            const float inv_sum = 1.f / prob_sum;
            for (int key_index = 0; key_index < seq_len; ++key_index) {
                scores[key_index] *= inv_sum;
            }
        }
        else {
            const float uniform_prob = 1.f / static_cast<float>(seq_len);
            for (int key_index = 0; key_index < seq_len; ++key_index) {
                scores[key_index] = uniform_prob;
            }
        }
    }
    __syncthreads();

    for (int dim = tid; dim < head_dim; dim += blockDim.x) {
        float acc = 0.f;
        for (int key_index = 0; key_index < seq_len; ++key_index) {
            const int value_token = begin + key_index;
            const int value_base  = (value_token * head_num + head) * head_dim;
            acc += scores[key_index] * LoadAsFloat(value, value_base + dim);
        }
        StoreFromFloat(output, query_base + dim, acc);
    }
}

template<class T>
void InvokePrepareChunkFeatures(Tensor&       output,
                                const Tensor& input,
                                const Buffer_<int>& audio_ids,
                                const Buffer_<int>& starts,
                                const Buffer_<int>& lengths,
                                int           chunk_frames,
                                cudaStream_t  stream)
{
    const int chunk_count = output.shape(0);
    const int mel_bins    = output.shape(2);
    const int total       = output.size();
    const int threads     = 256;
    const int blocks      = (total + threads - 1) / threads;
    PrepareChunkFeaturesKernel<<<blocks, threads, 0, stream>>>((T*)output.raw_data(),
                                                               (const T*)input.raw_data(),
                                                               audio_ids.data(),
                                                               starts.data(),
                                                               lengths.data(),
                                                               chunk_count,
                                                               mel_bins,
                                                               input.shape(2),
                                                               chunk_frames);
}

template<class T>
void InvokeConv2dGelu(
    Tensor& output, const Tensor& input, const Tensor& weight, const Tensor& bias, cudaStream_t stream)
{
    if (UseGemmConv()) {
        const int batch = input.shape(0);
        const int in_channels = input.shape(1);
        const int in_freq = input.shape(2);
        const int in_time = input.shape(3);
        const int out_channels = output.shape(1);
        const int out_freq = output.shape(2);
        const int out_time = output.shape(3);
        const int kernel_dim = in_channels * 3 * 3;
        const int rows = batch * out_freq * out_time;

        Tensor columns{{rows, kernel_dim}, input.dtype(), kDEVICE};
        const int im2col_total = columns.size();
        const int threads = 256;
        const int im2col_blocks = (im2col_total + threads - 1) / threads;
        Conv2dIm2ColKernel<<<im2col_blocks, threads, 0, stream>>>((T*)columns.raw_data(),
                                                                  (const T*)input.raw_data(),
                                                                  batch,
                                                                  in_channels,
                                                                  in_freq,
                                                                  in_time,
                                                                  out_freq,
                                                                  out_time);
        if (auto* profile = CurrentAudioProfile()) {
            profile->Mark("conv.im2col");
        }

        Tensor gemm_output{{rows, out_channels}, kFloat, kDEVICE};
        const float alpha = 1.f;
        const float beta = 0.f;
        auto handle = GetAudioCublasHandle().Get(stream);
        check_cuda_error(cublasGemmEx(handle,
                                      CUBLAS_OP_T,
                                      CUBLAS_OP_N,
                                      out_channels,
                                      rows,
                                      kernel_dim,
                                      &alpha,
                                      weight.raw_data(),
                                      to_cuda_dtype(weight.dtype()),
                                      kernel_dim,
                                      columns.raw_data(),
                                      to_cuda_dtype(columns.dtype()),
                                      kernel_dim,
                                      &beta,
                                      gemm_output.raw_data(),
                                      to_cuda_dtype(gemm_output.dtype()),
                                      out_channels,
                                      CUDA_R_32F,
                                      ConvGemmAlgo()));
        if (auto* profile = CurrentAudioProfile()) {
            profile->Mark("conv.gemm");
        }

        const int output_total = output.size();
        const int output_blocks = (output_total + threads - 1) / threads;
        Conv2dRowMajorToNchwGeluKernel<<<output_blocks, threads, 0, stream>>>((T*)output.raw_data(),
                                                                              (const float*)gemm_output.raw_data(),
                                                                              (const T*)bias.raw_data(),
                                                                              batch,
                                                                              out_channels,
                                                                              out_freq,
                                                                              out_time);
        if (auto* profile = CurrentAudioProfile()) {
            profile->Mark("conv.bias_gelu_layout");
        }
        sync_check_cuda_error();
        return;
    }

    if (UseWarpConv() && input.shape(1) >= 32) {
        constexpr int kWarpsPerBlock = 8;
        constexpr int kThreads = 32 * kWarpsPerBlock;
        const int total = output.size();
        const int blocks = (total + kWarpsPerBlock - 1) / kWarpsPerBlock;
        Conv2dGeluWarpKernel<<<blocks, kThreads, 0, stream>>>((T*)output.raw_data(),
                                                             (const T*)input.raw_data(),
                                                             (const T*)weight.raw_data(),
                                                             (const T*)bias.raw_data(),
                                                             input.shape(0),
                                                             input.shape(1),
                                                             input.shape(2),
                                                             input.shape(3),
                                                             output.shape(1),
                                                             output.shape(2),
                                                             output.shape(3));
        if (auto* profile = CurrentAudioProfile()) {
            profile->Mark("conv.warp");
        }
        return;
    }

    const int total   = output.size();
    const int threads = 256;
    const int blocks  = (total + threads - 1) / threads;
    Conv2dGeluKernel<<<blocks, threads, 0, stream>>>((T*)output.raw_data(),
                                                     (const T*)input.raw_data(),
                                                     (const T*)weight.raw_data(),
                                                     (const T*)bias.raw_data(),
                                                     input.shape(0),
                                                     input.shape(1),
                                                     input.shape(2),
                                                     input.shape(3),
                                                     output.shape(1),
                                                     output.shape(2),
                                                     output.shape(3));
    if (auto* profile = CurrentAudioProfile()) {
        profile->Mark("conv.scalar");
    }
}

template<class T>
void InvokeFlattenConvOutput(Tensor& output, const Tensor& input, cudaStream_t stream)
{
    const int total   = output.size();
    const int threads = 256;
    const int blocks  = (total + threads - 1) / threads;
    FlattenConvOutputKernel<<<blocks, threads, 0, stream>>>((T*)output.raw_data(),
                                                            (const T*)input.raw_data(),
                                                            input.shape(0),
                                                            input.shape(1),
                                                            input.shape(2),
                                                            input.shape(3));
}

template<class T>
void InvokeAddSinusoidPosition(Tensor& data, int batch, int time, int dim, cudaStream_t stream)
{
    const int total   = data.size();
    const int threads = 256;
    const int blocks  = (total + threads - 1) / threads;
    AddSinusoidPositionKernel<<<blocks, threads, 0, stream>>>((T*)data.raw_data(), batch, time, dim);
}

template<class T>
void InvokeGatherValidChunks(Tensor&           output,
                             const Tensor&     input,
                             const Buffer_<int>& valid_lens,
                             const Buffer_<int>& output_offsets,
                             cudaStream_t      stream)
{
    const int dim     = output.shape(1);
    const int threads = 256;
    const dim3 blocks((dim + threads - 1) / threads, input.shape(1), input.shape(0));
    GatherValidChunksKernel<<<blocks, threads, 0, stream>>>((T*)output.raw_data(),
                                                            (const T*)input.raw_data(),
                                                            valid_lens.data(),
                                                            output_offsets.data(),
                                                            input.shape(0),
                                                            input.shape(1),
                                                            dim);
}

template<class T>
void InvokeLayerNorm(
    Tensor& output, const Tensor& input, const Tensor& weight, const Tensor& bias, float eps, cudaStream_t stream)
{
    LayerNormKernel<<<input.shape(0), 1024, 0, stream>>>(
        (T*)output.raw_data(), (const T*)input.raw_data(), (const T*)weight.raw_data(), (const T*)bias.raw_data(),
        input.shape(0), input.shape(1), eps);
}

template<class T>
void InvokeGeluInplace(Tensor& data, cudaStream_t stream)
{
    const int total   = data.size();
    const int threads = 256;
    const int blocks  = (total + threads - 1) / threads;
    GeluInplaceKernel<<<blocks, threads, 0, stream>>>((T*)data.raw_data(), total);
}

template<class T>
void InvokeAddResidual(Tensor& output, const Tensor& residual, bool clamp_fp16, cudaStream_t stream)
{
    const int total   = output.size();
    const int threads = 256;
    const int blocks  = (total + threads - 1) / threads;
    AddResidualKernel<<<blocks, threads, 0, stream>>>(
        (T*)output.raw_data(), (const T*)residual.raw_data(), total, clamp_fp16);
}

template<class T>
void InvokeAudioAttention(Tensor&           output,
                          const Tensor&     query,
                          const Tensor&     key,
                          const Tensor&     value,
                          const Buffer_<int>& cu_seqlens,
                          int              head_num,
                          int              head_dim,
                          int              max_segment_len,
                          cudaStream_t      stream)
{
    const int segment_count = cu_seqlens.size() - 1;
    const dim3 blocks(max_segment_len, head_num, segment_count);
    const int threads = 128;
    const int shared_bytes = max_segment_len * sizeof(float);
    const float scale = 1.f / std::sqrt(static_cast<float>(head_dim));
    AudioAttentionKernel<<<blocks, threads, shared_bytes, stream>>>((T*)output.raw_data(),
                                                                    (const T*)query.raw_data(),
                                                                    (const T*)key.raw_data(),
                                                                    (const T*)value.raw_data(),
                                                                    cu_seqlens.data(),
                                                                    head_num,
                                                                    head_dim,
                                                                    max_segment_len,
                                                                    scale);
}

Buffer_<int> CopyToDevice(const Buffer_<int>& cpu_buffer)
{
    Buffer_<int> device_buffer{cpu_buffer.size(), kDEVICE};
    Copy(cpu_buffer, device_buffer);
    return device_buffer;
}

Buffer_<int> MakeCpuBuffer(const std::vector<int>& values)
{
    Buffer_<int> buffer{static_cast<ssize_t>(values.size()), kCPU};
    std::copy(values.begin(), values.end(), buffer.begin());
    return buffer;
}

Tensor EnsureDeviceTensor(const Tensor& input)
{
    if (input.device().type == kDEVICE) {
        return input;
    }
    Tensor output{input.layout(), input.dtype(), kDEVICE};
    Copy(input, output);
    return output;
}

}  // namespace

Qwen3AsrAudioTower::Qwen3AsrAudioTower(const AudioParam& audio, LlamaLinear& linear):
    audio_{audio}
{
    (void)linear;
}

int Qwen3AsrAudioTower::FeatureLengthAfterConv(int input_length)
{
    const int full_chunks = input_length / 100;
    const int tail = input_length % 100;
    if (tail == 0) {
        return full_chunks * 13;
    }
    return full_chunks * 13 + ConvOutputLength(ConvOutputLength(ConvOutputLength(tail)));
}

auto Qwen3AsrAudioTower::BuildChunkPlan(const Tensor& feature_lens) const -> ChunkPlan
{
    TM_CHECK_EQ(feature_lens.device().type, kCPU);
    TM_CHECK_EQ(feature_lens.dtype(), kInt);

    const int audio_count = feature_lens.shape(0);
    const int chunk_frames = audio_.n_window * 2;
    const int window_factor = std::max(1, audio_.n_window_infer / chunk_frames);
    const int* lens = feature_lens.data<int>();

    std::vector<int> audio_ids;
    std::vector<int> starts;
    std::vector<int> lengths;
    std::vector<int> valid_lens;
    std::vector<int> output_offsets;
    std::vector<int> audio_after_conv_lens;

    int token_count = 0;
    int max_chunk_len = 0;
    int max_conv_len = 0;
    for (int audio_id = 0; audio_id < audio_count; ++audio_id) {
        const int audio_len = lens[audio_id];
        TM_CHECK_GT(audio_len, 0);
        audio_after_conv_lens.push_back(FeatureLengthAfterConv(audio_len));
        for (int start = 0; start < audio_len; start += chunk_frames) {
            const int chunk_len = std::min(chunk_frames, audio_len - start);
            const int conv_len = FeatureLengthAfterConv(chunk_len);
            audio_ids.push_back(audio_id);
            starts.push_back(start);
            lengths.push_back(chunk_len);
            valid_lens.push_back(conv_len);
            output_offsets.push_back(token_count);
            token_count += conv_len;
            max_chunk_len = std::max(max_chunk_len, chunk_len);
            max_conv_len = std::max(max_conv_len, conv_len);
        }
    }

    std::vector<int> cu_seqlens{0};
    int max_segment_len = 0;
    const int window_after_conv = std::max(1, max_conv_len * window_factor);
    for (const int audio_len : audio_after_conv_lens) {
        int remaining = audio_len;
        while (remaining > 0) {
            const int segment_len = std::min(window_after_conv, remaining);
            cu_seqlens.push_back(cu_seqlens.back() + segment_len);
            max_segment_len = std::max(max_segment_len, segment_len);
            remaining -= segment_len;
        }
    }

    ChunkPlan plan;
    plan.audio_ids      = MakeCpuBuffer(audio_ids);
    plan.starts         = MakeCpuBuffer(starts);
    plan.lengths        = MakeCpuBuffer(lengths);
    plan.valid_lens     = MakeCpuBuffer(valid_lens);
    plan.output_offsets = MakeCpuBuffer(output_offsets);
    plan.cu_seqlens     = MakeCpuBuffer(cu_seqlens);
    plan.max_chunk_len  = max_chunk_len;
    plan.max_conv_len   = max_conv_len;
    plan.max_segment_len = max_segment_len;
    plan.chunk_count    = static_cast<int>(audio_ids.size());
    plan.token_count    = token_count;
    return plan;
}

Tensor Qwen3AsrAudioTower::RunConvFrontend(const Tensor&                  audio_features,
                                           const ChunkPlan&               plan,
                                           const Qwen3AsrAudioTowerWeight& weights)
{
    const auto stream = core::Context::stream().handle();
    const auto dtype = audio_features.dtype();

    auto audio_ids      = CopyToDevice(plan.audio_ids);
    auto starts         = CopyToDevice(plan.starts);
    auto lengths        = CopyToDevice(plan.lengths);
    auto valid_lens     = CopyToDevice(plan.valid_lens);
    auto output_offsets = CopyToDevice(plan.output_offsets);
    if (auto* profile = CurrentAudioProfile()) {
        profile->Mark("conv.copy_plan");
    }

    Tensor padded{{plan.chunk_count, 1, audio_.num_mel_bins, plan.max_chunk_len}, dtype, kDEVICE};
    auto prepare = [&](auto t) {
        using T = decltype(t);
        InvokePrepareChunkFeatures<T>(padded, audio_features, audio_ids, starts, lengths, plan.max_chunk_len, stream);
    };
    TM_DISPATCH_PRIMARY_DTYPES(dtype, prepare);
    sync_check_cuda_error();
    if (auto* profile = CurrentAudioProfile()) {
        profile->Mark("conv.prepare");
    }

    const int freq1 = ConvOutputLength(audio_.num_mel_bins);
    const int time1 = ConvOutputLength(plan.max_chunk_len);
    Tensor conv1{{plan.chunk_count, audio_.downsample_hidden_size, freq1, time1}, dtype, kDEVICE};
    auto conv2d1 = [&](auto t) {
        using T = decltype(t);
        InvokeConv2dGelu<T>(conv1, padded, weights.conv2d1_weight, weights.conv2d1_bias, stream);
    };
    TM_DISPATCH_PRIMARY_DTYPES(dtype, conv2d1);
    sync_check_cuda_error();

    const int freq2 = ConvOutputLength(freq1);
    const int time2 = ConvOutputLength(time1);
    Tensor conv2{{plan.chunk_count, audio_.downsample_hidden_size, freq2, time2}, dtype, kDEVICE};
    auto conv2d2 = [&](auto t) {
        using T = decltype(t);
        InvokeConv2dGelu<T>(conv2, conv1, weights.conv2d2_weight, weights.conv2d2_bias, stream);
    };
    TM_DISPATCH_PRIMARY_DTYPES(dtype, conv2d2);
    sync_check_cuda_error();

    const int freq3 = ConvOutputLength(freq2);
    const int time3 = ConvOutputLength(time2);
    Tensor conv3{{plan.chunk_count, audio_.downsample_hidden_size, freq3, time3}, dtype, kDEVICE};
    auto conv2d3 = [&](auto t) {
        using T = decltype(t);
        InvokeConv2dGelu<T>(conv3, conv2, weights.conv2d3_weight, weights.conv2d3_bias, stream);
    };
    TM_DISPATCH_PRIMARY_DTYPES(dtype, conv2d3);
    sync_check_cuda_error();

    Tensor flat{{plan.chunk_count * time3, audio_.downsample_hidden_size * freq3}, dtype, kDEVICE};
    auto flatten = [&](auto t) {
        using T = decltype(t);
        InvokeFlattenConvOutput<T>(flat, conv3, stream);
    };
    TM_DISPATCH_PRIMARY_DTYPES(dtype, flatten);
    sync_check_cuda_error();
    if (auto* profile = CurrentAudioProfile()) {
        profile->Mark("conv.flatten");
    }

    Tensor padded_embed = linear_.Forward(flat, weights.conv_out);
    sync_check_cuda_error();
    if (auto* profile = CurrentAudioProfile()) {
        profile->Mark("conv.out_linear");
    }

    auto add_pos = [&](auto t) {
        using T = decltype(t);
        InvokeAddSinusoidPosition<T>(padded_embed, plan.chunk_count, time3, audio_.d_model, stream);
    };
    TM_DISPATCH_PRIMARY_DTYPES(dtype, add_pos);
    sync_check_cuda_error();
    if (auto* profile = CurrentAudioProfile()) {
        profile->Mark("conv.add_pos");
    }

    Tensor hidden{{plan.token_count, audio_.d_model}, dtype, kDEVICE};
    auto gather = [&](auto t) {
        using T = decltype(t);
        InvokeGatherValidChunks<T>(hidden, padded_embed.view({plan.chunk_count, time3, audio_.d_model}),
                                   valid_lens, output_offsets, stream);
    };
    TM_DISPATCH_PRIMARY_DTYPES(dtype, gather);
    sync_check_cuda_error();
    if (auto* profile = CurrentAudioProfile()) {
        profile->Mark("conv.gather");
    }
    sync_check_cuda_error();
    return hidden;
}

Tensor Qwen3AsrAudioTower::RunEncoder(Tensor                             hidden_states,
                                      const Buffer_<int>&                 cu_seqlens,
                                      int                                max_segment_len,
                                      const Qwen3AsrAudioTowerWeight&     weights)
{
    const auto stream = core::Context::stream().handle();
    const auto dtype  = hidden_states.dtype();
    const int head_dim = audio_.d_model / audio_.encoder_attention_heads;

    int layer_idx = 0;
    for (const auto& layer_ptr : weights.layers) {
        const auto& layer = *layer_ptr;
        Tensor normed{{hidden_states.shape(0), audio_.d_model}, dtype, kDEVICE};
        auto norm_attn = [&](auto t) {
            using T = decltype(t);
            InvokeLayerNorm<T>(normed,
                               hidden_states,
                               layer.self_attn_layer_norm_weight,
                               layer.self_attn_layer_norm_bias,
                               kLayerNormEps,
                               stream);
        };
        TM_DISPATCH_PRIMARY_DTYPES(dtype, norm_attn);
        sync_check_cuda_error();
        if (auto* profile = CurrentAudioProfile()) {
            profile->Mark("encoder.norm_attn");
        }

        Tensor query = linear_.Forward(normed, layer.q_proj);
        ApplyBias(query, layer.q_proj.bias, stream);
        Tensor key = linear_.Forward(normed, layer.k_proj);
        ApplyBias(key, layer.k_proj.bias, stream);
        Tensor value = linear_.Forward(normed, layer.v_proj);
        ApplyBias(value, layer.v_proj.bias, stream);
        sync_check_cuda_error();
        if (auto* profile = CurrentAudioProfile()) {
            profile->Mark("encoder.qkv");
        }

        Tensor attn{{hidden_states.shape(0), audio_.d_model}, dtype, kDEVICE};
        auto attention = [&](auto t) {
            using T = decltype(t);
            InvokeAudioAttention<T>(attn,
                                    query.view({query.shape(0), audio_.encoder_attention_heads, head_dim}),
                                    key.view({key.shape(0), audio_.encoder_attention_heads, head_dim}),
                                    value.view({value.shape(0), audio_.encoder_attention_heads, head_dim}),
                                    cu_seqlens,
                                    audio_.encoder_attention_heads,
                                    head_dim,
                                    max_segment_len,
                                    stream);
        };
        TM_DISPATCH_PRIMARY_DTYPES(dtype, attention);
        sync_check_cuda_error();
        if (auto* profile = CurrentAudioProfile()) {
            profile->Mark("encoder.attention");
        }

        Tensor attn_out = linear_.Forward(attn, layer.out_proj);
        ApplyBias(attn_out, layer.out_proj.bias, stream);
        sync_check_cuda_error();
        if (auto* profile = CurrentAudioProfile()) {
            profile->Mark("encoder.out_proj");
        }
        auto add_attn_residual = [&](auto t) {
            using T = decltype(t);
            InvokeAddResidual<T>(attn_out, hidden_states, false, stream);
        };
        TM_DISPATCH_PRIMARY_DTYPES(dtype, add_attn_residual);
        sync_check_cuda_error();
        if (auto* profile = CurrentAudioProfile()) {
            profile->Mark("encoder.attn_residual");
        }
        hidden_states = attn_out;

        Tensor ffn_normed{{hidden_states.shape(0), audio_.d_model}, dtype, kDEVICE};
        auto norm_ffn = [&](auto t) {
            using T = decltype(t);
            InvokeLayerNorm<T>(
                ffn_normed, hidden_states, layer.final_layer_norm_weight, layer.final_layer_norm_bias, kLayerNormEps,
                stream);
        };
        TM_DISPATCH_PRIMARY_DTYPES(dtype, norm_ffn);
        sync_check_cuda_error();
        if (auto* profile = CurrentAudioProfile()) {
            profile->Mark("encoder.norm_ffn");
        }

        Tensor ffn = linear_.Forward(ffn_normed, layer.fc1);
        ApplyBias(ffn, layer.fc1.bias, stream);
        auto gelu = [&](auto t) {
            using T = decltype(t);
            InvokeGeluInplace<T>(ffn, stream);
        };
        TM_DISPATCH_PRIMARY_DTYPES(dtype, gelu);
        sync_check_cuda_error();
        if (auto* profile = CurrentAudioProfile()) {
            profile->Mark("encoder.fc1_gelu");
        }

        Tensor ffn_out = linear_.Forward(ffn, layer.fc2);
        ApplyBias(ffn_out, layer.fc2.bias, stream);
        auto add_ffn_residual = [&](auto t) {
            using T = decltype(t);
            InvokeAddResidual<T>(ffn_out, hidden_states, dtype == kHalf, stream);
        };
        TM_DISPATCH_PRIMARY_DTYPES(dtype, add_ffn_residual);
        sync_check_cuda_error();
        if (auto* profile = CurrentAudioProfile()) {
            profile->Mark("encoder.fc2_residual");
        }
        hidden_states = ffn_out;
        ++layer_idx;
    }

    Tensor normed{{hidden_states.shape(0), audio_.d_model}, dtype, kDEVICE};
    auto norm_post = [&](auto t) {
        using T = decltype(t);
        InvokeLayerNorm<T>(normed, hidden_states, weights.ln_post_weight, weights.ln_post_bias, kLayerNormEps, stream);
    };
    TM_DISPATCH_PRIMARY_DTYPES(dtype, norm_post);
    sync_check_cuda_error();
    if (auto* profile = CurrentAudioProfile()) {
        profile->Mark("encoder.post_norm");
    }

    Tensor projected = linear_.Forward(normed, weights.proj1);
    ApplyBias(projected, weights.proj1.bias, stream);
    auto gelu = [&](auto t) {
        using T = decltype(t);
        InvokeGeluInplace<T>(projected, stream);
    };
    TM_DISPATCH_PRIMARY_DTYPES(dtype, gelu);
    sync_check_cuda_error();
    if (auto* profile = CurrentAudioProfile()) {
        profile->Mark("encoder.proj1_gelu");
    }

    Tensor output = linear_.Forward(projected, weights.proj2);
    ApplyBias(output, weights.proj2.bias, stream);
    sync_check_cuda_error();
    if (auto* profile = CurrentAudioProfile()) {
        profile->Mark("encoder.proj2");
    }
    return output;
}

Tensor Qwen3AsrAudioTower::Forward(const Tensor&                  audio_features,
                                   const Tensor&                  feature_lens,
                                   const Qwen3AsrAudioTowerWeight& weights)
{
    NvtxScope scope("qwen3_asr_audio_tower");
    TM_CHECK(audio_.enabled);
    TM_CHECK_EQ(audio_features.ndim(), 3);
    TM_CHECK_EQ(audio_features.shape(1), audio_.num_mel_bins);
    TM_CHECK_EQ(audio_features.dtype(), weights.conv2d1_weight.dtype());
    TM_CHECK_EQ(feature_lens.ndim(), 1);
    TM_CHECK_EQ(feature_lens.shape(0), audio_features.shape(0));

    const auto stream = core::Context::stream().handle();
    AudioTowerProfile profile{stream};
    ScopedAudioTowerProfile scoped_profile{profile.enabled() ? &profile : nullptr};

    const auto build_plan_start = std::chrono::steady_clock::now();
    auto plan = BuildChunkPlan(feature_lens);
    const auto build_plan_end = std::chrono::steady_clock::now();
    profile.AddCpu("cpu.build_plan",
                   std::chrono::duration<float, std::milli>(build_plan_end - build_plan_start).count());
    if (plan.token_count == 0) {
        return Tensor{{0, audio_.output_dim}, audio_features.dtype(), kDEVICE};
    }

    Tensor device_features = EnsureDeviceTensor(audio_features);
    profile.Mark("forward.ensure_device_features");
    Tensor hidden_states   = RunConvFrontend(device_features, plan, weights);
    auto cu_seqlens        = CopyToDevice(plan.cu_seqlens);
    profile.Mark("forward.copy_cu_seqlens");
    Tensor output          = RunEncoder(hidden_states, cu_seqlens, plan.max_segment_len, weights);
    profile.Flush(audio_features.shape(0), plan.chunk_count, plan.token_count, plan.max_segment_len);
    return output;
}

}  // namespace turbomind
