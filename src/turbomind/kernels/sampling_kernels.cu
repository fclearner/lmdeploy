#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11000)
#include <cub/cub.cuh>
#else
#include "3rdparty/cub/cub.cuh"
#endif
#include "src/turbomind/kernels/sampling_kernels.h"
#include "src/turbomind/kernels/sampling_topp_kernels.h"
#include "src/turbomind/utils/constant.h"

namespace turbomind {

template<typename T, int BLOCK_SIZE>
__global__ void sampling(const T*       logits,
                         const int      stride,
                         const int*     indices,
                         const int*     kept,
                         curandState_t* curandstate,
                         int*           output_ids,
                         int*           sequence_length,
                         const int*     forced_ids,
                         T*             sampled_logprobs,
                         int*           sampled_indexes,
                         int*           sampled_nums)
{
    int tid      = threadIdx.x;
    int batch_id = blockIdx.x;
    int n        = kept[batch_id];

    logits += stride * batch_id;
    indices += stride * batch_id;

    const int forced_id = forced_ids == nullptr ? -1 : forced_ids[batch_id];
    if (forced_id >= 0) {
        if (tid == 0) {
            output_ids[batch_id] = forced_id;
            sequence_length[batch_id] += 1;
            if (sampled_logprobs != nullptr && sampled_indexes != nullptr && sampled_nums != nullptr) {
                sampled_logprobs[batch_id * kMaxLogProb] = 0.f;
                sampled_indexes[batch_id * kMaxLogProb]  = forced_id;
                sampled_nums[batch_id]                   = 1;
            }
        }
        return;
    }

    __shared__ float rand_num_s;
    __shared__ int   selected;
    if (tid == 0) {
        rand_num_s = curand_uniform(curandstate + batch_id);
    }
    __syncthreads();

    typedef cub::BlockScan<float, BLOCK_SIZE>  BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;

    float                 local_rand = rand_num_s;
    float                 prefix_sum = 0.f;
    BlockPrefixCallbackOp prefix_op{0};
    int                   end = (n + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
    for (int i = tid; i < end; i += BLOCK_SIZE) {
        float thread_logit = (i < n) ? static_cast<float>(logits[i]) : 0.f;
        BlockScan(temp_storage).InclusiveSum(thread_logit, prefix_sum, prefix_op);
        auto count = __syncthreads_count(prefix_sum > local_rand);
        if (count != 0 || (i + BLOCK_SIZE) >= end) {
            if (tid == min(BLOCK_SIZE - count, BLOCK_SIZE - 1)) {
                selected             = min(i, n - 1);
                output_ids[batch_id] = indices[selected];
            }
            break;
        }
    }

    if (tid == 0) {
        sequence_length[batch_id] += 1;
    }

    if (sampled_logprobs != nullptr && sampled_indexes != nullptr && sampled_nums != nullptr) {
        __syncthreads();
        sampled_logprobs += batch_id * kMaxLogProb;
        sampled_indexes += batch_id * kMaxLogProb;
        int end = min(n, kMaxLogProb);
        for (int i = tid; i < end; i += BLOCK_SIZE) {
            sampled_logprobs[i] = logf(logits[i]);
            sampled_indexes[i]  = indices[i];
        }
        if (n > kMaxLogProb && selected >= kMaxLogProb) {
            if ((kMaxLogProb - 1 + BLOCK_SIZE - tid) % BLOCK_SIZE == 0) {
                sampled_logprobs[kMaxLogProb - 1] = logf(logits[selected]);
                sampled_indexes[kMaxLogProb - 1]  = indices[selected];
            }
        }
        sampled_nums[batch_id] = min(n, kMaxLogProb);
    }
}

template<typename T>
void invokeSampling(SamplingParams& params, cudaStream_t stream)
{
    const int grid  = params.batch_size;
    const int block = 256;
    sampling<T, block><<<grid, block, 0, stream>>>((T*)params.logits,
                                                   params.stride,
                                                   params.indices,
                                                   params.kept,
                                                   params.curandstate,
                                                   params.output_ids,
                                                   params.sequence_length,
                                                   params.forced_ids,
                                                   (T*)params.sampled_logprobs,
                                                   params.sampled_indexes,
                                                   params.sampled_nums);
}

template void invokeSampling<float>(SamplingParams& params, cudaStream_t stream);

__global__ void tokenDecisionFromProbs(float*       probs,
                                       int          stride,
                                       int          vocab_size,
                                       const int*   infer_types,
                                       const int*   valid_ids,
                                       const int*   invalid_ids,
                                       const int*   end_ids,
                                       const float* certainty_thresholds,
                                       const float* completion_thresholds,
                                       const float* invalid_biases,
                                       int*         forced_ids,
                                       int*         top_ks,
                                       int*         kept,
                                       int*         indices)
{
    const int batch_id   = blockIdx.x;
    const int infer_type = infer_types[batch_id];
    int       forced_id  = -1;

    if (infer_type == 0) {
        const int valid_id   = valid_ids[batch_id];
        const int invalid_id = invalid_ids[batch_id];
        if (0 <= valid_id && valid_id < vocab_size && 0 <= invalid_id && invalid_id < vocab_size) {
            const float valid_prob   = probs[(size_t)batch_id * stride + valid_id];
            const float invalid_prob = probs[(size_t)batch_id * stride + invalid_id];
            if (fmaxf(valid_prob, invalid_prob) > certainty_thresholds[batch_id]) {
                forced_id = valid_prob > invalid_prob + invalid_biases[batch_id] ? valid_id : invalid_id;
            }
        }
    }
    else if (infer_type > 0) {
        const int end_id = end_ids[batch_id];
        if (0 <= end_id && end_id < vocab_size) {
            const float end_prob = probs[(size_t)batch_id * stride + end_id];
            if (end_prob > completion_thresholds[batch_id]) {
                forced_id = end_id;
            }
        }
    }

    forced_ids[batch_id] = forced_id;
    if (forced_id >= 0) {
        top_ks[batch_id]                    = 1;
        kept[batch_id]                      = 1;
        indices[(size_t)batch_id * stride] = forced_id;
    }
}

void invokeTokenDecisionFromProbs(float*       probs,
                                  int          stride,
                                  int          vocab_size,
                                  int          batch_size,
                                  const int*   infer_types,
                                  const int*   valid_ids,
                                  const int*   invalid_ids,
                                  const int*   end_ids,
                                  const float* certainty_thresholds,
                                  const float* completion_thresholds,
                                  const float* invalid_biases,
                                  int*         forced_ids,
                                  int*         top_ks,
                                  int*         kept,
                                  int*         indices,
                                  cudaStream_t stream)
{
    tokenDecisionFromProbs<<<batch_size, 1, 0, stream>>>(probs,
                                                        stride,
                                                        vocab_size,
                                                        infer_types,
                                                        valid_ids,
                                                        invalid_ids,
                                                        end_ids,
                                                        certainty_thresholds,
                                                        completion_thresholds,
                                                        invalid_biases,
                                                        forced_ids,
                                                        top_ks,
                                                        kept,
                                                        indices);
}

}  // namespace turbomind
