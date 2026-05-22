/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstdint>

#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace turbomind {

struct SamplingParams {
    void*          logits;
    int            stride;
    int*           indices;
    int*           kept;
    curandState_t* curandstate;
    size_t         batch_size;
    int*           output_ids;
    int*           sequence_length;
    int*           forced_ids;
    void*          sampled_logprobs;
    int*           sampled_indexes;
    int*           sampled_nums;
};

template<typename T>
void invokeSampling(SamplingParams& params, cudaStream_t stream);

void invokeGreedyFromLogits(float*       logits,
                            int          stride,
                            int          vocab_size,
                            int          batch_size,
                            int*         output_ids,
                            int*         sequence_length,
                            cudaStream_t stream);

void invokeTokenDecisionFromLogits(float*       logits,
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
                                   const int*   greedy_fallbacks,
                                   int*         forced_ids,
                                   int*         top_ks,
                                   int*         kept,
                                   int*         indices,
                                   cudaStream_t stream);

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
                                  const int*   greedy_fallbacks,
                                  int*         forced_ids,
                                  int*         top_ks,
                                  int*         kept,
                                  int*         indices,
                                  cudaStream_t stream);

}  // namespace turbomind
