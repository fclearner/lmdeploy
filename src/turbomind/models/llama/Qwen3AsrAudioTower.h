// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <vector>

#include "src/turbomind/core/core.h"
#include "src/turbomind/models/llama/LlamaLinear.h"
#include "src/turbomind/models/llama/Qwen3AsrAudioTowerWeight.h"
#include "src/turbomind/models/llama/llama_params.h"

namespace turbomind {

class Qwen3AsrAudioTower {
public:
    Qwen3AsrAudioTower(const AudioParam& audio, LlamaLinear& linear);

    Tensor Forward(const Tensor& audio_features,
                   const Tensor& feature_lens,
                   const Qwen3AsrAudioTowerWeight& weights);

private:
    struct ChunkPlan {
        Buffer_<int> audio_ids;
        Buffer_<int> starts;
        Buffer_<int> lengths;
        Buffer_<int> valid_lens;
        Buffer_<int> output_offsets;
        Buffer_<int> cu_seqlens;
        int          max_chunk_len = 0;
        int          max_conv_len  = 0;
        int          max_segment_len = 0;
        int          chunk_count   = 0;
        int          token_count   = 0;
    };

    static int FeatureLengthAfterConv(int input_length);

    ChunkPlan BuildChunkPlan(const Tensor& feature_lens) const;

    Tensor RunConvFrontend(const Tensor& audio_features,
                           const ChunkPlan& plan,
                           const Qwen3AsrAudioTowerWeight& weights);

    Tensor RunEncoder(Tensor hidden_states,
                      const Buffer_<int>& cu_seqlens,
                      int max_segment_len,
                      const Qwen3AsrAudioTowerWeight& weights);

private:
    AudioParam  audio_;
    LlamaLinear linear_;
};

}  // namespace turbomind
