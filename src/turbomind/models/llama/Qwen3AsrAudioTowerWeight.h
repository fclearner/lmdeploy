// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <memory>
#include <vector>

#include "src/turbomind/core/module.h"
#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "src/turbomind/models/llama/llama_params.h"

namespace turbomind {

struct Qwen3AsrAudioEncoderLayerWeight: public core::Module {
    Qwen3AsrAudioEncoderLayerWeight(DataType data_type, const AudioParam& audio, DataType weight_type, int group_size);

    void prepare();

    Tensor self_attn_layer_norm_weight;
    Tensor self_attn_layer_norm_bias;
    Tensor final_layer_norm_weight;
    Tensor final_layer_norm_bias;

    LlamaDenseWeight q_proj;
    LlamaDenseWeight k_proj;
    LlamaDenseWeight v_proj;
    LlamaDenseWeight out_proj;
    LlamaDenseWeight fc1;
    LlamaDenseWeight fc2;
};

struct Qwen3AsrAudioTowerWeight: public core::Module {
    Qwen3AsrAudioTowerWeight(DataType data_type, const AudioParam& audio, DataType weight_type, int group_size);

    void prepare();

    Tensor conv2d1_weight;
    Tensor conv2d1_bias;
    Tensor conv2d2_weight;
    Tensor conv2d2_bias;
    Tensor conv2d3_weight;
    Tensor conv2d3_bias;
    Tensor ln_post_weight;
    Tensor ln_post_bias;

    LlamaDenseWeight conv_out;
    LlamaDenseWeight proj1;
    LlamaDenseWeight proj2;

    std::vector<std::unique_ptr<Qwen3AsrAudioEncoderLayerWeight>> layers;
};

}  // namespace turbomind
