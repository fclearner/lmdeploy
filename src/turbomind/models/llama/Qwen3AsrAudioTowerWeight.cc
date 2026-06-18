// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/llama/Qwen3AsrAudioTowerWeight.h"

namespace turbomind {

Qwen3AsrAudioEncoderLayerWeight::Qwen3AsrAudioEncoderLayerWeight(DataType          data_type,
                                                                 const AudioParam& audio,
                                                                 DataType          weight_type,
                                                                 int               group_size)
{
    const int hidden_dim = audio.d_model;
    const int ffn_dim    = audio.encoder_ffn_dim;

    q_proj.emplace(hidden_dim, hidden_dim, data_type, true, weight_type, group_size);
    k_proj.emplace(hidden_dim, hidden_dim, data_type, true, weight_type, group_size);
    v_proj.emplace(hidden_dim, hidden_dim, data_type, true, weight_type, group_size);
    out_proj.emplace(hidden_dim, hidden_dim, data_type, true, weight_type, group_size);
    fc1.emplace(hidden_dim, ffn_dim, data_type, true, weight_type, group_size);
    fc2.emplace(ffn_dim, hidden_dim, data_type, true, weight_type, group_size);

    register_module("self_attn.q_proj", q_proj);
    register_module("self_attn.k_proj", k_proj);
    register_module("self_attn.v_proj", v_proj);
    register_module("self_attn.out_proj", out_proj);
    register_module("fc1", fc1);
    register_module("fc2", fc2);

    self_attn_layer_norm_weight = Tensor{{hidden_dim}, data_type, kDEVICE};
    self_attn_layer_norm_bias   = Tensor{{hidden_dim}, data_type, kDEVICE};
    final_layer_norm_weight     = Tensor{{hidden_dim}, data_type, kDEVICE};
    final_layer_norm_bias       = Tensor{{hidden_dim}, data_type, kDEVICE};

    register_parameter("self_attn_layer_norm.weight", self_attn_layer_norm_weight);
    register_parameter("self_attn_layer_norm.bias", self_attn_layer_norm_bias);
    register_parameter("final_layer_norm.weight", final_layer_norm_weight);
    register_parameter("final_layer_norm.bias", final_layer_norm_bias);
}

void Qwen3AsrAudioEncoderLayerWeight::prepare()
{
    q_proj.prepare();
    k_proj.prepare();
    v_proj.prepare();
    out_proj.prepare();
    fc1.prepare();
    fc2.prepare();
}

Qwen3AsrAudioTowerWeight::Qwen3AsrAudioTowerWeight(DataType          data_type,
                                                   const AudioParam& audio,
                                                   DataType          weight_type,
                                                   int               group_size)
{
    const int conv_out_freq = (((audio.num_mel_bins + 1) / 2 + 1) / 2 + 1) / 2;
    const int conv_out_dim  = audio.downsample_hidden_size * conv_out_freq;

    conv2d1_weight = Tensor{{audio.downsample_hidden_size, 1, 3, 3}, data_type, kDEVICE};
    conv2d1_bias   = Tensor{{audio.downsample_hidden_size}, data_type, kDEVICE};
    conv2d2_weight = Tensor{{audio.downsample_hidden_size, audio.downsample_hidden_size, 3, 3}, data_type, kDEVICE};
    conv2d2_bias   = Tensor{{audio.downsample_hidden_size}, data_type, kDEVICE};
    conv2d3_weight = Tensor{{audio.downsample_hidden_size, audio.downsample_hidden_size, 3, 3}, data_type, kDEVICE};
    conv2d3_bias   = Tensor{{audio.downsample_hidden_size}, data_type, kDEVICE};
    ln_post_weight = Tensor{{audio.d_model}, data_type, kDEVICE};
    ln_post_bias   = Tensor{{audio.d_model}, data_type, kDEVICE};

    register_parameter("conv2d1.weight", conv2d1_weight);
    register_parameter("conv2d1.bias", conv2d1_bias);
    register_parameter("conv2d2.weight", conv2d2_weight);
    register_parameter("conv2d2.bias", conv2d2_bias);
    register_parameter("conv2d3.weight", conv2d3_weight);
    register_parameter("conv2d3.bias", conv2d3_bias);
    register_parameter("ln_post.weight", ln_post_weight);
    register_parameter("ln_post.bias", ln_post_bias);

    conv_out.emplace(conv_out_dim, audio.d_model, data_type, false, weight_type, group_size);
    proj1.emplace(audio.d_model, audio.d_model, data_type, true, weight_type, group_size);
    proj2.emplace(audio.d_model, audio.output_dim, data_type, true, weight_type, group_size);
    register_module("conv_out", conv_out);
    register_module("proj1", proj1);
    register_module("proj2", proj2);

    layers.reserve(audio.encoder_layers);
    for (int i = 0; i < audio.encoder_layers; ++i) {
        layers.emplace_back(new Qwen3AsrAudioEncoderLayerWeight(data_type, audio, weight_type, group_size));
        register_module("layers", *layers.back(), i);
    }
}

void Qwen3AsrAudioTowerWeight::prepare()
{
    conv_out.prepare();
    proj1.prepare();
    proj2.prepare();
    for (auto& layer : layers) {
        layer->prepare();
    }
}

}  // namespace turbomind
