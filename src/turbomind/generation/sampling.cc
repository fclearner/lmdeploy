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

#include "src/turbomind/generation/sampling.h"

#include "src/turbomind/kernels/sampling_kernels.h"
#include "src/turbomind/kernels/sampling_topk_kernels.h"
#include "src/turbomind/kernels/sampling_topp_kernels.h"

#include "src/turbomind/engine/batch.h"
#include "src/turbomind/engine/request.h"

#include "src/turbomind/core/logger.h"
#include "src/turbomind/utils/constant.h"

namespace turbomind {

struct SamplingData {

    explicit SamplingData(int max_batch_size, DeviceType device)
    {
        top_k_buf = {max_batch_size, device};
        top_p_buf = {max_batch_size, device};
        min_p_buf = {max_batch_size, device};
        kept_buf  = {max_batch_size, device};
        token_decision_infer_type_buf           = {max_batch_size, device};
        token_decision_valid_id_buf             = {max_batch_size, device};
        token_decision_invalid_id_buf           = {max_batch_size, device};
        token_decision_end_id_buf               = {max_batch_size, device};
        token_decision_certainty_threshold_buf  = {max_batch_size, device};
        token_decision_completion_threshold_buf = {max_batch_size, device};
        token_decision_invalid_bias_buf         = {max_batch_size, device};
        token_decision_greedy_fallback_buf      = {max_batch_size, device};
        forced_ids_buf                          = {max_batch_size, device};

        sampled_logprobs = {max_batch_size * (ssize_t)kMaxLogProb, device};
        sampled_indices  = {max_batch_size * (ssize_t)kMaxLogProb, device};
        sampled_nums     = {max_batch_size, device};
    }

    int   max_topk = 0;
    int   min_topk = 0;
    float min_topp = 0;
    float max_minp = 0;

    Buffer_<int>   top_k_buf;
    Buffer_<float> top_p_buf;
    Buffer_<float> min_p_buf;
    Buffer_<int>   token_decision_infer_type_buf;
    Buffer_<int>   token_decision_valid_id_buf;
    Buffer_<int>   token_decision_invalid_id_buf;
    Buffer_<int>   token_decision_end_id_buf;
    Buffer_<float> token_decision_certainty_threshold_buf;
    Buffer_<float> token_decision_completion_threshold_buf;
    Buffer_<float> token_decision_invalid_bias_buf;
    Buffer_<int>   token_decision_greedy_fallback_buf;
    Buffer_<int>   forced_ids_buf;

    Buffer_<int> kept_buf;  // kept sample

    bool output_logprobs     = 0;
    bool has_greedy          = 0;
    bool all_greedy          = 0;
    bool has_token_decision  = 0;
    bool all_token_decision  = 0;

    Buffer_<float> sampled_logprobs;
    Buffer_<int>   sampled_indices;
    Buffer_<int>   sampled_nums;
};

Sampling::Sampling(const BaseGenerationParam& base, int phases): BaseGenerationParam{base}
{
    top_k_ = {max_batch_size_, kCPUpinned};
    top_p_ = {max_batch_size_, kCPUpinned};
    min_p_ = {max_batch_size_, kCPUpinned};
    kept_  = {max_batch_size_, kCPUpinned};
    token_decision_infer_type_           = {max_batch_size_, kCPUpinned};
    token_decision_valid_id_             = {max_batch_size_, kCPUpinned};
    token_decision_invalid_id_           = {max_batch_size_, kCPUpinned};
    token_decision_end_id_               = {max_batch_size_, kCPUpinned};
    token_decision_certainty_threshold_  = {max_batch_size_, kCPUpinned};
    token_decision_completion_threshold_ = {max_batch_size_, kCPUpinned};
    token_decision_invalid_bias_         = {max_batch_size_, kCPUpinned};
    token_decision_greedy_fallback_      = {max_batch_size_, kCPUpinned};

    sampled_logprobs_buf_ = {max_batch_size_ * (ssize_t)kMaxLogProb, kCPUpinned};
    sampled_indices_buf_  = {max_batch_size_ * (ssize_t)kMaxLogProb, kCPUpinned};
    sampled_nums_buf_     = {max_batch_size_, kCPUpinned};

    // constant array
    std::fill_n(kept_.data(), max_batch_size_, vocab_size_);

    for (int i = 0; i < phases; ++i) {
        data_.push_back(std::make_shared<SamplingData>(max_batch_size_, kDEVICE));
    }
}

void Sampling::Forward(int phase, TensorMap& args)
{
    // step1:
    //  - use topk / topp_minp kernel to sort and filter the scores
    //  - softmax the left score
    // step2:
    //  - sampling from left and sorted scores

    TM_LOG_DEBUG("{} start", __PRETTY_FUNCTION__);

    auto& d = *data_.at(phase);

    Tensor_<float> logits = args.at("logits");

    const auto bsz = logits.shape(0);

    auto stream = core::Context::stream().handle();

    if (d.all_greedy) {
        invokeGreedyFromLogits(logits.data(),
                               vocab_size_padded_,
                               vocab_size_,
                               bsz,
                               args.at("output_ids").data<int>(),
                               args.at("sequence_length").data<int>(),
                               stream);
        sync_check_cuda_error();
        TM_LOG_DEBUG("{} stop", __PRETTY_FUNCTION__);
        return;
    }

    Buffer_<int> indices(bsz * vocab_size_padded_, kDEVICE);

    if (d.all_token_decision) {
        invokeTokenDecisionFromLogits(logits.data(),
                                      vocab_size_padded_,
                                      vocab_size_,
                                      bsz,
                                      d.token_decision_infer_type_buf.data(),
                                      d.token_decision_valid_id_buf.data(),
                                      d.token_decision_invalid_id_buf.data(),
                                      d.token_decision_end_id_buf.data(),
                                      d.token_decision_certainty_threshold_buf.data(),
                                      d.token_decision_completion_threshold_buf.data(),
                                      d.token_decision_invalid_bias_buf.data(),
                                      d.token_decision_greedy_fallback_buf.data(),
                                      d.forced_ids_buf.data(),
                                      d.top_k_buf.data(),
                                      d.kept_buf.data(),
                                      indices.data(),
                                      stream);

        SamplingParams params{};
        params.logits          = logits.data();
        params.stride          = vocab_size_padded_;
        params.indices         = indices.data();
        params.kept            = d.kept_buf.data();
        params.curandstate     = (curandState_t*)args.at("curand_state").raw_data();
        params.batch_size      = bsz;
        params.output_ids      = args.at("output_ids").data<int>();  // (B, 1)
        params.sequence_length = args.at("sequence_length").data<int>();
        params.forced_ids      = d.forced_ids_buf.data();
        invokeSampling<float>(params, stream);
        sync_check_cuda_error();
        TM_LOG_DEBUG("{} stop", __PRETTY_FUNCTION__);
        return;
    }

    // use topk sort if some request use topk filter
    if (d.max_topk > 0) {
        // TODO: top_k >= 64 is much slower than torch.topk()
        TopKSortFilterParams params{};
        params.logits            = logits.data();
        params.sorted_logits     = logits.data();
        params.sorted_indices    = indices.data();
        params.kept              = d.kept_buf.data();
        params.top_ks            = d.top_k_buf.data();
        params.max_top_k         = d.max_topk;
        params.batch_size        = bsz;
        params.vocab_size        = vocab_size_;
        params.vocab_size_padded = vocab_size_padded_;
        invokeTopKSortFilter<float>(params, stream);
    }

    // use topp sort if some request skip topk filter
    if (d.min_topk == 0) {
        invokeSoftmax<float>(logits.data(), vocab_size_padded_, vocab_size_, bsz, d.kept_buf.data(), stream);

        if (d.has_token_decision) {
            invokeTokenDecisionFromProbs(logits.data(),
                                         vocab_size_padded_,
                                         vocab_size_,
                                         bsz,
                                         d.token_decision_infer_type_buf.data(),
                                         d.token_decision_valid_id_buf.data(),
                                         d.token_decision_invalid_id_buf.data(),
                                         d.token_decision_end_id_buf.data(),
                                         d.token_decision_certainty_threshold_buf.data(),
                                         d.token_decision_completion_threshold_buf.data(),
                                         d.token_decision_invalid_bias_buf.data(),
                                         d.token_decision_greedy_fallback_buf.data(),
                                         d.forced_ids_buf.data(),
                                         d.top_k_buf.data(),
                                         d.kept_buf.data(),
                                         indices.data(),
                                         stream);
        }

        if (!d.all_token_decision) {
            TopPSortParams params{};
            params.logits            = logits.data();
            params.sorted_logits     = logits.data();
            params.sorted_indices    = indices.data();
            params.kept              = d.kept_buf.data();
            params.top_ks            = d.top_k_buf.data();
            params.top_ps            = d.top_p_buf.data();
            params.batch_size        = bsz;
            params.vocab_size        = vocab_size_;
            params.vocab_size_padded = vocab_size_padded_;
            invokeTopPSort<float>(params, stream);
        }
    }

    // apply topp minp filter
    if (!d.all_token_decision && (d.max_minp != 0.f || d.min_topp != 1.f)) {
        TopPMinPFilterParams params{};
        params.sorted_logits     = logits.data();
        params.sorted_indices    = indices.data();
        params.kept              = d.kept_buf.data();
        params.top_ps            = d.top_p_buf.data();
        params.min_ps            = d.min_p_buf.data();
        params.batch_size        = bsz;
        params.vocab_size        = vocab_size_;
        params.vocab_size_padded = vocab_size_padded_;
        invokeTopPMinPFilter<float>(params, stream);
    }

    // sample
    {
        SamplingParams params{};
        params.logits          = logits.data();
        params.stride          = vocab_size_padded_;
        params.indices         = indices.data();
        params.kept            = d.kept_buf.data();
        params.curandstate     = (curandState_t*)args.at("curand_state").raw_data();
        params.batch_size      = bsz;
        params.output_ids      = args.at("output_ids").data<int>();  // (B, 1)
        params.sequence_length = args.at("sequence_length").data<int>();
        params.forced_ids      = d.has_token_decision ? d.forced_ids_buf.data() : nullptr;

        if (d.output_logprobs) {
            params.sampled_logprobs = d.sampled_logprobs.data();
            params.sampled_indexes  = d.sampled_indices.data();
            params.sampled_nums     = d.sampled_nums.data();
        }

        invokeSampling<float>(params, stream);
        sync_check_cuda_error();
    }

    TM_LOG_DEBUG("{} stop", __PRETTY_FUNCTION__);
}

void Sampling::Setup(int phase, TensorMap& env)
{

    const auto& rc   = env.at("batch").data<BatchData*>()[0]->rc;
    auto&       copy = *env.at("copy").data<BatchCopy*>()[0];

    const auto bsz = rc.size();

    auto& d = *data_.at(phase);
    d.output_logprobs = std::any_of(rc.begin(), rc.end(), [](auto& x) { return x->gen_cfg.output_logprobs; });
    d.has_greedy = false;
    d.all_greedy = true;
    d.has_token_decision = false;
    d.all_token_decision = true;

    for (int i = 0; i < bsz; ++i) {
        const auto& g = rc[i]->gen_cfg;
        top_k_[i] = g.top_k;
        top_p_[i] = g.top_p;
        min_p_[i] = g.min_p;

        const bool direct_validity_decision = (g.token_decision_infer_type == 0
                                               && g.token_decision_certainty_threshold <= 0.f
                                               && g.token_decision_invalid_bias == 0.f);
        const bool zero_threshold_decision = direct_validity_decision
                                            || (g.token_decision_infer_type > 0
                                                && g.token_decision_completion_threshold <= 0.f);
        const bool greedy_fallback_decision = g.token_decision_infer_type >= 0 && !zero_threshold_decision
                                              && g.top_k <= 1 && g.top_p == 1.f && g.min_p == 0.f;
        const bool sampling_token_decision = zero_threshold_decision || greedy_fallback_decision;
        const bool greedy_sampling         = g.token_decision_infer_type < 0 && !d.output_logprobs && g.top_k <= 1
                                             && g.top_p == 1.f && g.min_p == 0.f;

        token_decision_infer_type_[i]           = sampling_token_decision ? g.token_decision_infer_type : -1;
        token_decision_valid_id_[i]             = g.token_decision_valid_id;
        token_decision_invalid_id_[i]           = g.token_decision_invalid_id;
        token_decision_end_id_[i]               = g.token_decision_end_id;
        token_decision_certainty_threshold_[i]  = g.token_decision_certainty_threshold;
        token_decision_completion_threshold_[i] = g.token_decision_completion_threshold;
        token_decision_invalid_bias_[i]         = g.token_decision_invalid_bias;
        token_decision_greedy_fallback_[i]      = (greedy_fallback_decision || direct_validity_decision) ? 1 : 0;

        if (sampling_token_decision) {
            d.has_token_decision = true;
            d.all_greedy = false;
            // Token decision uses full-vocabulary probabilities. For zero-threshold
            // requests the decision always forces a token, so the row can use
            // Sampling's existing full softmax and skip later sorting/sampling.
            top_k_[i] = 0;
        }
        else {
            d.all_token_decision = false;
            if (greedy_sampling) {
                d.has_greedy = true;
            }
            else {
                d.all_greedy = false;
            }
        }
    }

    d.all_token_decision = d.all_token_decision && d.has_token_decision;
    d.all_greedy         = d.all_greedy && d.has_greedy;

    d.max_topk = *std::max_element(top_k_.begin(), top_k_.begin() + bsz);
    d.min_topk = *std::min_element(top_k_.begin(), top_k_.begin() + bsz);
    d.min_topp = *std::min_element(top_p_.begin(), top_p_.begin() + bsz);
    d.max_minp = *std::max_element(min_p_.begin(), min_p_.begin() + bsz);

    copy(top_k_.data(), bsz, d.top_k_buf.data());
    copy(top_p_.data(), bsz, d.top_p_buf.data());

    copy(min_p_.data(), bsz, d.min_p_buf.data());
    copy(kept_.data(), bsz, d.kept_buf.data());

    if (d.has_token_decision) {
        copy(token_decision_infer_type_.data(), bsz, d.token_decision_infer_type_buf.data());
        copy(token_decision_valid_id_.data(), bsz, d.token_decision_valid_id_buf.data());
        copy(token_decision_invalid_id_.data(), bsz, d.token_decision_invalid_id_buf.data());
        copy(token_decision_end_id_.data(), bsz, d.token_decision_end_id_buf.data());
        copy(token_decision_certainty_threshold_.data(), bsz, d.token_decision_certainty_threshold_buf.data());
        copy(token_decision_completion_threshold_.data(), bsz, d.token_decision_completion_threshold_buf.data());
        copy(token_decision_invalid_bias_.data(), bsz, d.token_decision_invalid_bias_buf.data());
        copy(token_decision_greedy_fallback_.data(), bsz, d.token_decision_greedy_fallback_buf.data());
    }
}

void Sampling::Fetch(int phase, TensorMap& env)
{
    auto& d    = *data_.at(phase);
    auto& b    = *env.at("batch").data<BatchData*>()[0];
    auto& copy = *env.at("copy").data<BatchCopy*>()[0];

    if (d.output_logprobs) {
        copy(d.sampled_logprobs, b.bsz * kMaxLogProb, sampled_logprobs_buf_);
        copy(d.sampled_indices, b.bsz * kMaxLogProb, sampled_indices_buf_);
        copy(d.sampled_nums, b.bsz, sampled_nums_buf_);
    }
}

void Sampling::Update(int phase, TensorMap& env)
{
    auto& d = *data_.at(phase);
    auto& b = *env.at("batch").data<BatchData*>()[0];

    if (d.output_logprobs) {
        float* logprob_buf = sampled_logprobs_buf_.data();
        int*   indices_buf = sampled_indices_buf_.data();
        int*   n_buf       = sampled_nums_buf_.data();
        for (int i = 0; i < b.rc.size(); ++i) {
            if (auto& x = *b.rc[i]; x.gen_cfg.output_logprobs) {
                // output buffers
                auto logprob_out = x.req->outputs.at("logprob_vals").data<float>();
                auto indices_out = x.req->outputs.at("logprob_indexes").data<int>();
                auto n_out       = x.req->outputs.at("logprob_nums").data<int>();
                // offset into output buffers
                const int offset = x.seq_len - x.prompt_len;
                std::copy_n(logprob_buf + i * kMaxLogProb, n_buf[i], logprob_out + offset * kMaxLogProb);
                std::copy_n(indices_buf + i * kMaxLogProb, n_buf[i], indices_out + offset * kMaxLogProb);
                n_out[offset] = n_buf[i];
            }
        }
    }
}

}  // namespace turbomind
