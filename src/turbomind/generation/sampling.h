
#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/generation/base_param.h"

namespace turbomind {

struct SamplingData;

class Sampling: public BaseGenerationParam {
public:
    explicit Sampling(const BaseGenerationParam& base, int phases);

    void Setup(int phase, TensorMap& env);

    void Forward(int phase, TensorMap& env);

    void Fetch(int phase, TensorMap& env);

    void Update(int phase, TensorMap& env);

private:
    std::vector<std::shared_ptr<SamplingData>> data_;

    // host buffer
    Buffer_<int>   kept_;
    Buffer_<int>   top_k_;
    Buffer_<float> top_p_;
    Buffer_<float> min_p_;
    Buffer_<int>   token_decision_infer_type_;
    Buffer_<int>   token_decision_valid_id_;
    Buffer_<int>   token_decision_invalid_id_;
    Buffer_<int>   token_decision_end_id_;
    Buffer_<float> token_decision_certainty_threshold_;
    Buffer_<float> token_decision_completion_threshold_;
    Buffer_<float> token_decision_invalid_bias_;

    Buffer_<float> sampled_logprobs_buf_;
    Buffer_<int>   sampled_indices_buf_;
    Buffer_<int>   sampled_nums_buf_;
};

}  // namespace turbomind
