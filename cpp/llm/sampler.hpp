#pragma once

#include "tensor.hpp"
#include <cstdint>
#include <random>
#include <vector>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cmath>

namespace prime::llm {

struct SamplingConfig {
    float    temperature  = 1.0f;   // 0 = greedy
    int      top_k        = 0;      // 0 = disabled
    float    top_p        = 1.0f;   // 1.0 = disabled
    uint64_t seed         = 0;      // 0 = random seed each call
    bool     seed_set     = false;
};

class Sampler {
public:
    explicit Sampler(const SamplingConfig& cfg = {}) : cfg_(cfg) {
        if (cfg_.seed_set && cfg_.seed != 0) {
            rng_.seed(cfg_.seed);
        } else {
            std::random_device rd;
            rng_.seed(rd());
        }
    }

    // Sample next token from logits[vocab_size].
    // Modifies logits in-place (applies temperature + softmax).
    int32_t sample(float* logits, uint32_t vocab_size) {
        if (vocab_size == 0) throw std::runtime_error("Sampler: vocab_size is 0");

        // Greedy: just take argmax.
        if (cfg_.temperature <= 0.0f) {
            return static_cast<int32_t>(
                std::max_element(logits, logits + vocab_size) - logits);
        }

        // Apply temperature.
        float inv_temp = 1.0f / cfg_.temperature;
        for (uint32_t i = 0; i < vocab_size; ++i) logits[i] *= inv_temp;

        // Softmax.
        softmax_inplace(logits, vocab_size);

        // Top-k filtering.
        std::vector<uint32_t> indices(vocab_size);
        std::iota(indices.begin(), indices.end(), 0u);

        if (cfg_.top_k > 0 && cfg_.top_k < static_cast<int>(vocab_size)) {
            std::partial_sort(indices.begin(),
                              indices.begin() + cfg_.top_k,
                              indices.end(),
                              [&](uint32_t a, uint32_t b){ return logits[a] > logits[b]; });
            // Zero out tokens outside top-k.
            for (uint32_t i = static_cast<uint32_t>(cfg_.top_k); i < vocab_size; ++i) {
                logits[indices[i]] = 0.0f;
            }
            indices.resize(static_cast<std::size_t>(cfg_.top_k));
        } else {
            std::sort(indices.begin(), indices.end(),
                      [&](uint32_t a, uint32_t b){ return logits[a] > logits[b]; });
        }

        // Top-p (nucleus) filtering.
        if (cfg_.top_p < 1.0f) {
            float cumulative = 0.0f;
            std::size_t cutoff = indices.size();
            for (std::size_t i = 0; i < indices.size(); ++i) {
                cumulative += logits[indices[i]];
                if (cumulative >= cfg_.top_p) {
                    cutoff = i + 1;
                    break;
                }
            }
            for (std::size_t i = cutoff; i < indices.size(); ++i) {
                logits[indices[i]] = 0.0f;
            }
            indices.resize(cutoff);
        }

        // Re-normalise.
        float sum = 0.0f;
        for (auto idx : indices) sum += logits[idx];
        if (sum <= 0.0f) return static_cast<int32_t>(indices[0]);
        float inv_sum = 1.0f / sum;
        for (auto idx : indices) logits[idx] *= inv_sum;

        // Multinomial sample.
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float r = dist(rng_);
        float cdf = 0.0f;
        for (auto idx : indices) {
            cdf += logits[idx];
            if (r < cdf) return static_cast<int32_t>(idx);
        }
        return static_cast<int32_t>(indices.back());
    }

private:
    SamplingConfig  cfg_;
    std::mt19937_64 rng_;
};

} // namespace prime::llm
