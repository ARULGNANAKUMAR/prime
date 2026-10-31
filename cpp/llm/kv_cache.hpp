#pragma once

#include "tensor.hpp"
#include <vector>
#include <cstdint>
#include <stdexcept>

namespace prime::llm {

// KVCache stores key/value vectors for each layer to enable efficient
// autoregressive generation (avoid recomputing past tokens each step).
//
// Layout: per layer, per head, sequence of head_dim vectors.
// Storage: flat F32 arrays [num_layers][max_seq][num_kv_heads][head_dim]
class KVCache {
public:
    KVCache() = default;

    void init(uint32_t num_layers, uint32_t num_kv_heads,
              uint32_t head_dim, uint32_t max_seq) {
        num_layers_   = num_layers;
        num_kv_heads_ = num_kv_heads;
        head_dim_     = head_dim;
        max_seq_      = max_seq;
        seq_len_      = 0;

        std::size_t layer_elems = static_cast<std::size_t>(max_seq) * num_kv_heads * head_dim;
        k_cache_.assign(num_layers, std::vector<float>(layer_elems, 0.0f));
        v_cache_.assign(num_layers, std::vector<float>(layer_elems, 0.0f));
    }

    void reset() { seq_len_ = 0; }

    uint32_t seq_len()  const { return seq_len_; }
    uint32_t max_seq()  const { return max_seq_; }
    bool     is_full()  const { return seq_len_ >= max_seq_; }

    // Append a new position's keys and values for a layer.
    // k_vec / v_vec: [num_kv_heads * head_dim] float arrays
    void append(uint32_t layer, const float* k_vec, const float* v_vec) {
        if (seq_len_ >= max_seq_) {
            throw std::runtime_error("KVCache: context exceeded max_seq");
        }
        std::size_t stride = num_kv_heads_ * head_dim_;
        std::size_t pos    = static_cast<std::size_t>(seq_len_) * stride;
        std::memcpy(k_cache_[layer].data() + pos, k_vec, stride * sizeof(float));
        std::memcpy(v_cache_[layer].data() + pos, v_vec, stride * sizeof(float));
    }

    // Advance position counter (called after appending all layers for one token).
    void advance() { ++seq_len_; }

    // Get pointer to K vectors for this layer [seq_len * kv_heads * head_dim].
    const float* k(uint32_t layer) const { return k_cache_[layer].data(); }
    const float* v(uint32_t layer) const { return v_cache_[layer].data(); }

    uint32_t num_kv_heads() const { return num_kv_heads_; }
    uint32_t head_dim()     const { return head_dim_; }

private:
    uint32_t num_layers_   = 0;
    uint32_t num_kv_heads_ = 0;
    uint32_t head_dim_     = 0;
    uint32_t max_seq_      = 0;
    uint32_t seq_len_      = 0;

    // [num_layers][max_seq * num_kv_heads * head_dim]
    std::vector<std::vector<float>> k_cache_;
    std::vector<std::vector<float>> v_cache_;
};

} // namespace prime::llm
