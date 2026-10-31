#pragma once

#include "model_config.hpp"
#include "tensor.hpp"
#include "kv_cache.hpp"
#include "gguf.hpp"
#include "../core/status.hpp"

#include <vector>
#include <string>
#include <unordered_map>
#include <memory>

namespace prime::llm {

// Weights for one transformer layer (LLaMA-style).
struct TransformerLayerWeights {
    // Attention
    std::vector<float> attn_norm;    // [embed_dim] RMSNorm weights
    std::vector<float> wq;           // [num_heads * head_dim, embed_dim]
    std::vector<float> wk;           // [num_kv_heads * head_dim, embed_dim]
    std::vector<float> wv;           // [num_kv_heads * head_dim, embed_dim]
    std::vector<float> wo;           // [embed_dim, num_heads * head_dim]

    // FFN (SwiGLU: gate + up + down)
    std::vector<float> ffn_norm;     // [embed_dim]
    std::vector<float> gate_proj;    // [ffn_hidden_dim, embed_dim]
    std::vector<float> up_proj;      // [ffn_hidden_dim, embed_dim]
    std::vector<float> down_proj;    // [embed_dim, ffn_hidden_dim]
};

// Full transformer model weights.
struct TransformerWeights {
    // Token embedding table [vocab_size, embed_dim]
    std::vector<float> token_embedding;

    // Per-layer weights
    std::vector<TransformerLayerWeights> layers;

    // Final norm + LM head
    std::vector<float> output_norm;  // [embed_dim]
    std::vector<float> lm_head;      // [vocab_size, embed_dim] (may alias token_embedding)
    bool lm_head_tied = false;       // true if lm_head == token_embedding
};

// Transformer: loads weights and runs the forward pass.
class Transformer {
public:
    Transformer() = default;

    // Load all weights from an open GGUF file.
    prime::Status load(GGUFFile& gguf, const ModelConfig& cfg);

    // Forward pass: given one token at position pos, update KV cache,
    // and write logits[vocab_size] to the provided buffer.
    // pos: current generation position (used for RoPE).
    prime::Status forward(int32_t token_id, uint32_t pos,
                          KVCache& kv, float* logits_out);

    const ModelConfig& config() const { return cfg_; }
    bool loaded() const { return loaded_; }

private:
    // Helpers
    void apply_rope(float* q, float* k,
                    uint32_t num_q_heads, uint32_t num_kv_heads,
                    uint32_t head_dim, uint32_t pos) const;

    void attention_layer(const TransformerLayerWeights& lw,
                         uint32_t layer_idx,
                         const float* x, float* out,
                         uint32_t pos, KVCache& kv) const;

    void ffn_layer(const TransformerLayerWeights& lw,
                   const float* x, float* out) const;

    prime::Status load_tensor_vec(GGUFFile& gguf,
                                   const std::string& name,
                                   std::vector<float>& dst);

    ModelConfig cfg_;
    TransformerWeights weights_;
    bool loaded_ = false;

    // Scratch buffers (reused across forward calls to avoid allocation).
    mutable std::vector<float> buf_x_;
    mutable std::vector<float> buf_residual_;
    mutable std::vector<float> buf_normed_;
    mutable std::vector<float> buf_q_;
    mutable std::vector<float> buf_k_;
    mutable std::vector<float> buf_v_;
    mutable std::vector<float> buf_attn_out_;
    mutable std::vector<float> buf_ffn_gate_;
    mutable std::vector<float> buf_ffn_up_;
    mutable std::vector<float> buf_scores_;
};

} // namespace prime::llm
