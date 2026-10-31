#include "transformer.hpp"
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace prime::llm {

// ---------------------------------------------------------------------------
// Weight loading helpers
// ---------------------------------------------------------------------------
prime::Status Transformer::load_tensor_vec(GGUFFile& gguf,
                                             const std::string& name,
                                             std::vector<float>& dst) {
    Tensor t;
    auto s = gguf.load_tensor(name, t);
    if (!s.ok()) return s;
    dst.assign(t.f32(), t.f32() + t.numel());
    return prime::Status::ok_status();
}

prime::Status Transformer::load(GGUFFile& gguf, const ModelConfig& cfg) {
    cfg_ = cfg;
    auto& w = weights_;

    // Token embeddings
    auto s = load_tensor_vec(gguf, "token_embd.weight", w.token_embedding);
    if (!s.ok()) return s;

    // Per-layer weights
    w.layers.resize(cfg_.num_layers);
    for (uint32_t l = 0; l < cfg_.num_layers; ++l) {
        std::string prefix = "blk." + std::to_string(l) + ".";
        auto& lw = w.layers[l];

        s = load_tensor_vec(gguf, prefix + "attn_norm.weight",  lw.attn_norm);  if (!s.ok()) return s;
        s = load_tensor_vec(gguf, prefix + "attn_q.weight",     lw.wq);         if (!s.ok()) return s;
        s = load_tensor_vec(gguf, prefix + "attn_k.weight",     lw.wk);         if (!s.ok()) return s;
        s = load_tensor_vec(gguf, prefix + "attn_v.weight",     lw.wv);         if (!s.ok()) return s;
        s = load_tensor_vec(gguf, prefix + "attn_output.weight",lw.wo);         if (!s.ok()) return s;
        s = load_tensor_vec(gguf, prefix + "ffn_norm.weight",   lw.ffn_norm);   if (!s.ok()) return s;
        s = load_tensor_vec(gguf, prefix + "ffn_gate.weight",   lw.gate_proj);  if (!s.ok()) return s;
        s = load_tensor_vec(gguf, prefix + "ffn_up.weight",     lw.up_proj);    if (!s.ok()) return s;
        s = load_tensor_vec(gguf, prefix + "ffn_down.weight",   lw.down_proj);  if (!s.ok()) return s;
    }

    // Output norm
    s = load_tensor_vec(gguf, "output_norm.weight", w.output_norm);
    if (!s.ok()) return s;

    // LM head (may be tied to token_embedding)
    Tensor lm_head_t;
    auto lm_s = gguf.load_tensor("output.weight", lm_head_t);
    if (lm_s.ok()) {
        w.lm_head.assign(lm_head_t.f32(), lm_head_t.f32() + lm_head_t.numel());
        w.lm_head_tied = false;
    } else {
        // Tied: share token embedding weights for LM head.
        w.lm_head_tied = true;
    }

    // Allocate scratch buffers.
    uint32_t D  = cfg_.embed_dim;
    uint32_t H  = cfg_.num_heads;
    uint32_t Hk = cfg_.num_kv_heads;
    uint32_t dh = cfg_.head_dim();
    uint32_t F  = cfg_.ffn_hidden_dim;

    buf_x_.resize(D);
    buf_residual_.resize(D);
    buf_normed_.resize(D);
    buf_q_.resize(H * dh);
    buf_k_.resize(Hk * dh);
    buf_v_.resize(Hk * dh);
    buf_attn_out_.resize(D);
    buf_ffn_gate_.resize(F);
    buf_ffn_up_.resize(F);
    // scores: heads × max_context; allocate lazily up to context_length
    buf_scores_.resize(H * cfg_.context_length);

    loaded_ = true;
    return prime::Status::ok_status();
}

// ---------------------------------------------------------------------------
// RoPE (Rotary Position Embedding)
// ---------------------------------------------------------------------------
void Transformer::apply_rope(float* q, float* k,
                              uint32_t num_q_heads, uint32_t num_kv_heads,
                              uint32_t head_dim, uint32_t pos) const {
    float theta = cfg_.rope_theta;
    for (uint32_t h = 0; h < num_q_heads; ++h) {
        float* qh = q + h * head_dim;
        for (uint32_t i = 0; i < head_dim / 2; ++i) {
            float freq = 1.0f / std::pow(theta, (2.0f * i) / head_dim);
            float val  = pos * freq;
            float cos_v = std::cos(val);
            float sin_v = std::sin(val);
            float q0 = qh[2*i], q1 = qh[2*i+1];
            qh[2*i]   = q0 * cos_v - q1 * sin_v;
            qh[2*i+1] = q0 * sin_v + q1 * cos_v;
        }
    }
    for (uint32_t h = 0; h < num_kv_heads; ++h) {
        float* kh = k + h * head_dim;
        for (uint32_t i = 0; i < head_dim / 2; ++i) {
            float freq = 1.0f / std::pow(theta, (2.0f * i) / head_dim);
            float val  = pos * freq;
            float cos_v = std::cos(val);
            float sin_v = std::sin(val);
            float k0 = kh[2*i], k1 = kh[2*i+1];
            kh[2*i]   = k0 * cos_v - k1 * sin_v;
            kh[2*i+1] = k0 * sin_v + k1 * cos_v;
        }
    }
}

// ---------------------------------------------------------------------------
// Attention layer
// ---------------------------------------------------------------------------
void Transformer::attention_layer(const TransformerLayerWeights& lw,
                                   uint32_t layer_idx,
                                   const float* x, float* out,
                                   uint32_t pos, KVCache& kv) const {
    uint32_t D  = cfg_.embed_dim;
    uint32_t H  = cfg_.num_heads;
    uint32_t Hk = cfg_.num_kv_heads;
    uint32_t dh = cfg_.head_dim();
    uint32_t seq = kv.seq_len();  // number of past tokens (before this one)

    // Q K V projections: [H*dh] = W[H*dh, D] × x[D]
    matmul_f32(lw.wq.data(), x, buf_q_.data(), H * dh, D);
    matmul_f32(lw.wk.data(), x, buf_k_.data(), Hk * dh, D);
    matmul_f32(lw.wv.data(), x, buf_v_.data(), Hk * dh, D);

    // Apply RoPE to Q and K.
    apply_rope(buf_q_.data(), buf_k_.data(), H, Hk, dh, pos);

    // Append current K and V to KV cache.
    kv.append(layer_idx, buf_k_.data(), buf_v_.data());
    // Note: seq_len_ is advanced by the caller (LLMEngine) after all layers.

    uint32_t total_seq = seq + 1;  // past + current

    // GQA: map query heads to kv heads.
    uint32_t kv_repeat = H / Hk;  // how many Q heads share one KV head

    // Compute attention scores for each Q head.
    float scale = 1.0f / std::sqrt(static_cast<float>(dh));
    std::fill(buf_attn_out_.begin(), buf_attn_out_.end(), 0.0f);

    const float* kc = kv.k(layer_idx);  // [total_seq * Hk * dh]
    const float* vc = kv.v(layer_idx);

    for (uint32_t h = 0; h < H; ++h) {
        uint32_t kv_h = h / kv_repeat;
        const float* q_head = buf_q_.data() + h * dh;

        // Compute attention scores: score[t] = dot(q, k_t) * scale
        // Clamp to [-30, 30] before softmax to prevent exp() overflow with
        // untrained / randomly initialised weights.
        float* scores = buf_scores_.data() + h * cfg_.context_length;
        for (uint32_t t = 0; t < total_seq; ++t) {
            const float* k_head = kc + t * Hk * dh + kv_h * dh;
            float s = vec_dot_f32(q_head, k_head, dh) * scale;
            scores[t] = (s > 30.0f) ? 30.0f : (s < -30.0f) ? -30.0f : s;
        }

        // Softmax over sequence positions.
        softmax_inplace(scores, total_seq);

        // Weighted sum of value vectors.
        float* out_head = buf_attn_out_.data() + h * dh;
        std::fill(out_head, out_head + dh, 0.0f);
        for (uint32_t t = 0; t < total_seq; ++t) {
            const float* v_head = vc + t * Hk * dh + kv_h * dh;
            for (uint32_t i = 0; i < dh; ++i) {
                out_head[i] += scores[t] * v_head[i];
            }
        }
    }

    // Output projection: [D] = Wo[D, H*dh] × attn_out[H*dh]
    matmul_f32(lw.wo.data(), buf_attn_out_.data(), out, D, H * dh);
}

// ---------------------------------------------------------------------------
// FFN layer (SwiGLU)
// ---------------------------------------------------------------------------
void Transformer::ffn_layer(const TransformerLayerWeights& lw,
                              const float* x, float* out) const {
    uint32_t D = cfg_.embed_dim;
    uint32_t F = cfg_.ffn_hidden_dim;

    // gate = silu(gate_proj × x), up = up_proj × x
    matmul_f32(lw.gate_proj.data(), x, buf_ffn_gate_.data(), F, D);
    matmul_f32(lw.up_proj.data(),  x, buf_ffn_up_.data(),   F, D);

    for (uint32_t i = 0; i < F; ++i) {
        buf_ffn_gate_[i] = silu(buf_ffn_gate_[i]) * buf_ffn_up_[i];
    }

    // out = down_proj × (gate * up)
    matmul_f32(lw.down_proj.data(), buf_ffn_gate_.data(), out, D, F);
}

// ---------------------------------------------------------------------------
// Forward pass: single token
// ---------------------------------------------------------------------------
prime::Status Transformer::forward(int32_t token_id, uint32_t pos,
                                    KVCache& kv, float* logits_out) {
    if (!loaded_) {
        return prime::Status::error(prime::StatusCode::ERROR_NOT_INITIALIZED,
                                    "Transformer not loaded");
    }
    if (token_id < 0 || static_cast<uint32_t>(token_id) >= cfg_.vocab_size) {
        return prime::Status::error(prime::StatusCode::ERROR_INVALID_CONFIG,
                                    "token_id out of range: " + std::to_string(token_id));
    }
    if (kv.is_full()) {
        return prime::Status::error(prime::StatusCode::ERROR_INVALID_CONFIG,
                                    "Context length exceeded");
    }

    uint32_t D = cfg_.embed_dim;

    // Token embedding lookup.
    const float* emb = weights_.token_embedding.data() +
                       static_cast<std::size_t>(token_id) * D;
    vec_copy(buf_x_.data(), emb, D);

    // Process each transformer layer.
    for (uint32_t l = 0; l < cfg_.num_layers; ++l) {
        const auto& lw = weights_.layers[l];

        // Attention sub-layer (with pre-norm + residual).
        rms_norm(buf_x_.data(), lw.attn_norm.data(), buf_normed_.data(), D);
        attention_layer(lw, l, buf_normed_.data(), buf_residual_.data(), pos, kv);
        vec_add_inplace(buf_x_.data(), buf_residual_.data(), D);

        // FFN sub-layer (with pre-norm + residual).
        rms_norm(buf_x_.data(), lw.ffn_norm.data(), buf_normed_.data(), D);
        ffn_layer(lw, buf_normed_.data(), buf_residual_.data());
        vec_add_inplace(buf_x_.data(), buf_residual_.data(), D);
    }

    // Final norm.
    rms_norm(buf_x_.data(), weights_.output_norm.data(), buf_normed_.data(), D);

    // LM head projection → logits[vocab_size].
    const float* lm_w = weights_.lm_head_tied
        ? weights_.token_embedding.data()
        : weights_.lm_head.data();
    matmul_f32(lm_w, buf_normed_.data(), logits_out, cfg_.vocab_size, D);

    return prime::Status::ok_status();
}

} // namespace prime::llm
