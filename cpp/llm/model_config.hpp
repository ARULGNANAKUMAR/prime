#pragma once

#include "tensor.hpp"
#include <string>
#include <cstddef>
#include <cstdint>

namespace prime::llm {

// ModelArchitecture identifies the transformer variant.
// Phase 2 supports LlamaStyle (GPT-2/LLaMA decoder-only transformer).
// Additional architectures are added in future phases by extending this enum
// and adding a corresponding backend in transformer.cpp.
enum class ModelArchitecture {
    UNKNOWN,
    LLAMA,   // LLaMA / Mistral / GPT-NeoX style decoder-only transformer
};

inline std::string arch_name(ModelArchitecture a) {
    switch (a) {
        case ModelArchitecture::LLAMA: return "llama";
        default:                       return "unknown";
    }
}

// ModelConfig holds the hyperparameters extracted from the model file.
// These are the fields Phase 2 actually uses; unused GGUF metadata is discarded.
struct ModelConfig {
    ModelArchitecture arch = ModelArchitecture::UNKNOWN;

    std::string model_name;     // from general.name if present

    uint32_t vocab_size     = 0;
    uint32_t context_length = 0;  // max sequence length the model was trained on
    uint32_t embed_dim      = 0;  // d_model / hidden_size
    uint32_t num_layers     = 0;
    uint32_t num_heads      = 0;  // query heads
    uint32_t num_kv_heads   = 0;  // GQA kv heads (= num_heads if not GQA)
    uint32_t ffn_hidden_dim = 0;  // intermediate size in FFN

    float    rope_theta     = 10000.0f;

    DType    weight_dtype   = DType::F32;

    // BOS / EOS token IDs (may be absent in some models)
    int32_t  bos_token_id   = 1;
    int32_t  eos_token_id   = 2;

    // Derived helpers
    uint32_t head_dim() const { return (num_heads > 0) ? embed_dim / num_heads : 0; }
    uint32_t kv_head_dim() const { return head_dim(); }
};

} // namespace prime::llm
