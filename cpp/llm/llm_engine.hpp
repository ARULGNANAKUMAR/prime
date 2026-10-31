#pragma once

#include "gguf.hpp"
#include "tokenizer.hpp"
#include "transformer.hpp"
#include "sampler.hpp"
#include "kv_cache.hpp"
#include "../core/status.hpp"

#include <string>
#include <vector>
#include <functional>
#include <cstdint>

namespace prime::llm {

// GenerationParams controls a single generate() call.
struct GenerationParams {
    int      max_new_tokens = 128;
    float    temperature    = 1.0f;
    int      top_k          = 0;
    float    top_p          = 1.0f;
    uint64_t seed           = 0;
    bool     seed_set       = false;
    std::vector<int32_t> stop_token_ids;  // defaults to {eos_id}
};

// StreamCallback: called with each decoded token piece during streaming.
// Return false to stop generation early.
using StreamCallback = std::function<bool(const std::string& piece)>;

// LLMEngine: owns the loaded model and provides generate() and stream_generate().
// One instance per loaded model file.
class LLMEngine {
public:
    LLMEngine() = default;

    // Load model from a GGUF file.
    prime::Status load(const std::string& model_path);

    // Generate text given a prompt. Returns the full generated string.
    prime::Status generate(const std::string& prompt,
                            const GenerationParams& params,
                            std::string& output);

    // Streaming variant: calls callback for each token piece.
    prime::Status stream_generate(const std::string& prompt,
                                   const GenerationParams& params,
                                   StreamCallback callback);

    // Metadata
    const ModelConfig& model_config() const { return transformer_.config(); }
    bool loaded() const { return loaded_; }
    const std::string& model_path() const { return model_path_; }
    uint32_t vocab_size() const { return tokenizer_.vocab_size(); }
    uint32_t context_length() const { return model_config().context_length; }

private:
    prime::Status run_generation(const std::string& prompt,
                                  const GenerationParams& params,
                                  StreamCallback callback,
                                  std::string* full_output);

    GGUFFile     gguf_;
    Tokenizer    tokenizer_;
    Transformer  transformer_;
    KVCache      kv_cache_;
    std::string  model_path_;
    bool         loaded_ = false;

    // Logit scratch buffer (reused across calls).
    std::vector<float> logits_;
};

} // namespace prime::llm
