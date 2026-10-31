#include "llm_engine.hpp"
#include <algorithm>

namespace prime::llm {

prime::Status LLMEngine::load(const std::string& model_path) {
    model_path_ = model_path;

    // Open and parse GGUF.
    auto s = gguf_.open(model_path);
    if (!s.ok()) return s;

    // Build model config from metadata.
    ModelConfig cfg;
    s = gguf_.build_model_config(cfg);
    if (!s.ok()) return s;

    // Load tokenizer vocabulary from GGUF.
    s = tokenizer_.load(gguf_);
    if (!s.ok()) return s;

    // Sync vocab size from tokenizer if not in metadata.
    if (cfg.vocab_size == 0) {
        cfg.vocab_size = tokenizer_.vocab_size();
    }
    if (cfg.vocab_size == 0) {
        return prime::Status::error(prime::StatusCode::ERROR_INVALID_CONFIG,
                                    "LLMEngine: could not determine vocab size");
    }

    // Load transformer weights.
    s = transformer_.load(gguf_, cfg);
    if (!s.ok()) return s;

    // Initialise KV cache.
    kv_cache_.init(cfg.num_layers, cfg.num_kv_heads,
                   cfg.head_dim(), cfg.context_length);

    // Allocate logit buffer.
    logits_.resize(cfg.vocab_size);

    loaded_ = true;
    return prime::Status::ok_status();
}

prime::Status LLMEngine::generate(const std::string& prompt,
                                   const GenerationParams& params,
                                   std::string& output) {
    return run_generation(prompt, params, nullptr, &output);
}

prime::Status LLMEngine::stream_generate(const std::string& prompt,
                                          const GenerationParams& params,
                                          StreamCallback callback) {
    return run_generation(prompt, params, std::move(callback), nullptr);
}

prime::Status LLMEngine::run_generation(const std::string& prompt,
                                         const GenerationParams& params,
                                         StreamCallback callback,
                                         std::string* full_output) {
    if (!loaded_) {
        return prime::Status::error(prime::StatusCode::ERROR_NOT_INITIALIZED,
                                    "LLMEngine: model not loaded");
    }
    if (params.max_new_tokens <= 0) {
        return prime::Status::error(prime::StatusCode::ERROR_INVALID_CONFIG,
                                    "max_new_tokens must be > 0");
    }

    // Build stop token set.
    std::vector<int32_t> stop_ids = params.stop_token_ids;
    if (stop_ids.empty()) stop_ids.push_back(tokenizer_.eos_id());

    // Tokenise prompt.
    auto input_ids = tokenizer_.encode(prompt, /*add_bos=*/true);
    if (input_ids.empty()) {
        return prime::Status::error(prime::StatusCode::ERROR_INVALID_CONFIG,
                                    "Tokenizer produced empty token sequence");
    }

    uint32_t ctx_len = model_config().context_length;
    if (input_ids.size() >= ctx_len) {
        return prime::Status::error(prime::StatusCode::ERROR_INVALID_CONFIG,
                                    "Prompt is longer than model context length");
    }

    // Reset KV cache for this generation.
    kv_cache_.reset();

    // Set up sampler.
    SamplingConfig sc;
    sc.temperature = params.temperature;
    sc.top_k       = params.top_k;
    sc.top_p       = params.top_p;
    sc.seed        = params.seed;
    sc.seed_set    = params.seed_set;
    Sampler sampler(sc);

    if (full_output) full_output->clear();

    // Prefill: run all prompt tokens through the model.
    for (std::size_t i = 0; i < input_ids.size(); ++i) {
        auto s = transformer_.forward(input_ids[i],
                                       static_cast<uint32_t>(i),
                                       kv_cache_, logits_.data());
        if (!s.ok()) return s;
        kv_cache_.advance();
    }

    // Decode: generate up to max_new_tokens new tokens.
    uint32_t pos = static_cast<uint32_t>(input_ids.size());
    int32_t next_token = sampler.sample(logits_.data(),
                                         model_config().vocab_size);

    for (int step = 0; step < params.max_new_tokens; ++step) {
        // Check stop condition.
        bool is_stop = false;
        for (int32_t sid : stop_ids) {
            if (next_token == sid) { is_stop = true; break; }
        }
        if (is_stop) break;
        if (kv_cache_.is_full()) break;

        // Decode this token to text.
        std::string piece = tokenizer_.decode_token(next_token);

        if (callback) {
            if (!callback(piece)) break;  // caller requested stop
        }
        if (full_output) *full_output += piece;

        // Run forward pass for next step.
        auto s = transformer_.forward(next_token, pos, kv_cache_, logits_.data());
        if (!s.ok()) return s;
        kv_cache_.advance();
        ++pos;

        next_token = sampler.sample(logits_.data(), model_config().vocab_size);
    }

    return prime::Status::ok_status();
}

} // namespace prime::llm
