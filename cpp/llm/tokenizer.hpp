#pragma once

#include "../core/status.hpp"
#include "gguf.hpp"

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>

namespace prime::llm {

// Token ID type
using TokenId = int32_t;

// Vocabulary entry
struct VocabEntry {
    std::string piece;  // byte-string or BPE piece
    float       score;  // merge priority (lower = higher priority in BPE)
    int32_t     type;   // 0=normal, 1=unknown, 2=control, 3=user-defined, 6=byte
};

// Tokenizer: wraps vocabulary loaded from GGUF and provides encode/decode.
// Phase 2 implements SPM-byte-fallback BPE (used by LLaMA/Mistral).
class Tokenizer {
public:
    Tokenizer() = default;

    // Load vocabulary from an open GGUF file.
    prime::Status load(const GGUFFile& gguf);

    // Encode text → token IDs.
    // add_bos: prepend BOS token if set.
    std::vector<TokenId> encode(const std::string& text, bool add_bos = true) const;

    // Decode token IDs → text.
    std::string decode(const std::vector<TokenId>& ids,
                       bool skip_special = true) const;

    // Decode a single token piece (for streaming).
    std::string decode_token(TokenId id) const;

    // Vocabulary info
    uint32_t vocab_size() const { return static_cast<uint32_t>(vocab_.size()); }
    TokenId  bos_id()     const { return bos_id_; }
    TokenId  eos_id()     const { return eos_id_; }
    TokenId  unk_id()     const { return unk_id_; }

    bool is_special(TokenId id) const {
        if (id < 0 || id >= static_cast<TokenId>(vocab_.size())) return false;
        return vocab_[id].type == 2 || vocab_[id].type == 3;  // control or user
    }

    bool loaded() const { return !vocab_.empty(); }

private:
    // Byte-pair encoding helpers
    void init_byte_pieces();
    std::vector<TokenId> bpe_encode(const std::string& word) const;

    // Greedy BPE: encode a single UTF-8 word segment.
    struct BPEMerge {
        float    score;
        TokenId  id;
    };

    std::vector<VocabEntry>             vocab_;
    std::unordered_map<std::string, TokenId> token_to_id_;
    std::string byte_pieces_[256];  // single-byte fallback pieces

    TokenId bos_id_ = 1;
    TokenId eos_id_ = 2;
    TokenId unk_id_ = 0;
};

} // namespace prime::llm
