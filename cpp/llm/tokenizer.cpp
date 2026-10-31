#include "tokenizer.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <cassert>
#include <cstring>

namespace prime::llm {

// ---------------------------------------------------------------------------
// Load vocabulary from GGUF
// ---------------------------------------------------------------------------
prime::Status Tokenizer::load(const GGUFFile& gguf) {
    const auto* tokens_val = gguf.get_meta("tokenizer.ggml.tokens");
    if (!tokens_val || tokens_val->type != GGUFValueType::ARRAY) {
        return prime::Status::error(prime::StatusCode::ERROR_INIT_FAILED,
                                    "Tokenizer: missing tokenizer.ggml.tokens");
    }
    const auto& tokens_arr = tokens_val->array;
    std::size_t n = tokens_arr.size();
    vocab_.resize(n);

    // Scores (optional)
    const auto* scores_val = gguf.get_meta("tokenizer.ggml.scores");

    // Token types (optional)
    const auto* types_val = gguf.get_meta("tokenizer.ggml.token_type");

    // Helper: visit a GGUFScalar as float (skipping non-numeric types).
    auto scalar_to_float = [](const GGUFScalar& s) -> float {
        return std::visit([](auto&& v) -> float {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_arithmetic_v<T> && !std::is_same_v<T, bool>)
                return static_cast<float>(v);
            return 0.0f;
        }, s);
    };
    auto scalar_to_i32 = [](const GGUFScalar& s) -> int32_t {
        return std::visit([](auto&& v) -> int32_t {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_arithmetic_v<T> && !std::is_same_v<T, bool>)
                return static_cast<int32_t>(v);
            return 0;
        }, s);
    };

    for (std::size_t i = 0; i < n; ++i) {
        auto& entry = vocab_[i];
        if (std::holds_alternative<std::string>(tokens_arr[i])) {
            entry.piece = std::get<std::string>(tokens_arr[i]);
        } else {
            entry.piece = "";
        }
        entry.score = (scores_val && i < scores_val->array.size())
            ? scalar_to_float(scores_val->array[i])
            : 0.0f;
        entry.type = (types_val && i < types_val->array.size())
            ? scalar_to_i32(types_val->array[i])
            : 0;

        token_to_id_[entry.piece] = static_cast<TokenId>(i);
    }

    bos_id_ = gguf.meta_i32("tokenizer.ggml.bos_token_id", 1);
    eos_id_ = gguf.meta_i32("tokenizer.ggml.eos_token_id", 2);
    unk_id_ = gguf.meta_i32("tokenizer.ggml.unknown_token_id", 0);

    init_byte_pieces();
    return prime::Status::ok_status();
}

void Tokenizer::init_byte_pieces() {
    // Build single-byte fallback pieces: <0x00> through <0xFF>
    for (int i = 0; i < 256; ++i) {
        char buf[8];
        snprintf(buf, sizeof(buf), "<0x%02X>", i);
        byte_pieces_[i] = std::string(buf);
    }
}

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------
// Greedy BPE merge: given a sequence of symbols (initially individual chars),
// repeatedly find the best merge pair and apply it.
std::vector<TokenId> Tokenizer::bpe_encode(const std::string& word) const {
    if (word.empty()) return {};

    // Start: each UTF-8 byte as a separate symbol, looked up by byte piece.
    struct Symbol {
        std::string text;
        TokenId     id;
    };
    std::vector<Symbol> syms;
    syms.reserve(word.size());
    for (unsigned char c : word) {
        std::string piece(1, static_cast<char>(c));
        // Try to find the exact single-char piece in vocab.
        auto it = token_to_id_.find(piece);
        if (it != token_to_id_.end()) {
            syms.push_back({piece, it->second});
        } else {
            // Fall back to <0xNN>
            syms.push_back({byte_pieces_[c], -1});
        }
    }

    // Also try to look up the full word first (common for single tokens).
    {
        auto it = token_to_id_.find(word);
        if (it != token_to_id_.end()) {
            return {it->second};
        }
    }

    // BPE merge loop
    bool changed = true;
    while (changed && syms.size() > 1) {
        changed = false;
        float best_score = std::numeric_limits<float>::lowest();
        std::size_t best_i = 0;
        TokenId best_id = -1;

        for (std::size_t i = 0; i + 1 < syms.size(); ++i) {
            std::string merged = syms[i].text + syms[i+1].text;
            auto it = token_to_id_.find(merged);
            if (it != token_to_id_.end()) {
                TokenId id = it->second;
                float score = vocab_[static_cast<std::size_t>(id)].score;
                if (score > best_score) {
                    best_score = score;
                    best_i     = i;
                    best_id    = id;
                }
            }
        }

        if (best_id >= 0) {
            // Apply the merge
            syms[best_i].text += syms[best_i + 1].text;
            syms[best_i].id    = best_id;
            syms.erase(syms.begin() + static_cast<std::ptrdiff_t>(best_i) + 1);
            changed = true;
        }
    }

    // Collect token IDs; for un-merged byte pieces look them up again.
    std::vector<TokenId> ids;
    ids.reserve(syms.size());
    for (auto& sym : syms) {
        if (sym.id >= 0) {
            ids.push_back(sym.id);
        } else {
            // Unknown piece: emit UNK or byte tokens
            auto it = token_to_id_.find(sym.text);
            if (it != token_to_id_.end()) {
                ids.push_back(it->second);
            } else {
                ids.push_back(unk_id_);
            }
        }
    }
    return ids;
}

std::vector<TokenId> Tokenizer::encode(const std::string& text, bool add_bos) const {
    std::vector<TokenId> ids;
    if (add_bos && bos_id_ >= 0) {
        ids.push_back(bos_id_);
    }

    if (text.empty()) return ids;

    // Simple whitespace-aware pre-tokenisation (SentencePiece style):
    // prepend a space to the first word, then split on spaces keeping the space
    // attached to the following word.
    std::string processed;
    processed.reserve(text.size() + 1);
    bool at_start = true;
    for (std::size_t i = 0; i < text.size(); ) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        if (c == ' ') {
            // Start of a new word segment; flush current, start new with space.
            if (!processed.empty()) {
                auto word_ids = bpe_encode(processed);
                ids.insert(ids.end(), word_ids.begin(), word_ids.end());
                processed.clear();
            }
            // Attach the space to the next word (SentencePiece convention: '▁').
            processed += '\xe2'; processed += '\x96'; processed += '\x81'; // U+2581 ▁
            ++i;
            at_start = false;
        } else {
            if (at_start) {
                // Prepend leading space marker for very first word.
                processed += '\xe2'; processed += '\x96'; processed += '\x81';
                at_start = false;
            }
            // Determine UTF-8 code-point length.
            int bytes = 1;
            if      ((c & 0xE0) == 0xC0) bytes = 2;
            else if ((c & 0xF0) == 0xE0) bytes = 3;
            else if ((c & 0xF8) == 0xF0) bytes = 4;
            for (int b = 0; b < bytes && i < text.size(); ++b, ++i) {
                processed += text[i];
            }
        }
    }
    if (!processed.empty()) {
        auto word_ids = bpe_encode(processed);
        ids.insert(ids.end(), word_ids.begin(), word_ids.end());
    }

    return ids;
}

// ---------------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------------
std::string Tokenizer::decode_token(TokenId id) const {
    if (id < 0 || id >= static_cast<TokenId>(vocab_.size())) return "";
    const auto& piece = vocab_[id].piece;

    // Replace SentencePiece space marker ▁ (U+2581, 3 bytes) with regular space.
    std::string result;
    result.reserve(piece.size());
    for (std::size_t i = 0; i < piece.size(); ) {
        if (i + 2 < piece.size() &&
            (unsigned char)piece[i]   == 0xE2 &&
            (unsigned char)piece[i+1] == 0x96 &&
            (unsigned char)piece[i+2] == 0x81) {
            result += ' ';
            i += 3;
        } else {
            result += piece[i++];
        }
    }

    // Resolve byte tokens <0xNN> → actual byte.
    if (result.size() == 6 && result[0] == '<' && result[1] == '0' &&
        result[2] == 'x' && result[5] == '>') {
        char hex[3] = { result[3], result[4], '\0' };
        unsigned int byte_val = 0;
        if (sscanf(hex, "%x", &byte_val) == 1) {
            return std::string(1, static_cast<char>(byte_val));
        }
    }

    return result;
}

std::string Tokenizer::decode(const std::vector<TokenId>& ids,
                               bool skip_special) const {
    std::string out;
    out.reserve(ids.size() * 4);
    for (TokenId id : ids) {
        if (skip_special && is_special(id)) continue;
        out += decode_token(id);
    }
    // Trim a leading space if present (BOS artefact).
    if (!out.empty() && out[0] == ' ') out = out.substr(1);
    return out;
}

} // namespace prime::llm
