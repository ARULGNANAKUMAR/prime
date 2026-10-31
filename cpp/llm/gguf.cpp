#include "gguf.hpp"
#include <cstring>
#include <stdexcept>
#include <algorithm>

namespace prime::llm {

// GGUF tensor type → our DType + bytes-per-block
struct GGUFTypeInfo {
    DType    dtype;
    uint32_t block_size;    // number of elements per block
    uint32_t block_bytes;   // bytes per block
};

static const std::unordered_map<uint32_t, GGUFTypeInfo> GGUF_TYPE_MAP = {
    {0,  {DType::F32,  1,  4}},   // GGML_TYPE_F32
    {1,  {DType::F16,  1,  2}},   // GGML_TYPE_F16
    {2,  {DType::Q4_0, 32, 18}},  // GGML_TYPE_Q4_0
    {8,  {DType::Q8_0, 32, 34}},  // GGML_TYPE_Q8_0
};

// ---------------------------------------------------------------------------
// Low-level file reading helpers
// ---------------------------------------------------------------------------
template<typename T>
T GGUFFile::read_pod() {
    T val;
    if (!file_.read(reinterpret_cast<char*>(&val), sizeof(T))) {
        throw std::runtime_error("GGUF: unexpected end of file reading " +
                                 std::string(typeid(T).name()));
    }
    return val;
}

std::string GGUFFile::read_gguf_string() {
    uint64_t len = read_pod<uint64_t>();
    if (len > 1024 * 1024) {  // sanity cap: 1 MB string
        throw std::runtime_error("GGUF: string length too large: " + std::to_string(len));
    }
    std::string s(len, '\0');
    if (!file_.read(s.data(), static_cast<std::streamsize>(len))) {
        throw std::runtime_error("GGUF: unexpected end of file reading string");
    }
    return s;
}

GGUFScalar GGUFFile::read_scalar(GGUFValueType t) {
    switch (t) {
        case GGUFValueType::UINT8:   return read_pod<uint8_t>();
        case GGUFValueType::INT8:    return read_pod<int8_t>();
        case GGUFValueType::UINT16:  return read_pod<uint16_t>();
        case GGUFValueType::INT16:   return read_pod<int16_t>();
        case GGUFValueType::UINT32:  return read_pod<uint32_t>();
        case GGUFValueType::INT32:   return read_pod<int32_t>();
        case GGUFValueType::FLOAT32: return read_pod<float>();
        case GGUFValueType::BOOL:    return static_cast<bool>(read_pod<uint8_t>());
        case GGUFValueType::STRING:  return read_gguf_string();
        case GGUFValueType::UINT64:  return read_pod<uint64_t>();
        case GGUFValueType::INT64:   return read_pod<int64_t>();
        case GGUFValueType::FLOAT64: return read_pod<double>();
        default:
            throw std::runtime_error("GGUF: unknown scalar type " +
                                     std::to_string(static_cast<uint32_t>(t)));
    }
}

GGUFValue GGUFFile::read_value() {
    GGUFValue val;
    val.type = static_cast<GGUFValueType>(read_pod<uint32_t>());

    if (val.type == GGUFValueType::ARRAY) {
        auto elem_type = static_cast<GGUFValueType>(read_pod<uint32_t>());
        uint64_t count = read_pod<uint64_t>();
        val.array.reserve(static_cast<std::size_t>(count));
        for (uint64_t i = 0; i < count; ++i) {
            val.array.push_back(read_scalar(elem_type));
        }
    } else {
        val.scalar = read_scalar(val.type);
    }
    return val;
}

// ---------------------------------------------------------------------------
// open()
// ---------------------------------------------------------------------------
prime::Status GGUFFile::open(const std::string& path) {
    path_ = path;
    file_.open(path, std::ios::binary);
    if (!file_.is_open()) {
        return prime::Status::error(prime::StatusCode::ERROR_INIT_FAILED,
                                    "GGUF: cannot open file: " + path);
    }

    // Magic
    char magic[4];
    if (!file_.read(magic, 4) || std::memcmp(magic, "GGUF", 4) != 0) {
        return prime::Status::error(prime::StatusCode::ERROR_INIT_FAILED,
                                    "GGUF: invalid magic bytes in " + path);
    }

    version_ = read_pod<uint32_t>();
    if (version_ < 2 || version_ > 3) {
        return prime::Status::error(prime::StatusCode::ERROR_INIT_FAILED,
                                    "GGUF: unsupported version " +
                                    std::to_string(version_) + " in " + path);
    }

    n_tensors_ = read_pod<uint64_t>();
    n_meta_kv_ = read_pod<uint64_t>();

    try {
        auto s = read_metadata();
        if (!s.ok()) return s;
        s = read_tensor_index();
        if (!s.ok()) return s;
    } catch (const std::exception& e) {
        return prime::Status::error(prime::StatusCode::ERROR_INIT_FAILED,
                                    std::string("GGUF parse error: ") + e.what());
    }

    is_open_ = true;
    return prime::Status::ok_status();
}

prime::Status GGUFFile::read_metadata() {
    for (uint64_t i = 0; i < n_meta_kv_; ++i) {
        std::string key = read_gguf_string();
        GGUFValue   val = read_value();
        metadata_[key]  = std::move(val);
    }
    return prime::Status::ok_status();
}

prime::Status GGUFFile::read_tensor_index() {
    for (uint64_t i = 0; i < n_tensors_; ++i) {
        GGUFTensorInfo info;
        info.name = read_gguf_string();

        uint32_t ndim = read_pod<uint32_t>();
        info.dims.resize(ndim);
        for (uint32_t d = 0; d < ndim; ++d) {
            info.dims[d] = read_pod<uint64_t>();
        }

        info.gguf_type = read_pod<uint32_t>();
        info.offset    = read_pod<uint64_t>();

        auto it = GGUF_TYPE_MAP.find(info.gguf_type);
        if (it == GGUF_TYPE_MAP.end()) {
            return prime::Status::error(prime::StatusCode::ERROR_INIT_FAILED,
                                        "GGUF: unsupported tensor type " +
                                        std::to_string(info.gguf_type) +
                                        " for tensor '" + info.name + "'");
        }
        const auto& ti = it->second;
        info.dtype = ti.dtype;

        uint64_t numel = 1;
        for (auto d : info.dims) numel *= d;
        uint64_t nblocks = (numel + ti.block_size - 1) / ti.block_size;
        info.nbytes = static_cast<std::size_t>(nblocks * ti.block_bytes);

        tensor_names_.push_back(info.name);
        tensor_index_[info.name] = std::move(info);
    }

    // Alignment: GGUF aligns tensor data section to 32 bytes.
    uint64_t pos = static_cast<uint64_t>(file_.tellg());
    uint64_t align = 32;
    uint64_t remainder = pos % align;
    if (remainder != 0) {
        file_.seekg(static_cast<std::streamoff>(align - remainder), std::ios::cur);
    }
    data_offset_ = static_cast<uint64_t>(file_.tellg());

    return prime::Status::ok_status();
}

// ---------------------------------------------------------------------------
// Metadata accessors
// ---------------------------------------------------------------------------
bool GGUFFile::has_meta(const std::string& key) const {
    return metadata_.count(key) > 0;
}

const GGUFValue* GGUFFile::get_meta(const std::string& key) const {
    auto it = metadata_.find(key);
    return (it != metadata_.end()) ? &it->second : nullptr;
}

template<typename T>
static T scalar_as(const GGUFScalar& s) {
    return std::visit([](auto&& v) -> T {
        using V = std::decay_t<decltype(v)>;
        if constexpr (std::is_convertible_v<V, T>) {
            return static_cast<T>(v);
        }
        return T{};
    }, s);
}

uint32_t GGUFFile::meta_u32(const std::string& key, uint32_t def) const {
    auto* v = get_meta(key);
    if (!v || v->type == GGUFValueType::ARRAY) return def;
    return scalar_as<uint32_t>(v->scalar);
}

int32_t GGUFFile::meta_i32(const std::string& key, int32_t def) const {
    auto* v = get_meta(key);
    if (!v || v->type == GGUFValueType::ARRAY) return def;
    return scalar_as<int32_t>(v->scalar);
}

uint64_t GGUFFile::meta_u64(const std::string& key, uint64_t def) const {
    auto* v = get_meta(key);
    if (!v || v->type == GGUFValueType::ARRAY) return def;
    return scalar_as<uint64_t>(v->scalar);
}

float GGUFFile::meta_f32(const std::string& key, float def) const {
    auto* v = get_meta(key);
    if (!v || v->type == GGUFValueType::ARRAY) return def;
    return scalar_as<float>(v->scalar);
}

std::string GGUFFile::meta_str(const std::string& key, const std::string& def) const {
    auto* v = get_meta(key);
    if (!v || v->type != GGUFValueType::STRING) return def;
    return std::get<std::string>(v->scalar);
}

// ---------------------------------------------------------------------------
// Tensor loading
// ---------------------------------------------------------------------------
const GGUFTensorInfo* GGUFFile::tensor_info(const std::string& name) const {
    auto it = tensor_index_.find(name);
    return (it != tensor_index_.end()) ? &it->second : nullptr;
}

prime::Status GGUFFile::load_tensor(const std::string& name, Tensor& out) {
    auto* info = tensor_info(name);
    if (!info) {
        return prime::Status::error(prime::StatusCode::ERROR_MODULE_NOT_FOUND,
                                    "GGUF: tensor not found: " + name);
    }

    // Allocate raw buffer and read from file.
    auto buf = std::shared_ptr<uint8_t[]>(new uint8_t[info->nbytes]());
    uint64_t file_pos = data_offset_ + info->offset;
    file_.seekg(static_cast<std::streamoff>(file_pos));
    if (!file_.read(reinterpret_cast<char*>(buf.get()),
                    static_cast<std::streamsize>(info->nbytes))) {
        return prime::Status::error(prime::StatusCode::ERROR_INIT_FAILED,
                                    "GGUF: failed to read tensor: " + name);
    }

    // Build shape (GGUF stores innermost dim first; we keep that convention).
    std::vector<std::size_t> shape;
    shape.reserve(info->dims.size());
    for (auto d : info->dims) shape.push_back(static_cast<std::size_t>(d));

    // We store a shared_ptr via a lambda-based deleter in Tensor.
    // Since Tensor::view doesn't own memory, we wrap in an owning helper.
    // Use a trick: store the shared_ptr in a shared_ptr<void> with custom deleter.
    struct OwnedBuf {
        std::shared_ptr<uint8_t[]> data;
    };
    auto owned = std::make_shared<OwnedBuf>(OwnedBuf{buf});

    // Create a Tensor that keeps the shared_ptr alive via the owned float[] trick.
    // For simplicity, we build an F32 tensor (dequantizing here if needed).
    if (info->dtype == DType::F32) {
        Tensor raw = Tensor::view(buf.get(), DType::F32, shape, info->nbytes);
        // Convert to owned F32
        out = raw.to_f32();
    } else {
        // Build a raw view for dequantisation, then convert.
        Tensor raw = Tensor::view(buf.get(), info->dtype, shape, info->nbytes);
        out = raw.to_f32();
    }

    return prime::Status::ok_status();
}

// ---------------------------------------------------------------------------
// build_model_config
// ---------------------------------------------------------------------------
prime::Status GGUFFile::build_model_config(ModelConfig& cfg) const {
    // Architecture
    std::string arch_str = meta_str("general.architecture", "");
    if (arch_str == "llama" || arch_str == "mistral" ||
        arch_str == "phi2" || arch_str.empty()) {
        cfg.arch = ModelArchitecture::LLAMA;
    } else {
        return prime::Status::error(prime::StatusCode::ERROR_INIT_FAILED,
                                    "GGUF: unsupported architecture: " + arch_str);
    }

    cfg.model_name = meta_str("general.name", path_);

    // Use architecture-prefixed keys (llama.*, fallback to unprefixed for fixtures).
    auto get_u32 = [&](const std::string& key, const std::string& fallback,
                        uint32_t def) -> uint32_t {
        if (has_meta(key))      return meta_u32(key, def);
        if (has_meta(fallback)) return meta_u32(fallback, def);
        return def;
    };

    cfg.vocab_size     = get_u32("llama.vocab_size",
                                  "prime.vocab_size", 0);
    if (cfg.vocab_size == 0) {
        // Infer from tokenizer.ggml.tokens array length
        auto* v = get_meta("tokenizer.ggml.tokens");
        if (v && v->type == GGUFValueType::ARRAY) {
            cfg.vocab_size = static_cast<uint32_t>(v->array.size());
        }
    }

    cfg.context_length = get_u32("llama.context_length",
                                  "prime.context_length", 512);
    cfg.embed_dim      = get_u32("llama.embedding_length",
                                  "prime.embed_dim", 0);
    cfg.num_layers     = get_u32("llama.block_count",
                                  "prime.num_layers", 0);
    cfg.num_heads      = get_u32("llama.attention.head_count",
                                  "prime.num_heads", 0);
    cfg.num_kv_heads   = get_u32("llama.attention.head_count_kv",
                                  "prime.num_kv_heads", cfg.num_heads);
    cfg.ffn_hidden_dim = get_u32("llama.feed_forward_length",
                                  "prime.ffn_hidden_dim", 0);
    cfg.rope_theta     = meta_f32("llama.rope.freq_base", 10000.0f);

    cfg.bos_token_id   = meta_i32("tokenizer.ggml.bos_token_id", 1);
    cfg.eos_token_id   = meta_i32("tokenizer.ggml.eos_token_id", 2);

    if (cfg.embed_dim == 0 || cfg.num_layers == 0 || cfg.num_heads == 0) {
        return prime::Status::error(prime::StatusCode::ERROR_INVALID_CONFIG,
                                    "GGUF: missing required model dimensions");
    }
    if (cfg.ffn_hidden_dim == 0) {
        // Default: 4x embed (GPT-2 style) or 8/3 * embed (LLaMA SwiGLU)
        cfg.ffn_hidden_dim = cfg.embed_dim * 4;
    }

    return prime::Status::ok_status();
}

} // namespace prime::llm
