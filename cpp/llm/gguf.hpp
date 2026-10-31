#pragma once

#include "model_config.hpp"
#include "tensor.hpp"
#include "../core/status.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include <variant>
#include <memory>
#include <fstream>

namespace prime::llm {

// GGUF metadata value types (enum matches GGUF spec).
enum class GGUFValueType : uint32_t {
    UINT8   = 0,
    INT8    = 1,
    UINT16  = 2,
    INT16   = 3,
    UINT32  = 4,
    INT32   = 5,
    FLOAT32 = 6,
    BOOL    = 7,
    STRING  = 8,
    ARRAY   = 9,
    UINT64  = 10,
    INT64   = 11,
    FLOAT64 = 12,
};

// A GGUF metadata value (scalar or array variant).
using GGUFScalar = std::variant<
    uint8_t, int8_t, uint16_t, int16_t,
    uint32_t, int32_t, uint64_t, int64_t,
    float, double, bool, std::string
>;

struct GGUFValue {
    GGUFValueType type;
    GGUFScalar    scalar;            // used when type != ARRAY
    std::vector<GGUFScalar> array;   // used when type == ARRAY
};

// Describes one tensor in the GGUF file (location + layout; no data loaded yet).
struct GGUFTensorInfo {
    std::string name;
    std::vector<uint64_t> dims;  // shape in GGUF order (innermost first)
    uint32_t gguf_type;          // GGUF tensor type enum
    uint64_t offset;             // byte offset from tensor data section start
    DType    dtype;              // translated to our DType
    std::size_t nbytes;          // total bytes for this tensor
};

// GGUFFile: opens a GGUF file and provides metadata + lazy tensor loading.
// No tensor data is read until load_tensor() is called.
class GGUFFile {
public:
    GGUFFile() = default;
    ~GGUFFile() = default;

    GGUFFile(const GGUFFile&) = delete;
    GGUFFile& operator=(const GGUFFile&) = delete;

    // Open and parse the GGUF header + metadata + tensor index.
    // Tensor data is NOT loaded yet.
    prime::Status open(const std::string& path);

    // Access parsed metadata.
    bool has_meta(const std::string& key) const;
    const GGUFValue* get_meta(const std::string& key) const;

    // Convenience helpers for common metadata types.
    uint32_t     meta_u32(const std::string& key, uint32_t def = 0) const;
    int32_t      meta_i32(const std::string& key, int32_t  def = 0) const;
    uint64_t     meta_u64(const std::string& key, uint64_t def = 0) const;
    float        meta_f32(const std::string& key, float    def = 0) const;
    std::string  meta_str(const std::string& key, const std::string& def = "") const;

    // List all tensor names.
    const std::vector<std::string>& tensor_names() const { return tensor_names_; }

    // Get tensor info (without loading data).
    const GGUFTensorInfo* tensor_info(const std::string& name) const;

    // Load tensor data into a Tensor (backed by file-mapped or read memory).
    // Returns invalid Tensor on error.
    prime::Status load_tensor(const std::string& name, Tensor& out);

    // Build a ModelConfig from GGUF metadata.
    prime::Status build_model_config(ModelConfig& cfg) const;

    bool is_open() const { return is_open_; }
    const std::string& path() const { return path_; }

private:
    prime::Status read_metadata();
    prime::Status read_tensor_index();

    template<typename T> T read_pod();
    std::string read_gguf_string();
    GGUFScalar  read_scalar(GGUFValueType t);
    GGUFValue   read_value();

    std::ifstream file_;
    std::string   path_;
    bool          is_open_ = false;

    uint32_t version_     = 0;
    uint64_t n_tensors_   = 0;
    uint64_t n_meta_kv_   = 0;
    uint64_t data_offset_ = 0;   // byte offset in file where tensor data begins

    std::unordered_map<std::string, GGUFValue> metadata_;
    std::vector<std::string>                   tensor_names_;
    std::unordered_map<std::string, GGUFTensorInfo> tensor_index_;
};

} // namespace prime::llm
