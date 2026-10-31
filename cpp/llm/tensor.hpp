#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>
#include <string>

namespace prime::llm {

// Numeric types supported by the runtime.
enum class DType {
    F32,    // 32-bit float (full precision)
    F16,    // 16-bit float (storage / future)
    Q4_0,   // 4-bit quantized, block-size 32, zero-point 0
    Q8_0,   // 8-bit quantized, block-size 32, zero-point 0
};

inline std::string dtype_name(DType dt) {
    switch (dt) {
        case DType::F32:  return "F32";
        case DType::F16:  return "F16";
        case DType::Q4_0: return "Q4_0";
        case DType::Q8_0: return "Q8_0";
        default:          return "UNKNOWN";
    }
}

// Q4_0 block: 16 bytes per 32 weights.
// Layout: [f16 scale][32 x 4-bit nibbles packed in 16 bytes]
struct Q4_0Block {
    uint16_t scale_f16;
    uint8_t  qs[16];      // 32 nibbles packed
};
static_assert(sizeof(Q4_0Block) == 18, "Q4_0Block size mismatch");

// Q8_0 block: 36 bytes per 32 weights.
struct Q8_0Block {
    uint16_t scale_f16;
    int8_t   qs[32];
};
static_assert(sizeof(Q8_0Block) == 34, "Q8_0Block size mismatch");

// Decode IEEE 754 float16 → float32.
inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1u;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t man  = h & 0x3FFu;
    uint32_t f;
    if (exp == 0) {
        if (man == 0) { f = sign << 31; }
        else {
            // Subnormal
            exp = 1;
            while (!(man & 0x400)) { man <<= 1; exp--; }
            man &= 0x3FF;
            f = (sign << 31) | ((exp + 127 - 15) << 23) | (man << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7F800000u | (man << 13);
    } else {
        f = (sign << 31) | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float result;
    std::memcpy(&result, &f, 4);
    return result;
}

// Dequantize a Q4_0 block into dst (must have room for 32 floats).
inline void dequantize_q4_0_block(const Q4_0Block& blk, float* dst) {
    float scale = f16_to_f32(blk.scale_f16);
    for (int i = 0; i < 16; ++i) {
        int lo = (blk.qs[i] & 0x0F) - 8;
        int hi = (blk.qs[i] >> 4)   - 8;
        dst[i*2    ] = lo * scale;
        dst[i*2 + 1] = hi * scale;
    }
}

// Dequantize a Q8_0 block into dst (must have room for 32 floats).
inline void dequantize_q8_0_block(const Q8_0Block& blk, float* dst) {
    float scale = f16_to_f32(blk.scale_f16);
    for (int i = 0; i < 32; ++i) {
        dst[i] = blk.qs[i] * scale;
    }
}

// Tensor: a named, typed, contiguous buffer of numbers.
// Owns its data when created via allocate(); acts as a view otherwise.
class Tensor {
public:
    Tensor() = default;

    // Create an F32 tensor, allocating fresh memory.
    static Tensor allocate_f32(std::vector<std::size_t> shape) {
        Tensor t;
        t.shape_ = std::move(shape);
        t.dtype_ = DType::F32;
        std::size_t n = t.numel();
        t.owned_ = std::shared_ptr<float[]>(new float[n]());
        t.data_  = t.owned_.get();
        t.nbytes_ = n * sizeof(float);
        return t;
    }

    // Create a view over external memory (no ownership transfer).
    static Tensor view(void* data, DType dtype,
                       std::vector<std::size_t> shape, std::size_t nbytes) {
        Tensor t;
        t.data_   = data;
        t.dtype_  = dtype;
        t.shape_  = std::move(shape);
        t.nbytes_ = nbytes;
        return t;
    }

    DType dtype()  const { return dtype_; }
    const std::vector<std::size_t>& shape() const { return shape_; }
    std::size_t nbytes() const { return nbytes_; }
    void* raw()  { return data_; }
    const void* raw() const { return data_; }
    float* f32() { return static_cast<float*>(data_); }
    const float* f32() const { return static_cast<const float*>(data_); }

    std::size_t numel() const {
        if (shape_.empty()) return 0;
        std::size_t n = 1;
        for (auto d : shape_) n *= d;
        return n;
    }

    bool valid() const { return data_ != nullptr; }

    // Dequantize this tensor into a fresh F32 tensor.
    // For F32, returns a view (no copy). For quantized, allocates.
    Tensor to_f32() const;

private:
    void*  data_  = nullptr;
    DType  dtype_ = DType::F32;
    std::vector<std::size_t> shape_;
    std::size_t nbytes_ = 0;
    std::shared_ptr<float[]> owned_;  // non-null only when we own the data
};

inline Tensor Tensor::to_f32() const {
    if (dtype_ == DType::F32) {
        // Return a view — no copy needed.
        return Tensor::view(const_cast<void*>(data_), DType::F32, shape_, nbytes_);
    }
    std::size_t n = numel();
    Tensor out = Tensor::allocate_f32(shape_);
    float* dst = out.f32();

    if (dtype_ == DType::Q4_0) {
        std::size_t nblocks = n / 32;
        const auto* blocks = static_cast<const Q4_0Block*>(data_);
        for (std::size_t b = 0; b < nblocks; ++b) {
            dequantize_q4_0_block(blocks[b], dst + b * 32);
        }
    } else if (dtype_ == DType::Q8_0) {
        std::size_t nblocks = n / 32;
        const auto* blocks = static_cast<const Q8_0Block*>(data_);
        for (std::size_t b = 0; b < nblocks; ++b) {
            dequantize_q8_0_block(blocks[b], dst + b * 32);
        }
    } else if (dtype_ == DType::F16) {
        const uint16_t* src = static_cast<const uint16_t*>(data_);
        for (std::size_t i = 0; i < n; ++i) {
            dst[i] = f16_to_f32(src[i]);
        }
    } else {
        throw std::runtime_error("Tensor::to_f32: unsupported dtype");
    }
    return out;
}

// ---------------------------------------------------------------------------
// Math primitives (F32)
// ---------------------------------------------------------------------------

// dot product of two F32 vectors
inline float vec_dot_f32(const float* a, const float* b, std::size_t n) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

// matrix-vector multiply: out[M] = mat[M][K] * vec[K]
inline void matmul_f32(const float* mat, const float* vec,
                       float* out, std::size_t M, std::size_t K) {
    for (std::size_t m = 0; m < M; ++m) {
        out[m] = vec_dot_f32(mat + m * K, vec, K);
    }
}

// element-wise add: a += b
inline void vec_add_inplace(float* a, const float* b, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) a[i] += b[i];
}

// element-wise multiply: a *= b
inline void vec_mul_inplace(float* a, const float* b, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) a[i] *= b[i];
}

// scale: a *= s
inline void vec_scale_inplace(float* a, float s, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) a[i] *= s;
}

// copy
inline void vec_copy(float* dst, const float* src, std::size_t n) {
    std::memcpy(dst, src, n * sizeof(float));
}

// RMS norm: out[i] = x[i] / rms(x) * weight[i]
inline void rms_norm(const float* x, const float* weight,
                     float* out, std::size_t n, float eps = 1e-5f) {
    float ss = 0.0f;
    for (std::size_t i = 0; i < n; ++i) ss += x[i] * x[i];
    float scale = 1.0f / std::sqrt(ss / static_cast<float>(n) + eps);
    for (std::size_t i = 0; i < n; ++i) out[i] = x[i] * scale * weight[i];
}

// SiLU activation: x * sigmoid(x)
inline float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

// Softmax in-place over n elements
inline void softmax_inplace(float* x, std::size_t n) {
    float mx = x[0];
    for (std::size_t i = 1; i < n; ++i) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (std::size_t i = 0; i < n; ++i) { x[i] = std::exp(x[i] - mx); sum += x[i]; }
    float inv = 1.0f / sum;
    for (std::size_t i = 0; i < n; ++i) x[i] *= inv;
}

} // namespace prime::llm
