"""
create_fixture.py — Generate a minimal valid GGUF v3 file for automated tests.

The fixture contains:
  - A tiny LLaMA-style transformer (2 layers, 4 heads, d=32, ffn=64)
  - A small vocabulary of 64 tokens (printable ASCII + special)
  - Random F32 weights (seeded for reproducibility)
  - All metadata required by PRIME's GGUF parser

Run from the project root:
    python3 tests/fixtures/create_fixture.py

Output: tests/fixtures/tiny_llama.gguf
"""

import struct
import os
import numpy as np

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "tiny_llama.gguf")

# Model hyperparameters (deliberately tiny for fast tests)
VOCAB_SIZE     = 64
CONTEXT_LEN    = 64
EMBED_DIM      = 32
NUM_LAYERS     = 2
NUM_HEADS      = 4
NUM_KV_HEADS   = 4
FFN_HIDDEN     = 64
HEAD_DIM       = EMBED_DIM // NUM_HEADS   # 8
ROPE_THETA     = 10000.0

BOS_ID = 1
EOS_ID = 2

rng = np.random.default_rng(seed=42)


# ---------------------------------------------------------------------------
# GGUF serialisation helpers
# ---------------------------------------------------------------------------
def write_u8(f, v):   f.write(struct.pack('<B', v))
def write_u16(f, v):  f.write(struct.pack('<H', v))
def write_i16(f, v):  f.write(struct.pack('<h', v))
def write_u32(f, v):  f.write(struct.pack('<I', v))
def write_i32(f, v):  f.write(struct.pack('<i', v))
def write_u64(f, v):  f.write(struct.pack('<Q', v))
def write_f32(f, v):  f.write(struct.pack('<f', v))

def write_str(f, s: str):
    b = s.encode('utf-8')
    write_u64(f, len(b))
    f.write(b)

# GGUF value types
GGUF_TYPE_UINT8   = 0
GGUF_TYPE_INT8    = 1
GGUF_TYPE_UINT32  = 4
GGUF_TYPE_INT32   = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL    = 7
GGUF_TYPE_STRING  = 8
GGUF_TYPE_ARRAY   = 9
GGUF_TYPE_UINT64  = 10

def write_kv_string(f, key: str, value: str):
    write_str(f, key)
    write_u32(f, GGUF_TYPE_STRING)
    write_str(f, value)

def write_kv_u32(f, key: str, value: int):
    write_str(f, key)
    write_u32(f, GGUF_TYPE_UINT32)
    write_u32(f, value)

def write_kv_i32(f, key: str, value: int):
    write_str(f, key)
    write_u32(f, GGUF_TYPE_INT32)
    write_i32(f, value)

def write_kv_f32(f, key: str, value: float):
    write_str(f, key)
    write_u32(f, GGUF_TYPE_FLOAT32)
    write_f32(f, value)

def write_kv_string_array(f, key: str, values: list):
    write_str(f, key)
    write_u32(f, GGUF_TYPE_ARRAY)
    write_u32(f, GGUF_TYPE_STRING)
    write_u64(f, len(values))
    for v in values:
        write_str(f, v)

def write_kv_f32_array(f, key: str, values: list):
    write_str(f, key)
    write_u32(f, GGUF_TYPE_ARRAY)
    write_u32(f, GGUF_TYPE_FLOAT32)
    write_u64(f, len(values))
    for v in values:
        write_f32(f, v)

def write_kv_i32_array(f, key: str, values: list):
    write_str(f, key)
    write_u32(f, GGUF_TYPE_ARRAY)
    write_u32(f, GGUF_TYPE_INT32)
    write_u64(f, len(values))
    for v in values:
        write_i32(f, v)


# ---------------------------------------------------------------------------
# Vocabulary (64 tokens)
# ---------------------------------------------------------------------------
def build_vocab():
    tokens = []
    scores = []
    types  = []

    # 0: <unk>, 1: <s>, 2: </s>
    specials = [("<unk>", 0, 2), ("<s>", 0, 3), ("</s>", 0, 3)]
    for tok, score, ttype in specials:
        tokens.append(tok)
        scores.append(score)
        types.append(ttype)

    # 3..34: single printable ASCII chars (space, A-Z, a-z subset)
    ascii_chars = [' '] + [chr(c) for c in range(ord('a'), ord('a') + 26)] + \
                  [chr(c) for c in range(ord('A'), ord('A') + 3)]
    for ch in ascii_chars:
        tokens.append(ch)
        scores.append(-float(len(tokens)))
        types.append(0)

    # Fill to VOCAB_SIZE with BPE merge tokens
    merges = ["he", "ll", "o ", "wo", "rld", " h", "el", "lo", " w", "or",
              "ld", "the", " t", "is", " i", "an", " a", "in", " in",
              "er", " e", "re", "at", " at", "on", " o", "nd", " n"]
    for m in merges:
        if len(tokens) >= VOCAB_SIZE:
            break
        tokens.append(m)
        scores.append(-float(len(tokens)))
        types.append(0)

    # Pad to exact VOCAB_SIZE
    while len(tokens) < VOCAB_SIZE:
        tokens.append(f"<pad{len(tokens)}>")
        scores.append(-float(len(tokens)))
        types.append(2)

    assert len(tokens) == VOCAB_SIZE
    return tokens, scores, types


# ---------------------------------------------------------------------------
# Weight tensors
# ---------------------------------------------------------------------------
def small_rand(shape, scale=0.1):
    """Small random F32 weights."""
    return (rng.standard_normal(shape) * scale).astype(np.float32)

def ones(shape):
    return np.ones(shape, dtype=np.float32)

def build_weights():
    D  = EMBED_DIM
    L  = NUM_LAYERS
    H  = NUM_HEADS
    Hk = NUM_KV_HEADS
    dh = HEAD_DIM
    F  = FFN_HIDDEN
    V  = VOCAB_SIZE

    tensors = {}
    # Token embedding
    tensors["token_embd.weight"] = small_rand((V, D))

    for l in range(L):
        p = f"blk.{l}."
        tensors[p + "attn_norm.weight"]   = ones((D,))
        tensors[p + "attn_q.weight"]      = small_rand((H * dh, D))
        tensors[p + "attn_k.weight"]      = small_rand((Hk * dh, D))
        tensors[p + "attn_v.weight"]      = small_rand((Hk * dh, D))
        tensors[p + "attn_output.weight"] = small_rand((D, H * dh))
        tensors[p + "ffn_norm.weight"]    = ones((D,))
        tensors[p + "ffn_gate.weight"]    = small_rand((F, D))
        tensors[p + "ffn_up.weight"]      = small_rand((F, D))
        tensors[p + "ffn_down.weight"]    = small_rand((D, F))

    tensors["output_norm.weight"] = ones((D,))
    tensors["output.weight"]      = small_rand((V, D))

    return tensors


# ---------------------------------------------------------------------------
# GGUF tensor type constants
# ---------------------------------------------------------------------------
GGML_TYPE_F32 = 0


def write_gguf(path: str):
    tokens, scores, token_types = build_vocab()
    weights = build_weights()

    # Pre-serialise tensor data to know offsets.
    tensor_data_parts = []
    tensor_infos = []  # (name, dims, offset, nbytes)

    offset = 0
    for name, arr in weights.items():
        # GGUF stores dims innermost-first; arr.shape is outermost-first.
        dims = list(reversed(arr.shape))
        data = arr.tobytes()
        tensor_infos.append((name, dims, offset, len(data)))
        tensor_data_parts.append(data)
        # Pad each tensor to 32-byte alignment within data section.
        remainder = len(data) % 32
        if remainder:
            pad = 32 - remainder
            tensor_data_parts.append(b'\x00' * pad)
            offset += len(data) + pad
        else:
            offset += len(data)

    n_tensors  = len(tensor_infos)
    n_meta_kvs = 14  # number of metadata key-value pairs written below

    with open(path, 'wb') as f:
        # Magic + version + tensor count + meta kv count
        f.write(b'GGUF')
        write_u32(f, 3)               # version 3
        write_u64(f, n_tensors)
        write_u64(f, n_meta_kvs)

        # Metadata
        write_kv_string(f,    "general.architecture",            "llama")
        write_kv_string(f,    "general.name",                    "PRIME-tiny-fixture")
        write_kv_u32(f,       "prime.vocab_size",                VOCAB_SIZE)
        write_kv_u32(f,       "prime.context_length",            CONTEXT_LEN)
        write_kv_u32(f,       "prime.embed_dim",                 EMBED_DIM)
        write_kv_u32(f,       "prime.num_layers",                NUM_LAYERS)
        write_kv_u32(f,       "prime.num_heads",                 NUM_HEADS)
        write_kv_u32(f,       "prime.num_kv_heads",              NUM_KV_HEADS)
        write_kv_u32(f,       "prime.ffn_hidden_dim",            FFN_HIDDEN)
        write_kv_i32(f,       "tokenizer.ggml.bos_token_id",    BOS_ID)
        write_kv_i32(f,       "tokenizer.ggml.eos_token_id",    EOS_ID)
        write_kv_string_array(f, "tokenizer.ggml.tokens",       tokens)
        write_kv_f32_array(f, "tokenizer.ggml.scores",          scores)
        write_kv_i32_array(f, "tokenizer.ggml.token_type",      token_types)

        # Tensor index
        for name, dims, toffset, nbytes in tensor_infos:
            write_str(f,  name)
            write_u32(f,  len(dims))
            for d in dims:
                write_u64(f, d)
            write_u32(f,  GGML_TYPE_F32)
            write_u64(f,  toffset)

        # Align to 32 bytes before tensor data section.
        pos = f.tell()
        remainder = pos % 32
        if remainder:
            f.write(b'\x00' * (32 - remainder))

        # Tensor data
        for part in tensor_data_parts:
            f.write(part)

    size_kb = os.path.getsize(path) / 1024
    print(f"Created: {path}")
    print(f"  Vocab size:  {VOCAB_SIZE}")
    print(f"  Layers:      {NUM_LAYERS}")
    print(f"  Embed dim:   {EMBED_DIM}")
    print(f"  FFN hidden:  {FFN_HIDDEN}")
    print(f"  File size:   {size_kb:.1f} KB")


if __name__ == "__main__":
    write_gguf(OUTPUT_PATH)
