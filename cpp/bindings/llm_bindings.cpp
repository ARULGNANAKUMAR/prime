/*
 * llm_bindings.cpp — Python C API binding for PRIME LLM engine (Phase 2).
 *
 * Architecture:
 *   Python (prime/llm.py)
 *     ↓
 *   Python wrapper (prime/llm.py)
 *     ↓
 *   this file (Python C API)
 *     ↓
 *   cpp/llm/llm_engine.hpp
 *
 * The binding is intentionally thin: all logic lives in C++ source modules.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "../llm/llm_engine.hpp"
#include "../core/status.hpp"

#include <memory>
#include <string>

using namespace prime;
using namespace prime::llm;

// ---------------------------------------------------------------------------
// Exception
// ---------------------------------------------------------------------------
static PyObject* LLMError = nullptr;

static void raise_llm_error(const prime::Status& s) {
    PyErr_SetString(LLMError, s.message.c_str());
}

// ---------------------------------------------------------------------------
// LLMEngineObject — wraps a prime::llm::LLMEngine
// ---------------------------------------------------------------------------
struct LLMEngineObject {
    PyObject_HEAD
    LLMEngine* engine;
};

static void LLMEngineObject_dealloc(LLMEngineObject* self) {
    delete self->engine;
    self->engine = nullptr;
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

static PyObject* LLMEngineObject_new(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
    static const char* kwlist[] = {"model_path", nullptr};
    const char* model_path = nullptr;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s",
                                      const_cast<char**>(kwlist), &model_path)) {
        return nullptr;
    }

    LLMEngineObject* self = reinterpret_cast<LLMEngineObject*>(type->tp_alloc(type, 0));
    if (!self) return nullptr;

    try {
        self->engine = new LLMEngine();
        auto s = self->engine->load(model_path);
        if (!s.ok()) {
            delete self->engine;
            self->engine = nullptr;
            Py_DECREF(self);
            PyErr_SetString(LLMError, s.message.c_str());
            return nullptr;
        }
    } catch (const std::exception& e) {
        Py_DECREF(self);
        PyErr_SetString(LLMError, e.what());
        return nullptr;
    }
    return reinterpret_cast<PyObject*>(self);
}

// generate(prompt, max_new_tokens, temperature, top_k, top_p, seed) → str
static PyObject* LLMEngineObject_generate(LLMEngineObject* self, PyObject* args, PyObject* kwargs) {
    static const char* kwlist[] = {
        "prompt", "max_new_tokens", "temperature", "top_k", "top_p", "seed", nullptr
    };
    const char* prompt = nullptr;
    int         max_new_tokens = 128;
    float       temperature    = 1.0f;
    int         top_k          = 0;
    float       top_p          = 1.0f;
    long long   seed           = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|ififi",
                                      const_cast<char**>(kwlist),
                                      &prompt, &max_new_tokens, &temperature,
                                      &top_k, &top_p, &seed)) {
        return nullptr;
    }

    GenerationParams gp;
    gp.max_new_tokens = max_new_tokens;
    gp.temperature    = temperature;
    gp.top_k          = top_k;
    gp.top_p          = top_p;
    gp.seed           = static_cast<uint64_t>(seed);
    gp.seed_set       = (seed != 0);

    std::string output;
    prime::Status s;
    Py_BEGIN_ALLOW_THREADS
    try {
        s = self->engine->generate(prompt, gp, output);
    } catch (const std::exception& e) {
        s = prime::Status::error(prime::StatusCode::ERROR_INIT_FAILED, e.what());
    }
    Py_END_ALLOW_THREADS

    if (!s.ok()) { raise_llm_error(s); return nullptr; }
    return PyUnicode_FromStringAndSize(output.data(),
                                        static_cast<Py_ssize_t>(output.size()));
}

// stream_generate(prompt, callback, max_new_tokens, temperature, top_k, top_p, seed) → None
static PyObject* LLMEngineObject_stream_generate(LLMEngineObject* self,
                                                   PyObject* args, PyObject* kwargs) {
    static const char* kwlist[] = {
        "prompt", "callback", "max_new_tokens", "temperature",
        "top_k", "top_p", "seed", nullptr
    };
    const char* prompt     = nullptr;
    PyObject*   callback   = nullptr;
    int         max_new_tokens = 128;
    float       temperature    = 1.0f;
    int         top_k          = 0;
    float       top_p          = 1.0f;
    long long   seed           = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sO|ififi",
                                      const_cast<char**>(kwlist),
                                      &prompt, &callback, &max_new_tokens,
                                      &temperature, &top_k, &top_p, &seed)) {
        return nullptr;
    }
    if (!PyCallable_Check(callback)) {
        PyErr_SetString(PyExc_TypeError, "callback must be callable");
        return nullptr;
    }

    GenerationParams gp;
    gp.max_new_tokens = max_new_tokens;
    gp.temperature    = temperature;
    gp.top_k          = top_k;
    gp.top_p          = top_p;
    gp.seed           = static_cast<uint64_t>(seed);
    gp.seed_set       = (seed != 0);

    // We need to call back into Python from the C++ callback,
    // so we must hold the GIL during stream generation.
    PyObject* py_callback = callback;
    Py_INCREF(py_callback);

    bool python_error = false;

    prime::Status s = self->engine->stream_generate(
        prompt, gp,
        [&](const std::string& piece) -> bool {
            // Re-acquire GIL for Python callback (we released it for generation).
            PyObject* py_piece = PyUnicode_FromStringAndSize(
                piece.data(), static_cast<Py_ssize_t>(piece.size()));
            if (!py_piece) { python_error = true; return false; }
            PyObject* result = PyObject_CallOneArg(py_callback, py_piece);
            Py_DECREF(py_piece);
            if (!result) { python_error = true; return false; }
            bool cont = PyObject_IsTrue(result);
            Py_DECREF(result);
            return cont;
        });

    Py_DECREF(py_callback);
    if (python_error) return nullptr;
    if (!s.ok()) { raise_llm_error(s); return nullptr; }
    Py_RETURN_NONE;
}

static PyObject* LLMEngineObject_vocab_size(LLMEngineObject* self, PyObject*) {
    return PyLong_FromUnsignedLong(self->engine->vocab_size());
}

static PyObject* LLMEngineObject_context_length(LLMEngineObject* self, PyObject*) {
    return PyLong_FromUnsignedLong(self->engine->context_length());
}

static PyObject* LLMEngineObject_model_path(LLMEngineObject* self, PyObject*) {
    return PyUnicode_FromString(self->engine->model_path().c_str());
}

static PyObject* LLMEngineObject_arch_name(LLMEngineObject* self, PyObject*) {
    auto a = self->engine->model_config().arch;
    return PyUnicode_FromString(arch_name(a).c_str());
}

static PyMethodDef LLMEngineObject_methods[] = {
    {"generate",
     reinterpret_cast<PyCFunction>(reinterpret_cast<void*>(LLMEngineObject_generate)),
     METH_VARARGS | METH_KEYWORDS, "Generate text from a prompt."},
    {"stream_generate",
     reinterpret_cast<PyCFunction>(reinterpret_cast<void*>(LLMEngineObject_stream_generate)),
     METH_VARARGS | METH_KEYWORDS, "Stream-generate text; calls callback(piece) per token."},
    {"vocab_size",      (PyCFunction)LLMEngineObject_vocab_size,
     METH_NOARGS, "Return vocabulary size."},
    {"context_length",  (PyCFunction)LLMEngineObject_context_length,
     METH_NOARGS, "Return model context length."},
    {"model_path",      (PyCFunction)LLMEngineObject_model_path,
     METH_NOARGS, "Return path to the loaded model file."},
    {"arch_name",       (PyCFunction)LLMEngineObject_arch_name,
     METH_NOARGS, "Return model architecture name."},
    {nullptr, nullptr, 0, nullptr}
};

// Use zero-initialised struct + targeted assignments to avoid aggregate-init warnings.
static PyTypeObject LLMEngineType = []() {
    PyTypeObject t = {};
    t.tp_name      = "_prime_llm._LLMEngine";
    t.tp_basicsize = sizeof(LLMEngineObject);
    t.tp_dealloc   = reinterpret_cast<destructor>(LLMEngineObject_dealloc);
    t.tp_flags     = Py_TPFLAGS_DEFAULT;
    t.tp_doc       = "Internal C++ LLM engine wrapper.";
    t.tp_methods   = LLMEngineObject_methods;
    t.tp_new       = LLMEngineObject_new;
    return t;
}();

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------
static PyMethodDef llm_module_methods[] = {
    {nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef llm_module_def = {
    PyModuleDef_HEAD_INIT,
    "_prime_llm",
    "PRIME LLM C++ extension module (Phase 2).",
    -1,
    llm_module_methods,
};

PyMODINIT_FUNC PyInit__prime_llm(void) {
    if (PyType_Ready(&LLMEngineType) < 0) return nullptr;

    PyObject* module = PyModule_Create(&llm_module_def);
    if (!module) return nullptr;

    LLMError = PyErr_NewExceptionWithDoc(
        "_prime_llm.LLMError",
        "Raised when a PRIME LLM operation fails.",
        nullptr, nullptr);
    if (!LLMError) { Py_DECREF(module); return nullptr; }
    Py_INCREF(LLMError);
    if (PyModule_AddObject(module, "LLMError", LLMError) < 0) {
        Py_DECREF(LLMError); Py_DECREF(module); return nullptr;
    }

    Py_INCREF(&LLMEngineType);
    if (PyModule_AddObject(module, "_LLMEngine",
                            reinterpret_cast<PyObject*>(&LLMEngineType)) < 0) {
        Py_DECREF(&LLMEngineType); Py_DECREF(module); return nullptr;
    }

    return module;
}
