/*
 * prime_bindings.cpp — Python C API binding layer for PRIME.
 *
 * This layer is intentionally thin: it translates between Python objects
 * and the C++ Runtime/Config. No business logic lives here.
 *
 * Architecture:
 *   Python (prime/__init__.py)
 *     ↓
 *   Python wrapper (prime/core.py)
 *     ↓
 *   this file (Python C API)
 *     ↓
 *   cpp/runtime/runtime.hpp and cpp/core/ headers
 *
 * Design note: This uses the Python C API directly (the same layer pybind11
 * wraps) because pybind11 is unavailable without network access in this
 * environment. The interface is designed so a future pybind11 replacement
 * is straightforward: each function maps 1-to-1 to a pybind11 def().
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "../runtime/runtime.hpp"
#include "../core/config.hpp"
#include "../core/status.hpp"

#include <memory>
#include <string>

using namespace prime;

// ---------------------------------------------------------------------------
// Python exception type for PRIME errors
// ---------------------------------------------------------------------------
static PyObject* PrimeError = nullptr;

static void raise_prime_error(const Status& s) {
    PyErr_SetString(PrimeError, s.message.c_str());
}

// ---------------------------------------------------------------------------
// RuntimeObject — wraps a prime::Runtime instance
// ---------------------------------------------------------------------------
struct RuntimeObject {
    PyObject_HEAD
    Runtime* runtime;  // owned; deleted in tp_dealloc
};

static void RuntimeObject_dealloc(RuntimeObject* self) {
    delete self->runtime;
    self->runtime = nullptr;
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

// _prime_runtime_create(log_level: int, compute_backend: str, max_memory_bytes: int)
static PyObject* RuntimeObject_new(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
    static const char* kwlist[] = {
        "log_level", "compute_backend", "max_memory_bytes", nullptr
    };

    int log_level = static_cast<int>(LogLevel::INFO);
    const char* compute_backend = "cpu";
    Py_ssize_t max_memory_bytes = 0;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "|isn",
            const_cast<char**>(kwlist),
            &log_level, &compute_backend, &max_memory_bytes)) {
        return nullptr;
    }

    RuntimeConfig config;
    config.log_level = static_cast<LogLevel>(log_level);
    config.compute_backend = compute_backend;
    config.max_memory_bytes = static_cast<std::size_t>(max_memory_bytes);

    std::string config_err;
    if (!config.validate(config_err)) {
        PyErr_SetString(PrimeError, config_err.c_str());
        return nullptr;
    }

    RuntimeObject* self = reinterpret_cast<RuntimeObject*>(type->tp_alloc(type, 0));
    if (!self) return nullptr;

    try {
        self->runtime = new Runtime(std::move(config));
    } catch (const std::exception& e) {
        Py_DECREF(self);
        PyErr_SetString(PrimeError, e.what());
        return nullptr;
    }

    return reinterpret_cast<PyObject*>(self);
}

static PyObject* RuntimeObject_initialize(RuntimeObject* self, PyObject* /*args*/) {
    if (!self->runtime) {
        PyErr_SetString(PrimeError, "Runtime object is in an invalid state");
        return nullptr;
    }
    try {
        Status s = self->runtime->initialize();
        if (!s.ok()) { raise_prime_error(s); return nullptr; }
    } catch (const std::exception& e) {
        PyErr_SetString(PrimeError, e.what());
        return nullptr;
    }
    Py_RETURN_NONE;
}

static PyObject* RuntimeObject_shutdown(RuntimeObject* self, PyObject* /*args*/) {
    if (!self->runtime) {
        PyErr_SetString(PrimeError, "Runtime object is in an invalid state");
        return nullptr;
    }
    try {
        Status s = self->runtime->shutdown();
        if (!s.ok()) { raise_prime_error(s); return nullptr; }
    } catch (const std::exception& e) {
        PyErr_SetString(PrimeError, e.what());
        return nullptr;
    }
    Py_RETURN_NONE;
}

static PyObject* RuntimeObject_status_string(RuntimeObject* self, PyObject* /*args*/) {
    if (!self->runtime) {
        PyErr_SetString(PrimeError, "Runtime object is in an invalid state");
        return nullptr;
    }
    try {
        std::string s = self->runtime->status_string();
        return PyUnicode_FromStringAndSize(s.data(), static_cast<Py_ssize_t>(s.size()));
    } catch (const std::exception& e) {
        PyErr_SetString(PrimeError, e.what());
        return nullptr;
    }
}

static PyObject* RuntimeObject_get_state(RuntimeObject* self, PyObject* /*args*/) {
    if (!self->runtime) {
        PyErr_SetString(PrimeError, "Runtime object is in an invalid state");
        return nullptr;
    }
    int state = static_cast<int>(self->runtime->state());
    return PyLong_FromLong(state);
}

static PyMethodDef RuntimeObject_methods[] = {
    {"initialize",    (PyCFunction)RuntimeObject_initialize,    METH_NOARGS, "Initialize the runtime."},
    {"shutdown",      (PyCFunction)RuntimeObject_shutdown,      METH_NOARGS, "Shut down the runtime."},
    {"status_string", (PyCFunction)RuntimeObject_status_string, METH_NOARGS, "Return a status summary string."},
    {"get_state",     (PyCFunction)RuntimeObject_get_state,     METH_NOARGS, "Return the runtime state as int."},
    {nullptr, nullptr, 0, nullptr}
};

static PyTypeObject RuntimeType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    /* tp_name */        "_prime._Runtime",
    /* tp_basicsize */   sizeof(RuntimeObject),
    /* tp_itemsize */    0,
    /* tp_dealloc */     (destructor)RuntimeObject_dealloc,
    /* tp_vectorcall_offset */ 0,
    /* tp_getattr */     nullptr,
    /* tp_setattr */     nullptr,
    /* tp_as_async */    nullptr,
    /* tp_repr */        nullptr,
    /* tp_as_number */   nullptr,
    /* tp_as_sequence */ nullptr,
    /* tp_as_mapping */  nullptr,
    /* tp_hash */        nullptr,
    /* tp_call */        nullptr,
    /* tp_str */         nullptr,
    /* tp_getattro */    nullptr,
    /* tp_setattro */    nullptr,
    /* tp_as_buffer */   nullptr,
    /* tp_flags */       Py_TPFLAGS_DEFAULT,
    /* tp_doc */         "Internal C++ Runtime wrapper.",
    /* tp_traverse */    nullptr,
    /* tp_clear */       nullptr,
    /* tp_richcompare */ nullptr,
    /* tp_weaklistoffset */ 0,
    /* tp_iter */        nullptr,
    /* tp_iternext */    nullptr,
    /* tp_methods */     RuntimeObject_methods,
    /* tp_members */     nullptr,
    /* tp_getset */      nullptr,
    /* tp_base */        nullptr,
    /* tp_dict */        nullptr,
    /* tp_descr_get */   nullptr,
    /* tp_descr_set */   nullptr,
    /* tp_dictoffset */  0,
    /* tp_init */        nullptr,
    /* tp_alloc */       nullptr,
    /* tp_new */         RuntimeObject_new,
};

// ---------------------------------------------------------------------------
// Module-level helpers
// ---------------------------------------------------------------------------
static PyObject* prime_version(PyObject* /*self*/, PyObject* /*args*/) {
    return PyUnicode_FromString("0.1.0.dev1");
}

static PyMethodDef prime_module_methods[] = {
    {"version", prime_version, METH_NOARGS, "Return the PRIME version string."},
    {nullptr, nullptr, 0, nullptr}
};

// ---------------------------------------------------------------------------
// Module constants (mirror C++ enums for Python)
// ---------------------------------------------------------------------------
static bool add_int_const(PyObject* module, const char* name, long value) {
    PyObject* obj = PyLong_FromLong(value);
    if (!obj) return false;
    int result = PyModule_AddObject(module, name, obj);
    if (result < 0) { Py_DECREF(obj); return false; }
    return true;
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------
static struct PyModuleDef prime_module_def = {
    PyModuleDef_HEAD_INIT,
    "_prime",
    "PRIME internal C++ extension module.",
    -1,
    prime_module_methods,
};

PyMODINIT_FUNC PyInit__prime(void) {
    // Finalize the RuntimeType before use.
    if (PyType_Ready(&RuntimeType) < 0) return nullptr;

    PyObject* module = PyModule_Create(&prime_module_def);
    if (!module) return nullptr;

    // Register the PrimeError exception.
    PrimeError = PyErr_NewExceptionWithDoc(
        "_prime.PrimeError",
        "Raised when a PRIME C++ operation fails.",
        nullptr, nullptr);
    if (!PrimeError) { Py_DECREF(module); return nullptr; }
    Py_INCREF(PrimeError);
    if (PyModule_AddObject(module, "PrimeError", PrimeError) < 0) {
        Py_DECREF(PrimeError);
        Py_DECREF(module);
        return nullptr;
    }

    // Register the Runtime type.
    Py_INCREF(&RuntimeType);
    if (PyModule_AddObject(module, "_Runtime", reinterpret_cast<PyObject*>(&RuntimeType)) < 0) {
        Py_DECREF(&RuntimeType);
        Py_DECREF(module);
        return nullptr;
    }

    // LogLevel constants.
    if (!add_int_const(module, "LOG_SILENT", 0)) { Py_DECREF(module); return nullptr; }
    if (!add_int_const(module, "LOG_ERROR",  1)) { Py_DECREF(module); return nullptr; }
    if (!add_int_const(module, "LOG_WARN",   2)) { Py_DECREF(module); return nullptr; }
    if (!add_int_const(module, "LOG_INFO",   3)) { Py_DECREF(module); return nullptr; }
    if (!add_int_const(module, "LOG_DEBUG",  4)) { Py_DECREF(module); return nullptr; }

    // RuntimeState constants.
    if (!add_int_const(module, "STATE_UNINITIALIZED", 0)) { Py_DECREF(module); return nullptr; }
    if (!add_int_const(module, "STATE_RUNNING",       1)) { Py_DECREF(module); return nullptr; }
    if (!add_int_const(module, "STATE_SHUTDOWN",      2)) { Py_DECREF(module); return nullptr; }

    return module;
}
