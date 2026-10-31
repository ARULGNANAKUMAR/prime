/*
 * memory_bindings.cpp — Python C API binding for the PRIME Memory Engine.
 *
 * Architecture:
 *   Python (prime/memory.py)
 *     ↓
 *   this file (_prime_memory C extension)
 *     ↓
 *   cpp/memory/memory_engine.hpp
 *
 * Exposed types:
 *   _prime_memory._MemoryEngine  — wraps MemoryEngine
 *   _prime_memory._MemoryItem    — wraps MemoryItem (read-only, returned by search/get)
 *   _prime_memory.MemoryError    — exception class
 *
 * Type/tier constants (mirror C++ enums):
 *   _prime_memory.MEMORY_WORKING, MEMORY_SHORT_TERM, MEMORY_LONG_TERM
 *   _prime_memory.EVICT_FIFO, EVICT_LRU, EVICT_IMPORTANCE_ASC
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "../memory/memory_engine.hpp"
#include "../memory/memory_types.hpp"

#include <memory>
#include <string>
#include <stdexcept>

using namespace prime;
using namespace prime::memory;

// ---------------------------------------------------------------------------
// Exception
// ---------------------------------------------------------------------------
static PyObject* MemoryError_type = nullptr;

static void set_memory_error(const char* msg) {
    PyErr_SetString(MemoryError_type, msg);
}

// ---------------------------------------------------------------------------
// _MemoryItem — thin Python wrapper for MemoryItem
// ---------------------------------------------------------------------------
struct PyMemoryItem {
    PyObject_HEAD
    MemoryItem item;
};

static void PyMemoryItem_dealloc(PyMemoryItem* self) {
    self->item.~MemoryItem();
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

static PyObject* PyMemoryItem_get_id(PyMemoryItem* self, void*) {
    return PyLong_FromUnsignedLongLong(self->item.id);
}
static PyObject* PyMemoryItem_get_type(PyMemoryItem* self, void*) {
    return PyUnicode_FromString(memory_type_name(self->item.type));
}
static PyObject* PyMemoryItem_get_content(PyMemoryItem* self, void*) {
    return PyUnicode_FromStringAndSize(self->item.content.data(),
                                       static_cast<Py_ssize_t>(self->item.content.size()));
}
static PyObject* PyMemoryItem_get_importance(PyMemoryItem* self, void*) {
    return PyFloat_FromDouble(self->item.importance);
}
static PyObject* PyMemoryItem_get_confidence(PyMemoryItem* self, void*) {
    return PyFloat_FromDouble(self->item.confidence);
}
static PyObject* PyMemoryItem_get_created_at(PyMemoryItem* self, void*) {
    return PyFloat_FromDouble(self->item.created_at);
}
static PyObject* PyMemoryItem_get_last_accessed(PyMemoryItem* self, void*) {
    return PyFloat_FromDouble(self->item.last_accessed);
}
static PyObject* PyMemoryItem_get_access_count(PyMemoryItem* self, void*) {
    return PyLong_FromUnsignedLongLong(self->item.access_count);
}
static PyObject* PyMemoryItem_get_expires_at(PyMemoryItem* self, void*) {
    return PyFloat_FromDouble(self->item.expires_at);
}
static PyObject* PyMemoryItem_get_metadata(PyMemoryItem* self, void*) {
    PyObject* d = PyDict_New();
    if (!d) return nullptr;
    for (auto& [k, v] : self->item.metadata) {
        PyObject* pk = PyUnicode_FromString(k.c_str());
        PyObject* pv = PyUnicode_FromString(v.c_str());
        if (!pk || !pv) { Py_XDECREF(pk); Py_XDECREF(pv); Py_DECREF(d); return nullptr; }
        if (PyDict_SetItem(d, pk, pv) < 0) {
            Py_DECREF(pk); Py_DECREF(pv); Py_DECREF(d); return nullptr;
        }
        Py_DECREF(pk); Py_DECREF(pv);
    }
    return d;
}

static PyGetSetDef PyMemoryItem_getset[] = {
    {(char*)"id",            (getter)PyMemoryItem_get_id,            nullptr, (char*)"Memory ID",          nullptr},
    {(char*)"type",          (getter)PyMemoryItem_get_type,          nullptr, (char*)"Memory tier",        nullptr},
    {(char*)"content",       (getter)PyMemoryItem_get_content,       nullptr, (char*)"Memory content",     nullptr},
    {(char*)"importance",    (getter)PyMemoryItem_get_importance,    nullptr, (char*)"Importance score",   nullptr},
    {(char*)"confidence",    (getter)PyMemoryItem_get_confidence,    nullptr, (char*)"Confidence score",   nullptr},
    {(char*)"created_at",    (getter)PyMemoryItem_get_created_at,    nullptr, (char*)"Creation timestamp", nullptr},
    {(char*)"last_accessed", (getter)PyMemoryItem_get_last_accessed, nullptr, (char*)"Last access time",   nullptr},
    {(char*)"access_count",  (getter)PyMemoryItem_get_access_count,  nullptr, (char*)"Access count",      nullptr},
    {(char*)"expires_at",    (getter)PyMemoryItem_get_expires_at,    nullptr, (char*)"Expiry timestamp",   nullptr},
    {(char*)"metadata",      (getter)PyMemoryItem_get_metadata,      nullptr, (char*)"Metadata dict",      nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr}
};

static PyObject* PyMemoryItem_repr(PyMemoryItem* self) {
    char buf[256];
    snprintf(buf, sizeof(buf), "<MemoryItem id=%llu type=%s importance=%.2f>",
             (unsigned long long)self->item.id,
             memory_type_name(self->item.type),
             self->item.importance);
    return PyUnicode_FromString(buf);
}

static PyTypeObject PyMemoryItemType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    /* tp_name */      "_prime_memory._MemoryItem",
    /* tp_basicsize */ sizeof(PyMemoryItem),
    /* tp_itemsize */  0,
    /* tp_dealloc */   (destructor)PyMemoryItem_dealloc,
    /* tp_vectorcall_offset */ 0,
    /* tp_getattr */   nullptr, /* tp_setattr */ nullptr,
    /* tp_as_async */  nullptr, /* tp_repr */ (reprfunc)PyMemoryItem_repr,
    /* tp_as_number */ nullptr, /* tp_as_sequence */ nullptr,
    /* tp_as_mapping */nullptr, /* tp_hash */ nullptr, /* tp_call */ nullptr,
    /* tp_str */       nullptr, /* tp_getattro */ nullptr,
    /* tp_setattro */  nullptr, /* tp_as_buffer */ nullptr,
    /* tp_flags */     Py_TPFLAGS_DEFAULT,
    /* tp_doc */       "A single PRIME memory item.",
    /* tp_traverse */  nullptr, /* tp_clear */ nullptr,
    /* tp_richcompare */nullptr, /* tp_weaklistoffset */ 0,
    /* tp_iter */      nullptr, /* tp_iternext */ nullptr,
    /* tp_methods */   nullptr,
    /* tp_members */   nullptr,
    /* tp_getset */    PyMemoryItem_getset,
};

// Factory: create a Python _MemoryItem from a C++ MemoryItem
static PyObject* make_py_item(const MemoryItem& item) {
    PyMemoryItem* obj = PyObject_New(PyMemoryItem, &PyMemoryItemType);
    if (!obj) return nullptr;
    new (&obj->item) MemoryItem(item);
    return reinterpret_cast<PyObject*>(obj);
}

// ---------------------------------------------------------------------------
// _MemoryEngine
// ---------------------------------------------------------------------------
struct PyMemoryEngine {
    PyObject_HEAD
    MemoryEngine* engine;  // owned
};

static void PyMemoryEngine_dealloc(PyMemoryEngine* self) {
    delete self->engine;
    self->engine = nullptr;
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

/*
 * __new__(db_path="", max_working=16, max_short_term=128,
 *         max_long_term=0, recency_half_life=86400.0,
 *         auto_consolidate=False, consolidation_threshold=0.8,
 *         eviction_policy="lru")
 */
static PyObject* PyMemoryEngine_new(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
    static const char* kwlist[] = {
        "db_path", "max_working", "max_short_term", "max_long_term",
        "recency_half_life", "auto_consolidate", "consolidation_threshold",
        "eviction_policy", nullptr
    };
    const char* db_path = "";
    int  max_working   = 16;
    int  max_short     = 128;
    int  max_long      = 0;
    double half_life   = 86400.0;
    int    auto_cons   = 0;
    double cons_thresh = 0.8;
    const char* evict_str = "lru";

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|siiidpds",
                                      const_cast<char**>(kwlist),
                                      &db_path, &max_working, &max_short,
                                      &max_long, &half_life, &auto_cons,
                                      &cons_thresh, &evict_str)) {
        return nullptr;
    }

    MemoryConfig cfg;
    cfg.db_path                           = db_path;
    cfg.max_working                        = (max_working > 0) ? static_cast<std::size_t>(max_working) : 0;
    cfg.max_short_term                     = (max_short > 0)   ? static_cast<std::size_t>(max_short)   : 0;
    cfg.max_long_term                      = (max_long > 0)    ? static_cast<std::size_t>(max_long)    : 0;
    cfg.recency_half_life_seconds          = half_life;
    cfg.auto_consolidate                   = (auto_cons != 0);
    cfg.consolidation_importance_threshold = cons_thresh;

    // Parse eviction policy
    try {
        EvictionPolicy ep = eviction_policy_from_string(evict_str);
        cfg.working_eviction    = ep;
        cfg.short_term_eviction = ep;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return nullptr;
    }

    PyMemoryEngine* self = reinterpret_cast<PyMemoryEngine*>(type->tp_alloc(type, 0));
    if (!self) return nullptr;

    try {
        self->engine = new MemoryEngine(std::move(cfg));
        Status s = self->engine->initialize();
        if (!s.ok()) {
            delete self->engine;
            self->engine = nullptr;
            Py_DECREF(self);
            set_memory_error(s.message.c_str());
            return nullptr;
        }
    } catch (const std::exception& e) {
        Py_DECREF(self);
        set_memory_error(e.what());
        return nullptr;
    }

    return reinterpret_cast<PyObject*>(self);
}

// Helper: extract string → MemoryType
static bool parse_memory_type(const char* s, MemoryType& out) {
    try { out = memory_type_from_string(s); return true; }
    catch (...) { PyErr_Format(PyExc_ValueError, "Unknown memory type: %s", s); return false; }
}

// Helper: Python dict → Metadata
static bool dict_to_metadata(PyObject* d, Metadata& out) {
    if (!d || d == Py_None) return true;
    if (!PyDict_Check(d)) { PyErr_SetString(PyExc_TypeError, "metadata must be a dict"); return false; }
    PyObject* k; PyObject* v; Py_ssize_t pos = 0;
    while (PyDict_Next(d, &pos, &k, &v)) {
        const char* ks = PyUnicode_AsUTF8(k);
        const char* vs = PyUnicode_AsUTF8(v);
        if (!ks || !vs) return false;
        out[ks] = vs;
    }
    return true;
}

/*
 * store(content, type="long_term", importance=0.5, confidence=1.0,
 *       metadata=None, expires_at=0.0) -> int
 */
static PyObject* PyMemoryEngine_store(PyMemoryEngine* self, PyObject* args, PyObject* kwargs) {
    static const char* kwlist[] = {
        "content", "type", "importance", "confidence",
        "metadata", "expires_at", nullptr
    };
    const char* content     = nullptr;
    const char* type_str    = "long_term";
    double      importance  = 0.5;
    double      confidence  = 1.0;
    PyObject*   meta_obj    = nullptr;
    double      expires_at  = 0.0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|sddOd",
                                      const_cast<char**>(kwlist),
                                      &content, &type_str, &importance, &confidence,
                                      &meta_obj, &expires_at)) {
        return nullptr;
    }

    MemoryType mtype;
    if (!parse_memory_type(type_str, mtype)) return nullptr;

    Metadata metadata;
    if (!dict_to_metadata(meta_obj, metadata)) return nullptr;

    try {
        uint64_t id = self->engine->store(content, mtype, importance, confidence,
                                           metadata, expires_at);
        return PyLong_FromUnsignedLongLong(id);
    } catch (const std::exception& e) {
        set_memory_error(e.what());
        return nullptr;
    }
}

/*
 * get(memory_id) -> _MemoryItem
 */
static PyObject* PyMemoryEngine_get(PyMemoryEngine* self, PyObject* args) {
    unsigned long long id_ull = 0;
    if (!PyArg_ParseTuple(args, "K", &id_ull)) return nullptr;
    try {
        MemoryItem item = self->engine->get(static_cast<uint64_t>(id_ull));
        return make_py_item(item);
    } catch (const std::exception& e) {
        set_memory_error(e.what());
        return nullptr;
    }
}

/*
 * update(memory_id, content=None, importance=None, confidence=None,
 *         metadata=None, type=None, expires_at=None) -> None
 */
static PyObject* PyMemoryEngine_update(PyMemoryEngine* self, PyObject* args, PyObject* kwargs) {
    static const char* kwlist[] = {
        "memory_id", "content", "importance", "confidence",
        "metadata", "type", "expires_at", nullptr
    };
    unsigned long long id_ull = 0;
    PyObject* content_obj   = Py_None;
    PyObject* imp_obj        = Py_None;
    PyObject* conf_obj       = Py_None;
    PyObject* meta_obj       = Py_None;
    PyObject* type_obj       = Py_None;
    PyObject* exp_obj        = Py_None;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "K|OOOOOO",
                                      const_cast<char**>(kwlist),
                                      &id_ull,
                                      &content_obj, &imp_obj, &conf_obj,
                                      &meta_obj, &type_obj, &exp_obj)) {
        return nullptr;
    }

    std::optional<std::string>  content;
    std::optional<double>       importance;
    std::optional<double>       confidence;
    std::optional<Metadata>     metadata;
    std::optional<MemoryType>   new_type;
    std::optional<double>       expires_at;

    if (content_obj && content_obj != Py_None) {
        const char* s = PyUnicode_AsUTF8(content_obj);
        if (!s) return nullptr;
        content = s;
    }
    if (imp_obj && imp_obj != Py_None) {
        double v = PyFloat_AsDouble(imp_obj);
        if (v == -1.0 && PyErr_Occurred()) return nullptr;
        importance = v;
    }
    if (conf_obj && conf_obj != Py_None) {
        double v = PyFloat_AsDouble(conf_obj);
        if (v == -1.0 && PyErr_Occurred()) return nullptr;
        confidence = v;
    }
    if (meta_obj && meta_obj != Py_None) {
        Metadata m;
        if (!dict_to_metadata(meta_obj, m)) return nullptr;
        metadata = m;
    }
    if (type_obj && type_obj != Py_None) {
        const char* s = PyUnicode_AsUTF8(type_obj);
        if (!s) return nullptr;
        MemoryType mt;
        if (!parse_memory_type(s, mt)) return nullptr;
        new_type = mt;
    }
    if (exp_obj && exp_obj != Py_None) {
        double v = PyFloat_AsDouble(exp_obj);
        if (v == -1.0 && PyErr_Occurred()) return nullptr;
        expires_at = v;
    }

    try {
        self->engine->update(static_cast<uint64_t>(id_ull),
                             content, importance, confidence,
                             metadata, new_type, expires_at);
        Py_RETURN_NONE;
    } catch (const std::exception& e) {
        set_memory_error(e.what());
        return nullptr;
    }
}

/*
 * delete(memory_id) -> None
 */
static PyObject* PyMemoryEngine_delete(PyMemoryEngine* self, PyObject* args) {
    unsigned long long id_ull = 0;
    if (!PyArg_ParseTuple(args, "K", &id_ull)) return nullptr;
    try {
        self->engine->delete_memory(static_cast<uint64_t>(id_ull));
        Py_RETURN_NONE;
    } catch (const std::exception& e) {
        set_memory_error(e.what());
        return nullptr;
    }
}

/*
 * search(query, limit=10, type=None, metadata=None) -> list[_MemoryItem]
 */
static PyObject* PyMemoryEngine_search(PyMemoryEngine* self, PyObject* args, PyObject* kwargs) {
    static const char* kwlist[] = {"query", "limit", "type", "metadata", nullptr};
    const char* query    = "";
    int         limit    = 10;
    PyObject*   type_obj = Py_None;
    PyObject*   meta_obj = Py_None;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|siOO",
                                      const_cast<char**>(kwlist),
                                      &query, &limit, &type_obj, &meta_obj)) {
        return nullptr;
    }

    std::optional<MemoryType> filter_type;
    if (type_obj && type_obj != Py_None) {
        const char* s = PyUnicode_AsUTF8(type_obj);
        if (!s) return nullptr;
        MemoryType mt;
        if (!parse_memory_type(s, mt)) return nullptr;
        filter_type = mt;
    }

    Metadata filter_meta;
    if (!dict_to_metadata(meta_obj, filter_meta)) return nullptr;

    try {
        auto results = self->engine->search(
            query,
            static_cast<std::size_t>(limit > 0 ? limit : 10),
            filter_type,
            filter_meta);

        PyObject* list = PyList_New(static_cast<Py_ssize_t>(results.size()));
        if (!list) return nullptr;
        for (std::size_t i = 0; i < results.size(); ++i) {
            PyObject* item = make_py_item(results[i]);
            if (!item) { Py_DECREF(list); return nullptr; }
            PyList_SET_ITEM(list, static_cast<Py_ssize_t>(i), item);
        }
        return list;
    } catch (const std::exception& e) {
        set_memory_error(e.what());
        return nullptr;
    }
}

/*
 * retrieve_context(query, limit=5) -> list[_MemoryItem]
 */
static PyObject* PyMemoryEngine_retrieve_context(PyMemoryEngine* self, PyObject* args, PyObject* kwargs) {
    static const char* kwlist[] = {"query", "limit", nullptr};
    const char* query = "";
    int limit = 5;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|i",
                                      const_cast<char**>(kwlist), &query, &limit)) {
        return nullptr;
    }
    try {
        auto results = self->engine->retrieve_context(query, static_cast<std::size_t>(limit));
        PyObject* list = PyList_New(static_cast<Py_ssize_t>(results.size()));
        if (!list) return nullptr;
        for (std::size_t i = 0; i < results.size(); ++i) {
            PyObject* item = make_py_item(results[i]);
            if (!item) { Py_DECREF(list); return nullptr; }
            PyList_SET_ITEM(list, static_cast<Py_ssize_t>(i), item);
        }
        return list;
    } catch (const std::exception& e) {
        set_memory_error(e.what());
        return nullptr;
    }
}

// count(type=None) -> int
static PyObject* PyMemoryEngine_count(PyMemoryEngine* self, PyObject* args, PyObject* kwargs) {
    static const char* kwlist[] = {"type", nullptr};
    PyObject* type_obj = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O",
                                      const_cast<char**>(kwlist), &type_obj)) {
        return nullptr;
    }
    std::optional<MemoryType> ft;
    if (type_obj && type_obj != Py_None) {
        const char* s = PyUnicode_AsUTF8(type_obj);
        if (!s) return nullptr;
        MemoryType mt; if (!parse_memory_type(s, mt)) return nullptr;
        ft = mt;
    }
    try {
        return PyLong_FromSsize_t(static_cast<Py_ssize_t>(self->engine->count(ft)));
    } catch (const std::exception& e) { set_memory_error(e.what()); return nullptr; }
}

// exists(memory_id) -> bool
static PyObject* PyMemoryEngine_exists(PyMemoryEngine* self, PyObject* args) {
    unsigned long long id_ull = 0;
    if (!PyArg_ParseTuple(args, "K", &id_ull)) return nullptr;
    try {
        return PyBool_FromLong(self->engine->exists(static_cast<uint64_t>(id_ull)) ? 1 : 0);
    } catch (const std::exception& e) { set_memory_error(e.what()); return nullptr; }
}

// clear(type=None) -> None
static PyObject* PyMemoryEngine_clear(PyMemoryEngine* self, PyObject* args, PyObject* kwargs) {
    static const char* kwlist[] = {"type", nullptr};
    PyObject* type_obj = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O",
                                      const_cast<char**>(kwlist), &type_obj)) return nullptr;
    std::optional<MemoryType> ft;
    if (type_obj && type_obj != Py_None) {
        const char* s = PyUnicode_AsUTF8(type_obj);
        if (!s) return nullptr;
        MemoryType mt; if (!parse_memory_type(s, mt)) return nullptr;
        ft = mt;
    }
    try { self->engine->clear(ft); Py_RETURN_NONE; }
    catch (const std::exception& e) { set_memory_error(e.what()); return nullptr; }
}

// expire_all() -> int  (number of items removed)
static PyObject* PyMemoryEngine_expire_all(PyMemoryEngine* self, PyObject*) {
    try {
        std::size_t n = self->engine->expire_all();
        return PyLong_FromSsize_t(static_cast<Py_ssize_t>(n));
    } catch (const std::exception& e) { set_memory_error(e.what()); return nullptr; }
}

// consolidate(importance_threshold=-1.0) -> int
static PyObject* PyMemoryEngine_consolidate(PyMemoryEngine* self, PyObject* args, PyObject* kwargs) {
    static const char* kwlist[] = {"importance_threshold", nullptr};
    double thresh = -1.0;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|d",
                                      const_cast<char**>(kwlist), &thresh)) return nullptr;
    try {
        std::size_t n = self->engine->consolidate(thresh);
        return PyLong_FromSsize_t(static_cast<Py_ssize_t>(n));
    } catch (const std::exception& e) { set_memory_error(e.what()); return nullptr; }
}

// working_add(content, importance=0.5, metadata=None) -> int
static PyObject* PyMemoryEngine_working_add(PyMemoryEngine* self, PyObject* args, PyObject* kwargs) {
    static const char* kwlist[] = {"content", "importance", "metadata", nullptr};
    const char* content = nullptr;
    double importance = 0.5;
    PyObject* meta_obj = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|dO",
                                      const_cast<char**>(kwlist),
                                      &content, &importance, &meta_obj)) return nullptr;
    Metadata metadata;
    if (!dict_to_metadata(meta_obj, metadata)) return nullptr;
    try {
        uint64_t id = self->engine->working_add(content, importance, metadata);
        return PyLong_FromUnsignedLongLong(id);
    } catch (const std::exception& e) { set_memory_error(e.what()); return nullptr; }
}

// working_clear() -> None
static PyObject* PyMemoryEngine_working_clear(PyMemoryEngine* self, PyObject*) {
    try { self->engine->working_clear(); Py_RETURN_NONE; }
    catch (const std::exception& e) { set_memory_error(e.what()); return nullptr; }
}

static PyMethodDef PyMemoryEngine_methods[] = {
    {"store",            (PyCFunction)(void*)PyMemoryEngine_store,            METH_VARARGS|METH_KEYWORDS, "Store a new memory item."},
    {"get",              (PyCFunction)PyMemoryEngine_get,                     METH_VARARGS,               "Get memory item by ID."},
    {"update",           (PyCFunction)(void*)PyMemoryEngine_update,           METH_VARARGS|METH_KEYWORDS, "Update memory item."},
    {"delete",           (PyCFunction)PyMemoryEngine_delete,                  METH_VARARGS,               "Delete memory item by ID."},
    {"search",           (PyCFunction)(void*)PyMemoryEngine_search,           METH_VARARGS|METH_KEYWORDS, "Search memories."},
    {"retrieve_context", (PyCFunction)(void*)PyMemoryEngine_retrieve_context, METH_VARARGS|METH_KEYWORDS, "Retrieve context for a query."},
    {"count",            (PyCFunction)(void*)PyMemoryEngine_count,            METH_VARARGS|METH_KEYWORDS, "Count memories."},
    {"exists",           (PyCFunction)PyMemoryEngine_exists,                  METH_VARARGS,               "Check if memory ID exists."},
    {"clear",            (PyCFunction)(void*)PyMemoryEngine_clear,            METH_VARARGS|METH_KEYWORDS, "Clear memories."},
    {"expire_all",       (PyCFunction)PyMemoryEngine_expire_all,              METH_NOARGS,                "Expire all expired memories."},
    {"consolidate",      (PyCFunction)(void*)PyMemoryEngine_consolidate,      METH_VARARGS|METH_KEYWORDS, "Promote STM items to LTM."},
    {"working_add",      (PyCFunction)(void*)PyMemoryEngine_working_add,      METH_VARARGS|METH_KEYWORDS, "Add item to working memory."},
    {"working_clear",    (PyCFunction)PyMemoryEngine_working_clear,           METH_NOARGS,                "Clear working memory."},
    {nullptr, nullptr, 0, nullptr}
};

static PyTypeObject PyMemoryEngineType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    /* tp_name */      "_prime_memory._MemoryEngine",
    /* tp_basicsize */ sizeof(PyMemoryEngine),
    /* tp_itemsize */  0,
    /* tp_dealloc */   (destructor)PyMemoryEngine_dealloc,
    /* tp_vectorcall_offset */ 0,
    /* tp_getattr */   nullptr, /* tp_setattr */ nullptr,
    /* tp_as_async */  nullptr, /* tp_repr */    nullptr,
    /* tp_as_number */ nullptr, /* tp_as_sequence */ nullptr,
    /* tp_as_mapping */nullptr, /* tp_hash */    nullptr,
    /* tp_call */      nullptr, /* tp_str */     nullptr,
    /* tp_getattro */  nullptr, /* tp_setattro */nullptr,
    /* tp_as_buffer */ nullptr,
    /* tp_flags */     Py_TPFLAGS_DEFAULT,
    /* tp_doc */       "PRIME Memory Engine.",
    /* tp_traverse */  nullptr, /* tp_clear */   nullptr,
    /* tp_richcompare */nullptr, /* tp_weaklistoffset */ 0,
    /* tp_iter */      nullptr, /* tp_iternext */nullptr,
    /* tp_methods */   PyMemoryEngine_methods,
    /* tp_members */   nullptr, /* tp_getset */  nullptr,
    /* tp_base */      nullptr, /* tp_dict */    nullptr,
    /* tp_descr_get */ nullptr, /* tp_descr_set */nullptr,
    /* tp_dictoffset */0, /* tp_init */          nullptr,
    /* tp_alloc */     nullptr, /* tp_new */     PyMemoryEngine_new,
};

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------
static PyMethodDef module_methods[] = {
    {nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef memory_module_def = {
    PyModuleDef_HEAD_INIT,
    "_prime_memory",
    "PRIME Memory Engine C++ extension (Phase 3).",
    -1,
    module_methods,
};

static bool add_int_const(PyObject* m, const char* name, long v) {
    PyObject* obj = PyLong_FromLong(v);
    if (!obj) return false;
    int rc = PyModule_AddObject(m, name, obj);
    if (rc < 0) { Py_DECREF(obj); return false; }
    return true;
}

PyMODINIT_FUNC PyInit__prime_memory(void) {
    if (PyType_Ready(&PyMemoryItemType) < 0)   return nullptr;
    if (PyType_Ready(&PyMemoryEngineType) < 0) return nullptr;

    PyObject* m = PyModule_Create(&memory_module_def);
    if (!m) return nullptr;

    // Exception
    MemoryError_type = PyErr_NewExceptionWithDoc(
        "_prime_memory.MemoryError",
        "Raised when a PRIME memory operation fails.",
        nullptr, nullptr);
    if (!MemoryError_type) { Py_DECREF(m); return nullptr; }
    Py_INCREF(MemoryError_type);
    if (PyModule_AddObject(m, "MemoryError", MemoryError_type) < 0) {
        Py_DECREF(MemoryError_type); Py_DECREF(m); return nullptr;
    }

    // Types
    Py_INCREF(&PyMemoryItemType);
    if (PyModule_AddObject(m, "_MemoryItem",
                            reinterpret_cast<PyObject*>(&PyMemoryItemType)) < 0) {
        Py_DECREF(&PyMemoryItemType); Py_DECREF(m); return nullptr;
    }
    Py_INCREF(&PyMemoryEngineType);
    if (PyModule_AddObject(m, "_MemoryEngine",
                            reinterpret_cast<PyObject*>(&PyMemoryEngineType)) < 0) {
        Py_DECREF(&PyMemoryEngineType); Py_DECREF(m); return nullptr;
    }

    // Memory tier constants
    if (!add_int_const(m, "MEMORY_WORKING",    0)) { Py_DECREF(m); return nullptr; }
    if (!add_int_const(m, "MEMORY_SHORT_TERM", 1)) { Py_DECREF(m); return nullptr; }
    if (!add_int_const(m, "MEMORY_LONG_TERM",  2)) { Py_DECREF(m); return nullptr; }

    // Eviction policy constants
    if (!add_int_const(m, "EVICT_FIFO",           0)) { Py_DECREF(m); return nullptr; }
    if (!add_int_const(m, "EVICT_LRU",            1)) { Py_DECREF(m); return nullptr; }
    if (!add_int_const(m, "EVICT_IMPORTANCE_ASC", 2)) { Py_DECREF(m); return nullptr; }

    return m;
}
