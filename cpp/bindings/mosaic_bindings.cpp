// mosaic_bindings.cpp — Python C API bindings for PRIME Phase 6 MOSAIC.
//
// Exposes:
//   _prime_mosaic.MOSAICEngine   — module registration, planning, explanation
//   _prime_mosaic.MOSAICError    — exception class
//   Free functions:
//     create_engine(config_dict)
//     register_module(engine_capsule, module_dict)
//     compose(engine_capsule, requirements_dict, config_dict?)
//     validate_plan(engine_capsule, plan_dict)
//     explain_plan(engine_capsule, plan_dict)
//     explain_trace(engine_capsule, trace_dict)
//     set_module_status(engine_capsule, id, status)
//     list_modules(engine_capsule)
//     remove_module(engine_capsule, id)
//     analyze_task(engine_capsule, task_dict)
//     get_module(engine_capsule, id)

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "mosaic/mosaic_types.hpp"
#include "mosaic/module_registry.hpp"
#include "mosaic/task_analyzer.hpp"
#include "mosaic/composition_planner.hpp"
#include "mosaic/mosaic_engine.hpp"

#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------
static PyObject* MOSAICError = nullptr;

// ---------------------------------------------------------------------------
// Engine capsule helpers
// ---------------------------------------------------------------------------
static const char* CAPSULE_NAME = "prime.mosaic.engine";

static void engine_destructor(PyObject* cap) {
    auto* e = static_cast<prime::mosaic::MOSAICEngine*>(PyCapsule_GetPointer(cap, CAPSULE_NAME));
    delete e;
}

static prime::mosaic::MOSAICEngine* get_engine(PyObject* cap) {
    return static_cast<prime::mosaic::MOSAICEngine*>(
        PyCapsule_GetPointer(cap, CAPSULE_NAME));
}

// ---------------------------------------------------------------------------
// Python dict helpers
// ---------------------------------------------------------------------------
static std::string py_str(PyObject* d, const char* k, const char* dflt = "") {
    PyObject* v = PyDict_GetItemString(d, k);
    if (!v || !PyUnicode_Check(v)) return dflt;
    return PyUnicode_AsUTF8(v);
}

static int py_int(PyObject* d, const char* k, int dflt = 0) {
    PyObject* v = PyDict_GetItemString(d, k);
    if (!v) return dflt;
    if (PyLong_Check(v))  return (int)PyLong_AsLong(v);
    if (PyFloat_Check(v)) return (int)PyFloat_AsDouble(v);
    return dflt;
}

static double py_dbl(PyObject* d, const char* k, double dflt = 0.0) {
    PyObject* v = PyDict_GetItemString(d, k);
    if (!v) return dflt;
    if (PyFloat_Check(v)) return PyFloat_AsDouble(v);
    if (PyLong_Check(v))  return (double)PyLong_AsLong(v);
    return dflt;
}

static bool py_bool(PyObject* d, const char* k, bool dflt = false) {
    PyObject* v = PyDict_GetItemString(d, k);
    if (!v) return dflt;
    return PyObject_IsTrue(v) == 1;
}

static std::vector<std::string> py_str_list(PyObject* d, const char* k) {
    std::vector<std::string> out;
    PyObject* v = PyDict_GetItemString(d, k);
    if (!v || !PyList_Check(v)) return out;
    for (Py_ssize_t i = 0; i < PyList_Size(v); ++i) {
        PyObject* item = PyList_GetItem(v, i);
        if (item && PyUnicode_Check(item))
            out.push_back(PyUnicode_AsUTF8(item));
    }
    return out;
}

// ---------------------------------------------------------------------------
// Config parsing
// ---------------------------------------------------------------------------
static prime::mosaic::MOSAICConfig parse_config(PyObject* d) {
    prime::mosaic::MOSAICConfig cfg;
    if (!d || !PyDict_Check(d)) return cfg;
    cfg.max_modules         = py_int(d, "max_modules",         10);
    cfg.max_execution_steps = py_int(d, "max_execution_steps", 50);
    cfg.max_latency_ms      = py_dbl(d, "max_latency_ms",   30000.0);
    cfg.max_retries         = py_int(d, "max_retries",          1);
    cfg.allow_llm           = py_bool(d, "allow_llm",      false);
    cfg.allow_fallback      = py_bool(d, "allow_fallback",   true);
    cfg.prefer_lightweight  = py_bool(d, "prefer_lightweight",true);
    cfg.use_cache           = py_bool(d, "use_cache",        true);
    cfg.max_total_cost      = py_dbl(d, "max_total_cost",   100.0);
    cfg.preferred_modules   = py_str_list(d, "preferred_modules");
    cfg.excluded_modules    = py_str_list(d, "excluded_modules");
    return cfg;
}

// ---------------------------------------------------------------------------
// Module metadata parsing
// ---------------------------------------------------------------------------
static prime::mosaic::Capability parse_capability(PyObject* d) {
    prime::mosaic::Capability cap;
    cap.name        = py_str(d, "name");
    cap.description = py_str(d, "description");

    std::string cat = py_str(d, "category", "general");
    if      (cat == "representation")      cap.category = prime::mosaic::CapabilityCategory::REPRESENTATION;
    else if (cat == "retrieval")           cap.category = prime::mosaic::CapabilityCategory::RETRIEVAL;
    else if (cat == "arithmetic")          cap.category = prime::mosaic::CapabilityCategory::ARITHMETIC;
    else if (cat == "rule_reasoning")      cap.category = prime::mosaic::CapabilityCategory::RULE_REASONING;
    else if (cat == "symbolic_reasoning")  cap.category = prime::mosaic::CapabilityCategory::SYMBOLIC_REASONING;
    else if (cat == "constraint_checking") cap.category = prime::mosaic::CapabilityCategory::CONSTRAINT_CHECKING;
    else if (cat == "language")            cap.category = prime::mosaic::CapabilityCategory::LANGUAGE;
    else if (cat == "validation")          cap.category = prime::mosaic::CapabilityCategory::VALIDATION;
    else if (cat == "storage")             cap.category = prime::mosaic::CapabilityCategory::STORAGE;
    else                                    cap.category = prime::mosaic::CapabilityCategory::GENERAL;

    cap.input_types  = py_str_list(d, "input_types");
    cap.output_types = py_str_list(d, "output_types");
    cap.confidence   = py_dbl(d, "confidence", 1.0);
    return cap;
}

static prime::mosaic::ModuleMetadata parse_module_meta(PyObject* d) {
    prime::mosaic::ModuleMetadata m;
    m.id      = py_str(d, "id");
    m.name    = py_str(d, "name");
    m.version = py_str(d, "version", "1.0");
    m.priority     = py_int(d, "priority");
    m.deterministic= py_bool(d, "deterministic", true);
    m.is_fallback  = py_bool(d, "is_fallback",   false);
    m.fallback_for = py_str(d, "fallback_for");
    m.input_types  = py_str_list(d, "input_types");
    m.output_types = py_str_list(d, "output_types");
    m.dependencies = py_str_list(d, "dependencies");

    // Resource cost
    PyObject* cost = PyDict_GetItemString(d, "cost");
    if (cost && PyDict_Check(cost)) {
        m.cost.cpu_relative    = py_dbl(cost, "cpu_relative",    1.0);
        m.cost.memory_relative = py_dbl(cost, "memory_relative", 1.0);
        m.cost.latency_ms      = py_dbl(cost, "latency_ms",      0.0);
        m.cost.requires_model  = py_bool(cost, "requires_model", false);
    }

    // Capabilities
    PyObject* caps = PyDict_GetItemString(d, "capabilities");
    if (caps && PyList_Check(caps)) {
        for (Py_ssize_t i = 0; i < PyList_Size(caps); ++i) {
            PyObject* c = PyList_GetItem(caps, i);
            if (c && PyDict_Check(c))
                m.capabilities.push_back(parse_capability(c));
        }
    }

    // Status
    std::string status_str = py_str(d, "status", "available");
    if      (status_str == "unavailable") m.status = prime::mosaic::ModuleStatus::UNAVAILABLE;
    else if (status_str == "degraded")    m.status = prime::mosaic::ModuleStatus::DEGRADED;
    else if (status_str == "disabled")    m.status = prime::mosaic::ModuleStatus::DISABLED;
    else                                   m.status = prime::mosaic::ModuleStatus::AVAILABLE;

    return m;
}

// ---------------------------------------------------------------------------
// Build Python dict from CompositionPlan
// ---------------------------------------------------------------------------
static PyObject* plan_to_pydict(const prime::mosaic::CompositionPlan& plan) {
    PyObject* d = PyDict_New();

    PyDict_SetItemString(d, "task_id",
        PyUnicode_FromString(plan.task_id.c_str()));
    PyDict_SetItemString(d, "domain",
        PyUnicode_FromString(plan.domain.c_str()));
    PyDict_SetItemString(d, "status",
        PyUnicode_FromString(prime::mosaic::composition_status_to_string(plan.status).c_str()));
    PyDict_SetItemString(d, "valid",
        PyBool_FromLong(plan.valid()));
    PyDict_SetItemString(d, "status_detail",
        PyUnicode_FromString(plan.status_detail.c_str()));
    PyDict_SetItemString(d, "estimated_cost",
        PyFloat_FromDouble(plan.estimated_cost));

    // module_ids
    PyObject* mids = PyList_New(0);
    for (auto& id : plan.module_ids)
        PyList_Append(mids, PyUnicode_FromString(id.c_str()));
    PyDict_SetItemString(d, "module_ids", mids);
    Py_DECREF(mids);

    // selection_reasons
    PyObject* reasons = PyList_New(0);
    for (auto& r : plan.selection_reasons)
        PyList_Append(reasons, PyUnicode_FromString(r.c_str()));
    PyDict_SetItemString(d, "selection_reasons", reasons);
    Py_DECREF(reasons);

    // requirements summary
    PyObject* req = PyDict_New();
    PyDict_SetItemString(req, "domain",
        PyUnicode_FromString(plan.requirements.domain.c_str()));

    auto strvec_to_pylist = [](const std::vector<std::string>& v) {
        PyObject* l = PyList_New(0);
        for (auto& s : v) PyList_Append(l, PyUnicode_FromString(s.c_str()));
        return l;
    };

    PyObject* rcaps = strvec_to_pylist(plan.requirements.required_capabilities);
    PyDict_SetItemString(req, "required_capabilities", rcaps); Py_DECREF(rcaps);
    PyObject* ocaps = strvec_to_pylist(plan.requirements.optional_capabilities);
    PyDict_SetItemString(req, "optional_capabilities", ocaps); Py_DECREF(ocaps);
    PyDict_SetItemString(req, "needs_memory", PyBool_FromLong(plan.requirements.needs_memory));
    PyDict_SetItemString(req, "needs_llm",    PyBool_FromLong(plan.requirements.needs_llm));
    PyDict_SetItemString(req, "has_arithmetic",PyBool_FromLong(plan.requirements.has_arithmetic));
    PyDict_SetItemString(req, "has_rules",    PyBool_FromLong(plan.requirements.has_rules));
    PyDict_SetItemString(req, "has_constraints",PyBool_FromLong(plan.requirements.has_constraints));
    PyDict_SetItemString(d, "requirements", req);
    Py_DECREF(req);

    return d;
}

// ---------------------------------------------------------------------------
// Build Python dict from TaskRequirements
// ---------------------------------------------------------------------------
static PyObject* requirements_to_pydict(const prime::mosaic::TaskRequirements& req) {
    PyObject* d = PyDict_New();
    PyDict_SetItemString(d, "task_id", PyUnicode_FromString(req.task_id.c_str()));
    PyDict_SetItemString(d, "domain",  PyUnicode_FromString(req.domain.c_str()));
    PyDict_SetItemString(d, "needs_memory", PyBool_FromLong(req.needs_memory));
    PyDict_SetItemString(d, "needs_llm",    PyBool_FromLong(req.needs_llm));
    PyDict_SetItemString(d, "has_arithmetic",PyBool_FromLong(req.has_arithmetic));
    PyDict_SetItemString(d, "has_rules",    PyBool_FromLong(req.has_rules));
    PyDict_SetItemString(d, "has_constraints",PyBool_FromLong(req.has_constraints));

    auto strvec_to_pylist = [](const std::vector<std::string>& v) {
        PyObject* l = PyList_New(0);
        for (auto& s : v) PyList_Append(l, PyUnicode_FromString(s.c_str()));
        return l;
    };
    PyObject* rcaps = strvec_to_pylist(req.required_capabilities);
    PyDict_SetItemString(d, "required_capabilities", rcaps); Py_DECREF(rcaps);
    PyObject* ocaps = strvec_to_pylist(req.optional_capabilities);
    PyDict_SetItemString(d, "optional_capabilities", ocaps); Py_DECREF(ocaps);
    return d;
}

// ---------------------------------------------------------------------------
// Module dict builder
// ---------------------------------------------------------------------------
static PyObject* module_meta_to_pydict(const prime::mosaic::ModuleMetadata& m) {
    PyObject* d = PyDict_New();
    PyDict_SetItemString(d, "id",      PyUnicode_FromString(m.id.c_str()));
    PyDict_SetItemString(d, "name",    PyUnicode_FromString(m.name.c_str()));
    PyDict_SetItemString(d, "version", PyUnicode_FromString(m.version.c_str()));
    PyDict_SetItemString(d, "status",
        PyUnicode_FromString(prime::mosaic::module_status_to_string(m.status).c_str()));
    PyDict_SetItemString(d, "priority",     PyLong_FromLong(m.priority));
    PyDict_SetItemString(d, "deterministic",PyBool_FromLong(m.deterministic));
    PyDict_SetItemString(d, "is_fallback",  PyBool_FromLong(m.is_fallback));
    PyDict_SetItemString(d, "fallback_for", PyUnicode_FromString(m.fallback_for.c_str()));

    PyObject* caps = PyList_New(0);
    for (auto& c : m.capabilities) {
        PyObject* cd = PyDict_New();
        PyDict_SetItemString(cd, "name",     PyUnicode_FromString(c.name.c_str()));
        PyDict_SetItemString(cd, "category",
            PyUnicode_FromString(prime::mosaic::category_to_string(c.category).c_str()));
        PyDict_SetItemString(cd, "description", PyUnicode_FromString(c.description.c_str()));
        PyList_Append(caps, cd); Py_DECREF(cd);
    }
    PyDict_SetItemString(d, "capabilities", caps); Py_DECREF(caps);

    auto strvec_to_pylist = [](const std::vector<std::string>& v) {
        PyObject* l = PyList_New(0);
        for (auto& s : v) PyList_Append(l, PyUnicode_FromString(s.c_str()));
        return l;
    };
    PyObject* it = strvec_to_pylist(m.input_types);
    PyDict_SetItemString(d, "input_types", it); Py_DECREF(it);
    PyObject* ot = strvec_to_pylist(m.output_types);
    PyDict_SetItemString(d, "output_types", ot); Py_DECREF(ot);
    PyObject* dt = strvec_to_pylist(m.dependencies);
    PyDict_SetItemString(d, "dependencies", dt); Py_DECREF(dt);

    PyObject* cost = PyDict_New();
    PyDict_SetItemString(cost, "cpu_relative",    PyFloat_FromDouble(m.cost.cpu_relative));
    PyDict_SetItemString(cost, "memory_relative", PyFloat_FromDouble(m.cost.memory_relative));
    PyDict_SetItemString(cost, "latency_ms",      PyFloat_FromDouble(m.cost.latency_ms));
    PyDict_SetItemString(cost, "requires_model",  PyBool_FromLong(m.cost.requires_model));
    PyDict_SetItemString(d, "cost", cost); Py_DECREF(cost);

    return d;
}

// ===========================================================================
// Exposed Python functions
// ===========================================================================

// create_engine(config_dict) → capsule
static PyObject* py_create_engine(PyObject*, PyObject* args) {
    PyObject* cfg_dict = nullptr;
    if (!PyArg_ParseTuple(args, "|O", &cfg_dict)) return nullptr;

    prime::mosaic::MOSAICConfig cfg;
    if (cfg_dict && PyDict_Check(cfg_dict)) cfg = parse_config(cfg_dict);

    auto* engine = new prime::mosaic::MOSAICEngine(cfg);
    return PyCapsule_New(engine, CAPSULE_NAME, engine_destructor);
}

// register_module(capsule, module_dict) → bool
static PyObject* py_register_module(PyObject*, PyObject* args) {
    PyObject* cap;
    PyObject* meta_dict;
    if (!PyArg_ParseTuple(args, "OO!", &cap, &PyDict_Type, &meta_dict))
        return nullptr;

    auto* engine = get_engine(cap);
    if (!engine) { PyErr_SetString(MOSAICError, "Invalid engine capsule"); return nullptr; }

    try {
        auto meta = parse_module_meta(meta_dict);
        bool ok   = engine->register_module(std::move(meta));
        return PyBool_FromLong(ok);
    } catch (const std::exception& e) {
        PyErr_SetString(MOSAICError, e.what());
        return nullptr;
    }
}

// update_module(capsule, module_dict) → None
static PyObject* py_update_module(PyObject*, PyObject* args) {
    PyObject* cap;
    PyObject* meta_dict;
    if (!PyArg_ParseTuple(args, "OO!", &cap, &PyDict_Type, &meta_dict))
        return nullptr;
    auto* engine = get_engine(cap);
    if (!engine) { PyErr_SetString(MOSAICError, "Invalid engine capsule"); return nullptr; }
    try {
        engine->update_module(parse_module_meta(meta_dict));
        Py_RETURN_NONE;
    } catch (const std::exception& e) {
        PyErr_SetString(MOSAICError, e.what()); return nullptr;
    }
}

// remove_module(capsule, id) → bool
static PyObject* py_remove_module(PyObject*, PyObject* args) {
    PyObject* cap;
    const char* id;
    if (!PyArg_ParseTuple(args, "Os", &cap, &id)) return nullptr;
    auto* engine = get_engine(cap);
    if (!engine) { PyErr_SetString(MOSAICError, "Invalid engine capsule"); return nullptr; }
    return PyBool_FromLong(engine->remove_module(id));
}

// set_module_status(capsule, id, status_str) → bool
static PyObject* py_set_module_status(PyObject*, PyObject* args) {
    PyObject*   cap;
    const char* id;
    const char* status_str;
    if (!PyArg_ParseTuple(args, "Oss", &cap, &id, &status_str)) return nullptr;
    auto* engine = get_engine(cap);
    if (!engine) { PyErr_SetString(MOSAICError, "Invalid engine capsule"); return nullptr; }

    prime::mosaic::ModuleStatus status = prime::mosaic::ModuleStatus::AVAILABLE;
    std::string ss(status_str);
    if      (ss == "unavailable") status = prime::mosaic::ModuleStatus::UNAVAILABLE;
    else if (ss == "degraded")    status = prime::mosaic::ModuleStatus::DEGRADED;
    else if (ss == "disabled")    status = prime::mosaic::ModuleStatus::DISABLED;

    return PyBool_FromLong(engine->set_module_status(id, status));
}

// list_modules(capsule) → list of str
static PyObject* py_list_modules(PyObject*, PyObject* args) {
    PyObject* cap;
    if (!PyArg_ParseTuple(args, "O", &cap)) return nullptr;
    auto* engine = get_engine(cap);
    if (!engine) { PyErr_SetString(MOSAICError, "Invalid engine capsule"); return nullptr; }

    auto ids = engine->list_modules();
    PyObject* lst = PyList_New(0);
    for (auto& id : ids) PyList_Append(lst, PyUnicode_FromString(id.c_str()));
    return lst;
}

// get_module(capsule, id) → dict or None
static PyObject* py_get_module(PyObject*, PyObject* args) {
    PyObject*   cap;
    const char* id;
    if (!PyArg_ParseTuple(args, "Os", &cap, &id)) return nullptr;
    auto* engine = get_engine(cap);
    if (!engine) { PyErr_SetString(MOSAICError, "Invalid engine capsule"); return nullptr; }

    auto* m = engine->get_module(id);
    if (!m) Py_RETURN_NONE;
    return module_meta_to_pydict(*m);
}

// analyze_task(capsule, task_dict) → requirements_dict
static PyObject* py_analyze_task(PyObject*, PyObject* args) {
    PyObject* cap;
    PyObject* task_dict;
    if (!PyArg_ParseTuple(args, "OO!", &cap, &PyDict_Type, &task_dict))
        return nullptr;
    auto* engine = get_engine(cap);
    if (!engine) { PyErr_SetString(MOSAICError, "Invalid engine capsule"); return nullptr; }

    try {
        auto ops  = py_str_list(task_dict, "operation_types");
        auto gts  = py_str_list(task_dict, "goal_targets");
        auto req  = engine->analyze_task(
            py_str(task_dict, "task_id"),
            py_str(task_dict, "domain"),
            py_bool(task_dict, "has_expressions"),
            py_bool(task_dict, "has_rules"),
            py_bool(task_dict, "has_constraints"),
            py_bool(task_dict, "has_goals"),
            py_bool(task_dict, "has_entities"),
            py_bool(task_dict, "has_logic_clauses"),
            py_bool(task_dict, "has_relationships"),
            ops, gts,
            py_int(task_dict, "entity_count"),
            py_bool(task_dict, "has_memory_context"));
        return requirements_to_pydict(req);
    } catch (const std::exception& e) {
        PyErr_SetString(MOSAICError, e.what()); return nullptr;
    }
}

// compose(capsule, requirements_dict, config_dict=None) → plan_dict
static PyObject* py_compose(PyObject*, PyObject* args) {
    PyObject* cap;
    PyObject* req_dict;
    PyObject* cfg_dict = nullptr;
    if (!PyArg_ParseTuple(args, "OO!|O", &cap, &PyDict_Type, &req_dict, &cfg_dict))
        return nullptr;

    auto* engine = get_engine(cap);
    if (!engine) { PyErr_SetString(MOSAICError, "Invalid engine capsule"); return nullptr; }

    try {
        // Parse requirements from dict
        prime::mosaic::TaskRequirements req;
        req.task_id  = py_str(req_dict, "task_id");
        req.domain   = py_str(req_dict, "domain");
        req.required_capabilities = py_str_list(req_dict, "required_capabilities");
        req.optional_capabilities = py_str_list(req_dict, "optional_capabilities");
        req.needs_memory    = py_bool(req_dict, "needs_memory");
        req.needs_llm       = py_bool(req_dict, "needs_llm");
        req.has_arithmetic  = py_bool(req_dict, "has_arithmetic");
        req.has_rules       = py_bool(req_dict, "has_rules");
        req.has_constraints = py_bool(req_dict, "has_constraints");

        prime::mosaic::CompositionPlan plan;
        if (cfg_dict && PyDict_Check(cfg_dict)) {
            auto cfg = parse_config(cfg_dict);
            plan = engine->compose(req, cfg);
        } else {
            plan = engine->compose(req);
        }
        return plan_to_pydict(plan);
    } catch (const std::exception& e) {
        PyErr_SetString(MOSAICError, e.what()); return nullptr;
    }
}

// explain_plan(capsule, plan_dict) → str
static PyObject* py_explain_plan(PyObject*, PyObject* args) {
    PyObject* cap;
    PyObject* plan_dict;
    if (!PyArg_ParseTuple(args, "OO!", &cap, &PyDict_Type, &plan_dict))
        return nullptr;
    auto* engine = get_engine(cap);
    if (!engine) { PyErr_SetString(MOSAICError, "Invalid engine capsule"); return nullptr; }

    // Rebuild plan from dict (lean — just needs module_ids and basic fields)
    prime::mosaic::CompositionPlan plan;
    plan.task_id    = py_str(plan_dict, "task_id");
    plan.domain     = py_str(plan_dict, "domain");
    plan.module_ids = py_str_list(plan_dict, "module_ids");
    plan.selection_reasons = py_str_list(plan_dict, "selection_reasons");
    plan.estimated_cost    = py_dbl(plan_dict, "estimated_cost");

    std::string status_str = py_str(plan_dict, "status", "VALID");
    if      (status_str == "VALID")               plan.status = prime::mosaic::CompositionStatus::VALID;
    else if (status_str == "COMPOSITION_CYCLE")   plan.status = prime::mosaic::CompositionStatus::COMPOSITION_CYCLE;
    else if (status_str == "MISSING_CAPABILITY")  plan.status = prime::mosaic::CompositionStatus::MISSING_CAPABILITY;
    else if (status_str == "INPUT_MISMATCH")      plan.status = prime::mosaic::CompositionStatus::INPUT_MISMATCH;
    else if (status_str == "RESOURCE_EXCEEDED")   plan.status = prime::mosaic::CompositionStatus::RESOURCE_EXCEEDED;
    else if (status_str == "EMPTY_PLAN")          plan.status = prime::mosaic::CompositionStatus::EMPTY_PLAN;
    else                                           plan.status = prime::mosaic::CompositionStatus::INVALID;

    plan.status_detail = py_str(plan_dict, "status_detail");

    // Requirements sub-dict
    PyObject* req_d = PyDict_GetItemString(plan_dict, "requirements");
    if (req_d && PyDict_Check(req_d)) {
        plan.requirements.domain = py_str(req_d, "domain");
        plan.requirements.required_capabilities = py_str_list(req_d, "required_capabilities");
        plan.requirements.optional_capabilities = py_str_list(req_d, "optional_capabilities");
    }

    return PyUnicode_FromString(engine->explain_plan(plan).c_str());
}

// validate_plan(capsule, plan_dict) → str (status string)
static PyObject* py_validate_plan(PyObject*, PyObject* args) {
    PyObject* cap;
    PyObject* plan_dict;
    if (!PyArg_ParseTuple(args, "OO!", &cap, &PyDict_Type, &plan_dict))
        return nullptr;
    auto* engine = get_engine(cap);
    if (!engine) { PyErr_SetString(MOSAICError, "Invalid engine capsule"); return nullptr; }

    prime::mosaic::CompositionPlan plan;
    plan.module_ids = py_str_list(plan_dict, "module_ids");
    auto status = engine->validate_plan(plan);
    return PyUnicode_FromString(prime::mosaic::composition_status_to_string(status).c_str());
}

// ---------------------------------------------------------------------------
// Method table
// ---------------------------------------------------------------------------
static PyMethodDef methods[] = {
    {"create_engine",      py_create_engine,      METH_VARARGS, "create_engine([config]) -> capsule"},
    {"register_module",    py_register_module,    METH_VARARGS, "register_module(capsule, meta_dict) -> bool"},
    {"update_module",      py_update_module,      METH_VARARGS, "update_module(capsule, meta_dict) -> None"},
    {"remove_module",      py_remove_module,      METH_VARARGS, "remove_module(capsule, id) -> bool"},
    {"set_module_status",  py_set_module_status,  METH_VARARGS, "set_module_status(capsule, id, status) -> bool"},
    {"list_modules",       py_list_modules,       METH_VARARGS, "list_modules(capsule) -> list"},
    {"get_module",         py_get_module,         METH_VARARGS, "get_module(capsule, id) -> dict|None"},
    {"analyze_task",       py_analyze_task,       METH_VARARGS, "analyze_task(capsule, task_dict) -> requirements_dict"},
    {"compose",            py_compose,            METH_VARARGS, "compose(capsule, req_dict[, cfg_dict]) -> plan_dict"},
    {"explain_plan",       py_explain_plan,       METH_VARARGS, "explain_plan(capsule, plan_dict) -> str"},
    {"validate_plan",      py_validate_plan,      METH_VARARGS, "validate_plan(capsule, plan_dict) -> str"},
    {nullptr, nullptr, 0, nullptr},
};

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT, "_prime_mosaic",
    "PRIME Phase 6 — MOSAIC Dynamic Intelligence Composition Engine", -1, methods,
};

PyMODINIT_FUNC PyInit__prime_mosaic(void) {
    PyObject* m = PyModule_Create(&module_def);
    if (!m) return nullptr;
    MOSAICError = PyErr_NewException("_prime_mosaic.MOSAICError", nullptr, nullptr);
    Py_INCREF(MOSAICError);
    PyModule_AddObject(m, "MOSAICError", MOSAICError);
    PyModule_AddStringConstant(m, "VERSION", "0.6.0.dev1");
    return m;
}
