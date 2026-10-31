/*
 * atlas_bindings.cpp — Python C API binding for the PRIME ATLAS engine.
 *
 * Architecture:
 *   Python (prime/atlas.py)
 *     ↓
 *   this file (_prime_atlas C extension)
 *     ↓
 *   cpp/atlas/atlas_engine.hpp
 *
 * Exposed types:
 *   _prime_atlas.ATLASError           — exception class
 *   _prime_atlas._ATLASEngine         — wraps ATLASEngine
 *   _prime_atlas._UniversalProblem    — wraps UniversalProblem (returned from parse)
 *
 * Constants:
 *   _prime_atlas.SCHEMA_VERSION       — "1.0"
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "../atlas/atlas_engine.hpp"

#include <memory>
#include <string>
#include <stdexcept>

using namespace prime::atlas;

// ---------------------------------------------------------------------------
// Exception
// ---------------------------------------------------------------------------
static PyObject* ATLASError_type = nullptr;

static void set_atlas_error(const char* msg) {
    PyErr_SetString(ATLASError_type, msg);
}

// ---------------------------------------------------------------------------
// _UniversalProblem — Python wrapper around UniversalProblem
// ---------------------------------------------------------------------------
struct PyUniversalProblem {
    PyObject_HEAD
    UniversalProblem* problem;  // heap allocated, owned by this object
};

static void PyUniversalProblem_dealloc(PyUniversalProblem* self) {
    delete self->problem;
    Py_TYPE(self)->tp_free(self);
}

// Helper: build a Python dict from a std::unordered_map<str,str>
static PyObject* meta_to_pydict(const std::unordered_map<std::string,std::string>& m) {
    PyObject* d = PyDict_New();
    if (!d) return nullptr;
    for (auto& [k, v] : m) {
        PyObject* key = PyUnicode_FromString(k.c_str());
        PyObject* val = PyUnicode_FromString(v.c_str());
        if (!key || !val) { Py_XDECREF(key); Py_XDECREF(val); Py_DECREF(d); return nullptr; }
        PyDict_SetItem(d, key, val);
        Py_DECREF(key); Py_DECREF(val);
    }
    return d;
}

// Helper: build a Python list from a vector<string>
static PyObject* strvec_to_pylist(const std::vector<std::string>& v) {
    PyObject* lst = PyList_New(static_cast<Py_ssize_t>(v.size()));
    if (!lst) return nullptr;
    for (std::size_t i = 0; i < v.size(); ++i) {
        PyObject* s = PyUnicode_FromString(v[i].c_str());
        if (!s) { Py_DECREF(lst); return nullptr; }
        PyList_SET_ITEM(lst, static_cast<Py_ssize_t>(i), s);
    }
    return lst;
}

// Helper: build a Quantity dict
static PyObject* quantity_to_pydict(const Quantity& q) {
    PyObject* d = PyDict_New();
    if (!d) return nullptr;
    PyDict_SetItemString(d, "value",     PyFloat_FromDouble(q.value));
    PyDict_SetItemString(d, "unit",      PyUnicode_FromString(q.unit.c_str()));
    PyDict_SetItemString(d, "dimension", PyUnicode_FromString(q.dimension.c_str()));
    return d;
}

// Helper: build entity dict
static PyObject* entity_to_pydict(const Entity& e) {
    PyObject* d = PyDict_New();
    if (!d) return nullptr;
    PyDict_SetItemString(d, "id",         PyUnicode_FromString(e.id.c_str()));
    PyDict_SetItemString(d, "type",       PyUnicode_FromString(e.type.c_str()));
    PyDict_SetItemString(d, "name",       PyUnicode_FromString(e.name.c_str()));
    PyDict_SetItemString(d, "confidence", PyFloat_FromDouble(e.confidence));
    if (e.quantity) {
        PyDict_SetItemString(d, "quantity", quantity_to_pydict(*e.quantity));
    } else {
        PyDict_SetItemString(d, "quantity", Py_None); Py_INCREF(Py_None);
    }
    // Attributes list
    PyObject* attrs = PyList_New(static_cast<Py_ssize_t>(e.attributes.size()));
    for (std::size_t i = 0; i < e.attributes.size(); ++i) {
        auto& a = e.attributes[i];
        PyObject* ad = PyDict_New();
        PyDict_SetItemString(ad, "id",         PyUnicode_FromString(a.id.c_str()));
        PyDict_SetItemString(ad, "name",       PyUnicode_FromString(a.name.c_str()));
        PyDict_SetItemString(ad, "value_text", PyUnicode_FromString(a.value_text.c_str()));
        if (a.quantity) PyDict_SetItemString(ad, "quantity", quantity_to_pydict(*a.quantity));
        else           { PyDict_SetItemString(ad, "quantity", Py_None); Py_INCREF(Py_None); }
        PyList_SET_ITEM(attrs, static_cast<Py_ssize_t>(i), ad);
    }
    PyDict_SetItemString(d, "attributes", attrs); Py_DECREF(attrs);
    PyObject* meta = meta_to_pydict(e.metadata);
    PyDict_SetItemString(d, "metadata", meta); Py_DECREF(meta);
    return d;
}

// Helper: relationship dict
static PyObject* rel_to_pydict(const Relationship& r) {
    PyObject* d = PyDict_New();
    if (!d) return nullptr;
    PyDict_SetItemString(d, "id",         PyUnicode_FromString(r.id.c_str()));
    PyDict_SetItemString(d, "source_id",  PyUnicode_FromString(r.source_id.c_str()));
    PyDict_SetItemString(d, "relation",   PyUnicode_FromString(r.relation.c_str()));
    PyDict_SetItemString(d, "target_id",  PyUnicode_FromString(r.target_id.c_str()));
    PyDict_SetItemString(d, "confidence", PyFloat_FromDouble(r.confidence));
    PyObject* meta = meta_to_pydict(r.metadata);
    PyDict_SetItemString(d, "metadata", meta); Py_DECREF(meta);
    return d;
}

// Helper: constraint dict
static PyObject* con_to_pydict(const Constraint& c) {
    PyObject* d = PyDict_New();
    if (!d) return nullptr;
    PyDict_SetItemString(d, "id",         PyUnicode_FromString(c.id.c_str()));
    PyDict_SetItemString(d, "left",       PyUnicode_FromString(c.left.c_str()));
    PyDict_SetItemString(d, "op",         PyUnicode_FromString(constraint_op_to_string(c.op).c_str()));
    PyDict_SetItemString(d, "right",      PyUnicode_FromString(c.right.c_str()));
    PyDict_SetItemString(d, "type",       PyUnicode_FromString(c.type.c_str()));
    PyDict_SetItemString(d, "confidence", PyFloat_FromDouble(c.confidence));
    return d;
}

// Helper: goal dict
static PyObject* goal_to_pydict(const Goal& g) {
    PyObject* d = PyDict_New();
    if (!d) return nullptr;
    PyDict_SetItemString(d, "id",            PyUnicode_FromString(g.id.c_str()));
    PyDict_SetItemString(d, "description",   PyUnicode_FromString(g.description.c_str()));
    PyDict_SetItemString(d, "target",        PyUnicode_FromString(g.target.c_str()));
    PyDict_SetItemString(d, "desired_state", PyUnicode_FromString(g.desired_state.c_str()));
    PyDict_SetItemString(d, "priority",      PyLong_FromLong(g.priority));
    PyDict_SetItemString(d, "confidence",    PyFloat_FromDouble(g.confidence));
    return d;
}

// Helper: operation dict
static PyObject* op_to_pydict(const Operation& op) {
    PyObject* d = PyDict_New();
    if (!d) return nullptr;
    PyDict_SetItemString(d, "id",   PyUnicode_FromString(op.id.c_str()));
    PyDict_SetItemString(d, "type", PyUnicode_FromString(op.type.c_str()));
    PyObject* ins  = strvec_to_pylist(op.input_ids);
    PyObject* outs = strvec_to_pylist(op.output_ids);
    PyDict_SetItemString(d, "input_ids",  ins);  Py_DECREF(ins);
    PyDict_SetItemString(d, "output_ids", outs); Py_DECREF(outs);
    return d;
}

// Helper: expression dict
static PyObject* expr_to_pydict(const Expression& ex) {
    PyObject* d = PyDict_New();
    if (!d) return nullptr;
    PyDict_SetItemString(d, "id",        PyUnicode_FromString(ex.id.c_str()));
    PyDict_SetItemString(d, "op",        PyUnicode_FromString(ex.op.c_str()));
    PyDict_SetItemString(d, "output_id", PyUnicode_FromString(ex.output_id.c_str()));
    PyDict_SetItemString(d, "text",      PyUnicode_FromString(ex.text.c_str()));
    PyObject* operands = strvec_to_pylist(ex.operand_ids);
    PyDict_SetItemString(d, "operand_ids", operands); Py_DECREF(operands);
    return d;
}

// Helper: assumption dict
static PyObject* assume_to_pydict(const Assumption& a) {
    PyObject* d = PyDict_New();
    PyDict_SetItemString(d, "id",          PyUnicode_FromString(a.id.c_str()));
    PyDict_SetItemString(d, "description", PyUnicode_FromString(a.description.c_str()));
    PyDict_SetItemString(d, "confidence",  PyFloat_FromDouble(a.confidence));
    return d;
}

// Helper: state dict
static PyObject* state_to_pydict(const State& s) {
    PyObject* d = PyDict_New();
    PyDict_SetItemString(d, "id",    PyUnicode_FromString(s.id.c_str()));
    PyDict_SetItemString(d, "label", PyUnicode_FromString(s.label.c_str()));
    PyObject* eids = strvec_to_pylist(s.entity_ids);
    PyObject* cids = strvec_to_pylist(s.constraint_ids);
    PyDict_SetItemString(d, "entity_ids",     eids); Py_DECREF(eids);
    PyDict_SetItemString(d, "constraint_ids", cids); Py_DECREF(cids);
    return d;
}

// Build the complete Python dict representation of a UniversalProblem
static PyObject* problem_to_pydict(const UniversalProblem& p) {
    PyObject* d = PyDict_New();
    if (!d) return nullptr;

    #define SET_STR(key, val) PyDict_SetItemString(d, key, PyUnicode_FromString(val))
    SET_STR("problem_id",           p.problem_id.c_str());
    SET_STR("atlas_schema_version", p.atlas_schema_version.c_str());
    SET_STR("original_input",       p.original_input.c_str());
    SET_STR("normalized_input",     p.normalized_input.c_str());
    SET_STR("domain",               domain_to_string(p.domain).c_str());
    PyDict_SetItemString(d, "domain_confidence", PyFloat_FromDouble(p.domain_confidence));
    #undef SET_STR

    // entities
    PyObject* elist = PyList_New(static_cast<Py_ssize_t>(p.entities.size()));
    for (std::size_t i = 0; i < p.entities.size(); ++i)
        PyList_SET_ITEM(elist, static_cast<Py_ssize_t>(i), entity_to_pydict(p.entities[i]));
    PyDict_SetItemString(d, "entities", elist); Py_DECREF(elist);

    // relationships
    PyObject* rlist = PyList_New(static_cast<Py_ssize_t>(p.relationships.size()));
    for (std::size_t i = 0; i < p.relationships.size(); ++i)
        PyList_SET_ITEM(rlist, static_cast<Py_ssize_t>(i), rel_to_pydict(p.relationships[i]));
    PyDict_SetItemString(d, "relationships", rlist); Py_DECREF(rlist);

    // constraints
    PyObject* clist = PyList_New(static_cast<Py_ssize_t>(p.constraints.size()));
    for (std::size_t i = 0; i < p.constraints.size(); ++i)
        PyList_SET_ITEM(clist, static_cast<Py_ssize_t>(i), con_to_pydict(p.constraints[i]));
    PyDict_SetItemString(d, "constraints", clist); Py_DECREF(clist);

    // goals
    PyObject* glist = PyList_New(static_cast<Py_ssize_t>(p.goals.size()));
    for (std::size_t i = 0; i < p.goals.size(); ++i)
        PyList_SET_ITEM(glist, static_cast<Py_ssize_t>(i), goal_to_pydict(p.goals[i]));
    PyDict_SetItemString(d, "goals", glist); Py_DECREF(glist);

    // operations
    PyObject* olist = PyList_New(static_cast<Py_ssize_t>(p.operations.size()));
    for (std::size_t i = 0; i < p.operations.size(); ++i)
        PyList_SET_ITEM(olist, static_cast<Py_ssize_t>(i), op_to_pydict(p.operations[i]));
    PyDict_SetItemString(d, "operations", olist); Py_DECREF(olist);

    // expressions
    PyObject* exlist = PyList_New(static_cast<Py_ssize_t>(p.expressions.size()));
    for (std::size_t i = 0; i < p.expressions.size(); ++i)
        PyList_SET_ITEM(exlist, static_cast<Py_ssize_t>(i), expr_to_pydict(p.expressions[i]));
    PyDict_SetItemString(d, "expressions", exlist); Py_DECREF(exlist);

    // assumptions
    PyObject* alist = PyList_New(static_cast<Py_ssize_t>(p.assumptions.size()));
    for (std::size_t i = 0; i < p.assumptions.size(); ++i)
        PyList_SET_ITEM(alist, static_cast<Py_ssize_t>(i), assume_to_pydict(p.assumptions[i]));
    PyDict_SetItemString(d, "assumptions", alist); Py_DECREF(alist);

    // states
    PyObject* slist = PyList_New(static_cast<Py_ssize_t>(p.states.size()));
    for (std::size_t i = 0; i < p.states.size(); ++i)
        PyList_SET_ITEM(slist, static_cast<Py_ssize_t>(i), state_to_pydict(p.states[i]));
    PyDict_SetItemString(d, "states", slist); Py_DECREF(slist);

    // logic_clauses
    PyObject* lclist = PyList_New(static_cast<Py_ssize_t>(p.logic_clauses.size()));
    for (std::size_t i = 0; i < p.logic_clauses.size(); ++i) {
        auto& lc = p.logic_clauses[i];
        PyObject* lcd = PyDict_New();
        PyDict_SetItemString(lcd, "id",          PyUnicode_FromString(lc.id.c_str()));
        PyDict_SetItemString(lcd, "op",          PyUnicode_FromString(logic_op_to_string(lc.op).c_str()));
        PyDict_SetItemString(lcd, "condition",   PyUnicode_FromString(lc.condition.c_str()));
        PyDict_SetItemString(lcd, "implication", PyUnicode_FromString(lc.implication.c_str()));
        PyList_SET_ITEM(lclist, static_cast<Py_ssize_t>(i), lcd);
    }
    PyDict_SetItemString(d, "logic_clauses", lclist); Py_DECREF(lclist);

    // evidence
    PyObject* evlist = PyList_New(static_cast<Py_ssize_t>(p.evidence.size()));
    for (std::size_t i = 0; i < p.evidence.size(); ++i) {
        auto& ev = p.evidence[i];
        PyObject* evd = PyDict_New();
        PyDict_SetItemString(evd, "id",         PyUnicode_FromString(ev.id.c_str()));
        PyDict_SetItemString(evd, "source",     PyUnicode_FromString(ev.source.c_str()));
        PyDict_SetItemString(evd, "content",    PyUnicode_FromString(ev.content.c_str()));
        PyDict_SetItemString(evd, "confidence", PyFloat_FromDouble(ev.confidence));
        PyList_SET_ITEM(evlist, static_cast<Py_ssize_t>(i), evd);
    }
    PyDict_SetItemString(d, "evidence", evlist); Py_DECREF(evlist);

    // metadata
    PyObject* metad = meta_to_pydict(p.metadata);
    PyDict_SetItemString(d, "metadata", metad); Py_DECREF(metad);

    // normalized_input
    PyDict_SetItemString(d, "normalized_input",
                         PyUnicode_FromString(p.normalized_input.c_str()));

    // graph summary
    PyObject* graph_d = PyDict_New();
    PyDict_SetItemString(graph_d, "node_count", PyLong_FromSize_t(p.graph.node_count()));
    PyDict_SetItemString(graph_d, "edge_count", PyLong_FromSize_t(p.graph.edge_count()));
    PyDict_SetItemString(d, "graph", graph_d); Py_DECREF(graph_d);

    return d;
}

// ---------------------------------------------------------------------------
// PyUniversalProblem attributes
// ---------------------------------------------------------------------------
static PyObject* PyUP_get_problem_id(PyUniversalProblem* self, void*) {
    return PyUnicode_FromString(self->problem->problem_id.c_str());
}
static PyObject* PyUP_get_domain(PyUniversalProblem* self, void*) {
    return PyUnicode_FromString(domain_to_string(self->problem->domain).c_str());
}
static PyObject* PyUP_get_domain_confidence(PyUniversalProblem* self, void*) {
    return PyFloat_FromDouble(self->problem->domain_confidence);
}
static PyObject* PyUP_get_original_input(PyUniversalProblem* self, void*) {
    return PyUnicode_FromString(self->problem->original_input.c_str());
}
static PyObject* PyUP_get_schema_version(PyUniversalProblem* self, void*) {
    return PyUnicode_FromString(self->problem->atlas_schema_version.c_str());
}
static PyObject* PyUP_get_entities(PyUniversalProblem* self, void*) {
    PyObject* lst = PyList_New(static_cast<Py_ssize_t>(self->problem->entities.size()));
    for (std::size_t i = 0; i < self->problem->entities.size(); ++i)
        PyList_SET_ITEM(lst, static_cast<Py_ssize_t>(i), entity_to_pydict(self->problem->entities[i]));
    return lst;
}
static PyObject* PyUP_get_relationships(PyUniversalProblem* self, void*) {
    PyObject* lst = PyList_New(static_cast<Py_ssize_t>(self->problem->relationships.size()));
    for (std::size_t i = 0; i < self->problem->relationships.size(); ++i)
        PyList_SET_ITEM(lst, static_cast<Py_ssize_t>(i), rel_to_pydict(self->problem->relationships[i]));
    return lst;
}
static PyObject* PyUP_get_constraints(PyUniversalProblem* self, void*) {
    PyObject* lst = PyList_New(static_cast<Py_ssize_t>(self->problem->constraints.size()));
    for (std::size_t i = 0; i < self->problem->constraints.size(); ++i)
        PyList_SET_ITEM(lst, static_cast<Py_ssize_t>(i), con_to_pydict(self->problem->constraints[i]));
    return lst;
}
static PyObject* PyUP_get_goals(PyUniversalProblem* self, void*) {
    PyObject* lst = PyList_New(static_cast<Py_ssize_t>(self->problem->goals.size()));
    for (std::size_t i = 0; i < self->problem->goals.size(); ++i)
        PyList_SET_ITEM(lst, static_cast<Py_ssize_t>(i), goal_to_pydict(self->problem->goals[i]));
    return lst;
}
static PyObject* PyUP_get_operations(PyUniversalProblem* self, void*) {
    PyObject* lst = PyList_New(static_cast<Py_ssize_t>(self->problem->operations.size()));
    for (std::size_t i = 0; i < self->problem->operations.size(); ++i)
        PyList_SET_ITEM(lst, static_cast<Py_ssize_t>(i), op_to_pydict(self->problem->operations[i]));
    return lst;
}
static PyObject* PyUP_get_expressions(PyUniversalProblem* self, void*) {
    PyObject* lst = PyList_New(static_cast<Py_ssize_t>(self->problem->expressions.size()));
    for (std::size_t i = 0; i < self->problem->expressions.size(); ++i)
        PyList_SET_ITEM(lst, static_cast<Py_ssize_t>(i), expr_to_pydict(self->problem->expressions[i]));
    return lst;
}
static PyObject* PyUP_get_assumptions(PyUniversalProblem* self, void*) {
    PyObject* lst = PyList_New(static_cast<Py_ssize_t>(self->problem->assumptions.size()));
    for (std::size_t i = 0; i < self->problem->assumptions.size(); ++i)
        PyList_SET_ITEM(lst, static_cast<Py_ssize_t>(i), assume_to_pydict(self->problem->assumptions[i]));
    return lst;
}
static PyObject* PyUP_get_states(PyUniversalProblem* self, void*) {
    PyObject* lst = PyList_New(static_cast<Py_ssize_t>(self->problem->states.size()));
    for (std::size_t i = 0; i < self->problem->states.size(); ++i)
        PyList_SET_ITEM(lst, static_cast<Py_ssize_t>(i), state_to_pydict(self->problem->states[i]));
    return lst;
}
static PyObject* PyUP_get_graph(PyUniversalProblem* self, void*) {
    PyObject* d = PyDict_New();
    PyDict_SetItemString(d, "node_count", PyLong_FromSize_t(self->problem->graph.node_count()));
    PyDict_SetItemString(d, "edge_count", PyLong_FromSize_t(self->problem->graph.edge_count()));
    return d;
}

static PyGetSetDef PyUP_getset[] = {
    {"problem_id",        (getter)PyUP_get_problem_id,        nullptr, "Problem ID",        nullptr},
    {"domain",            (getter)PyUP_get_domain,            nullptr, "Detected domain",   nullptr},
    {"domain_confidence", (getter)PyUP_get_domain_confidence, nullptr, "Domain confidence", nullptr},
    {"original_input",    (getter)PyUP_get_original_input,    nullptr, "Original input",    nullptr},
    {"schema_version",    (getter)PyUP_get_schema_version,    nullptr, "Schema version",    nullptr},
    {"entities",          (getter)PyUP_get_entities,          nullptr, "List of entity dicts",       nullptr},
    {"relationships",     (getter)PyUP_get_relationships,     nullptr, "List of relationship dicts", nullptr},
    {"constraints",       (getter)PyUP_get_constraints,       nullptr, "List of constraint dicts",   nullptr},
    {"goals",             (getter)PyUP_get_goals,             nullptr, "List of goal dicts",         nullptr},
    {"operations",        (getter)PyUP_get_operations,        nullptr, "List of operation dicts",    nullptr},
    {"expressions",       (getter)PyUP_get_expressions,       nullptr, "List of expression dicts",   nullptr},
    {"assumptions",       (getter)PyUP_get_assumptions,       nullptr, "List of assumption dicts",   nullptr},
    {"states",            (getter)PyUP_get_states,            nullptr, "List of state dicts",        nullptr},
    {"graph",             (getter)PyUP_get_graph,             nullptr, "Graph summary dict",         nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr}
};

// to_dict method
static PyObject* PyUP_to_dict(PyUniversalProblem* self, PyObject* /*args*/) {
    return problem_to_pydict(*self->problem);
}

static PyMethodDef PyUP_methods[] = {
    {"to_dict", (PyCFunction)PyUP_to_dict, METH_NOARGS, "Return dict representation"},
    {nullptr, nullptr, 0, nullptr}
};

static PyObject* PyUP_repr(PyUniversalProblem* self) {
    return PyUnicode_FromFormat(
        "<UniversalProblem id=%s domain=%s entities=%zu goals=%zu>",
        self->problem->problem_id.c_str(),
        domain_to_string(self->problem->domain).c_str(),
        self->problem->entities.size(),
        self->problem->goals.size()
    );
}

static PyTypeObject PyUniversalProblemType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "_prime_atlas._UniversalProblem",
    sizeof(PyUniversalProblem),
    0,
    (destructor)PyUniversalProblem_dealloc,
    0, 0, 0, 0,
    (reprfunc)PyUP_repr,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    Py_TPFLAGS_DEFAULT,
    "ATLAS Universal Problem Representation",
    0, 0, 0, 0, 0, 0,
    PyUP_methods,
    0,
    PyUP_getset,
    0, 0, 0, 0, 0, 0, 0,
    PyType_GenericNew,
};

// Helper: alloc a PyUniversalProblem and populate it
static PyObject* make_py_problem(UniversalProblem&& p) {
    PyUniversalProblem* obj = PyObject_New(PyUniversalProblem, &PyUniversalProblemType);
    if (!obj) return nullptr;
    obj->problem = new UniversalProblem(std::move(p));
    return (PyObject*)obj;
}

// ---------------------------------------------------------------------------
// _ATLASEngine
// ---------------------------------------------------------------------------
struct PyATLASEngine {
    PyObject_HEAD
    ATLASEngine* engine;
};

static void PyATLASEngine_dealloc(PyATLASEngine* self) {
    delete self->engine;
    Py_TYPE(self)->tp_free(self);
}

static int PyATLASEngine_init(PyATLASEngine* self, PyObject* args, PyObject* kwargs) {
    // Optional keyword args for limits
    static const char* kwlist[] = {
        "max_input_bytes", "max_entities", "max_relationships",
        "max_constraints", "max_goals", "max_operations", "max_graph_nodes",
        nullptr
    };
    AtlasLimits lim;
    unsigned long long mib = lim.max_input_bytes, me = lim.max_entities,
                       mr  = lim.max_relationships, mc = lim.max_constraints,
                       mg  = lim.max_goals,          mo = lim.max_operations,
                       mgn = lim.max_graph_nodes;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs,
            "|KKKKKKK", (char**)kwlist,
            &mib, &me, &mr, &mc, &mg, &mo, &mgn))
        return -1;

    lim.max_input_bytes   = static_cast<std::size_t>(mib);
    lim.max_entities      = static_cast<std::size_t>(me);
    lim.max_relationships = static_cast<std::size_t>(mr);
    lim.max_constraints   = static_cast<std::size_t>(mc);
    lim.max_goals         = static_cast<std::size_t>(mg);
    lim.max_operations    = static_cast<std::size_t>(mo);
    lim.max_graph_nodes   = static_cast<std::size_t>(mgn);

    try {
        self->engine = new ATLASEngine(lim);
    } catch (std::exception& ex) {
        set_atlas_error(ex.what());
        return -1;
    }
    return 0;
}

// parse(text) → _UniversalProblem
static PyObject* PyATLAS_parse(PyATLASEngine* self, PyObject* args) {
    const char* text = nullptr;
    if (!PyArg_ParseTuple(args, "s", &text)) return nullptr;
    try {
        return make_py_problem(self->engine->parse(text ? text : ""));
    } catch (AtlasError& ex) {
        set_atlas_error(ex.what());
        return nullptr;
    } catch (std::exception& ex) {
        set_atlas_error(ex.what());
        return nullptr;
    }
}

// to_json(problem) → str
static PyObject* PyATLAS_to_json(PyATLASEngine* self, PyObject* args) {
    PyObject* prob_obj = nullptr;
    if (!PyArg_ParseTuple(args, "O!", &PyUniversalProblemType, &prob_obj)) return nullptr;
    auto* pyup = (PyUniversalProblem*)prob_obj;
    try {
        std::string json = self->engine->to_json(*pyup->problem);
        return PyUnicode_FromString(json.c_str());
    } catch (std::exception& ex) {
        set_atlas_error(ex.what());
        return nullptr;
    }
}

// from_json(json_str) → _UniversalProblem
static PyObject* PyATLAS_from_json(PyATLASEngine* self, PyObject* args) {
    const char* json_str = nullptr;
    if (!PyArg_ParseTuple(args, "s", &json_str)) return nullptr;
    try {
        return make_py_problem(self->engine->from_json(json_str ? json_str : ""));
    } catch (AtlasError& ex) {
        set_atlas_error(ex.what());
        return nullptr;
    } catch (std::exception& ex) {
        set_atlas_error(ex.what());
        return nullptr;
    }
}

// validate(problem) → dict {valid: bool, errors: [...], warnings: [...]}
static PyObject* PyATLAS_validate(PyATLASEngine* self, PyObject* args) {
    PyObject* prob_obj = nullptr;
    if (!PyArg_ParseTuple(args, "O!", &PyUniversalProblemType, &prob_obj)) return nullptr;
    auto* pyup = (PyUniversalProblem*)prob_obj;
    try {
        auto result = self->engine->validate(*pyup->problem);
        PyObject* d = PyDict_New();
        PyDict_SetItemString(d, "valid", result.valid ? Py_True : Py_False);
        Py_INCREF(result.valid ? Py_True : Py_False);

        PyObject* errs = PyList_New(static_cast<Py_ssize_t>(result.errors.size()));
        for (std::size_t i = 0; i < result.errors.size(); ++i)
            PyList_SET_ITEM(errs, static_cast<Py_ssize_t>(i),
                            PyUnicode_FromString(result.errors[i].c_str()));
        PyDict_SetItemString(d, "errors", errs); Py_DECREF(errs);

        PyObject* warns = PyList_New(static_cast<Py_ssize_t>(result.warnings.size()));
        for (std::size_t i = 0; i < result.warnings.size(); ++i)
            PyList_SET_ITEM(warns, static_cast<Py_ssize_t>(i),
                            PyUnicode_FromString(result.warnings[i].c_str()));
        PyDict_SetItemString(d, "warnings", warns); Py_DECREF(warns);
        return d;
    } catch (std::exception& ex) {
        set_atlas_error(ex.what());
        return nullptr;
    }
}

// schema_version() → str
static PyObject* PyATLAS_schema_version(PyATLASEngine* /*self*/, PyObject* /*args*/) {
    return PyUnicode_FromString(ATLASEngine::schema_version());
}

static PyMethodDef PyATLAS_methods[] = {
    {"parse",          (PyCFunction)PyATLAS_parse,          METH_VARARGS, "Parse text → UniversalProblem"},
    {"to_json",        (PyCFunction)PyATLAS_to_json,        METH_VARARGS, "Serialize UniversalProblem → JSON str"},
    {"from_json",      (PyCFunction)PyATLAS_from_json,      METH_VARARGS, "Deserialize JSON str → UniversalProblem"},
    {"validate",       (PyCFunction)PyATLAS_validate,       METH_VARARGS, "Validate UniversalProblem → dict"},
    {"schema_version", (PyCFunction)PyATLAS_schema_version, METH_NOARGS,  "Return ATLAS schema version string"},
    {nullptr, nullptr, 0, nullptr}
};

static PyObject* PyATLAS_repr(PyATLASEngine* self) {
    (void)self;
    return PyUnicode_FromString("<_prime_atlas._ATLASEngine>");
}

static PyTypeObject PyATLASEngineType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "_prime_atlas._ATLASEngine",    /* tp_name */
    sizeof(PyATLASEngine),          /* tp_basicsize */
    0,                              /* tp_itemsize */
    (destructor)PyATLASEngine_dealloc, /* tp_dealloc */
    0,                              /* tp_vectorcall_offset */
    0,                              /* tp_getattr */
    0,                              /* tp_setattr */
    0,                              /* tp_as_async */
    (reprfunc)PyATLAS_repr,         /* tp_repr */
    0,                              /* tp_as_number */
    0,                              /* tp_as_sequence */
    0,                              /* tp_as_mapping */
    0,                              /* tp_hash */
    0,                              /* tp_call */
    0,                              /* tp_str */
    0,                              /* tp_getattro */
    0,                              /* tp_setattro */
    0,                              /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    "ATLAS Engine — parse / validate / serialize UniversalProblem", /* tp_doc */
    0,                              /* tp_traverse */
    0,                              /* tp_clear */
    0,                              /* tp_richcompare */
    0,                              /* tp_weaklistoffset */
    0,                              /* tp_iter */
    0,                              /* tp_iternext */
    PyATLAS_methods,                /* tp_methods */
    0,                              /* tp_members */
    0,                              /* tp_getset */
    0,                              /* tp_base */
    0,                              /* tp_dict */
    0,                              /* tp_descr_get */
    0,                              /* tp_descr_set */
    0,                              /* tp_dictoffset */
    (initproc)PyATLASEngine_init,   /* tp_init */
    0,                              /* tp_alloc */
    PyType_GenericNew,              /* tp_new */
};

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------
static PyMethodDef module_methods[] = {
    {nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_prime_atlas",
    "PRIME ATLAS — Universal Problem Representation Engine (C++ backend)",
    -1,
    module_methods,
    nullptr, nullptr, nullptr, nullptr
};

PyMODINIT_FUNC PyInit__prime_atlas(void) {
    if (PyType_Ready(&PyATLASEngineType) < 0)      return nullptr;
    if (PyType_Ready(&PyUniversalProblemType) < 0) return nullptr;

    PyObject* m = PyModule_Create(&moduledef);
    if (!m) return nullptr;

    // Exception
    ATLASError_type = PyErr_NewException("_prime_atlas.ATLASError", nullptr, nullptr);
    if (!ATLASError_type) { Py_DECREF(m); return nullptr; }
    Py_INCREF(ATLASError_type);
    PyModule_AddObject(m, "ATLASError", ATLASError_type);

    // Types
    Py_INCREF(&PyATLASEngineType);
    PyModule_AddObject(m, "_ATLASEngine", (PyObject*)&PyATLASEngineType);
    Py_INCREF(&PyUniversalProblemType);
    PyModule_AddObject(m, "_UniversalProblem", (PyObject*)&PyUniversalProblemType);

    // Constants
    PyModule_AddStringConstant(m, "SCHEMA_VERSION", ATLAS_SCHEMA_VERSION);

    return m;
}
