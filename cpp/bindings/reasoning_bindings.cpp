// reasoning_bindings.cpp — Python C API bindings for the PRIME Phase 5 Reasoning Engine.
//
// Exposes:
//   _prime_reasoning.ReasoningEngine
//   _prime_reasoning.ReasoningResult
//   _prime_reasoning.ReasoningStep
//   _prime_reasoning.ReasoningTrace
//   _prime_reasoning.Rule
//   _prime_reasoning.Condition
//   _prime_reasoning.Conclusion
//   _prime_reasoning.ReasoningContext
//   _prime_reasoning.solve_context()
//   _prime_reasoning.parse_condition()
//   _prime_reasoning.eval_expression()
//   _prime_reasoning.ReasoningError

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "reasoning/reasoning_types.hpp"
#include "reasoning/expression_evaluator.hpp"
#include "reasoning/rule_engine.hpp"
#include "reasoning/constraint_engine.hpp"
#include "reasoning/symbolic_engine.hpp"
#include "reasoning/reasoning_engine.hpp"

#include <string>
#include <vector>
#include <unordered_map>
#include <sstream>

// ---------------------------------------------------------------------------
// ReasoningError exception type
// ---------------------------------------------------------------------------
static PyObject* ReasoningError = nullptr;

// ---------------------------------------------------------------------------
// Helpers: Python dict/list → C++ structs
// ---------------------------------------------------------------------------

static double py_get_double(PyObject* dict, const char* key, double dflt = 0.0) {
    PyObject* v = PyDict_GetItemString(dict, key);
    if (!v) return dflt;
    if (PyFloat_Check(v)) return PyFloat_AsDouble(v);
    if (PyLong_Check(v))  return (double)PyLong_AsLong(v);
    return dflt;
}

static std::string py_get_str(PyObject* dict, const char* key, const char* dflt = "") {
    PyObject* v = PyDict_GetItemString(dict, key);
    if (!v || !PyUnicode_Check(v)) return dflt;
    return PyUnicode_AsUTF8(v);
}

static bool py_get_bool(PyObject* dict, const char* key, bool dflt = false) {
    PyObject* v = PyDict_GetItemString(dict, key);
    if (!v) return dflt;
    return PyObject_IsTrue(v) == 1;
}

static int py_get_int(PyObject* dict, const char* key, int dflt = 0) {
    PyObject* v = PyDict_GetItemString(dict, key);
    if (!v) return dflt;
    if (PyLong_Check(v))  return (int)PyLong_AsLong(v);
    if (PyFloat_Check(v)) return (int)PyFloat_AsDouble(v);
    return dflt;
}

// Parse a Python condition dict into a prime::reasoning::Condition
static prime::reasoning::Condition parse_condition_dict(PyObject* d) {
    using namespace prime::reasoning;
    Condition c;
    c.left  = py_get_str(d, "left");
    c.right = py_get_str(d, "right");

    std::string op_str = py_get_str(d, "op", "=");
    if      (op_str == "="  || op_str == "==") c.op = CondOp::EQ;
    else if (op_str == "!=" || op_str == "<>") c.op = CondOp::NEQ;
    else if (op_str == "<")                     c.op = CondOp::LT;
    else if (op_str == "<=")                    c.op = CondOp::LTE;
    else if (op_str == ">")                     c.op = CondOp::GT;
    else if (op_str == ">=")                    c.op = CondOp::GTE;
    else if (op_str == "AND")                   c.op = CondOp::AND;
    else if (op_str == "OR")                    c.op = CondOp::OR;
    else if (op_str == "NOT")                   c.op = CondOp::NOT;
    else if (op_str == "EXISTS")                c.op = CondOp::EXISTS;
    else                                         c.op = CondOp::EQ;

    c.right_is_fact = py_get_bool(d, "right_is_fact");

    // Sub-conditions
    PyObject* subs = PyDict_GetItemString(d, "sub_conditions");
    if (subs && PyList_Check(subs)) {
        for (Py_ssize_t i = 0; i < PyList_Size(subs); ++i) {
            PyObject* sub = PyList_GetItem(subs, i);
            if (sub && PyDict_Check(sub))
                c.sub_conditions.push_back(parse_condition_dict(sub));
        }
    }
    return c;
}

// Parse a Python rule dict into a prime::reasoning::Rule
static prime::reasoning::Rule parse_rule_dict(PyObject* d) {
    using namespace prime::reasoning;
    Rule r;
    r.id         = py_get_str(d, "id");
    r.name       = py_get_str(d, "name");
    r.priority   = py_get_int(d, "priority");
    r.confidence = py_get_double(d, "confidence", 1.0);

    PyObject* conds = PyDict_GetItemString(d, "conditions");
    if (conds && PyList_Check(conds)) {
        for (Py_ssize_t i = 0; i < PyList_Size(conds); ++i) {
            PyObject* c = PyList_GetItem(conds, i);
            if (c && PyDict_Check(c))
                r.conditions.push_back(parse_condition_dict(c));
        }
    }

    PyObject* concs = PyDict_GetItemString(d, "conclusions");
    if (concs && PyList_Check(concs)) {
        for (Py_ssize_t i = 0; i < PyList_Size(concs); ++i) {
            PyObject* c = PyList_GetItem(concs, i);
            if (!c || !PyDict_Check(c)) continue;
            Conclusion conc;
            conc.name        = py_get_str(c, "name");
            conc.value_expr  = py_get_str(c, "value");
            conc.unit        = py_get_str(c, "unit");
            conc.confidence  = py_get_double(c, "confidence", 1.0);
            // Check if numeric value is provided
            PyObject* nv = PyDict_GetItemString(c, "numeric_value");
            if (nv && (PyFloat_Check(nv) || PyLong_Check(nv))) {
                conc.is_numeric    = true;
                conc.numeric_value = PyFloat_Check(nv) ? PyFloat_AsDouble(nv)
                                                        : (double)PyLong_AsLong(nv);
            }
            r.conclusions.push_back(conc);
        }
    }
    return r;
}

// Parse entity list from ATLAS into facts
static prime::reasoning::ReasoningContext
build_context_from_dict(PyObject* ctx_dict) {
    using namespace prime::reasoning;
    ReasoningContext ctx;

    // Problem metadata
    ctx.problem_id     = py_get_str(ctx_dict, "problem_id");
    ctx.problem_domain = py_get_str(ctx_dict, "problem_domain");

    // Config
    PyObject* cfg = PyDict_GetItemString(ctx_dict, "config");
    if (cfg && PyDict_Check(cfg)) {
        std::string mode_str = py_get_str(cfg, "mode", "hybrid");
        if      (mode_str == "rule")         ctx.config.mode = ReasoningMode::RULE;
        else if (mode_str == "symbolic")     ctx.config.mode = ReasoningMode::SYMBOLIC;
        else if (mode_str == "llm_assisted") ctx.config.mode = ReasoningMode::LLM_ASSISTED;
        else                                  ctx.config.mode = ReasoningMode::HYBRID;

        ctx.config.max_steps           = py_get_int(cfg, "max_steps",        1000);
        ctx.config.max_depth           = py_get_int(cfg, "max_depth",          50);
        ctx.config.max_rule_firings    = py_get_int(cfg, "max_rule_firings",   500);
        ctx.config.max_expression_size = py_get_int(cfg, "max_expression_size",256);
        ctx.config.timeout_ms          = py_get_double(cfg, "timeout_ms",    5000.0);
        ctx.config.use_rules           = py_get_bool(cfg, "use_rules",    true);
        ctx.config.use_symbolic        = py_get_bool(cfg, "use_symbolic",  true);
        ctx.config.use_llm             = py_get_bool(cfg, "use_llm",      false);
        ctx.config.detect_circular     = py_get_bool(cfg, "detect_circular", true);
        ctx.config.confidence_threshold= py_get_double(cfg, "confidence_threshold", 0.0);
    }

    // Entities → facts
    PyObject* entities = PyDict_GetItemString(ctx_dict, "entities");
    if (entities && PyList_Check(entities)) {
        for (Py_ssize_t i = 0; i < PyList_Size(entities); ++i) {
            PyObject* e = PyList_GetItem(entities, i);
            if (!e || !PyDict_Check(e)) continue;

            std::string eid        = py_get_str(e, "id");
            std::string ename      = py_get_str(e, "name");
            std::string etype      = py_get_str(e, "type");
            double      confidence = py_get_double(e, "confidence", 1.0);
            bool        has_qty    = false;
            double      qty_val    = 0.0;
            std::string qty_unit;

            PyObject* qty = PyDict_GetItemString(e, "quantity");
            if (qty && qty != Py_None && PyDict_Check(qty)) {
                has_qty  = true;
                qty_val  = py_get_double(qty, "value");
                qty_unit = py_get_str(qty, "unit");
            }

            Fact f = entity_to_fact(eid, ename, etype, has_qty, qty_val, qty_unit, confidence);
            ctx.facts[eid] = f;
            // Also index by human name
            if (!ename.empty() && ename != eid)
                ctx.facts[ename] = f;
        }
    }

    // Additional pre-bound facts (user-supplied)
    PyObject* facts = PyDict_GetItemString(ctx_dict, "facts");
    if (facts && PyDict_Check(facts)) {
        PyObject *key, *val;
        Py_ssize_t pos = 0;
        while (PyDict_Next(facts, &pos, &key, &val)) {
            if (!PyUnicode_Check(key)) continue;
            std::string kstr = PyUnicode_AsUTF8(key);
            Fact f;
            f.id     = kstr;
            f.name   = kstr;
            f.source = "initial";
            f.bound  = true;
            if (PyFloat_Check(val) || PyLong_Check(val)) {
                f.value = FactValue{PyFloat_Check(val) ? PyFloat_AsDouble(val)
                                                       : (double)PyLong_AsLong(val)};
            } else if (PyBool_Check(val)) {
                f.value = FactValue{val == Py_True};
            } else if (PyUnicode_Check(val)) {
                std::string sv = PyUnicode_AsUTF8(val);
                if (sv == "true")       f.value = FactValue{true};
                else if (sv == "false") f.value = FactValue{false};
                else {
                    try { f.value = FactValue{std::stod(sv)}; }
                    catch (...) { f.value = FactValue{sv}; f.bound = true; }
                }
            } else if (PyDict_Check(val)) {
                // Could be {value: ..., unit: ...}
                double dv = py_get_double(val, "value");
                f.value   = FactValue{dv};
                f.unit    = py_get_str(val, "unit");
            } else {
                f.value = FactValue{kstr};
                f.bound = false;
            }
            f.confidence = 1.0;
            ctx.facts[kstr] = f;
        }
    }

    // Constraints
    PyObject* consts = PyDict_GetItemString(ctx_dict, "constraints");
    if (consts && PyList_Check(consts)) {
        for (Py_ssize_t i = 0; i < PyList_Size(consts); ++i) {
            PyObject* c = PyList_GetItem(consts, i);
            if (!c || !PyDict_Check(c)) continue;
            ReasoningConstraint rc = atlas_constraint_to_reasoning(
                py_get_str(c, "id"),
                py_get_str(c, "left"),
                py_get_str(c, "op", "="),
                py_get_str(c, "right"),
                py_get_double(c, "confidence", 1.0));
            ctx.constraints.push_back(rc);
        }
    }

    // Goals
    PyObject* goals = PyDict_GetItemString(ctx_dict, "goals");
    if (goals && PyList_Check(goals)) {
        for (Py_ssize_t i = 0; i < PyList_Size(goals); ++i) {
            PyObject* g = PyList_GetItem(goals, i);
            if (!g || !PyDict_Check(g)) continue;
            Goal goal;
            goal.id          = py_get_str(g, "id");
            goal.target      = py_get_str(g, "target");
            goal.description = py_get_str(g, "description");
            goal.priority    = py_get_int(g, "priority");
            goal.confidence  = py_get_double(g, "confidence", 1.0);
            ctx.goals.push_back(goal);
        }
    }

    // Expressions (from ATLAS)
    PyObject* exprs = PyDict_GetItemString(ctx_dict, "expressions");
    if (exprs && PyList_Check(exprs)) {
        for (Py_ssize_t i = 0; i < PyList_Size(exprs); ++i) {
            PyObject* ex = PyList_GetItem(exprs, i);
            if (!ex || !PyDict_Check(ex)) continue;
            AtlasExpression ae;
            ae.id        = py_get_str(ex, "id");
            ae.op        = py_get_str(ex, "op");
            ae.output_id = py_get_str(ex, "output_id");
            ae.text      = py_get_str(ex, "text");

            PyObject* ops = PyDict_GetItemString(ex, "operand_ids");
            if (ops && PyList_Check(ops)) {
                for (Py_ssize_t j = 0; j < PyList_Size(ops); ++j) {
                    PyObject* op = PyList_GetItem(ops, j);
                    if (op && PyUnicode_Check(op))
                        ae.operand_ids.push_back(PyUnicode_AsUTF8(op));
                }
            }
            ctx.expressions.push_back(ae);
        }
    }

    // User-provided rules
    PyObject* rules = PyDict_GetItemString(ctx_dict, "rules");
    if (rules && PyList_Check(rules)) {
        for (Py_ssize_t i = 0; i < PyList_Size(rules); ++i) {
            PyObject* r = PyList_GetItem(rules, i);
            if (r && PyDict_Check(r))
                ctx.rules.push_back(parse_rule_dict(r));
        }
    }

    return ctx;
}

// ---------------------------------------------------------------------------
// Build Python result dict from ReasoningResult
// ---------------------------------------------------------------------------
static PyObject* result_to_pydict(const prime::reasoning::ReasoningResult& res) {
    using namespace prime::reasoning;

    PyObject* d = PyDict_New();
    PyDict_SetItemString(d, "status",
        PyUnicode_FromString(status_to_string(res.status).c_str()));
    PyDict_SetItemString(d, "answer",
        PyUnicode_FromString(res.answer.c_str()));
    PyDict_SetItemString(d, "answer_unit",
        PyUnicode_FromString(res.answer_unit.c_str()));
    if (res.answer_is_numeric)
        PyDict_SetItemString(d, "answer_numeric", PyFloat_FromDouble(res.answer_numeric));
    else
        PyDict_SetItemString(d, "answer_numeric", Py_None); Py_INCREF(Py_None);
    PyDict_SetItemString(d, "answer_is_numeric", PyBool_FromLong(res.answer_is_numeric));
    PyDict_SetItemString(d, "confidence",        PyFloat_FromDouble(res.confidence));
    PyDict_SetItemString(d, "error_message",
        PyUnicode_FromString(res.error_message.c_str()));
    PyDict_SetItemString(d, "mode_used",
        PyUnicode_FromString(res.mode_used.c_str()));

    // Trace
    PyObject* trace_d = PyDict_New();
    PyDict_SetItemString(trace_d, "total_steps",
        PyLong_FromLong(res.trace.total_steps));
    PyDict_SetItemString(trace_d, "rules_fired",
        PyLong_FromLong(res.trace.rules_fired));
    PyDict_SetItemString(trace_d, "expressions_evaluated",
        PyLong_FromLong(res.trace.expressions_evaluated));
    PyDict_SetItemString(trace_d, "constraints_checked",
        PyLong_FromLong(res.trace.constraints_checked));

    PyObject* steps_list = PyList_New(0);
    for (auto& step : res.trace.steps) {
        PyObject* s = PyDict_New();
        PyDict_SetItemString(s, "step_id",    PyLong_FromLong(step.step_id));
        PyDict_SetItemString(s, "type",       PyUnicode_FromString(step_type_to_string(step.type).c_str()));
        PyDict_SetItemString(s, "operation",  PyUnicode_FromString(step.operation.c_str()));
        PyDict_SetItemString(s, "rule_id",    PyUnicode_FromString(step.rule_id.c_str()));
        PyDict_SetItemString(s, "description",PyUnicode_FromString(step.description.c_str()));
        PyDict_SetItemString(s, "confidence", PyFloat_FromDouble(step.confidence));
        PyDict_SetItemString(s, "success",    PyBool_FromLong(step.success));

        PyObject* inf = PyList_New(0);
        for (auto& f : step.input_facts)
            PyList_Append(inf, PyUnicode_FromString(f.c_str()));
        PyDict_SetItemString(s, "input_facts", inf);
        Py_DECREF(inf);

        PyObject* outf = PyList_New(0);
        for (auto& f : step.output_facts)
            PyList_Append(outf, PyUnicode_FromString(f.c_str()));
        PyDict_SetItemString(s, "output_facts", outf);
        Py_DECREF(outf);

        PyList_Append(steps_list, s);
        Py_DECREF(s);
    }
    PyDict_SetItemString(trace_d, "steps", steps_list);
    Py_DECREF(steps_list);
    PyDict_SetItemString(d, "trace", trace_d);
    Py_DECREF(trace_d);

    // Goal results
    PyObject* grl = PyList_New(0);
    for (auto& gr : res.goal_results) {
        PyObject* g = PyDict_New();
        PyDict_SetItemString(g, "id",            PyUnicode_FromString(gr.id.c_str()));
        PyDict_SetItemString(g, "target",        PyUnicode_FromString(gr.target.c_str()));
        PyDict_SetItemString(g, "satisfied",     PyBool_FromLong(gr.satisfied));
        PyDict_SetItemString(g, "derived_value", PyUnicode_FromString(gr.derived_value.c_str()));
        PyDict_SetItemString(g, "unit",          PyUnicode_FromString(gr.unit.c_str()));
        PyDict_SetItemString(g, "confidence",    PyFloat_FromDouble(gr.confidence));
        PyList_Append(grl, g);
        Py_DECREF(g);
    }
    PyDict_SetItemString(d, "goal_results", grl);
    Py_DECREF(grl);

    // Constraint results
    PyObject* crl = PyList_New(0);
    for (auto& cr : res.constraint_results) {
        PyObject* c = PyDict_New();
        PyDict_SetItemString(c, "id",          PyUnicode_FromString(cr.id.c_str()));
        const char* rs = (cr.result == prime::reasoning::ConstraintResult::SATISFIED) ? "satisfied"
                       : (cr.result == prime::reasoning::ConstraintResult::VIOLATED)  ? "violated"
                       : "unknown";
        PyDict_SetItemString(c, "result",      PyUnicode_FromString(rs));
        PyDict_SetItemString(c, "description", PyUnicode_FromString(cr.description.c_str()));
        PyList_Append(crl, c);
        Py_DECREF(c);
    }
    PyDict_SetItemString(d, "constraint_results", crl);
    Py_DECREF(crl);

    // Derived facts
    PyObject* dfl = PyDict_New();
    for (auto& [k, v] : res.derived_facts) {
        PyObject* fv;
        if (prime::reasoning::fact_value_is_numeric(v.value))
            fv = PyFloat_FromDouble(prime::reasoning::fact_value_to_double(v.value));
        else
            fv = PyUnicode_FromString(prime::reasoning::fact_value_to_string(v.value).c_str());
        PyDict_SetItemString(dfl, k.c_str(), fv);
        Py_DECREF(fv);
    }
    PyDict_SetItemString(d, "derived_facts", dfl);
    Py_DECREF(dfl);

    return d;
}

// ---------------------------------------------------------------------------
// Module-level functions
// ---------------------------------------------------------------------------

// solve_context(dict) → dict
static PyObject* py_solve_context(PyObject* /*self*/, PyObject* args) {
    PyObject* ctx_dict;
    if (!PyArg_ParseTuple(args, "O!", &PyDict_Type, &ctx_dict))
        return nullptr;
    try {
        auto ctx    = build_context_from_dict(ctx_dict);
        prime::reasoning::ReasoningEngine engine;
        auto result = engine.solve(std::move(ctx));
        return result_to_pydict(result);
    } catch (const std::exception& e) {
        PyErr_SetString(ReasoningError, e.what());
        return nullptr;
    }
}

// eval_expression(expr_str, bindings_dict) → float
static PyObject* py_eval_expression(PyObject* /*self*/, PyObject* args) {
    const char* expr_str;
    PyObject*   bindings_dict;
    if (!PyArg_ParseTuple(args, "sO!", &expr_str, &PyDict_Type, &bindings_dict))
        return nullptr;

    std::unordered_map<std::string, double> bindings;
    PyObject* key;
    PyObject* val;
    Py_ssize_t pos = 0;
    while (PyDict_Next(bindings_dict, &pos, &key, &val)) {
        if (!PyUnicode_Check(key)) continue;
        double d = 0.0;
        if (PyFloat_Check(val)) d = PyFloat_AsDouble(val);
        else if (PyLong_Check(val)) d = (double)PyLong_AsLong(val);
        bindings[PyUnicode_AsUTF8(key)] = d;
    }

    try {
        double result = prime::reasoning::eval_expression(expr_str, bindings);
        return PyFloat_FromDouble(result);
    } catch (const prime::reasoning::ExprParseError& e) {
        PyErr_SetString(ReasoningError, e.what());
        return nullptr;
    } catch (const prime::reasoning::ExprEvalError& e) {
        PyErr_SetString(ReasoningError, e.what());
        return nullptr;
    } catch (const std::exception& e) {
        PyErr_SetString(ReasoningError, e.what());
        return nullptr;
    }
}

// check_constraint(left_name, op, right, facts_dict) → str ("satisfied" / "violated" / "unknown")
static PyObject* py_check_constraint(PyObject* /*self*/, PyObject* args) {
    const char* left;
    const char* op_str;
    const char* right;
    PyObject*   facts_dict;
    if (!PyArg_ParseTuple(args, "sssO!", &left, &op_str, &right, &PyDict_Type, &facts_dict))
        return nullptr;

    using namespace prime::reasoning;
    std::unordered_map<std::string, Fact> facts;
    PyObject* key;
    PyObject* val;
    Py_ssize_t pos = 0;
    while (PyDict_Next(facts_dict, &pos, &key, &val)) {
        if (!PyUnicode_Check(key)) continue;
        std::string ks = PyUnicode_AsUTF8(key);
        Fact f;
        f.id = f.name = ks;
        f.source = "check";
        f.bound  = true;
        if (PyFloat_Check(val)) f.value = FactValue{PyFloat_AsDouble(val)};
        else if (PyLong_Check(val)) f.value = FactValue{(double)PyLong_AsLong(val)};
        else if (PyBool_Check(val)) f.value = FactValue{val == Py_True};
        else if (PyUnicode_Check(val)) {
            std::string sv = PyUnicode_AsUTF8(val);
            try { f.value = FactValue{std::stod(sv)}; } catch (...) { f.value = FactValue{sv}; }
        } else f.value = FactValue{ks};
        facts[ks] = f;
    }

    ReasoningConstraint c = atlas_constraint_to_reasoning("c0", left, op_str, right, 1.0);
    ConstraintEngine ce;
    ReasoningTrace trace;
    auto r = ce.check(c, facts);
    const char* rs = (r.result == ConstraintResult::SATISFIED) ? "satisfied"
                   : (r.result == ConstraintResult::VIOLATED)  ? "violated"
                   : "unknown";
    return PyUnicode_FromString(rs);
}

// ---------------------------------------------------------------------------
// Method table
// ---------------------------------------------------------------------------
static PyMethodDef methods[] = {
    {"solve_context",     py_solve_context,     METH_VARARGS,
     "solve_context(ctx_dict) -> result_dict\n"
     "Solve a reasoning context dict. Returns a result dict."},
    {"eval_expression",   py_eval_expression,   METH_VARARGS,
     "eval_expression(expr_str, bindings) -> float\n"
     "Evaluate an arithmetic expression string with variable bindings."},
    {"check_constraint",  py_check_constraint,  METH_VARARGS,
     "check_constraint(left, op, right, facts) -> str\n"
     "Check whether a constraint is satisfied/violated/unknown."},
    {nullptr, nullptr, 0, nullptr},
};

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------
static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "_prime_reasoning",
    "PRIME Phase 5 — Reasoning Engine C extension",
    -1,
    methods,
};

PyMODINIT_FUNC PyInit__prime_reasoning(void) {
    PyObject* m = PyModule_Create(&module_def);
    if (!m) return nullptr;

    // Create and register ReasoningError
    ReasoningError = PyErr_NewException("_prime_reasoning.ReasoningError", nullptr, nullptr);
    Py_INCREF(ReasoningError);
    PyModule_AddObject(m, "ReasoningError", ReasoningError);

    PyModule_AddStringConstant(m, "VERSION", "0.5.0.dev1");

    return m;
}
