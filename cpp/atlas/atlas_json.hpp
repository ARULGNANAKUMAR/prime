#pragma once
/*
 * atlas_json.hpp — Hand-rolled JSON serialization/deserialization for ATLAS types.
 *
 * No external JSON library required. Uses the same pattern as the Phase 3
 * memory engine (hand-rolled, zero external deps).
 *
 * Public API:
 *   std::string atlas_to_json(const UniversalProblem&);
 *   UniversalProblem atlas_from_json(const std::string&);
 */

#include "atlas_types.hpp"
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <cctype>

namespace prime {
namespace atlas {

// ---------------------------------------------------------------------------
// Minimal JSON writer helpers
// ---------------------------------------------------------------------------
namespace detail {

static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\t";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += (char)c;
                }
        }
    }
    return out;
}

static std::string jstr(const std::string& s) {
    return "\"" + json_escape(s) + "\"";
}

static std::string jkv(const std::string& k, const std::string& v_json) {
    return jstr(k) + ": " + v_json;
}

static std::string jkvs(const std::string& k, const std::string& v) {
    return jkv(k, jstr(v));
}

static std::string jkvd(const std::string& k, double v) {
    if (std::isnan(v) || std::isinf(v)) return jkv(k, "0.0");
    std::ostringstream oss;
    oss << v;
    return jkv(k, oss.str());
}

static std::string jkvi(const std::string& k, int v) {
    return jkv(k, std::to_string(v));
}


static std::string obj(const std::vector<std::string>& fields) {
    std::string out = "{";
    for (std::size_t i = 0; i < fields.size(); ++i) {
        if (i) out += ", ";
        out += fields[i];
    }
    return out + "}";
}

static std::string arr(const std::vector<std::string>& items) {
    std::string out = "[";
    for (std::size_t i = 0; i < items.size(); ++i) {
        if (i) out += ", ";
        out += items[i];
    }
    return out + "]";
}

static std::string meta_to_json(const std::unordered_map<std::string,std::string>& m) {
    std::vector<std::string> pairs;
    pairs.reserve(m.size());
    for (auto& [k,v] : m) pairs.push_back(jkvs(k, v));
    return obj(pairs);
}

static std::string quantity_to_json(const Quantity& q) {
    return obj({ jkvd("value", q.value), jkvs("unit", q.unit), jkvs("dimension", q.dimension) });
}

static std::string strvec_to_json(const std::vector<std::string>& v) {
    std::vector<std::string> items;
    for (auto& s : v) items.push_back(jstr(s));
    return arr(items);
}

// ---- Serialize sub-types -----------------------------------------------

static std::string attribute_to_json(const Attribute& a) {
    std::vector<std::string> f = {
        jkvs("id", a.id),
        jkvs("name", a.name),
        jkvs("value_text", a.value_text)
    };
    if (a.quantity) f.push_back(jkv("quantity", quantity_to_json(*a.quantity)));
    else            f.push_back(jkv("quantity", "null"));
    f.push_back(jkv("metadata", meta_to_json(a.metadata)));
    return obj(f);
}

static std::string entity_to_json(const Entity& e) {
    std::vector<std::string> attrs_json;
    for (auto& a : e.attributes) attrs_json.push_back(attribute_to_json(a));
    std::vector<std::string> f = {
        jkvs("id", e.id),
        jkvs("type", e.type),
        jkvs("name", e.name),
        jkv("quantity", e.quantity ? quantity_to_json(*e.quantity) : "null"),
        jkv("attributes", arr(attrs_json)),
        jkvd("confidence", e.confidence),
        jkv("metadata", meta_to_json(e.metadata))
    };
    return obj(f);
}

static std::string relationship_to_json(const Relationship& r) {
    return obj({
        jkvs("id", r.id),
        jkvs("source_id", r.source_id),
        jkvs("relation", r.relation),
        jkvs("target_id", r.target_id),
        jkvd("confidence", r.confidence),
        jkv("metadata", meta_to_json(r.metadata))
    });
}

static std::string constraint_to_json(const Constraint& c) {
    return obj({
        jkvs("id", c.id),
        jkvs("left", c.left),
        jkvs("op", constraint_op_to_string(c.op)),
        jkvs("right", c.right),
        jkvs("type", c.type),
        jkvd("confidence", c.confidence),
        jkv("metadata", meta_to_json(c.metadata))
    });
}

static std::string goal_to_json(const Goal& g) {
    return obj({
        jkvs("id", g.id),
        jkvs("description", g.description),
        jkvs("target", g.target),
        jkvs("desired_state", g.desired_state),
        jkvi("priority", g.priority),
        jkvd("confidence", g.confidence),
        jkv("metadata", meta_to_json(g.metadata))
    });
}

static std::string operation_to_json(const Operation& op) {
    return obj({
        jkvs("id", op.id),
        jkvs("type", op.type),
        jkv("input_ids",  strvec_to_json(op.input_ids)),
        jkv("output_ids", strvec_to_json(op.output_ids)),
        jkv("metadata",   meta_to_json(op.metadata))
    });
}

static std::string expression_to_json(const Expression& ex) {
    return obj({
        jkvs("id", ex.id),
        jkvs("op", ex.op),
        jkv("operand_ids", strvec_to_json(ex.operand_ids)),
        jkvs("output_id", ex.output_id),
        jkvs("text", ex.text),
        jkv("metadata", meta_to_json(ex.metadata))
    });
}

static std::string state_to_json(const State& s) {
    return obj({
        jkvs("id", s.id),
        jkvs("label", s.label),
        jkv("entity_ids", strvec_to_json(s.entity_ids)),
        jkv("constraint_ids", strvec_to_json(s.constraint_ids)),
        jkv("metadata", meta_to_json(s.metadata))
    });
}

static std::string assumption_to_json(const Assumption& a) {
    return obj({
        jkvs("id", a.id),
        jkvs("description", a.description),
        jkvd("confidence", a.confidence),
        jkv("metadata", meta_to_json(a.metadata))
    });
}

static std::string evidence_to_json(const Evidence& ev) {
    return obj({
        jkvs("id", ev.id),
        jkvs("source", ev.source),
        jkvs("content", ev.content),
        jkvd("confidence", ev.confidence),
        jkv("metadata", meta_to_json(ev.metadata))
    });
}

static std::string logic_clause_to_json(const LogicClause& lc) {
    return obj({
        jkvs("id", lc.id),
        jkvs("op", logic_op_to_string(lc.op)),
        jkvs("condition", lc.condition),
        jkvs("implication", lc.implication),
        jkv("metadata", meta_to_json(lc.metadata))
    });
}

static std::string graph_node_to_json(const GraphNode& n) {
    return obj({ jkvs("id", n.id), jkvs("kind", n.kind), jkvs("ref", n.ref) });
}

static std::string graph_edge_to_json(const GraphEdge& e) {
    return obj({ jkvs("from_id", e.from_id), jkvs("to_id", e.to_id), jkvs("label", e.label) });
}

} // namespace detail

// ---------------------------------------------------------------------------
// Public: atlas_to_json
// ---------------------------------------------------------------------------
inline std::string atlas_to_json(const UniversalProblem& p) {
    using namespace detail;

    auto vec_to_json = [](const auto& vec, auto fn) -> std::string {
        std::vector<std::string> items;
        items.reserve(vec.size());
        for (auto& x : vec) items.push_back(fn(x));
        return arr(items);
    };

    // Graph
    std::vector<std::string> gnodes, gedges;
    for (auto& n : p.graph.nodes) gnodes.push_back(graph_node_to_json(n));
    for (auto& e : p.graph.edges) gedges.push_back(graph_edge_to_json(e));
    std::string graph_json = obj({
        jkv("nodes", arr(gnodes)),
        jkv("edges", arr(gedges))
    });

    std::vector<std::string> fields = {
        jkvs("atlas_schema_version", p.atlas_schema_version),
        jkvs("problem_id", p.problem_id),
        jkvs("original_input", p.original_input),
        jkvs("normalized_input", p.normalized_input),
        jkvs("domain", domain_to_string(p.domain)),
        jkvd("domain_confidence", p.domain_confidence),
        jkv("entities",      vec_to_json(p.entities,      entity_to_json)),
        jkv("relationships", vec_to_json(p.relationships, relationship_to_json)),
        jkv("constraints",   vec_to_json(p.constraints,   constraint_to_json)),
        jkv("goals",         vec_to_json(p.goals,         goal_to_json)),
        jkv("operations",    vec_to_json(p.operations,    operation_to_json)),
        jkv("expressions",   vec_to_json(p.expressions,   expression_to_json)),
        jkv("states",        vec_to_json(p.states,        state_to_json)),
        jkv("assumptions",   vec_to_json(p.assumptions,   assumption_to_json)),
        jkv("evidence",      vec_to_json(p.evidence,      evidence_to_json)),
        jkv("logic_clauses", vec_to_json(p.logic_clauses, logic_clause_to_json)),
        jkv("graph",         graph_json),
        jkv("metadata",      meta_to_json(p.metadata))
    };
    return detail::obj(fields);
}

// ---------------------------------------------------------------------------
// Minimal JSON parser — tokenizer + value extractor
// ---------------------------------------------------------------------------
namespace detail {

struct JsonTok {
    enum Type { LBRACE, RBRACE, LBRACKET, RBRACKET, COLON, COMMA,
                STRING, NUMBER, BOOL_TRUE, BOOL_FALSE, NULL_VAL, END };
    Type        type;
    std::string sval;
    double      dval = 0.0;
};

struct JsonLexer {
    const std::string& src;
    std::size_t        pos = 0;

    explicit JsonLexer(const std::string& s) : src(s) {}

    void skip_ws() {
        while (pos < src.size() && std::isspace((unsigned char)src[pos])) ++pos;
    }

    JsonTok next() {
        skip_ws();
        if (pos >= src.size()) return {JsonTok::END};
        char c = src[pos];
        switch (c) {
            case '{': ++pos; return {JsonTok::LBRACE};
            case '}': ++pos; return {JsonTok::RBRACE};
            case '[': ++pos; return {JsonTok::LBRACKET};
            case ']': ++pos; return {JsonTok::RBRACKET};
            case ':': ++pos; return {JsonTok::COLON};
            case ',': ++pos; return {JsonTok::COMMA};
            case '"': return read_string();
            case 't': pos += 4; return {JsonTok::BOOL_TRUE,  "true",  1.0};
            case 'f': pos += 5; return {JsonTok::BOOL_FALSE, "false", 0.0};
            case 'n': pos += 4; return {JsonTok::NULL_VAL,   "null"};
            default:
                if (c == '-' || std::isdigit((unsigned char)c)) return read_number();
                throw AtlasError(std::string("Unexpected char in JSON: ") + c);
        }
    }

    JsonTok peek() {
        std::size_t saved = pos;
        JsonTok t = next();
        pos = saved;
        return t;
    }

    JsonTok read_string() {
        ++pos; // skip opening "
        std::string out;
        while (pos < src.size()) {
            char c = src[pos++];
            if (c == '"') break;
            if (c == '\\' && pos < src.size()) {
                char e = src[pos++];
                switch (e) {
                    case '"':  out += '"';  break;
                    case '\\': out += '\\'; break;
                    case '/':  out += '/';  break;
                    case 'n':  out += '\n'; break;
                    case 'r':  out += '\r'; break;
                    case 't':  out += '\t'; break;
                    case 'u': {
                        // minimal: skip 4 hex digits, replace with '?'
                        pos += 4;
                        out += '?';
                        break;
                    }
                    default:   out += e;    break;
                }
            } else {
                out += c;
            }
        }
        return {JsonTok::STRING, out};
    }

    JsonTok read_number() {
        std::size_t start = pos;
        if (src[pos] == '-') ++pos;
        while (pos < src.size() && (std::isdigit((unsigned char)src[pos]) || src[pos] == '.' ||
               src[pos] == 'e' || src[pos] == 'E' || src[pos] == '+' || src[pos] == '-')) {
            ++pos;
        }
        std::string ns = src.substr(start, pos - start);
        JsonTok t;
        t.type  = JsonTok::NUMBER;
        t.sval  = ns;
        t.dval  = std::stod(ns);
        return t;
    }
};

// Simple JSON value representation for parsing
struct JValue {
    enum Kind { OBJECT, ARRAY, STRING, NUMBER, BOOL_V, NULL_V };
    Kind                                     kind = NULL_V;
    std::string                              sval;
    double                                   dval = 0.0;
    bool                                     bval = false;
    std::vector<JValue>                      arr_val;
    std::vector<std::pair<std::string,JValue>> obj_val;

    const JValue* get(const std::string& k) const {
        for (auto& [key, val] : obj_val) if (key == k) return &val;
        return nullptr;
    }
    std::string str(const std::string& k, const std::string& def = "") const {
        auto* v = get(k);
        if (!v || v->kind != STRING) return def;
        return v->sval;
    }
    double num(const std::string& k, double def = 0.0) const {
        auto* v = get(k);
        if (!v || v->kind != NUMBER) return def;
        return v->dval;
    }
    int inum(const std::string& k, int def = 0) const {
        return static_cast<int>(num(k, static_cast<double>(def)));
    }
};

static JValue parse_value(JsonLexer& lex);

static JValue parse_object(JsonLexer& lex) {
    JValue v;
    v.kind = JValue::OBJECT;
    JsonTok t = lex.next(); // consume {
    if (t.type != JsonTok::LBRACE) throw AtlasError("Expected '{'");
    while (true) {
        t = lex.peek();
        if (t.type == JsonTok::RBRACE) { lex.next(); break; }
        if (t.type == JsonTok::COMMA)  { lex.next(); continue; }
        JsonTok key = lex.next();
        if (key.type != JsonTok::STRING) throw AtlasError("Expected string key");
        JsonTok colon = lex.next();
        if (colon.type != JsonTok::COLON) throw AtlasError("Expected ':'");
        JValue val = parse_value(lex);
        v.obj_val.push_back({key.sval, std::move(val)});
    }
    return v;
}

static JValue parse_array(JsonLexer& lex) {
    JValue v;
    v.kind = JValue::ARRAY;
    JsonTok t = lex.next(); // consume [
    if (t.type != JsonTok::LBRACKET) throw AtlasError("Expected '['");
    while (true) {
        t = lex.peek();
        if (t.type == JsonTok::RBRACKET) { lex.next(); break; }
        if (t.type == JsonTok::COMMA)    { lex.next(); continue; }
        v.arr_val.push_back(parse_value(lex));
    }
    return v;
}

static JValue parse_value(JsonLexer& lex) {
    JsonTok t = lex.peek();
    if (t.type == JsonTok::LBRACE)   return parse_object(lex);
    if (t.type == JsonTok::LBRACKET) return parse_array(lex);
    lex.next();
    JValue v;
    switch (t.type) {
        case JsonTok::STRING:     v.kind = JValue::STRING; v.sval = t.sval; break;
        case JsonTok::NUMBER:     v.kind = JValue::NUMBER; v.dval = t.dval; break;
        case JsonTok::BOOL_TRUE:  v.kind = JValue::BOOL_V; v.bval = true;   break;
        case JsonTok::BOOL_FALSE: v.kind = JValue::BOOL_V; v.bval = false;  break;
        default:                  v.kind = JValue::NULL_V; break;
    }
    return v;
}

// Helpers to reconstruct C++ objects from JValue
static Quantity quantity_from_jv(const JValue& jv) {
    return { jv.num("value"), jv.str("unit"), jv.str("dimension") };
}

static std::unordered_map<std::string,std::string> meta_from_jv(const JValue* jv) {
    std::unordered_map<std::string,std::string> m;
    if (!jv || jv->kind != JValue::OBJECT) return m;
    for (auto& [k,v] : jv->obj_val)
        if (v.kind == JValue::STRING) m[k] = v.sval;
    return m;
}

static std::vector<std::string> strvec_from_jv(const JValue* jv) {
    std::vector<std::string> out;
    if (!jv || jv->kind != JValue::ARRAY) return out;
    for (auto& v : jv->arr_val)
        if (v.kind == JValue::STRING) out.push_back(v.sval);
    return out;
}

static Attribute attribute_from_jv(const JValue& jv) {
    Attribute a;
    a.id         = jv.str("id");
    a.name       = jv.str("name");
    a.value_text = jv.str("value_text");
    auto* q = jv.get("quantity");
    if (q && q->kind == JValue::OBJECT) a.quantity = quantity_from_jv(*q);
    a.metadata = meta_from_jv(jv.get("metadata"));
    return a;
}

static Entity entity_from_jv(const JValue& jv) {
    Entity e;
    e.id   = jv.str("id");
    e.type = jv.str("type");
    e.name = jv.str("name");
    auto* q = jv.get("quantity");
    if (q && q->kind == JValue::OBJECT) e.quantity = quantity_from_jv(*q);
    e.confidence = jv.num("confidence", 1.0);
    e.metadata   = meta_from_jv(jv.get("metadata"));
    auto* attrs = jv.get("attributes");
    if (attrs && attrs->kind == JValue::ARRAY)
        for (auto& av : attrs->arr_val)
            if (av.kind == JValue::OBJECT) e.attributes.push_back(attribute_from_jv(av));
    return e;
}

static Relationship relationship_from_jv(const JValue& jv) {
    return { jv.str("id"), jv.str("source_id"), jv.str("relation"),
             jv.str("target_id"), jv.num("confidence", 1.0),
             meta_from_jv(jv.get("metadata")) };
}

static Constraint constraint_from_jv(const JValue& jv) {
    return { jv.str("id"), jv.str("left"),
             constraint_op_from_string(jv.str("op")),
             jv.str("right"), jv.str("type"),
             jv.num("confidence", 1.0),
             meta_from_jv(jv.get("metadata")) };
}

static Goal goal_from_jv(const JValue& jv) {
    return { jv.str("id"), jv.str("description"), jv.str("target"),
             jv.str("desired_state"), jv.inum("priority", 0),
             jv.num("confidence", 1.0),
             meta_from_jv(jv.get("metadata")) };
}

static Operation operation_from_jv(const JValue& jv) {
    return { jv.str("id"), jv.str("type"),
             strvec_from_jv(jv.get("input_ids")),
             strvec_from_jv(jv.get("output_ids")),
             meta_from_jv(jv.get("metadata")) };
}

static Expression expression_from_jv(const JValue& jv) {
    return { jv.str("id"), jv.str("op"),
             strvec_from_jv(jv.get("operand_ids")),
             jv.str("output_id"), jv.str("text"),
             meta_from_jv(jv.get("metadata")) };
}

static State state_from_jv(const JValue& jv) {
    return { jv.str("id"), jv.str("label"),
             strvec_from_jv(jv.get("entity_ids")),
             strvec_from_jv(jv.get("constraint_ids")),
             meta_from_jv(jv.get("metadata")) };
}

static Assumption assumption_from_jv(const JValue& jv) {
    return { jv.str("id"), jv.str("description"),
             jv.num("confidence", 1.0),
             meta_from_jv(jv.get("metadata")) };
}

static Evidence evidence_from_jv(const JValue& jv) {
    return { jv.str("id"), jv.str("source"), jv.str("content"),
             jv.num("confidence", 1.0),
             meta_from_jv(jv.get("metadata")) };
}

static LogicOp logic_op_from_string(const std::string& s) {
    if (s == "AND")  return LogicOp::AND;
    if (s == "OR")   return LogicOp::OR;
    if (s == "NOT")  return LogicOp::NOT;
    if (s == "IF")   return LogicOp::IF;
    if (s == "THEN") return LogicOp::THEN;
    return LogicOp::UNKNOWN;
}

static LogicClause logic_clause_from_jv(const JValue& jv) {
    return { jv.str("id"),
             logic_op_from_string(jv.str("op")),
             jv.str("condition"), jv.str("implication"),
             meta_from_jv(jv.get("metadata")) };
}

} // namespace detail

// ---------------------------------------------------------------------------
// Public: atlas_from_json
// ---------------------------------------------------------------------------
inline UniversalProblem atlas_from_json(const std::string& json_text) {
    using namespace detail;

    if (json_text.empty()) throw AtlasError("Empty JSON input");
    if (json_text.size() > 16 * 1024 * 1024)
        throw AtlasError("JSON input exceeds maximum size (16 MB)");

    JsonLexer lex(json_text);
    JValue root = parse_object(lex);

    // Schema version check
    std::string schema_ver = root.str("atlas_schema_version");
    if (!schema_ver.empty() && schema_ver != ATLAS_SCHEMA_VERSION)
        throw AtlasError("Unsupported atlas_schema_version: " + schema_ver);

    UniversalProblem p;
    p.atlas_schema_version = schema_ver.empty() ? std::string(ATLAS_SCHEMA_VERSION) : schema_ver;
    p.problem_id           = root.str("problem_id");
    p.original_input       = root.str("original_input");
    p.normalized_input     = root.str("normalized_input");
    p.domain               = domain_from_string(root.str("domain", "unknown"));
    p.domain_confidence    = root.num("domain_confidence", 0.0);
    p.metadata             = meta_from_jv(root.get("metadata"));

    auto load_vec = [&](const char* key, auto& vec, auto fn) {
        auto* jv = root.get(key);
        if (!jv || jv->kind != JValue::ARRAY) return;
        for (auto& item : jv->arr_val)
            if (item.kind == JValue::OBJECT) vec.push_back(fn(item));
    };

    load_vec("entities",      p.entities,      entity_from_jv);
    load_vec("relationships", p.relationships, relationship_from_jv);
    load_vec("constraints",   p.constraints,   constraint_from_jv);
    load_vec("goals",         p.goals,         goal_from_jv);
    load_vec("operations",    p.operations,    operation_from_jv);
    load_vec("expressions",   p.expressions,   expression_from_jv);
    load_vec("states",        p.states,        state_from_jv);
    load_vec("assumptions",   p.assumptions,   assumption_from_jv);
    load_vec("evidence",      p.evidence,      evidence_from_jv);
    load_vec("logic_clauses", p.logic_clauses, logic_clause_from_jv);

    // Graph
    auto* gjv = root.get("graph");
    if (gjv && gjv->kind == JValue::OBJECT) {
        auto* nodes_jv = gjv->get("nodes");
        if (nodes_jv && nodes_jv->kind == JValue::ARRAY)
            for (auto& n : nodes_jv->arr_val)
                if (n.kind == JValue::OBJECT)
                    p.graph.nodes.push_back({n.str("id"), n.str("kind"), n.str("ref")});
        auto* edges_jv = gjv->get("edges");
        if (edges_jv && edges_jv->kind == JValue::ARRAY)
            for (auto& e : edges_jv->arr_val)
                if (e.kind == JValue::OBJECT)
                    p.graph.edges.push_back({e.str("from_id"), e.str("to_id"), e.str("label")});
    }

    return p;
}

} // namespace atlas
} // namespace prime
