#pragma once
/*
 * atlas_types.hpp — Core data types for ATLAS: Universal Problem Representation Engine.
 *
 * ATLAS converts heterogeneous problem inputs into a structured, domain-independent
 * UniversalProblem representation that future PRIME reasoning phases can consume.
 *
 * ATLAS does NOT solve problems. It represents them.
 *
 * Schema version: 1.0
 */

#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include <stdexcept>
#include <cstdint>
#include <algorithm>
#include <sstream>

namespace prime {
namespace atlas {

// ---------------------------------------------------------------------------
// Schema version
// ---------------------------------------------------------------------------
constexpr const char* ATLAS_SCHEMA_VERSION = "1.0";

// ---------------------------------------------------------------------------
// Resource limits (defaults; configurable)
// ---------------------------------------------------------------------------
struct AtlasLimits {
    std::size_t max_input_bytes      = 1024 * 1024;  // 1 MB
    std::size_t max_entities         = 512;
    std::size_t max_relationships    = 1024;
    std::size_t max_constraints      = 256;
    std::size_t max_goals            = 64;
    std::size_t max_operations       = 128;
    std::size_t max_graph_nodes      = 2048;
};

// ---------------------------------------------------------------------------
// AtlasError
// ---------------------------------------------------------------------------
class AtlasError : public std::runtime_error {
public:
    explicit AtlasError(const std::string& msg) : std::runtime_error(msg) {}
};

// ---------------------------------------------------------------------------
// Domain classification
// ---------------------------------------------------------------------------
enum class Domain {
    MATHEMATICS,
    PHYSICS,
    PROGRAMMING,
    BUSINESS,
    ENGINEERING,
    BIOLOGY,
    GENERAL,
    UNKNOWN
};

inline std::string domain_to_string(Domain d) {
    switch (d) {
        case Domain::MATHEMATICS:  return "mathematics";
        case Domain::PHYSICS:      return "physics";
        case Domain::PROGRAMMING:  return "programming";
        case Domain::BUSINESS:     return "business";
        case Domain::ENGINEERING:  return "engineering";
        case Domain::BIOLOGY:      return "biology";
        case Domain::GENERAL:      return "general";
        default:                   return "unknown";
    }
}

inline Domain domain_from_string(const std::string& s) {
    if (s == "mathematics")  return Domain::MATHEMATICS;
    if (s == "physics")      return Domain::PHYSICS;
    if (s == "programming")  return Domain::PROGRAMMING;
    if (s == "business")     return Domain::BUSINESS;
    if (s == "engineering")  return Domain::ENGINEERING;
    if (s == "biology")      return Domain::BIOLOGY;
    if (s == "general")      return Domain::GENERAL;
    return Domain::UNKNOWN;
}

// ---------------------------------------------------------------------------
// Quantity — a value with a unit and optional dimension
// ---------------------------------------------------------------------------
struct Quantity {
    double      value     = 0.0;
    std::string unit;       // e.g. "km", "kg", "s", "m/s", "%"
    std::string dimension;  // e.g. "length", "mass", "time", "speed", "temperature"

    bool has_unit() const { return !unit.empty(); }

    std::string to_string() const {
        std::ostringstream oss;
        oss << value;
        if (!unit.empty()) oss << " " << unit;
        return oss.str();
    }
};

// ---------------------------------------------------------------------------
// Attribute — generic key/value with optional Quantity
// ---------------------------------------------------------------------------
struct Attribute {
    std::string                id;
    std::string                name;
    std::string                value_text;          // raw string value
    std::optional<Quantity>    quantity;            // parsed numeric quantity
    std::unordered_map<std::string,std::string> metadata;
};

// ---------------------------------------------------------------------------
// Entity — a named object, concept, or quantity in the problem
// ---------------------------------------------------------------------------
struct Entity {
    std::string                              id;
    std::string                              type;      // "object", "quantity", "concept", "variable"
    std::string                              name;
    std::optional<Quantity>                  quantity;  // set when entity is a measured value
    std::vector<Attribute>                   attributes;
    double                                   confidence = 1.0;
    std::unordered_map<std::string,std::string> metadata;
};

// ---------------------------------------------------------------------------
// Relationship — a directed edge between two entity IDs
// ---------------------------------------------------------------------------
struct Relationship {
    std::string id;
    std::string source_id;   // entity or node id
    std::string relation;    // "travels", "has_unit", "depends_on", "equals", ...
    std::string target_id;   // entity or node id
    double      confidence   = 1.0;
    std::unordered_map<std::string,std::string> metadata;
};

// ---------------------------------------------------------------------------
// Constraint — a logical or mathematical constraint on entities
// ---------------------------------------------------------------------------
enum class ConstraintOp {
    EQ, NEQ, LT, LTE, GT, GTE, IN, NOT_IN, CUSTOM
};

inline std::string constraint_op_to_string(ConstraintOp op) {
    switch (op) {
        case ConstraintOp::EQ:     return "=";
        case ConstraintOp::NEQ:    return "!=";
        case ConstraintOp::LT:     return "<";
        case ConstraintOp::LTE:    return "<=";
        case ConstraintOp::GT:     return ">";
        case ConstraintOp::GTE:    return ">=";
        case ConstraintOp::IN:     return "in";
        case ConstraintOp::NOT_IN: return "not_in";
        default:                   return "custom";
    }
}

inline ConstraintOp constraint_op_from_string(const std::string& s) {
    if (s == "=")      return ConstraintOp::EQ;
    if (s == "!=")     return ConstraintOp::NEQ;
    if (s == "<")      return ConstraintOp::LT;
    if (s == "<=")     return ConstraintOp::LTE;
    if (s == ">")      return ConstraintOp::GT;
    if (s == ">=")     return ConstraintOp::GTE;
    if (s == "in")     return ConstraintOp::IN;
    if (s == "not_in") return ConstraintOp::NOT_IN;
    return ConstraintOp::CUSTOM;
}

struct Constraint {
    std::string   id;
    std::string   left;       // entity id or literal
    ConstraintOp  op          = ConstraintOp::EQ;
    std::string   right;      // entity id, literal, or list repr
    std::string   type;       // "mathematical", "logical", "domain", "physical"
    double        confidence  = 1.0;
    std::unordered_map<std::string,std::string> metadata;
};

// ---------------------------------------------------------------------------
// Goal — what the problem asks for
// ---------------------------------------------------------------------------
struct Goal {
    std::string id;
    std::string description;
    std::string target;        // entity id or quantity name to compute/find
    std::string desired_state; // free-form description of the desired outcome
    int         priority       = 0;  // lower = higher priority
    double      confidence     = 1.0;
    std::unordered_map<std::string,std::string> metadata;
};

// ---------------------------------------------------------------------------
// Operation — an action or computation referenced by the problem
// ---------------------------------------------------------------------------
struct Operation {
    std::string              id;
    std::string              type;      // "calculate", "compare", "sort", "classify", ...
    std::vector<std::string> input_ids; // entity or expression ids
    std::vector<std::string> output_ids;
    std::unordered_map<std::string,std::string> metadata;
};

// ---------------------------------------------------------------------------
// Expression — a lightweight structural representation of a formula/expression
// ---------------------------------------------------------------------------
struct Expression {
    std::string              id;
    std::string              op;        // "add", "subtract", "multiply", "divide", "assign", ...
    std::vector<std::string> operand_ids; // entity or expression ids
    std::string              output_id;   // entity id that the result maps to
    std::string              text;        // raw expression text, e.g. "speed = distance / time"
    std::unordered_map<std::string,std::string> metadata;
};

// ---------------------------------------------------------------------------
// State — a snapshot of entity values at a point in problem narrative
// ---------------------------------------------------------------------------
struct State {
    std::string              id;
    std::string              label;       // "initial", "intermediate", "goal"
    std::vector<std::string> entity_ids;
    std::vector<std::string> constraint_ids;
    std::unordered_map<std::string,std::string> metadata;
};

// ---------------------------------------------------------------------------
// Assumption — an explicitly represented assumption
// ---------------------------------------------------------------------------
struct Assumption {
    std::string id;
    std::string description;
    double      confidence = 1.0;
    std::unordered_map<std::string,std::string> metadata;
};

// ---------------------------------------------------------------------------
// Evidence — a piece of evidence/source information
// ---------------------------------------------------------------------------
struct Evidence {
    std::string id;
    std::string source;
    std::string content;
    double      confidence = 1.0;
    std::unordered_map<std::string,std::string> metadata;
};

// ---------------------------------------------------------------------------
// Logical structure — simple IF/AND/OR/NOT representation
// ---------------------------------------------------------------------------
enum class LogicOp { AND, OR, NOT, IF, THEN, UNKNOWN };

inline std::string logic_op_to_string(LogicOp op) {
    switch (op) {
        case LogicOp::AND:  return "AND";
        case LogicOp::OR:   return "OR";
        case LogicOp::NOT:  return "NOT";
        case LogicOp::IF:   return "IF";
        case LogicOp::THEN: return "THEN";
        default:            return "UNKNOWN";
    }
}

struct LogicClause {
    std::string id;
    LogicOp     op           = LogicOp::UNKNOWN;
    std::string condition;   // free-form or constraint_id
    std::string implication; // free-form or constraint_id / goal_id
    std::unordered_map<std::string,std::string> metadata;
};

// ---------------------------------------------------------------------------
// Graph node/edge — lightweight internal graph for the problem
// ---------------------------------------------------------------------------
struct GraphNode {
    std::string id;
    std::string kind;  // "entity", "constraint", "goal", "operation", "expression"
    std::string ref;   // id of the referenced object
};

struct GraphEdge {
    std::string from_id;
    std::string to_id;
    std::string label;
};

struct ProblemGraph {
    std::vector<GraphNode>  nodes;
    std::vector<GraphEdge>  edges;

    void add_node(const std::string& id, const std::string& kind, const std::string& ref) {
        nodes.push_back({id, kind, ref});
    }
    void add_edge(const std::string& from, const std::string& to, const std::string& label) {
        edges.push_back({from, to, label});
    }
    bool has_node(const std::string& id) const {
        for (auto& n : nodes) if (n.id == id) return true;
        return false;
    }
    std::size_t node_count() const { return nodes.size(); }
    std::size_t edge_count() const { return edges.size(); }
};

// ---------------------------------------------------------------------------
// UniversalProblem — the complete ATLAS representation
// ---------------------------------------------------------------------------
struct UniversalProblem {
    std::string  problem_id;
    std::string  atlas_schema_version  = ATLAS_SCHEMA_VERSION;
    std::string  original_input;       // preserved verbatim
    std::string  normalized_input;     // after normalization
    Domain       domain                = Domain::UNKNOWN;
    double       domain_confidence     = 0.0;

    std::vector<Entity>       entities;
    std::vector<Relationship> relationships;
    std::vector<Constraint>   constraints;
    std::vector<Goal>         goals;
    std::vector<Operation>    operations;
    std::vector<Expression>   expressions;
    std::vector<State>        states;
    std::vector<Assumption>   assumptions;
    std::vector<Evidence>     evidence;
    std::vector<LogicClause>  logic_clauses;

    ProblemGraph  graph;

    std::unordered_map<std::string,std::string> metadata;

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------
    const Entity*       find_entity(const std::string& id) const {
        for (auto& e : entities) if (e.id == id) return &e;
        return nullptr;
    }
    const Constraint*   find_constraint(const std::string& id) const {
        for (auto& c : constraints) if (c.id == id) return &c;
        return nullptr;
    }
    const Goal*         find_goal(const std::string& id) const {
        for (auto& g : goals) if (g.id == id) return &g;
        return nullptr;
    }

    // ------------------------------------------------------------------
    // Counters
    // ------------------------------------------------------------------
    std::size_t entity_count()       const { return entities.size(); }
    std::size_t relationship_count() const { return relationships.size(); }
    std::size_t constraint_count()   const { return constraints.size(); }
    std::size_t goal_count()         const { return goals.size(); }
    std::size_t operation_count()    const { return operations.size(); }
    std::size_t expression_count()   const { return expressions.size(); }
};

} // namespace atlas
} // namespace prime
