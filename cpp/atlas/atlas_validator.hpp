#pragma once
/*
 * atlas_validator.hpp — Validates a UniversalProblem representation.
 *
 * Checks required fields, ID uniqueness, relationship endpoint validity,
 * confidence ranges, and structural integrity.
 */

#include "atlas_types.hpp"
#include <unordered_set>
#include <vector>
#include <string>
#include <cmath>

namespace prime {
namespace atlas {

struct ValidationResult {
    bool                     valid = true;
    std::vector<std::string> errors;
    std::vector<std::string> warnings;

    void add_error(const std::string& msg) {
        valid = false;
        errors.push_back(msg);
    }
    void add_warning(const std::string& msg) {
        warnings.push_back(msg);
    }
    bool ok() const { return valid; }
};

class AtlasValidator {
public:
    ValidationResult validate(const UniversalProblem& p) const {
        ValidationResult r;

        // Problem-level
        if (p.problem_id.empty())
            r.add_error("problem_id is empty");
        if (p.atlas_schema_version.empty())
            r.add_error("atlas_schema_version is empty");
        if (p.atlas_schema_version != ATLAS_SCHEMA_VERSION)
            r.add_error("Unsupported atlas_schema_version: " + p.atlas_schema_version);

        // Collect all entity IDs for reference checking
        std::unordered_set<std::string> entity_ids;
        std::unordered_set<std::string> all_ids;

        // Entities
        for (auto& e : p.entities) {
            if (e.id.empty())
                r.add_error("Entity with empty id");
            else if (!all_ids.insert(e.id).second)
                r.add_error("Duplicate entity id: " + e.id);
            else
                entity_ids.insert(e.id);

            if (e.name.empty())
                r.add_warning("Entity '" + e.id + "' has empty name");
            if (!valid_confidence(e.confidence))
                r.add_error("Entity '" + e.id + "' has invalid confidence: " +
                            std::to_string(e.confidence));
            if (e.quantity) {
                if (std::isnan(e.quantity->value) || std::isinf(e.quantity->value))
                    r.add_error("Entity '" + e.id + "' quantity value is NaN/Inf");
            }
        }

        // Relationships
        std::unordered_set<std::string> rel_ids;
        for (auto& rel : p.relationships) {
            if (rel.id.empty())
                r.add_error("Relationship with empty id");
            else if (!rel_ids.insert(rel.id).second)
                r.add_error("Duplicate relationship id: " + rel.id);
            if (rel.relation.empty())
                r.add_warning("Relationship '" + rel.id + "' has empty relation");
            if (!valid_confidence(rel.confidence))
                r.add_error("Relationship '" + rel.id + "' has invalid confidence");
            // Endpoints may be entity IDs or free-form labels — only warn if both empty
            if (rel.source_id.empty() || rel.target_id.empty())
                r.add_warning("Relationship '" + rel.id + "' has empty source or target");
        }

        // Constraints
        std::unordered_set<std::string> con_ids;
        for (auto& c : p.constraints) {
            if (c.id.empty())
                r.add_error("Constraint with empty id");
            else if (!con_ids.insert(c.id).second)
                r.add_error("Duplicate constraint id: " + c.id);
            if (c.left.empty())
                r.add_warning("Constraint '" + c.id + "' has empty left side");
            if (!valid_confidence(c.confidence))
                r.add_error("Constraint '" + c.id + "' has invalid confidence");
        }

        // Goals
        std::unordered_set<std::string> goal_ids;
        for (auto& g : p.goals) {
            if (g.id.empty())
                r.add_error("Goal with empty id");
            else if (!goal_ids.insert(g.id).second)
                r.add_error("Duplicate goal id: " + g.id);
            if (g.description.empty())
                r.add_warning("Goal '" + g.id + "' has empty description");
            if (!valid_confidence(g.confidence))
                r.add_error("Goal '" + g.id + "' has invalid confidence");
        }

        // Operations
        std::unordered_set<std::string> op_ids;
        for (auto& op : p.operations) {
            if (op.id.empty())
                r.add_error("Operation with empty id");
            else if (!op_ids.insert(op.id).second)
                r.add_error("Duplicate operation id: " + op.id);
            if (op.type.empty())
                r.add_warning("Operation '" + op.id + "' has empty type");
        }

        // Expressions
        std::unordered_set<std::string> expr_ids;
        for (auto& ex : p.expressions) {
            if (ex.id.empty())
                r.add_error("Expression with empty id");
            else if (!expr_ids.insert(ex.id).second)
                r.add_error("Duplicate expression id: " + ex.id);
            if (ex.op.empty())
                r.add_warning("Expression '" + ex.id + "' has empty operator");
        }

        // States
        for (auto& s : p.states) {
            if (s.id.empty())
                r.add_error("State with empty id");
            if (s.label.empty())
                r.add_warning("State '" + s.id + "' has empty label");
        }

        // Assumptions
        for (auto& a : p.assumptions) {
            if (a.id.empty())
                r.add_error("Assumption with empty id");
            if (!valid_confidence(a.confidence))
                r.add_error("Assumption '" + a.id + "' has invalid confidence");
        }

        // Evidence
        for (auto& ev : p.evidence) {
            if (ev.id.empty())
                r.add_error("Evidence with empty id");
            if (!valid_confidence(ev.confidence))
                r.add_error("Evidence '" + ev.id + "' has invalid confidence");
        }

        // Graph node/edge basic checks
        std::unordered_set<std::string> graph_node_ids;
        for (auto& n : p.graph.nodes) {
            if (n.id.empty())
                r.add_error("Graph node with empty id");
            else
                graph_node_ids.insert(n.id);
        }
        for (auto& e : p.graph.edges) {
            if (e.from_id.empty() || e.to_id.empty())
                r.add_warning("Graph edge with empty endpoint");
        }

        return r;
    }

private:
    static bool valid_confidence(double c) {
        return c >= 0.0 && c <= 1.0;
    }
};

} // namespace atlas
} // namespace prime
