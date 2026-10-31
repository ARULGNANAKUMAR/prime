#pragma once
// rule_engine.hpp — Lightweight rule engine for PRIME Phase 5.
//
// Supports:
//   - Rule storage with priority ordering
//   - Condition evaluation (=, !=, <, <=, >, >=, AND, OR, NOT, EXISTS)
//   - Forward chaining with loop / depth / step limits
//   - Conflict resolution: priority → confidence → specificity
//   - Circular-fact detection

#include "reasoning_types.hpp"
#include <algorithm>
#include <sstream>

namespace prime::reasoning {

// ---------------------------------------------------------------------------
// ConditionEvaluator
// ---------------------------------------------------------------------------
class ConditionEvaluator {
public:
    // Evaluate a single atomic condition against the fact map.
    static bool evaluate(const Condition& cond,
                         const std::unordered_map<std::string, Fact>& facts) {
        switch (cond.op) {
            case CondOp::AND: {
                for (auto& sub : cond.sub_conditions)
                    if (!evaluate(sub, facts)) return false;
                return true;
            }
            case CondOp::OR: {
                for (auto& sub : cond.sub_conditions)
                    if (evaluate(sub, facts)) return true;
                return false;
            }
            case CondOp::NOT: {
                if (cond.sub_conditions.empty()) return false;
                return !evaluate(cond.sub_conditions[0], facts);
            }
            case CondOp::EXISTS: {
                return facts.count(cond.left) > 0;
            }
            default:
                return evaluate_comparison(cond, facts);
        }
    }

private:
    static bool evaluate_comparison(const Condition& cond,
                                    const std::unordered_map<std::string, Fact>& facts) {
        // Resolve left-hand side
        auto lit = facts.find(cond.left);
        if (lit == facts.end()) return false; // fact not present
        const Fact& left_fact = lit->second;

        // Resolve right-hand side
        FactValue right_val;
        if (cond.right_is_fact) {
            auto rit = facts.find(cond.right);
            if (rit == facts.end()) return false;
            right_val = rit->second.value;
        } else {
            // Parse the string literal
            right_val = parse_literal(cond.right);
        }

        // Numeric comparison
        if (fact_value_is_numeric(left_fact.value) && fact_value_is_numeric(right_val)) {
            double l = fact_value_to_double(left_fact.value);
            double r = fact_value_to_double(right_val);
            switch (cond.op) {
                case CondOp::EQ:  return l == r;
                case CondOp::NEQ: return l != r;
                case CondOp::LT:  return l <  r;
                case CondOp::LTE: return l <= r;
                case CondOp::GT:  return l >  r;
                case CondOp::GTE: return l >= r;
                default: return false;
            }
        }

        // Boolean comparison
        if (std::holds_alternative<bool>(left_fact.value) ||
            std::holds_alternative<bool>(right_val)) {
            bool l = std::holds_alternative<bool>(left_fact.value)
                     ? std::get<bool>(left_fact.value)
                     : (fact_value_to_string(left_fact.value) == "true");
            bool r = std::holds_alternative<bool>(right_val)
                     ? std::get<bool>(right_val)
                     : (fact_value_to_string(right_val) == "true");
            switch (cond.op) {
                case CondOp::EQ:  return l == r;
                case CondOp::NEQ: return l != r;
                default: return false;
            }
        }

        // String comparison (only EQ / NEQ)
        std::string ls = fact_value_to_string(left_fact.value);
        std::string rs = fact_value_to_string(right_val);
        switch (cond.op) {
            case CondOp::EQ:  return ls == rs;
            case CondOp::NEQ: return ls != rs;
            default: return false;
        }
    }

    static FactValue parse_literal(const std::string& s) {
        if (s == "true")  return FactValue{true};
        if (s == "false") return FactValue{false};
        try {
            size_t idx;
            double d = std::stod(s, &idx);
            if (idx == s.size()) return FactValue{d};
        } catch (...) {}
        return FactValue{s};
    }
};

// ---------------------------------------------------------------------------
// RuleEngine
// ---------------------------------------------------------------------------
class RuleEngine {
public:
    // Add a rule to the knowledge base. Returns false if ID already exists.
    bool add_rule(Rule rule) {
        if (rule_index_.count(rule.id)) return false;
        rule_index_[rule.id] = rules_.size();
        rules_.push_back(std::move(rule));
        sorted_ = false;
        return true;
    }

    size_t rule_count() const { return rules_.size(); }

    void clear_rules() {
        rules_.clear();
        rule_index_.clear();
        sorted_ = true;
    }

    // Run forward chaining. Returns new facts derived.
    // Modifies facts in place. Appends steps to trace.
    bool forward_chain(
            std::unordered_map<std::string, Fact>& facts,
            ReasoningTrace& trace,
            const ReasoningConfig& cfg)
    {
        ensure_sorted();
        int  firings     = 0;
        int  depth       = 0;
        bool any_fired   = false;

        // Circular detection: track which (rule_id, key_fact_snapshot) pairs we've seen
        std::unordered_set<std::string> fired_signatures;

        while (depth < cfg.max_depth && firings < cfg.max_rule_firings) {
            bool fired_this_round = false;

            // Collect applicable rules (not fired if one-shot, or priority-ordered)
            std::vector<size_t> applicable;
            for (size_t i = 0; i < rules_.size(); ++i) {
                auto& rule = rules_[i];
                if (!conditions_met(rule, facts)) continue;

                // Build signature to detect circular re-firings
                std::string sig = rule.id + "|" + facts_snapshot(rule, facts);
                if (cfg.detect_circular && fired_signatures.count(sig)) continue;

                applicable.push_back(i);
            }

            if (applicable.empty()) break;

            // Conflict resolution: highest priority first, then confidence, then specificity
            std::stable_sort(applicable.begin(), applicable.end(),
                [this](size_t a, size_t b) {
                    const auto& ra = rules_[a];
                    const auto& rb = rules_[b];
                    if (ra.priority != rb.priority) return ra.priority > rb.priority;
                    if (ra.confidence != rb.confidence) return ra.confidence > rb.confidence;
                    return ra.conditions.size() > rb.conditions.size(); // specificity
                });

            for (size_t idx : applicable) {
                auto& rule = rules_[idx];
                std::string sig = rule.id + "|" + facts_snapshot(rule, facts);
                fired_signatures.insert(sig);

                ReasoningStep step;
                step.type       = StepType::RULE_FIRE;
                step.rule_id    = rule.id;
                step.operation  = "RULE_FIRE";
                step.confidence = rule.confidence;
                step.description = "Rule '" + (rule.name.empty() ? rule.id : rule.name) + "' fired";

                // Collect input facts used by conditions
                collect_condition_facts(rule.conditions, step.input_facts);

                bool produced = apply_conclusions(rule, facts, step);
                step.success = produced || !rule.conclusions.empty();

                trace.add_step(step);
                rule.fired = true;

                if (produced) {
                    fired_this_round = true;
                    any_fired        = true;
                    ++firings;
                    if (firings >= cfg.max_rule_firings) break;
                }
            }

            if (!fired_this_round) break;
            ++depth;
        }

        return any_fired;
    }

    // Attempt backward chaining: can we derive `goal_name` from known facts?
    // Returns true if derivable. Does NOT modify facts permanently (uses copy).
    bool backward_chain(
            const std::string& goal_name,
            std::unordered_map<std::string, Fact> facts, // by value
            ReasoningTrace& trace,
            const ReasoningConfig& cfg,
            int depth = 0)
    {
        if (depth > cfg.max_depth) return false;

        // Goal already in facts?
        if (facts.count(goal_name)) return true;

        ensure_sorted();

        // Find rules whose conclusions produce goal_name
        for (auto& rule : rules_) {
            bool produces_goal = false;
            for (auto& conc : rule.conclusions) {
                if (conc.name == goal_name) { produces_goal = true; break; }
            }
            if (!produces_goal) continue;

            // Check if preconditions can be satisfied
            bool all_satisfied = true;
            for (auto& cond : rule.conditions) {
                if (!ConditionEvaluator::evaluate(cond, facts)) {
                    // Try to derive the required fact recursively
                    if (!backward_chain(cond.left, facts, trace, cfg, depth + 1)) {
                        all_satisfied = false;
                        break;
                    }
                }
            }

            if (all_satisfied && conditions_met(rule, facts)) {
                ReasoningStep step;
                step.type        = StepType::RULE_FIRE;
                step.rule_id     = rule.id;
                step.operation   = "BACKWARD_CHAIN";
                step.confidence  = rule.confidence;
                step.description = "Backward chain: rule '" + rule.id + "' produces '" + goal_name + "'";
                collect_condition_facts(rule.conditions, step.input_facts);
                apply_conclusions(rule, facts, step);
                trace.add_step(step);
                return true;
            }
        }
        return false;
    }

    const std::vector<Rule>& rules() const { return rules_; }

private:
    std::vector<Rule>                   rules_;
    std::unordered_map<std::string, size_t> rule_index_;
    bool                                sorted_ = true;

    void ensure_sorted() {
        if (!sorted_) {
            std::stable_sort(rules_.begin(), rules_.end(),
                [](const Rule& a, const Rule& b) {
                    if (a.priority != b.priority) return a.priority > b.priority;
                    return a.confidence > b.confidence;
                });
            // Rebuild index
            rule_index_.clear();
            for (size_t i = 0; i < rules_.size(); ++i)
                rule_index_[rules_[i].id] = i;
            sorted_ = true;
        }
    }

    bool conditions_met(const Rule& rule,
                        const std::unordered_map<std::string, Fact>& facts) const {
        if (rule.conditions.empty()) return true;
        // If single compound condition
        if (rule.conditions.size() == 1)
            return ConditionEvaluator::evaluate(rule.conditions[0], facts);
        // Multiple conditions = implicit AND
        for (auto& c : rule.conditions)
            if (!ConditionEvaluator::evaluate(c, facts)) return false;
        return true;
    }

    bool apply_conclusions(const Rule& rule,
                           std::unordered_map<std::string, Fact>& facts,
                           ReasoningStep& step) {
        bool produced = false;
        for (auto& conc : rule.conclusions) {
            Fact f;
            f.id         = "fact_" + rule.id + "_" + conc.name;
            f.name       = conc.name;
            f.source     = "rule:" + rule.id;
            f.confidence = conc.confidence;
            f.unit       = conc.unit;
            f.bound      = true;

            if (conc.is_numeric) {
                f.value = FactValue{conc.numeric_value};
            } else {
                // Try to interpret as bool or number
                if (conc.value_expr == "true")  f.value = FactValue{true};
                else if (conc.value_expr == "false") f.value = FactValue{false};
                else {
                    try {
                        size_t idx;
                        double d = std::stod(conc.value_expr, &idx);
                        if (idx == conc.value_expr.size()) f.value = FactValue{d};
                        else f.value = FactValue{conc.value_expr};
                    } catch (...) {
                        f.value = FactValue{conc.value_expr};
                    }
                }
            }

            bool is_new = (facts.count(f.name) == 0) ||
                          (fact_value_to_string(facts[f.name].value) !=
                           fact_value_to_string(f.value));

            facts[f.name] = f;
            step.output_facts.push_back(f.name + "=" + fact_value_to_string(f.value));
            if (is_new) produced = true;
        }
        return produced;
    }

    std::string facts_snapshot(const Rule& rule,
                               const std::unordered_map<std::string, Fact>& facts) const {
        std::string snap;
        for (auto& cond : rule.conditions) {
            auto it = facts.find(cond.left);
            if (it != facts.end())
                snap += cond.left + "=" + fact_value_to_string(it->second.value) + ";";
        }
        return snap;
    }

    void collect_condition_facts(const std::vector<Condition>& conds,
                                 std::vector<std::string>& out) const {
        for (auto& c : conds) {
            if (!c.left.empty()) out.push_back(c.left);
            collect_condition_facts(c.sub_conditions, out);
        }
    }
};

} // namespace prime::reasoning
