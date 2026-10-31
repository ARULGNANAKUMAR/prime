#pragma once
/*
 * atlas_parser.hpp — Lightweight deterministic text parser for ATLAS.
 *
 * Pipeline:
 *   raw text
 *     → InputNormalizer     (whitespace, units, sentence splitting)
 *     → ProblemClassifier   (domain detection)
 *     → NumericExtractor    (entities with quantities)
 *     → EntityExtractor     (concept entities)
 *     → RelationshipExtractor
 *     → ConstraintExtractor
 *     → GoalExtractor
 *     → OperationExtractor
 *     → ExpressionExtractor (formula representation)
 *     → LogicExtractor      (IF/AND/OR)
 *     → AssumptionExtractor
 *     → StateBuilder
 *     → GraphBuilder
 *
 * ATLAS does NOT solve problems. It only represents them.
 * No LLM required. No external NLP libraries.
 */

#include "atlas_types.hpp"
#include <regex>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <unordered_map>
#include <unordered_set>
#include <atomic>

namespace prime {
namespace atlas {

// ---------------------------------------------------------------------------
// ID generator
// ---------------------------------------------------------------------------
class IdGen {
public:
    explicit IdGen(const std::string& prefix) : prefix_(prefix) {}
    std::string next() {
        return prefix_ + "_" + std::to_string(++counter_);
    }
    void reset() { counter_ = 0; }
private:
    std::string  prefix_;
    std::uint64_t counter_ = 0;
};

// ---------------------------------------------------------------------------
// Unit normalization table
// ---------------------------------------------------------------------------
struct UnitInfo {
    std::string canonical;   // canonical unit name
    std::string dimension;   // "length", "mass", "time", "speed", "temperature",
                             // "area", "volume", "energy", "force", "pressure",
                             // "electric_current", "angle", "data", "currency", "percent"
};

inline const std::unordered_map<std::string, UnitInfo>& unit_table() {
    static const std::unordered_map<std::string, UnitInfo> table = {
        // length
        {"km",   {"km",   "length"}}, {"kilometer",  {"km",  "length"}}, {"kilometers", {"km",  "length"}},
        {"m",    {"m",    "length"}}, {"meter",      {"m",   "length"}}, {"meters",     {"m",   "length"}},
        {"cm",   {"cm",   "length"}}, {"centimeter", {"cm",  "length"}}, {"centimeters",{"cm",  "length"}},
        {"mm",   {"mm",   "length"}}, {"millimeter", {"mm",  "length"}},
        {"mi",   {"mi",   "length"}}, {"mile",       {"mi",  "length"}}, {"miles",      {"mi",  "length"}},
        {"ft",   {"ft",   "length"}}, {"foot",       {"ft",  "length"}}, {"feet",       {"ft",  "length"}},
        {"in",   {"in",   "length"}}, {"inch",       {"in",  "length"}}, {"inches",     {"in",  "length"}},
        // mass
        {"kg",   {"kg",   "mass"}},   {"kilogram",  {"kg",  "mass"}},   {"kilograms",  {"kg",  "mass"}},
        {"g",    {"g",    "mass"}},   {"gram",      {"g",   "mass"}},   {"grams",      {"g",   "mass"}},
        {"mg",   {"mg",   "mass"}},   {"lb",        {"lb",  "mass"}},   {"lbs",        {"lb",  "mass"}},
        {"pound",{"lb",   "mass"}},   {"pounds",    {"lb",  "mass"}},   {"ton",        {"ton", "mass"}},
        {"tonne",{"tonne","mass"}},   {"oz",        {"oz",  "mass"}},
        // time
        {"s",    {"s",    "time"}},   {"sec",       {"s",   "time"}},   {"second",     {"s",   "time"}},
        {"seconds",{"s",  "time"}},   {"min",       {"min", "time"}},   {"minute",     {"min", "time"}},
        {"minutes",{"min","time"}},   {"h",         {"h",   "time"}},   {"hr",         {"h",   "time"}},
        {"hour", {"h",    "time"}},   {"hours",     {"h",   "time"}},   {"day",        {"day", "time"}},
        {"days", {"day",  "time"}},   {"week",      {"wk",  "time"}},   {"weeks",      {"wk",  "time"}},
        {"month",{"mo",   "time"}},   {"year",      {"yr",  "time"}},   {"years",      {"yr",  "time"}},
        // speed (composite)
        {"km/h", {"km/h", "speed"}},  {"km/hr",     {"km/h","speed"}},  {"kph",        {"km/h","speed"}},
        {"m/s",  {"m/s",  "speed"}},  {"mph",       {"mph", "speed"}},
        // temperature
        {"c",    {"°C",   "temperature"}}, {"°c", {"°C","temperature"}},{"celsius",    {"°C", "temperature"}},
        {"f",    {"°F",   "temperature"}}, {"°f", {"°F","temperature"}},{"fahrenheit", {"°F", "temperature"}},
        {"k",    {"K",    "temperature"}}, {"kelvin",{"K","temperature"}},
        // area
        {"m2",   {"m²",   "area"}},   {"m²",        {"m²", "area"}},   {"km2",        {"km²","area"}},
        {"km²",  {"km²",  "area"}},   {"sqft",      {"ft²","area"}},
        // volume
        {"l",    {"l",    "volume"}}, {"liter",     {"l",  "volume"}}, {"liters",     {"l",  "volume"}},
        {"ml",   {"ml",   "volume"}}, {"milliliter",{"ml", "volume"}},
        {"m3",   {"m³",   "volume"}}, {"m³",        {"m³", "volume"}},
        // energy / force / pressure
        {"j",    {"J",    "energy"}}, {"joule",     {"J",  "energy"}}, {"kj",         {"kJ", "energy"}},
        {"n",    {"N",    "force"}},  {"newton",    {"N",  "force"}},  {"pa",         {"Pa", "pressure"}},
        {"pascal",{"Pa",  "pressure"}},
        // data
        {"b",    {"B",    "data"}},   {"kb",        {"KB", "data"}},   {"mb",         {"MB", "data"}},
        {"gb",   {"GB",   "data"}},   {"tb",        {"TB", "data"}},
        {"byte", {"B",    "data"}},   {"bytes",     {"B",  "data"}},   {"bit",        {"bit","data"}},
        {"bits", {"bit",  "data"}},
        // currency
        {"usd",  {"USD",  "currency"}},{"$",        {"USD","currency"}},
        {"eur",  {"EUR",  "currency"}},{"€",        {"EUR","currency"}},
        // percent
        {"%",    {"%",    "percent"}}, {"percent",  {"%",  "percent"}},{"percentage", {"%",  "percent"}},
    };
    return table;
}

inline std::optional<UnitInfo> lookup_unit(const std::string& raw) {
    std::string low = raw;
    std::transform(low.begin(), low.end(), low.begin(), [](unsigned char c){ return std::tolower(c); });
    auto& tbl = unit_table();
    auto it = tbl.find(low);
    if (it != tbl.end()) return it->second;
    return std::nullopt;
}

// ---------------------------------------------------------------------------
// InputNormalizer
// ---------------------------------------------------------------------------
class InputNormalizer {
public:
    struct Result {
        std::string              normalized;
        std::string              original;
        std::vector<std::string> sentences;
    };

    Result normalize(const std::string& raw) const {
        if (raw.empty()) {
            return {"", "", {}};
        }

        Result r;
        r.original   = raw;
        r.normalized = collapse_whitespace(raw);
        r.sentences  = split_sentences(r.normalized);
        return r;
    }

private:
    static std::string collapse_whitespace(const std::string& s) {
        std::string out;
        out.reserve(s.size());
        bool in_ws = false;
        for (unsigned char c : s) {
            if (std::isspace(c)) {
                if (!in_ws) { out += ' '; in_ws = true; }
            } else {
                out += (char)c;
                in_ws = false;
            }
        }
        // trim
        std::size_t start = 0;
        while (start < out.size() && out[start] == ' ') ++start;
        std::size_t end = out.size();
        while (end > start && out[end-1] == ' ') --end;
        return out.substr(start, end - start);
    }

    static std::vector<std::string> split_sentences(const std::string& s) {
        std::vector<std::string> sents;
        std::string cur;
        for (std::size_t i = 0; i < s.size(); ++i) {
            cur += s[i];
            if ((s[i] == '.' || s[i] == '!' || s[i] == '?') &&
                (i + 1 >= s.size() || std::isspace((unsigned char)s[i+1]))) {
                std::string t = cur;
                // trim
                while (!t.empty() && t.front() == ' ') t.erase(t.begin());
                while (!t.empty() && t.back()  == ' ') t.pop_back();
                if (!t.empty()) sents.push_back(t);
                cur.clear();
            }
        }
        if (!cur.empty()) {
            while (!cur.empty() && cur.front() == ' ') cur.erase(cur.begin());
            while (!cur.empty() && cur.back()  == ' ') cur.pop_back();
            if (!cur.empty()) sents.push_back(cur);
        }
        if (sents.empty() && !s.empty()) sents.push_back(s);
        return sents;
    }
};

// ---------------------------------------------------------------------------
// ProblemClassifier
// ---------------------------------------------------------------------------
class ProblemClassifier {
public:
    struct ClassificationResult {
        Domain domain;
        double confidence;
    };

    ClassificationResult classify(const std::string& normalized_text) const {
        std::string low = to_lower(normalized_text);

        struct Rule {
            Domain domain;
            std::vector<std::string> keywords;
            double base_weight;
        };

        static const std::vector<Rule> rules = {
            { Domain::MATHEMATICS, {
                "equation", "solve", "sum", "product", "integral", "derivative",
                "matrix", "vector", "polynomial", "factor", "prime", "divisible",
                "angle", "triangle", "circle", "radius", "perimeter",
                "probability", "mean", "median", "mode", "fraction",
                "algebra", "geometry", "calculus", "statistics", "arithmetic",
                "quadratic", "exponent", "logarithm", "modulo"
            }, 1.0 },
            { Domain::PHYSICS, {
                "force", "acceleration", "velocity", "mass", "energy", "momentum",
                "gravity", "friction", "torque", "pressure", "temperature", "heat",
                "wave", "frequency", "amplitude", "newton", "joule", "watt",
                "electric", "magnetic", "field", "charge", "current", "voltage",
                "photon", "quantum", "relativity", "thermodynamic", "entropy"
            }, 1.0 },
            { Domain::PROGRAMMING, {
                "algorithm", "sort", "array", "list", "function", "loop", "code",
                "variable", "class", "object", "database", "query", "api",
                "python", "java", "c++", "javascript", "binary", "tree", "graph",
                "stack", "queue", "hash", "recursion", "complexity", "bug", "debug",
                "compile", "runtime", "pointer", "memory", "thread"
            }, 1.0 },
            { Domain::BUSINESS, {
                "revenue", "profit", "cost", "budget", "sales", "market", "customer",
                "price", "investment", "return", "roi", "loss", "expense", "income",
                "tax", "interest", "rate", "loan", "mortgage", "stock", "portfolio",
                "forecast", "growth", "demand", "supply", "margin"
            }, 1.0 },
            { Domain::ENGINEERING, {
                "circuit", "voltage", "resistance", "load", "beam", "stress", "strain",
                "torque", "gear", "pump", "valve", "flow", "pressure", "thermal",
                "material", "alloy", "steel", "concrete", "signal", "bandwidth",
                "efficiency", "machine", "motor", "turbine", "pipeline", "capacity"
            }, 1.0 },
            { Domain::BIOLOGY, {
                "cell", "dna", "rna", "protein", "enzyme", "gene", "organism",
                "species", "evolution", "photosynthesis", "ecosystem", "population",
                "membrane", "nucleus", "mitosis", "meiosis", "virus", "bacteria",
                "metabolism", "neural", "tissue", "organ", "chromosome"
            }, 1.0 },
        };

        std::unordered_map<Domain, double> scores;
        for (auto& rule : rules) {
            double s = 0.0;
            for (auto& kw : rule.keywords)
                if (low.find(kw) != std::string::npos) s += rule.base_weight;
            if (s > 0.0) scores[rule.domain] = s;
        }

        // Motion terms that strongly indicate physics regardless of math keywords
        {
            bool has_motion = (low.find("travel") != std::string::npos ||
                               low.find("speed")  != std::string::npos ||
                               low.find("velocity")!= std::string::npos ||
                               low.find("distance")!= std::string::npos ||
                               low.find("km/h")   != std::string::npos ||
                               low.find("m/s")    != std::string::npos ||
                               low.find("mph")    != std::string::npos);
            bool has_unit = (low.find(" km") != std::string::npos ||
                             low.find(" m ")  != std::string::npos ||
                             low.find(" kg")  != std::string::npos ||
                             low.find(" hour")!= std::string::npos ||
                             low.find(" min") != std::string::npos);
            if (has_motion || has_unit) scores[Domain::PHYSICS] += 1.2;
        }

        Domain best   = Domain::UNKNOWN;
        double best_s = 0.0;
        for (auto& [d, s] : scores)
            if (s > best_s) { best = d; best_s = s; }

        // Also check for "general" motion/unit patterns → physics default
        if (best == Domain::UNKNOWN) {
            bool has_number_and_unit = false;
            std::regex num_unit(R"(\d+\s*(?:km|m|kg|s|hour|min|mph|m/s|°C|°F))");
            if (std::regex_search(low, num_unit)) has_number_and_unit = true;
            if (has_number_and_unit) {
                best   = Domain::PHYSICS;
                best_s = 0.3;
            }
        }

        // Normalize confidence (cap at 0.99)
        double total_kw_count = 0.0;
        for (auto& [d, s] : scores) total_kw_count += s;
        double conf = total_kw_count > 0 ? std::min(0.99, best_s / (total_kw_count + best_s)) : 0.1;
        if (best == Domain::UNKNOWN) conf = 0.1;

        return { best, conf };
    }

private:
    static std::string to_lower(const std::string& s) {
        std::string out = s;
        std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c){ return std::tolower(c); });
        return out;
    }
};

// ---------------------------------------------------------------------------
// NumericExtractor — finds "VALUE UNIT" patterns in text
// ---------------------------------------------------------------------------
struct NumericMatch {
    std::string full_match;   // raw matched text
    double      value;
    std::string unit_raw;
    std::string unit_canonical;
    std::string dimension;
    std::size_t pos;
    std::size_t len;
};

class NumericExtractor {
public:
    std::vector<NumericMatch> extract(const std::string& text) const {
        std::vector<NumericMatch> results;

        // Pattern: optional minus, integer or decimal, optional %, optional space, optional unit
        // Groups: 1=sign+number, 2=unit
        static const std::regex pat(
            R"((-?\d+(?:\.\d+)?)\s*(%|km/h|km/hr|m/s|mph|kph|km²|m²|m³|km|m|cm|mm|mi|ft|in|kg|g|mg|lb|lbs|oz|ton|tonne|s|sec|min|h|hr|day|days|week|weeks|month|year|years|l|ml|j|kj|n|pa|kb|mb|gb|tb|°[CFK]|K)?)",
            std::regex::icase
        );

        auto begin = std::sregex_iterator(text.begin(), text.end(), pat);
        auto end   = std::sregex_iterator();

        for (auto it = begin; it != end; ++it) {
            auto& m = *it;
            std::string num_str = m[1].str();
            if (num_str.empty()) continue;

            double val = 0.0;
            try { val = std::stod(num_str); }
            catch (...) { continue; }

            std::string unit_raw = m[2].str();
            std::string unit_can, dim;
            if (!unit_raw.empty()) {
                if (auto info = lookup_unit(unit_raw)) {
                    unit_can = info->canonical;
                    dim      = info->dimension;
                } else {
                    unit_can = unit_raw;
                    dim      = "";
                }
            }

            NumericMatch nm;
            nm.full_match      = m[0].str();
            nm.value           = val;
            nm.unit_raw        = unit_raw;
            nm.unit_canonical  = unit_can;
            nm.dimension       = dim;
            nm.pos             = static_cast<std::size_t>(m.position());
            nm.len             = static_cast<std::size_t>(m.length());

            results.push_back(nm);
        }
        return results;
    }
};

// ---------------------------------------------------------------------------
// KeywordEntityExtractor — extracts concept entities by domain keyword lists
// ---------------------------------------------------------------------------
class KeywordEntityExtractor {
public:
    struct ConceptMatch {
        std::string name;
        std::string type; // "concept", "variable", "object"
    };

    std::vector<ConceptMatch> extract(const std::string& text, Domain domain) const {
        std::string low = to_lower(text);
        std::vector<ConceptMatch> results;

        // Common physics/math variables
        static const std::vector<std::pair<std::string,std::string>> common = {
            {"distance","variable"}, {"speed","variable"}, {"velocity","variable"},
            {"time","variable"}, {"acceleration","variable"}, {"force","variable"},
            {"mass","variable"}, {"energy","variable"}, {"temperature","variable"},
            {"pressure","variable"}, {"height","variable"}, {"weight","variable"},
            {"rate","variable"}, {"cost","variable"}, {"revenue","variable"},
            {"profit","variable"}, {"price","variable"}, {"budget","variable"},
            {"array","concept"}, {"list","concept"}, {"string","concept"},
            {"function","concept"}, {"algorithm","concept"},
            {"car","object"}, {"train","object"}, {"plane","object"},
            {"truck","object"}, {"ball","object"}, {"box","object"},
            {"machine","object"}, {"motor","object"}, {"circuit","object"},
            {"cell","concept"}, {"organism","concept"}, {"population","concept"},
        };

        std::unordered_set<std::string> seen;
        for (auto& [name, type] : common) {
            if (low.find(name) != std::string::npos && !seen.count(name)) {
                results.push_back({name, type});
                seen.insert(name);
            }
        }

        (void)domain; // future: domain-specific extraction

        return results;
    }

private:
    static std::string to_lower(const std::string& s) {
        std::string out = s;
        std::transform(out.begin(), out.end(), out.begin(),
                       [](unsigned char c){ return std::tolower(c); });
        return out;
    }
    std::unordered_set<std::string> dummy_; // avoid unused warning
};

// ---------------------------------------------------------------------------
// GoalExtractor — identifies what the problem is asking for
// ---------------------------------------------------------------------------
class GoalExtractor {
public:
    struct GoalMatch {
        std::string description;
        std::string target;     // entity name if identifiable
    };

    std::vector<GoalMatch> extract(const std::string& text) const {
        std::vector<GoalMatch> results;
        std::string low = to_lower(text);

        // Patterns indicating a question/goal
        static const std::vector<std::pair<std::string,std::string>> patterns = {
            { "what is",      "calculate" },
            { "what are",     "enumerate" },
            { "find",         "find" },
            { "calculate",    "calculate" },
            { "determine",    "determine" },
            { "compute",      "compute" },
            { "how many",     "count" },
            { "how much",     "calculate" },
            { "solve",        "solve" },
            { "predict",      "predict" },
            { "classify",     "classify" },
            { "sort",         "sort" },
            { "optimize",     "optimize" },
            { "maximize",     "maximize" },
            { "minimize",     "minimize" },
        };

        std::vector<GoalMatch> found;
        for (auto& [kw, action] : patterns) {
            std::size_t p = low.find(kw);
            if (p != std::string::npos) {
                // Extract the remainder of the sentence after the keyword
                std::string fragment = text.substr(p);
                if (fragment.size() > 80) fragment = fragment.substr(0, 80) + "…";
                // Avoid duplicates
                bool dup = false;
                for (auto& g : found)
                    if (g.description.find(kw) != std::string::npos) { dup = true; break; }
                if (!dup)
                    found.push_back({ fragment, action });
            }
        }

        // If nothing found but there's a '?' — it's probably a question
        if (found.empty() && low.find('?') != std::string::npos) {
            found.push_back({ text.substr(0, std::min(text.size(), std::size_t(120))), "answer" });
        }

        return found;
    }

private:
    static std::string to_lower(const std::string& s) {
        std::string out = s;
        std::transform(out.begin(), out.end(), out.begin(),
                       [](unsigned char c){ return std::tolower(c); });
        return out;
    }
};

// ---------------------------------------------------------------------------
// ConstraintExtractor — finds explicit constraints in text
// ---------------------------------------------------------------------------
class ConstraintExtractor {
public:
    struct ConstraintMatch {
        std::string left;
        ConstraintOp op;
        std::string right;
        std::string type;
    };

    std::vector<ConstraintMatch> extract(const std::string& text) const {
        std::vector<ConstraintMatch> results;

        // Simple numeric constraint patterns: "X < Y", "X >= Y", "X = Y"
        static const std::regex con_pat(
            R"((\w[\w\s]*?)\s*(>=|<=|!=|>|<|=)\s*(-?\d+(?:\.\d+)?(?:\s*\w+)?))"
        );
        auto begin = std::sregex_iterator(text.begin(), text.end(), con_pat);
        auto end   = std::sregex_iterator();
        for (auto it = begin; it != end; ++it) {
            auto& m = *it;
            std::string left  = trim(m[1].str());
            std::string op_s  = m[2].str();
            std::string right = trim(m[3].str());
            if (left.empty() || right.empty()) continue;
            results.push_back({ left, constraint_op_from_string(op_s), right, "mathematical" });
        }

        // Textual patterns: "must be", "at least", "at most", "between"
        std::string low = to_lower(text);

        if (low.find("at least") != std::string::npos)
            results.push_back({"constraint", ConstraintOp::GTE, "threshold", "domain"});
        if (low.find("at most") != std::string::npos)
            results.push_back({"constraint", ConstraintOp::LTE, "threshold", "domain"});
        if (low.find("must not exceed") != std::string::npos)
            results.push_back({"constraint", ConstraintOp::LTE, "limit", "domain"});
        if (low.find("maximum") != std::string::npos && results.empty())
            results.push_back({"constraint", ConstraintOp::LTE, "maximum", "domain"});
        if (low.find("minimum") != std::string::npos && results.empty())
            results.push_back({"constraint", ConstraintOp::GTE, "minimum", "domain"});

        return results;
    }

private:
    static std::string trim(const std::string& s) {
        std::size_t a = 0, b = s.size();
        while (a < b && std::isspace((unsigned char)s[a])) ++a;
        while (b > a && std::isspace((unsigned char)s[b-1])) --b;
        return s.substr(a, b - a);
    }
    static std::string to_lower(const std::string& s) {
        std::string out = s;
        std::transform(out.begin(), out.end(), out.begin(),
                       [](unsigned char c){ return std::tolower(c); });
        return out;
    }
};

// ---------------------------------------------------------------------------
// OperationExtractor
// ---------------------------------------------------------------------------
class OperationExtractor {
public:
    struct OpMatch { std::string type; };

    std::vector<OpMatch> extract(const std::string& text) const {
        std::string low = to_lower(text);
        std::vector<OpMatch> results;

        static const std::vector<std::string> ops = {
            "calculate", "compute", "find", "determine", "solve",
            "sort", "search", "compare", "classify", "predict",
            "transform", "optimize", "maximize", "minimize", "count",
            "sum", "average", "multiply", "divide"
        };

        std::unordered_set<std::string> seen;
        for (auto& op : ops)
            if (low.find(op) != std::string::npos && !seen.count(op)) {
                results.push_back({op});
                seen.insert(op);
            }
        return results;
    }

private:
    static std::string to_lower(const std::string& s) {
        std::string out = s;
        std::transform(out.begin(), out.end(), out.begin(),
                       [](unsigned char c){ return std::tolower(c); });
        return out;
    }
    std::unordered_set<std::string> dummy_;
};

// ---------------------------------------------------------------------------
// ExpressionExtractor — detects formula-like text
// ---------------------------------------------------------------------------
class ExpressionExtractor {
public:
    struct ExprMatch {
        std::string text;
        std::string op;        // "divide", "multiply", "add", "subtract", "assign"
        std::string lhs;
        std::string rhs_left;
        std::string rhs_right;
    };

    std::vector<ExprMatch> extract(const std::string& text) const {
        std::vector<ExprMatch> results;

        // Pattern: "X = Y / Z", "X = Y * Z", "X = Y + Z", "X = Y - Z"
        static const std::regex expr_pat(
            R"((\w+)\s*=\s*(\w+)\s*([+\-*/])\s*(\w+))"
        );
        auto begin = std::sregex_iterator(text.begin(), text.end(), expr_pat);
        auto end   = std::sregex_iterator();
        for (auto it = begin; it != end; ++it) {
            auto& m = *it;
            ExprMatch em;
            em.text      = m[0].str();
            em.lhs       = m[1].str();
            em.rhs_left  = m[2].str();
            em.rhs_right = m[4].str();
            char opch    = m[3].str()[0];
            switch (opch) {
                case '+': em.op = "add";      break;
                case '-': em.op = "subtract"; break;
                case '*': em.op = "multiply"; break;
                case '/': em.op = "divide";   break;
                default:  em.op = "custom";   break;
            }
            results.push_back(em);
        }

        // Also look for "speed = distance / time" phrasing in words
        std::string low = to_lower(text);
        struct WordExpr { std::string trigger; std::string lhs; std::string op; std::string a; std::string b; };
        static const std::vector<WordExpr> word_exprs = {
            { "average speed", "speed", "divide", "distance", "time" },
            { "profit",        "profit","subtract","revenue","cost" },
            { "acceleration",  "acceleration","divide","force","mass" },
        };
        for (auto& we : word_exprs) {
            if (low.find(we.trigger) != std::string::npos) {
                // Only add if not already found via regex
                bool found = false;
                for (auto& r : results) if (r.lhs == we.lhs) { found = true; break; }
                if (!found)
                    results.push_back({ we.trigger + " formula", we.op, we.lhs, we.a, we.b });
            }
        }

        return results;
    }

private:
    static std::string to_lower(const std::string& s) {
        std::string out = s;
        std::transform(out.begin(), out.end(), out.begin(),
                       [](unsigned char c){ return std::tolower(c); });
        return out;
    }
};

// ---------------------------------------------------------------------------
// LogicExtractor — detects IF/AND/OR/NOT clauses
// ---------------------------------------------------------------------------
class LogicExtractor {
public:
    struct LogicMatch {
        LogicOp     op;
        std::string condition;
        std::string implication;
    };

    std::vector<LogicMatch> extract(const std::string& text) const {
        std::vector<LogicMatch> results;
        std::string low = to_lower(text);

        // IF ... THEN ...
        std::regex if_then(R"(if\s+(.+?)[,.]?\s+then\s+(.+?)(?:[.!]|$))",
                           std::regex::icase | std::regex::multiline);
        auto begin = std::sregex_iterator(text.begin(), text.end(), if_then);
        auto end   = std::sregex_iterator();
        for (auto it = begin; it != end; ++it) {
            auto& m = *it;
            results.push_back({ LogicOp::IF, trim(m[1].str()), trim(m[2].str()) });
        }

        // Simple AND/OR detection
        if (low.find(" and ") != std::string::npos)
            results.push_back({ LogicOp::AND, "", "" });
        if (low.find(" or ") != std::string::npos)
            results.push_back({ LogicOp::OR,  "", "" });

        return results;
    }

private:
    static std::string to_lower(const std::string& s) {
        std::string out = s;
        std::transform(out.begin(), out.end(), out.begin(),
                       [](unsigned char c){ return std::tolower(c); });
        return out;
    }
    static std::string trim(const std::string& s) {
        std::size_t a = 0, b = s.size();
        while (a < b && std::isspace((unsigned char)s[a])) ++a;
        while (b > a && std::isspace((unsigned char)s[b-1])) --b;
        return s.substr(a, b - a);
    }
};

// ---------------------------------------------------------------------------
// AssumptionExtractor
// ---------------------------------------------------------------------------
class AssumptionExtractor {
public:
    struct AssumptionMatch { std::string description; };

    std::vector<AssumptionMatch> extract(const std::string& text) const {
        std::vector<AssumptionMatch> results;
        std::string low = to_lower(text);

        static const std::vector<std::string> triggers = {
            "assume", "assuming", "neglect", "ignore", "ideal",
            "approximately", "roughly", "approximately", "without loss",
            "no friction", "no air resistance", "frictionless", "constant"
        };
        for (auto& t : triggers)
            if (low.find(t) != std::string::npos)
                results.push_back({ t });

        return results;
    }

private:
    static std::string to_lower(const std::string& s) {
        std::string out = s;
        std::transform(out.begin(), out.end(), out.begin(),
                       [](unsigned char c){ return std::tolower(c); });
        return out;
    }
};

// ---------------------------------------------------------------------------
// ATLASParser — main orchestrator
// ---------------------------------------------------------------------------
class ATLASParser {
public:
    explicit ATLASParser(const AtlasLimits& limits = AtlasLimits{})
        : limits_(limits) {}

    // Parse a plain text string into a UniversalProblem
    UniversalProblem parse(const std::string& text) const {
        if (text.size() > limits_.max_input_bytes)
            throw AtlasError("Input exceeds maximum size");

        UniversalProblem p;
        p.problem_id  = generate_problem_id();

        // Normalize
        InputNormalizer norm;
        auto norm_result = norm.normalize(text);
        p.original_input   = norm_result.original;
        p.normalized_input = norm_result.normalized;

        // Classify
        ProblemClassifier cls;
        auto cls_result = cls.classify(p.normalized_input);
        p.domain            = cls_result.domain;
        p.domain_confidence = cls_result.confidence;

        // Build representation
        build_from_text(p, norm_result);

        return p;
    }

    // Accept an already-structured (partially filled) UniversalProblem
    UniversalProblem parse_structured(UniversalProblem proto) const {
        if (proto.problem_id.empty())
            proto.problem_id = generate_problem_id();
        if (proto.atlas_schema_version.empty())
            proto.atlas_schema_version = ATLAS_SCHEMA_VERSION;
        // Build graph from existing entities/relationships
        build_graph(proto);
        // Build initial state if none exists
        if (proto.states.empty()) build_initial_state(proto);
        return proto;
    }

private:
    AtlasLimits limits_;
    mutable std::atomic<std::uint64_t> id_counter_{0};

    std::string generate_problem_id() const {
        return "prob_" + std::to_string(++id_counter_);
    }
    std::string gen_id(const std::string& prefix) const {
        return prefix + "_" + std::to_string(++id_counter_);
    }

    void build_from_text(UniversalProblem& p,
                         const InputNormalizer::Result& norm) const {
        const std::string& text = norm.normalized;

        // 1. Extract numeric entities
        NumericExtractor num_ext;
        auto nums = num_ext.extract(text);
        for (auto& nm : nums) {
            if (p.entities.size() >= limits_.max_entities) break;
            Entity e;
            e.id   = gen_id("e");
            e.type = "quantity";
            e.name = nm.full_match;
            e.quantity = Quantity{ nm.value, nm.unit_canonical, nm.dimension };
            e.confidence = 0.9;
            e.metadata["raw"] = nm.full_match;
            p.entities.push_back(std::move(e));
        }

        // 2. Extract concept entities
        KeywordEntityExtractor kw_ext;
        auto concepts = kw_ext.extract(text, p.domain);
        for (auto& cm : concepts) {
            if (p.entities.size() >= limits_.max_entities) break;
            // Skip if already have a quantity entity with same name
            bool dup = false;
            for (auto& e : p.entities)
                if (to_lower(e.name) == cm.name) { dup = true; break; }
            if (!dup) {
                Entity e;
                e.id   = gen_id("e");
                e.type = cm.type;
                e.name = cm.name;
                e.confidence = 0.7;
                p.entities.push_back(std::move(e));
            }
        }

        // 3. Extract expressions (formulas)
        ExpressionExtractor expr_ext;
        auto exprs = expr_ext.extract(text);
        for (auto& em : exprs) {
            if (p.expressions.size() >= limits_.max_entities) break;
            Expression ex;
            ex.id   = gen_id("ex");
            ex.op   = em.op;
            ex.text = em.text;
            // Map operand names to entity IDs
            auto find_eid = [&](const std::string& name) -> std::string {
                std::string low_name = to_lower(name);
                for (auto& e : p.entities)
                    if (to_lower(e.name) == low_name) return e.id;
                return name; // use name as fallback
            };
            if (!em.lhs.empty())       ex.output_id = find_eid(em.lhs);
            if (!em.rhs_left.empty())  ex.operand_ids.push_back(find_eid(em.rhs_left));
            if (!em.rhs_right.empty()) ex.operand_ids.push_back(find_eid(em.rhs_right));
            p.expressions.push_back(std::move(ex));
        }

        // 4. Build relationships between quantity entities
        // "X travels Y" → entity(X) -travels→ entity(distance_entity)
        build_inferred_relationships(p, text);

        // 5. Extract constraints
        ConstraintExtractor con_ext;
        auto cons = con_ext.extract(text);
        for (auto& cm : cons) {
            if (p.constraints.size() >= limits_.max_constraints) break;
            Constraint c;
            c.id   = gen_id("c");
            c.left = cm.left;
            c.op   = cm.op;
            c.right= cm.right;
            c.type = cm.type;
            c.confidence = 0.75;
            p.constraints.push_back(std::move(c));
        }

        // 6. Extract goals
        GoalExtractor goal_ext;
        auto goals = goal_ext.extract(text);
        for (auto& gm : goals) {
            if (p.goals.size() >= limits_.max_goals) break;
            Goal g;
            g.id          = gen_id("g");
            g.description = gm.description;
            g.target      = gm.target;
            g.confidence  = 0.8;
            p.goals.push_back(std::move(g));
        }

        // 7. Extract operations
        OperationExtractor op_ext;
        auto ops = op_ext.extract(text);
        for (auto& om : ops) {
            if (p.operations.size() >= limits_.max_operations) break;
            Operation op;
            op.id   = gen_id("op");
            op.type = om.type;
            p.operations.push_back(std::move(op));
        }

        // 8. Extract logic clauses
        LogicExtractor logic_ext;
        auto logic_matches = logic_ext.extract(text);
        for (auto& lm : logic_matches) {
            LogicClause lc;
            lc.id          = gen_id("lc");
            lc.op          = lm.op;
            lc.condition   = lm.condition;
            lc.implication = lm.implication;
            p.logic_clauses.push_back(std::move(lc));
        }

        // 9. Extract assumptions
        AssumptionExtractor ass_ext;
        auto assumptions = ass_ext.extract(text);
        for (auto& am : assumptions) {
            Assumption a;
            a.id          = gen_id("a");
            a.description = am.description;
            a.confidence  = 0.7;
            p.assumptions.push_back(std::move(a));
        }

        // 10. Build problem graph
        build_graph(p);

        // 11. Build initial state
        build_initial_state(p);
    }

    void build_inferred_relationships(UniversalProblem& p,
                                      const std::string& text) const {
        std::string low = to_lower(text);

        // "X travels Y distance" → (object) -travels→ (quantity)
        // Heuristic: pair object entities with their closest quantity entities
        std::vector<Entity*> objects, quantities;
        for (auto& e : p.entities) {
            if (e.type == "object" || e.type == "concept") objects.push_back(&e);
            else if (e.type == "quantity") quantities.push_back(&e);
        }

        // "travels" → motion relation
        if (low.find("travel") != std::string::npos && !objects.empty() && !quantities.empty()) {
            Relationship r;
            r.id        = gen_id("r");
            r.source_id = objects[0]->id;
            r.relation  = "travels";
            r.target_id = quantities[0]->id;
            r.confidence= 0.7;
            if (p.relationships.size() < limits_.max_relationships)
                p.relationships.push_back(std::move(r));
        }

        // Quantity has_unit relationship
        for (auto& e : p.entities) {
            if (e.type == "quantity" && e.quantity && !e.quantity->unit.empty()) {
                Relationship r;
                r.id        = gen_id("r");
                r.source_id = e.id;
                r.relation  = "has_unit";
                r.target_id = e.quantity->unit;
                r.confidence= 1.0;
                if (p.relationships.size() < limits_.max_relationships)
                    p.relationships.push_back(std::move(r));
            }
        }

        // Expression operand relationships
        for (auto& ex : p.expressions) {
            for (auto& oid : ex.operand_ids) {
                Relationship r;
                r.id        = gen_id("r");
                r.source_id = ex.id;
                r.relation  = "uses";
                r.target_id = oid;
                r.confidence= 0.9;
                if (p.relationships.size() < limits_.max_relationships)
                    p.relationships.push_back(std::move(r));
            }
        }
    }

    void build_graph(UniversalProblem& p) const {
        std::size_t node_count = 0;
        auto add_node = [&](const std::string& id, const std::string& kind) {
            if (node_count >= limits_.max_graph_nodes) return;
            if (!p.graph.has_node(id)) {
                p.graph.add_node(id, kind, id);
                ++node_count;
            }
        };

        for (auto& e  : p.entities)      add_node(e.id,   "entity");
        for (auto& c  : p.constraints)   add_node(c.id,   "constraint");
        for (auto& g  : p.goals)         add_node(g.id,   "goal");
        for (auto& op : p.operations)    add_node(op.id,  "operation");
        for (auto& ex : p.expressions)   add_node(ex.id,  "expression");

        for (auto& r : p.relationships) {
            // Add endpoint nodes if they don't already exist (unit nodes, etc.)
            if (!p.graph.has_node(r.source_id)) {
                p.graph.add_node(r.source_id, "entity", r.source_id);
            }
            if (!p.graph.has_node(r.target_id)) {
                p.graph.add_node(r.target_id, "entity", r.target_id);
            }
            p.graph.add_edge(r.source_id, r.target_id, r.relation);
        }

        // Constraint → entity edges
        for (auto& c : p.constraints) {
            for (auto& e : p.entities)
                if (e.name == c.left || e.id == c.left)
                    p.graph.add_edge(c.id, e.id, "constrains");
        }

        // Goal → operation/expression edges
        for (auto& g : p.goals) {
            for (auto& op : p.operations)
                p.graph.add_edge(g.id, op.id, "requires");
        }
    }

    void build_initial_state(UniversalProblem& p) const {
        State s;
        s.id    = gen_id("s");
        s.label = "initial";
        for (auto& e : p.entities)    s.entity_ids.push_back(e.id);
        for (auto& c : p.constraints) s.constraint_ids.push_back(c.id);
        p.states.push_back(std::move(s));
    }

    static std::string to_lower(const std::string& str) {
        std::string out = str;
        std::transform(out.begin(), out.end(), out.begin(),
                       [](unsigned char c){ return std::tolower(c); });
        return out;
    }
};

} // namespace atlas
} // namespace prime
