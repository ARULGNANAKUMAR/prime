#pragma once
/*
 * atlas_engine.hpp — Top-level ATLAS engine API.
 *
 * Combines parser, validator, and serializer into one clean surface.
 * This is what the Python binding wraps.
 */

#include "atlas_types.hpp"
#include "atlas_parser.hpp"
#include "atlas_validator.hpp"
#include "atlas_json.hpp"

namespace prime {
namespace atlas {

class ATLASEngine {
public:
    explicit ATLASEngine(const AtlasLimits& limits = AtlasLimits{})
        : limits_(limits), parser_(limits), validator_() {}

    // ------------------------------------------------------------------
    // Parse text input → UniversalProblem
    // ------------------------------------------------------------------
    UniversalProblem parse(const std::string& text) const {
        if (text.empty())
            throw AtlasError("Input text is empty");
        return parser_.parse(text);
    }

    // ------------------------------------------------------------------
    // Accept a pre-structured UniversalProblem (e.g. from Python dict)
    // ------------------------------------------------------------------
    UniversalProblem parse_structured(UniversalProblem proto) const {
        return parser_.parse_structured(std::move(proto));
    }

    // ------------------------------------------------------------------
    // Validate
    // ------------------------------------------------------------------
    ValidationResult validate(const UniversalProblem& p) const {
        return validator_.validate(p);
    }

    // ------------------------------------------------------------------
    // Serialization
    // ------------------------------------------------------------------
    std::string to_json(const UniversalProblem& p) const {
        return atlas_to_json(p);
    }

    UniversalProblem from_json(const std::string& json_text) const {
        return atlas_from_json(json_text);
    }

    // ------------------------------------------------------------------
    // Schema version
    // ------------------------------------------------------------------
    static const char* schema_version() { return ATLAS_SCHEMA_VERSION; }

    // ------------------------------------------------------------------
    // Limits
    // ------------------------------------------------------------------
    const AtlasLimits& limits() const { return limits_; }

private:
    AtlasLimits     limits_;
    ATLASParser     parser_;
    AtlasValidator  validator_;
};

} // namespace atlas
} // namespace prime
