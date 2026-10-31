#pragma once
// expression_evaluator.hpp — Safe arithmetic expression parser and evaluator.
//
// Supports: +, -, *, /, %, ^, unary -, min, max, abs.
// Grammar (recursive descent):
//   expr   = term (('+' | '-') term)*
//   term   = power (('*' | '/' | '%') power)*
//   power  = unary ('^' unary)*   (right-assoc)
//   unary  = '-' unary | atom
//   atom   = NUMBER | IDENT | IDENT '(' arglist ')' | '(' expr ')'
//   arglist = expr (',' expr)*
//
// No eval(), no external libraries.

#include "reasoning_types.hpp"
#include <stdexcept>
#include <cmath>
#include <cctype>
#include <sstream>
#include <algorithm>

namespace prime::reasoning {

// ---------------------------------------------------------------------------
// Parse error
// ---------------------------------------------------------------------------
struct ExprParseError : std::runtime_error {
    explicit ExprParseError(const std::string& msg)
        : std::runtime_error("ExprParseError: " + msg) {}
};

struct ExprEvalError : std::runtime_error {
    explicit ExprEvalError(const std::string& msg)
        : std::runtime_error("ExprEvalError: " + msg) {}
};

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------
enum class TokKind { NUMBER, IDENT, PLUS, MINUS, STAR, SLASH, PERCENT,
                     CARET, LPAREN, RPAREN, COMMA, END };

struct Token {
    TokKind     kind;
    double      number = 0.0;
    std::string ident;
};

class Lexer {
public:
    explicit Lexer(const std::string& src) : src_(src), pos_(0) {}

    Token next() {
        skip_ws();
        if (pos_ >= src_.size()) return {TokKind::END};
        char c = src_[pos_];

        if (std::isdigit(c) || (c == '.' && pos_+1 < src_.size() && std::isdigit(src_[pos_+1]))) {
            return lex_number();
        }
        if (std::isalpha(c) || c == '_') return lex_ident();

        ++pos_;
        switch (c) {
            case '+': return {TokKind::PLUS};
            case '-': return {TokKind::MINUS};
            case '*': return {TokKind::STAR};
            case '/': return {TokKind::SLASH};
            case '%': return {TokKind::PERCENT};
            case '^': return {TokKind::CARET};
            case '(': return {TokKind::LPAREN};
            case ')': return {TokKind::RPAREN};
            case ',': return {TokKind::COMMA};
            default:
                throw ExprParseError(std::string("Unexpected character: '") + c + "'");
        }
    }

    Token peek() {
        auto saved = pos_;
        auto tok   = next();
        pos_ = saved;
        return tok;
    }

private:
    std::string src_;
    size_t      pos_;

    void skip_ws() {
        while (pos_ < src_.size() && std::isspace(static_cast<unsigned char>(src_[pos_])))
            ++pos_;
    }

    Token lex_number() {
        size_t start = pos_;
        while (pos_ < src_.size() && (std::isdigit(src_[pos_]) || src_[pos_] == '.'))
            ++pos_;
        // Scientific notation
        if (pos_ < src_.size() && (src_[pos_] == 'e' || src_[pos_] == 'E')) {
            ++pos_;
            if (pos_ < src_.size() && (src_[pos_] == '+' || src_[pos_] == '-'))
                ++pos_;
            while (pos_ < src_.size() && std::isdigit(src_[pos_])) ++pos_;
        }
        Token t;
        t.kind   = TokKind::NUMBER;
        t.number = std::stod(src_.substr(start, pos_ - start));
        return t;
    }

    Token lex_ident() {
        size_t start = pos_;
        while (pos_ < src_.size() && (std::isalnum(src_[pos_]) || src_[pos_] == '_'))
            ++pos_;
        Token t;
        t.kind  = TokKind::IDENT;
        t.ident = src_.substr(start, pos_ - start);
        return t;
    }
};

// ---------------------------------------------------------------------------
// Parser — builds an ExprNode tree
// ---------------------------------------------------------------------------
class ExprParser {
public:
    explicit ExprParser(const std::string& src) : lex_(src) {
        advance();
    }

    ExprNodePtr parse() {
        auto node = parse_expr();
        if (cur_.kind != TokKind::END)
            throw ExprParseError("Unexpected token after expression");
        return node;
    }

private:
    Lexer  lex_;
    Token  cur_;

    void advance() { cur_ = lex_.next(); }

    bool match(TokKind k) {
        if (cur_.kind == k) { advance(); return true; }
        return false;
    }

    // expr = term (('+' | '-') term)*
    ExprNodePtr parse_expr() {
        auto left = parse_term();
        while (cur_.kind == TokKind::PLUS || cur_.kind == TokKind::MINUS) {
            BinOp op = (cur_.kind == TokKind::PLUS) ? BinOp::ADD : BinOp::SUB;
            advance();
            auto right = parse_term();
            auto node  = std::make_shared<ExprNode>();
            node->type  = ExprNodeType::BINOP;
            node->binop = op;
            node->left  = left;
            node->right = right;
            left = node;
        }
        return left;
    }

    // term = power (('*' | '/' | '%') power)*
    ExprNodePtr parse_term() {
        auto left = parse_power();
        while (cur_.kind == TokKind::STAR || cur_.kind == TokKind::SLASH
               || cur_.kind == TokKind::PERCENT) {
            BinOp op;
            if      (cur_.kind == TokKind::STAR)    op = BinOp::MUL;
            else if (cur_.kind == TokKind::SLASH)   op = BinOp::DIV;
            else                                     op = BinOp::MOD;
            advance();
            auto right = parse_power();
            auto node  = std::make_shared<ExprNode>();
            node->type  = ExprNodeType::BINOP;
            node->binop = op;
            node->left  = left;
            node->right = right;
            left = node;
        }
        return left;
    }

    // power = unary ('^' unary)*   (right-assoc handled via recursion)
    ExprNodePtr parse_power() {
        auto base = parse_unary();
        if (cur_.kind == TokKind::CARET) {
            advance();
            auto exp  = parse_power(); // right-assoc
            auto node = std::make_shared<ExprNode>();
            node->type  = ExprNodeType::BINOP;
            node->binop = BinOp::POW;
            node->left  = base;
            node->right = exp;
            return node;
        }
        return base;
    }

    // unary = '-' unary | atom
    ExprNodePtr parse_unary() {
        if (cur_.kind == TokKind::MINUS) {
            advance();
            auto child = parse_unary();
            auto node  = std::make_shared<ExprNode>();
            node->type  = ExprNodeType::UNOP;
            node->unop  = UnOp::NEG;
            node->child = child;
            return node;
        }
        return parse_atom();
    }

    // atom = NUMBER | IDENT | IDENT '(' arglist ')' | '(' expr ')'
    ExprNodePtr parse_atom() {
        if (cur_.kind == TokKind::NUMBER) {
            auto node = std::make_shared<ExprNode>();
            node->type   = ExprNodeType::NUMBER;
            node->number = cur_.number;
            advance();
            return node;
        }
        if (cur_.kind == TokKind::IDENT) {
            std::string name = cur_.ident;
            advance();
            if (cur_.kind == TokKind::LPAREN) {
                // Function call
                advance();
                auto node = std::make_shared<ExprNode>();
                node->type      = ExprNodeType::FUNCTION;
                node->func_name = name;
                if (cur_.kind != TokKind::RPAREN) {
                    node->func_args.push_back(parse_expr());
                    while (cur_.kind == TokKind::COMMA) {
                        advance();
                        node->func_args.push_back(parse_expr());
                    }
                }
                if (cur_.kind != TokKind::RPAREN)
                    throw ExprParseError("Expected ')' after function arguments");
                advance();
                return node;
            }
            // Variable reference
            auto node = std::make_shared<ExprNode>();
            node->type     = ExprNodeType::VARIABLE;
            node->var_name = name;
            return node;
        }
        if (cur_.kind == TokKind::LPAREN) {
            advance();
            auto inner = parse_expr();
            if (cur_.kind != TokKind::RPAREN)
                throw ExprParseError("Expected ')'");
            advance();
            return inner;
        }
        throw ExprParseError("Unexpected token in expression");
    }
};

// ---------------------------------------------------------------------------
// Evaluator — walks the AST with variable bindings
// ---------------------------------------------------------------------------
class ExprEvaluator {
public:
    using Bindings = std::unordered_map<std::string, double>;

    static double evaluate(ExprNodePtr node, const Bindings& bindings) {
        if (!node) throw ExprEvalError("Null expression node");
        switch (node->type) {
            case ExprNodeType::NUMBER:
                return node->number;
            case ExprNodeType::VARIABLE: {
                auto it = bindings.find(node->var_name);
                if (it == bindings.end())
                    throw ExprEvalError("Unbound variable: " + node->var_name);
                return it->second;
            }
            case ExprNodeType::BINOP: {
                double l = evaluate(node->left,  bindings);
                double r = evaluate(node->right, bindings);
                switch (node->binop) {
                    case BinOp::ADD: return l + r;
                    case BinOp::SUB: return l - r;
                    case BinOp::MUL: return l * r;
                    case BinOp::DIV:
                        if (r == 0.0) throw ExprEvalError("Division by zero");
                        return l / r;
                    case BinOp::MOD:
                        if (r == 0.0) throw ExprEvalError("Modulo by zero");
                        return std::fmod(l, r);
                    case BinOp::POW: return std::pow(l, r);
                    default: throw ExprEvalError("Unknown binary operator");
                }
            }
            case ExprNodeType::UNOP: {
                double v = evaluate(node->child, bindings);
                switch (node->unop) {
                    case UnOp::NEG: return -v;
                    case UnOp::ABS: return std::abs(v);
                    default: throw ExprEvalError("Unknown unary operator");
                }
            }
            case ExprNodeType::FUNCTION: {
                std::string fn = node->func_name;
                // Convert to lowercase for matching
                std::transform(fn.begin(), fn.end(), fn.begin(), ::tolower);
                if (fn == "abs") {
                    if (node->func_args.size() != 1)
                        throw ExprEvalError("abs() requires 1 argument");
                    return std::abs(evaluate(node->func_args[0], bindings));
                }
                if (fn == "min") {
                    if (node->func_args.size() < 1)
                        throw ExprEvalError("min() requires at least 1 argument");
                    double m = evaluate(node->func_args[0], bindings);
                    for (size_t i = 1; i < node->func_args.size(); ++i)
                        m = std::min(m, evaluate(node->func_args[i], bindings));
                    return m;
                }
                if (fn == "max") {
                    if (node->func_args.size() < 1)
                        throw ExprEvalError("max() requires at least 1 argument");
                    double m = evaluate(node->func_args[0], bindings);
                    for (size_t i = 1; i < node->func_args.size(); ++i)
                        m = std::max(m, evaluate(node->func_args[i], bindings));
                    return m;
                }
                if (fn == "sqrt") {
                    if (node->func_args.size() != 1)
                        throw ExprEvalError("sqrt() requires 1 argument");
                    double v = evaluate(node->func_args[0], bindings);
                    if (v < 0) throw ExprEvalError("sqrt() of negative number");
                    return std::sqrt(v);
                }
                throw ExprEvalError("Unknown function: " + node->func_name);
            }
            default:
                throw ExprEvalError("Unknown expression node type");
        }
    }

    // Collect variable names used in an expression
    static void collect_vars(ExprNodePtr node, std::unordered_set<std::string>& out) {
        if (!node) return;
        if (node->type == ExprNodeType::VARIABLE) {
            out.insert(node->var_name);
            return;
        }
        collect_vars(node->left,  out);
        collect_vars(node->right, out);
        collect_vars(node->child, out);
        for (auto& a : node->func_args) collect_vars(a, out);
    }
};

// ---------------------------------------------------------------------------
// Convenience: parse and evaluate a string expression
// ---------------------------------------------------------------------------
inline double eval_expression(
        const std::string& expr_str,
        const std::unordered_map<std::string, double>& bindings) {
    ExprParser p(expr_str);
    auto tree = p.parse();
    return ExprEvaluator::evaluate(tree, bindings);
}

// ---------------------------------------------------------------------------
// ExpressionEngine — higher-level interface
// ---------------------------------------------------------------------------
class ExpressionEngine {
public:
    // Parse the expression string and return an AST.
    // Throws ExprParseError on syntax errors.
    ExprNodePtr compile(const std::string& src) const {
        ExprParser p(src);
        return p.parse();
    }

    // Evaluate a compiled expression against a binding map.
    // Throws ExprEvalError on unbound variables or runtime errors.
    double evaluate(ExprNodePtr tree,
                    const std::unordered_map<std::string, double>& bindings) const {
        return ExprEvaluator::evaluate(tree, bindings);
    }

    // Collect variable names referenced by a compiled expression.
    std::unordered_set<std::string> variables(ExprNodePtr tree) const {
        std::unordered_set<std::string> out;
        ExprEvaluator::collect_vars(tree, out);
        return out;
    }

    // Parse + evaluate in one call (convenience).
    double eval(const std::string& src,
                const std::unordered_map<std::string, double>& bindings) const {
        return eval_expression(src, bindings);
    }
};

} // namespace prime::reasoning
