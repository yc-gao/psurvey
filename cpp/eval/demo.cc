#include <cassert>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <map>
#include <stack>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

enum class TokenType { BRACKET, OP, INT, VAR };

struct Token {
  TokenType type;
  std::string tok;
  union {
    std::int64_t val;
    char op;
  };

  Token(TokenType type, std::string tok) : type(type), tok(tok) {}
  Token(TokenType type, char op) : type(type), op(op) {}
  Token(TokenType type, std::int64_t val) : type(type), val(val) {}

  friend std::ostream &operator<<(std::ostream &os, const Token &tok) {
    os << "Token{type=" << static_cast<int>(tok.type);
    if (tok.type == TokenType::OP || tok.type == TokenType::BRACKET) {
      os << ", op=" << tok.op;
    } else if (tok.type == TokenType::INT) {
      os << ", val=" << tok.val;
    } else {
      os << ", tok=" << tok.tok;
    }
    os << "}";
    return os;
  }
};

class TokenEvaler {
  std::vector<Token> tokens;

  void ShuntingYard(std::vector<Token> &tokens) {
    std::vector<Token> res;
    std::stack<Token> sop;

    std::map<char, int> op2prio{{'(', 0}, {'+', 1}, {'-', 1},
                                {'*', 2}, {'/', 2}, {'%', 2}};

    res.reserve(tokens.size());
    for (int i = 0; i < tokens.size(); i++) {
      switch (tokens[i].type) {
      case TokenType::INT:
      case TokenType::VAR:
        res.emplace_back(std::move(tokens[i]));
        break;
      case TokenType::OP: {
        if (sop.empty() || sop.top().op == '(' ||
            op2prio[sop.top().op] < op2prio[tokens[i].op]) {
          sop.emplace(std::move(tokens[i]));
        } else {
          while (!sop.empty() &&
                 op2prio[sop.top().op] >= op2prio[tokens[i].op]) {
            res.emplace_back(std::move(sop.top()));
            sop.pop();
          }
          sop.emplace(std::move(tokens[i]));
        }
      } break;
      case TokenType::BRACKET: {
        if (tokens[i].op == '(') {
          sop.emplace(std::move(tokens[i]));
        } else {
          while (!sop.empty() && sop.top().type == TokenType::OP) {
            res.emplace_back(std::move(sop.top()));
            sop.pop();
          }
          if (sop.empty() || sop.top().type != TokenType::BRACKET ||
              sop.top().op != '(') {
            throw std::logic_error("illegal expr");
          }
          sop.pop();
        }
      } break;
      default:
        throw std::logic_error("illegal token type");
        break;
      }
    }
    while (!sop.empty()) {
      res.emplace_back(std::move(sop.top()));
      sop.pop();
    }
    tokens = std::move(res);
  }

public:
  TokenEvaler(const TokenEvaler &) = default;
  TokenEvaler(TokenEvaler &&) = default;
  TokenEvaler &operator=(const TokenEvaler &) = default;
  TokenEvaler &operator=(TokenEvaler &&) = default;

  TokenEvaler(std::vector<Token> tokens) : tokens(std::move(tokens)) {
    ShuntingYard(this->tokens);
  }

  std::int64_t Eval(const std::map<std::string, std::int64_t> &vars) {
    std::stack<std::int64_t> st;
    for (int i = 0; i < tokens.size(); i++) {
      switch (tokens[i].type) {
      case TokenType::INT:
        st.push(tokens[i].val);
        break;
      case TokenType::VAR:
        st.push(vars.at(tokens[i].tok));
        break;
      case TokenType::OP: {
        std::int64_t rhs = st.top();
        st.pop();
        std::int64_t lhs = st.top();
        st.pop();
        switch (tokens[i].op) {
        case '+':
          st.push(lhs + rhs);
          break;
        case '-':
          st.push(lhs - rhs);
          break;
        case '*':
          st.push(lhs * rhs);
          break;
        case '/':
          st.push(lhs / rhs);
          break;
        case '%':
          st.push(lhs % rhs);
          break;
        default:
          throw std::logic_error("illegal token");
          break;
        }
      } break;
      default:
        throw std::logic_error("illegal expr");
        break;
      }
    }
    if (st.size() != 1) {
      throw std::logic_error("illegal expr");
    }
    return st.top();
  }
};

class Tokenizer {
  const char *buf_;
  std::size_t size_;

public:
  struct TokenizerIterator
      : public std::iterator<std::input_iterator_tag, std::string> {

    Tokenizer *tokenizer;
    std::size_t pos;
    std::size_t size;
    TokenType type;

    TokenizerIterator() : TokenizerIterator(nullptr, 0, 0, TokenType()) {}

    TokenizerIterator(Tokenizer *tokenizer, std::size_t pos, std::size_t size,
                      TokenType type)
        : tokenizer(tokenizer), pos(pos), size(size), type(type) {}

    Token operator*() const {
      if (type == TokenType::OP || type == TokenType::BRACKET) {
        return Token(type, tokenizer->At(pos));
      } else if (type == TokenType::INT) {
        std::int64_t val = std::stoll(tokenizer->Str(pos, size));
        return Token(type, val);
      } else {
        return Token(type, tokenizer->Str(pos, size));
      }
    }

    TokenizerIterator &operator++() {
      *this = tokenizer->begin(pos + size);
      return *this;
    }
    TokenizerIterator operator++(int) {
      auto tmp = *this;
      ++*this;
      return tmp;
    }
    bool operator==(const TokenizerIterator &other) const {
      return this == &other || (tokenizer == other.tokenizer &&
                                pos == other.pos && size == other.size);
    }
    bool operator!=(const TokenizerIterator &other) const {
      return !(*this == other);
    }
  };

  Tokenizer(const char *str, std::size_t size) : buf_(str), size_(size) {}
  Tokenizer(const std::string &str) : Tokenizer(str.data(), str.size()) {}

  char At(std::size_t pos) const { return buf_[pos]; }
  std::string Str(std::size_t pos, std::size_t size) const {
    if (pos + size > size_) {
      throw std::out_of_range("out of range exception");
    }
    return std::string(buf_ + pos, size);
  }

  Token GetToken(std::size_t pos = 0) { return *begin(pos); }

  TokenizerIterator begin(std::size_t pos = 0) {
    if (pos >= size_ || !buf_) {
      return {};
    }
    switch (buf_[pos]) {
    case ' ':
      return begin(pos + 1);
      break;
    case '+':
    case '-':
    case '*':
    case '/':
    case '%':
      return {this, pos, 1, TokenType::OP};
      break;
    case ')':
    case '(':
      return {this, pos, 1, TokenType::BRACKET};
      break;
    default: {
      bool has_num = false;
      bool has_alpha = false;

      auto end = pos;
      while (end < size_) {
        bool at_end = false;
        switch (buf_[end]) {
        case ' ':
        case '+':
        case '-':
        case '*':
        case '/':
        case '%':
        case '(':
        case ')':
          at_end = true;
          break;
        default: {
          if (buf_[end] >= '0' && buf_[end] <= '9') {
            has_num = true;
          } else {
            has_alpha = true;
          }
          end++;
        } break;
        }
        if (at_end) {
          break;
        }
      }
      if (has_num && !has_alpha) {
        return {this, pos, end - pos, TokenType::INT};
      } else {
        return {this, pos, end - pos, TokenType::VAR};
      }
    } break;
    }
  }
  TokenizerIterator end() { return begin(size_); }
};

struct TimeMersure {
  std::string name;
  std::chrono::time_point<std::chrono::steady_clock> start;

  ~TimeMersure() {
    auto end = std::chrono::steady_clock::now();
    std::cout << "name: " << name << ", duration: "
              << std::chrono::duration_cast<std::chrono::nanoseconds>(end -
                                                                      start)
                     .count()
              << "ns" << std::endl;
  }
  TimeMersure(std::string name) : name(std::move(name)) {
    start = std::chrono::steady_clock::now();
  }
};

void print(const std::vector<Token> &tokens) {
  int idx = 0;
  for (auto tok : tokens) {
    std::cout << idx++ << ": " << tok << std::endl;
  }
}

int main(int argc, char *argv[]) {
  std::string str("frame % 5");
  Tokenizer tokenizer(str);

  std::vector<Token> tokens;
  std::copy(tokenizer.begin(), tokenizer.end(), std::back_inserter(tokens));

  TokenEvaler evaler(std::move(tokens));
  std::int64_t val;
  {
    TimeMersure demo("evaler");
    for (std::size_t i = 0; i < 100; i++) {
      val = evaler.Eval({{"frame", i}});
    }
  }
  {
    TimeMersure demo("native");
    for (std::size_t i = 0; i < 100; i++) {
      val = i % 5;
    }
  }
  return 0;
}
