#pragma once

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

inline std::vector<std::string> StrSplit(const std::string &str, char c) {
  std::vector<std::string> tokens;

  std::istringstream is(str);
  for (std::string line; std::getline(is, line, c);) {
    tokens.emplace_back(std::move(line));
  }
  return tokens;
}

template <typename Op,
          typename R = std::vector<std::invoke_result_t<Op, std::string>>>
inline R StrSplit(const std::string &str, char c, Op &&op) {
  R tokens;
  std::istringstream is(str);
  for (std::string token; std::getline(is, token, c);) {
    tokens.emplace_back(op(std::move(token)));
  }
  return tokens;
}


inline std::uint64_t Event2Id(const std::string &tracefs, std::string e) {
  std::replace(e.begin(), e.end(), ':', '/');
  std::string path = tracefs + '/' + e;
  std::ifstream ifs(path);
  std::uint64_t id;
  ifs >> id;
  return id;
}
