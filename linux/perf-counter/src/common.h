#pragma once

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <linux/perf_event.h>

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
  std::string path = tracefs + "/events/" + e + "/id";
  std::ifstream ifs(path);
  std::uint64_t id;
  ifs >> id;
  return id;
}

inline std::uint64_t TryConvertHwId(const std::string &e) {
  static std::unordered_map<std::string, std::uint64_t> e2id{
      {"PERF_COUNT_HW_CPU_CYCLES", PERF_COUNT_HW_CPU_CYCLES},
      {"PERF_COUNT_HW_INSTRUCTIONS", PERF_COUNT_HW_INSTRUCTIONS},
      {"PERF_COUNT_HW_CACHE_REFERENCES", PERF_COUNT_HW_CACHE_REFERENCES},
      {"PERF_COUNT_HW_CACHE_MISSES", PERF_COUNT_HW_CACHE_MISSES},
      {"PERF_COUNT_HW_BRANCH_INSTRUCTIONS", PERF_COUNT_HW_BRANCH_INSTRUCTIONS},
      {"PERF_COUNT_HW_BRANCH_MISSES", PERF_COUNT_HW_BRANCH_MISSES},
      {"PERF_COUNT_HW_BUS_CYCLES", PERF_COUNT_HW_BUS_CYCLES},
      {"PERF_COUNT_HW_STALLED_CYCLES_FRONTEND",
       PERF_COUNT_HW_STALLED_CYCLES_FRONTEND},
      {"PERF_COUNT_HW_STALLED_CYCLES_BACKEND",
       PERF_COUNT_HW_STALLED_CYCLES_BACKEND},
      {"PERF_COUNT_HW_REF_CPU_CYCLES", PERF_COUNT_HW_REF_CPU_CYCLES},
  };
  auto iter = e2id.find(e);
  if (iter == e2id.end()) {
    return -1ul;
  }
  return iter->second;
}

inline std::uint64_t TryConvertSwId(const std::string &e) {
  static std::unordered_map<std::string, std::uint64_t> e2id{
      {"PERF_COUNT_SW_CPU_CLOCK", PERF_COUNT_SW_CPU_CLOCK},
      {"PERF_COUNT_SW_TASK_CLOCK", PERF_COUNT_SW_TASK_CLOCK},
      {"PERF_COUNT_SW_PAGE_FAULTS", PERF_COUNT_SW_PAGE_FAULTS},
      {"PERF_COUNT_SW_CONTEXT_SWITCHES", PERF_COUNT_SW_CONTEXT_SWITCHES},
      {"PERF_COUNT_SW_CPU_MIGRATIONS", PERF_COUNT_SW_CPU_MIGRATIONS},
      {"PERF_COUNT_SW_PAGE_FAULTS_MIN", PERF_COUNT_SW_PAGE_FAULTS_MIN},
      {"PERF_COUNT_SW_PAGE_FAULTS_MAJ", PERF_COUNT_SW_PAGE_FAULTS_MAJ},
      {"PERF_COUNT_SW_ALIGNMENT_FAULTS", PERF_COUNT_SW_ALIGNMENT_FAULTS},
      {"PERF_COUNT_SW_EMULATION_FAULTS", PERF_COUNT_SW_EMULATION_FAULTS},
      {"PERF_COUNT_SW_DUMMY", PERF_COUNT_SW_DUMMY},
      {"PERF_COUNT_SW_BPF_OUTPUT", PERF_COUNT_SW_BPF_OUTPUT},
  };
  auto iter = e2id.find(e);
  if (iter == e2id.end()) {
    return -1ul;
  }
  return iter->second;
}
