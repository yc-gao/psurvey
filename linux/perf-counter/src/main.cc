#include <chrono>
#include <csignal>
#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "gflags/gflags.h"

#include "PerfMonitor.h"
#include "Rate.h"
#include "common.h"

DEFINE_string(tracefs, "/sys/kernel/tracing", "rate to sampling");

DEFINE_int32(rate, 1000, "rate to sampling");
DEFINE_int32(cpu, -1, "cpu to sampling");
DEFINE_string(pids, "", "comma-separated list of pids to attach");
DEFINE_string(events, "", "comma-separated list of events to trace");

bool running{true};

std::uint64_t TryConvertHwId(const std::string &e) {
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

std::uint64_t TryConvertSwId(const std::string &e) {
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

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  std::signal(SIGINT, [](int) { running = false; });
  std::signal(SIGTERM, [](int) { running = false; });

  std::vector<pid_t> pids = StrSplit(
      FLAGS_pids, ',', [](std::string token) { return std::stoi(token); });
  std::vector<std::string> events = StrSplit(FLAGS_events, ',');

  std::vector<std::uint64_t> counter(pids.size() * events.size());
  std::vector<PerfMonitor> monitors(pids.size());

  {
    std::uint64_t idx = 0;
    for (const auto &e : events) {
      perf_type_id tid;
      std::uint64_t eid = -1;
      if ((eid = TryConvertHwId(e)) != -1ul) {
        tid = PERF_TYPE_HARDWARE;
      } else if ((eid = TryConvertSwId(e)) != -1ul) {
        tid = PERF_TYPE_SOFTWARE;
      } else {
        tid = PERF_TYPE_TRACEPOINT;
        eid = Event2Id(FLAGS_tracefs, e);
      }
      for (int i = 0; i < pids.size(); i++) {
        monitors[i].Monitor(tid, eid, pids[i], FLAGS_cpu, &counter[idx]);
        idx++;
      }
    }
  }

  for (auto &&item : monitors) {
    item.Begin();
  }
  Rate rate(FLAGS_rate);
  while (running && rate.Ok()) {
    for (auto &&item : monitors) {
      item.Update();
    }
    {
      // dump result
      auto tm = std::chrono::steady_clock::now().time_since_epoch();
      for (int i = 0; i < pids.size(); i++) {
        std::cout << tm.count() << '\t' << pids[i];
        for (int j = 0; j < events.size(); j++) {
          std::cout << '\t' << counter[i * events.size() + j];
        }
        std::cout << '\n';
      }
    }
  }
  for (auto &&item : monitors) {
    item.End();
  }

  return 0;
}
