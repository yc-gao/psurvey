#include <iostream>

#include "degine/degine.h"

void Usage(int argc, char *argv[]) {
  std::cout << "usage: " << argv[0] << " <model>" << std::endl;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    Usage(argc, argv);
    return 1;
  }
  using namespace degine;
  EngineOptions options{argv[1], "", "cuda"};
  auto engine = Engine::BuildEngine(options);
  return 0;
}
