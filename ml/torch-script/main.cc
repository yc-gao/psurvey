#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>
#include <vector>

int main(int argc, const char *argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }

  torch::jit::script::Module module;
  try {
    module = torch::jit::load(argv[1]);
  } catch (const c10::Error &e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::randn({4}));
  std::cout << "input:\n" << inputs[0] << std::endl;

  at::Tensor output = module.forward(inputs).toTensor();
  std::cout << "output:\n" << output << std::endl;
}
