#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <vector>

class NvLogger : public nvinfer1::ILogger {
  void log(Severity severity, const char* msg) noexcept override {}
};

void test_trt(void* buf, std::size_t size) {
  NvLogger logger;
  auto runtime = nvinfer1::createInferRuntime(logger);
  auto engine = runtime->deserializeCudaEngine(buf, size);
  auto context = engine->createExecutionContext();

  float* d_input;
  float* d_output;
  cudaMallocManaged((void**)&d_input, sizeof(float) * 1 * 3 * 224 * 224);
  cudaMallocManaged((void**)&d_output, sizeof(float) * 1000);
  std::fill_n(d_input, 1 * 3 * 224 * 224, 1);
  std::fill_n(d_output, 1000, 0);

  context->setTensorAddress("x", d_input);
  context->setTensorAddress("y", d_output);

  context->enqueueV3(0);
  cudaStreamSynchronize(0);

  for (auto i = 0; i < 10; i++) {
    std::cout << d_output[i] << '\n';
  }
}

void test_trt(char const* filename) {
  std::vector<char> buffer;
  {
    std::ifstream ifs(filename, std::ios::binary);
    ifs.seekg(0, ifs.end);
    buffer.resize(ifs.tellg());
    ifs.seekg(0, ifs.beg);
    ifs.read(buffer.data(), buffer.size());
  }

  test_trt(buffer.data(), buffer.size());
}

void test_onnx(char const* filename) {
  NvLogger logger;

  auto builder = nvinfer1::createInferBuilder(logger);
  auto network = builder->createNetworkV2(0);
  auto builder_config = builder->createBuilderConfig();

  auto onnx_parser = nvonnxparser::createParser(*network, logger);
  onnx_parser->parseFromFile(filename,
                             static_cast<int>(NvLogger::Severity::kWARNING));

  auto serializedModel =
      builder->buildSerializedNetwork(*network, *builder_config);

  test_trt(serializedModel->data(), serializedModel->size());
}

int main(int argc, char* argv[]) {
  std::string filename = argv[1];
  if (std::equal(filename.rbegin(), filename.rbegin() + 3, "trt")) {
    test_trt(argv[1]);
  } else {
    test_onnx(argv[1]);
  }
  return 0;
}
