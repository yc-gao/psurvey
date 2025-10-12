#include <cstdint>
#include <iostream>

#include <cudnn.h>
#include <cudnn_frontend.h>
#include <cudnn_frontend/graph_properties.h>

#include "utils.h"

auto b = 16, m = 1024, n = 1024, k = 1024;

void do_gemm_basic(cudnnHandle_t cudnn_handler) {
  auto [graph, A, B, C] = [=]() {
    auto graph = cudnn_frontend::graph::Graph();
    auto A =
        graph.tensor(cudnn_frontend::graph::Tensor_attributes()
                         .set_name("A")
                         .set_dim({b, m, k})
                         .set_stride({m * k, k, 1})
                         .set_data_type(cudnn_frontend::DataType_t::FLOAT));
    auto B =
        graph.tensor(cudnn_frontend::graph::Tensor_attributes()
                         .set_name("B")
                         .set_dim({b, k, n})
                         .set_stride({k * n, n, 1})
                         .set_data_type(cudnn_frontend::DataType_t::FLOAT));
    auto C = graph.matmul(
        A, B,
        cudnn_frontend::graph::Matmul_attributes()
            .set_name("GEMM")
            .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT));
    C->set_data_type(cudnn_frontend::DataType_t::FLOAT);

    C->set_output(true);
    return std::make_tuple(graph, A, B, C);
  }();

  COMMON_ASSERT(graph.validate().is_good(), "graph invalied");
  COMMON_ASSERT(graph.build_operation_graph(cudnn_handler).is_good(),
                "build operation graph failed");
  COMMON_ASSERT(
      graph.create_execution_plans({cudnn_frontend::HeurMode_t::A}).is_good(),
      "create execution plans failed");
  COMMON_ASSERT(
      graph.build_plans({cudnn_frontend::BuildPlanPolicy_t::ALL}).is_good(),
      "build plans failed");

  std::int64_t workspace_size = 0;
  COMMON_ASSERT(graph.get_workspace_size(workspace_size).is_good(),
                "get workspace size failed");

  device_vector<float> A_vec(A->get_volume());
  device_vector<float> B_vec(B->get_volume());
  device_vector<float> C_vec(C->get_volume());
  device_vector<std::int8_t> w_vec(workspace_size);

  std::unordered_map<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>,
                     void*>
      args_pack{
          {A, A_vec.data()},
          {B, B_vec.data()},
          {C, C_vec.data()},
      };

  COMMON_ASSERT(graph.execute(cudnn_handler, args_pack, w_vec.data()).is_good(),
                "execute graph failed");
  cudaDeviceSynchronize();
}

void do_gemm_int8(cudnnHandle_t cudnn_handler) {
  // per tensor symmetric quant for A, B
  // per tensor symmetric quant for C
  auto [graph, A, B, C] = [=]() {
    auto graph = cudnn_frontend::graph::Graph();
    auto A = graph.tensor(cudnn_frontend::graph::Tensor_attributes()
                              .set_name("A")
                              .set_dim({b, m, k})
                              .set_stride({m * k, k, 1})
                              .set_data_type(cudnn_frontend::DataType_t::INT8));
    auto B = graph.tensor(cudnn_frontend::graph::Tensor_attributes()
                              .set_name("B")
                              .set_dim({b, k, n})
                              .set_stride({k * n, n, 1})
                              .set_data_type(cudnn_frontend::DataType_t::INT8));
    auto C = graph.matmul(
        A, B,
        cudnn_frontend::graph::Matmul_attributes()
            .set_name("GEMM")
            .set_compute_data_type(cudnn_frontend::DataType_t::INT32));
    C->set_data_type(cudnn_frontend::DataType_t::INT32);

    C = graph.pointwise(
        C, graph.tensor(5.0f),
        cudnn_frontend::graph::Pointwise_attributes()
            .set_mode(cudnn_frontend::PointwiseMode_t::MUL)
            .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT));
    C->set_data_type(cudnn_frontend::DataType_t::FLOAT);

    C = graph.pointwise(
        C, graph.tensor(1.0f),
        cudnn_frontend::graph::Pointwise_attributes()
            .set_mode(cudnn_frontend::PointwiseMode_t::ADD)
            .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT));
    C->set_data_type(cudnn_frontend::DataType_t::FLOAT);

    C = graph.pointwise(
        C, graph.tensor(5.0f),
        cudnn_frontend::graph::Pointwise_attributes()
            .set_mode(cudnn_frontend::PointwiseMode_t::MUL)
            .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT));
    C->set_data_type(cudnn_frontend::DataType_t::INT8);

    C->set_output(true);
    return std::make_tuple(graph, A, B, C);
  }();

  COMMON_ASSERT(graph.validate().is_good(), "graph invalied");
  COMMON_ASSERT(graph.build_operation_graph(cudnn_handler).is_good(),
                "build operation graph failed");
  COMMON_ASSERT(
      graph.create_execution_plans({cudnn_frontend::HeurMode_t::A}).is_good(),
      "create execution plans failed");
  COMMON_ASSERT(
      graph.build_plans({cudnn_frontend::BuildPlanPolicy_t::ALL}).is_good(),
      "build plans failed");

  std::int64_t workspace_size = 0;
  COMMON_ASSERT(graph.get_workspace_size(workspace_size).is_good(),
                "get workspace size failed");

  device_vector<float> A_vec(A->get_volume());
  device_vector<float> B_vec(B->get_volume());
  device_vector<float> C_vec(C->get_volume());
  device_vector<std::int8_t> w_vec(workspace_size);

  std::unordered_map<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>,
                     void*>
      args_pack{
          {A, A_vec.data()},
          {B, B_vec.data()},
          {C, C_vec.data()},
      };

  COMMON_ASSERT(graph.execute(cudnn_handler, args_pack, w_vec.data()).is_good(),
                "execute graph failed");
  cudaDeviceSynchronize();
}

int main(int argc, char* argv[]) {
  auto cudnn_handler = create_cudnn_handle();
  do_gemm_basic(*cudnn_handler);
  do_gemm_int8(*cudnn_handler);

  return 0;
}
