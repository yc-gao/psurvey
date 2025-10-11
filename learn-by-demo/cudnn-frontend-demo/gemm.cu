#include <cstdint>

#include <cudnn.h>
#include <cudnn_frontend.h>
#include <cudnn_frontend/graph_properties.h>
#include <thrust/device_vector.h>

#include "utils.h"

void do_test0(cudnnHandle_t cudnn_handler) {
  auto b = 16, m = 32, n = 64, k = 128;

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

  thrust::device_vector<std::int8_t> A_vec(A->get_volume());
  thrust::device_vector<std::int8_t> B_vec(B->get_volume());
  thrust::device_vector<float> C_vec(C->get_volume());
  thrust::device_vector<std::int8_t> w_vec(workspace_size);

  std::unordered_map<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>,
                     void*>
      args_pack{
          {A, A_vec.data().get()},
          {B, B_vec.data().get()},
          {C, C_vec.data().get()},
      };

  COMMON_ASSERT(
      graph.execute(cudnn_handler, args_pack, w_vec.data().get()).is_good(),
      "execute graph failed");
}

void do_test1(cudnnHandle_t cudnn_handler) {
  auto b = 16, m = 32, n = 64, k = 128;

  auto [graph, A, B, Bias, C] = [=]() {
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
    C->set_data_type(cudnn_frontend::DataType_t::FLOAT);

    C = graph.pointwise(
        C, graph.tensor(3.0f),
        cudnn_frontend::graph::Pointwise_attributes()
            .set_mode(cudnn_frontend::PointwiseMode_t::MUL)
            .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT));
    C->set_data_type(cudnn_frontend::DataType_t::FLOAT);

    auto Bias =
        graph.tensor(cudnn_frontend::graph::Tensor_attributes()
                         .set_dim({1, 1, n})
                         .set_stride({n, n, 1})
                         .set_data_type(cudnn_frontend::DataType_t::FLOAT));
    C = graph.pointwise(
        C, Bias,
        cudnn_frontend::graph::Pointwise_attributes()
            .set_mode(cudnn_frontend::PointwiseMode_t::ADD)
            .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT));
    C->set_data_type(cudnn_frontend::DataType_t::FLOAT);

    C = graph.pointwise(
        C, graph.tensor(3.0f),
        cudnn_frontend::graph::Pointwise_attributes()
            .set_mode(cudnn_frontend::PointwiseMode_t::MUL)
            .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT));
    C->set_data_type(cudnn_frontend::DataType_t::INT8);

    C->set_output(true);
    return std::make_tuple(graph, A, B, Bias, C);
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

  thrust::device_vector<std::int8_t> A_vec(A->get_volume());
  thrust::device_vector<std::int8_t> B_vec(B->get_volume());
  thrust::device_vector<float> Bias_vec(Bias->get_volume());
  thrust::device_vector<std::int8_t> C_vec(C->get_volume());
  thrust::device_vector<std::int8_t> w_vec(workspace_size);

  std::unordered_map<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>,
                     void*>
      args_pack{
          {A, A_vec.data().get()},
          {B, B_vec.data().get()},
          {Bias, Bias_vec.data().get()},
          {C, C_vec.data().get()},
      };

  COMMON_ASSERT(
      graph.execute(cudnn_handler, args_pack, w_vec.data().get()).is_good(),
      "execute graph failed");
}

int main(int argc, char* argv[]) {
  auto cudnn_handler = create_cudnn_handle();
  do_test0(*cudnn_handler);
  do_test1(*cudnn_handler);

  return 0;
}
