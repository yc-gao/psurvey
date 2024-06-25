#include <pybind11/pybind11.h>

#include <iostream>
#include <string>

int add(int i, int j) { return i + j; }

struct Pet {
  std::string name;

  Pet(const std::string &name) : name(name) {
    std::cout << "new pet, name: " << name << std::endl;
  }

  void setName(const std::string &name_) { name = name_; }
  const std::string &getName() const { return name; }
};

namespace py = pybind11;
PYBIND11_MODULE(demo, m) {
  m.doc() = "pybind11 example plugin";  // optional module docstring

  m.def("add", &add, "A function that adds two numbers");

  py::class_<Pet>(m, "Pet")
      .def(py::init<const std::string &>())
      .def("setName", &Pet::setName)
      .def("getName", &Pet::getName);
}
