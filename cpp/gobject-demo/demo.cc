#include <iostream>

#include "TInt.h"

int main(int argc, char *argv[]) {
  TInt *num = T_INT(g_object_new(t_int_get_type(), "value", 123, nullptr));
  std::cout << "value: " << num->value << std::endl;
  g_object_unref(num);
  return 0;
}
