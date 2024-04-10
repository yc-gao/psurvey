#include <iostream>

#include "TInt.h"
#include "TNumber.h"

int main(int argc, char *argv[]) {
  {
    TInt *num = T_INT(g_object_new(T_TYPE_INT, "value", 123, nullptr));
    std::cout << "value: " << num->value << std::endl;
    g_object_unref(num);
  }
  {
    TNumber *num = T_NUMBER(g_object_new(T_TYPE_NUMBER, nullptr));
    t_number_print(num);
    g_object_unref(num);
  }

  return 0;
}
