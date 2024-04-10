#include "TNumber.h"
#include <cstdio>

struct TNumberPrivate {};

G_DEFINE_TYPE_WITH_PRIVATE(TNumber, t_number, G_TYPE_OBJECT);

static void t_number_class_init(TNumberClass *klass) {
  klass->print = [](TNumber *self) {
    g_return_if_fail(T_IS_NUMBER(self));
    printf("TNumber print\n");
  };
}
static void t_number_init(TNumber *self) {
  TNumberPrivate *priv = (TNumberPrivate *)t_number_get_instance_private(self);
}

void t_number_print(TNumber *self) {
  g_return_if_fail(T_IS_NUMBER(self));
  TNumberClass *klass = T_NUMBER_GET_CLASS(self);
  g_return_if_fail(klass->print != NULL);
  klass->print(self);
}
