#include "TAddable.h"

G_DEFINE_INTERFACE(TAddable, t_addable, G_TYPE_OBJECT)

static void t_addable_default_init(TAddableInterface *iface) {
  iface->inc = [](TAddable *) {
    g_warning("Default perform_action called. This should be implemented by "
              "the interface user.");
  };
}

void t_addable_inc(TAddable *self) {
  g_return_if_fail(T_IS_ADDABLE(self));
  TAddableInterface *iface = T_ADDABLE_GET_IFACE(self);
  iface->inc(self);
}
