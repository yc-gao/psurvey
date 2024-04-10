#pragma once

#include <glib-object.h>

G_BEGIN_DECLS

#define T_TYPE_NUMBER (t_number_get_type())
G_DECLARE_DERIVABLE_TYPE(TNumber, t_number, T, NUMBER, GObject)

struct _TNumberClass {
  GObjectClass parent;
  void (*print)(TNumber *);
};

void t_number_print(TNumber *);

G_END_DECLS
