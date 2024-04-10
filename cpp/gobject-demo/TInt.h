#pragma once

#include <glib-object.h>

G_BEGIN_DECLS

#define T_TYPE_INT (t_int_get_type())
G_DECLARE_FINAL_TYPE(TInt, t_int, T, INT, GObject)
struct _TInt {
  GObject parent;
  int value;
};

G_END_DECLS
