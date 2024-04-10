#pragma once

#include <glib-object.h>

G_BEGIN_DECLS

#define T_TYPE_ADDABLE (t_addable_get_type())
G_DECLARE_INTERFACE(TAddable, t_addable, T, ADDABLE, GObject)

struct _TAddableInterface {
  GTypeInterface parent;
  void (*inc)(TAddable *);
};

void t_addable_inc(TAddable *);

G_END_DECLS
