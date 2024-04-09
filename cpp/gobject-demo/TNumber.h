#pragma once

#include <glib-object.h>

G_BEGIN_DECLS

G_DECLARE_DERIVABLE_TYPE(TNumber, t_number, T, NUMBER, GObject)
struct _TNumberClass {
  GObjectClass parent;
};

G_END_DECLS
