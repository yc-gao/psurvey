#include "TInt.h"

static void t_int_init(TInt *) {}
static void t_int_class_init(TIntClass *kclass) {
  GObjectClass *object_class = G_OBJECT_CLASS(kclass);
  object_class->set_property = [](GObject *object, guint property_id,
                                  const GValue *value, GParamSpec *pspec) {
    TInt *self = T_INT(object);
    if (property_id == 1) {
      self->value = g_value_get_int(value);
    } else {
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, pspec);
    }
  };
  object_class->get_property = [](GObject *object, guint property_id,
                                  GValue *value, GParamSpec *pspec) {
    TInt *self = T_INT(object);
    if (property_id == 1) {
      g_value_set_int(value, self->value);
    } else {
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, pspec);
    }
  };
  g_object_class_install_property(
      object_class, 1,
      g_param_spec_int("value", "val", "Integer value", G_MININT, G_MAXINT, 0,
                       G_PARAM_READWRITE));
}

G_DEFINE_TYPE(TInt, t_int, G_TYPE_OBJECT)
