#include "TInt.h"

G_DEFINE_TYPE(TInt, t_int, G_TYPE_OBJECT)

static void t_int_set_property(GObject *object, guint property_id,
                               const GValue *value, GParamSpec *pspec) {

  TInt *self = T_INT(object);
  switch (property_id) {
  case 1:
    self->value = g_value_get_int(value);
    break;
  default:
    G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, pspec);
    break;
  }
}

static void t_int_get_property(GObject *object, guint property_id,
                               GValue *value, GParamSpec *pspec) {
  TInt *self = T_INT(object);
  switch (property_id) {
  case 1:
    g_value_set_int(value, self->value);
    break;
  default:
    G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, pspec);
    break;
  }
}

static void t_int_class_init(TIntClass *klass) {
  GObjectClass *object_class = G_OBJECT_CLASS(klass);
  object_class->set_property = t_int_set_property;
  object_class->get_property = t_int_get_property;
  g_object_class_install_property(object_class, 1,
                                  g_param_spec_int("value", "Value", "int num",
                                                   G_MININT, // min value
                                                   G_MAXINT, // max value
                                                   0,        // default value
                                                   G_PARAM_READWRITE));
}
static void t_int_init(_TInt *) {}
