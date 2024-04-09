#include "TNumber.h"

static void t_number_init(TNumber *) {}
static void t_number_class_init(TNumberClass *) {}

G_DEFINE_TYPE(TNumber, t_number, G_TYPE_OBJECT)
