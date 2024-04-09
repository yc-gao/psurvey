#pragma once
#include "TNumber.h"

G_BEGIN_DECLS

G_DECLARE_FINAL_TYPE(TInt, t_int, T, INT, TNumber)
struct _TInt {
  TNumber parent;
  int value;
};

G_END_DECLS
