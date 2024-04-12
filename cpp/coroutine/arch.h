#pragma once

extern "C" void CoSwitch(void **current_sp, void **next_sp);
extern "C" void CoInit(void **sp, void (*func)(void *), void *data);
