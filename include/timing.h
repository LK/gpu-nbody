#include <stdbool.h>

typedef double measure_t;

measure_t *start_timer();
double end_timer(measure_t *timer);
double end_timer_silent(measure_t *timer);
