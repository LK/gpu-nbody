#include "timing.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

void timing(double *wcTime) {
  struct timeval tp;

  gettimeofday(&tp, NULL);
  *wcTime = (double)(tp.tv_sec + tp.tv_usec / 1000000.0);
}

measure_t *start_timer() {
  measure_t *c = (measure_t *)malloc(sizeof(measure_t));
  timing(c);
  return c;
}

double end_timer(measure_t *timer) {
  double time = end_timer_silent(timer);
  fprintf(stderr, "RUNTIME: %f\n", time);
  return time;
}

double end_timer_silent(measure_t *timer) {
  double end;
  timing(&end);
  end -= *timer;
  free(timer);
  return end;
}
