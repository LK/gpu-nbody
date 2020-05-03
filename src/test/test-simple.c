#include "nbodysim.h"
#include "timing.h"
#include <stdio.h>
#include <stdlib.h>

void dumpt(simdata_t *sdata, int step) {
  printf("=============== STEP %d ===============\n", step);
  float *pos, *vel;
  for (int i = 0; i < sdata->nparticles; i++) {
    pos = simdata_pos_ptr(sdata, i);
    vel = simdata_vel_ptr(sdata, i);
    printf("\t(%.04f, %.04f)\t(%.04f, %.04f)\n", pos[0], pos[1], vel[0],
           vel[1]);
  }
}

int main() {
  simdata_t *sdata = simdata_create(2, 1, 2);
  sdata->data[0] = 10;
  sdata->data[1] = 0;
  sdata->data[2] = 0;
  sdata->data[3] = 0;
  sdata->data[4] = 10;
  sdata->data[5] = -10;
  sdata->data[6] = 0;
  sdata->data[7] = 0;
  sdata->data[8] = 0;
  sdata->data[9] = 10;
  measure_t *timer = start_timer();
  run_simulation(sdata, INT_EULER, FORCE_NEWTONIAN_SIMPLE, .1, 100);
  double runtime = end_timer(timer);
  dumpt(sdata, 1);
  printf("RUNTIME: %f\n", runtime);
  simdata_free(sdata);

  return 0;
}
