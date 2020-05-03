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
    printf("%4d: \t(%.03f, %.03f, %.03f)\t(%.03f, %.03f, %.03f)\n", i, pos[0],
           pos[1], pos[2], vel[0], vel[1], vel[2]);
  }
}

int main() {
  simdata_t *sdata = simdata_create(3, 1, 1024);
  srand(216);
  for (int i = 0; i < 1024; i++) {
    float *pos = simdata_pos_ptr(sdata, i);
    float *vel = simdata_vel_ptr(sdata, i);
    simdata_feat_ptr(sdata, i)[0] = ((float)rand() / RAND_MAX) * 100.0f;
    for (int j = 0; j < 3; j++) {
      pos[j] = ((float)rand() / RAND_MAX - 0.5f) * 1000.0f;
      vel[j] = ((float)rand() / RAND_MAX - 0.5f) * 10.0f;
    }
  }
  simconfig_t sconfig = {.precompute = false};

  measure_t *timer = start_timer();
  run_simulation(sdata, &sconfig, INT_LEAPFROG, FORCE_NEWTONIAN_SIMPLE, .1,
                 1000);
  double runtime = end_timer(timer);
  dumpt(sdata, 1);
  printf("RUNTIME: %f\n", runtime);
  simdata_free(sdata);

  return 0;
}
