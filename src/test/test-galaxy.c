#include "nbodysim.h"
#include "timing.h"
#include <stdio.h>
#include <stdlib.h>

#define MAXSTR (256)
#define TIME_STEP (0.01)
#define STEPS (1000)

float scaleFactor = 1.5f;     // 10.0f, 50
float velFactor = 8.0f;       // 15.0f, 100
float massFactor = 120000.0f; // 50000000.0,
int NUM_BODIES = 4096;

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

void load_data(simdata_t *sdata) {
  int skip = 40960 / NUM_BODIES;

  FILE *galaxy_in;

  if ((galaxy_in = fopen("./data/dubinski.tab", "r"))) {
    char buf[MAXSTR];
    float v[7];
    float *position, *velocity, *features;

    // total 81920 particles
    // 16384 Gal. Disk
    // 16384 And. Disk
    // 8192  Gal. bulge
    // 8192  And. bulge
    // 16384 Gal. halo
    // 16384 And. halo
    int k = 0;
    for (int i = 0; i < NUM_BODIES; i++, k++) {
      // depend on input size...
      for (int j = 0; j < skip; j++, k++)
        fgets(buf, MAXSTR, galaxy_in); // lead line

      sscanf(buf, "%f %f %f %f %f %f %f", v + 0, v + 1, v + 2, v + 3, v + 4,
             v + 5, v + 6);

      // position
      position = simdata_pos_ptr(sdata, i);
      velocity = simdata_vel_ptr(sdata, i);
      features = simdata_feat_ptr(sdata, i);

      position[0] = v[1] * scaleFactor;
      position[1] = v[2] * scaleFactor;
      position[2] = v[3] * scaleFactor;

      // mass
      features[0] = v[0] * massFactor;

      // velocity
      velocity[0] = v[4] * velFactor;
      velocity[1] = v[5] * velFactor;
      velocity[2] = v[6] * velFactor;
    }
  }

  fclose(galaxy_in);
}

int main(int argc, char **argv) {

  int steps = STEPS;
  if (argc == 2)
    steps = atoi(argv[1]);
  else if (argc == 3) {
    steps = atoi(argv[1]);
    if (atoi(argv[2]) % 2 != 0) {
      printf("Bodies must be multiples of 2\n");
      exit(1);
    }
    NUM_BODIES = atoi(argv[2]);
  }

  simdata_t *sdata = simdata_create(3, 1, NUM_BODIES);
  load_data(sdata);

  simconfig_t sconfig = {.precompute = false};
  run_simulation(sdata, &sconfig, INT_LEAPFROG, FORCE_NEWTONIAN_SIMPLE,
                 TIME_STEP, steps);

  simdata_free(sdata);

  return 0;
}
