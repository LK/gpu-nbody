#include "nbodysim.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void getForce(float *force, float *position, float *features,
              float *positionActor, float *featuresActor, simdata_t *sdata) {

  float feature = features[0];
  float featureActor = featuresActor[0];

  float distance = 0;
  for (int i = 0; i < sdata->posdim; i++) {
    float pos = position[i];
    float posActor = positionActor[i];
    float deltaPos = pos - posActor;
    distance += deltaPos * deltaPos;
  }
  distance = sqrt(distance);

  for (int i = 0; i < sdata->posdim; i++) {
    float pos = position[i];
    float posActor = positionActor[i];
    float deltaPos = pos - posActor;

    float f =
        feature * featureActor / distance / distance * deltaPos / distance;
    force[i] -= f;
  }
}

void integrator(float *position, float *velocity, float *acceleration,
                float timeStep, simdata_t *sdata) {
  for (int i = 0; i < sdata->posdim; i++) {
    position[i] +=
        velocity[i] * timeStep + 0.5 * acceleration[i] * timeStep * timeStep;
    velocity[i] += acceleration[i] * timeStep;
  }
}

void dump(simdata_t *sdata, int step) {
  printf("=============== STEP %d ===============\n", step);
  float *pos, *vel;
  for (int i = 0; i < sdata->nparticles; i++) {
    pos = simdata_pos_ptr(sdata, i);
    vel = simdata_vel_ptr(sdata, i);
    printf("\t(%.04f, %.04f)\t(%.04f, %.04f)\n", pos[0], pos[1], vel[0],
           vel[1]);
  }
}

void run_simulation(simdata_t *sdata, integrator_t int_type, force_t force_type,
                    simulator_mode_t mode, float time_step, int steps) {
  float *accelerations =
      (float *)malloc(sizeof(float) * sdata->posdim * sdata->nparticles);
  for (int step = 0; step < steps; step++) {
    for (int i = 0; i < sdata->nparticles; i++) {
      float *acceleration = &accelerations[i * sdata->posdim];
      memset(acceleration, 0, sdata->posdim * sizeof(float));

      float *position = simdata_pos_ptr(sdata, i);
      float *features = simdata_feat_ptr(sdata, i);

      for (int j = 0; j < sdata->nparticles; j++) {
        if (i == j)
          continue;
        float *positionActor = simdata_pos_ptr(sdata, j);
        float *featuresActor = simdata_feat_ptr(sdata, j);

        getForce(acceleration, position, features, positionActor, featuresActor,
                 sdata);
      }
      for (int j = 0; j < sdata->posdim; j++) {
        acceleration[j] = acceleration[j] * get_multiplier(mode) / features[0];
      }
    }

    for (int i = 0; i < sdata->nparticles; i++) {
      integrator(simdata_pos_ptr(sdata, i), simdata_vel_ptr(sdata, i),
                 accelerations + i * sdata->posdim, time_step, sdata);
    }
  }
  free(accelerations);
}
