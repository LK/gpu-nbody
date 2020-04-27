#include "nbodysim.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//#define G (6.673 * pow(10, -11))
#define G 1

#define GET_POSITION(sdata, i)                                                 \
  (sdata->data + i * (sdata->posdim * 2 + sdata->featdim))
#define GET_VELOCITY(sdata, i)                                                 \
  (sdata->data + i * (sdata->posdim * 2 + sdata->featdim) + sdata->posdim)
#define GET_FEATURES(sdata, i)                                                 \
  (sdata->data + i * (sdata->posdim * 2 + sdata->featdim) + 2 * sdata->posdim)

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
  for (int i = 0; i < sdata->nparticles; i++) {
    printf("\t(%.04f, %.04f)\t(%.04f, %.04f)\n", GET_POSITION(sdata, i)[0],
           GET_POSITION(sdata, i)[1], GET_VELOCITY(sdata, i)[0],
           GET_VELOCITY(sdata, i)[1]);
  }
}

void runSimulation(simdata_t *sdata, integrator_t int_type, force_t force_type,
                   float timeStep, int steps) {
  float *accelerations =
      (float *)malloc(sizeof(float) * sdata->posdim * sdata->nparticles);
  for (int step = 0; step < steps; step++) {
    for (int i = 0; i < sdata->nparticles; i++) {
      float *acceleration = &accelerations[i * sdata->posdim];
      memset(acceleration, 0, sdata->posdim * sizeof(float));

      float *position = GET_POSITION(sdata, i);
      float *features = GET_FEATURES(sdata, i);

      for (int j = 0; j < sdata->nparticles; j++) {
        if (i == j)
          continue;
        float *positionActor = GET_POSITION(sdata, j);
        float *featuresActor = GET_FEATURES(sdata, j);

        getForce(acceleration, position, features, positionActor, featuresActor,
                 sdata);
      }
      for (int j = 0; j < sdata->posdim; j++) {
        acceleration[j] = acceleration[j] * G / features[0];
      }
    }

    for (int i = 0; i < sdata->nparticles; i++) {
      integrator(GET_POSITION(sdata, i), GET_VELOCITY(sdata, i),
                 accelerations + i * sdata->posdim, timeStep, sdata);
    }
    dump(sdata, step);
  }
}
