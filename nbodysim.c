#include "nbodysim.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//#define G (6.673 * pow(10, -11))
#define G 1

#define GET_POSITION(sdata, i)                                                 \
  (sdata.data + i * (sdata.dim * 2 + sdata.dataDim))
#define GET_VELOCITY(sdata, i)                                                 \
  (sdata.data + i * (sdata.dim * 2 + sdata.dataDim) + sdata.dim)
#define GET_FEATURES(sdata, i)                                                 \
  (sdata.data + i * (sdata.dim * 2 + sdata.dataDim) + 2 * sdata.dim)

void getForce(float *force, float *position, float *features,
              float *positionActor, float *featuresActor, struct simdata data) {

  float feature = features[0];
  float featureActor = featuresActor[0];

  float distance = 0;
  for (int i = 0; i < data.dim; i++) {
    float pos = position[i];
    float posActor = positionActor[i];
    float deltaPos = pos - posActor;
    distance += deltaPos * deltaPos;
  }
  distance = sqrt(distance);

  for (int i = 0; i < data.dim; i++) {
    float pos = position[i];
    float posActor = positionActor[i];
    float deltaPos = pos - posActor;

    float f =
        feature * featureActor / distance / distance * deltaPos / distance;
    force[i] -= f;
  }
}

void integrator(float *position, float *velocity, float *acceleration,
                float timeStep, struct simdata sdata) {
  for (int i = 0; i < sdata.dim; i++) {
    position[i] +=
        velocity[i] * timeStep + 0.5 * acceleration[i] * timeStep * timeStep;
    velocity[i] += acceleration[i] * timeStep;
  }
}

void dump(struct simdata data, int step) {
  printf("=============== STEP %d ===============\n", step);
  for (int i = 0; i < data.numParticles; i++) {
    printf("\t(%.04f, %.04f)\t(%.04f, %.04f)\n", GET_POSITION(data, i)[0],
           GET_POSITION(data, i)[1], GET_VELOCITY(data, i)[0],
           GET_VELOCITY(data, i)[1]);
  }
}

void runSimulation(struct simdata data, float timeStep, int steps) {
  float *accelerations =
      (float *)malloc(sizeof(float) * data.dim * data.numParticles);
  for (int step = 0; step < steps; step++) {
    for (int i = 0; i < data.numParticles; i++) {
      float *acceleration = &accelerations[i * data.dim];
      memset(acceleration, 0, data.dim * sizeof(float));

      float *position = GET_POSITION(data, i);
      float *features = GET_FEATURES(data, i);

      for (int j = 0; j < data.numParticles; j++) {
        if (i == j)
          continue;
        float *positionActor = GET_POSITION(data, j);
        float *featuresActor = GET_FEATURES(data, j);

        getForce(acceleration, position, features, positionActor, featuresActor,
                 data);
      }
      for (int j = 0; j < data.dim; j++) {
        acceleration[j] = acceleration[j] * G / features[0];
      }
    }

    for (int i = 0; i < data.numParticles; i++) {
      integrator(GET_POSITION(data, i), GET_VELOCITY(data, i),
                 accelerations + i * data.dim, timeStep, data);
    }
    dump(data, step);
  }
}

int main() {
  struct simdata data;
  data.dim = 2;
  data.numParticles = 2;
  data.dataDim = 1;
  data.data = (float *)malloc(sizeof(float) * 10);
  data.data[0] = 10;
  data.data[1] = 0;
  data.data[2] = 0;
  data.data[3] = 0;
  data.data[4] = 10;

  data.data[5] = -10;
  data.data[6] = 0;
  data.data[7] = 0;
  data.data[8] = 0;
  data.data[9] = 10;

  runSimulation(data, .1, 100);

  return 0;
}
