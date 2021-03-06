#include "nbodysim.h"
#include "timing.h"
#include "tsneforces.c"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/// Compute Newtonian force between two particles.
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

/// Compute the first half of the Leapfrog integration routine.
void leapfrog_part1(float *accelerations, float timeStep, simdata_t *sdata) {
  float *position, *velocity, *acceleration;
  for (int i = 0; i < sdata->nparticles; i++) {
    position = simdata_pos_ptr(sdata, i);
    velocity = simdata_vel_ptr(sdata, i);
    acceleration = accelerations + i * sdata->posdim;
    for (int j = 0; j < sdata->posdim; j++) {
      velocity[j] += acceleration[j] * timeStep * 0.5;
      position[j] += timeStep * velocity[j];
    }
  }
}

/// Compute the second half of the Leapfrog integration routine.
void leapfrog_part2(float *accelerations, float timeStep, simdata_t *sdata) {
  float *velocity, *acceleration;
  for (int i = 0; i < sdata->nparticles; i++) {
    velocity = simdata_vel_ptr(sdata, i);
    acceleration = accelerations + i * sdata->posdim;
    for (int j = 0; j < sdata->posdim; j++) {
      velocity[j] += timeStep * acceleration[j] * 0.5;
    }
  }
}

/// Compute the Euler integration routine.
void euler_part2(float *accelerations, float timeStep, simdata_t *sdata) {
  float *position, *velocity, *acceleration;
  for (int j = 0; j < sdata->nparticles; j++) {
    position = simdata_pos_ptr(sdata, j);
    velocity = simdata_vel_ptr(sdata, j);
    acceleration = accelerations + j * sdata->posdim;
    for (int i = 0; i < sdata->posdim; i++) {
      position[i] +=
          velocity[i] * timeStep + 0.5 * acceleration[i] * timeStep * timeStep;
      velocity[i] += acceleration[i] * timeStep;
    }
  }
}

/// Debug logging.
void dump(simdata_t *sdata, int step) {
  printf("=============== STEP %d ===============\n", step);
  float *pos, *vel;
  for (int i = 0; i < sdata->nparticles; i++) {
    pos = simdata_pos_ptr(sdata, i);
    vel = simdata_vel_ptr(sdata, i);
    printf("\t(%.04f, %.04f,%.04f)\t(%.04f, %.04f,%.04f)\n", pos[0], pos[1],
           pos[2], vel[0], vel[1], vel[2]);
  }
}

/// Run n-body simulation.
void run_simulation(simdata_t *sdata, simconfig_t *sconfig,
                    integrator_t int_type, force_t force_type, float time_step,
                    int steps) {
  double force_calc_time = 0;
  float *accelerations =
      (float *)malloc(sizeof(float) * sdata->posdim * sdata->nparticles);
  memset(accelerations, 0, sdata->posdim * sdata->nparticles * sizeof(float));

  double integration_time = 0;
  measure_t *fulltimer = start_timer();

  measure_t *timer = start_timer();
  float *aux = NULL;
  // If enabled and supported, pre-compute some of the force computations.
  if (sconfig->precompute) {
    switch (force_type) {
    case FORCE_TSNE:
      aux = tsne_precompute(sdata, sdata->nparticles);
      break;
    case FORCE_NEWTONIAN:
      break;
    case FORCE_NEWTONIAN_SIMPLE:
      // aux = newtonian_precompute(d_sdata, sdata->nparticles);
      break;
    }
  }
  double precompute_time = end_timer_silent(timer);

  for (int step = 0; step < steps; step++) {
    timer = start_timer();
    switch (int_type) {
    case INT_LEAPFROG:
      leapfrog_part1(accelerations, time_step, sdata);
      break;
    default:
      break;
    }
    switch (force_type) {
    case FORCE_TSNE:
      setQjiDenominator(sdata, aux);
      break;
    default:
      break;
    }
    integration_time += end_timer_silent(timer);

    if (step > 0) {
      memset(accelerations, 0,
             sizeof(float) * sdata->posdim * sdata->nparticles);
    }
    timer = start_timer();
    for (int i = 0; i < sdata->nparticles; i++) {
      float *acceleration = &accelerations[i * sdata->posdim];

      float *position = simdata_pos_ptr(sdata, i);
      float *features = simdata_feat_ptr(sdata, i);

      for (int j = 0; j < sdata->nparticles; j++) {
        if (i == j)
          continue;
        float *positionActor = simdata_pos_ptr(sdata, j);
        float *featuresActor = simdata_feat_ptr(sdata, j);

        switch (force_type) {
        case FORCE_TSNE:
          getTsneForce(acceleration, position, i, positionActor, j, sdata, aux);
          break;
        case FORCE_NEWTONIAN:
        case FORCE_NEWTONIAN_SIMPLE:
          getForce(acceleration, position, features, positionActor,
                   featuresActor, sdata);
          break;
        }
      }
      for (int j = 0; j < sdata->posdim; j++) {
        switch (force_type) {
        case FORCE_TSNE:
          break;
        case FORCE_NEWTONIAN:
          acceleration[j] = acceleration[j] *
                            (6.673 * pow(10, -11) * 13.3153474) / features[0];
          break;
        case FORCE_NEWTONIAN_SIMPLE:
          acceleration[j] = acceleration[j] / features[0];
          break;
        }
      }
    }
    force_calc_time += end_timer_silent(timer);
    timer = start_timer();
    switch (int_type) {
    case INT_LEAPFROG:
      leapfrog_part2(accelerations, time_step, sdata);
      break;
    default:
      euler_part2(accelerations, time_step, sdata);
      break;
    }
    integration_time += end_timer_silent(timer);
  }
  double fulltime = end_timer_silent(fulltimer);
  printf("%f, %f, %f, %f\n", precompute_time, force_calc_time, integration_time,
         fulltime);

  if (aux)
    free(aux);

  free(accelerations);
}
