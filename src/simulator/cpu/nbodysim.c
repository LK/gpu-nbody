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

float getl2NormDiff(float *a, float *b, int dim)
{
    float norm = 0;
    for (int i = 0; i < dim; i++)
    {
        float diff = a[i] - b[i];
        norm += diff * diff;
    }
    return norm;
}
void setQjiDenominator(simdata_t *sdata, float *aux)
{
    float denominator = 0;
    for (int k = 0; k < sdata->nparticles; k++)
    {
        for (int l = 0; l < sdata->nparticles; l++)
        {
            if (l == k)
                continue;
            denominator += 1 / (1 + getl2NormDiff(simdata_pos_ptr(sdata, l), simdata_pos_ptr(sdata, k), sdata->posdim));
        }
    }
    aux[sdata->nparticles * sdata->nparticles] = denominator;
}
float getPji(int dataSize, int i, int j, float sd, float *l2NormDiffs)
{
    float denominator = 0;
    for (int k = 0; k < dataSize; k++)
    {
        if (k == i)
            continue;
        denominator += exp(-l2NormDiffs[i * dataSize + k] / 2 / sd / sd);
    }
    float numerator = exp(-l2NormDiffs[i * dataSize + j] / 2 / sd / sd);
    return numerator / denominator;
}
float getPerp(int dataSize, int i, float sd, float *l2NormDiffs)
{
    float shEntr = 0;
    for (int j = 0; j < dataSize; j++)
    {
        float pji = getPji(dataSize, i, j, sd, l2NormDiffs);
        shEntr -= (pji * log2(pji));
    }
    return pow(2.0, shEntr);
}
float findSd(int dataSize, int i, float *l2NormDiffs)
{
    float MIN_SD = 0.00000001;
    float MAX_SD = 100000000;
    float PERPLEXITY = 50;

    float currentPerp = getPerp(dataSize, i, (MIN_SD + MAX_SD) / 2, l2NormDiffs);
    while (fabsf(PERPLEXITY - currentPerp) > 0.01)
    {
        if (currentPerp > PERPLEXITY)
        {
            MAX_SD = (MIN_SD + MAX_SD) / 2;
        }
        else
        {
            MIN_SD = (MIN_SD + MAX_SD) / 2;
        }
        currentPerp = getPerp(dataSize, i, (MIN_SD + MAX_SD) / 2, l2NormDiffs);
    }
    float sd = (MAX_SD + MIN_SD) / 2;
    return sd;
}
float* tsne_precompute(simdata_t *sdata, int dataSize)
{
  float *l2NormDiffs = malloc(sizeof(float) * dataSize * dataSize);
  for (int i = 0; i < dataSize; i++)
  {
      for (int j = 0; j < dataSize; j++)
      {
          l2NormDiffs[i * dataSize + j] = getl2NormDiff(simdata_feat_ptr(sdata, i), simdata_feat_ptr(sdata, j), sdata->featdim);
      }
  }
  
  float *stdevs = malloc(sizeof(float) * dataSize);
  for (int i = 0; i < dataSize; i++)
  {
      stdevs[i] = findSd(dataSize, i, l2NormDiffs);
  }

  float *aux = malloc(sizeof(float) * (dataSize * dataSize + 1));

  for (int j = 0; j < dataSize; j++)
  {
      for (int i = 0; i < dataSize; i++)
      {
          aux[j * dataSize + i] = getPji(dataSize, i, j, stdevs[i], l2NormDiffs);
      }
  }

  free(l2NormDiffs);
  free(stdevs);
  return aux;
}
void getTsneForce(float *force, float *position, int dataIndex,
              float *positionActor, int actorDataIndex, simdata_t *sdata, float * aux) {

  int dataSize = sdata->nparticles;
  float pji = aux[actorDataIndex * dataSize + dataIndex];
  float pij = aux[dataIndex * dataSize + actorDataIndex];
  float qjiNumerator = 1 / (1 + getl2NormDiff(position, positionActor, sdata->posdim));
  float qjiDenominator = aux[sdata->nparticles * sdata->nparticles];

  float scalar = 4 * ((pij + pji) / 2 / dataSize - qjiNumerator / qjiDenominator) / (1 + getl2NormDiff(position, positionActor, sdata->posdim));

  for (int i = 0; i < sdata->posdim; i++) {  
      force[i] += scalar * (positionActor[i] - position[i]);
  }
}
void run_simulation(simdata_t *sdata, simconfig_t *sconfig,
                    integrator_t int_type, force_t force_type, float time_step,
                    int steps) {
  float *accelerations =
      (float *)malloc(sizeof(float) * sdata->posdim * sdata->nparticles);
  memset(accelerations, 0, sdata->posdim * sdata->nparticles * sizeof(float));

  float *aux = NULL;
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
  
  for (int step = 0; step < steps; step++) {

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

    if (step > 0) {
      memset(accelerations, 0,
             sizeof(float) * sdata->posdim * sdata->nparticles);
    }

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
          getTsneForce(acceleration, position, i, positionActor, j,
                 sdata, aux);
          break;
        case FORCE_NEWTONIAN:
          getForce(acceleration, position, features, positionActor, featuresActor,
                 sdata);
          break;
        case FORCE_NEWTONIAN_SIMPLE:
          getForce(acceleration, position, features, positionActor, featuresActor,
                 sdata);
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

    switch (int_type) {
    case INT_LEAPFROG:
      leapfrog_part2(accelerations, time_step, sdata);
      break;
    default:
      euler_part2(accelerations, time_step, sdata);
      break;
    }
  }
  free(accelerations);
}
