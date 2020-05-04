#define PERPLEXITY 50
#define MIN_SD 0.00000001
#define MAX_SD 100000000

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
    float minSd = MIN_SD;
    float maxSD = MAX_SD;
    float currentPerp = getPerp(dataSize, i, (minSd + maxSD) / 2, l2NormDiffs);
    while (fabsf(PERPLEXITY - currentPerp) > 0.01)
    {
        if (currentPerp > PERPLEXITY)
        {
            maxSD = (minSd + maxSD) / 2;
        }
        else
        {
            minSd = (minSd + maxSD) / 2;
        }
        currentPerp = getPerp(dataSize, i, (minSd + maxSD) / 2, l2NormDiffs);
    }
    float sd = (maxSD + minSd) / 2;
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