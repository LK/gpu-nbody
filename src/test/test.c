#include "nbodysim.h"
#include <stdlib.h>

int main() {
  simdata_t data;
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

  runSimulation(data, INT_EULER, FORCE_NEWTONIAN, .1, 100);

  return 0;
}
