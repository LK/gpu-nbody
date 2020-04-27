#include "nbodysim.h"
#include <stdlib.h>

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

  runSimulation(sdata, INT_EULER, FORCE_NEWTONIAN, .1, 100);

  return 0;
}
