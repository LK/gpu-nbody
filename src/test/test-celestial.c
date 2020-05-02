#include "nbodysim.h"
#include "solarsystemdata.h"
#include <stdlib.h>

#define SOLAR_DELTA_T (0.01)

void solar_system_test(celestial_t test_planet);

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

int main() {
  solar_system_test(SATURN);
  return 0;
}

void solar_system_test(celestial_t test_planet) {
  simdata_t *sdata = simdata_create(3, 1, 10);

  for (int i = 0; i < 10; i++) {
    load_index_vectors(0, sdata->data + i * 7, sdata->data + i * 7 + 3,
                       (celestial_t)i);
    sdata->data[i * 7 + 6] = get_planet_mass((celestial_t)i);
  }

  float *julian_dates;
  int num_dates = get_julian_dates(test_planet, &julian_dates);

  float julian_date_offset;
  float real_position[3], real_velocity[3];

  for (int i = 1; i < num_dates; i++) {
    julian_date_offset = julian_dates[i - 1];
    run_simulation(
        sdata, INT_LEAPFROG, FORCE_NEWTONIAN, MODE_CELESTIAL, SOLAR_DELTA_T,
        (int)((julian_dates[i] - julian_date_offset) * (1.0 / SOLAR_DELTA_T)));
    load_index_vectors(i, real_position, real_velocity, test_planet);

    printf("JULIAN DATE %f - POSITION ERROR = %f --- VELOCITY ERROR = %f\n",
           julian_dates[i],
           relative_error(sdata->data + test_planet * 7, real_position, 3),
           relative_error(sdata->data + test_planet * 7 + 3, real_velocity, 3));
  }
  free(julian_dates);
  simdata_free(sdata);
}
