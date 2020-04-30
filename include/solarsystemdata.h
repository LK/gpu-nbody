#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define FILE_LINES (500)

const char* get_field(char* line, int num);

typedef enum celestial_t { SUN, MERCURY, VENUS, EARTH, MARS, JUPITER, SATURN, URANUS, NEPTUNE, PLUTO } celestial_t;

char* get_planet_filename(celestial_t planet);

int get_julian_dates(celestial_t planet,float** julian_dates);

float relative_error(float *estimate, float *real,int dim);