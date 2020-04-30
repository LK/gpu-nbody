#include "solarsystemdata.h"

const char* get_field(char* line, int num)
{
    const char* tok;
    const char* split = ", ";
    for (tok = strtok(line, split); tok && *tok; tok = strtok(NULL,split))
    {
        if (!num--)
            return tok;
    }
    return NULL;
}

float relative_error(float *estimate, float *real,int dim){
    float error_vec[dim];
    for(int i = 0; i < dim; i++) error_vec[i] = estimate[i] - real[i];

    float error_norm, real_norm;
    error_norm = real_norm = 0;
    for(int i = 0; i < dim; i++){
        error_norm += error_vec[i]*error_vec[i];
        real_norm += real[i]*real[i];
    }

    return sqrt(error_norm)/sqrt(real_norm);
}

int get_julian_dates(celestial_t planet,float** julian_dates){
    FILE* stream = fopen(get_planet_filename(planet),"r");
    *julian_dates = (float *)malloc(sizeof(float)*FILE_LINES);

    char line[512];
    int i = 0;
    float* temp = *julian_dates;
    while(fgets(line,512,stream)){
        const char *date_value = get_field(line,0);
        temp[i] = atof(date_value);
        i++;
    }

    return i;
}

char* get_planet_filename(celestial_t planet){

    switch(planet){
        case SUN: return "./data/sun.csv";
        case MERCURY: return "./data/mercury.csv";
        case VENUS: return "./data/venus.csv";
        case EARTH: return "./data/earth.csv";
        case MARS: return "./data/mars.csv";
        case JUPITER: return "./data/jupiter.csv";
        case SATURN: return "./data/saturn.csv";
        case URANUS: return "./data/uranus.csv";
        case NEPTUNE: return "./data/neptune.csv";
        case PLUTO: return "./data/pluto.csv";
    }
}