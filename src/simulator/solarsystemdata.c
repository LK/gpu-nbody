#include "solarsystemdata.h"

const char* get_field(char* line, int num)
{
    const char* tok;
    for (tok = strtok(line, ","); tok && *tok; tok = strtok(NULL,",\n"))
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
    fclose(stream);
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

float get_planet_mass(celestial_t planet){
    switch(planet){
        case SUN: return 332946;
        case MERCURY: return 0.0553;
        case VENUS: return 0.815;
        case EARTH: return 1;
        case MARS: return 0.107;
        case JUPITER: return 317.8;
        case SATURN: return 95.2;
        case URANUS: return 14.5;
        case NEPTUNE: return 17.1;
        case PLUTO: return 0.0025;
    }
}

void load_index_vectors(int index,float* position_dest,float* velocity_dest,celestial_t planet) {
    FILE* stream = fopen(get_planet_filename(planet),"r");
    char line[512];
    for(int i = 0; i <= index;i++) fgets(line,512,stream);
    for(int i = 0; i < 3; i++){
        char* tmp = strdup(line);
        char* tmp2 = strdup(line);
        const char* position = get_field(tmp,3 + i);
        const char* velocity = get_field(tmp2,6 + i);
        position_dest[i] = atof(position);
        velocity_dest[i] = atof(velocity);
        free(tmp);
        free(tmp2);
    }
    fclose(stream);
}
