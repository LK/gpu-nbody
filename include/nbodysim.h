typedef struct simdata_t {
  float *data; // position, velocity, features
  unsigned int posdim;
  unsigned int featdim;
  unsigned int nparticles;
} simdata_t;

simdata_t *simdata_create(unsigned int posdim, unsigned int featdim,
                          unsigned int nparticles);
void simdata_free(simdata_t *sdata);

typedef enum integrator_t { INT_EULER = 0 } integrator_t;
typedef enum force_t { FORCE_NEWTONIAN = 0 } force_t;

void runSimulation(simdata_t *data, integrator_t int_type, force_t force_type,
                   float timeStep, int steps);
