typedef struct simdata_t {
  float *data; // position, velocity, features
  unsigned int dim;
  unsigned int numParticles;
  unsigned int dataDim;
} simdata_t;

typedef enum integrator_t { INT_EULER = 0 } integrator_t;
typedef enum force_t { FORCE_NEWTONIAN = 0 } force_t;

void runSimulation(simdata_t data, integrator_t int_type, force_t force_type,
                   float timeStep, int steps);
