

struct simdata {
  float *data; // position, velocity, features
  unsigned int dim;
  unsigned int numParticles;
  unsigned int dataDim;
};

void runSimulation(struct simdata data, float timeStep, int steps);
