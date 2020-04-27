typedef struct simdata_t {
  float *data; // position, velocity, features
  unsigned int posdim;
  unsigned int featdim;
  unsigned int nparticles;
} simdata_t;

simdata_t *simdata_create(unsigned int posdim, unsigned int featdim,
                          unsigned int nparticles);
void simdata_free(simdata_t *sdata);
float *simdata_pos_ptr(simdata_t *sdata, unsigned int idx);
float *simdata_vel_ptr(simdata_t *sdata, unsigned int idx);
float *simdata_feat_ptr(simdata_t *sdata, unsigned int idx);

typedef enum integrator_t { INT_EULER = 0 } integrator_t;
typedef enum force_t { FORCE_NEWTONIAN = 0 } force_t;

void run_simulation(simdata_t *data, integrator_t int_type, force_t force_type,
                    float time_step, int steps);
