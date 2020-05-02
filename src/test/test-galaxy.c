#include "nbodysim.h"
#include <stdio.h>
#include <stdlib.h>

#define MAXSTR (256)
#define NUM_BODIES (2048)
#define TIME_STEP (0.01)
#define STEPS (10000)


//unsure if this should be here
float 	scaleFactor = 1.5f;		// 10.0f, 50
float 	velFactor = 8.0f;			// 15.0f, 100
float	massFactor = 120000.0f;	// 50000000.0,

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

void load_data(simdata_t *sdata) {
	int skip = 81920 / NUM_BODIES;

	FILE *galaxy_in;

	if((galaxy_in= fopen("./data/dubinski.tab", "r"))){
		char buf[MAXSTR];
    	float v[7];
    	float *position,*velocity,*features;
    	 
    	// total 81920 particles
    	// 16384 Gal. Disk
    	// 16384 And. Disk
    	// 8192  Gal. bulge
    	// 8192  And. bulge
    	// 16384 Gal. halo
    	// 16384 And. halo
    	int k=0;
    	for (int i=0; i< NUM_BODIES; i++,k++)
    	{
    		// depend on input size...
    		for (int j=0; j < skip; j++,k++)
    			fgets (buf, MAXSTR, galaxy_in);	// lead line
    		
    		sscanf(buf, "%f %f %f %f %f %f %f", v+0, v+1, v+2, v+3, v+4, v+5, v+6);
    		
    		
    		// position
    		position = simdata_pos_ptr(sdata, i);
    		velocity = simdata_vel_ptr(sdata, i);
    		features = simdata_feat_ptr(sdata, i);

    		position[0] = v[1] * scaleFactor;
    		position[1] = v[2] * scaleFactor;
    		position[2] = v[3] * scaleFactor;
    		
    		// mass
    		features[0] = v[0]*massFactor;
    		
    		// velocity
    		velocity[0] = v[4]*velFactor;
    		velocity[1] = v[5]*velFactor;
    		velocity[2] = v[6]*velFactor;
    		
    	}   
	}

	fclose(galaxy_in);
}

int main(int argc,char** argv) {
	simdata_t* sdata = simdata_create(3,1,NUM_BODIES);
	load_data(sdata);

	int steps = STEPS;
	if(argc > 1) steps = atoi(argv[1]);



	run_simulation(sdata, INT_LEAPFROG, FORCE_NEWTONIAN,MODE_GALAXY,TIME_STEP,steps);

	simdata_free(sdata);

  	return 0;
}

