#include <stdio.h>

#define G (6.673 * pow(10, -11))

#define GET_POSITION(sdata, i) (sdata.data + i * (sdata.dim * 2 + sdata.dataDim)) 
#define GET_VELOCITY(sdata, i) (sdata.data + i * (sdata.dim * 2 + sdata.dataDim) + sdata.dim) 
#define GET_FEATURES(sdata, i) (sdata.data + i * (sdata.dim * 2 + sdata.dataDim) + 2 * sdata.dim) 

void getForce(float * force, float * position, float * features, float * positionActor, float * featuresActor, struct simdata data)
{

    float feature = features[0];
    float featureActor = featuresActor[0];

    float distance = 0; 
    for(int i = 0; i < data.dim; i++)
    {
        float pos = position[i];
        float posActor = positionActor[i];
        float deltaPos = pos - posActor;
        distance += deltaPos*deltaPos;
    }
    distance = sqrt(distance);

    for(int i = 0; i < data.dim; i++)
    {
        float pos = position[i];
        float posActor = positionActor[i];
        float deltaPos = pos - posActor;

        float f = feature * featureActor / distance / distance * deltaPos / distance;
        force[i] += f;
    }
}

void runSimulation(struct simdata data, float timeStep, int steps)
{
    for(int i = 0; i < data.numParticles; i++)
    {
        float force[data.dim];
        memset(force, 0, data.dim * sizeof(float));

        float * position = GET_POSITION(data, i);
        float * features = GET_FEATURES(data, i);

        for(int j = 0; j < data.numParticles; j++)
        {
            if(i == j) continue;
            float * positionActor = GET_POSITION(data, j);
            float * featuresActor = GET_FEATURES(data, j);

            getForce(force, position, feature, positionActor, featuresActor, data);
        }
        for(int j = 0; j < data.dim; j++)
        {
            force[j] = force[j] * G;
        }
    }

    


}

int main ()
{
    
    return 0;
}