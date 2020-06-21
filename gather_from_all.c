#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <math.h>

bool continueGatheringPossibleCollisions = false;

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  int worldSize = 0;
  int worldRank = 0;
  float objectParameters[4];

  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

  continueGatheringPossibleCollisions = true;
  const int kNumberOfDecimals = std::pow(10, 5);
  float* objectsData = nullptr;
  float* readBuf = (float*)malloc(worldSize*4*sizeof(float));
 
  while(continueGatheringPossibleCollisions)
  {
    if (worldRank == 0)
    {
      objectsData = new float[worldSize * 4];
      srand(time(NULL));
      float* kSphereRadiuses = new float[worldSize];
      for (int i = 0; i < worldSize; ++i)
      {
	kSphereRadiuses[i] = static_cast<float>(i + 1);

	printf("%d Sphere radius %.5f\n", i, kSphereRadiuses[i]);
      }

      for (int i = 0; i < worldSize; ++i)
      {
	float x = 0;
	while(x == 0)
	{
	  x = static_cast<float>(rand()%kNumberOfDecimals)/static_cast<float>(kNumberOfDecimals);
        }
	// usleep(100);
	float y = 0;
	while(y == 0)
	{
	  y = static_cast<float>(rand()%kNumberOfDecimals)/static_cast<float>(kNumberOfDecimals);
	}
	// usleep(100);
	float z = 0;
	while(z == 0)
	{
	  z = static_cast<float>(rand()%kNumberOfDecimals)/static_cast<float>(kNumberOfDecimals);
	}
	// usleep(100);
    
	objectsData[i*4] = kSphereRadiuses[i];
	printf("objectsData[%d] = %.5f\n", i, kSphereRadiuses[i]);
	
	objectsData[i*4 + 1] = x;
	printf("objectsData[%d] = %.5f\n", i + 1, kSphereRadiuses[i + 1]);

	objectsData[i*4 + 2] = y;
	printf("objectsData[%d] = %.5f\n", i + 2, kSphereRadiuses[i + 2]);

	objectsData[i*4 + 3] = z;
	printf("objectsData[%d] = %.5f\n", i + 3, kSphereRadiuses[i + 3]);
      }
    
      printf("#######################################\n");
      printf("Scatter Data\n");
      for (int i = 0; i < worldSize; ++i)
	{
	  printf("\tSphere radius: %.5f\n", objectsData[i*4]);
	  printf("\tx: %.5f\n", objectsData[i*4 + 1]);
	  printf("\ty: %.5f\n", objectsData[i*4 + 2]);
	  printf("\tz: %.5f\n", objectsData[i*4 + 3]);
	}
      printf("#######################################\n");
    }
    float* objectData = new float[4];
    MPI_Scatter(objectsData, 4, MPI_FLOAT, objectData, 4, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Allgather(objectData, 4, MPI_FLOAT, readBuf, 4, MPI_FLOAT, MPI_COMM_WORLD);

    for (int i = 0; i < worldSize*4; i+=4)
    {
      printf("Rank %d: Sphere size %.5f, Sphere coordinates( %.5f, %.5f, %.5f )", worldRank, readBuf[i], readBuf[i+1], readBuf[i+2], readBuf[i+3]);
	printf("\n");
    }
    sleep(2);
  }

  if (nullptr != readBuf) free(readBuf);

  delete[] objectsData;

  MPI_Finalize();

  return 0;
}
