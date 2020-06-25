#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <vector>
#include <tuple>
#include <chrono>
#include <iostream>
#include <limits.h>

#include "mpi.h"

bool detectCollisions = false;
const float kNull = 0.0;

float GenerateNonNullRandomNumber()
{
  const int kNumberOfDecimals = std::pow(10, 5);
  float x = 0.0;
  while (kNull == x)
  {
    x = static_cast<float>(rand()%kNumberOfDecimals)/static_cast<float>(kNumberOfDecimals);
  }
  return x;
}

float GenerateTrajectoryNextStep()
{
  const int kMaxStepSize = 10001;
  const int kStepGranularity = std::pow(10, 5);
  float x = 0.0;
  while (kNull == x)
  {
    x = static_cast<float>(rand() % kMaxStepSize)/static_cast<float>(kStepGranularity);
  }

  return x;
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  int worldSize = 0;
  int worldRank = 0;
  float objectParameters[4];

  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

  float* objectsData = nullptr;
  float* readBuf = (float*)malloc(worldSize*4*sizeof(float));

  float* sphereSizes = new float[worldSize];
  const float kSphereSizeProportion = 1.0/32.0;
  for (int i = 0; i < worldSize; ++i)
  {
    sphereSizes[i] = GenerateNonNullRandomNumber() * kSphereSizeProportion; 
  }

  srand(time(NULL));
  auto collisionStart = std::chrono::high_resolution_clock::now();
  detectCollisions = true;

  if (!detectCollisions) return 0;
  
  if (worldRank == 0)
  {
    objectsData = new float[worldSize * 4];
    for (int i = 0; i < worldSize; ++i)
    {
      float x = GenerateNonNullRandomNumber();
      float y = GenerateNonNullRandomNumber();
      float z = GenerateNonNullRandomNumber();
      objectsData[i*4] = sphereSizes[i];
      objectsData[i*4 + 1] = x;
      objectsData[i*4 + 2] = y;
      objectsData[i*4 + 3] = z;
      }
  }
  float* objectData = new float[4];
  /**
     The Sphere size and location data will be scattered
     and will be received by each processor according to their rank.
     Each processor receives a chunk of data from the objectsData array.
     After this call each of the processors will have a sphere with its location and size.
     This first MPI call solves the problem of generating random number for the start locations
     of the spheres.
   */
  MPI_Scatter(objectsData, 4, MPI_FLOAT, objectData, 4, MPI_FLOAT, 0, MPI_COMM_WORLD);

  while(detectCollisions)
  {
    MPI_Allgather(objectData, 4, MPI_FLOAT, readBuf, 4, MPI_FLOAT, MPI_COMM_WORLD);
    
    float myRankX(0.0), myRankY(0.0), myRankZ(0.0);
    myRankX = objectData[1];
    myRankY = objectData[2];
    myRankZ = objectData[3];

    bool collisionDetected = false;
    for (int i = 0, rank = 0; i < worldSize*4; i+=4, ++rank)
    {
      if (rank != worldRank)
      {
	/**
	 * \desc This code will detect the collision using formula: d1 <= r1 + r2, where
	 * d1 is the euclidean distance between the origins of the two spheres and r1 + r2 
	 * is the sum of the sphere's radiuses.
	 */
	float euclideanDistance = std::sqrt((myRankX - readBuf[i + 1]) * (myRankX - readBuf[i + 1]) +
					    (myRankY - readBuf[i + 2]) * (myRankY - readBuf[i + 2]) +
					    (myRankZ - readBuf[i + 3]) * (myRankZ - readBuf[i + 3])
					    );
	float summedRadiuses = readBuf[i] + sphereSizes[worldRank];
	
	if (euclideanDistance <= summedRadiuses)
	{
	  auto pointOfCollision = std::chrono::high_resolution_clock::now();
	  auto collisionTimestamp = std::chrono::duration_cast<std::chrono::microseconds>(pointOfCollision - collisionStart).count();
	  printf("@%dus Sphere(%.5f, %.5f, %.5f, %.5f) %d colided with sphere(%.5f, %.5f, %.5f, %.5f) %d.\n",
		 collisionTimestamp,
		 readBuf[i], readBuf[i + 1], readBuf[i + 2], readBuf[i + 3],
		 rank,
		 sphereSizes[worldRank], myRankX, myRankY, myRankZ,
		 worldRank);
	  collisionDetected = true;
	}
      }	
    }
    if (collisionDetected)
    {
      objectData[1] = GenerateNonNullRandomNumber();
      objectData[2] = GenerateNonNullRandomNumber();
      objectData[3] = GenerateNonNullRandomNumber();      
    }
    else
    {
      objectData[1] += GenerateTrajectoryNextStep();
      objectData[2] += GenerateTrajectoryNextStep();
      objectData[3] += GenerateTrajectoryNextStep();
    }
  }

  if (nullptr != readBuf) free(readBuf);

  delete[] objectsData;

  MPI_Finalize();

  return 0;
}
