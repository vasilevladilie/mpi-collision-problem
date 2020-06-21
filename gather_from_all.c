#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <vector>
#include <tuple>
#include <chrono>
#include <iostream>

#include "mpi.h"

bool detectCollisions = false;

float GenerateNonNullRandomNumber()
{
  const int kNumberOfDecimals = std::pow(10, 5);
  float x = 0.0;
  const float kNull = 0.0;
  while (kNull == x)
  {
    x = static_cast<float>(rand()%kNumberOfDecimals)/static_cast<float>(kNumberOfDecimals);
  }
  return x;
}

std::vector<std::tuple<int, int> > collisions;
bool IsPreviouslyFoundCollision(int rank, int worldRank)
{
  for (int i = 0; i < collisions.size(); ++i)
  {
    if (std::get<0>(collisions[i]) == rank && std::get<1>(collisions[i]) == worldRank)
      return true;
  }
  return false;
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
  while(detectCollisions)
  {
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
    MPI_Scatter(objectsData, 4, MPI_FLOAT, objectData, 4, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Allgather(objectData, 4, MPI_FLOAT, readBuf, 4, MPI_FLOAT, MPI_COMM_WORLD);

    float myRankX(0.0), myRankY(0.0), myRankZ(0.0);
    myRankX = objectData[1];
    myRankY = objectData[2];
    myRankZ = objectData[3];
    
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
	
	if (euclideanDistance <= summedRadiuses && !IsPreviouslyFoundCollision(rank, worldRank))
	{
	  auto pointOfCollision = std::chrono::high_resolution_clock::now();
	  auto collisionTimestamp = std::chrono::duration_cast<std::chrono::microseconds>(pointOfCollision - collisionStart).count();
	  printf("@%dus Sphere(%.5f, %.5f, %.5f, %.5f) %d colided with sphere(%.5f, %.5f, %.5f, %.5f) %d.\n",
		 collisionTimestamp,
		 readBuf[i], readBuf[i + 1], readBuf[i + 2], readBuf[i + 3],
		 rank,
		 sphereSizes[worldRank], myRankX, myRankY, myRankZ,
		 worldRank);
	  collisions.push_back(std::make_tuple(rank, worldRank));
	}
      }	
    }
  }

  if (nullptr != readBuf) free(readBuf);

  delete[] objectsData;

  MPI_Finalize();

  return 0;
}
