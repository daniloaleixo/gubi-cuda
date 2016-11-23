#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>

typedef struct {
  float ***data;
  int width, height;
} Imagem;

Imagem* new_image(int w, int h) {
  Imagem *res = (Imagem*)malloc(3*w*h*sizeof(float));
  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      for (int k = 0; k < 3; k++) {
        res->data[i][j][k] = 0;
      }
    }
  }
  res->width = w;
  res->height = h;
  return res;
}

#define PI acosf(-1)

__global__ void kernel(float *R, float *G, float *B, float *nR, float *nG, float *nB) {
	int x = blockIdx.x;
	int y = blockIdx.y;

  float theta = G[y*32 + x]*2*PI;

  float Rx = R[y*32 + x]*sinf(theta);
  float Ry = R[y*32 + x]*cosf(theta);

  float Bx = -B[y*32 + x]*sinf(theta);
  float By = -B[y*32 + x]*cosf(theta);

  if (x % 17 == 0) {
    printf("[%d %d] = (%f, %f, %f)\nAng G: %f Rx: %f Ry: %f Bx: %f By: %f\n", x, y, R[y*32 + x], G[y*32 + x], B[y*32 + x], theta, Rx, Ry, Bx, By);
  }

  int xx = 0;
  int yy = 0;

  if (Rx > 0) {
    if (x < 32 - 1) {
      xx = 1;
    }
  } else {
    if (x > 0) {
      xx = -1;
    }
  }
  if (Ry > 0) {
    if (y < 32 - 1) {
      yy = 1;
    }
  } else {
    if (y > 0) {
      yy = -1;
    }
  }
	
  return;
}

// the wrapper around the kernel call for main program to call.
extern "C" void kernel_wrapper(int num_procs, float *R, float *G, float *B, float *nR, float *nG, float *nB) {
  dim3 image_size(32, 32);
	kernel<<<image_size, 1>>>(R, G, B, nR, nG, nB);
}


int main(int argc, char const *argv[]) {	
  if (argc != 5) {
    printf("Uso:\n");
    printf("%s entrada saida num_iters num_procs", argv[0]);
    return 0;
  }

  int iters = atoi(argv[3]);
  int num_procs = atoi(argv[4]);

  srand(time(NULL));

  float *R = (float*)malloc(32*32*sizeof(float));
  float *G = (float*)malloc(32*32*sizeof(float));
  float *B = (float*)malloc(32*32*sizeof(float));
  float *gR, *gG, *gB, *nR, *nG, *nB;

  for (int i = 0; i < 32; i++) {
    for (int j = 0; j < 32; j++) {
      R[i*32 + j] = ((float)rand()/RAND_MAX);
      G[i*32 + j] = ((float)rand()/RAND_MAX);
      B[i*32 + j] = ((float)rand()/RAND_MAX);
    }
  }

  cudaMalloc((void**)&gR, 32*32*sizeof(float));
  cudaMalloc((void**)&gG, 32*32*sizeof(float));
  cudaMalloc((void**)&gB, 32*32*sizeof(float));
  cudaMalloc((void**)&nR, 32*32*sizeof(float));
  cudaMalloc((void**)&nG, 32*32*sizeof(float));
  cudaMalloc((void**)&nB, 32*32*sizeof(float));

  cudaMemcpy(gR, R, 32*32*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gG, G, 32*32*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gB, B, 32*32*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(nR, R, 32*32*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(nG, G, 32*32*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(nB, B, 32*32*sizeof(float), cudaMemcpyHostToDevice);
    
	assert(cudaGetLastError() == cudaSuccess);
	printf("%s\n", cudaGetErrorString(cudaGetLastError()));
	
	clock_t start, stop;
	
	for(int i = 0; i < iters; i++) {
		printf("========= Iteracao %d\n", i+1);
		
		start = clock();
    // bagulhos aqui
    kernel_wrapper(num_procs, gR, gG, gB, nR, nG, nB);
		stop = clock();

    cudaMemcpy(R, gR, 32*32*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(G, gG, 32*32*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(B, gB, 32*32*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(nR, R, 32*32*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(nG, G, 32*32*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(nB, B, 32*32*sizeof(float), cudaMemcpyHostToDevice);
    assert(cudaGetLastError() == cudaSuccess);
	}

  cudaDeviceSynchronize();
	
	return 0;
}
