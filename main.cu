#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define SIZE 32

#define FAKE_SIZE (SIZE + 2)

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

__device__ void get_components(int x, int y, float theta, float *V, float *outx, float *outy) {
  *outx = V[y*FAKE_SIZE + x]*sinf(theta);
  *outy = V[y*FAKE_SIZE + x]*cosf(theta);
}

__device__ float get_theta(int x, int y, float *V) {
  return V[y*FAKE_SIZE + x]*2*PI;
}

__global__ void kernel(float *R, float *G, float *B, float *Rx, float *Ry, float *Bx, float *By) {
	int x = blockIdx.x + 1;
	int y = blockIdx.y + 1;

  float theta = get_theta(x, y, G);

  float contribRx, contribRy;
  get_components(x, y, theta, R, &contribRx, &contribRy);

  float contribBx, contribBy;
  get_components(x, y, theta, B, &contribBx, &contribBy);
  contribBx *= -1;
  contribBy *= -1;

  int xx = 0;
  int yy = 0;

  if (contribRx > 0) {
    xx = 1;
  } else {
    xx = -1;
  }
  if (contribRy > 0) {
    yy = 1;
  } else {
    yy = -1;
  }

    printf("[%d %d] = (%f, %f, %f)\nAng G: %f contribRx: %f contribRy: %f contribBx: %f contribBy: %f\n", x, y, R[y*FAKE_SIZE + x], G[y*FAKE_SIZE + x], B[y*FAKE_SIZE + x], theta, contribRx, contribRy, contribBx, contribBy);
    printf("[%d, %d] -> [%d, %d]\n", x, y, x + xx, y);
    printf("[%d, %d] -> [%d, %d]\n", x, y, x, y + yy);

  float deltaRx = (1 - R[y*FAKE_SIZE + (x + xx)])*contribRx/4.0;
  float deltaRy = (1 - R[(y + yy)*FAKE_SIZE + x])*contribRy/4.0;

  float deltaBx = (1 - B[y*FAKE_SIZE + (x - xx)])*contribBx/4.0;
  float deltaBy = (1 - B[(y - yy)*FAKE_SIZE + x])*contribBy/4.0;

  if (xx != 0) {
    atomicAdd(&Rx[y*FAKE_SIZE + (x + xx)], deltaRx);
    atomicSub(&Rx[y*FAKE_SIZE + x], deltaRx);

    atomicAdd(&Bx[y*FAKE_SIZE + (x - xx)], deltaBx);
    atomicSub(&Bx[y*FAKE_SIZE + x], deltaBx);
  }
  if (yy != 0) {
    atomicAdd(&Ry[(y + yy)*FAKE_SIZE + x], deltaRy);
    atomicSub(&Ry[y*FAKE_SIZE + x], deltaRy);

    atomicAdd(&By[(y - yy)*FAKE_SIZE + x], deltaBy);
    atomicSub(&By[y*FAKE_SIZE + x], deltaBy);
  }
	
  return;
}

__global__ void calc_components(float *R, float *G, float *B, float *Rx, float *Ry, float *Bx, float *By) {
	int x = blockIdx.x + 1;
	int y = blockIdx.y + 1;

  get_components(x, y, get_theta(x, y, R), R, &Rx[y*FAKE_SIZE + x], &Ry[y*FAKE_SIZE + x]);
  get_components(x, y, get_theta(x, y, B), B, &Bx[y*FAKE_SIZE + x], &By[y*FAKE_SIZE + x]);
  Bx[y*FAKE_SIZE + x] *= -1;
  By[y*FAKE_SIZE + x] *= -1;
}

__global__ void recalc_magnitudes(float *Rx, float *Ry, float *Bx, float *By, float *R, float *B) {
	int x = blockIdx.x + 1;
	int y = blockIdx.y + 1;

  float rx = Rx[y*FAKE_SIZE + x];
  float ry = Ry[y*FAKE_SIZE + x];
  float r = sqrtf((rx*rx)+(ry*ry));
  R[y*FAKE_SIZE + x] = r;

  float bx = Bx[y*FAKE_SIZE + x];
  float by = By[y*FAKE_SIZE + x];
  float b = sqrtf((bx*bx)+(by*by));
  B[y*FAKE_SIZE + x] = b;

  //atomicAdd(&G[y*FAKE_SIZE + x], atan2f(b, r));
}

// the wrapper around the kernel call for main program to call.
extern "C" void kernel_wrapper(int num_procs, float *R, float *G, float *B, float *Rx, float *Ry, float *Bx, float *By) {
  dim3 image_size(SIZE, SIZE);
  calc_components<<<image_size, 1>>>(R, G, B, Rx, Ry, Bx, By);
	kernel<<<image_size, 1>>>(R, G, B, Rx, Ry, Bx, By);
  recalc_magnitudes<<<image_size, 1>>>(Rx, Ry, Bx, By, R, B);
  //redistribuicao
  // re-redistribui das bordas pra dentro
  // corta > 1
  // calc G`
  // copia valores da gpu pra cpu ??
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

  float *R = (float*)malloc(FAKE_SIZE*FAKE_SIZE*sizeof(float));
  float *G = (float*)malloc(FAKE_SIZE*FAKE_SIZE*sizeof(float));
  float *B = (float*)malloc(FAKE_SIZE*FAKE_SIZE*sizeof(float));
  float *gR, *gG, *gB, *nR, *nG, *nB, *nN;

  for (int i = 0; i < FAKE_SIZE; i++) {
    for (int j = 0; j < FAKE_SIZE; j++) {
      R[i*FAKE_SIZE + j] = ((float)rand()/RAND_MAX);
      G[i*FAKE_SIZE + j] = ((float)rand()/RAND_MAX);
      B[i*FAKE_SIZE + j] = ((float)rand()/RAND_MAX);
    }
  }

  cudaMalloc((void**)&gR, FAKE_SIZE*FAKE_SIZE*sizeof(float));
  cudaMalloc((void**)&gG, FAKE_SIZE*FAKE_SIZE*sizeof(float));
  cudaMalloc((void**)&gB, FAKE_SIZE*FAKE_SIZE*sizeof(float));
  cudaMalloc((void**)&nR, FAKE_SIZE*FAKE_SIZE*sizeof(float));
  cudaMalloc((void**)&nG, FAKE_SIZE*FAKE_SIZE*sizeof(float));
  cudaMalloc((void**)&nB, FAKE_SIZE*FAKE_SIZE*sizeof(float));
  cudaMalloc((void**)&nN, FAKE_SIZE*FAKE_SIZE*sizeof(float));

  cudaMemcpy(gR, R, FAKE_SIZE*FAKE_SIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gG, G, FAKE_SIZE*FAKE_SIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gB, B, FAKE_SIZE*FAKE_SIZE*sizeof(float), cudaMemcpyHostToDevice);
    
	assert(cudaGetLastError() == cudaSuccess);
	printf("%s\n", cudaGetErrorString(cudaGetLastError()));
	
	clock_t start, stop;
	
	for(int i = 0; i < iters; i++) {
		printf("========= Iteracao %d\n", i+1);
		
		start = clock();
    // bagulhos aqui
    kernel_wrapper(num_procs, gR, gG, gB, nR, nG, nB, nN);
		stop = clock();

    assert(cudaGetLastError() == cudaSuccess);
	}

  cudaDeviceSynchronize();
	
	return 0;
}
