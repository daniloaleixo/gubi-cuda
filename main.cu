#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>

__constant__ int WIDTH, HEIGHT;
int WIDTHH, HEIGHTH;

#define SIZE (WIDTH*HEIGHT)
#define SIZEH (WIDTHH*HEIGHTH)
#define FAKE_SIZE (SIZE + 2)
#define FAKE_SIZEH (SIZEH + 2)

#define PI acosf(-1)

__device__ void get_components(int x, int y, float theta, float *V, float *outx, float *outy) {
  *outx = V[y*FAKE_SIZE + x]*sinf(theta);
  *outy = V[y*FAKE_SIZE + x]*cosf(theta);
}

__device__ float get_theta(int x, int y, float *V) {
  return V[y*FAKE_SIZE + x]*2*PI;
}

__global__ void kernel(float *R, float *G, float *B, float *Rx, float *Ry, float *Bx, float *By) {
  printf("%d %d\n", WIDTH, HEIGHT);
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

  if (x % 17 == 0 && y % 17 == 0) {
    printf("[%d %d] = (%f, %f, %f)\nAng G: %f contribRx: %f contribRy: %f contribBx: %f contribBy: %f\n", x, y, R[y*FAKE_SIZE + x], G[y*FAKE_SIZE + x], B[y*FAKE_SIZE + x], theta, contribRx, contribRy, contribBx, contribBy);
    printf("[%d, %d] -> [%d, %d]\n", x, y, x + xx, y);
    printf("[%d, %d] -> [%d, %d]\n", x, y, x, y + yy);
  }

  float deltaRx = (1 - R[y*FAKE_SIZE + (x + xx)])*contribRx/4.0;
  float deltaRy = (1 - R[(y + yy)*FAKE_SIZE + x])*contribRy/4.0;

  float deltaBx = (1 - B[y*FAKE_SIZE + (x - xx)])*contribBx/4.0;
  float deltaBy = (1 - B[(y - yy)*FAKE_SIZE + x])*contribBy/4.0;

  if (xx != 0) {
    atomicAdd(&Rx[y*FAKE_SIZE + (x + xx)], deltaRx);
    atomicAdd(&Rx[y*FAKE_SIZE + x], -deltaRx);

    atomicAdd(&Bx[y*FAKE_SIZE + (x - xx)], deltaBx);
    atomicAdd(&Bx[y*FAKE_SIZE + x], -deltaBx);
  }
  if (yy != 0) {
    atomicAdd(&Ry[(y + yy)*FAKE_SIZE + x], deltaRy);
    atomicAdd(&Ry[y*FAKE_SIZE + x], -deltaRy);

    atomicAdd(&By[(y - yy)*FAKE_SIZE + x], deltaBy);
    atomicAdd(&By[y*FAKE_SIZE + x], -deltaBy);
  }
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

  Rx[y*FAKE_SIZE + x] = r;
  Bx[y*FAKE_SIZE + x] = b;
}

__global__ void redist(float *nR, float *nB, float *R, float *B) {
  int x = blockIdx.x + 1;
  int y = blockIdx.y + 1;

  if (R[y*FAKE_SIZE + x] > 1) {
    float tmp = 1 - R[y*FAKE_SIZE + x];
    nR[y*FAKE_SIZE + x] = 1;
    atomicAdd(&nR[y*FAKE_SIZE + x + 1], tmp/4);
    atomicAdd(&nR[y*FAKE_SIZE + x - 1], tmp/4);
    atomicAdd(&nR[y*FAKE_SIZE + x + FAKE_SIZE], tmp/4);
    atomicAdd(&nR[y*FAKE_SIZE + x - FAKE_SIZE], tmp/4);
  }
  if (B[y*FAKE_SIZE + x] > 1) {
    float tmp = 1 - B[y*FAKE_SIZE + x];
    nB[y*FAKE_SIZE + x] = 1;
    atomicAdd(&nB[y*FAKE_SIZE + x + 1], tmp/4);
    atomicAdd(&nB[y*FAKE_SIZE + x - 1], tmp/4);
    atomicAdd(&nB[y*FAKE_SIZE + x + FAKE_SIZE], tmp/4);
    atomicAdd(&nB[y*FAKE_SIZE + x - FAKE_SIZE], tmp/4);
  }
}

__global__ void re_redist_w(float *R, float *B) {
  int x = blockIdx.x + 1;

  atomicAdd(&R[x + FAKE_SIZE], R[x]);
  atomicAdd(&R[FAKE_SIZE * FAKE_SIZE - 1 - x - FAKE_SIZE], R[FAKE_SIZE * FAKE_SIZE - 1 - x]);
}

__global__ void re_redist_h(float *R, float *B) {
  int x = blockIdx.x + 1;

  atomicAdd(&R[x * FAKE_SIZE + 1], R[x * FAKE_SIZE]);
  atomicAdd(&R[x * FAKE_SIZE + FAKE_SIZE - 2], R[x * FAKE_SIZE + FAKE_SIZE - 1]);
}

__global__ void finalize(float *R, float *G, float *B) {
  int x = blockIdx.x + 1;
  int y = blockIdx.y + 1;

  if (R[y*FAKE_SIZE + x] > 1) R[y*FAKE_SIZE + x] = 1;
  if (B[y*FAKE_SIZE + x] > 1) B[y*FAKE_SIZE + x] = 1;

  atomicAdd(&G[y*FAKE_SIZE + x], atan2f(B[y*FAKE_SIZE + x], R[y*FAKE_SIZE + x])/(2*PI));

  if (G[y*FAKE_SIZE + x] > 1) G[y*FAKE_SIZE + x] = 0;
}

// the wrapper around the kernel call for main program to call.
extern "C" void kernel_wrapper(int num_procs, float *R, float *G, float *B, float *Rx, float *Ry, float *Bx, float *By) {
  dim3 image_size(WIDTHH, HEIGHTH);
  calc_components<<<image_size, 1>>>(R, G, B, Rx, Ry, Bx, By);
  assert(cudaGetLastError() == cudaSuccess);
  kernel<<<image_size, 1>>>(R, G, B, Rx, Ry, Bx, By);
  assert(cudaGetLastError() == cudaSuccess);
  recalc_magnitudes<<<image_size, 1>>>(Rx, Ry, Bx, By, R, B);
  assert(cudaGetLastError() == cudaSuccess);
  redist<<<image_size, 1>>>(R, B, Rx, Bx);
  assert(cudaGetLastError() == cudaSuccess);
  re_redist_w<<<WIDTHH, 1>>>(R, B);
  assert(cudaGetLastError() == cudaSuccess);
  re_redist_h<<<HEIGHTH, 1>>>(R, B);
  assert(cudaGetLastError() == cudaSuccess);
  finalize<<<image_size, 1>>>(R, G, B);
  assert(cudaGetLastError() == cudaSuccess);
  // copia valores da gpu pra cpu ??
}

void writePPM(const char* file, float *R, float *G, float *B) {
  FILE *fp = fopen(file, "w");
  if (fp == NULL) {
    fprintf(stderr, "Erro ao escrever %s\n", file);
    return;
  }
  fprintf(fp, "P3\n%d %d\n255\n", WIDTHH, HEIGHTH);
  for (int i = 1; i <= HEIGHTH; i++) {
    for (int j = 1; j <= WIDTHH; j++) {
      int idx = i*FAKE_SIZEH + j;
      fprintf(fp, "%d %d %d ", (int)(R[idx]*255), (int)(G[idx]*255), (int)(B[idx]*255));
      printf("[%d, %d] -> %f, %f, %f\n", i, j, R[idx], G[idx], B[idx]);
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
}

void readPPM(const char* file, float **R, float **G, float **B) {
  FILE *fp = fopen(file, "r");
  if (fp == NULL) {
    fprintf(stderr, "Erro ao abrir %s\n", file);
    return;
  }
  int w, h;
  int *gw, *gh;
  scanf("P3 %d %d 255 ", &w, &h);
  WIDTHH = w;
  HEIGHTH = h;
  cudaGetSymbolAddress((void**)&gw, WIDTH);
  cudaGetSymbolAddress((void**)&gh, HEIGHT);
  cudaMemcpy(&WIDTH, &w, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(&HEIGHT, &h, sizeof(int), cudaMemcpyHostToDevice);
  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      // TODO
    }
  }
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

  float *R, *G, *B, *gR, *gG, *gB, *nR, *nG, *nB, *nN;

  readPPM("test.ppm", &R, &G, &B);
  /*for (int i = 0; i < FAKE_SIZEH; i++) {
    for (int j = 0; j < FAKE_SIZEH; j++) {
      R[i*FAKE_SIZEH + j] = ((float)rand()/RAND_MAX);
      G[i*FAKE_SIZEH + j] = ((float)rand()/RAND_MAX);
      B[i*FAKE_SIZEH + j] = ((float)rand()/RAND_MAX);
      //printf("%f %f %f\n", R[i*FAKE_SIZEH + j], G[i*FAKE_SIZEH + j], B[i*FAKE_SIZEH + j]);
    }
  }*/

  writePPM("a.ppm", R, G, B);

  cudaMalloc((void**)&gR, FAKE_SIZEH*sizeof(float));
  cudaMalloc((void**)&gG, FAKE_SIZEH*sizeof(float));
  cudaMalloc((void**)&gB, FAKE_SIZEH*sizeof(float));
  cudaMalloc((void**)&nR, FAKE_SIZEH*sizeof(float));
  cudaMalloc((void**)&nG, FAKE_SIZEH*sizeof(float));
  cudaMalloc((void**)&nB, FAKE_SIZEH*sizeof(float));
  cudaMalloc((void**)&nN, FAKE_SIZEH*sizeof(float));

  cudaMemcpy(gR, R, FAKE_SIZEH*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gG, G, FAKE_SIZEH*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gB, B, FAKE_SIZEH*sizeof(float), cudaMemcpyHostToDevice);

  assert(cudaGetLastError() == cudaSuccess);
  printf("%s\n", cudaGetErrorString(cudaGetLastError()));

  clock_t start, stop;

  for(int i = 0; i < iters; i++) {
    printf("========= Iteracao %d\n", i+1);

    start = clock();
    // bagulhos aqui
    kernel_wrapper(num_procs, gR, gG, gB, nR, nG, nB, nN);
    cudaMemcpy(R, gR, FAKE_SIZEH*sizeof(float), cudaMemcpyDeviceToHost);
    assert(cudaGetLastError() == cudaSuccess);
    cudaMemcpy(G, gG, FAKE_SIZEH*sizeof(float), cudaMemcpyDeviceToHost);
    assert(cudaGetLastError() == cudaSuccess);
    cudaMemcpy(B, gB, FAKE_SIZEH*sizeof(float), cudaMemcpyDeviceToHost);
    assert(cudaGetLastError() == cudaSuccess);
    stop = clock();

  }

  writePPM("b.ppm", R, G, B);

  cudaDeviceSynchronize();

  return 0;
}
