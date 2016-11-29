/*
 *  MAC0431 - Introducao a Programacao Paralela e Distribuida
 *
 * Fisica Alternativa
 *
 * Bruno Endo       - 7990982
 * Danilo Aleixo    - 7972370
 * Gustavo Caparica - 7991020
 *
 */
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>

#include <cuda.h>
#include <cuda_runtime.h>

__constant__ int WIDTH, HEIGHT;
int WIDTHH, HEIGHTH;

#define SIZE (WIDTH*HEIGHT)
#define SIZEH (WIDTHH*HEIGHTH)
#define BORDERW (WIDTH+2)
#define BORDERH (HEIGHT+2)
#define FAKE_SIZE (BORDERW*BORDERH)

#define FAKE_SIZEH ((WIDTHH+2)*(HEIGHTH+2))
#define MAXLINE 128

#define PI acosf(-1)

__device__ void get_components(int x, int y, float theta, float *V, float *outx, float *outy) {
  *outx = V[y*BORDERW + x]*sinf(theta);
  *outy = V[y*BORDERW + x]*cosf(theta);
}

__device__ float get_theta(int x, int y, float *V) {
  return V[y*BORDERW + x]*2*PI;
}

__global__ void calc_contributions(float *R, float *G, float *B, float *Rx, float *Ry, float *Bx, float *By) {
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

  float deltaRx = (1 - R[y*BORDERW + (x + xx)])*contribRx/4.0;
  float deltaRy = (1 - R[(y + yy)*BORDERW + x])*contribRy/4.0;

  float deltaBx = (1 - B[y*BORDERW + (x - xx)])*contribBx/4.0;
  float deltaBy = (1 - B[(y - yy)*BORDERW + x])*contribBy/4.0;

  if (xx != 0) {
    atomicAdd(&Rx[y*BORDERW + (x + xx)], deltaRx);
    atomicAdd(&Rx[y*BORDERW + x], -deltaRx);

    atomicAdd(&Bx[y*BORDERW + (x - xx)], deltaBx);
    atomicAdd(&Bx[y*BORDERW + x], -deltaBx);
  }
  if (yy != 0) {
    atomicAdd(&Ry[(y + yy)*BORDERW + x], deltaRy);
    atomicAdd(&Ry[y*BORDERW + x], -deltaRy);

    atomicAdd(&By[(y - yy)*BORDERW + x], deltaBy);
    atomicAdd(&By[y*BORDERW + x], -deltaBy);
  }
}

__global__ void calc_components(float *R, float *G, float *B, float *Rx, float *Ry, float *Bx, float *By) {
  int x = blockIdx.x + 1;
  int y = blockIdx.y + 1;

  get_components(x, y, get_theta(x, y, R), R, &Rx[y*BORDERW + x], &Ry[y*BORDERW + x]);
  get_components(x, y, get_theta(x, y, B), B, &Bx[y*BORDERW + x], &By[y*BORDERW + x]);
  Bx[y*BORDERW + x] *= -1;
  By[y*BORDERW + x] *= -1;
}

__global__ void recalc_magnitudes(float *Rx, float *Ry, float *Bx, float *By, float *R, float *B) {
  int x = blockIdx.x + 1;
  int y = blockIdx.y + 1;

  float rx = Rx[y*BORDERW + x];
  float ry = Ry[y*BORDERW + x];
  float r = sqrtf((rx*rx)+(ry*ry));
  R[y*BORDERW + x] = r;

  float bx = Bx[y*BORDERW + x];
  float by = By[y*BORDERW + x];
  float b = sqrtf((bx*bx)+(by*by));
  B[y*BORDERW + x] = b;

  Rx[y*BORDERW + x] = r;
  Bx[y*BORDERW + x] = b;
}

__global__ void redist(float *nR, float *nB, float *R, float *B) {
  int x = blockIdx.x + 1;
  int y = blockIdx.y + 1;

  if (R[y*BORDERW + x] > 1) {
    float tmp = 1 - R[y*BORDERW + x];
    nR[y*BORDERW + x] = 1;
    atomicAdd(&nR[y*BORDERW + x + 1], tmp/4);
    atomicAdd(&nR[y*BORDERW + x - 1], tmp/4);
    atomicAdd(&nR[(y+1)*BORDERW + x], tmp/4);
    atomicAdd(&nR[(y-1)*BORDERW + x], tmp/4);
  }
  if (B[y*BORDERW + x] > 1) {
    float tmp = 1 - B[y*BORDERW + x];
    nB[y*BORDERW + x] = 1;
    atomicAdd(&nB[y*BORDERW + x + 1], tmp/4);
    atomicAdd(&nB[y*BORDERW + x - 1], tmp/4);
    atomicAdd(&nB[(y+1)*BORDERW + x], tmp/4);
    atomicAdd(&nB[(y-1)*BORDERW + x], tmp/4);
  }
}

__global__ void re_redist_w(float *R, float *B) {
  int x = blockIdx.x + 1;

  atomicAdd(&R[x + BORDERW], R[x]);
  atomicAdd(&R[FAKE_SIZE - 1 - x - BORDERW], R[FAKE_SIZE - 1 - x]);

  atomicAdd(&B[x + BORDERW], B[x]);
  atomicAdd(&B[FAKE_SIZE - 1 - x - BORDERW], B[FAKE_SIZE - 1 - x]);

  R[x] = 0;
  R[FAKE_SIZE - 1 - x] = 0;

  B[x] = 0;
  B[FAKE_SIZE - 1 - x] = 0;
}

__global__ void re_redist_h(float *R, float *B) {
  int x = blockIdx.x + 1;

  atomicAdd(&R[x * BORDERW + 1], R[x * BORDERW]);
  atomicAdd(&R[x * BORDERW + BORDERW - 2], R[x * BORDERW + BORDERW - 1]);

  atomicAdd(&B[x * BORDERW + 1], B[x * BORDERW]);
  atomicAdd(&B[x * BORDERW + BORDERW - 2], B[x * BORDERW + BORDERW - 1]);

  R[x * BORDERW] = 0;
  R[x * BORDERW + BORDERW - 1] = 0;

  B[x * BORDERW] = 0;
  B[x * BORDERW + BORDERW - 1] = 0;
}

__global__ void finalize(float *R, float *G, float *B) {
  int x = blockIdx.x + 1;
  int y = blockIdx.y + 1;

  if (R[y*BORDERW + x] > 1) R[y*BORDERW + x] = 1;
  if (B[y*BORDERW + x] > 1) B[y*BORDERW + x] = 1;

  if (R[y*BORDERW + x] < 0) R[y*BORDERW + x] = 0;
  if (B[y*BORDERW + x] < 0) B[y*BORDERW + x] = 0;

  atomicAdd(&G[y*BORDERW + x], atan2f(B[y*BORDERW + x], R[y*BORDERW + x])/(2*PI));

  if (G[y*BORDERW + x] > 1) G[y*BORDERW + x] = 0;
  if (G[y*BORDERW + x] < 0) G[y*BORDERW + x] = 0;
}

// the wrapper around the kernel call for main program to call.
extern "C" void kernel_wrapper(float *R, float *G, float *B, float *Rx, float *Ry, float *Bx, float *By) {
  dim3 image_size(WIDTHH, HEIGHTH);

  calc_components<<<image_size, 1>>>(R, G, B, Rx, Ry, Bx, By);
  assert(cudaGetLastError() == cudaSuccess);

  calc_contributions<<<image_size, 1>>>(R, G, B, Rx, Ry, Bx, By);
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
      int idx = i*(WIDTHH+2) + j;
      fprintf(fp, "%d %d %d ", (int)(R[idx]*255), (int)(G[idx]*255), (int)(B[idx]*255));
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
}

int readPPM(const char* file, float **R, float **G, float **B) {
  FILE *fp = fopen(file, "rb");
  if (fp == NULL) {
    fprintf(stderr, "Erro ao abrir %s\n", file);
    return 0;
  }
  int w, h, k;
  int red, green, blue;
  int *gw, *gh;
  char c;
  char tmp[MAXLINE];
	c = getc(fp);
	if (c == 'P' || c == 'p') {
		c = getc(fp);
  }

	if (c != '3') {
    fprintf(stderr, "Erro: formato do PPM (P%c) nao suportado\n", c);
    return 0;
  }

  c = getc(fp);
  if (c == '\n' || c == '\r') {
    c = getc(fp);
    while(c == '#') {
      fscanf(fp, "%[^\n\r] ", tmp);
      c = getc(fp);
    }
    ungetc(c,fp); 
  }

  fscanf(fp, "%d %d %d", &w, &h, &k);

  WIDTHH = w;
  HEIGHTH = h;

  cudaGetSymbolAddress((void**)&gw, WIDTH);
  assert(cudaGetLastError() == cudaSuccess);
  cudaGetSymbolAddress((void**)&gh, HEIGHT);
  assert(cudaGetLastError() == cudaSuccess);
  cudaMemcpy(gw, &w, sizeof(int), cudaMemcpyHostToDevice);
  assert(cudaGetLastError() == cudaSuccess);
  cudaMemcpy(gh, &h, sizeof(int), cudaMemcpyHostToDevice);
  assert(cudaGetLastError() == cudaSuccess);

  *R = (float*)malloc(FAKE_SIZEH*sizeof(float));
  *G = (float*)malloc(FAKE_SIZEH*sizeof(float));
  *B = (float*)malloc(FAKE_SIZEH*sizeof(float));

  for (int i = 1; i <= h; i++) {
    for (int j = 1; j <= w; j++) {
      fscanf(fp, "%d %d %d", &red, &green, &blue );
      (*R)[i*(w+2) + j] = (float)red/(float)k;
      (*G)[i*(w+2) + j] = (float)green/(float)k;
      (*B)[i*(w+2) + j] = (float)blue/(float)k;
    }
  }

  for (int i = 0; i < w + 2; i++) {
    (*R)[i] = (*G)[i] = (*B)[i] = 0;
    (*R)[(h+1)*(w+2) + i] = (*G)[(h+1)*(w+2) + i] = (*B)[(h+1)*(w+2) + i] = 0;
  }

  for (int i = 0; i < h + 2; i++) {
    (*R)[i*(w+2)] = (*G)[i*(w+2)] = (*B)[i*(w+2)] = 0;
    (*R)[i*(w+2) + w + 1] = (*G)[i*(w+2) + w + 1] = (*B)[i*(w+2) + w + 1] = 0;
  }

  return 1;
}

int main(int argc, char const *argv[]) {	
  if (argc != 5) {
    printf("Uso:\n");
    printf("%s entrada saida num_iters num_procs\n", argv[0]);
    printf("Nota: num_procs é ignorado pois usamos CUDA\n");
    return 0;
  }

  int iters = atoi(argv[3]);

  srand(time(NULL));

  float *R, *G, *B, *gR, *gG, *gB, *nR, *nG, *nB, *nN;

  if (!readPPM(argv[1], &R, &G, &B)) {
    fprintf(stderr, "Erro durante a leitura\n");
    return 1;
  }

  cudaMalloc((void**)&gR, FAKE_SIZEH*sizeof(float));
  assert(cudaGetLastError() == cudaSuccess);
  cudaMalloc((void**)&gG, FAKE_SIZEH*sizeof(float));
  assert(cudaGetLastError() == cudaSuccess);
  cudaMalloc((void**)&gB, FAKE_SIZEH*sizeof(float));
  assert(cudaGetLastError() == cudaSuccess);
  cudaMalloc((void**)&nR, FAKE_SIZEH*sizeof(float));
  assert(cudaGetLastError() == cudaSuccess);
  cudaMalloc((void**)&nG, FAKE_SIZEH*sizeof(float));
  assert(cudaGetLastError() == cudaSuccess);
  cudaMalloc((void**)&nB, FAKE_SIZEH*sizeof(float));
  assert(cudaGetLastError() == cudaSuccess);
  cudaMalloc((void**)&nN, FAKE_SIZEH*sizeof(float));
  assert(cudaGetLastError() == cudaSuccess);

  cudaMemcpy(gR, R, FAKE_SIZEH*sizeof(float), cudaMemcpyHostToDevice);
  assert(cudaGetLastError() == cudaSuccess);
  cudaMemcpy(gG, G, FAKE_SIZEH*sizeof(float), cudaMemcpyHostToDevice);
  assert(cudaGetLastError() == cudaSuccess);
  cudaMemcpy(gB, B, FAKE_SIZEH*sizeof(float), cudaMemcpyHostToDevice);
  assert(cudaGetLastError() == cudaSuccess);

  clock_t start, stop;

  start = clock();
  for(int i = 0; i < iters; i++) {
    kernel_wrapper(gR, gG, gB, nR, nG, nB, nN);
  }
  cudaDeviceSynchronize();
  stop = clock();

  cudaMemcpy(R, gR, FAKE_SIZEH*sizeof(float), cudaMemcpyDeviceToHost);
  assert(cudaGetLastError() == cudaSuccess);
  cudaMemcpy(G, gG, FAKE_SIZEH*sizeof(float), cudaMemcpyDeviceToHost);
  assert(cudaGetLastError() == cudaSuccess);
  cudaMemcpy(B, gB, FAKE_SIZEH*sizeof(float), cudaMemcpyDeviceToHost);
  assert(cudaGetLastError() == cudaSuccess);

  float tempo = (float)(stop - start) / CLOCKS_PER_SEC;
  printf("Tempo total: %fs (tempo medio por iteracao: %fs)\n", tempo, tempo/iters);
  printf("Nota: tempo nao inclui tempo de copia de/para a GPU\n");

  writePPM(argv[2], R, G, B);

  return 0;
}
