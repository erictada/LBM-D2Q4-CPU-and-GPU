// LBM Code for 2 - D, diffusion problems, D2Q4
// Adapted from the book Lattice Boltzmann Method - Fundamentals
// and Engineering Applications with Computer Codes by A. Mohamad
// Output file can be opened by the free software ParaView

// Eric Tada, April 24th, 2019

#include "stdio.h"
#include "math.h"
#include "stdlib.h"
#include <chrono>
#include "string.h"
#include "cuda_runtime.h"

#define m 100 //m is the number of lattice nodes (y)
#define n 100 //n is the number of lattice nodes (x)


// Collision kernel, done in parallel
__global__ void collision(float *f1, float *f2, float *f3, float *f4, float *rho, float *omega) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x + (blockDim.y * blockIdx.y + threadIdx.y) * (n+1);

	float feq = 0.25*rho[tid];
	f1[tid] = omega[0]*feq + (1.0 - omega[0])*f1[tid];
	f2[tid] = omega[0]*feq + (1.0 - omega[0])*f2[tid];
	f3[tid] = omega[0]*feq + (1.0 - omega[0])*f3[tid];
	f4[tid] = omega[0]*feq + (1.0 - omega[0])*f4[tid];
}

// Streaming kernel for f1 and f2, done in series for each row
__global__ void streaming12(float *f1, float *f2) {
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	for (int i = 1; i <= n; i++) {
		f1[j*(n + 1) + n - i] = f1[j*(n + 1) + n - i - 1];
		f2[j*(n+1) + i - 1] = f2[j*(n + 1) + i];
	}
}

// Streaming kernel for f3 and f4, done in series for each column
__global__ void streaming34(float *f3, float *f4) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	for (int j = 1; j <= m; j++) {
		f3[(m-j)*(n + 1) + i] = f3[(m - j - 1)*(n + 1) + i];
		f4[(j - 1)*(n + 1) + i] = f4[(j)*(n + 1) + i];
	}
}

// Kernel to apply boundary conditions (1)
__global__ void bound1(float *f1, float *f2, float *f3, float *f4) {
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	if (j == 0 || j == n) {
	}
	else {
		f1[j*(n + 1)] = 0.5 - f2[j*(n + 1)];
		f3[j*(n + 1)] = 0.5 - f4[j*(n + 1)];
		f1[j*(n + 1) + n] = 0.0;
		f2[j*(n + 1) + n] = 0.0;
		f3[j*(n + 1) + n] = 0.0;
		f4[j*(n + 1) + n] = 0.0;
	}
}

// Kernel to apply boundary conditions (2)
__global__ void bound2(float *f1, float *f2, float *f3, float *f4) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i == 0 || i == n) {
	}
	else {
		f1[m*(n + 1) + i] = 0.0;
		f2[m*(n + 1) + i] = 0.0;
		f3[m*(n + 1) + i] = 0.0;
		f4[m*(n + 1) + i] = 0.0;
		f1[i] = f1[n + 1 + i];
		f2[i] = f2[n + 1 + i];
		f3[i] = f3[n + 1 + i];
		f4[i] = f4[n + 1 + i];
	}
}

// Kernel to update rho value
__global__ void update(float *f1, float *f2, float *f3, float *f4, float *rho) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x + (blockDim.y * blockIdx.y + threadIdx.y) * (n + 1);

	rho[tid] = f1[tid] + f2[tid] + f3[tid] + f4[tid];
}

// CPU function to output results
void print(float x[n + 1], float y[m + 1], float rho[(n + 1)*(m + 1)], int step) {
	char str[20];
	sprintf(str, "step%06d.vtk", step);

	FILE *res;
	res = fopen(str, "w");
	int i, j;

	fprintf(res, "# vtk DataFile Version 3.0\r\nvtk output\r\nASCII\r\nDATASET RECTILINEAR_GRID\r\nDIMENSIONS %d %d 1\r\n\r\n", n + 1, m + 1);
	fprintf(res, "X_COORDINATES %d  float\r\n", n + 1);
	for (i = 0; i <= n; i++) { fprintf(res, "%f ", x[i]); }
	fprintf(res, "\r\nY_COORDINATES %d  float\r\n", m + 1);
	for (j = 0; j <= m; j++) { fprintf(res, "%f ", y[j]); }
	fprintf(res, "\r\nZ_COORDINATES 1 float\r\n0\r\n\r\n");
	fprintf(res, "POINT_DATA %d\r\n", (n + 1)*(m + 1));
	fprintf(res, "FIELD FieldData 1\r\nv 1 %d float\r\n", (n + 1)*(m + 1));
	for (j = 0; j <= n; j++) {
		for (i = 0; i <= m; i++) {
			fprintf(res, "%f ", rho[j*(n+1) + i]);
		}
		fprintf(res, "\r\n");
	}
	fclose(res);
}

int main() {
	float f1[(n + 1)*(m + 1)], f2[(n + 1)*(m + 1)], f3[(n + 1)*(m + 1)], f4[(n + 1)*(m + 1)];
	float rho[(n + 1)*(m + 1)], x[n + 1], y[m + 1];
	int i, j;

	float dx = 1.0;
	float dy = dx;
	float dt = 1.0;

	x[0] = 0.0;
	y[0] = 0.0;
	for (i = 1; i <= n; i++) {
		x[i] = x[i - 1] + dx;
	}
	for (j = 1; j <= m; j++) {
		y[j] = y[j - 1] + dy;
	}
	float csq = dx*dx / (dt*dt);
	float alpha = 0.25;
	float omega[1];
	omega[0] = 1.0 / (2.*alpha / (dt*csq) + 0.5);
	float mstep = 4000;
	for (j = 0; j <= m; j++) {
		for (i = 0; i <= n; i++) {
			rho[j*(n+1) + i] = 0.0; //initial values of the dependent variable
		}
	}
	for (j = 0; j <= m; j++) {
		for (i = 0; i <= n; i++) {
			f1[j*(n + 1) + i] = 0.25*rho[j*(n + 1) + i];
			f2[j*(n + 1) + i] = 0.25*rho[j*(n + 1) + i];
			f3[j*(n + 1) + i] = 0.25*rho[j*(n + 1) + i];
			f4[j*(n + 1) + i] = 0.25*rho[j*(n + 1) + i];
		}
	}

	print(x, y, rho, 0);

	// Create GPU variables
	float *d_f1, float *d_f2, float *d_f3, float *d_f4, float *d_rho, float *d_omega;

	// Allocate memory to GPU
	cudaMalloc((void**)&d_f1, ((n+1)*(m+1)) * sizeof(float));
	cudaMalloc((void**)&d_f2, ((n + 1)*(m + 1)) * sizeof(float));
	cudaMalloc((void**)&d_f3, ((n + 1)*(m + 1)) * sizeof(float));
	cudaMalloc((void**)&d_f4, ((n + 1)*(m + 1)) * sizeof(float));
	cudaMalloc((void**)&d_rho, ((n + 1)*(m + 1)) * sizeof(float));
	cudaMalloc((void**)&d_omega, (1 * sizeof(float)));

	dim3 blocksij(n+1, m+1, 1), threads(1, 1, 1);
	dim3 blocksi(n + 1, 1, 1);
	dim3 blocksj(1, m + 1, 1);

	// Copy from host to device
	cudaMemcpy(d_f1, f1, (n+1)*(m+1)*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_f2, f2, (n + 1)*(m + 1) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_f3, f3, (n + 1)*(m + 1) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_f4, f4, (n + 1)*(m + 1) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_rho, rho, (n + 1)*(m + 1) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_omega, omega, sizeof(float), cudaMemcpyHostToDevice);

	// Record start time
	auto start = std::chrono::high_resolution_clock::now();

	for (int kk = 1; kk <= mstep; kk++) {

		//collision
		collision << < blocksij, threads >> > (d_f1, d_f2, d_f3, d_f4, d_rho, d_omega);

		//streaming
		streaming12 << <blocksj, threads >> > (d_f1, d_f2);
		streaming34 << <blocksi, threads >> > (d_f3, d_f4);

		//boundary conditions
		bound1 << <blocksj, threads >> > (d_f1, d_f2, d_f3, d_f4);
		bound2 << <blocksi, threads >> > (d_f1, d_f2, d_f3, d_f4);

		//update rho
		update << < blocksij, threads >> > (d_f1, d_f2, d_f3, d_f4, d_rho);

		//output result
		if (kk % 20 == 0) {
			cudaMemcpy(rho, d_rho, (n + 1)*(m + 1) * sizeof(float), cudaMemcpyDeviceToHost);
			print(x, y, rho, kk);
		}
	}

	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	printf("Elapsed time for LBM: %f s\n", elapsed.count());

	getchar();
}