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

#define m 100 //m is the number of lattice nodes (y)
#define n 100 //n is the number of lattice nodes (x)

void print(float x[n+1], float y[m+1], float rho[n+1][m+1], int step) {
	//print result

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
			fprintf(res, "%f ", rho[i][j]);
		}
		fprintf(res, "\r\n");
	}
	fclose(res);
}

int main() {
	float f1[n + 1][m + 1], f2[n + 1][m + 1], f3[n + 1][m + 1], f4[n + 1][m + 1];
	float rho[n + 1][m + 1], feq, x[n + 1], y[m + 1];
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
	float omega = 1.0 / (2.*alpha / (dt*csq) + 0.5);
	float mstep = 4000;
	for (j = 0; j <= m; j++) {
		for (i = 0; i <= n; i++) {
			rho[i][j] = 0.0; //initial values of the dependent variable
		}
	}
	for (j = 0; j <= m; j++) {
		for (i = 0; i <= n; i++) {
			f1[i][j] = 0.25*rho[i][j];
			f2[i][j] = 0.25*rho[i][j];
			f3[i][j] = 0.25*rho[i][j];
			f4[i][j] = 0.25*rho[i][j];
		}
	}

	print(x, y, rho, 0);


	// Record start time
	auto start = std::chrono::high_resolution_clock::now();

	for (int kk = 1; kk <= mstep; kk++) {

		//collision
		for (j = 0; j <= m; j++) {
			for (i = 0; i <= n; i++) {
				feq = 0.25*rho[i][j];
				f1[i][j] = omega*feq + (1.0 - omega)*f1[i][j];
				f2[i][j] = omega*feq + (1.0 - omega)*f2[i][j];
				f3[i][j] = omega*feq + (1.0 - omega)*f3[i][j];
				f4[i][j] = omega*feq + (1.0 - omega)*f4[i][j];
			}
		}

		//streaming
		for (j = 0; j <= m; j++) {
			for (i = 1; i <= n; i++) {
				f1[n - i][j] = f1[n - i - 1][j];
				f2[i - 1][j] = f2[i][j];
			}
		}
		for (j = 1; j <= m; j++) {
			for (i = 0; i <= n; i++) {
				f3[i][m - j] = f3[i][m - j - 1];
				f4[i][j - 1] = f4[i][j];
			}
		}

		//boundary conditions
		for (j = 1; j < m; j++) { //Dependent variable equal to 1 in left boundary and equal to 0 in right boundary
			f1[0][j] = 0.5 - f2[0][j];
			f3[0][j] = 0.5 - f4[0][j];
			f1[n][j] = 0.0;
			f2[n][j] = 0.0;
			f3[n][j] = 0.0;
			f4[n][j] = 0.0;
		}
		for (i = 1; i < n; i++) { // Dependent variable equal to 0 in upper boundary, Neumann down
			f1[i][m] = 0.0;
			f2[i][m] = 0.0;
			f3[i][m] = 0.0;
			f4[i][m] = 0.0;
			f1[i][0] = f1[i][1];
			f2[i][0] = f2[i][1];
			f3[i][0] = f3[i][1];
			f4[i][0] = f4[i][1];
		}

		//update rho
		for (j = 0; j < m; j++) {
			for (i = 0; i < n; i++) {
				rho[i][j] = f1[i][j] + f2[i][j] + f3[i][j] + f4[i][j];
			}
		}
		rho[0][0] = 1.0;

		if (kk % 20 == 0) {
			print(x, y, rho, kk);
		}
	}

	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	printf("Elapsed time for LBM: %f s\n", elapsed.count());			
															
	getchar();
}