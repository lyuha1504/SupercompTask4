#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

const int BLOCK_SIZE = 512;

//parameters for debugging
bool print_solve_iter = false; //print results of method iterations
bool print_the_solution = false; //print the final solution


//////////////////////////////////////////// Auxilliary functions and structures //////////////////////////////


//structure for describing grid parameters, process domain boundaries and its location
struct GridDomain {
    const double A1 = -1.0;
    const double A2 = 2.0;
    const double B1 = -2.0;
    const double B2 = 2.0;
    //process domain = [i_beg, i_end) X [j_beg, j_end)
    int i_beg;
    int i_end;
    int j_beg;
    int j_end;
    //process domain size by X and Y and overall size
    int i_size;
    int j_size;
    int size;
    //indexes for process neighbours (> 0 if there is such neighbour, = 0 - otherwise)
    int down_nb;
    int left_nb;
    int right_nb;
    int up_nb;
    //number of cells in overall grid and grid steps by X and Y
    int M;
    int N;
    double h1;
    double h2;
    //some useful products (for further calculations)
    double h1_by_h2;   // h1 * h2
    double h1_over_h2; // h1 / h2
    double h2_over_h1; // h2 / h1
};


//auxilliary function for getting the node local index in domain
__host__ __device__ int Index(int i, int j, GridDomain* gd) {

    //error (outside the grid)
    if (i < 0) return (j - gd->j_beg) * gd->i_size;
    if (i > gd->M) return (j - gd->j_beg) * gd->i_size + gd->i_size - 1;
    if (j < 1) return (j + 1 - gd->j_beg) * gd->i_size + (i - gd->i_beg);
    if (j > gd->N) return (j - 1 - gd->j_beg) * gd->i_size + (i - gd->i_beg);

    //node inside the domain
    if (i >= gd->i_beg && i < gd->i_end && j >= gd->j_beg && j < gd->j_end)
        return (j - gd->j_beg) * gd->i_size + (i - gd->i_beg);

    //lower halo node
    if (j == gd->j_beg - 1)
        return gd->size + i - gd->i_beg;
    //left halo node
    if (i == gd->i_beg - 1)
        return gd->size + j - gd->j_beg + gd->down_nb;
    //right halo node
    if (i == gd->i_end)
        return gd->size + j - gd->j_beg + gd->down_nb + gd->left_nb;
    //upper halo node
    if (i == gd->j_end)
        return gd->size + i - gd->i_beg + gd->down_nb + gd->left_nb + gd->right_nb;

    return 0; //never returned

}


/////////////////////////////// Poisson equation parameters and boundary conditions /////////////////////////////


//Poisson equation koefficient (in Laplace operator)
__device__ double k(double x, double y) {
    return 4.0 + x;
}

//potential
__device__ double q(double x, double y) {
    return (x + y) * (x + y);
}

//right side function of the equation
__device__ double F(double x, double y) {
    return ((x + y + 1.0) * (x + y + 1.0) - 1.0 + 4.0 * (4.0 + x) * (1.0 - 2 * (x + y) * (x + y))) * exp(1.0 - (x + y) * (x + y));
}

//1st type condition on the bottom
__host__ __device__ double phi(double x, const double B1) {
    return exp(1.0 - (x + B1) * (x + B1));
}

//2nd type condition on the right
__device__ double psiR(double y, const double A2) {
    return -12.0 * (A2 + y) * exp(1.0 - (A2 + y) * (A2 + y));
}

//2nd type condition on the top
__device__ double psiT(double x, const double B2) {
    return -2.0 * (4.0 + x) * (x + B2) * exp(1.0 - (x + B2) * (x + B2));
}

//2nd type condition on the left
__device__ double psiL(double y, const double A1) {
    return 6.0 * (A1 + y) * exp(1.0 - (A1 + y) * (A1 + y));
}


///////////////////////////// coefficients a and b and coordinates x and y ///////////////////////////////


//find x and y according to indexes
__device__ double x(int i, double h1, const double A1) {
    return i * h1 + A1;
}

__device__ double y(int j, double h2, const double B1) {
    return j * h2 + B1;
}


//coefficients in the linear system
__device__ double a(int i, int j, GridDomain* gd) {
    return gd->h2_over_h1 * k(x(i, gd->h1, gd->A1) - 0.5 * gd->h1, y(j, gd->h2, gd->B1));
}

__device__ double b(int i, int j, GridDomain* gd) {
    return gd->h1_over_h2 * k(x(i, gd->h1, gd->A1), y(j, gd->h2, gd->B1) - 0.5 * gd->h2);
}


/////////////////////////// Kernels and functions for solving the system of linear equations //////////////////


//kernel for filling the initial estimate, initial discrepancy and rho function vector
__global__ void Initialize_kernel(double* omega, double* r, GridDomain* gd, int N_OWN) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= N_OWN) return;

    int i = index % gd->i_size + gd->i_beg;
    int j = index / gd->i_size + gd->j_beg;

    double r_elem = gd->h1_by_h2 * F(x(i, gd->h1, gd->A1), y(j, gd->h2, gd->B1));

    if (i == 0) r_elem += 2 * gd->h2 * psiL(y(j, gd->h2, gd->B1), gd->A1);

    if (i == gd->M) r_elem += 2 * gd->h2 * psiR(y(j, gd->h2, gd->B1), gd->A2);

    if (j == 1) r_elem += b(i, j, gd) * phi(x(i, gd->h1, gd->A1), gd->B1);

    if (j == gd->N) r_elem += 2 * gd->h1 * psiT(x(i, gd->h1, gd->A1), gd->B2);

    r[index] = -r_elem;
    omega[index] = 0.0;

}


//Laplace operator on the grid
__device__ double Laplace_operator(double* r, int i, int j, GridDomain* gd) {

    double center_coeff = b(i, j, gd);
    if (j == gd->N) center_coeff *= 2;
    double down_coeff = (j > 1) ? center_coeff : 0.0;
    double left_coeff = (i > 0) ? a(i, j, gd) : 0.0;
    if (i == gd->M) left_coeff *= 2;
    double right_coeff = (i < gd->M) ? a(i + 1, j, gd) : 0.0;
    if (i == 0) right_coeff *= 2;
    double up_coeff = (j < gd->N) ? b(i, j + 1, gd) : 0.0;

    return (center_coeff + left_coeff + right_coeff + up_coeff) * r[Index(i, j, gd)] -
        down_coeff * r[Index(i, j - 1, gd)] - left_coeff * r[Index(i - 1, j, gd)] -
        right_coeff * r[Index(i + 1, j, gd)] - up_coeff * r[Index(i, j + 1, gd)];

}


//find the vector image after applying operator A (left side of linear system)
__global__ void A_operator_kernel(double* Ar, double* r, GridDomain* gd, int N_OWN) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= N_OWN) return;

    int i = index % gd->i_size + gd->i_beg;
    int j = index / gd->i_size + gd->j_beg;

    Ar[Index(i, j, gd)] = gd->h1_by_h2 * q(x(i, gd->h1, gd->A1), y(j, gd->h2, gd->B1)) * r[Index(i, j, gd)] +
        Laplace_operator(r, i, j, gd);

}


//two vectors' scalar product kernel
__global__ void Dot_kernel(double* res, double* vec1, double* vec2, GridDomain* gd, int N_OWN) {

    int idx = threadIdx.x;
    double local_sum = 0.0;

    for (int index = idx; index < N_OWN; index += BLOCK_SIZE) {
        double s = gd->h1_by_h2 * vec1[index] * vec2[index];
        int i = index % gd->i_size + gd->i_beg;
        int j = index / gd->i_size + gd->j_beg;
        if (i == 0 || i == gd->M) s *= 0.5;
        if (j == gd->N) s *= 0.5;
        local_sum += s;
    }

    __shared__ double sums[BLOCK_SIZE];

    sums[idx] = local_sum;
    __syncthreads();

    for (int size = BLOCK_SIZE / 2; size > 0; size /= 2) {
        if (idx < size) sums[idx] += sums[idx + size];
        __syncthreads();
    }

    if (idx == 0) *res = sums[0];

}


//two vectors' linear combination kernel
__global__ void Axpby_kernel(double* omega, double* r, double* p, double tau, int N_OWN) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= N_OWN) return;

    omega[index] += tau * r[index];
    r[index] += tau * p[index];

}


//function for calculating the vector norm
double grid_norm(double* res_device, double* r, GridDomain* gd, int N_OWN) {

    double res;

    Dot_kernel<<<1, BLOCK_SIZE>>>(&res_device[0], r, r, gd, N_OWN); // (r, r)

    cudaMemcpy(&res, &res_device[0], 1 * sizeof(double), cudaMemcpyDeviceToHost);

    return sqrt(res);

}


//function for calculating the iteration parameter
double find_tau(double* res_device, double* Ar, double* r, GridDomain* gd, int N_OWN) {

    double res[2];

    Dot_kernel<<<1, BLOCK_SIZE>>>(&res_device[0], Ar, r, gd, N_OWN); // (Ar, r)
    Dot_kernel<<<1, BLOCK_SIZE>>>(&res_device[1], Ar, Ar, gd, N_OWN); // (Ar, Ar)

    cudaMemcpy(res, res_device, 2 * sizeof(double), cudaMemcpyDeviceToHost);

    return -res[0] / res[1];

}


//////////////////// function for solving the system of linear equations ////////////////////////////////


//solve the linear system using the minimal discrepancies method
double* Solve(GridDomain& gd_host, bool print_iter = false) {

    const double eps = 0.000001; //solution calculation's accuracy
    double tau, discrep_norm, accuracy; //iteration parameter, discrepancy norm and accuracy
    int N_OWN = gd_host.size;
    int N_ALL = gd_host.size + gd_host.down_nb + gd_host.left_nb + gd_host.right_nb + gd_host.up_nb;
    double* omega; //solution vector on device
    double* r; //discrepancy vector on device
    double* Ar; //discrepancy image after applying operator A (on device)
    double* dot_res_device; //scalar products results on device
    GridDomain* gd; //grid and domain structure on device
    int k = 0;
    int nblocks = (N_OWN + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaMalloc((void**)&omega, N_OWN * sizeof(double));
    cudaMalloc((void**)&r, N_ALL * sizeof(double));
    cudaMalloc((void**)&Ar, N_OWN * sizeof(double));
    cudaMalloc((void**)&dot_res_device, 2 * sizeof(double));

    cudaMalloc((void**)&gd, 1 * sizeof(GridDomain));
    cudaMemcpy(gd, &gd_host, 1 * sizeof(GridDomain), cudaMemcpyHostToDevice);

    Initialize_kernel<<<nblocks, BLOCK_SIZE>>>(omega, r, gd, N_OWN);

    discrep_norm = grid_norm(dot_res_device, r, gd, N_OWN);

    if (print_iter)
        printf("Start:       ||A omega(0) - B|| = %12.8f\n", discrep_norm);

    do {

        A_operator_kernel<<<nblocks, BLOCK_SIZE>>>(Ar, r, gd, N_OWN); // p = Ar

        tau = find_tau(dot_res_device, Ar, r, gd, N_OWN); //find the iteration parameter

        Axpby_kernel<<<nblocks, BLOCK_SIZE>>>(omega, r, Ar, tau, N_OWN); //omega = omega - tau * r, r = r - tau * p

        accuracy = fabs(tau) * discrep_norm; // ||omega(k+1) - omega(k)||
        discrep_norm = grid_norm(dot_res_device, r, gd, N_OWN);

        k++;

        if (print_iter) {
            printf("iteration %d: ||A omega(%d) - B|| = %12.8f, ", k, k, discrep_norm);
            printf("||omega(%d) - omega(%d)|| = %12.8f\n", k, k - 1, accuracy);
        }

    } while (accuracy > eps);

    double* omega_host = new double[N_OWN];
    cudaMemcpy(omega_host, omega, N_OWN * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(omega);
    cudaFree(r);
    cudaFree(Ar);
    cudaFree(dot_res_device);
    cudaFree(gd);

    if (print_iter) printf("\n");
    printf("Number of iterations:   k = %d\n\n", k);
    printf("||omega(k) - omega(k-1)|| = %11.9f\n\n", accuracy);
    printf("||A omega(k) - B||        = %11.9f\n\n", discrep_norm);

    return omega_host;

}


////////////////////// main function, initializing functions and solution printing function ///////////////


//function for identifying the grid parameters, the domain borders, its size and neighbours
void grid_and_domain(GridDomain& gd, int M, int N) {

    gd.M = M;
    gd.N = N;
    gd.h1 = (gd.A2 - gd.A1) / M;
    gd.h2 = (gd.B2 - gd.B1) / N;
    gd.h1_by_h2 = gd.h1 * gd.h2;
    gd.h1_over_h2 = gd.h1 / gd.h2;
    gd.h2_over_h1 = gd.h2 / gd.h1;

    gd.i_beg = 0;
    gd.i_size = M + 1;
    gd.i_end = M + 1;
    gd.j_beg = 1;
    gd.j_size = N;
    gd.j_end = N + 1;
    gd.size = gd.i_size * gd.j_size;

    gd.up_nb = 0;
    gd.down_nb = 0;
    gd.left_nb = 0;
    gd.right_nb = 0;

}


//function that prints the solution data for graphics
void print_solution(double* omega, GridDomain& gd) {

    printf("Solution:\n");
    for (int i = 0; i < gd.M + 1; ++i) {
        double x = gd.A1 + gd.h1 * i;
        printf("%6.3f %6.3f %10.7f\n", x, gd.B1, phi(x, gd.B1));
    }
    for (int i = gd.i_beg; i < gd.i_end; ++i)
        for (int j = gd.j_beg; j < gd.j_end; ++j)
            printf("%7.4f %7.4f %10.7f\n", i, j, omega[Index(i, j, &gd)]);

}


int main(int argc, char** argv) {

    if (argc != 3) {

        printf("Please execute the program with the following string:\n\nexe_file_name M N\n\n");
        printf("where M and N are the numbers of cells on X and Y axis (positive integers)\n");

    }
    else {

        int M = atoi(argv[1]);
        int N = atoi(argv[2]); //determine grid sizes

        if (M <= 0 || N <= 0) printf("Please give all the parameters integer positive values\n");
        else {

            GridDomain gd;
            grid_and_domain(gd, M, N);

            double* omega = Solve(gd, print_solve_iter);

            if (print_the_solution) print_solution(omega, gd);

            delete[] omega;

        }

    }

    return 0;

}