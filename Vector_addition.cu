#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <random>
#include <functional> // std::bind
#include <memory> // unique_ptr
// cuda
#include "cuda_runtime.h" // cudaMalloc, cudaFree in cudart.lib
#include "device_launch_parameters.h" 
#include "cublas_v2.h"
#include <curand_kernel.h>
// boost
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/random.hpp>
//#include <boost/range/combine.hpp>
// Eigen
#include <Eigen/Dense> // C:\installed\eigen-3.4.0\   additional include library
// Armadillo
#include <armadillo> // C:\installed\armadillo-12.8.0\include
// addtional lib directories : C:\installed\armadillo-12.8.0\examples\lib_win64         can not be saved ???
// linker->input -> addtional dependuncies : libopenblas.lib

template<class T>
void addvec(T* c, T* a, T* b, size_t N)
{
    //std::cout << "addvec* called" << std::endl; // C, C/C++, vector.data()
    for (int i = 0; i < N; i++)
        c[i] = a[i] + b[i];
}

template<class T>
void addvec(T& c, T& a, T& b, size_t N)
{
    //std::cout << "addvec& called" << std::endl; // C++, ublas
    for (int i = 0; i < N; i++)
        c[i] = a[i] + b[i];
}

template<class T>
void addvec(std::vector<T>& c, std::vector<T>& a, std::vector<T>& const b)
{
    //std::cout << "addvec3 called" << std::endl; // C++ vector 
    auto csize = c.size(); // call the size function once, not N times
    for (int i = 0; i < csize; i++)
        c[i] = a[i] + b[i];
}

__global__ void
vectorAdd(const float* A, const float* B, float* C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

template<class T>
void matvec(T* c, T* a, T* b, size_t N)
{
    for (int i = 0; i < N; i++)
    {   c[i] = 0;
        for (int j = 0; j < N; j++)
            c[i] += a[i*N+j] * b[j];
    }
}

template<class T>
void matmat(T* c, T* a, T* b, size_t N)
{   // assume matrix is a set of column vectors
    for (int i = 0; i < N; i++) // row
    {
        for (int j = 0; j < N; j++) // column
        {
            c[i*N+j] = 0;
            for (int k = 0; k < N; k++)
                c[i*N+j] += a[i + N * k] * b[ j * N + k ];
        }
    }
}


void matvec_Vector(std::vector<float>& c, const std::vector<float>& a, const std::vector<float>& b, int N) 
{
    // Check if the matrix and vector dimensions are compatible
    //if (a.size() != N * N || b.size() != N) {
    //    std::cerr << "Error: Matrix and vector dimensions are not compatible." << std::endl;
    //    exit(EXIT_FAILURE);
    //}
    // Perform matrix-vector multiplication
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            c[i] += a[i * N + j] * b[j];
        }
    }
}

void matmat_Vector(std::vector<float>& c, const std::vector<float>& a, const std::vector<float>& b, int N)
{
    // Check if the matrix and vector dimensions are compatible
    //if (a.size() != N * N || b.size() != N) {
    //    std::cerr << "Error: Matrix and vector dimensions are not compatible." << std::endl;
    //    exit(EXIT_FAILURE);
    //}
    // Perform matrix-vector multiplication
    for (int i = 0; i < N; i++) // row
    {
        for (int j = 0; j < N; j++) // column
        {
            c[i * N + j] = 0;
            for (int k = 0; k < N; k++)
                c[i * N + j] += a[i + N * k] * b[j * N + k];
        }
    }
}



/*
template<class T>
void addveciter(std::vector<T>& c, std::vector<T>& const a, std::vector<T>& const b)
{   // using iterators instead of the index
    auto cend = c.end(); // 
    for (auto ci = c.begin(), bi = b.begin(), ai = a.begin(); ci != cend; ci++, bi++, ai++)
        *ci = *ai + *bi;
}
*/


__global__ void cuda_matVec(float* matrix, float* vector, float* result, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        result[i] = 0.0;
        float sum = 0.0;
        for (int j = 0; j < N; ++j) {
            sum += matrix[i * N + j] * vector[j];
        }
        result[i] = sum;
    }
}

__global__ void cuda_matmat(const float* A, const float* B,
        float* C, int m, int n, int k) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < m && col < n) {
            float sum = 0.0f;
            for (int i = 0; i < k; i++) {
                sum += A[row * k + i] * B[i * n + col];
            }
            C[row * n + col] = sum;
        }
    }



// CUDA kernel for matrix-vector multiplication
__global__ void matrixVectorMul(float* d_matrix, float* d_vector, float* d_result, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {
        float sum = 0.0f;
        for (int i = 0; i < size; ++i) {
            sum += d_matrix[tid * size + i] * d_vector[i];
        }
        d_result[tid] = sum;
    }
}


__global__ void initializeRandom(float* array, unsigned long long seed, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    curandState state;
    curand_init(seed, i, 0, &state);

    if (i < size) {
        array[i] = curand_uniform(&state);
    }
}

int Cuda_C_Style(int nStart, int nEnd)
{
    using Duration = std::chrono::nanoseconds;
    std::mt19937 engine(std::random_device{}()); // Mersenne twister MT19937
    std::uniform_real_distribution<float> dist(0.f, 1.f); //Values between 0 and 1
    std::vector<std::string> vs = { "Size","Allocation","Initialize","memcpy","Calculation","memcpy","Verifying","memRelease","Verified" };
    std::cout << std::setw(6);
    for (auto& s : vs) std::cout << std::left << s << std::setw(13);
    std::cout << std::setw(0) << std::endl;
    cudaError_t err = cudaSuccess;
    err = cudaSetDevice(0);
    for (auto i = nStart; i <= nEnd; i++)
    {
        std::vector<Duration> times;
        size_t N = (size_t)pow(10, i);
        auto tNow = std::chrono::high_resolution_clock::now(); auto tPre = tNow;
        size_t tSize = N * sizeof(float);
        float* h_a = (float*)malloc(tSize);// memory allocation at host
        float* h_b = (float*)malloc(tSize);
        float* h_c = (float*)malloc(tSize);
        float* d_a = NULL; err = cudaMalloc((void**)&d_a, tSize);// memory allocation at device
        float* d_b = NULL; err = cudaMalloc((void**)&d_b, tSize);
        float* d_c = NULL; err = cudaMalloc((void**)&d_c, tSize);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        for (auto j = 0; j < N; j++)
        {
            h_a[j] = dist(engine); h_b[j] = dist(engine);
        }
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        err = cudaMemcpy(d_a, h_a, tSize, cudaMemcpyHostToDevice);
        err = cudaMemcpy(d_b, h_b, tSize, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to memcpy to device\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        int threadsPerBlock = 3840; // rtx3060, 3840 cores
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        vectorAdd << <blocksPerGrid, threadsPerBlock >> > (d_a, d_b, d_c, N);
        err = cudaGetLastError();
        err = cudaDeviceSynchronize(); // wait for all threads done
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to kernel\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        err = cudaMemcpy(h_c, d_c, tSize, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to memcpy to Host\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        bool bResult = true;
        for (auto j = 0; j < N; j++)
        {
            if (fabs(h_c[j] - h_a[j] - h_b[j]) > 0.0001)
            {
                bResult = false; break;
            }
            //if (i == 3 && j < 3) std::cout << h_c[j] << " " << h_a[j] << " " << h_b[j] << std::endl;
        }
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        free(h_a); free(h_b); free(h_c);
        err = cudaFree(d_a); err = cudaFree(d_b); err = cudaFree(d_b);
        err = cudaDeviceReset();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to release device memory\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        std::cout << "10^" << i;
        for (auto& t : times) std::cout << "  " << std::scientific << std::right << std::setprecision(4 + 1) << 1.e-9 * t.count();
        std::cout << std::setw(6) << std::boolalpha << bResult << std::endl;
        std::cout << "blocksPerGrid: " << blocksPerGrid << "\tthreadsPerBlock: " << threadsPerBlock << std::endl;
    }
    return 0;
}

int CPP_Vector_ublas(int nStart, int nEnd)
{
    using Duration = std::chrono::nanoseconds;
    std::mt19937 engine(std::random_device{}()); // Mersenne twister MT19937
    std::uniform_real_distribution<float> dist(0.f, 1.f); //Values between 0 and 1
    std::vector<std::string> vs = { "Size","Allocation","Initialize","Cal_adddvec2","Cal_ublas","Verifying","Verified" };
    std::cout << std::setw(6);
    for (auto& s : vs) std::cout << std::left << s << std::setw(13);
    std::cout << std::setw(0) << std::endl;
    for (auto i = nStart; i <= nEnd; i++)
    {
        std::vector<Duration> times;
        size_t N = (size_t)pow(10, i);
        auto tNow = std::chrono::high_resolution_clock::now(); auto tPre = tNow;
        boost::numeric::ublas::vector<float> a(N), b(N),c(N);
        tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        for (auto j = 0; j < N; j++)
        {
            a[j] = dist(engine); b[j] = dist(engine);
        }
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        addvec(c, a, b,N);
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));

        auto d = a + b;
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        bool bResult = true;
        for (auto j = 0; j < N; j++)
        {
            if (fabs(d[j] - a[j] - b[j]) > 0.0001)
            {
                bResult = false; break;
            }
            //if (i == 3 && j < 3) std::cout << c[j] << " " << a[j] << " " << b[j] << std::endl;
        }
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        std::cout << "10^" << i;
        for (auto& t : times) std::cout << "  " << std::scientific << std::right << std::setprecision(4 + 1) << 1.e-9 * t.count();
        std::cout << std::setw(6) << std::boolalpha << bResult << std::endl;

    }
    return 0;
}

int CPP_Vector(int nStart, int nEnd)
{
    using Duration = std::chrono::nanoseconds;
    std::mt19937 engine(std::random_device{}()); // Mersenne twister MT19937
    std::uniform_real_distribution<float> dist(0.f, 1.f); //Values between 0 and 1
    std::vector<std::string> vs = { "Size","Allocation","Initialize","Cal_vector","Cal_vecdata","Verifying","Verified" };
    std::cout << std::setw(6);
    for (auto& s : vs) std::cout << std::left << s << std::setw(13);
    std::cout << std::setw(0) << std::endl;
    for (auto i = nStart; i <= nEnd; i++)
    {
        std::vector<Duration> times;
        size_t N = (size_t)pow(10, i);
        auto tNow = std::chrono::high_resolution_clock::now(); auto tPre = tNow;
        std::vector<float> a(N), b(N), c(N);
        tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        for (auto j = 0; j < N; j++)
        {
            a[j] = dist(engine); b[j] = dist(engine);
        }
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        addvec(c, a, b);
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        addvec(c.data(), a.data(), b.data(), c.size());
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        bool bResult = true;
        for (auto j = 0; j < N; j++)
        {
            if (fabs(c[j] - a[j] - b[j]) > 0.0001)
            {
                bResult = false; break;
            }
            //if (j < 3) std::cout << c[j] << " " << a[j] << " " << b[j] << std::endl;
        }
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        std::cout << "10^" << i;
        for (auto& t : times) std::cout << "  " << std::scientific << std::right << std::setprecision(4 + 1) << 1.e-9 * t.count();
        std::cout << std::setw(6) << std::boolalpha << bResult << std::endl;
    }
    return 0;
}

int CPP_Style(int nStart, int nEnd)
{
    using Duration = std::chrono::nanoseconds;
    std::mt19937 engine(std::random_device{}()); // Mersenne twister MT19937
    std::uniform_real_distribution<float> dist(0.f, 1.f); //Values between 0 and 1
    std::vector<std::string> vs = { "Size","Allocation","Initialize","Calculation","Verifying","Verified" };
    std::cout << std::setw(6);
    for (auto& s : vs) std::cout << std::left << s << std::setw(13);
    std::cout << std::setw(0) << std::endl;
    for (auto i = nStart; i <= nEnd; i++)
    {
        std::vector<Duration> times;
        size_t N = (unsigned int)pow(10, i);
        auto tNow = std::chrono::high_resolution_clock::now(); auto tPre = tNow;
        auto a = std::make_unique<float[]>(N);
        auto b = std::make_unique<float[]>(N);
        auto c = std::make_unique<float[]>(N);
        tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        for (auto j = 0; j < N; j++)
        {
            a[j] = dist(engine); b[j] = dist(engine);
        }
        tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        addvec(c, a, b, N); // call by reference
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        bool bResult = true;
        for (auto j = 0; j < N; j++)
        {
            if (fabs(c[j] - a[j] - b[j]) > 0.0001)
            {
                bResult = false; break;
            }
            //if (j < 3) std::cout << c[j] << " " << a[j] << " " << b[j] << std::endl;
        }
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        std::cout << "10^" << i;
        for (auto& t : times) std::cout << "  " << std::scientific << std::right << std::setprecision(4 + 1) << 1.e-9 * t.count();
        std::cout << std::setw(6) << std::boolalpha << bResult << std::endl;
    }
    return 0;
}

int Old_CPP_Style(int nStart, int nEnd)
{
    using Duration = std::chrono::nanoseconds;
    std::mt19937 engine(std::random_device{}()); // Mersenne twister MT19937
    std::uniform_real_distribution<float> dist(0.f, 1.f); //Values between 0 and 1
    std::vector<std::string> vs = { "Size","Allocation","Initialize","Calculation","Verifying","ReleaseMem","Verified" };
    std::cout << std::setw(6);
    for (auto& s : vs) std::cout << std::left << s << std::setw(13);
    std::cout << std::setw(0) << std::endl;
    for (auto i = nStart; i <= nEnd; i++)
    {
        std::vector<Duration> times;
        size_t N = (size_t)pow(10, i);
        auto tNow = std::chrono::high_resolution_clock::now(); auto tPre = tNow;
        float* a = new float[N]; float* b = new float[N]; float* c = new float[N];
        tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        for (auto j = 0; j < N; j++)
        {
            a[j] = dist(engine); b[j] = dist(engine);
        }
        tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        addvec(c, a, b, N);
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        bool bResult = true;
        for (auto j = 0; j < N; j++)
        {
            if (fabs(c[j] - a[j] - b[j]) > 0.0001)
            {
                bResult = false; break;
            }
            //if (j < 3) std::cout << c[j] << " " << a[j] << " " << b[j] << std::endl;
        }
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        delete a; delete b; delete c;
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        std::cout << "10^" << i;
        for (auto& t : times) std::cout << "  " << std::scientific << std::right << std::setprecision(4 + 1) << 1.e-9 * t.count();
        std::cout << std::setw(6) << std::boolalpha << bResult << std::endl;
    }
    return 0;
}

int Old_C_Style(int nStart, int nEnd)
{
    using Duration = std::chrono::nanoseconds;
    std::mt19937 engine(std::random_device{}()); // Mersenne twister MT19937
    std::uniform_real_distribution<float> dist(0.f, 1.f); //Values between 0 and 1
    std::vector<std::string> vs = { "Size","Allocation","Initialize","Calculation","Verifying","ReleaseMem","Verified"};
    std::cout << std::setw(6);
    for (auto& s : vs ) std::cout << std::left << s << std::setw(13)  ;
    std::cout << std::setw(0) << std::endl;
    for (auto i = nStart; i <= nEnd; i++)
    {
        std::vector<Duration> times;
        size_t N = (size_t)pow(10, i);
        auto tNow = std::chrono::high_resolution_clock::now(); auto tPre = tNow;
        float* a = (float*)malloc(N * sizeof(float));
        float* b = (float*)malloc(N * sizeof(float));
        float* c = (float*)malloc(N * sizeof(float));
        tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        for (auto j = 0; j < N; j++)
        {
            a[j] = dist(engine); b[j] = dist(engine);
        }
        //for (auto k = 0; k < N; k++) std::cout << a[k] << "\t" << b[k] << std::endl; // check the random number
        tPre = tNow;tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        addvec(c, a, b, N);
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        bool bResult = true;
        for (auto j = 0; j < N; j++)
        {   if (fabs(c[j] - a[j] - b[j]) > 0.0001)
            {   bResult = false; break;
            }
            //if (j < 3) std::cout << c[j] << " " << a[j] << " " << b[j] << std::endl;
        }
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        free(a); free(b); free(c);
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        std::cout << "10^" << i;
        for (auto& t : times) std::cout << "  " << std::scientific << std::right << std::setprecision(4 + 1) << 1.e-9 * t.count();
        std::cout << std::setw(6) << std::boolalpha << bResult << std::endl;
    }
    return 0;
}

int Old_C_Style_MV(int nStart, int nEnd)
{
    using Duration = std::chrono::nanoseconds;
    std::mt19937 engine(std::random_device{}()); // Mersenne twister MT19937
    std::uniform_real_distribution<float> dist(0.f, 1.f); //Values between 0 and 1
    std::vector<std::string> vs = { "Size","Allocation","Initialize","Calculation","Verifying","ReleaseMem","Verified" };
    std::cout << std::setw(6);
    for (auto& s : vs) std::cout << std::left << s << std::setw(13);
    std::cout << std::setw(0) << std::endl;
    for (auto i = nStart; i <= nEnd; i++)
    {
        std::vector<Duration> times;
        size_t N = (size_t)pow(10, i);
        auto tNow = std::chrono::high_resolution_clock::now(); auto tPre = tNow;
        float* a = (float*)malloc(N * N * sizeof(float));
        float* b = (float*)malloc(N * sizeof(float));
        float* c = (float*)malloc(N * sizeof(float));
        tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        for (auto j = 0; j < N; j++)
        {   for( auto k = 0;k<N;k++)
                a[j] = dist(engine);
            b[j] = dist(engine);
        }
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        //addvec(c, a, b, N);
        matvec(c, a, b, N);
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        bool bResult = true; float sum = 0;
        for (auto j = 0; j < N; j++)
        {   sum += a[j] * b[j];
        }
        if (fabs(c[0] - sum ) > 0.0001)
        {bResult = false;
        }
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        free(a); free(b); free(c);
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        std::cout << "10^" << i;
        for (auto& t : times) std::cout << "  " << std::scientific << std::right << std::setprecision(4 + 1) << 1.e-9 * t.count();
        std::cout << std::setw(6) << std::boolalpha << bResult << std::endl;
    }
    return 0;
}
int Old_CPP_Style_MV(int nStart, int nEnd)
{
    using Duration = std::chrono::nanoseconds;
    std::mt19937 engine(std::random_device{}()); // Mersenne twister MT19937
    std::uniform_real_distribution<float> dist(0.f, 1.f); //Values between 0 and 1
    std::vector<std::string> vs = { "Size","Allocation","Initialize","Calculation","Verifying","ReleaseMem","Verified" };
    std::cout << std::setw(6);
    for (auto& s : vs) std::cout << std::left << s << std::setw(13);
    std::cout << std::setw(0) << std::endl;
    for (auto i = nStart; i <= nEnd; i++)
    {
        std::vector<Duration> times;
        size_t N = (size_t)pow(10, i);
        auto tNow = std::chrono::high_resolution_clock::now(); auto tPre = tNow;
        float* a = new float[N*N]; float* b = new float[N]; float* c = new float[N];
        tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        for (auto j = 0; j < N; j++)
        {
            for (auto k = 0; k < N; k++)
                a[j] = dist(engine);
            b[j] = dist(engine);
        }
        tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        matvec(c, a, b, N);
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        bool bResult = true; float sum = 0;
        for (auto j = 0; j < N; j++)
        {
            sum += a[j] * b[j];
        }
        if (fabs(c[0] - sum) > 0.0001)
        {
            bResult = false;
        }
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        delete a; delete b; delete c;

        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        std::cout << "10^" << i;
        for (auto& t : times) std::cout << "  " << std::scientific << std::right << std::setprecision(4 + 1) << 1.e-9 * t.count();
        std::cout << std::setw(6) << std::boolalpha << bResult << std::endl;
    }
    return 0;
}

int CPP_Style_MV(int nStart, int nEnd)
{
    using Duration = std::chrono::nanoseconds;
    std::mt19937 engine(std::random_device{}()); // Mersenne twister MT19937
    std::uniform_real_distribution<float> dist(0.f, 1.f); //Values between 0 and 1
    std::vector<std::string> vs = { "Size","Allocation","Initialize","Calculation","Verifying","Verified" };
    std::cout << std::setw(6);
    for (auto& s : vs) std::cout << std::left << s << std::setw(13);
    std::cout << std::setw(0) << std::endl;
    for (auto i = nStart; i <= nEnd; i++)
    {
        std::vector<Duration> times;
        size_t N = (size_t)pow(10, i);
        auto tNow = std::chrono::high_resolution_clock::now(); auto tPre = tNow;
        auto a = std::make_unique<float[]>(N*N);
        auto b = std::make_unique<float[]>(N);
        auto c = std::make_unique<float[]>(N);
        tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        for (auto j = 0; j < N; j++)
        {
            for (auto k = 0; k < N; k++)
                a[j*N+k] = dist(engine);
            b[j] = dist(engine);
        }
        tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        matvec(c.get(), a.get(), b.get(), N); // using smart pointers in NOT smart way.
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        bool bResult = true; float sum = 0;
        for (auto j = 0; j < N; j++)
        {
            sum += a[j] * b[j];
        }
        if (fabs(c[0] - sum) > 0.0001)
        {
            bResult = false;
        }
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        std::cout << "10^" << i;
        for (auto& t : times) std::cout << "  " << std::scientific << std::right << std::setprecision(4 + 1) << 1.e-9 * t.count();
        std::cout << std::setw(6) << std::boolalpha << bResult << std::endl;
    }
    return 0;
}

int CPP_Vector_MV(int nStart, int nEnd)
{
    using Duration = std::chrono::nanoseconds;
    std::mt19937 engine(std::random_device{}()); // Mersenne twister MT19937
    std::uniform_real_distribution<float> dist(0.f, 1.f); //Values between 0 and 1
    std::vector<std::string> vs = { "Size","Allocation","Initialize","Cal_vector","Cal_ptr","Verifying","Verified" };
    std::cout << std::setw(6);
    for (auto& s : vs) std::cout << std::left << s << std::setw(13);
    std::cout << std::setw(0) << std::endl;
    for (auto i = nStart; i <= nEnd; i++)
    {
        std::vector<Duration> times;
        size_t N = (size_t)pow(10, i);
        auto tNow = std::chrono::high_resolution_clock::now(); auto tPre = tNow;
        std::vector<float> a(N*N, 0.0);std::vector<float> b(N, 0.0); std::vector<float> c(N, 0.0);
        tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        for (auto j = 0; j < N; j++)
        {
            for (auto k = 0; k < N; k++)
                a[j*N+k] = dist(engine);
            b[j] = dist(engine);
        }
        tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        matvec_Vector(c, a, b, N);
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        matvec(c.data(), a.data(), b.data(), N);
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        bool bResult = true; float sum = 0;
        for (auto j = 0; j < N; j++)
        {
            sum += a[j] * b[j];
        }
        if (fabs(c[0] - sum) > 0.0001)
        {
            bResult = false;
        }
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        std::cout << "10^" << i;
        for (auto& t : times) std::cout << "  " << std::scientific << std::right << std::setprecision(4 + 1) << 1.e-9 * t.count();
        std::cout << std::setw(6) << std::boolalpha << bResult << std::endl;
    }
    return 0;
}

int CPP_ublas_MV(int nStart, int nEnd)
{
    using Duration = std::chrono::nanoseconds;
    std::mt19937 engine(std::random_device{}()); // Mersenne twister MT19937
    std::uniform_real_distribution<float> dist(0.f, 1.f); //Values between 0 and 1
    std::vector<std::string> vs = { "Size","Allocation","Initialize","Cal_prod","Verifying","Verified" };
    std::cout << std::setw(6);
    for (auto& s : vs) std::cout << std::left << s << std::setw(13);
    std::cout << std::setw(0) << std::endl;
    for (auto i = nStart; i <= nEnd; i++)
    {
        std::vector<Duration> times;
        size_t N = (size_t)pow(10, i);
        auto tNow = std::chrono::high_resolution_clock::now(); auto tPre = tNow;
        boost::numeric::ublas::matrix<float> a(N, N);
        boost::numeric::ublas::vector<float> b(N);
        boost::numeric::ublas::vector<float> c(N);
        tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        for (auto j = 0; j < N; j++)
        {
            for (auto k = 0; k < N; k++)
                a(j,k) = dist(engine);
            b(j) = dist(engine);
        }
        tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        //boost::numeric::ublas::vector<float> c = boost::numeric::ublas::prod(a, b);
        c = boost::numeric::ublas::prod(a, b);
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        //c = a * b; // operator overloading
        //tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        //times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        bool bResult = true; float sum = 0;
        for (auto j = 0; j < N; j++)
        {
            sum += a(0,j) * b(j);
        }
        if (fabs(c(0) - sum) > 0.0001)
        {
            bResult = false;
        }
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        std::cout << "10^" << i;
        for (auto& t : times) std::cout << "  " << std::scientific << std::right << std::setprecision(4 + 1) << 1.e-9 * t.count();
        std::cout << std::setw(6) << std::boolalpha << bResult << std::endl;
    }
    return 0;
}

int cuda_MV(int nStart, int nEnd)
{
    using Duration = std::chrono::nanoseconds;
    std::mt19937 engine(std::random_device{}()); // Mersenne twister MT19937
    std::uniform_real_distribution<float> dist(0.f, 1.f); //Values between 0 and 1
    std::vector<std::string> vs = { "Size","Allocation","Initialize","cuda_mem&cpy","Cal_cuda","memcpy_back","cudablas","memcpy_back","verifying","memRelease","Verified" };
    std::cout << std::setw(6);
    for (auto& s : vs) std::cout << std::left << s << std::setw(13);
    std::cout << std::setw(0) << std::endl;
    cudaError_t err = cudaSuccess;
    //err = cudaSetDevice(0);
    for (auto i = nStart; i <= nEnd; i++)
    {
        std::vector<Duration> times;
        int N = (size_t)pow(10, i);
        auto tNow = std::chrono::high_resolution_clock::now(); auto tPre = tNow;
        float* h_a = (float*)malloc(N*N* sizeof(float));// memory allocation at host
        float* h_b = (float*)malloc(N * sizeof(float));
        float* h_c = (float*)malloc(N * sizeof(float));
        tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        for (auto j = 0; j < N; j++)
        {
            for (auto k = 0; k < N; k++)
                h_a[j * N + k] = dist(engine);
            h_b[j] = dist(engine);
            h_c[j] = 0;
        }
        tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        float* d_a, * d_b, * d_c;
        cudaMalloc((void**)&d_a, N * N * sizeof(float));
        cudaMalloc((void**)&d_b, N * sizeof(float));
        cudaMalloc((void**)&d_c, N * sizeof(float));
        err = cudaMemcpy(d_a, h_a, N * N * sizeof(float), cudaMemcpyHostToDevice);
        err = cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);
        err = cudaMemcpy(d_c, h_c, N * sizeof(float), cudaMemcpyHostToDevice);
        cublasHandle_t handle;
        cublasCreate(&handle);
        tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        int threadsPerBlock = 384; // rtx3060, 3840 cores
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        cuda_matVec <<<blocksPerGrid, threadsPerBlock >>> (d_a, d_b, d_c, N);
        tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        err = cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        // Perform matrix-vector multiplication using cuBLAS
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasSgemv(handle, CUBLAS_OP_T, N, N, &alpha, d_a, N, d_b, 1, &beta, d_c, 1);
        tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        err = cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);
        tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        bool bResult = true; float sum = 0;
        for (auto j = 0; j < N; j++)
        {
            //std::cout << h_a[j] << " " << h_b[j] << std::endl;
            sum += h_a[0*N+j] * h_b[j];
        }
        
        if (fabs(h_c[0] - sum) > 0.0001)
        {
            std::cout << sum << " " << h_c[0] << " " << fabs(h_c[0] - sum) << std::endl;
            bResult = false;
        }
        //for (auto j = 0; j < N; j++)std::cout << h_c[j] << " ";
        //std::cout << std::endl;
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        cublasDestroy(handle);
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        std::cout << "10^" << i;
        for (auto& t : times) std::cout << "  " << std::scientific << std::right << std::setprecision(4 + 1) << 1.e-9 * t.count();
        std::cout << std::setw(6) << std::boolalpha << bResult << std::endl;
    }// could be false on verification because the tolerence is also larger with greater N
    return 0;
}

int CPP_Eigen_MV(int nStart, int nEnd)
{
    using Duration = std::chrono::nanoseconds;
    std::mt19937 engine(std::random_device{}()); // Mersenne twister MT19937
    std::uniform_real_distribution<float> dist(0.f, 1.f); //Values between 0 and 1
    std::vector<std::string> vs = { "Size","Allocation","Initialize","Cal_*","Verifying","Verified" };
    std::cout << std::setw(6);
    for (auto& s : vs) std::cout << std::left << s << std::setw(13);
    std::cout << std::setw(0) << std::endl;
    for (auto i = nStart; i <= nEnd; i++)
    {
        std::vector<Duration> times;
        size_t N = (size_t)pow(10, i);
        auto tNow = std::chrono::high_resolution_clock::now(); auto tPre = tNow;
        Eigen::MatrixXd a(N, N);
        Eigen::VectorXd b(N);
        Eigen::VectorXd c(N);
        tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre)); // 1
        for (auto j = 0; j < N; j++)
        {
            for (auto k = 0; k < N; k++)
                a << dist(engine);//a(j, k) = dist(engine);
            b << dist(engine);//b(j) = dist(engine);
        }
        tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre)); // 2
        c = a * b;
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre)); // 3
        bool bResult = true; float sum = 0;
        for (auto j = 0; j < N; j++)
        {
            sum += a(0, j) * b(j);
        }
        if (fabs(c(0) - sum) > 0.0001)
        {
            bResult = false;
        }
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre)); // 4
        std::cout << "10^" << i;
        for (auto& t : times) std::cout << "  " << std::scientific << std::right << std::setprecision(4 + 1) << 1.e-9 * t.count();
        std::cout << std::setw(6) << std::boolalpha << bResult << std::endl;
    }
    return 0;
}


int CPP_Arma_MV(int nStart, int nEnd)
{   // C:\installed\armadillo-12.8.0\examples\lib_win64

    using Duration = std::chrono::nanoseconds;
    std::mt19937 engine(std::random_device{}()); // Mersenne twister MT19937
    std::uniform_real_distribution<float> dist(0.f, 1.f); //Values between 0 and 1
    std::vector<std::string> vs = { "Size","Alloc/w Rand","Cal_*","Verifying","Verified" };
    std::cout << std::setw(6);
    for (auto& s : vs) std::cout << std::left << s << std::setw(13);
    std::cout << std::setw(0) << std::endl;
    for (auto i = nStart; i <= nEnd; i++)
    {
        std::vector<Duration> times;
        size_t N = (size_t)pow(10, i);
        auto tNow = std::chrono::high_resolution_clock::now(); auto tPre = tNow;
        arma::arma_rng::set_seed(0);
        arma::mat a = arma::randu(N, N); // Generates uniform random numbers
        arma::vec b = arma::randu(N);
        arma::vec c(N);
        tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre)); // allocation with randu
        //for (auto j = 0; j < N; j++)
        //{
        //    for (auto k = 0; k < N; k++)
        //        a(j, k) = dist(engine);
        //    b(j) = dist(engine);
        //}
        //tNow = std::chrono::high_resolution_clock::now();
        //times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        c = a * b;
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre)); // cal
        //c = a * b; // operator overloading
        //tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        //times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        bool bResult = true; float sum = 0;
        for (auto j = 0; j < N; j++)
        {
            sum += a(0, j) * b(j);
        }
        if (fabs(c(0) - sum) > 0.0001)
        {
            bResult = false;
        }
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        std::cout << "10^" << i;
        for (auto& t : times) std::cout << "  " << std::scientific << std::right << std::setprecision(4 + 1) << 1.e-9 * t.count();
        std::cout << std::setw(6) << std::boolalpha << bResult << std::endl;
    }
    return 0;
}


//////////////////////////////////////////////////////////////////////
// MM
////////////////////

int Old_C_Style_MM(int nStart, int nEnd)
{
    using Duration = std::chrono::nanoseconds;
    std::mt19937 engine(std::random_device{}()); // Mersenne twister MT19937
    std::uniform_real_distribution<float> dist(0.f, 1.f); //Values between 0 and 1
    std::vector<std::string> vs = { "Size","Allocation","Initialize","Calculation","Verifying","ReleaseMem","Verified" };
    std::cout << std::setw(6);
    for (auto& s : vs) std::cout << std::left << s << std::setw(13);
    std::cout << std::setw(0) << std::endl;
    for (auto i = nStart; i <= nEnd; i++)
    {
        std::vector<Duration> times;
        size_t N = (size_t)pow(10, i);
        auto tNow = std::chrono::high_resolution_clock::now(); auto tPre = tNow;
        float* a = (float*)malloc(N * N * sizeof(float));
        float* b = (float*)malloc(N * N * sizeof(float));
        float* c = (float*)malloc(N * N * sizeof(float));
        tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        for (auto j = 0; j < N; j++)
        {
            for (auto k = 0; k < N; k++)
            {
                a[j] = dist(engine); b[j] = dist(engine);
            }
        }
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        matmat(c, a, b, N);
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        bool bResult = true; float sum = 0;
        for (auto j = 0; j < N; j++)
        {
            sum += a[j*N] * b[j];
        }
        if (fabs(c[0] - sum) > 0.0001)
        {
            bResult = false;
        }
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        free(a); free(b); free(c);
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        std::cout << "10^" << i;
        for (auto& t : times) std::cout << "  " << std::scientific << std::right << std::setprecision(4 + 1) << 1.e-9 * t.count();
        std::cout << std::setw(6) << std::boolalpha << bResult << std::endl;
    }
    return 0;
}
int Old_CPP_Style_MM(int nStart, int nEnd)
{
    using Duration = std::chrono::nanoseconds;
    std::mt19937 engine(std::random_device{}()); // Mersenne twister MT19937
    std::uniform_real_distribution<float> dist(0.f, 1.f); //Values between 0 and 1
    std::vector<std::string> vs = { "Size","Allocation","Initialize","Calculation","Verifying","ReleaseMem","Verified" };
    std::cout << std::setw(6);
    for (auto& s : vs) std::cout << std::left << s << std::setw(13);
    std::cout << std::setw(0) << std::endl;
    for (auto i = nStart; i <= nEnd; i++)
    {
        std::vector<Duration> times;
        size_t N = (size_t)pow(10, i);
        auto tNow = std::chrono::high_resolution_clock::now(); auto tPre = tNow;
        float* a = new float[N * N]; float* b = new float[N*N]; float* c = new float[N*N];
        tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        for (auto j = 0; j < N; j++)
        {
            for (auto k = 0; k < N; k++)
            {
                a[j] = dist(engine); b[j] = dist(engine);
            }
        }
        tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        matmat(c, a, b, N);
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        bool bResult = true; float sum = 0;
        for (auto j = 0; j < N; j++)
        {
            sum += a[j * N] * b[j];
        }
        if (fabs(c[0] - sum) > 0.0001)
        {
            bResult = false;
        }
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        delete a; delete b; delete c;

        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        std::cout << "10^" << i;
        for (auto& t : times) std::cout << "  " << std::scientific << std::right << std::setprecision(4 + 1) << 1.e-9 * t.count();
        std::cout << std::setw(6) << std::boolalpha << bResult << std::endl;
    }
    return 0;
}

int CPP_Style_MM(int nStart, int nEnd)
{
    using Duration = std::chrono::nanoseconds;
    std::mt19937 engine(std::random_device{}()); // Mersenne twister MT19937
    std::uniform_real_distribution<float> dist(0.f, 1.f); //Values between 0 and 1
    std::vector<std::string> vs = { "Size","Allocation","Initialize","Calculation","Verifying","Verified" };
    std::cout << std::setw(6);
    for (auto& s : vs) std::cout << std::left << s << std::setw(13);
    std::cout << std::setw(0) << std::endl;
    for (auto i = nStart; i <= nEnd; i++)
    {
        std::vector<Duration> times;
        size_t N = (size_t)pow(10, i);
        auto tNow = std::chrono::high_resolution_clock::now(); auto tPre = tNow;
        auto a = std::make_unique<float[]>(N * N);
        auto b = std::make_unique<float[]>(N*N);
        auto c = std::make_unique<float[]>(N*N);
        tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        for (auto j = 0; j < N; j++)
        {
            for (auto k = 0; k < N; k++)
            {
                a[j] = dist(engine); b[j] = dist(engine);
            }
        }
        tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        matmat(c.get(), a.get(), b.get(), N); // using smart pointers in NOT smart way.
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        bool bResult = true; float sum = 0;
        for (auto j = 0; j < N; j++)
        {
            sum += a[j * N] * b[j];
        }
        if (fabs(c[0] - sum) > 0.0001)
        {
            bResult = false;
        }
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        std::cout << "10^" << i;
        for (auto& t : times) std::cout << "  " << std::scientific << std::right << std::setprecision(4 + 1) << 1.e-9 * t.count();
        std::cout << std::setw(6) << std::boolalpha << bResult << std::endl;
    }
    return 0;
}

int CPP_Vector_MM(int nStart, int nEnd)
{
    using Duration = std::chrono::nanoseconds;
    std::mt19937 engine(std::random_device{}()); // Mersenne twister MT19937
    std::uniform_real_distribution<float> dist(0.f, 1.f); //Values between 0 and 1
    std::vector<std::string> vs = { "Size","Allocation","Initialize","Cal_vector","Cal_ptr","Verifying","Verified" };
    std::cout << std::setw(6);
    for (auto& s : vs) std::cout << std::left << s << std::setw(13);
    std::cout << std::setw(0) << std::endl;
    for (auto i = nStart; i <= nEnd; i++)
    {
        std::vector<Duration> times;
        size_t N = (size_t)pow(10, i);
        auto tNow = std::chrono::high_resolution_clock::now(); auto tPre = tNow;
        std::vector<float> a(N * N, 0.0); std::vector<float> b(N*N, 0.0); std::vector<float> c(N*N, 0.0);
        tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        for (auto j = 0; j < N; j++)
        {
            for (auto k = 0; k < N; k++)
            {
                a[j] = dist(engine); b[j] = dist(engine);
            }
        }
        tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        matmat_Vector(c, a, b, N);
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        matmat(c.data(), a.data(), b.data(), N);
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        bool bResult = true; float sum = 0;
        for (auto j = 0; j < N; j++)
        {
            sum += a[j * N] * b[j];
        }
        if (fabs(c[0] - sum) > 0.0001)
        {
            bResult = false;
        }
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        std::cout << "10^" << i;
        for (auto& t : times) std::cout << "  " << std::scientific << std::right << std::setprecision(4 + 1) << 1.e-9 * t.count();
        std::cout << std::setw(6) << std::boolalpha << bResult << std::endl;
    }
    return 0;
}

int CPP_ublas_MM(int nStart, int nEnd)
{
    using Duration = std::chrono::nanoseconds;
    std::mt19937 engine(std::random_device{}()); // Mersenne twister MT19937
    std::uniform_real_distribution<float> dist(0.f, 1.f); //Values between 0 and 1
    std::vector<std::string> vs = { "Size","Allocation","Initialize","Cal_prod","Verifying","Verified" };
    std::cout << std::setw(6);
    for (auto& s : vs) std::cout << std::left << s << std::setw(13);
    std::cout << std::setw(0) << std::endl;
    for (auto i = nStart; i <= nEnd; i++)
    {
        std::vector<Duration> times;
        size_t N = (size_t)pow(10, i);
        auto tNow = std::chrono::high_resolution_clock::now(); auto tPre = tNow;
        boost::numeric::ublas::matrix<float> a(N,N);
        boost::numeric::ublas::matrix<float> b(N,N);
        boost::numeric::ublas::matrix<float> c(N,N);
        tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        for (auto j = 0; j < N; j++)
        {
            for (auto k = 0; k < N; k++)
            {
                a(j,k) = dist(engine); 
                b(j,k) = dist(engine);
            }
        }
        tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        //boost::numeric::ublas::vector<float> c = boost::numeric::ublas::prod(a, b);
        c = boost::numeric::ublas::prod(a, b);
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        //c = a * b; // operator overloading
        //tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        //times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        bool bResult = true; float sum = 0;
        for (auto j = 0; j < N; j++)
        {
            sum += a(0,j) * b(j,0);
        }
        if (fabs(c(0,0) - sum) > 0.0001)
        {
            bResult = false;
        }
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        std::cout << "10^" << i;
        for (auto& t : times) std::cout << "  " << std::scientific << std::right << std::setprecision(4 + 1) << 1.e-9 * t.count();
        std::cout << std::setw(6) << std::boolalpha << bResult << std::endl;
    }
    return 0;
}

int cuda_MM(int nStart, int nEnd)
{
    using Duration = std::chrono::nanoseconds;
    std::mt19937 engine(std::random_device{}()); // Mersenne twister MT19937
    std::uniform_real_distribution<float> dist(0.f, 1.f); //Values between 0 and 1
    std::vector<std::string> vs = { "Size","Allocation","Initialize","cuda_mem&cpy","Cal_cuda","memcpy_back","cudablas","memcpy_back","verifying","memRelease","Verified" };
    std::cout << std::setw(6);
    for (auto& s : vs) std::cout << std::left << s << std::setw(13);
    std::cout << std::setw(0) << std::endl;
    cudaError_t err = cudaSuccess;
    //err = cudaSetDevice(0);
    for (auto i = nStart; i <= nEnd; i++)
    {
        std::vector<Duration> times;
        int N = (size_t)pow(10, i);
        auto tNow = std::chrono::high_resolution_clock::now(); auto tPre = tNow;
        float* h_a = (float*)malloc(N * N * sizeof(float));// memory allocation at host
        float* h_b = (float*)malloc(N * N*  sizeof(float));
        float* h_c = (float*)malloc(N * N * sizeof(float));
        tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        for (auto j = 0; j < N; j++)
        {
            for (auto k = 0; k < N; k++)
            {
                h_a[j * N + k] = dist(engine);
                h_b[j * N + k] = dist(engine);
            }
            h_c[j] = 0;
        }
        tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        float* d_a, * d_b, * d_c;
        cudaMalloc((void**)&d_a, N * N * sizeof(float));
        cudaMalloc((void**)&d_b, N * N * sizeof(float));
        cudaMalloc((void**)&d_c, N * N * sizeof(float));
        err = cudaMemcpy(d_a, h_a, N * N * sizeof(float), cudaMemcpyHostToDevice);
        err = cudaMemcpy(d_b, h_b, N * N * sizeof(float), cudaMemcpyHostToDevice);
        err = cudaMemcpy(d_c, h_c, N * N * sizeof(float), cudaMemcpyHostToDevice);
        cublasHandle_t handle;
        cublasCreate(&handle);
        tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        //int threadsPerBlock = 384; // rtx3060, 3840 cores
        //int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        // Threads per block
        dim3 threadsPerBlock(16, 16);
        // Blocks per grid
        dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
        cuda_matmat << <blocksPerGrid, threadsPerBlock >> > (d_a, d_b, d_c, N,N,N);
        tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        err = cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        // Perform matrix-vector multiplication using cuBLAS
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasSgemv(handle, CUBLAS_OP_T, N, N, &alpha, d_a, N, d_b, 1, &beta, d_c, 1);
        tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        err = cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);
        tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        bool bResult = true; float sum = 0;
        for (auto j = 0; j < N; j++)
        {
            //std::cout << h_a[j] << " " << h_b[j] << std::endl;
            sum += h_a[0 * N + j] * h_b[j];
        }

        if (fabs(h_c[0] - sum) > 0.0001)
        {
            std::cout << sum << " " << h_c[0] << " " << fabs(h_c[0] - sum) << std::endl;
            bResult = false;
        }
        //for (auto j = 0; j < N; j++)std::cout << h_c[j] << " ";
        //std::cout << std::endl;
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        cublasDestroy(handle);
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        std::cout << "10^" << i;
        for (auto& t : times) std::cout << "  " << std::scientific << std::right << std::setprecision(4 + 1) << 1.e-9 * t.count();
        std::cout << std::setw(6) << std::boolalpha << bResult << std::endl;
    }// could be false on verification because the tolerence is also larger with greater N
    return 0;
}

int CPP_Eigen_MM(int nStart, int nEnd)
{
    using Duration = std::chrono::nanoseconds;
    std::mt19937 engine(std::random_device{}()); // Mersenne twister MT19937
    std::uniform_real_distribution<float> dist(0.f, 1.f); //Values between 0 and 1
    std::vector<std::string> vs = { "Size","Allocation","Initialize","Cal_*","Verifying","Verified" };
    std::cout << std::setw(6);
    for (auto& s : vs) std::cout << std::left << s << std::setw(13);
    std::cout << std::setw(0) << std::endl;
    for (auto i = nStart; i <= nEnd; i++)
    {
        std::vector<Duration> times;
        size_t N = (size_t)pow(10, i);
        auto tNow = std::chrono::high_resolution_clock::now(); auto tPre = tNow;
        Eigen::MatrixXd a(N, N);
        Eigen::MatrixXd b(N ,N);
        Eigen::MatrixXd c(N, N);
        tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre)); // 1
        for (auto j = 0; j < N; j++)
        {
            for (auto k = 0; k < N; k++)
            {
                a << dist(engine);
                b << dist(engine);
            }
        }
        tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre)); // 2
        c = a * b;
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre)); // 3
        bool bResult = true; float sum = 0;
        for (auto j = 0; j < N; j++)
        {
            sum += a(0, j) * b(j,0);
        }
        if (fabs(c(0,0) - sum) > 0.0001)
        {
            bResult = false;
        }
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre)); // 4
        std::cout << "10^" << i;
        for (auto& t : times) std::cout << "  " << std::scientific << std::right << std::setprecision(4 + 1) << 1.e-9 * t.count();
        std::cout << std::setw(6) << std::boolalpha << bResult << std::endl;
    }
    return 0;
}


int CPP_Arma_MM(int nStart, int nEnd)
{   // C:\installed\armadillo-12.8.0\examples\lib_win64

    using Duration = std::chrono::nanoseconds;
    std::mt19937 engine(std::random_device{}()); // Mersenne twister MT19937
    std::uniform_real_distribution<float> dist(0.f, 1.f); //Values between 0 and 1
    std::vector<std::string> vs = { "Size","Alloc/w Rand","Cal_*","Verifying","Verified" };
    std::cout << std::setw(6);
    for (auto& s : vs) std::cout << std::left << s << std::setw(13);
    std::cout << std::setw(0) << std::endl;
    for (auto i = nStart; i <= nEnd; i++)
    {
        std::vector<Duration> times;
        size_t N = (size_t)pow(10, i);
        auto tNow = std::chrono::high_resolution_clock::now(); auto tPre = tNow;
        arma::arma_rng::set_seed(0);
        arma::mat a = arma::randu(N, N); // Generates uniform random numbers
        arma::mat b = arma::randu(N, N);
        arma::mat c(N,N);
        tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre)); // allocation with randu
        c = a * b;
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre)); // cal
        bool bResult = true; float sum = 0;
        for (auto j = 0; j < N; j++)
        {
            sum += a(0, j) * b(j,0);
        }
        if (fabs(c(0,0) - sum) > 0.0001)
        {
            bResult = false;
        }
        tPre = tNow; tNow = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<Duration>(tNow - tPre));
        std::cout << "10^" << i;
        for (auto& t : times) std::cout << "  " << std::scientific << std::right << std::setprecision(4 + 1) << 1.e-9 * t.count();
        std::cout << std::setw(6) << std::boolalpha << bResult << std::endl;
    }
    return 0;
}


void vector_addition(int nStart, int nEnd)
{
    std::cout << "C Style" << std::endl;
    Old_C_Style(nStart, nEnd);
    std::cout << "C/C++ Style" << std::endl;
    Old_CPP_Style(nStart, nEnd);
    std::cout << "C++ Style" << std::endl;
    CPP_Style(nStart, nEnd);
    std::cout << "C++ Vector " << std::endl;
    CPP_Vector(nStart, nEnd);
    std::cout << "boost ublas" << std::endl;
    CPP_Vector_ublas(nStart, nEnd);
    std::cout << "Cuda C Style" << std::endl;
    Cuda_C_Style(nStart, nEnd);
}

void matvec(int nStart, int nEnd)
{
    std::cout << "C Style" << std::endl;
    Old_C_Style_MV(nStart, nEnd);
    std::cout << "C/C++ Style" << std::endl;
    Old_CPP_Style_MV(nStart, nEnd);
    std::cout << "C++ Style" << std::endl;
    CPP_Style_MV(nStart, nEnd);
    std::cout << "C++ Vector " << std::endl;
    CPP_Vector_MV(nStart, nEnd);
    std::cout << "boost ublas" << std::endl;
    CPP_ublas_MV(nStart, nEnd);
    std::cout << "Cuda C Style" << std::endl;
    cuda_MV(nStart, nEnd);
    std::cout << "Eigen" << std::endl;
    CPP_Eigen_MV(nStart, nEnd);
    std::cout << "Armadillo" << std::endl;
    CPP_Arma_MV(nStart, nEnd);
}

void matmat(int nStart, int nEnd)
{
    std::cout << "C Style" << std::endl;
    Old_C_Style_MM(nStart, nEnd);
    std::cout << "C/C++ Style" << std::endl;
    Old_CPP_Style_MM(nStart, nEnd);
    std::cout << "C++ Style" << std::endl;
    CPP_Style_MM(nStart, nEnd);
    std::cout << "C++ Vector " << std::endl;
    CPP_Vector_MM(nStart, nEnd);
    std::cout << "boost ublas" << std::endl;
    CPP_ublas_MM(nStart, nEnd);
    std::cout << "Cuda C Style" << std::endl;
    cuda_MM(nStart, nEnd);
    std::cout << "Eigen" << std::endl;
    CPP_Eigen_MV(nStart, nEnd);
    std::cout << "Armadillo" << std::endl;
    CPP_Arma_MV(nStart, nEnd);
}

int main()
{
    // Vector Addition
    //vector_addition(4, 8);
    
    // Matrix - Vector Multiplication
    //matvec(2,4); // not 5, crashed 

    // Matrix - Matrix Multiplication
    matmat(1, 4);

    return 0;
}

