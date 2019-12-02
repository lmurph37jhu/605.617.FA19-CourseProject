/*
Luis Murphy Marcos
JHU Engineering for Professionals
605.617 Introduction to GPU Programming
*/

// Standard Library includes
#include <iostream>
#include <fstream>
#include <ctime>
#include <cstdio>
#include <cmath>

// CUDA includes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cufft.h>

// Local includes
#include "helper_functions.h"
#include "helper_cuda.h"

typedef float2 Complex;
const double C = 299792.458; //km/s
const double PI = 3.1415926535897932384626433;

__global__ 
void ComplexMUL(Complex *a, Complex *b)
{
    int i = threadIdx.x;
    a[i].x = a[i].x * b[i].x - a[i].y*b[i].y;
    a[i].y = a[i].x * b[i].y + a[i].y*b[i].x;
}

/** POMR Eq. 20.50
 * TODO: make this a kernel
 * @param A   - peak amplitude 
 * @param f0  - center frequency
 * @param B   - waveform bandwidth
 * @param tau - pulse duration
 * @param t   - time (seconds)
 */
__host__ 
template <typename T>
T Chirp(T A, T f0, T B, T tau, T t)
{
    return A*cos( 2*PI*f0*t + PI*B/tau*t*t );
}

int main()
{
    // Test with values from POMR pg 75
    // TODO: replace with input params?
    const float f0 = 9.3e9; // signal initial frequency (Hz)
    const float B = 0.2e9; // signal bandwidth (Hz)
    const float tau = 1.2e-6; // pulse width (s)
    const float PRI = 500e-6; // pulse repetition interval (s)
    const float CPI = 18.3e-3; // Processing dwell time (s)

    // Generate pulses
    const float signal_fs = 2*(f0+B); // sample frequency, Nyquist
    const float signal_Ts = 1/signal_fs;
    const int num_signal_samples = tau/signal_Ts; //22800

    Complex *signal = new Complex[num_signal_samples];
    int signal_bytes = sizeof(Complex) * num_signal_samples;
    for(int i = 0; i < num_signal_samples; ++i)
    {
        fg[i].x = Chirp<float>(1.0, f0, B, tau, i*signal_Ts); 
        fg[i].y = 0;
    }

    // Modify amplitude according to radar range equation

    // Modify phase according to Eq. 8.26 and Eq. 8.28

    // Get in-phase and quadrature components for analytic signal
    
    // Populate fast-time slow-time matrix

    // FFT along each slow-time row

    /*
    cufftComplex *d_signal;
    checkCudaErrors(cudaMalloc((void **) &d_signal, mem_size)); 
    checkCudaErrors(cudaMemcpy(d_signal, fg, mem_size, cudaMemcpyHostToDevice));

    cufftComplex *d_filter_kernel;
    checkCudaErrors(cudaMalloc((void **)&d_filter_kernel, mem_size));
    checkCudaErrors(cudaMemcpy(d_filter_kernel, fig, mem_size, cudaMemcpyHostToDevice));

    // CUFFT plan
    cufftHandle plan;
    cufftPlan2d(&plan, N, N, CUFFT_C2C);

    // Transform signal and filter
    printf("Transforming signal cufftExecR2C\n");
    cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_FORWARD);

    printf("Launching Complex multiplication<<< >>>\n");
    ComplexMUL <<< 32, 256 >> >(d_signal, d_filter_kernel);

    // Transform signal back
    printf("Transforming signal back cufftExecC2C\n");
    cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_INVERSE);

    Complex *result = new Complex[SIZE];
    cudaMemcpy(result, d_signal, sizeof(Complex)*SIZE, cudaMemcpyDeviceToHost);

    for (int i = 0; i < SIZE; i=i+5)
    {
        std::cout << result[i].x << " " << result[i + 1].x << " " << result[i + 2].x << " " << result[i + 3].x << " " << result[i + 4].x << std::endl;
    }

    delete result, fg, fig;
    cufftDestroy(plan);
    cudaFree(d_signal);
    cudaFree(d_filter_kernel);
    */
}