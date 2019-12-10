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
#include <random>

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

template <typename T>
T dBToPower(T dB){ return std::pow(10, dB/10); }

template <typename T>
T powerTodB(T ratio){ return 10*std::log10(ratio); }

// Bit twiddling hack from:
// https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
int nextPowerOfTwo(int v){ 
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

float complexAbs(Complex cplx) { return std::sqrt(cplx.x*cplx.x + cplx.y*cplx.y); }
float complexPhase(Complex cplx) {
    if(cplx.x==0 && cplx.y>=0) return PI/2;
    if(cplx.x==0 && cplx.y<0) return -PI/2;
    return std::atan(cplx.y / cplx.x);
}

int main()
{
    // 
    // Radar parameters from POMR Section 2.12
    const float Pt = 150;      // Transmit peak power (kW)
    const float f0  = 9.4e9;   // Signal frequency (Hz)
    const float tau = 1.2e-6;  // Pulse width (s)
    const float PRF = 2e3;     // Pulse repetition frequency (Hz)
    const float Da = 2.4;      // Antenna diameter (m)
    const float eta_a = 0.6;   // Antenna efficiency
    const float CPI = 16e-3;   // coherent processing interval / dwell time (s)
                               // Adjusted from 18.3 to get a power of 2 number of dwells
    const float Lt = 3.1;      // Transmit loss (dB)
    const float Lr = 2.4;      // Receive loss (dB)
    const float Lsp = 3.2;     // Signal processing loss (dB)
    const float La_per_km = 0.16;  // Atmospheric loss per km (dB/km)
    const float RCS = -10;       // Target RCS (dBsm)
    const float R0 = 25;       // Initial target range (km)
    const float vt = 120e-3;   // Target velocity (km/s)

    // Derived parameters
    const float lambda = C/f0; // Signal wavelength (km)
    const float PRI = 1.0/PRF; // Pulse repetition interval (s)
    const float Ae = PI * std::pow(Da/2, 2) * eta_a; // Antenna effective area (m^2)


    //
    // Transmitted pulse for display purposes (Eq. 8.25)
    std::cout << "Generating transmitted pulse..." << std::endl;
    const float fs_tx = 2*f0; // Nyquist rate (Hz)
    const float Ts_tx = 1/fs_tx;
    const int num_samples_tx = tau/Ts_tx;
    float *pulse_tx = new float[num_samples_tx];
    int signal_bytes = sizeof(float) * num_samples_tx;
    for(int i = 0; i < num_samples_tx; ++i)
    {
        pulse_tx[i] = std::cos(2*PI*f0*i*Ts_tx);
    }
    delete[] pulse_tx;

    const int num_pulses = CPI/PRI;
    const int num_range_bins = PRI/tau;
    std::cout << "Generating fast time slow time matrix...\n";
    std::cout << "Range bins: " << num_range_bins << "\n";
    std::cout << "Pulses: " << num_pulses << std::endl;

    Complex **data_matrix = new Complex*[num_pulses];
    for(int i = 0; i < num_pulses; ++i)
    {
        data_matrix[i] = new Complex[num_range_bins];
    }

    std::fstream data_matrix_file;
    data_matrix_file.open("data-matrix.txt",std::ios::out);
    if(!data_matrix_file)
    {
        std::cout<<"Error creating data matrix file"<< std::endl;
        return 0;
    }

    std::random_device rd;
    std::mt19937 gen{rd()};
    std::normal_distribution<float> dist{0.0, 0.1};

    for(int pulse = 0; pulse < num_pulses; ++pulse)
    {
        const float R = R0 + vt*(pulse*PRI);
        const float echo_start_time = 2*R/C;
        
        //
        // Radar range equation w/o noise (Eq. 2.17)
        const float Ls = dBToPower( Lt + La_per_km*2*R + Lr + Lsp ); // Two-way system loss (dB)
        float A_prime = (Pt * dBToPower(RCS) * Ae) / 
                        (std::pow(4*PI, 3) * std::pow(R, 4) * Ls);

        for(int i = 0; i < num_range_bins; ++i)
        {
            //
            // Received pulse with doppler phase shift (Eq. 8.26)
            const float t = i*tau - echo_start_time; // Use pulse arrival time as reference time
            // Evaluates to 0 or 1, using to avoid if statement in kernel
            const bool in_pulse = t >= 0 && t <= tau; // Inside pulse arrival window
            float signal_rx = A_prime * std::cos(2*PI*f0*i*tau - 4*PI*R/lambda) * in_pulse;

            // Add some Gaussian noise
            signal_rx += A_prime*dist(gen);

            //
            // Analytic signal from I/Q channels (Eq. 8.27)
            Complex signal_analytic;
            signal_analytic.x = signal_rx * 2*std::cos(2*PI*f0*i*tau);
            signal_analytic.y = signal_rx * 2*std::sin(2*PI*f0*i*tau);

            //
            // Populate fast-time slow-time matrix
            data_matrix[pulse][i] = signal_analytic;
            data_matrix_file << complexAbs(signal_analytic) << " ";
        }
        data_matrix_file << "\n";
    }
    data_matrix_file.close();

    //
    // FFT along each slow-time row
    std::fstream range_doppler_file;
    range_doppler_file.open("range-doppler.txt",std::ios::out);
    if(!range_doppler_file)
    {
        std::cout << "Error creating range-doppler file"<< std::endl;
        return 0;
    }

    std::cout << "Performing FFT on slow-time sequences..." << std::endl;
    const int fft_size = nextPowerOfTwo(num_pulses);

    // CUFFT plan
    cufftHandle plan;
    cufftPlan1d(&plan, fft_size, CUFFT_C2C, 1);

    Complex complex_zero; complex_zero.x=0; complex_zero.y=0;
    for(int range_bin = 0; range_bin < num_range_bins; ++range_bin)
    {
        Complex *slow_time_data = new Complex[fft_size];
        
        for(int pulse = 0; pulse < fft_size; ++pulse)
        {
            // Zero pad if needed
            slow_time_data[pulse] = (pulse < num_pulses) ? data_matrix[pulse][range_bin] : complex_zero;
        }

        cufftComplex *d_slow_time_data;
        int mem_size = sizeof(Complex)*fft_size;
        checkCudaErrors(cudaMalloc((void **) &d_slow_time_data, mem_size)); 
        checkCudaErrors(cudaMemcpy(d_slow_time_data, slow_time_data, mem_size, cudaMemcpyHostToDevice));

        // Transform slow time data
        cufftExecC2C(plan, (cufftComplex *)d_slow_time_data, (cufftComplex *)d_slow_time_data, CUFFT_FORWARD);

        // Retrieve range-doppler matrix row
        Complex *fft_data = new Complex[fft_size];
        cudaMemcpy(fft_data, d_slow_time_data, mem_size, cudaMemcpyDeviceToHost);

        for (int i = 0; i < fft_size; ++i)
        {
            range_doppler_file << complexAbs(fft_data[i]) << " ";
        }
        range_doppler_file << std::endl;

        delete[] slow_time_data;
        delete[] fft_data;
        cudaFree(d_slow_time_data);
    }
    range_doppler_file.close();

    std::cout << "Deleting data matrix..." << std::endl;
    for(int i = 0; i < num_pulses; ++i)
    {
        delete[] data_matrix[i];
    }
    delete[] data_matrix;
    std::cout << "Destroying plan..." << std::endl;
    cufftDestroy(plan);
    std::cout << "Success!" << std::endl;
}
