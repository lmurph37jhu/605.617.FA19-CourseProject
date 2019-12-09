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

template <typename T>
T dBToPower(T dB){ return 10*std::log10(dB); }

template <typename T>
T powerTodB(T ratio){ return std::pow(10, ratio/10); }

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
    const float RCS = 0;       // Target RCS (dBsm)
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
            const bool out_of_pulse = t < 0 || t > tau; // Inside pulse arrival window
            float signal_rx = A_prime * std::cos(2*PI*f0*i*tau - 4*PI*R/lambda) * out_of_pulse;

            //
            // Analytic signal from I/Q channels (Eq. 8.27)
            Complex signal_analytic;
            signal_analytic.x = signal_rx * 2*std::cos(2*PI*f0*i*tau);
            signal_analytic.y = signal_rx * 2*std::sin(2*PI*f0*i*tau);

            //
            // Populate fast-time slow-time matrix
            data_matrix[pulse][i] = signal_analytic;
            float abs_value = std::sqrt(signal_analytic.x*signal_analytic.x + signal_analytic.y*signal_analytic.y);
            std::cout << abs_value << " ";
        }
        std::cout << "\n";
    }

    //
    // FFT along each slow-time row
    std::cout << "Performing FFT on slow-time sequences..." << std::endl;

    // CUFFT plan
    cufftHandle plan;
    cufftPlan1d(&plan, num_pulses, CUFFT_C2C, 1);

    for(int range_bin = 0; range_bin < num_range_bins; ++range_bin)
    {
        Complex *slow_time_data = new Complex[num_pulses];
        cufftComplex *fft_data = new cufftComplex[num_pulses];
        for(int pulse = 0; pulse < num_pulses; ++pulse)
        {
            slow_time_data[pulse] = data_matrix[range_bin][pulse];
        }

        cufftComplex *d_slow_time_data;
        int mem_size = sizeof(cufftComplex)*num_pulses;
        checkCudaErrors(cudaMalloc((void **) &d_slow_time_data, mem_size)); 
        checkCudaErrors(cudaMemcpy(d_slow_time_data, slow_time_data, mem_size, cudaMemcpyHostToDevice));

        // Transform slow time data and retrieve
        cufftExecC2C(plan, (cufftComplex *)d_slow_time_data, (cufftComplex *)d_slow_time_data, CUFFT_FORWARD);
        cudaMemcpy(fft_data, d_slow_time_data, mem_size, cudaMemcpyDeviceToHost);

        for (int i = 0; i < num_pulses; ++i)
        {
            float abs_value = std::sqrt(fft_data[i].x*fft_data[i].x + fft_data[i].y*fft_data[i].y);
            std::cout << abs_value << " ";
        }
        std::cout << std::endl;
        delete slow_time_data, fft_data;
        cudaFree(d_slow_time_data);
    }

    for(int i = 0; i < num_pulses; ++i)
    {
        delete[] data_matrix[i];
    }
    delete[] data_matrix;
    cufftDestroy(plan);
    std::cout << "Success!" << std::endl;
}