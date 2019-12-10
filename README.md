# 605.617.FA19-CourseProject
## Summary
This program is a low/medium fidelity simulation of a signal being transmitted and
received in order to perform Range-Doppler processing and obtain a target's range
and velocity. 

The radar parameters used are drawn heavily from Principles of Modern Radar Section 
2.12 (A Graphical Example). Parts of the program were adapted from the course's Module 
8 code, namely the cuFFT invocation to be used to determine doppler.

## Compiling and Running
The code was tested using the Vocareum environment. Please make sure /usr/local/cuda/bin is 
in your path for nvcc to work. To compile, just run ```make``` from the top directory. This 
will generate the doppler binary in the ```bin/``` directory.

When you run ```doppler```, it will create 2 files in the run directory: data-matrix.txt 
and range-doppler.txt. These files can be processed with Microsoft Excel or Matlab to 
observe patterns in the data. The ```images/``` directory has images for the fast-time 
slow-time matrix and the resulting range-doppler "plot", both created with Excel.

## References:
  * Principles of Modern Radar Volume 1 - Richards, Scheer, Holm 
  * https://stackoverflow.com/questions/36889333/cuda-cufft-2d-example