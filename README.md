# 605.617.FA19-CourseProject
This program is a low/medium fidelity simulation of a signal being transmitted and
received in order to perform Range-Doppler processing and obtain a target's range
and velocity.
Moving most array related functions (like signal generation and modulation) is 
currently in the works. Further changes to the program's structure pending on 
code review feedback.
Parts of the program were adapted from the course's Module 8 code, namely the 
cuFFT invocation to be used to determine doppler.

References:
    - Principles of Modern Radar Volume 1 - Richards, Scheer, Holm
    - https://stackoverflow.com/questions/36889333/cuda-cufft-2d-example