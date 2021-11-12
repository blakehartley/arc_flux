# ARC

0. About

Adaptive Ray-tracing with CUDA (ARC) is an Astronomy module designed to efficiently track radiation in a cosmological volume and its effects on the ionization state of the gas using the power of CUDA to take advantage of the power of multi-level parallelization (multiple parallel GPUs, each with thousands of parallel cores). The version of the code presented here is designed to track a cube of fixed sized over any given period of cosmic time on a fixed grid with size 2^n. (Current version tracks (10 Mpc)^3 starting at z=30 with PLANCK parameters, but will soon be updated to allow freedom of choice for these parameters).

In depth code description and accuracy tests are presented in full detail here:

ARC: adaptive ray-tracing with CUDA, a new ray tracing code for parallel GPUs
https://arxiv.org/pdf/1807.07094.pdf

1. How to configure ARC

Several parameters are required before the code is compiled, such as the size of the fixed grid on which the code is to run. TODO These parameters need to be entered in the './configure' script at the top.

This code is run using MPI and CUDA, so make sure that openMPI and CUDA are both loaded, with both versions being mutually compatible (older versions of MPI are not CUDA-aware; this feature is required from the compiling and running of this code, as device arrays are shared via MPI).

Once the necessary parameters have been entered and modules are loaded, run './configure' to set up the Makefile for compiling the code.

2. How to compile ARC

A Makefile is used for the managing the directory structure and compiling of the code. The following codes may be used as desired once the code has been compiled:

'makedir' - This command needs to be run before the code can be compiled and run. It produces the necessary file structure 

'make' - This will compile the code, creating the necessary object files and binary, which is lockated in the directory './bin/'.

'cleandat' - This will clean out the data folders entirely. It is ideal for cleaning up unnecessary work.

3. How to run the code

Before the code may be run, the file 'parameters.txt', located in the same directory as this file, needs to be edited to suit the user's needs. Each parameter is given its own line, which a commented out description of what that parameter represents.

The current version of the code is designed to be run on a multiple of 8 parallel GPUs, using MPI and CUDA. The code is run with a command such as the following:

mpirun -n 8x ./bin/arc

where x is some integer, as desired. An example of the type of SLURM script used for running the code is included in './run.csh'. This script is one that I have used to run it on both UMD's Deepthought2 and JHU's Bluecrab. It is possible to run the code on a smaller or larger number, or non-multiple of 8, and MPI will take care of over/underloading GPU's as necessary. However, an integer multiple of 8 will ensure optimal saturation of computational resources (i.e. at 15 nodes, the final 16th sub-volume will take its own computational cycle, making the whole code hang up).

4. Code outputs

The outputs of the code are stored in the ./dat/ folder. The code produces the following outputs:

'./dat/sum.dat' - This file contains a summary of various parameters at each time step of the code. TODO: './dat/sum.info' contains a description of each of the parameter columns of sum.dat.

'./dat/hydrogen/' - 

'./dat/helium/' - 

'./dat/photons/' - 

'./dat/restart/' - 

5. Contact the author

I may be contacted at bth@astro.umd.edu for any questions regarding this code.

I wrote ARC from the ground up for my own use in my thesis work, and as such the code is self-contained and designed to work with the files I had at the outset of the project. However, I tried to keep the code as flexible and modular as possible, so it shouldn't be too difficult to apply it to different file types, or to utilize pieces of the code (the CUDA ray-tracing, CUDA ionization integrator, halo-matching algorithm, etc.) for different codes with comparable structure. CUDA is a uniquely powerful framework for these types of computations, and I believe that any large scale project with these types of calculations would benefit from the utilization of CUDA.
