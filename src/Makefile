# Copyright (c) 1993-2015, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# This Makefile was modified for use in the compiling of ARC by Blake Teixeira Hartley.

#CUDA_INSTALL_PATH = $(CUDA_PATH)

# bluecrab variables (not sure what is best here):
MPI_HOME = /opt/openmpi-gcc/1.6.5
#MPI_HOME = /cell_root/software/openmpi/1.8.6/gnu/4.8.1/threaded/sys
#MPI_HOME = /cm/shared/apps/Intel/openmpi/1.8.6

CUDA_PATH = /usr/local/cuda
#CUDA_PATH = /cell_root/software/cuda/7.5.18/sys
#CUDA_PATH = /cm/shared/apps/cuda/7.5

CUDA_INCDIR = $(CUDA_PATH)/include
CUDA_LIB64DIR = $(CUDA_PATH)/lib64

#CUDA_PATH = $(PATH)
#CIDA_INCDIR = $(INCLUDE)
#CIDA_LIB64DIR = $(LD_LIBRARY_PATH)

# Compilers
MPICC=$(PREP) mpicc
MPILD=$(PREP) mpic++
#NVCC=$(PREP) nvcc
NVCC=$(PREP) /usr/local/cuda/bin/nvcc

# Flags
CFLAGS = -std=c99 -O3 -march=native -Wall
CFLAGS += -I$(CUDA_INCDIR)
#CFLAGS += -L/etc/glue/nvidia/lib64 -Wl,-rpath,/etc/glue/nvidia/lib64 -lcudart

LDFLAGS = -L$(CUDA_LIB64DIR) -Wl,-rpath,$(CUDA_LIB64DIR) -lcudart
#LDFLAGS += -L/etc/glue/nvidia/lib64 -Wl,-rpath,/etc/glue/nvidia/lib64

MPICFLAGS=-I${MPI_HOME}/include
CUDACFLAGS=-I${CUDA_INCDIR}

#CUDACFLAGS=-I${CUDA_INSTALL_PATH}/include

GENCODE_SM13    := -gencode arch=compute_13,code=sm_13
GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30
GENCODE_SM35    := -gencode arch=compute_35,code=sm_35
GENCODE_SM52    := -gencode arch=compute_52,code=\"sm_52,compute_52\"
GENCODE_FLAGS   := $(GENCODE_SM35) $(GENCODE_SM37) $(GENCODE_SM50) $(GENCODE_SM52)

NVCCFLAGS=-O3 $(GENCODE_FLAGS) -Xcompiler -march=native

CUDALDFLAGS := -L${CUDA_INCDIR}/lib64 -lcudart

# Description of binaries
BINDIR=../bin
ARC=$(BINDIR)/arc
BINARIES=$(ARC)

# Commands
all: $(BINARIES)

test:
	@echo $(MPI_HOME)
	@echo $(CUDA_INCDIR)
	@echo $(CUDA_LIB64DIR)
	@echo $(CFLAGS)
	@echo $(LDFLAGS)

dev.o: arc.h arc_dev.cu Makefile
	$(NVCC) $(MPICFLAGS) $(NVCCFLAGS) -c arc_dev.cu -o dev.o

host.o: arc.h arc_host.cpp Makefile
	$(MPILD) $(LDFLAGS) $(CFLAGS) -c arc_host.cpp -o host.o
	
$(ARC): dev.o host.o Makefile
	mkdir -p $(BINDIR)
	$(MPILD) $(LDFLAGS) -o $(ARC) dev.o host.o 

doc: CUDA_Normal_MPI.c CUDA_Aware_MPI.c Device.cu Host.c Input.c Makefile Jacobi.c Jacobi.h Jacobi.doxygen
	doxygen Jacobi.doxygen
	
clean:
	rm -rf *.o *~ $(BINARIES)

cleandat:
	rm -rf ../dat/hydrogen/* ../dat/helium/* ../dat/photons/*
