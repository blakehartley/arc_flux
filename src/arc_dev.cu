#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include <iostream>
#include <fstream>
using namespace std;

#include "./inc/domain.h"
#include "./inc/ray.h"
#include "arc.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "./arc_kernels.cu"

// GLOBAL VARIABLES FOR DISPLAY
int numThreads1, numThreads2; 
int elements;

// Error handling macro
/*#define CUDA_CHECK(call) \
	if((call) != cudaSuccess) { \
		cudaError_t err = cudaGetLastError(); \
		cerr << "CUDA error calling \""#call"\", code is " << err << endl; \
		exit(-1);}*/
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Perform calculations to produce Flux arrays
// DensArray	= Array of densities in cm^-3
// x_NArray		= Array of neutral fractions
// particles	= Array of source objects
// FluxArray	= Array of total ionizations, per baryon per second /1Mpc
// EArray		= Array of energies per baryon
// numParts		= Number of source objects
// L, a			= Length of the side of the box and scale factor
// dt_ion		= last ionization time step
void rad(	float* DensArray_dev, float* x_NArray_dev, const source* source_dev,
		float* FluxArray_dev, float* dEArray_dev, Ray* RayBuf, 
		int* PartInfo, int* ndBuf, float L, float a,
		float *dt, float* nfSback, Domain domain, float dt_ion)
{
	int dim = domain.get_dim();
	
	// JET
	srand(PartInfo[0]);
	// JET
	
	/*cout << "Ray class test area! Stand clear!" << endl;
	
	Ray *ray;
	int temp = 2;
	ray = new Ray[temp];
	int X, Y;
	(ray+1)->set_pix(123456,8);
	(ray+1)->get_pix(&X, &Y);
	cout << "size = " << X << "\tsize = " << Y << endl;
	cout << "size = " << sizeof(ray) << "\tsize = " << sizeof(ray[0]) << endl;
	(ray+1)->tau[0]+=1;
	cout << "tau = " << (ray+1)->tau[0] << endl;
	
	cout << "Ray class test area! Stand clear!" << endl;*/
	
	/*int nDevices;

	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++)
	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		printf("  Memory Clock Rate (KHz): %d\n",
			prop.memoryClockRate);
		printf("  Memory Bus Width (bits): %d\n",
			prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
			2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
	}*/
	
	cudaEvent_t timeA, timeB, timeC, timeD;
	CUDA_CHECK( cudaEventCreate(&timeA) );
	CUDA_CHECK( cudaEventCreate(&timeB) );
	CUDA_CHECK( cudaEventCreate(&timeC) );
	CUDA_CHECK( cudaEventCreate(&timeD) );
	
	//size_t free0, free1, free2, total;
	//CUDA_CHECK( cudaMemGetInfo(&free0, &total) );
	//CUDA_CHECK( cudaMemGetInfo(&free1, &total) );
	
	// Allocate memory and copy arrays to the device
	CUDA_CHECK( cudaDeviceSynchronize() );
	CUDA_CHECK( cudaEventRecord( timeA, 0 ) );
	
	size_t sizeR = sizeof(RayBuf[0]);
	
	int* NumBuf_dev;
	CUDA_CHECK( cudaMalloc((void **)&NumBuf_dev, 8*sizeof(int)) );
	Ray* RayBuf_dev;
	CUDA_CHECK( cudaMalloc((void **)&RayBuf_dev, 8*NUM_BUF*sizeR) );
	
	float size3 = FREQ_BIN_NUM*sizeof(float);
	float* nfSback_dev;
	CUDA_CHECK( cudaMalloc((void **)&nfSback_dev, size3));
	CUDA_CHECK( cudaMemcpy(nfSback_dev, nfSback, size3, cudaMemcpyHostToDevice) );
	
	CUDA_CHECK( cudaEventRecord( timeB, 0 ) );
	
	// Ray data on the GPU host
	int nRays0, nRays;
	Ray *RayDat;
	
	// Pointers for the ray data
	Ray *RayDat0_dev;
	Ray *RayDat_dev;
	
	// Tracing rays from the buffer
	if(PartInfo[3] == 1)
	{
		nRays0 = 0;
		for(int i=0; i<8; i++)
		{
			nRays0 += ndBuf[i];
		}
		nRays = nRays0;
		
		RayDat = new Ray[nRays];
		
		int d_dat = 0;
		int d_buf = 0;
		for(int i=0; i<8; i++)
		{
			memcpy(	RayDat + d_dat,
					RayBuf + d_buf,
					ndBuf[i]*sizeR);
			
			// dev rays are listed together
			d_dat += ndBuf[i];
			// host rays are spaced by NUM_BUF
			d_buf += NUM_BUF;
		}
		
		CUDA_CHECK( cudaMalloc((void **)&RayDat0_dev, nRays0*sizeR) ); 
		CUDA_CHECK( cudaMemcpy(	RayDat0_dev,
								RayDat,
								nRays0*sizeR,
								cudaMemcpyHostToDevice) );
		
		/*int m = 0;
		for(int j=0; j<3; j++)
		{
			for(int i=0; i<ndBuf[j]; i++)
			{
				int n = j*NUM_BUF + i;
				printf("%d\t%d\t%f\n", domain.get_id(), RayDat[m].get_dom(), RayDat[m].R);
				m++;
			}
		}
		return;*/
		//printf("Buf Domain %d has %d rays.\n", domain.get_id(), nRays);
	}
	
	// Reset buffer counts
	memset(ndBuf, 0, 8*sizeof(int));
	CUDA_CHECK( cudaMemcpy(	NumBuf_dev, ndBuf,
							8*sizeof(int), cudaMemcpyHostToDevice) );
	
	if(PartInfo[3] == 0)	// Starting rays:
	{
		// Produce the initial ray arrays:
		int ord = 6;
		int Nside = pow(2, ord);
		int nPix = 12*Nside*Nside;
		
		int partNumAdd = MIN( PartInfo[1], 1);
		/*int partNumAdd;
		if(10 > partNumAdd)
			partNumAdd = 10;
		else
			partNumAdd = PartInfo[1];*/
		
		nRays0 = partNumAdd*nPix;
		nRays = nRays0;
	
		RayDat = new Ray[nRays];
		
		// After initializing a set of rays, we advance along the particle array
		//printf("A %d\t%d\t%d\t%d\n", domain.get_id(), PartInfo[1], PartInfo[2], partNumAdd);
		
		// Sources are located on the device. We need them on the host to initialize.
		source * source_host = new source[PartInfo[0]];
		CUDA_CHECK( cudaMemcpy( source_host, source_dev, PartInfo[0]*sizeof(source), cudaMemcpyDeviceToHost) );
		
		int index = 0;
		for(int i=0; i<partNumAdd; i++)
		{
			int jetOn = 0;
			int jetOnNum = Nside*Nside;
			int jetOnPix =  rand() % (nPix/jetOnNum);
			
			if(jetOn == 0)
			{
				for(int j=0; j<nPix; j++)
				{
					//int index = i*nPix + j;
					RayDat[index].set_dom(domain.get_id());
					RayDat[index].set_part(PartInfo[2]);
					RayDat[index].set_pix(j, ord);
					
					// Defining the initial flux array for initialization
					float flux_init[FREQ_BIN_NUM]={0};
					for(int nBin=0; nBin<FREQ_BIN_NUM; nBin++)
					{
						// Start with the total luminosity of the source
						flux_init[nBin] = source_host[PartInfo[2]].gam;
						
						// Break the fluxes down by frequency bin
						// Divide by initial number of pixels
						flux_init[nBin] *= gfn[nBin]/nPix;
					}
					RayDat[index].set_flux(flux_init);
					
					float position[3], direction[3]={0};
					position[0] = source_host[PartInfo[2]].x;
					position[1] = source_host[PartInfo[2]].y;
					position[2] = source_host[PartInfo[2]].z;
					RayDat[index].set_position(position, 0.0, direction);
					
					index++;
				}
			}
			else
			{
				for(int j=0; j<jetOnNum; j++)
				{
					int pix = jetOnPix*jetOnNum + j;
					RayDat[index].set_dom(domain.get_id());
					RayDat[index].set_part(PartInfo[2]);
					RayDat[index].set_pix(pix, ord);
					
					// Defining the initial flux array for initialization
					float flux_init[FREQ_BIN_NUM]={0};
					for(int nBin=0; nBin<FREQ_BIN_NUM; nBin++)
					{
						// Start with the total luminosity of the source
						flux_init[nBin] = source_host[PartInfo[2]].gam;
						
						// Break the fluxes down by frequency bin
						// Divide by initial number of pixels
						flux_init[nBin] *= gfn[nBin]/jetOnNum;
					}
					RayDat[index].set_flux(flux_init);
					
					float position[3], direction[3]={0};
					position[0] = source_host[PartInfo[2]].x;
					position[1] = source_host[PartInfo[2]].y;
					position[2] = source_host[PartInfo[2]].z;
					RayDat[index].set_position(position, 0.0, direction);
					
					index++;
				}
			}
			
			PartInfo[1] -= 1;
			PartInfo[2] += 1;
		}
		
		delete[] source_host;
		
		//printf("B %d\t%d\t%d\n", PartInfo[1], PartInfo[2], partNumAdd);
		
		CUDA_CHECK( cudaMalloc((void **)&RayDat0_dev, nRays0*sizeR) ); 
		CUDA_CHECK( cudaMemcpy(	RayDat0_dev,
								RayDat,
								nRays0*sizeR,
								cudaMemcpyHostToDevice) );
	}
	
	delete[] RayDat;
	RayDat = 0;
	
	// Number of active rays:
	int *nRays_dev;
	CUDA_CHECK( cudaMalloc((void **)&nRays_dev, sizeof(int)) );
	
	// Kernel loop
	for(;nRays0>0;)
	{
		//printf("Domain: %d\tnRays0 = %d\n", domain.get_id(), nRays0);
		// Set CUDA kernel variables
		dim3 threadsPB(16, 16, 1);
		//	int gridx = 65535;
		//	int gridy = (nInit/256)/65535 + 1;
		int grid = (int) sqrt(nRays0/256) + 1;
		dim3 blocksPG(grid, grid, 1);
 
		CUDA_CHECK( cudaMemcpy(nRays_dev, &nRays0, sizeof(int), cudaMemcpyHostToDevice) );
		
		//CUDA_CHECK( cudaMemGetInfo(&free2, &total) );
		//if(free2 < free1)
		//	free1 = free2;
		
		// Execute kernel
		CUDA_CHECK( cudaGetLastError() );
		
		rayTraceKernel<<<blocksPG,threadsPB>>>(	DensArray_dev, x_NArray_dev, 
												source_dev,	FluxArray_dev, dEArray_dev,
												RayDat0_dev, nRays_dev, nRays0,
												L, a, nfSback_dev, domain, dt_ion);
		
		CUDA_CHECK( cudaPeekAtLastError() );
		CUDA_CHECK( cudaDeviceSynchronize() );
		
		// Get the number of continuing arrays
		CUDA_CHECK( cudaMemcpy(&nRays, nRays_dev, sizeof(int), cudaMemcpyDeviceToHost) );
		CUDA_CHECK( cudaMemset(nRays_dev, 0, sizeof(int)) );
//			printf("-> %d\t%f\n", nSplit, temp);

		// Create the new RayDat array
		//printf("Node %d nRays0 = %d nRays = %d\n", domain.get_id(), nRays0, nRays);
		nRays *= 4;
		CUDA_CHECK( cudaMalloc((void **)&RayDat_dev, nRays*sizeR) );
		
		// Split rays into new array
		CUDA_CHECK( cudaDeviceSynchronize() );
		CUDA_CHECK( cudaPeekAtLastError() );
		//printf("%d\t%d\t%d\t%d\n", domain.get_id(), ndBuf[0], ndBuf[1], ndBuf[2]);
		raySplitKernel<<<blocksPG,threadsPB>>>(	RayDat0_dev, RayDat_dev,
												nRays_dev, nRays0,
												RayBuf_dev, NumBuf_dev,
												source_dev, domain);
		CUDA_CHECK( cudaPeekAtLastError() );
		CUDA_CHECK( cudaDeviceSynchronize() );
		
		// Get the number of continuing arrays (post-split)
		CUDA_CHECK( cudaMemcpy(&nRays, nRays_dev, sizeof(int), cudaMemcpyDeviceToHost) );
		CUDA_CHECK( cudaMemset(nRays_dev, 0, sizeof(int)) );
		
		// Free old array
		CUDA_CHECK( cudaFree(RayDat0_dev) );

		// Handle pointer referencing
		RayDat0_dev = RayDat_dev;
		RayDat_dev = 0;
		
		// Reset counter
		nRays0 = nRays;
		
		//printf("%d\t%d\t%d\t%d\n", domain.get_id(), PartInfo[0], PartInfo[1], PartInfo[2]);
	}
	
	// Free unecessary array(s)
	CUDA_CHECK( cudaDeviceSynchronize() );
	CUDA_CHECK( cudaFree(nRays_dev) );
	CUDA_CHECK( cudaFree(RayDat0_dev) );
//	CUDA_CHECK( cudaFree(RayDat_dev) );	//CHECKXXX
	
	CUDA_CHECK( cudaEventRecord( timeC, 0 ) );
	
	/*float f0 = (uint)free0/1048576.0;
	float f1 = (uint)free1/1048576.0;
	float t1 = (uint)total/1048576.0;
	printf("Memory:\tStart: %f\tMax: %f\tTotal:%f\n", f0, f1, t1);*/
	
	CUDA_CHECK( cudaMemcpy(	ndBuf, NumBuf_dev,
							8*sizeof(int), cudaMemcpyDeviceToHost) );
	CUDA_CHECK( cudaMemcpy( RayBuf, RayBuf_dev,
							8*NUM_BUF*sizeR, cudaMemcpyDeviceToHost) );
		
	// Unified Memory
	/*memcpy(FluxArray, FluxArray_dev, size0*SPECIES);
	memcpy(dEArray, dEArray_dev, size0);*/
	
	CUDA_CHECK( cudaMemcpy(nfSback, nfSback_dev, size3, cudaMemcpyDeviceToHost) );
	
	CUDA_CHECK( cudaFree(NumBuf_dev) );
	CUDA_CHECK( cudaFree(RayBuf_dev) );
	
	CUDA_CHECK( cudaFree(nfSback_dev) );
	
	CUDA_CHECK( cudaDeviceSynchronize() );
	CUDA_CHECK( cudaEventRecord( timeD, 0 ) );
	
	float time0, time1, time2;
	CUDA_CHECK( cudaDeviceSynchronize() );
	CUDA_CHECK( cudaEventElapsedTime( &time0, timeA, timeB ) );
	CUDA_CHECK( cudaDeviceSynchronize() );
	CUDA_CHECK( cudaEventElapsedTime( &time1, timeB, timeC ) );
	CUDA_CHECK( cudaDeviceSynchronize() );
	CUDA_CHECK( cudaEventElapsedTime( &time2, timeC, timeD ) );
	
	CUDA_CHECK( cudaEventDestroy(timeA) );
	CUDA_CHECK( cudaEventDestroy(timeB) );
	CUDA_CHECK( cudaEventDestroy(timeC) );
	CUDA_CHECK( cudaEventDestroy(timeD) );
	
	//printf("Ray Trace (node %d):\t%f\t%f\t%f\n", domain.get_id(), time0/1000, time1/1000, time2/1000);
	
	dt[0] = time1/1000;
}

// Integrate neutral fractions differential equations
// DensArray	= Array of densities in cm^-3
// x_NArray		= Array of neutral fractions
// FluxArray	= Array of total ionizations, per baryon per second /1Mpc
// EArray		= Array of energies per baryon
// background	= Ionizing background for each channel
// dt			= Time step
// fErr			= Error number
// a			= Scale factor
void ion(	float* DensArray_dev, float* x_NArray_dev, float* FluxArray_dev,
			float* EArray, float* dEArray_dev, const float* background, float* dt, float* fErr, float a, Domain domain)
{
	*fErr = 0.0;
	//long elements = DIMX*DIMY*DIMZ;
	int dim = domain.get_dim();
	long elements = dim*dim*dim;
	
	dim3 numThreads1(16,16,1);
	dim3 numBlocks1;
	/*numBlocks1.x = DIMX/16;
	numBlocks1.y = DIMY/16;
	numBlocks1.z = DIMZ;*/
	numBlocks1.x = dim/16;
	numBlocks1.y = dim/16;
	numBlocks1.z = dim;
	
	size_t size2 = elements*sizeof(float);
	cudaExtent size = make_cudaExtent(DIMX, DIMY, DIMZ);
	
	cudaEvent_t timeA, timeB, timeC, timeD;
	CUDA_CHECK( cudaEventCreate(&timeA) );
	CUDA_CHECK( cudaEventCreate(&timeB) );
	CUDA_CHECK( cudaEventCreate(&timeC) );
	CUDA_CHECK( cudaEventCreate(&timeD) );
	
	CUDA_CHECK( cudaEventRecord( timeA, 0 ) );
	
	// 2. Copy data to GPU on each node
	float* EArray_dev;
	float* back_dev;
	
	CUDA_CHECK( cudaMalloc((void **)&EArray_dev, size2) );
	CUDA_CHECK( cudaMalloc((void **)&back_dev, 2*SPECIES*sizeof(float)) );
	
	CUDA_CHECK( cudaMemcpy(EArray_dev, EArray, size2, cudaMemcpyHostToDevice) );
	CUDA_CHECK( cudaMemcpy(back_dev, background, 2*SPECIES*sizeof(float), cudaMemcpyHostToDevice) );
	
	float* err_dev;
	CUDA_CHECK( cudaMalloc((void **)&err_dev, sizeof(float)) ); 	
	CUDA_CHECK( cudaMemcpy(err_dev, fErr, sizeof(float), cudaMemcpyHostToDevice) );
	
	CUDA_CHECK( cudaEventRecord( timeB, 0 ) );
	
	CUDA_CHECK( cudaGetLastError() );
	ionization<<<numBlocks1, numThreads1>>>(*dt, err_dev, DensArray_dev, x_NArray_dev, FluxArray_dev, EArray_dev, dEArray_dev, back_dev, dim, a);
	CUDA_CHECK( cudaPeekAtLastError() );
	CUDA_CHECK( cudaDeviceSynchronize() );
	
	CUDA_CHECK( cudaEventRecord( timeC, 0 ) );
	
	CUDA_CHECK( cudaDeviceSynchronize() );
	
	CUDA_CHECK( cudaMemcpy(fErr, err_dev, sizeof(float), cudaMemcpyDeviceToHost) );
	CUDA_CHECK( cudaMemcpy(EArray, EArray_dev, size2, cudaMemcpyDeviceToHost) );
	
	CUDA_CHECK( cudaEventRecord( timeD, 0 ) );
	
	CUDA_CHECK( cudaFree(err_dev) );
	CUDA_CHECK( cudaFree(EArray_dev) );
	CUDA_CHECK( cudaFree(back_dev) );
	
	float time0, time1, time2;
	CUDA_CHECK( cudaEventElapsedTime( &time0, timeA, timeB ) );
	CUDA_CHECK( cudaEventElapsedTime( &time1, timeB, timeC ) );
	CUDA_CHECK( cudaEventElapsedTime( &time2, timeC, timeD ) );
	
	CUDA_CHECK( cudaEventDestroy(timeA) );
	CUDA_CHECK( cudaEventDestroy(timeB) );
	CUDA_CHECK( cudaEventDestroy(timeC) );
	CUDA_CHECK( cudaEventDestroy(timeD) );
	
	//printf("Ionization (node %d):\t%f\t%f\t%f\n", domain.get_id(), time0/1000, time1/1000, time2/1000);
}

void dt_H(	float* rate, float* DensArray_dev, float* x_NArray_dev,
			float* FluxArray_dev, float* EArray, 
			const float* background, const float L, const float a, Domain domain)
{
	//long elements = DIMX*DIMY*DIMZ;
	int dim = domain.get_dim();
	long elements = dim*dim*dim;
	
	dim3 numThreads1(16,16,1);
	dim3 numBlocks1;
	/*numBlocks1.x = DIMX/16;
	numBlocks1.y = DIMY/16;
	numBlocks1.z = DIMZ;*/
	numBlocks1.x = dim/16;
	numBlocks1.y = dim/16;
	numBlocks1.z = dim;
	
	size_t size2 = elements*sizeof(float);
	cudaExtent size = make_cudaExtent(DIMX, DIMY, DIMZ);
	
	float* EArray_dev;
	float* back_dev;
	
	// FILTER
	float* dtFilter = new float[elements];
	float* dtFilter_dev;
	CUDA_CHECK( cudaMalloc((void **)&dtFilter_dev, size2) );
	CUDA_CHECK( cudaMemcpy(dtFilter_dev, dtFilter, size2, cudaMemcpyHostToDevice) );
	
	CUDA_CHECK( cudaMalloc((void **)&EArray_dev, size2) );
	CUDA_CHECK( cudaMalloc((void **)&back_dev, 2*SPECIES*sizeof(float)) );
	
	CUDA_CHECK( cudaMemcpy(EArray_dev, EArray, size2, cudaMemcpyHostToDevice) );
	CUDA_CHECK( cudaMemcpy(back_dev, background, 2*SPECIES*sizeof(float), cudaMemcpyHostToDevice) );
	
	//rate[0] = 0.0;
	float* rate_dev;
	CUDA_CHECK( cudaMalloc((void **)&rate_dev, 2*sizeof(float)) ); 	
	CUDA_CHECK( cudaMemcpy(rate_dev, rate, 2*sizeof(float), cudaMemcpyHostToDevice) );
	
	CUDA_CHECK( cudaGetLastError() );
	timestep<<<numBlocks1, numThreads1>>>(	rate_dev, dtFilter_dev,	DensArray_dev, x_NArray_dev,
						FluxArray_dev, EArray_dev, back_dev, dim, L, a);

	CUDA_CHECK( cudaPeekAtLastError() );
	CUDA_CHECK( cudaDeviceSynchronize() );
	
	CUDA_CHECK( cudaDeviceSynchronize() );
	
	CUDA_CHECK( cudaMemcpy(rate, rate_dev, 2*sizeof(float), cudaMemcpyDeviceToHost) );
	
	CUDA_CHECK( cudaFree(EArray_dev) );
	CUDA_CHECK( cudaFree(back_dev) );
	CUDA_CHECK( cudaFree(rate_dev) );
	
	// FILTER
	CUDA_CHECK( cudaMemcpy(dtFilter, dtFilter_dev, size2, cudaMemcpyDeviceToHost) );
	
	/*float max = 1.e-20;
	float buffer[DIMX];
	for(int i=0; i<DIMX; i++)
	{
		for(int j=0; j<DIMX; j++)
		{
			for(int k=1; k<DIMX-1; k++)
			{
				//cout << i << "\t" << j << "\t" << k << endl;
				int ind = i + DIMX*j + DIMX*DIMY*k;
				int dn = DIMX*DIMY;
				
				buffer[k]	= 0.5*dtFilter[ind];
				buffer[k]	+= 0.25*dtFilter[ind+dn];
				buffer[k]	+= 0.25*dtFilter[ind-dn];
			}
			for(int k=1; k<DIMX-1; k++)
			{
				int ind = i + DIMX*j + DIMX*DIMY*k;
				dtFilter[ind]	= buffer[k];
			}
		}
	}
	for(int i=0; i<DIMX; i++)
	{
		for(int k=0; k<DIMX; k++)
		{
			for(int j=1; j<DIMX-1; j++)
			{
				int ind = i + DIMX*j + DIMX*DIMY*k;
				int dn = DIMX;
				
				buffer[j]	= 0.5*dtFilter[ind];
				buffer[j]	+= 0.25*dtFilter[ind+dn];
				buffer[j]	+= 0.25*dtFilter[ind-dn];
			}
			for(int j=1; j<DIMX-1; j++)
			{
				int ind = i + DIMX*j + DIMX*DIMY*k;
				dtFilter[ind]	= buffer[j];
			}
		}
	}
	for(int j=1; j<DIMX-1; j++)
	{
		for(int k=1; k<DIMX-1; k++)
		{
			for(int i=1; i<DIMX-1; i++)
			{
				int ind = i + DIMX*j + DIMX*DIMY*k;
				int dn = 1;
				
				buffer[i]	= 0.5*dtFilter[ind];
				buffer[i]	+= 0.25*dtFilter[ind+dn];
				buffer[i]	+= 0.25*dtFilter[ind-dn];
			}
			for(int i=1; i<DIMX-1; i++)
			{
				int ind = i + DIMX*j + DIMX*DIMY*k;
				dtFilter[ind]	= buffer[i];
				max = MAX(max, dtFilter[ind]);
			}
		}
	}
	cout << "Unfiltered = " << 0.05/3.154e13/rate[0] << endl;
	cout << "Filtered = " << 0.05/3.154e13/max << endl;
	rate[0] = max;*/
	
	CUDA_CHECK( cudaFree(dtFilter_dev) );
	delete[] dtFilter;
}
