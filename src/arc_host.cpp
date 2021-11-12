#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <algorithm>

#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;

#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "./inc/domain.h"
#include "./inc/ray.h"
#include "arc.h"
#include "./inc/background.h"

// Global variables
//#define SCALE 4.0
//#define sigma_eff_dx 1.0/SCALE
//#define BOX_SIZE 10.0	// Length of DIMX side in Mpc
//#define BOX_SIZE 0.015
#define BOX_SIZE 0.0066 // Test123
//#define BOX_SIZE 0.714286 // Test4
//#define HUBBLE 0.7			// h
#define HUBBLE 1.0			// Test
#define MLIM 100.0			// M_lim
#define SINGLE_SOURCE 0.5251

// Error handling macros
#define MPI_CHECK(call) \
	if((call) != MPI_SUCCESS) { \
		cerr << "MPI error calling \""#call"\"\n"; \
		my_abort(-1); }
		
void halomatch(float*, unsigned, float, float, int);
double schechter(double,double, double, double);
double model_Mstar(double, unsigned);
double model_log10phi(double, unsigned);
double model_alpha(double, unsigned);

double time(double redshift){
	double h=0.6711;
	double h0=h*3.246753e-18;
	double omegam=0.3;
	double yrtos=3.15569e7;
	double time = 2.*pow((1.+redshift),-3./2.)/(3.*h0*pow(omegam,0.5));
	time = time/(yrtos*1.e6);
	return time;
}

double redshift(double time){
	double h=0.6711;
	double h0=h*3.246753e-18;
	double omegam=0.3;
	double yrtos=3.15569e7;
	time = time*yrtos*1.e6;
	double redshift = pow((3.*h0*pow(omegam,0.5)*time/2.),-2./3.)-1.;
	return redshift;
}

void output(float* FluxArray, float* EArray, float* x_NArray, int elements, int filenum)
{
	char sFile0[32];
	char sFile1[32];
	char sFile2[32];
	sprintf(sFile0, "../dat/hydrogen/xhi%03d.bin", filenum);
	sprintf(sFile1, "../dat/helium/xhei%03d.bin", filenum);
	sprintf(sFile2, "../dat/photons/flux%03d.bin", filenum);
	ofstream ofsData0(sFile0, ios::out | ios::binary);
	ofstream ofsData1(sFile1, ios::out | ios::binary);
	ofstream ofsData2(sFile2, ios::out | ios::binary);
	
	char sFile3[32];
	sprintf(sFile3, "../dat/photons/temp%03d.bin", filenum);
	ofstream ofsData3(sFile3, ios::out | ios::binary);
	
	//	int mod = DIMX/128;
	//	cout << "mod = " << mod << endl;
	for (long i = 0; i<elements; i++)
	{
		/*int i0 = i % (DIMX);
		int j0 = i / (DIMX);
		int k0 = i / (DIMX*DIMY);
		if(i0%mod==0 && j0%mod==0 && k0%mod==0)
		{
		}*/
		ofsData0.write((char*)(x_NArray + i), sizeof(float));
		ofsData1.write((char*)(x_NArray + elements + i), sizeof(float));
		for (int nBin = 0; nBin<SPECIES; nBin++)
		{
			//ofsData2.write((char*)(FluxArray + elements*nBin + i), sizeof(float));
		}
		
		float xe = (1.0-Y_P)*(1.0-x_NArray[i])+0.25*Y_P*(1.0-x_NArray[elements + i]);
		float T = EArray[i]/(1.5*8.6173303e-5*(1.0+xe));
		ofsData3.write((char*)&T, sizeof(float));
	}

	ofsData0.close();
	ofsData1.close();
	ofsData2.close();
	ofsData3.close();
}

inline float frand()
{
    return rand() / (float) RAND_MAX;
}

inline float f_esc(float M)
{
	return 1.0-0.5*tanh(M/1.e9);
}

inline float Lum_S0(float mag)
{
	return 9.426e6*exp(0.4*2.30259*(-25.0-mag));
}

// More stable way to take the sum of a large cube of ints
inline float cubesum(float* arr, int side)
{
	int ind = 0;
	float lay = 0;
	for(int k=0; k<side; k++)
	{
		float col = 0;
		for(int j=0; j<side; j++)
		{
			float row = 0;
			for(int i=0; i<side; i++)
			{
				row += arr[ind];
				ind++;
			}
			//row /= ((float) side);
			col += row;
		}
		//col /= ((float) side);
		lay += col;
	}
	//lay /= ((float) side);
	
	return lay;
}

bool sortByGam(const source &lhs, const source &rhs) {return lhs.gam > rhs.gam;}

inline void arrayChange(float*);
inline void arrayChangeInv(float*);
inline void BinGridRead(float*, char*, int);

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int main(int argc, char** argv)
{
	string line, word;
	ifstream file("../parameters.txt");
	
	string PATH;
	float fDtConst, fFesc, fMdm, fTfinal, fDtFile;
	int nStepType, nLoopInit, nModel;
	
	if(file.good())
	{
		getline(file, line);
		PATH = line.substr(0, line.find("\t"));
		
		getline(file, line);
		nModel = atoi( line.substr(0, line.find("\t")).c_str() );
		
		getline(file, line);
		nStepType = atoi( line.substr(0, line.find("\t")).c_str() );
		
		getline(file, line);
		fDtConst = atof( line.substr(0, line.find("\t")).c_str() );
		
		getline(file, line);
		fTfinal = atof( line.substr(0, line.find("\t")).c_str() );
		
		getline(file, line);
		nLoopInit = atoi( line.substr(0, line.find("\t")).c_str() );
		
		getline(file, line);
		fDtFile = atof( line.substr(0, line.find("\t")).c_str() );

		getline(file, line);
		fFesc = atof( line.substr(0, line.find("\t")).c_str() );
		
		getline(file, line);
		fMdm = atof( line.substr(0, line.find("\t")).c_str() );
	}
	else
	{
		//cout << "Couldn't read input file at: " << file << endl;
		return 0;
	}
	file.close();
	
	FILE *report;
	report = fopen("../dat/report.dat", "w");
	fprintf(report, "Details of the present simulation run:\n");
	fprintf(report, "Time step:\t%f Myr\n", fDtConst);
	fprintf(report, "Length of sim:\t%e\n", fTfinal);
	fprintf(report, "Output produced every:\t%e Myrs\n", fDtFile);
	fclose(report);
	
	// Variables available on every node at all times
	// Number of sub-domains
	int nDomain = 8;
	
	// Size of host array and each sub-domain
	long nSizeHost = DIMX*DIMY*DIMZ;
	int nSize = nSizeHost/nDomain;
	
	// Initialize MPI state
	MPI_CHECK(MPI_Init(&argc, &argv));
	
	// Get our MPI node number and node count
	int commSize, commRank;
	MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &commSize));
	MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &commRank));
	
    // Check that we can actually divide the work correctly:
	if(commSize%nDomain != 0)
	{
		printf("commSize (%d) isn't divisble by nDomain (%d)!\nEXIT!\n", commSize, nDomain);
		MPI_CHECK(MPI_Finalize());
		return 0;
	}
	
	// Setup split communicators
	int nLayers = commSize / nDomain;
	int nGridDom = commRank % nDomain;
	int nGridLoc = commRank / nDomain;
	
	// MPI Grids: Dom for split of domain, Loc for "layers" of domains
	// X1 X2 X3 X4 ... (nGridLoc = 0)
	// X1 X2 X3 X4 ... (nGridLoc = 1)
	MPI_Comm commGridDom, commGridLoc;
	MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, nGridLoc, commRank, &commGridDom));
	MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, nGridDom, commRank, &commGridLoc));
	
	// Create a type for our source struct
	MPI_Datatype	mpi_source;
	{
	int				blocklengths[4] = {1,1,1,1};
	MPI_Datatype	types[4] = {MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT};
	MPI_Aint		offsets[4];
	
	offsets[0]		= offsetof(source, x);
	offsets[1]		= offsetof(source, y);
	offsets[2]		= offsetof(source, z);
	offsets[3]		= offsetof(source, gam);
	
	MPI_Type_create_struct(4, blocklengths, offsets, types, &mpi_source);
	MPI_Type_commit(&mpi_source);
	}
	
	// Create a type for our ray class
	MPI_Datatype	mpi_ray;
	{
	int				blocklengths[7] = {1,1,1,1,3,3,FREQ_BIN_NUM};
	MPI_Datatype	types[7] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT,
								MPI_FLOAT, MPI_FLOAT, MPI_FLOAT};
	MPI_Aint		offsets[7];
	
	offsets[0]		= offsetof(Ray, PID);
	offsets[1]		= offsetof(Ray, UID);
	offsets[2]		= offsetof(Ray, DOM);
	offsets[3]		= offsetof(Ray, R);
	offsets[4]		= offsetof(Ray, gridloc);
	offsets[5]		= offsetof(Ray, position);
	offsets[6]		= offsetof(Ray, flux);
	
	MPI_Type_create_struct(7, blocklengths, offsets, types, &mpi_ray);
	MPI_Type_commit(&mpi_ray);
	}
	
	if(commRank == 0)
	{
		cout << "Loading data from: " << PATH << endl;
		cout << "Model: " << nModel << endl;
		cout << "Time step: " << fDtConst << endl;
		cout << "Length of simulation: " << fTfinal << endl;
		cout << "Starting on output number: " << nLoopInit << endl;
		cout << "Output every: " << fDtFile << " Myr" << endl;
		cout << "Escape fraction: " << fFesc << endl;
	}
	
	// Define the local domain
	Domain domain(nGridDom, DIMX/2);
	
	// Declare CPU arrays on all nodes
	float *DensArray, *DensArrayHost;
	float *x_NArray, *x_NArrayHost;
	float *FluxArray, *FluxArrayHost;
	float *dEArray;
	
	float *Background, *BackgroundHost;
	float *EArray, *EArrayHost;
	
	// Allocate space for local arrays
	DensArray = new float[nSize];
	x_NArray = new float[nSize*SPECIES];
	FluxArray = new float[nSize*SPECIES];
	dEArray = new float[nSize];
	
	Background = new float[2*SPECIES];
	EArray = new float[nSize];
				
	source *particles;
	
	// Device Arrays for the rad calculation
	size_t nSizeBytes = nSize*sizeof(float);
	__device__ float *DensArray_dev;
	__device__ float *x_NArray_dev;
	__device__ float  *FluxArray_dev;
	__device__ float  *dEArray_dev;
	__device__ source *source_dev;
	
	CUDA_CHECK( cudaMalloc((void **)&DensArray_dev, nSizeBytes) ); 
	CUDA_CHECK( cudaMalloc((void **)&x_NArray_dev, nSizeBytes*SPECIES) );
	CUDA_CHECK( cudaMalloc((void **)&FluxArray_dev, nSizeBytes*SPECIES) );
	CUDA_CHECK( cudaMalloc((void **)&dEArray_dev, nSizeBytes) );
	
	// Generic string for storing names
	char name[100];
	
	// Loop variable declarations
	int counter = 0;
	float fTFile = fDtFile*(1.e0 + 1.e-3);
	int step = 0;
	int iFileNum = 0;
	int nOnLimit;	// Max halos (to balance S0)
	int nOnCount;	// Number on in current MODEL
	int nOnFinal;	// Final halo count after limits applied
	int* nOnLoc;	// Array containing local counts
	nOnLoc = new int[nDomain];
	int* nOnDisp;	// Array containing displacements for local counts
	nOnDisp = new int[nDomain];
	
	//float fT0 = time(30.0-0.01);
	float fT0 = time(0.0-0.01);
	//float fT0 = time(9.0-0.01);	//Test
	float fT = 0.0;
	float fDt = fDtConst;
	
	// Averaged values, and local counterparts
	double ndSum[6] = {1.0, 1.0, 1.0, 1.0, 0.0, 0.0};
	double ndSum_loc[6];
	double ndSum_old[6];
	float error_loc;
	
	// Background array declarations
	float ndot;
	const int BACK_BINS = 1000;
	float* Jback = new float[BACK_BINS];
	float* Nuback = new float[BACK_BINS];
	
	// Arrays for balancing radiation workload
	float* dtRad = new float[nLayers];
	int* haloCount = new int[nLayers];
	int* haloDisp = new int[nLayers];
	
	double dTStart=0,dT0=0,dT1=0,dT2=0;
	double dT0a=0, dT0b=0;
	if(commRank==0) dTStart=MPI_Wtime();
	
	// Summary file
	FILE *summary;
	
	// Initialize arrays on the root node
	if (commRank == 0)
	{
		cout << "Running on " << commSize << " nodes" << endl;
		
		// Grid properties written here
		DensArrayHost = new float[nSizeHost];
		x_NArrayHost = new float[nSizeHost*SPECIES];
		FluxArrayHost = new float[nSizeHost*SPECIES];
		BackgroundHost = new float[2*SPECIES];
		EArrayHost = new float[nSizeHost];
		
		// Initialize neutral arrays if beginning on the 0th step
		// Other possible t=t0 initializations written here
		if(nLoopInit == 0)
		{
			// Blank summary file
			summary = fopen("../dat/sum.dat", "w+");
			fclose(summary);
	
			for (long i = 0; i<nSizeHost; i++)
			{
				for(int j=0; j<SPECIES; j++)
					x_NArrayHost[i+j*nSizeHost] = 1.0;	//neutral fraction
				float kB = 8.6173303e-5;			// In eV
				EArrayHost[i] = (3./2.)*kB*TNAUGHT;
			}
			for(int i=0; i<BACK_BINS; i++)
			{
				Nuback[i] = pow(10., 1.+(2.-1.)*(i/((float) BACK_BINS)));
				Jback[i] = 0;
			}
		}
		else	// Here we load the data for resuming the code
		{
			// Loop data:
			sprintf(name, "../dat/restart/restart%03d.dat", nLoopInit);
			file.open(name);
			
			getline(file, line);
			
			istringstream iss(line);
			iss >> counter >> step >> iFileNum >> fT >> fDt >> fTFile >> fDtFile;
			file.close();
			/*cout << counter << "\t" << step << "\t" << iFileNum << endl;
			cout << fT << "\t" << fDt << "\t" << ftFile << "\t" << fDtFile << endl;*/
			
			// Copy previous "sum.dat" until the correct location
			int temp = rename( "../dat/sum.dat", "../dat/last_sum.dat");
			if( temp != 0 )
				printf( "Error renaming file\n");
	
			ofstream file0("../dat/sum.dat", ofstream::out);
			file0.precision(6);
			file0 << scientific;
	
			ifstream file1("../dat/last_sum.dat");
	
			float fTread = 0.0;
			while(fTread < fT)
			{
				if(!getline(file1, line))
				{
					perror("Problem copying summary file!\n");
					return 0;
				}
		
				istringstream iss(line);
				iss >> fTread;
		
				if( fTread > fT )
					break;
		
				file0 << fTread << "\t";
		
				float temp;
				for(int i=0; i<13; i++)
				{
					iss >> temp;
					file0 << temp << "\t";
				}
				file0 << "\n";
			}
			file0.close();
			file1.close();
			
			// Array data:
			sprintf(name, "../dat/hydrogen/xhi%03d.bin", nLoopInit);
			BinGridRead(x_NArrayHost, name, DIMX);
			
			sprintf(name, "../dat/helium/xhei%03d.bin", nLoopInit);
			BinGridRead(x_NArrayHost + nSizeHost, name, DIMX);
			
			sprintf(name, "../dat/photons/temp%03d.bin", nLoopInit);
			BinGridRead(EArrayHost, name, DIMX);
			
			sprintf(name, "%s/nanoJubilee/GridDensities/%d/GridDensities_%d_0%02d.bin", PATH.c_str(), DIMX, DIMX, iFileNum);
			//sprintf(name, "./reduced.bin");	//Test
			BinGridRead(DensArrayHost, name, DIMX);
			
			for(int i=0; i<nSizeHost; i++)
			{
				float kB = 8.6173303e-5;			// In eV
				float xe = (1.0-Y_P)*(1.0-x_NArrayHost[i]);
				xe += 0.25*Y_P*(1.0-x_NArrayHost[nSizeHost + i]);
				//EArrayHost[i] = (1.5*kB*(1.0+xe))*EArrayHost[i]; // Test
				EArrayHost[i] = (1.5*kB*(1.0+xe))*1.0e4;
				
				//DensArrayHost[i] = 2.43e-7*DensArrayHost[i];
				//float fA = 1.0/(1.0+redshift(fT+fT0));
				DensArrayHost[i] = 1.e-3;
				//DensArrayHost[i] = fA*fA*fA*DensArrayHost[i];	//Test4
			}
			
			cout << "First density element = " << DensArrayHost[0]/2.43e-7 << endl;
			
			// Particles:
			source *p_init = new source[400000];
			sprintf(name, "../dat/photons/slist%02d.dat", iFileNum-1);
			file.open(name);
			
			ndot = 0;
			nOnFinal = 0;
			for(int i=0; i<nDomain; i++)
			{
				nOnLoc[i] = 0;
			}
			while (getline(file, line))
			{
				float X, Y, Z, M, S0;
				istringstream iss(line);
					
				if(!(iss >> X >> Y >> Z >> S0 >> M))
					break;
				float fJet = 1.0;
				p_init[nOnFinal].x = X;
				p_init[nOnFinal].y = Y;
				p_init[nOnFinal].z = Z;
				p_init[nOnFinal].gam = S0*fFesc*fJet;	//CHECKXXX fJet
				
				float dim2 = DIMX/2.0;
				int ix = (int) p_init[nOnFinal].x/dim2;
				int iy = (int) p_init[nOnFinal].y/dim2;
				int iz = (int) p_init[nOnFinal].z/dim2;
				
				int ind = 4*iz + 2*iy + ix;
				nOnLoc[ind]++;
				
				// ndi is the photons per particle
				float ndi = p_init[nOnFinal].gam;
				ndi /= 2.43e-7*pow((float) BOX_SIZE,3.0)*3.086e24;
				ndot += ndi;
				 
				nOnFinal++;
			}
			
			// Order the particles by volume
			particles = new source[nOnFinal];
			for(int i=0; i<nOnFinal; i++)
			{
				particles[i].x = p_init[i].x;
				particles[i].y = p_init[i].y;
				particles[i].z = p_init[i].z;
				particles[i].gam = p_init[i].gam;
			}
			delete[] p_init;
			file.close();
			
			// Background arrays:
			sprintf(name, "../dat/background/spec%03d.dat", nLoopInit);
			file.open(name);
			
			int iback=0;
			while (getline(file, line))
			{
				double temp0, temp1;
				istringstream iss(line);
					
				if(!(iss >> temp0 >> temp1))
					break;
				
				Nuback[iback] = temp0;
				Jback[iback] = temp1;
				//cout << "Nu = " << Nuback[iback] <<  "J = " << Jback[iback] << "\n";
				iback++;
			}
			file.close();
			
			Background[0] = 0.0;
			Background[1] = 0.0;
			Background[2] = 0.0;
			Background[3] = 0.0;
			float dLogNu = log(Nuback[1]/Nuback[0])*4*3.14159;
			
			for(int iback=0; iback<BACK_BINS; iback++)
			{
				Background[0] += Jback[iback]*sig_H(Nuback[iback])*dLogNu;
				Background[1] += Jback[iback]*sig_He(Nuback[iback])*dLogNu;
				Background[2] += (Nuback[iback] - 13.6)*Background[0];
				Background[3] += (Nuback[iback] - 24.6)*Background[1];
			}
			
			fT += fDt;
			step++;
		}
		
		// Change to sub-domain ordering
		arrayChange(DensArrayHost);
		arrayChange(EArrayHost);
		for(int nSpe=0; nSpe < SPECIES; nSpe++)
		{
			arrayChange(x_NArrayHost + nSpe*nSizeHost);
			arrayChange(FluxArrayHost + nSpe*nSizeHost);
		}
	}
	
	MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
	
	// Send the loop variables to all nodes.
	MPI_CHECK(MPI_Bcast(&counter, 1, MPI_INT, 0, MPI_COMM_WORLD));
	MPI_CHECK(MPI_Bcast(&step, 1, MPI_INT, 0, MPI_COMM_WORLD));
	MPI_CHECK(MPI_Bcast(&iFileNum, 1, MPI_INT, 0, MPI_COMM_WORLD));
	MPI_CHECK(MPI_Bcast(&fT, 1, MPI_FLOAT, 0, MPI_COMM_WORLD));
	MPI_CHECK(MPI_Bcast(&fDt, 1, MPI_FLOAT, 0, MPI_COMM_WORLD));
	MPI_CHECK(MPI_Bcast(&fTFile, 1, MPI_FLOAT, 0, MPI_COMM_WORLD));
	MPI_CHECK(MPI_Bcast(&fDtFile, 1, MPI_FLOAT, 0, MPI_COMM_WORLD));
	MPI_CHECK(MPI_Bcast(&nOnFinal, 1, MPI_INT, 0, MPI_COMM_WORLD));
	
	// Scatter data from root node to nodes in top layer
	if(nGridLoc == 0)
	{
		for(int nSpe=0; nSpe < SPECIES; nSpe++)
		{
			int dloc = nSize*nSpe;
			int dhost = nSizeHost*nSpe;
			MPI_CHECK(MPI_Scatter(	x_NArrayHost + dhost, nSize, MPI_FLOAT, 
									x_NArray + dloc, nSize, MPI_FLOAT,
									0, commGridDom));
			MPI_CHECK(MPI_Scatter(	FluxArrayHost + dhost, nSize, MPI_FLOAT, 
									FluxArray + dloc, nSize, MPI_FLOAT,
									0, commGridDom));
		}
	
		MPI_CHECK(MPI_Scatter(	EArrayHost, nSize, MPI_FLOAT, 
								EArray, nSize, MPI_FLOAT, 0, commGridDom));
	}
	
	// Broadcast from top layer to all layers
	MPI_CHECK(MPI_Bcast( FluxArray, nSize*SPECIES, MPI_FLOAT, 0, commGridLoc));
	MPI_CHECK(MPI_Bcast( x_NArray, nSize*SPECIES, MPI_FLOAT, 0, commGridLoc));
	MPI_CHECK(MPI_Bcast( EArray, nSize, MPI_FLOAT, 0, commGridLoc));
	
	MPI_CHECK(MPI_Bcast(Background, 4, MPI_FLOAT, 0, MPI_COMM_WORLD));
	
	// These arrays only get sent if we're restarting:
	if(nLoopInit != 0)
	{
		if(nGridLoc == 0)
		{
			MPI_CHECK(MPI_Scatter(	DensArrayHost, nSize, MPI_FLOAT, 
									DensArray, nSize, MPI_FLOAT, 0, commGridDom));
		}
		MPI_CHECK(MPI_Bcast( DensArray, nSize, MPI_FLOAT, 0, commGridLoc));
		
		// Only the host node has the particle array allocated
		if(commRank != 0)
			particles = new source[nOnFinal];
		MPI_CHECK(MPI_Bcast(particles, nOnFinal, mpi_source, 0, MPI_COMM_WORLD));
		
		CUDA_CHECK( cudaMalloc((void **)&source_dev, nOnFinal*sizeof(source)) );
		CUDA_CHECK( cudaMemcpy(source_dev, particles, nOnFinal*sizeof(source), cudaMemcpyHostToDevice) );
	}
	else
	{
		// Create a dummy array so that the loop can delete the array safely
		CUDA_CHECK( cudaMalloc((void **)&source_dev, sizeof(source)) );
	}
	
	// Free large arrays from the root node
	if(commRank == 0)
	{
		delete[] DensArrayHost;
		delete[] x_NArrayHost;
		delete[] EArrayHost;
		delete[] FluxArrayHost;
	}
	
	// Copy initial arrays onto device memory
	CUDA_CHECK( cudaMemcpy(DensArray_dev, DensArray, nSizeBytes, cudaMemcpyHostToDevice) );
	CUDA_CHECK( cudaMemcpy(x_NArray_dev, x_NArray, nSizeBytes*SPECIES, cudaMemcpyHostToDevice) );
	CUDA_CHECK( cudaMemcpy(FluxArray_dev, FluxArray, nSizeBytes*SPECIES, cudaMemcpyHostToDevice) )
	
	// Distribute initail halo info
	MPI_CHECK(MPI_Bcast(nOnLoc, nDomain, MPI_INT, 0, MPI_COMM_WORLD));
	
	if(nLoopInit != 0)
	{
		// Set displacements for halo counts
		nOnDisp[0] = 0;
		for(int i=1; i<nDomain; i++)
		{
			nOnDisp[i] = nOnDisp[i-1] + nOnLoc[i-1];
		}
		
		// Set framework for adaptive halo calc
		int disp = 0;
		for(int i=0; i<nLayers-1; i++)
		{
			//int nRemCount = nOnLoc[nGridDom] - haloDisp[i];
			float fOnLoc = (float) nOnLoc[nGridDom];
			
			haloCount[i] = (int) (fOnLoc/nLayers);
			haloDisp[i] = disp;
			
			disp += haloCount[i];
		}
		
		haloCount[nLayers-1] = nOnLoc[nGridDom] - disp;
		haloDisp[nLayers-1] = disp;
	}
	
	// Load source file scale factors
	sprintf(name, "../outputs_a.txt");
	
	ifstream scalefactors;
	scalefactors.open(name);
	float nfA[94];	
	for(int i=0; i<94; i++)
	{
		scalefactors >> nfA[i];
	}
	scalefactors.close();
	
	ofstream escape ("../dat/escape.dat", ios::out);
	escape.close();
	
	double emitted_total = 0.0;
	double recombined_total = 0.0;
	ofstream conservation ("../dat/conservation.dat", ios::out);
	conservation << "Time\t\td(x_HII)\tGamma*dt\tAlpha*dt\tRatio\t\t";
	conservation << "x_HII\t\tGamma*T\t\tAlpha*T\t\tRatio\t\tFlux/S0" << endl;
	conservation.close();
	
	int nDuty = 0;
	float fDuty = 1.0;
	
	float fTslice = time(1.0 / nfA[iFileNum] - 1.0);
	
	for(; fT < fTfinal; step++)
	{
		if(commRank == 0)
			printf("\nBeginning step #%d\tT = %f\tT_n = %f\n", step, fT, fTslice);
		
		float fA = 1.0/(1.0+redshift(fT+fT0));
		
		//	Read particle/density file if past a time step	//
		//if( fT+fT0 > fTslice)
		if(iFileNum == 0) //Test
		{
			if(iFileNum != 0)
			{
				delete[] particles;
			}
			nOnCount = 0;
			//printf("HUH\n");
			// Distribute new densities:
			if(commRank == 0)
			{
				DensArrayHost = new float[nSizeHost];
				
				sprintf(name, "%s/nanoJubilee/GridDensities/%d/GridDensities_%d_0%02d.bin", PATH.c_str(), DIMX, DIMX, iFileNum);
				//sprintf(name, "./reduced.bin");	//Test
				ifstream ifsDens (name, ios::in | ios::binary);
				ifsDens.seekg (0, ios::beg);
				cout << "Read the particle files" << endl;
				double FillFact = 0.0;
				for(int i=0; i<nSizeHost; i++)
				{
					float buff;
					ifsDens.read((char*)&buff, 4);
					if(buff > 200.0)
					{
						DensArrayHost[i] = 2.43e-7*buff;
						FillFact += 1.0;
					}
					else
						DensArrayHost[i] = 2.43e-7*buff;
					//DensArrayHost[i] = 2.43e-7*buff;
					//DensArrayHost[i] = buff*fA*fA*fA;	//Test4
					DensArrayHost[i] = 1.e-3;	//Test123
				}
				
				arrayChange(DensArrayHost);
				cout << "First density element = " << DensArrayHost[0]/2.43e-7 << endl;
				cout << "Filling Factor = " << FillFact/nSizeHost << endl;
			}
			
			if(nGridLoc == 0)
			{
				MPI_CHECK(MPI_Scatter(	DensArrayHost, nSize, MPI_FLOAT, 
										DensArray, nSize, MPI_FLOAT, 0, commGridDom));
			}
			
			if(commRank == 0)
				delete[] DensArrayHost;
			
			MPI_CHECK(MPI_Bcast( DensArray, nSize, MPI_FLOAT, 0, commGridLoc));
			MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
			
			// Copy new array onto 
			CUDA_CHECK( cudaMemcpy(DensArray_dev, DensArray, nSizeBytes, cudaMemcpyHostToDevice) );
			
			// Open the particle files and figure out how many halos are on
			// 400000 is larger than the largest number of halos
			int *niOnHalos;
			float *mags;
			source *p_init;
			//cout << "Allocated room for halo counts" << endl;
			
			if(commRank == 0)
			{
				nOnLimit = 0;	// Max halos (to balance S0)
				nOnCount = 0;	// Number on in current MODEL
				
				// Run through the files to find memory requirements
				// Also take note of which halos are on
				
				niOnHalos = new int[400000]; // 400000 is larger than the largest halo file
				
				int iFileNumTemp = iFileNum;
				iFileNumTemp = 997; // Use this for the test file
				sprintf(name, "../slice%03d.dat", iFileNumTemp);
				//sprintf(name, "%s/nanoJubilee/Reduced/slice%02d.dat", PATH.c_str(), iFileNumTemp);
				//sprintf(name, "../sources.dat");	//Test4
				
				string line;
				ifstream file(name);
				
				int i=0;
				while (getline(file, line))
				{
					string ID;
					float X, Y, Z, a;
					float mStarC, mStarB;
					int mdm, slice, state;
					istringstream iss(line);
					
					if(!(	iss >> ID >> mdm >>
							X >> Y >> Z >> a >>
							slice >> state >> 
							mStarC >> mStarB))
					{
						break;
					}
					
					/*mdm=0;
					slice=0;
					state=0;
					if(!(iss >> X >> Y >> Z >> a))
						break;*///Test4
					
					float mass = 23936.0*mdm;
					
//					cout << slice << "\t" << state << endl;
					
					// Condition for a halo being on
					// Pop III star conditions would be set here
					niOnHalos[i] = 0;
					
					if(state==0)	//has merged
					{
						if(nModel == 0)
						{
							niOnHalos[i] = 1;
							nOnCount++;
						}
						
						if( (mass > fMdm && mStarB < 1.0) || // Massive enough and no stars
							((iFileNum - slice)%10 == 0 && mStarB > 1.0) ) // Bursting again
						{
							nOnLimit++;	// Number of bursty halos sets limit
						
							if(nModel == 1)
							{
								niOnHalos[i] = 1;
								nOnCount++;
							}
						}
					}
					
					i++;
				}
				
				// Run through the file again to copy relevant halo data
				p_init = new source[nOnCount];
				printf("nOnLimit = %d\tnOnCount = %d", nOnLimit, nOnCount);
				
				file.clear();
				file.seekg(0, ios::beg);
				
				i = 0;
				int j = 0;
				int min=100, max=0;
				while (getline(file, line))
				{
					string ID;
					float X, Y, Z, a;
					float temp;
					int mdm, slice, state;
					istringstream iss(line);
					
					// Read the line into the relevant variables
					if(!(iss >> ID >> mdm >> X >> Y >> Z >> a >> slice >> state >> temp >> temp))
						break;
					
					//cout << ID << "\t" << mdm << endl;
					//float mass = 23936.0*mdm;
					if(niOnHalos[i] == 1)
					{
						p_init[j].gam = (float) mdm;	// Mass in #dm particles
						p_init[j].x = (X/(BOX_SIZE*1.e3*HUBBLE))*DIMX;
						p_init[j].y = (Y/(BOX_SIZE*1.e3*HUBBLE))*DIMY;
						p_init[j].z = (Z/(BOX_SIZE*1.e3*HUBBLE))*DIMZ;
						
						j++;
					}
					
					/*if(!(iss >> X >> Y >> Z >> a))
						break;
					
					//cout << ID << "\t" << mdm << endl;
					if(niOnHalos[i] == 1)
					{
						p_init[j].gam = a;
						p_init[j].x = X;
						p_init[j].y = Y;
						p_init[j].z = Z;
						
						j++;
					}*///Test
					
					i++;
					
				}
				
				delete[] niOnHalos;
				file.close();
				
				// Put the halos in order of decreasing mass
				sort(p_init, p_init + nOnCount, sortByGam);
				//cout << "p_init[0] = " << p_init[0].gam << endl;
				//cout << "p_init[nOnLimit] = " << p_init[nOnLimit].gam << endl;
				
				// Calculate luminosities and how many halos are above MLIM
				//nOnLimit = nOnCount; //Test4
				mags = new float[nOnLimit];
				
				halomatch(mags, nOnLimit, nfA[iFileNum], pow(BOX_SIZE,3), 0);	// HUBBLE?
				
				nOnFinal = 0;
				for(int i=0; i<nDomain; i++)
				{
					nOnLoc[i] = 0;
				}
				for (int i=0; i<nOnLimit; i++)
				{
					//if( mags[i] <= MLIM )
					if(true)//Test
					{
						nOnFinal++;
						
						float dim2 = DIMX/2.0;
						int ix = (int) p_init[i].x/dim2;
						int iy = (int) p_init[i].y/dim2;
						int iz = (int) p_init[i].z/dim2;
						
						int ind = 4*iz + 2*iy + ix;
						nOnLoc[ind]++;
					}
				}
			}
			
			// Create particle arrays and assign position and luminosity
			MPI_CHECK(MPI_Bcast(&nOnFinal, 1, MPI_INT, 0, MPI_COMM_WORLD));
			MPI_CHECK(MPI_Bcast(nOnLoc, nDomain, MPI_INT, 0, MPI_COMM_WORLD));
			
			// Assign displacements along the particle array
			nOnDisp[0] = 0;
			for(int i=1; i<nDomain; i++)
			{
				nOnDisp[i] = nOnDisp[i-1] + nOnLoc[i-1];
			}
			
			particles = new source[nOnFinal];
			
			if(commRank == 0)
			{
				ndot = 0; // Total photon count for current slice
				
				// Load the relevant halos into the particle array
				int* nOnLocTemp = new int[nDomain];
				float *mass = new float[nOnFinal];
				for(int i=0; i<nDomain; i++)
					nOnLocTemp[i] = 0;
				
				float fJet = 1.0;
				for (int i=0; i<nOnFinal; i++)
				{
					float dim2 = DIMX/2.0;
					int ix = (int) p_init[i].x/dim2;
					int iy = (int) p_init[i].y/dim2;
					int iz = (int) p_init[i].z/dim2;
					int ind = 4*iz + 2*iy + ix;
					
					// Order the particles by volume
					int j = nOnDisp[ind] + nOnLocTemp[ind];
					nOnLocTemp[ind]++;
					
					//printf("%f\t%f\t%f\t%f\t%d\n", p_init[i].x, p_init[i].y, p_init[i].z, p_init[i].gam, ind);
					//printf("%d\t%d\t%d\t%d\n", ind, nOnDisp[ind], (nOnLocTemp[ind]-1), j);
					
					particles[j].x = p_init[i].x;
					particles[j].y = p_init[i].y;
					particles[j].z = p_init[i].z;
					particles[j].gam = p_init[i].gam;
					
					float S0 = Lum_S0(mags[i]);
					mass[j] = p_init[i].gam*23936.0;
					
					//particles[j].gam = fJet*fFesc*S0/fDuty;	// CHECK
					particles[j].gam = SINGLE_SOURCE;	// TEST
					//particles[j].gam = p_init[i].gam*1050.0;	// TEST
					
					// ndi is the photons per particle
					float ndi = particles[j].gam;
					ndi /= 2.43e-7*pow((float) BOX_SIZE,3.0)*3.086e24*fJet;
					//ndot += ndi;	//TestCHECKXXX
				}
				
				char mlname[40];
				sprintf(mlname, "../dat/photons/slist%02d.dat", iFileNum);
				FILE *ML;
				ML = fopen(mlname, "w");
				
				for(int i=0; i<nOnFinal; i++)
				{
					fprintf(ML, "%e\t%e\t", particles[i].x, particles[i].y);
					fprintf(ML, "%e\t%e\t", particles[i].z, particles[i].gam/fJet/fFesc);
					fprintf(ML, "%e\n", mass[i]);
				}
				
				fclose(ML);
				
				delete[] nOnLocTemp;
				delete[] p_init;
				delete[] mags;
				delete[] mass;
			}
			
			MPI_CHECK(MPI_Bcast(particles, nOnFinal, mpi_source, 0, MPI_COMM_WORLD));
			
			CUDA_CHECK( cudaFree(source_dev)); //CHECKXXX
			CUDA_CHECK( cudaMalloc((void **)&source_dev, nOnFinal*sizeof(source)) );
			CUDA_CHECK( cudaMemcpy( source_dev, particles, nOnFinal*sizeof(source), cudaMemcpyHostToDevice) );
			
			// Set framework for adaptive halo calc
			int disp = 0;
			for(int i=0; i<nLayers-1; i++)
			{
				float fOnLoc = (float) nOnLoc[nGridDom];
			
				haloCount[i] = (int) (fOnLoc/nLayers);
				haloDisp[i] = disp;
			
				disp += haloCount[i];
			}
		
			haloCount[nLayers-1] = nOnLoc[nGridDom] - disp;
			haloDisp[nLayers-1] = disp;
			
			// Send workload parameters to each node
			/*MPI_CHECK(MPI_Bcast(dtRad, nLayers, MPI_FLOAT, 0, commGridLoc));
			MPI_CHECK(MPI_Bcast(haloCount, nLayers, MPI_INT, 0, commGridLoc));
			MPI_CHECK(MPI_Bcast(haloDisp, nLayers, MPI_INT, 0, commGridLoc));*/
			
			iFileNum++;
			nDuty = 0;
			if(commRank == 0) {printf("T = %f Myr. Currently tracking %d halos.\n", fT, nOnFinal);}
		}
		
		fTslice = time(1.0 / nfA[iFileNum] - 1.0);
		float fTslice0 = time(1.0 / nfA[iFileNum-1] - 1.0);
		float fTsliceDuty = fTslice0*(1.0-fDuty) + fTslice*fDuty;
		
		//	rad step: computer the Flux array	//
		// 1. Allocate on all nodes and Bcast/Scatter arrays:
		MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
		if(commRank==0) {dT0=MPI_Wtime(); printf("0...\n");}
		
		// 2.-5. Done by rad, in 3Dtransfer.cu. Begin by making FluxArray = 0
		/*float* dEArray = new float[nSize];
		for (int i = 0; i<nSize; i++)
		{
			dEArray[i] = 0;
			//dEArray_red[i] = 0;
			for(int nSpe=0; nSpe<SPECIES; nSpe++)
			{
				FluxArray[i + nSpe*nSize] = 0;
			}
		}*/
		memset(FluxArray, 0.0, nSize*SPECIES*sizeof(float));
		cudaMemset(FluxArray_dev, 0.0, nSize*SPECIES*sizeof(float));
		
		memset(dEArray, 0.0, nSize*sizeof(float));
		cudaMemset(dEArray_dev, 0.0, nSize*sizeof(float));
		
		/*if(commRank == 0)
		{
			for(int i=0; i<10; i++)
			{
				Ray ray;
				ray.set_pix(0, i);
				int nside = ray.get_Nside();
				int Npix = 12*nside*nside;
				
				float Rsplit = sqrt(Npix/12.56636/OMEGA_RAY);
				
				printf("ord = %d\tRsplit = %e\n", i, Rsplit);
			}
		}*/
		
		float nfSback[FREQ_BIN_NUM] = {0};
		float nfSback_red[FREQ_BIN_NUM] = {0};
		MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
		
		float dtRadLoc=0.0, dtRadTemp;
		if(commRank==0)
			printf("fTs0 = %e, fTs = %e, fTsD = %e, fT=%e\n", fTslice0, fTslice, fTsliceDuty, fT+fT0);
		/*if(fT+fT0 > fTsliceDuty && nDuty == 0)
		{
			nOnFinal = 0;
			for(int i=0; i<8; i++)
			{
				haloCount[i] = 0;
			}
		}*///TEST
		
		// Particle and Ray info for the tracer to know what to do
		int PartInfo[4];
		
		// Total number of particles to copy
		PartInfo[0] = nOnFinal;
		// Number of particles to initiate
		PartInfo[1] = haloCount[nGridLoc];
		// Index of first particle to look at
		PartInfo[2] = nOnDisp[nGridDom] + haloDisp[nGridLoc];

		int nPartRem;
		MPI_CHECK(MPI_Allreduce( (PartInfo+1), &nPartRem, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
		
		// Create Ray buffers
		// Size dictated by domain geometry
		Ray *SendBuf = new Ray[8*NUM_BUF];
		Ray *RecvBuf = new Ray[8*NUM_BUF];
		//printf("PartInfo[1] ??? %d\n", PartInfo[1]);
		// Ray tracer decreases PartInfo[1] as it works through the particles
		
		while(nPartRem > 0)
		{
			// Tell the tracer whether to trace from sources or buffer
			PartInfo[3] = 0;
			//printf("Domain %d, PartInfo = (%d, %d, %d, %d)\n", domain.get_id(), PartInfo[0], PartInfo[1], PartInfo[2], PartInfo[3]);
			
			// Number of rays in each buffer
			int ndBuf[8];
			for(int i=0; i<8; i++)
			{
				ndBuf[i] = 0;
			}
			
			// Call initial ray tracing routine
			MPI_CHECK(MPI_Barrier(commGridDom));
			if(commRank==0) {dT0a=MPI_Wtime();}
			
			rad(	DensArray_dev, x_NArray_dev, source_dev, FluxArray_dev, dEArray_dev, SendBuf,
					PartInfo, ndBuf, BOX_SIZE, fA, &dtRadTemp, nfSback, domain, fDt);
			// CHECKING FOR ARRAY TROUBLES //
			float check3[1]={0};
			int asdf3=0;
			for(int i=0; i<DIMX*DIMX*DIMX/8; i++)
			{
				//CUDA_CHECK( cudaMemcpy( check3, x_NArray_dev+i, sizeof(float), cudaMemcpyDeviceToHost) );
				asdf3 += check3[0];
				if(isnan(check3[0]))
				{
					printf("%e HOST PROBLEM, at index %d\n", check3[0], i);
				}
			}
			//printf("HOST %d PULLED = %d\n", commRank, asdf3);
			// CHECKING FOR ARRAY TROUBLES //
			int asdf=0;
			for(int i=0; i<nSize; i++)
			{
				asdf += x_NArray[i];
				if(isnan(x_NArray[i]))
				{
					//printf("%e IS A PROBLEM, should be %d\n", x_NArray[i], i);
				}
			}
			//printf("asdf = %d\n", asdf);
			MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
			//assert(false);
			
			MPI_CHECK(MPI_Barrier(commGridDom));
			
			if(commRank==0) {dT0b=MPI_Wtime();}
			dtRadLoc += dtRadTemp;
			
			//printf("PartInfo[1] = %d, PartInfo[1] = %d, PartInfo[2] = %d\n", PartInfo[0], PartInfo[1], PartInfo[2]);
			// Distribute buffers and trace appropriately
			PartInfo[3] = 1;
			
			// Loop for propagating rays between domains
			int MAX_LOOPS = 8;
			for(int j=0; j<MAX_LOOPS; j++)
			{
				MPI_CHECK(MPI_Barrier(commGridDom));
				//if(commRank == 0) printf("|--------------------------------------------------|\n");
				//printf("Domain %d, PartInfo = (%d, %d, %d, %d)\n", domain.get_id(), PartInfo[0], PartInfo[1], PartInfo[2], PartInfo[3]);
				double dTRay = MPI_Wtime();
				MPI_Request mpi_req[8];
				MPI_Status mpi_stat[8];
				int ndBufRecv[8];
				
				// Non-blocking Irecv requests
				for(int i=0; i<8; i++)
				{
					int d_buf = i*NUM_BUF;
					if(i != domain.get_id())
					{
						MPI_CHECK( MPI_Irecv(	RecvBuf + d_buf, NUM_BUF, mpi_ray,
												i, 0, commGridDom,
												mpi_req + i ) );
					}
				}
				
				// Send the messages
				for(int i=0; i<8; i++)
				{
					int d_buf = i*NUM_BUF;
					if(i != domain.get_id())
					{
						MPI_CHECK( MPI_Send(	SendBuf + d_buf, ndBuf[i], mpi_ray,
												i, 0, commGridDom ) );
						
						MPI_CHECK( MPI_Wait( mpi_req + i, mpi_stat + i) );
						MPI_CHECK( MPI_Get_count( mpi_stat + i, mpi_ray, ndBufRecv + i) );
						if(ndBufRecv[i] != 0)
						{
							//printf("Node %d recv %d rays from dim %d\n", domain.get_id(), ndBufRecv[i], i);
						}
					}
					else
					{
						// Note that we aren't receiving anything from self
						ndBufRecv[i] = 0;
					}
				}
				MPI_CHECK(MPI_Barrier(commGridDom));
				
				// Copy received rays into the outgoing buffer
				for(int i=0; i<8; i++)
				{
					size_t sizeR = sizeof(SendBuf[0]);
					int d_buf = i*NUM_BUF;
					ndBuf[i] = ndBufRecv[i];
					memcpy(SendBuf + d_buf, RecvBuf + d_buf, ndBuf[i]*sizeR);
				}
				MPI_CHECK(MPI_Barrier(commGridDom));
			
				/*int neg = 0;
				for(int i=0; i<3; i++)
				{
					int disp = i*NUM_BUF;
					for(int j=0; j<ndBuf[i]; j++)
					{
						if(SendBuf[disp + j].get_dom() == -1)
						{
							neg++;
						}
					}
				}*/
				//printf("Domain %d recv (%d, %d, %d)\n", domain.get_id(), ndBuf[0], ndBuf[1], ndBuf[2]);
				MPI_CHECK(MPI_Barrier(commGridDom));
				//if(nGridDom == 0)
					//printf("-------------------- %f s --------------------\n", (MPI_Wtime() - dTRay));
			
				float dummy;
				rad(	DensArray_dev, x_NArray_dev, source_dev, FluxArray_dev,
						dEArray_dev, SendBuf, PartInfo, ndBuf,
						BOX_SIZE, fA, &dummy, nfSback, domain, fDt);
				
				/*nBufTot = 0;
				for(int j=0; j<3; j++)
				{
					int temp;
					MPI_CHECK(MPI_Allreduce(ndBuf+j, &temp, 1, MPI_INT, MPI_SUM, commGridDom));
					nBufTot += temp;
				}*/
				/*if(j==1)
					break;*/
			}
			
			MPI_CHECK(MPI_Allreduce( PartInfo+1, &nPartRem, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
			//printf("nOnFinal = %d, PartInfo[0] = %d, PartInfo[1] = %d\n", nOnFinal, nPartRem, PartInfo[1]);
		}
		
		delete[] SendBuf;
		delete[] RecvBuf;
		
		// Copy flux arrays off the devices
		CUDA_CHECK( cudaMemcpy(	FluxArray, FluxArray_dev, nSizeBytes*SPECIES, cudaMemcpyDeviceToHost) );
		CUDA_CHECK( cudaMemcpy(	dEArray, dEArray_dev, nSizeBytes, cudaMemcpyDeviceToHost) );
		
//		cout << "Process #" << commRank << " has finished the rad kernel." << endl;
		
		// 6. Reduce Flux and Energy Arrays
		float* FluxArray_red = new float[nSize*SPECIES];
		float* dEArray_red = new float[nSize];
		MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
		MPI_CHECK(MPI_Allreduce(	FluxArray, FluxArray_red, nSize*SPECIES,
									MPI_FLOAT, MPI_SUM, commGridLoc));
		MPI_CHECK(MPI_Allreduce(	dEArray, dEArray_red, nSize,
									MPI_FLOAT, MPI_SUM, commGridLoc));
		MPI_CHECK(MPI_Allreduce(	nfSback, nfSback_red, FREQ_BIN_NUM,
									MPI_FLOAT, MPI_SUM, commGridLoc));
		memset(nfSback, 0, FREQ_BIN_NUM*sizeof(float));
		MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
		
		// Copy the reduced flux array back onto the devices CHECKXXX
		CUDA_CHECK( cudaMemcpy(	FluxArray_dev, FluxArray_red, nSizeBytes*SPECIES, cudaMemcpyHostToDevice) );
		memcpy(FluxArray, FluxArray_red, nSizeBytes*SPECIES);
		delete[] FluxArray_red;
		
		CUDA_CHECK( cudaMemcpy(	dEArray_dev, dEArray_red, nSizeBytes, cudaMemcpyHostToDevice) );
		memcpy(dEArray, dEArray_red, nSizeBytes);
		delete[] dEArray_red;
		
		/*if(commRank == 0)
		{
			escape.open("../dat/escape.dat", ios::app);
			for(int nBin=0; nBin<FREQ_BIN_NUM; nBin++)
			{
				escape << nfSback_red[nBin] << "\t";
			}
			escape << ndot << "\n";
			escape.close();
		}*/
		
		MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
		if(commRank==0) {dT1=MPI_Wtime(); printf("1...\n");}
		
		// 7. Calculate the background:
		if(commRank == 0)
		{
			//float z = redshift(fT);
			//float dz = redshift(fT-fDt) - z;
			
			float z = redshift(fTslice);
			float dz = redshift(fTslice - fDt) - z;
			
			float nb = 2.43e-7*pow(1.+z,3.);
			// CHECKXXX
			float xh = ndSum[2];
			float xhe = ndSum[3];
			
			float nd = ndot*2.43e-7*pow(1+z, 3);	// Putting the correct units for the background code
			
			//cout << endl << "xh = " << xh << "\txhe = " << xhe << "\tndot = " << nd << endl;
			//cout << "a = " << fA << "\tz = " << (1./fA - 1.) << "\tt = " << fT << endl;
			
			float Back_full[2] = {0};
			
			float Jloc[BACK_BINS];
			memcpy(Jloc, Jback, BACK_BINS*sizeof(float));
			
			Jnew(Jloc, nd, dz, z, Nuback, nb, xh, xhe, 5.0, 0.0, BACK_BINS);
			Jnew(Jback, nd, dz, z, Nuback, nb, xh, xhe, 0.0, 0.0, BACK_BINS);
			Background[0] = 0.0;
			Background[1] = 0.0;
			Background[2] = 0.0;
			Background[3] = 0.0;
			float dLogNu = log(Nuback[1]/Nuback[0])*4*3.14159;
			
			for(int iback=0; iback<BACK_BINS; iback++)
			{
				//Background[0] += Jloc[iback]*sig_H(Nuback[iback])*dLogNu/3.15e13;
				//Background[1] += Jloc[iback]*sig_He(Nuback[iback])*dLogNu/3.15e13;
				
				Background[0] += Jloc[iback]*sig_H(Nuback[iback])*dLogNu/3.15e13;
				//Background[1] += Jloc[iback]*sig_He(Nuback[iback])*dLogNu/3.15e13;
				
				Background[2] += (Nuback[iback] - 13.6)*Background[0];
				//Background[3] += (Nuback[iback] - 24.6)*Background[0];
				
				Back_full[0] += Jback[iback]*sig_H(Nuback[iback])*dLogNu/3.15e13;
				Back_full[1] += Jback[iback]*sig_He(Nuback[iback])*dLogNu/3.15e13;
			}
			//Background[0] /= 3.1536e13*fDt*2.43e-7*pow(1+z, 3);
			//Background[1] /= 3.1536e13*fDt*2.43e-7*pow(1+z, 3);
			
			cout << "Background Check: " << Background[0] << "\t" << Background[1] << "\n";
			cout << "Background Check: " << Background[2] << "\t" << Background[3] << "\n";
			cout << "Background Check: " << Back_full[0] << "\t" << Back_full[1] << "\n";
		}
		//Background[0] = 0.0;
		Background[1] = 0.0;
		//Background[2] = 0.0;
		Background[3] = 0.0;
		
		MPI_CHECK(MPI_Bcast(Background, 4, MPI_FLOAT, 0, MPI_COMM_WORLD));
		MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
		
		// Determine the time step
		float max_ddt[2] = {0};
		float max_ddt_M[2] = {0};
		dt_H(max_ddt, DensArray_dev, x_NArray_dev, FluxArray_dev, EArray, Background, BOX_SIZE, fA, domain);
		//printf("TEST -> %e %e %e %e\n", DensArray[0], x_NArray[0], FluxArray[0], fA);
		float fDtLoc;
		
		MPI_CHECK(MPI_Allreduce(max_ddt, max_ddt_M, 2, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD));
		
		float fDt_H = 0.1/3.154e13/max_ddt_M[0];
		float fDt_I = 100.0/3.154e13/max_ddt_M[1];
		if(nStepType == 1)
		{
			float dtnew = fDt_H;
			fDtLoc = MIN(2.0*fDt, dtnew);
			//fDtLoc = MAX(fDt, 1.e-5);
		}
		else if(nStepType == 2)
		{
			float dtnew = MAX(fDt_I,fDt_H);
			fDtLoc = MIN(2.0*fDt, dtnew);
			//fDtLoc = MIN(fDt, 1.e0);
		}
		else
		{
			fDtLoc = fDtConst;
		}
		
		// We need the shortest time step for every sub-volume
		//MPI_CHECK(MPI_Allreduce(&fDtLoc, &fDt, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD));
		if(commRank == 0) printf("Step type %d, fDt = %f, fDtLoc = %f\n", nStepType, fDt, fDtLoc);
		fDtLoc = MIN(fDtLoc, fDtConst);
		// Ensure that the time step can't be larger than the time between file outputs
		fDt = MIN(fDtLoc, fDtFile);
		if(commRank==0) {printf("dt = %e\tdt_H = %e\tdt_I = %e\t\n", fDt, fDt_H, fDt_I);}
		//if(commRank==0) {printf("dt = %e\tdt_H = %e\tdt_I = %e\t\n", fDt, max_ddt[0], max_ddt[1]);}
		
		// Perform chemistry calculation
		error_loc = 0;
		ion(DensArray_dev, x_NArray_dev, FluxArray_dev, EArray, dEArray_dev, Background, &fDt, &error_loc, fA, domain);
		MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
		
		// Copy fraction and temperature arrays off the devices
		CUDA_CHECK( cudaMemcpy(	x_NArray, x_NArray_dev,
								nSizeBytes*SPECIES, cudaMemcpyDeviceToHost) );
		//CUDA_CHECK( cudaMemcpy(	EArray, EArray_dev,
		//						nSizeBytes*SPECIES, cudaMemcpyDeviceToHost) );
		
		// Calculate averages before looking at the background
		// ndSum[i]:
		// 0, 1: xHI, xHeI
		// 2, 3: nHI, nHeI
		memset(ndSum_loc, 0, 6*sizeof(double));
		copy(ndSum, ndSum+6, ndSum_old);
		double temp = 0;
		for(int i=0; i<nSize; i++)
		{
			float xHI = x_NArray[i];
			float xHeI = x_NArray[i + nSize];
			
			
			ndSum_loc[0] += xHI;
			ndSum_loc[1] += xHeI;
			
			ndSum_loc[2] += DensArray[i]*xHI;
			ndSum_loc[3] += DensArray[i]*xHeI;
			
			//ndSum_loc[1] += FluxArray[i];
			//if(FluxArray[i] > temp){temp = FluxArray[i];}
			ndSum_loc[4] += FluxArray[i];
			ndSum_loc[5] += isnan(FluxArray[i]);
		}
		MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
		//printf("ndSum[5] of rank (%d) = %e\n", commRank, ndSum_loc[5]);
		//ndSum_loc[1] = temp;
		
		for(int i=0; i<6; i++)
		{
			ndSum_loc[i] /= nSize;
		}
		MPI_CHECK( MPI_Allreduce(ndSum_loc, ndSum, 6, MPI_DOUBLE, MPI_SUM, commGridDom) );
		for(int i=0; i<6; i++)
		{
			ndSum[i] = ndSum[i]/nDomain;
		}
		
		// Output the temperature for testing
		double dError, dError_loc = (double) error_loc;
		MPI_CHECK( MPI_Allreduce(&dError_loc, &dError, 1, MPI_DOUBLE, MPI_SUM, commGridDom) );
		if(commRank == 0)
		{
			printf("Summed test variable = %e\n", dError);
			
			printf("ndSum (old) = ");
			for(int i=0; i<6; i++)
			{
				printf("%0.3e, ", ndSum_old[i]);
			}
			printf("\n");
			printf("ndSum (new) = ");
			for(int i=0; i<6; i++)
			{
				printf("%0.3e, ", ndSum[i]);
			}
			printf("\n");
		}
		
		if(commRank==0)
		{
			// Change in neutral fraction:
			double deltaN = -(ndSum[0] - ndSum_old[0]);
			
			// Escaped photons
			double escaped = 0;
			for(int nBin=0; nBin<FREQ_BIN_NUM; nBin++)
			{
				escaped += nfSback_red[nBin];
			}
			
			// Emitted photons
			double emitted = (SINGLE_SOURCE - escaped)*pow(3.086e24,2);
			emitted *= fDt*1.0e6*3.154e7;				// multiply by time step
			emitted /= 1.e-3*pow(BOX_SIZE*3.086e24,3);	// divide by number of atoms
			
			// Recombinations
			double recombined = dError;
			recombined *= fDt*1.0e6*3.154e7;	
			recombined /= pow(DIMX,3);
			//recombined /= 1.e-3*pow(BOX_SIZE*3.086e24,3);	// divide by number of atoms
			//recombined *= fDt*1.0e6*3.154e7;				// multiply by time step
			
			// Ratio
			double ratio = 1-(deltaN+recombined)/emitted;
			
			// Cumulative
			emitted_total += emitted;
			recombined_total += recombined;
			double ratio_total = 1-((1-ndSum[0]) + recombined_total)/emitted_total;
			
			// Flux Array
			double gamma_array = ndSum[4];
			gamma_array *= fDt*1.0e6*3.154e7;
			double ratio_gamma = 1 - gamma_array/emitted;
			
			printf("Time: %e\n", fT);
			printf("\nNeutral Change: %e\n", deltaN);
			printf("Emissions: %e\n", emitted);
			printf("Recombinations: %e\n", recombined);
			printf("Ratio: %e\n\n", ratio);
			
			printf("Background:");
			for(int nBin=0;nBin<FREQ_BIN_NUM;nBin++)
			{
				printf("%e\t", nfSback_red[nBin]);
			}
			printf("\n");
			
			printf("Array check: %f\n\n", ratio_gamma);
			
			conservation.open("../dat/conservation.dat", ios::app);
			conservation << scientific;
			conservation.precision(3);
			conservation << fT << "\t";
			conservation << deltaN << "\t";
			conservation << emitted << "\t";
			conservation << recombined << "\t";
			//conservation << fixed;
			conservation << ratio << "\t";
			conservation << 1-ndSum[0] << "\t";
			conservation << emitted_total << "\t";
			conservation << recombined_total << "\t";
			conservation << ratio_total << "\t";
			conservation << ratio_gamma << "\n";
			conservation.close();
		}
		
		MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
		if(commRank==0) {dT2=MPI_Wtime(); printf("2...\n");}
		
		//	output as necessary					//
		if ( (fT < fTFile) && (fT+fDt >= fTFile))
		{
			float fOutTime = MPI_Wtime();//CHECKXXX
			fTFile += fDtFile;
			counter++;
			
			if(nGridLoc == 0)
			{
				if(commRank == 0)
				{
					x_NArrayHost = new float[nSizeHost*SPECIES];
					FluxArrayHost = new float[nSizeHost*SPECIES];
					EArrayHost = new float[nSizeHost];
				}
				if(commRank == 0)
					printf("Output 0\n");
				for(int nSpe=0; nSpe < SPECIES; nSpe++)
				{
					int dloc = nSize*nSpe;
					int dhost = nSizeHost*nSpe;
					
					MPI_CHECK(MPI_Gather(	x_NArray + dloc, nSize, MPI_FLOAT, 
											x_NArrayHost + dhost, nSize, MPI_FLOAT,
											0, commGridDom));
					
					MPI_CHECK(MPI_Gather(	FluxArray + dloc, nSize, MPI_FLOAT, 
											FluxArrayHost + dhost, nSize, MPI_FLOAT,
											0, commGridDom));
				}
				
				MPI_CHECK(MPI_Gather(	EArray, nSize, MPI_FLOAT, 
										EArrayHost, nSize, MPI_FLOAT, 0, commGridDom));
				if(commRank == 0)
					printf("Output 1\n");
				if(commRank == 0)
				{
					for(int nSpe=0; nSpe < SPECIES; nSpe++)
					{
						arrayChangeInv(x_NArrayHost + nSpe*nSizeHost);
						arrayChangeInv(FluxArrayHost + nSpe*nSizeHost);
					}
					arrayChangeInv(EArrayHost);
				
					output(FluxArrayHost, EArrayHost, x_NArrayHost, nSizeHost, counter);
				
					delete[] x_NArrayHost;
					delete[] FluxArrayHost;
					delete[] EArrayHost;
			
					// Print the current background
					sprintf(name, "../dat/background/spec%03d.dat", counter);
					FILE *fp;
					fp = fopen(name, "w");
					for(int iback=0; iback<BACK_BINS; iback++)
					{
						fprintf(fp, "%e\t%e\n", Nuback[iback], Jback[iback]);
					}
					fclose(fp);
			
					// Print the current loop variables
					sprintf(name, "../dat/restart/restart%03d.dat", counter);
					FILE *re;
					re = fopen(name, "w");
					fprintf(re, "%d\t%d\t%d\t", counter, step, iFileNum);
					fprintf(re, "%e\t%e\t%e\t%e", fT, fDt, fTFile, fDtFile);
					fclose(re);
				}
				if(commRank == 0)
				{
					float fOutTime1 = MPI_Wtime();
					fOutTime = fOutTime1-fOutTime;
					printf("Output 2 took %f s\n", fOutTime);
				}
			}
		}
		
		if (step != nLoopInit)
		{
			// x_HI, x_HeI, n_e, gam_v, gam_m, rec_v, rec_m, T_v, T_m
			double ndOut[14] = {0};
			
			for(int ind=0; ind < nSize; ind++)
			{
				double xe_i = (1.0-Y_P)*(1.0-x_NArray[ind]);
				xe_i += 0.25*Y_P*(1.0-x_NArray[nSize + ind]);
				xe_i *= pow(fA,-3.0);
				double ne_i = DensArray[ind]*xe_i;
				
				double T = EArray[ind]/((3./2.)*8.6173303e-5)/(1.0+xe_i);
				double alpha;
				if(T < 1.e0)
				{
					alpha = 2.17e-10;
				}
				else if(T < 5.0e3)
				{
					alpha = 2.17e-10*powf(T,-0.6756);
					//al[1] = 0.0;
				}
				else
				{
					alpha = 4.36e-10*powf(T,-0.7573);
					//al[1] = 1.50e-10*powf(T,-0.6353);
				}
				
				double rec_i = alpha*(1.0-x_NArray[ind])*ne_i;
				
				ndOut[1] += ne_i;
				
				ndOut[2] += x_NArray[ind]*DensArray[ind];
				ndOut[3] += x_NArray[nSize + ind]*DensArray[ind];
				
				ndOut[5] += FluxArray[ind];
				ndOut[6] += FluxArray[ind]*DensArray[ind];
				
				ndOut[7] += rec_i;
				ndOut[8]+= rec_i*DensArray[ind];
				
				ndOut[10] += T;
				ndOut[11] += T*DensArray[ind];
			}
			// Divide by average comoving density for mass weights
			int massW[5] = {2, 3, 6, 8, 11};
			for(int i=0; i<5; i++)
			{
				ndOut[massW[i]] /= 2.43e-7;
			}
			
			// Outputs for collisional ionization
			//col[0] = 5.85e-11*powf(T,0.5)*expf(-157809.1/T);
			//col[1] = 2.38e-11*powf(T,0.5)*expf(-285335.4/T);
			//dydx[0] = -y[3] - col[0]*y[0]*ne + al[0]*(1.0-y[0])*ne;
			
			double ndOut_red[14];
			MPI_CHECK(MPI_Allreduce(ndOut, ndOut_red, 14, MPI_DOUBLE, MPI_SUM, commGridDom));
			memcpy(ndOut, ndOut_red, 14*sizeof(double));
			
			for(int i=0; i<14; i++)
			{
				ndOut[i] /= nSize*nDomain;
			}
			//Troubleshooting
			float A0=0, A1=0;
			for(int i=0; i<nSize; i++)
			{
				if(FluxArray[i] > A0)
					A0 += FluxArray[i];
				if(EArray[i] > A1)
					A1 += EArray[i];
			}
			
			//Troubleshooting
			//printf("ID = %d, F = %e, E = %e\n", domain.get_id(), A0/nSize, A1/nSize);
			//float ndotphys = ndot*pow(nfA[iFileNum],-3.0); // CHECKXXX
			
			ndOut[0] = fT;
			ndOut[4] = ndot;
			ndOut[9] = Background[0]/fDt;
			
			ndOut[12] = 1.0/3.15e13/max_ddt_M[0];
			ndOut[13] = 1.0/3.15e13/max_ddt_M[1];
			
			if(commRank == 0)
			{
				summary = fopen("../dat/sum.dat", "a");
				
				// ndOut: fDt, x_e, xHI, xHeI, ndot, F_V, F_M, G_V, G_M, Back,
				// ndOut: T_V, T_M, dt_H, dt_I
				for(int i=0; i<14; i++)
				{
					fprintf(summary, "%e\t", ndOut[i]);
				}
				fprintf(summary, "\n");
				fclose(summary);
			}
		}
		
		// Determine adaptive work distribution
		MPI_CHECK(MPI_Allgather(&dtRadLoc, 1, MPI_FLOAT, dtRad, 1, MPI_FLOAT, commGridLoc));
		
		float totalTime = 0.0;
		for(int i=0; i<nLayers; i++)
			totalTime += dtRad[i];
		
		int disp = 0;
		for(int i=0; i<nLayers-1; i++)
		{
			/*int nRemCount = nOnLoc[nGridDom] - haloDisp[i];
			float fRemCount = (float) nRemCount;
			
			float adjFac = totalTime/dtRad[i]/nLayers;
			int adjCount = (int) (adjFac*haloCount[i]);
			
			// Stop the adjusting algorithm from getting out of hand
			if(adjCount >= nRemCount)
				haloCount[i] = (int) (nRemCount/(nLayers-i));
			else
				haloCount[i] = adjCount;
			
			haloDisp[i] = disp;
			
			disp += haloCount[i];*/
			int nRemCount = nOnLoc[nGridDom] - haloDisp[i];
			float fRemCount = (float) nRemCount;
			
			float adjFac = totalTime/dtRad[i]/nLayers;
			int adjCount = (int) (adjFac*haloCount[i]);
			
			// Stop the adjusting algorithm from getting out of hand
			if(adjCount >= nRemCount)
				haloCount[i] = (int) (nRemCount/(nLayers-i));
			else
				haloCount[i] = adjCount;
			
			haloDisp[i] = disp;
			
			disp += haloCount[i];
		}
		
		haloCount[nLayers-1] = nOnLoc[nGridDom] - disp;
		haloDisp[nLayers-1] = disp;
		
		
		int *Counts = new int[nDomain*nLayers];
		float *Times = new float[nDomain*nLayers];
		
		MPI_CHECK(MPI_Gather(haloCount, nLayers, MPI_INT, Counts, nLayers, MPI_INT, 0, commGridDom));
		MPI_CHECK(MPI_Gather(dtRad, nLayers, MPI_FLOAT, Times, nLayers, MPI_FLOAT, 0, commGridDom));
		
		if(nGridLoc == 0)
		{
			if(nGridDom == 0)
			{
				printf("TArray:\n%f\t%f\t%f\n", dT1-dT0, dT0b-dT0a, dT2-dT1);
				
				int k=0;
				for(int j=0; j<nLayers; j++)
				{
					
					for(int i=0; i<nDomain; i++)
					{
						printf("%d\t", Counts[k]);
						k++;
					}
					printf("\n");
				}
				
				k=0;
				for(int j=0; j<nLayers; j++)
				{
					for(int i=0; i<nDomain; i++)
					{
						printf("%f\t", Times[k]);
						k++;
					}
					printf("\n");
				}
			}
		}
		
		
		MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
		if(commRank==0) printf("Step #%d, rad step: %f, ion step: %f\n", step, dT1-dT0, dT2-dT1);
		
		fT+=fDt;
		/*if(commRank != -1)
		{
			float AA=0, BB=0;
			for(int i=0; i<nSize; i++)
			{
				AA += x_NArray[i];
			}
			AA /= nSize;
			printf("Average at %d at the end = %f\n", domain.get_id(), AA);
		}*/
	}
	
	CUDA_CHECK( cudaFree(DensArray_dev) );
	CUDA_CHECK( cudaFree(x_NArray_dev) );
	CUDA_CHECK( cudaFree(FluxArray_dev) );
	CUDA_CHECK( cudaFree(source_dev));
	
	CUDA_CHECK( cudaDeviceReset() );
	
	delete[] particles;
	delete[] Nuback;
	delete[] Jback;
	
	// Free host memory
	delete[] DensArray;
	delete[] FluxArray;
	delete[] x_NArray;
	delete[] dEArray;
	
	if(commRank == 0) printf("Total run time: %f\n", MPI_Wtime()-dTStart);

	MPI_CHECK(MPI_Finalize());
	
	return 0;
}

inline void arrayChange(float* array)
{
	int mainIndex=0;
	int subIndex;
	int size = DIMX*DIMX*DIMX;
	int half = DIMX/2;
	
	float* buffer = new float[size];
	memcpy(buffer, array, size*sizeof(float));
	
	for(int k=0; k<DIMX; k++)
	{
		for(int j=0; j<DIMX; j++)
		{
			for(int i=0; i<DIMX; i++)
			{
				int tag = 4*(k/half) + 2*(j/half) + 1*(i/half);
				subIndex = half*half*(k%half);
				subIndex += half*(j%half);
				subIndex += i%half;
				subIndex += tag*half*half*half;
				
				array[subIndex] = buffer[mainIndex];
				
				mainIndex++;
			}
		}
	}
	
	delete[] buffer;
}
inline void arrayChangeInv(float* array)
{
	int mainIndex=0;
	int subIndex;
	int size = DIMX*DIMX*DIMX;
	int half = DIMX/2;
	
	float* buffer = new float[size];
	memcpy(buffer, array, size*sizeof(float));
	
	for(int k=0; k<DIMX; k++)
	{
		for(int j=0; j<DIMX; j++)
		{
			for(int i=0; i<DIMX; i++)
			{
				int tag = 4*(k/half) + 2*(j/half) + 1*(i/half);
				subIndex = half*half*(k%half);
				subIndex += half*(j%half);
				subIndex += i%half;
				subIndex += tag*half*half*half;
				
				array[mainIndex] = buffer[subIndex];
				
				mainIndex++;
			}
		}
	}
	
	delete[] buffer;
}

inline void BinGridRead(float* array, char* filename, int dim)
{
	ifstream ifsFile (filename, ios::in | ios::binary);
	ifsFile.seekg (0, ios::beg);
	
	unsigned long int num = dim*dim*dim;
	
	for (unsigned long int i = 0; i < num; i++)
	{
		float buff;
		
		ifsFile.read((char*)&buff, sizeof(float));
		array[i] = buff;
	}
	
	ifsFile.close();
}

// Shut down MPI cleanly if something goes wrong
void my_abort(int err)
{
    cout << "Test FAILED\n";
    MPI_Abort(MPI_COMM_WORLD, err);
}

// Returns magnitude of Nth halo
void halomatch(float* L, unsigned n, float a, float V, int mod)
{
	float z = 1.0/a-1.0;
	float lmin = 0, lmax = -25, dl = 0.01;
	float nschechter = 0;
	float ll = lmax;
	
	for(unsigned i=0; i<n; i++)
	{
		float dndl = schechter(ll, model_Mstar(z, mod), model_log10phi(z, mod), model_alpha(z, mod));
		while(nschechter + dl*dndl*10 > (i+1)/V && nschechter < (i+1)/V)
		{
			//cout << "Decreasing step size" << dl << " -> " << dl/10 << endl;
			dl = dl/10;
		}
		while(nschechter < (i+1)/V)
		{
			ll += dl;
			dndl = schechter(ll, model_Mstar(z, mod), model_log10phi(z, mod), model_alpha(z, mod));
			nschechter += dl*dndl;
		}
		
		//cout << "N = " << nschechter << "\tM = " << ll << endl;
		//cout << "i = " << i << "\tdndnl = " << dndl << endl;
		L[i] = ll;
	}
}

double schechter(	double M,			// Mass
					double Mstar,		// Shechter parameters 
					double log10phi,		// "
					double alpha)		// "
{
	double temp = pow(10.,0.4*(Mstar-M));
	return 0.921034*pow(10.,log10phi)*pow(temp,alpha+1.)*pow(2.718281828,-temp);
}

double model_Mstar(double z, unsigned mod)
{
	if(mod == 1)	{return -20.37 + 0.30*(z - 6.);}
	else if(mod == 2)	{return -20.47 + 0.24*(z - 6.);}
	return -20.42 + 0.27*(z - 6.);
}

double model_log10phi(double z, unsigned mod)
{
	if(mod == 1)	{return -3.05 - 0.09*(z - 6.);}
	else if(mod == 2)	{return -2.97 - 0.05*(z - 6.);}
	return -3.01 - 0.07*(z - 6.);
}

double model_alpha(double z, unsigned mod)
{
	if(mod == 1)	{return -1.80 - 0.04*(z - 6.);}
	else if(mod == 2)	{return -1.88 - 0.08*(z - 6.);}
	return -1.84 - 0.06*(z - 6.);
}
