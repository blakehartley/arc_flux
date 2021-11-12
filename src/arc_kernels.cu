#include <cuda.h>
#include <math_functions.h>

#include "./inc/chealpix.h"
#include "./inc/chealpix.cu"
#include "./inc/rates.cu"

//__device__ float gfn[FREQ_BIN_NUM] = {0.5, 0.5};
//__device__ float gfn[FREQ_BIN_NUM] = {0.25, 0.25, 0.25, 0.25};

//__device__ float energy[FREQ_BIN_NUM] = {16.74, 24.65, 34.49, 52.06};
__device__ float energy[FREQ_BIN_NUM] = {13.60, 24.65, 34.49, 52.06};
float gfn[FREQ_BIN_NUM] = {1.0, 0.0, 0.0, 0.0};
//float gfn[FREQ_BIN_NUM] = {	0.5, 0.5, 0.0, 0.0};
//float gfn[FREQ_BIN_NUM] = {	0.277, 0.335, 0.2, 0.188};
//__device__ float gfn[FREQ_BIN_NUM] = {	0.277, 0.335, 0.2, 0.188};
//__device__ float gfn[FREQ_BIN_NUM] = {	0.465, 0.335, 0.2, 0.0};

//__device__ float energy[FREQ_BIN_NUM] = {17.00, 17.00, 17.00, 17.00};
//float gfn[FREQ_BIN_NUM] = {1.0, 0.0, 0.0, 0.0};
//__device__ float gfn[FREQ_BIN_NUM] = {1.0, 0.0, 0.0, 0.0};

__device__ float time(float redshift) {
	float h = 0.6711;
	float h0 = h*3.246753e-18;
	float omegam = 0.3;
	float yrtos = 3.15569e7;
	float time = 2.*powf((1. + redshift), -3. / 2.) / (3.*h0*powf(omegam, 0.5));
	time = time / (yrtos*1.e6);
	return time;
}

__device__ float redshift(float time) {
	float h = 0.6711;
	float h0 = h*3.246753e-18;
	float omegam = 0.3;
	float yrtos = 3.15569e7;
	time = time*yrtos*1.e6;
	float redshift = powf((3.*h0*powf(omegam, 0.5)*time / 2.), -2. / 3.) - 1.;
	return redshift;
}

/*inline __device__ void sigma(float sig[][FREQ_BIN_NUM])
{
	sig[0][0] = 3.4630495932136023e-18;	// 17.0 eV
	sig[0][1] = 0.0;	// 10.0 eV
	sig[0][2] = 0.0;	// 10.0 eV
	sig[0][3] = 0.0;	// 10.0 eV
	sig[1][0] = 0.0;	// 17.0 eV
	sig[1][1] = 0.0;	// 10.0 eV
	sig[1][2] = 0.0;	// 10.0 eV
	sig[1][3] = 0.0;	// 10.0 eV
}*/

inline __device__ void sigma(float sig[][FREQ_BIN_NUM])
{
	sig[0][0] = 6.2998764713e-18;	// 13.6 eV
	sig[0][1] = 1.23034961639e-18;	// 24.65 eV
	sig[0][2] = 4.70442276047e-19;	// 34.49 eV
	sig[0][3] = 1.40170632038e-19;	// 52.06 eV
	
	sig[1][0] = 0;	// 16.74 eV
	sig[1][1] = 7.77983571966e-18;	// 24.65 eV
	sig[1][2] = 4.20352986507e-18;	// 34.49 eV
	sig[1][3] = 1.90620549625e-18;	// 52.06 eV
	//sig[1][1] = 0;	// 24.65 eV
	//sig[1][2] = 0;	// 34.49 eV
}

__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ void step_bdf(float* yout, float* y, float* dGdx, float n, float E, float dt)
{
	float xe = (1.0-Y_P)*(1.0-y[0])+0.25*Y_P*(1.0-y[1]);
	float ne = n*xe;
	float T  = E/((3./2.)*8.6173303e-5)/(1.0+xe);
	int CASE = 1;
	
	// Recombination rates (Black81)
	float al[2];
	al[0] = rec_HII(T, CASE);
	al[1] = rec_HeII(T, CASE);
	
	// Collision Excitation
	float col[2];
	col[0] = col_HI(T);
	col[1] = col_HeI(T);
	
	for(int nBin=0; nBin < SPECIES; nBin++)
	{
		// Constant alpha
		//float source = al[nBin]*(1.0-y[nBin])*ne*dt + y[nBin];
		//float sink = 1.0 + (dGdx[nBin] + col[nBin]*ne)*dt;
		// Variable alpha
		float source = al[nBin]*(1.0-y[nBin])*ne*dt + y[nBin];
		float sink = 1.0 + (dGdx[nBin] + al[nBin]*ne + col[nBin]*ne)*dt;
		// Basic, no alpha
		//float source = y[nBin];
		//float sink = 1.0 + (dGdx[nBin])*dt;
		
		yout[nBin] = source/sink;
		/*yout[nBin] = y[nBin];
		yout[nBin] += dt*(-gam[nBin] - col[nBin]*ne*y[nBin]);
		yout[nBin] += dt*al[nBin]*(1.0-y[nBin])*ne;*/
		//yout[nBin] = y[nBin];
		//yout[nBin] += dt*(-dGdx[nBin]*y[nBin]);
		//yout[nBin] += dt*al[nBin]*(1.0-y[nBin])*ne;
		
		// Quadratic formulation:
		/*float A = al[nBin]*n;
		float B = -dGdx[nBin] - 2.0*al[nBin]*n;
		float C = al[nBin]*n;
		
		float Anew = -A*dt;
		float Bnew = 1.0-dt*B;
		float Cnew = -y[nBin]-C*dt;
		
		yout[nBin] = (-Bnew + sqrt(Bnew*Bnew-4*Anew*Cnew))/(2*Anew);
		yout[nBin] = MAX(yout[nBin],1.0e-10);
		yout[nBin] = MIN(yout[nBin],1.0e0);*/
	}
}

__device__ float dnHIdt(float* y, float* dGdx, float n, float E)
{
	float xe = (1.0-Y_P)*(1.0-y[0])+0.25*Y_P*(1.0-y[1]);
	float ne = n*xe;
	float T  = E/((3./2.)*8.6173303e-5)/(1.0+xe);
	int CASE = 1;
	
	// Recombination rates (Black81)
	float al[2];
	al[0] = rec_HII(T, CASE);
	al[1] = rec_HeII(T, CASE);
	
	// Collision Excitation
	float col[2];
	col[0] = col_HI(T);
	col[1] = col_HeI(T);
	
	int nBin=0;
	
	float x = -dGdx[nBin]*y[nBin] - col[nBin]*ne*y[nBin] + al[nBin]*(1.0-y[nBin])*ne;
	return x;
}

__device__ float lambda(float E, float* y, float n, float a)
{
	int CASE = 1;
	
	float dEdt = 0.0;
	float xe = (1.0-Y_P)*(1.0-y[0])+0.25*Y_P*(1.0-y[1]);
	float ne = n*xe;
	float T  = E/((3./2.)*8.6173303e-5)/(1.0+xe);
	
	float xHI = (1-Y_P)*y[0];
	float xHII = (1-Y_P)*(1.0 - y[0]);
	float xHeI = 0.25*Y_P*y[1];
	float xHeII = 0.0;
	//float xHeIII = 0.0;
	
	float colH = col_cool_HI(T)*ne*xHI;
	float colHe = col_cool_HeI(T)*ne*xHeI;
	
	float recH =  rec_cool_HII(T, CASE)*ne*xHII;
	float recHe = rec_cool_HeII(T, CASE)*ne*xHeII;
	
	float colexH = colex_HI(T)*ne*xHI;
	
	float brem = 1.42e-27*powf(T,0.5)*ne*ne*6.242e11/n;
	
	dEdt += colH + recH + colexH;
	dEdt += colHe + recHe;
	dEdt += brem;
	
	// Adiabatic cooling:
	float H0 = 67.11*3.241e-20;
	dEdt += 3.0*H0*0.5477*powf(a, -1.5)*E;
	
	return dEdt;
}

__device__ float thin_source(float source, float fraction)
{
	if(fraction < 1.e-30)
	{
		return 0.0;
	}
	else
	{
		return source/fraction;
	}
}

__global__ void timestep(	float* rate, float* dt, float* density, float* x_N,
							float* FluxArray, float* EArray, float* background,
							int dim, float L, float a)
{
	int nelements=dim*dim*dim;
	
	int i0 = blockIdx.x*blockDim.x+threadIdx.x;
	int j0 = blockIdx.y*blockDim.y+threadIdx.y;
	int k0 = blockIdx.z;

	int ind=i0+dim*j0+dim*dim*k0;
	
	float xn[SPECIES];
	float Gamma[SPECIES];
	
	for(int nSpe=0; nSpe < SPECIES; nSpe++)
	{
		xn[nSpe] = x_N[ind + nSpe*nelements];
		Gamma[nSpe] = FluxArray[ind + nSpe*nelements];
		Gamma[nSpe] += background[nSpe];	// Ionization rate for each species (per Myr)
	}
	
	float dens = density[ind]/(a*a*a);	// Baryonic number density
	float E = EArray[ind];				// Energy (temperature) per baryon
	
	// Subcycle loop to advance the chemistry
	// First we calculate the optically thin approximation of source terms:
	float dGdx[SPECIES];
	for(int nSpe=0; nSpe<SPECIES; nSpe++)
	{
		dGdx[nSpe] = thin_source(Gamma[nSpe], xn[nSpe]);
	}
		
	// Find the max time step to advance hydrogen
	float dxdt;
	dxdt = abs(dnHIdt(xn, dGdx, dens, E));
	
	float dx = (L/DIMX)*a;
	float sig = 1.111e7; // sig[0][0]*cm in a Mpc
	float tau = max(dens*sig*dx, 3.0);
	
	// dIdt
	atomicMax(rate+1, dxdt*tau);
	
	// dnHdt
	if(tau > 0.5)
		atomicMax(rate, dxdt/xn[0]);
	dt[ind] = dxdt/xn[0];
	
	__syncthreads();
}

__global__ void ionization1(	float dt, float* error, float* density, float* x_N,
							float* FluxArray, float* EArray, float* dEArray,
							float* background, int dim, float a)
{
	int nelements=dim*dim*dim;
	
	int i0 = blockIdx.x*blockDim.x+threadIdx.x;
	int j0 = blockIdx.y*blockDim.y+threadIdx.y;
	int k0 = blockIdx.z;

	int index=i0+dim*j0+dim*dim*k0;
	
	float fDtMyr = dt*3.15e13;
	
//	float t = time(1.0/a - 1.0);
	
	float xn[SPECIES];
	float xn_out[SPECIES];
	float Gamma[SPECIES];
	
	float fCumFlux = 0;
	
	for(int nSpe=0; nSpe < SPECIES; nSpe++)
	{
		xn[nSpe] = x_N[index+nSpe*nelements];
		Gamma[nSpe] = FluxArray[index+nSpe*nelements];
		Gamma[nSpe] += background[nSpe];	// Ionization rate for each species (per Myr)
		
		fCumFlux += xn[3+nSpe]*fDtMyr;
	}
	
	float dens = density[index]/(a*a*a);	// Baryonic number density
	float E = EArray[index];				// Energy (temperature) per baryon
	float dEdt = dEArray[index];
	//float dEdt = dEArray[index]+background[2]+background[3];
	
	// Subcycle loop to advance the chemistry
	// First we calculate the optically thin approximation of source terms:
	float dEdx, dGdx[SPECIES];
	for(int nSpe=0; nSpe<SPECIES; nSpe++)
	{
		dGdx[nSpe] = thin_source(Gamma[nSpe], xn[nSpe]);
	}
	dEdx = thin_source(dEdt, xn[0]);
	
	E = E + dEdt*dt;
	
	//step_bdf(xn_out, xn, dGdx, dens, E, dtSub);
	//step_bdf(float* yout, float* y, float* dGdx, float n, float E, float dt)
	float xe = (1.0-Y_P)*(1.0-xn[0])+0.25*Y_P*(1.0-xn[1]);
	float ne = dens*xe;
	float T  = E/((3./2.)*8.6173303e-5)/(1.0+xe);
	int CASE = 1;
	
	// Recombination rates (Black81)
	float al[2];
	al[0] = rec_HII(T, CASE);
	al[1] = rec_HeII(T, CASE);
	
	// Collision Excitation
	float col[2];
	col[0] = col_HI(T);
	col[1] = col_HeI(T);
	
	float fRec;
	for(int nSpe=0; nSpe < SPECIES; nSpe++)
	{
		//float source = al[nBin]*(1.0-y[nBin])*ne*dt + y[nBin];
		//float sink = 1.0 + (dGdx[nBin] + col[nBin]*ne)*dt;
		//yout[nBin] = source/sink;
		//yout[nBin] = y[nBin];
		//yout[nBin] += dt*(-gam[nBin] - col[nBin]*ne*y[nBin]);
		//yout[nBin] += dt*al[nBin]*(1.0-y[nBin])*ne;
		//xn[nBin] = xn[nBin];
		//xn[nBin] += dt*(-dGdx[nBin]*xn[nBin]);
		//xn_out[nBin] -= dGdx[nBin]*xn[nBin]*fDtMyr;
		
		//xn_out[nSpe] = xn[nSpe] - dGdx[nSpe]*fDtMyr;
		xn_out[nSpe] = xn[nSpe] - Gamma[nSpe]*fDtMyr;
		fRec = al[nSpe]*(1.0-xn[nSpe])*ne;
		xn_out[nSpe] += fRec*fDtMyr;
		
		atomicAdd(error, fRec);
		//atomicAdd(error, 1);
		
		//xn_out[nBin] = xn[nBin]/(1+dGdx[nBin]*1*fDtMyr);
		//yout[nBin] += dt*al[nBin]*(1.0-y[nBin])*ne;
	}
	for(int nSpe=0; nSpe<SPECIES; nSpe++)
	{
		xn[nSpe] = xn_out[nSpe];
		xn[nSpe] = MAX(xn[nSpe], 1.0e-10);
		xn[nSpe] = MIN(xn[nSpe], 1.0);
	}
	
	
	__syncthreads();
	for(int nSpe=0; nSpe<SPECIES; nSpe++)
	{
		x_N[index+nSpe*nelements] = xn[nSpe];
	}
	
	//float xerr[SPECIES];
	// Change the energy array
	if(E <= 0)
		EArray[index] = 0.0;
		//EArray[index] = E;
	else
		EArray[index] = E;
	
	//for(int i=0; i<SPECIES; i++)
	//{
	//	xerr[i] = xn_out[i];
	//}
	
//	if(index == 1056832)
//		*error = xeq;
//	atomicMax(error, EArray[index]);
//	atomicAdd(error, al[nBin]*(1.0-y[nBin])*ne*dt);
//	__syncthreads();
}

// This is here because derivs is inherently inline under CUDA architecture.
//#include "./inc/rkck.cu"
//#include "./inc/simpr.cu"

__global__ void ionization(	float dt, float* error, float* density, float* x_N,
							float* FluxArray, float* EArray, float* dEArray,
							float* background, int dim, float a)
{
	int nelements=dim*dim*dim;
	
	int i0 = blockIdx.x*blockDim.x+threadIdx.x;
	int j0 = blockIdx.y*blockDim.y+threadIdx.y;
	int k0 = blockIdx.z;

	int index=i0+dim*j0+dim*dim*k0;
	
	float fDtMyr = dt*3.15e13;
	
//	float t = time(1.0/a - 1.0);
	
	float xn[SPECIES];
	float xn_out[SPECIES];
	float Gamma[SPECIES];
	
	float fCumFlux = 0;
	
	for(int nSpe=0; nSpe < SPECIES; nSpe++)
	{
		xn[nSpe] = x_N[index+nSpe*nelements];
		Gamma[nSpe] = FluxArray[index+nSpe*nelements];
		Gamma[nSpe] += background[nSpe];	// Ionization rate for each species (per Myr)
		
		fCumFlux += xn[3+nSpe]*fDtMyr;
	}
	
	float dens = density[index]/(a*a*a);	// Baryonic number density
	float E = EArray[index];				// Energy (temperature) per baryon
	float dEdt = dEArray[index];
	//float dEdt = dEArray[index]+background[2]+background[3];
	
	float eps = 0.1;	// Maximum fractional change during subcycle
	float fDtRem;		// Remaining time in the subcycle loop
	
	// Subcycle loop to advance the chemistry
	// First we calculate the optically thin approximation of source terms:
	float dEdx, dGdx[SPECIES];
	for(int nSpe=0; nSpe<SPECIES; nSpe++)
	{
		dGdx[nSpe] = thin_source(Gamma[nSpe], xn[nSpe]);
	}
	dEdx = thin_source(dEdt, xn[0]);
	
	// Recombination addition
	float xe = (1.0-Y_P)*(1.0-xn[0])+0.25*Y_P*(1.0-xn[1]);
	float ne = dens*xe;
	float T  = E/((3./2.)*8.6173303e-5)/(1.0+xe);
	int CASE = 1;
	float al[2];
	al[0] = rec_HII(T, CASE);
	al[1] = rec_HeII(T, CASE);
	float fRec;
	for(int nSpe=0; nSpe<SPECIES; nSpe++)
	{
		fRec = al[nSpe]*(1.0-xn[nSpe])*ne;
		atomicAdd(error, fRec);
	}
	
	// Subcycle:
	fDtRem = fDtMyr;
	while(fDtRem > 1.0)	// One second
	{
		// Find the max time step to advance E
		float dtSubE;
		float Lam = lambda(E, xn, dens, a);
		float Heat = dEdx*xn[0];
		float dEdt = Heat - Lam;
		float rate = abs(dEdt);
		if(rate < eps*E/fDtRem)
		{
			dtSubE = fDtRem;
		}
		else
		{
			dtSubE = eps*E/rate;
		}
		
		// Find the max time step to advance hydrogen
		float dtSubH;
		rate = abs(dnHIdt(xn, dGdx, dens, E));
		if(rate < eps*0.1/fDtRem)
		{
			dtSubH = fDtRem;
		}
		else
		{
			dtSubH = eps*0.1/rate;
		}
		
		float dtSub = min(dtSubE, dtSubH);
		
		// Updating energy
		//float E1 = E + dEdt*dtSub;
		//float dEdt1 = dEdx*xn[0] - lambda(E1, xn, dens, a);
		//E = MIN(2.e4*1.29e-4, E + (dEdt+dEdt1)*dtSub/2);
		E = E + dEdt*dtSub;
		
		step_bdf(xn_out, xn, dGdx, dens, E, dtSub);		
		for(int nSpe=0; nSpe<SPECIES; nSpe++)
		{
			if (xn_out[nSpe] < 0.0)
			{
				xn[nSpe] = 0.0;
			}
			else if (xn_out[nSpe] <= 1.0)
			{
				xn[nSpe] = xn_out[nSpe];
			}
			else
			{
				xn[nSpe] = 1.0;
			}
		}
		
		fDtRem = fDtRem - dtSub;
	}
	
	__syncthreads();
	for(int nSpe=0; nSpe<SPECIES; nSpe++)
	{
		x_N[index+nSpe*nelements] = xn[nSpe];
	}
	
	//float xerr[SPECIES];
	// Change the energy array
	if(E <= 0)
		EArray[index] = 0.0;
		//EArray[index] = E;
	else
		EArray[index] = E;
	
	//for(int i=0; i<SPECIES; i++)
	//{
	//	xerr[i] = xn_out[i];
	//}
	
//	if(index == 1056832)
//		*error = xeq;
//	atomicMax(error, EArray[index]);
	//atomicAdd(error, fCumFlux);
//	__syncthreads();
}

// Signum function
__device__ int sign(float x)
{
	return (x > 0) - (x < 0);
}

// Does the HEALPix math but gives a float
__device__ void fpix2vec_nest(long n, long m, float* vec)
{
	double temp[3];
	pix2vec_nest(n, m, temp);
	
	vec[0] = (float) temp[0];
	vec[1] = (float) temp[1];
	vec[2] = (float) temp[2];
}

// Takes position (x0) and direction(u) and takes a step along integer grid to x
__device__ float raystep(float* x, int* ijk, float* x0, int* ijk0, float* u)
{
	// Minimum projection, to prevent divide by 0
	float eps = 1.e-10;
	// Length of step
	float dS;
	
	// Direction of movement along each axis
	int s[3];
	for(int i=0;i<3;i++)
		s[i] = sign(u[i]);
	
	// Distance to nearest cell face along each axis
	float r[3];
	for(int i=0;i<3;i++)
	{
		if(s[i] != 0)
			r[i] = fabsf((ijk0[i] + (s[i]+1.0)/2.0) - x0[i])/MAX(eps,fabsf(u[i]));
		else
			r[i] = 1.0/eps;
	}
	
	// Initialize next step
	for(int i=0;i<3;i++)
		ijk[i] = ijk0[i];
	
	// Take the step
	if(r[0] <= r[1] && r[0] <= r[2])
	{
		dS		= r[0];
		ijk[0]	+= s[0];
	}
	if(r[1] <= r[0] && r[1] <= r[2])
	{
		dS		= r[1];
		ijk[1]	+= s[1];
	}
	if(r[2] <= r[0] && r[2] <= r[1])
	{
		dS		= r[2];
		ijk[2]	+= s[2];
	}
	
	for(int i=0;i<3;i++)
		x[i] = x0[i] + dS*u[i];
	
	return dS;
}

__device__ int rayFinish(Ray *ray, int nDom, Domain domain)
{
	if(nDom == domain.get_id())
	{
		printf("Problem: attempting to send ray to self.");
		ray->set_dom(-1);
		return 1;
	}
	
	for(int dom=0; dom<8; dom++)
	{
		if(nDom == dom)
		{
			ray->set_dom(dom);
			return 1;
		}
	}
	
	ray->set_dom(-1);
	return 1;
}

__device__ void round_down(int * I, float * X)
{
	I[0] = __double2int_rd(X[0]);
	I[1] = __double2int_rd(X[1]);
	I[2] = __double2int_rd(X[2]);
}

// Returns 1 if the ray is outside the domain
__device__ int BoundaryCheck(float * X, int * I, int DIM)
{
	for(int i=0; i<3; i++)
	{
		if(	X[i] < 0 || I[i] < 0 ||
			X[i] >= DIM || I[i] >= DIM)
		{
			return 1;
		}
	}
	
	return 0;
}

// For tracking rays
// X and I are ray position and gridloc, vec is the direction of the ray
// mode is for adjusting rays tracked by the tracer (0) or placed on the grid (1)
__device__ void BoundaryAdjust(float * X, int * I, float* vec, int mode, int DIM)
{
	for(int i=0; i<3; i++)
	{
		if(I[i] < 0)
		{
			X[i] += DIM;
			
			if(mode == 0)
			{
				I[i] += DIM;
			}
			else
			{
				I[i] = static_cast<int>(X[i]);
			}
		}
		
		if(I[i] >= DIM)
		{
			X[i] -= DIM;
			
			if(mode == 0)
			{
				I[i] -= DIM;
			}
			else
			{
				I[i] = static_cast<int>(X[i]);
			}
		}
	}
}

/*// For new rays
__device__ void BoundaryAdjust_new(float * X, int * I, int DIM)
{
	for(int i=0; i<3; i++)
	{
		if(X[i] < 0)
		{
			X[i] += DIM;
			I[i] = static_cast<int>(X[i]);
		}
		if(X[i] >= DIM)
		{
			X[i] -= DIM;
			I[i] = static_cast<int>(X[i]);
		}
	}
}*/

// This kernel traces rays until they split or end.
// nGrid:	number density of absorbers on the physical grid
// xGrid:	the neutral fraction of absorbers on the physical grid
// Parts:	Particles under consideration
// GamGrid:	rate of photon absorption on the physical grid
// PixList:	List of N pixels (in unique nested form)
// RayDat:	List of ray data in (R, tau_0, ..., tau_n) form
// N0:		Array of number of initial rays per particle
// Nside:	HEALPix parameter
// L:		Physical length of the side of the box
// int is used because 2e9 is enough to get to HEALPix order 13
// dt:		Length of the previous time step in Myr
__global__ void rayTraceKernel(	const float *nGrid, const float *xGrid,
								const source *Parts, float *GamGrid, float* dEArray,
								Ray *RayDat,  int *N, int N0,
								float L, float a, float *nfSback, Domain domain, float dt)
{
	// Determine properties of the ray to be traced:
	
	// 2+1D grid of 2D blocks. CHECKXXX
	// z dimension of grid is for particle ID
	// Blocks are 16x16 to fill the SM's in CC 3.5
	int blockID =	blockIdx.x + blockIdx.y * gridDim.x;
	int threadID = 	blockID * blockDim.x * blockDim.y
					+ threadIdx.y * blockDim.x + threadIdx.x;
	
	// Only computing Npix rays CHECKXXX
	
	if(threadID >= N0)
		return;
	
	int dim = domain.get_dim();
	int nElements = dim*dim*dim;
	int domID = domain.get_id();
	
	int xLim0[3];//, xLim1[3];
	domain.get_x0(xLim0);
	int d_ind = dim*dim*xLim0[2] + dim*xLim0[1] + xLim0[0];
	
	//domain.get_x1(xLim1);
	
	Ray *ray = RayDat + threadID;
	int partID = ray->get_part();
	int pixID, ord;
	ray->get_pix(&pixID, &ord);
	/*if(partID <0 || partID >1)
		printf("?!? %d\t%d\n", domID, partID);*/
	
	int Nside = (1 << ord);
	int Npix = 12 * Nside * Nside;
	
	// Find direction of ray
	float vec[3];
	fpix2vec_nest(Nside, pixID, vec);
	
	// Find position of the ray
	float * X;
	X = ray->position;
	
	int * I;
	I = ray->gridloc;
	
	int nDom = domain.loc(I);
	
	// Find distance to domanin wall
	int domID3[3];
	domain.get_id3(domID3);
	
	float XR[3];
	XR[0] = Parts[partID].x/dim;
	XR[1] = Parts[partID].y/dim;
	XR[2] = Parts[partID].z/dim;
	
	float r_dom = dim*raystep(XR, domID3, XR, domID3, vec);
	
	/*if(pixID < 10 && ord == 2)
		printf("%f for (%f, %f, %f)\n", r_dom, vec[0], vec[1], vec[2]);*/
	
//	printf("%d\t%e\t%e\t%e\t%e\n", pID, ray[0], X[0], X[1], X[2]);
//	printf("%d\t%e\t%d\t%d\t%d\n", pID, ray[0], I[0], I[1], I[2]);
//	printf("%d\t%e\t%e\t%e\t%e\n", pID, ray[0], vec[0], vec[1], vec[2]);
	
	// Grab the cross sections
	float sig[SPECIES][FREQ_BIN_NUM];
	sigma(sig);
	
	// Loop variables
	float X0[3], dR;
	int I0[3], ind;
	
	// Set the max distance to trace a ray
	float Rmax = 1.7320*DIMX;
	//float Rmax = 32;
	float Rsplit = sqrt(Npix/12.56636/OMEGA_RAY);
	
	float dcross = 	fabsf(Rsplit - r_dom);
	/*if( dcross < 2.0)
		Rsplit = Rsplit - 2.0;*/
	
	while(ray->R < Rsplit)
	{
		/*if(abs(X[0]-I[0]) > 2)// This is for checking boundary conditions REMOVEXXX
		if(pixID == 89829)
		{
			printf("%d %d %d %f %f %f\n", I[0], I[1], I[2], X[0], X[1], X[2]);
		}*/
		
		ind = I[0] + dim*I[1] + dim*dim*I[2] - d_ind;
		
		memcpy(I0, I, 3*sizeof(int));
		memcpy(X0, X, 3*sizeof(float));
		
		// Take a single step
		dR = raystep(X, I, X, I, vec);
		
		// Check if the ray is just outside the domain
		if(nDom != domID)
		{
			// Check if it come from the boundary
			if(PERIODIC == 1)
			{
				if(BoundaryCheck(X, I, DIMX))
				{
					BoundaryAdjust(X, I, vec, 0, DIMX);
				}
			}
			
			// Entered
			if(domain.loc(I) == domID)
			{
				ind = I[0] + dim*I[1] + dim*dim*I[2] - d_ind;
				
				memcpy(I0, I, 3*sizeof(int));
				memcpy(X0, X, 3*sizeof(float));
		
				dR += raystep(X0, I0, X0, I0, vec);
			}
			else
			{
				rayFinish(ray, nDom, domain);
				atomicSub(N, 1);
				return;
			}
		}
		
		ray->R += dR;
		
		/*if(ind < 0 || ind >= dim*dim*dim)
			printf("??? %d %d\n", domID, ind);*/
		
		// Calculate the column densities:
		float dL = (dR/DIMX)*L*a;
		
		// Hydrogen
		float nH		= nGrid[ind]*(1.0-Y_P);
		float nHI		= nH*xGrid[ind];
		float NcolHI	= 3.086e24*dL*pow(a,-3)*nHI;
		
		// Helium
		float nHe		= nGrid[ind]*0.25*Y_P;
		float nHeI		= nHe*xGrid[ind+nElements];
		float NcolHeI	= 3.086e24*dL*pow(a,-3)*nHeI;
		
		/////////	Adjacent pixel correction //////////
		float fc = 1.0;
		float Lpix = sqrtf(12.566*ray->R*ray->R/Npix);
		float Dedge = Lpix/2.0;
		int ind_c = ind;
		int del=1;
		
		float D[3];
		
		for(int i=0; i<3; i++)
		{
			float Dci;
			Dci = X0[i] + dR*vec[i]/2.0 - (I0[i]+0.5);
			Dci = Dci;
			if(abs(Dci) > 0.5)
			{
				//Dci = 1.0 - abs(Dci);
			}
			
			D[i] = Dci;
			
			float De = 0.5 - fabs(Dci);
			if(De < Dedge)
			{
				Dedge = De;
				if(Dci > 0 && I[i] + 1 < xLim0[i] + dim)//CHECKXXX
				{
					ind_c = ind + del;
				}
				else if(Dci < 0 && I[i] > xLim0[i])
				{
					ind_c = ind - del;
				}
			}
			del *= dim;
		}
		
		if(Dedge < Lpix/2.0)
			fc = powf(0.5 + Dedge/Lpix, 1.0);
		else
			fc = 1.0;
		
		//if(pixID == 10096)
		if(Dedge < 0)
		{
			fc = 1.0;
			//printf("%d: %f (%d %d %d)(%f %f %f)(%f %f %f)(%f %f %f)\n", pixID, ray->R, I0[0], I0[1], I0[2], X0[0], X0[1], X0[2], vec[0], vec[1], vec[2], D[0], D[1], D[2]);
		}
		if(D[0] > 0.5 || D[1] > 0.5 || D[2] > 0.5)
		{
			fc = 1.0;
		}
		/////////	Adjacent pixel correction //////////
		
		float gamH = 0;
		float gamHe = 0;
		float dE = 0;
		
		/////////	Absorption limiting constants //////////
		// Total flux along the ray
		double S_0=0.0;
		for(int nBin=0; nBin<FREQ_BIN_NUM; nBin++)
		{
			S_0 += ray->flux[nBin];
		}
		// "density" of rays within a cell/cell face
		double C = OMEGA_RAY*pow(Rsplit/ray->R, 2);
		// Size of and number of atoms in the cell
		double L_cell = (1.0/DIMX)*L*a;
		double LANX = pow(L_cell*3.086e24, 3)*nH;//*xGrid[ind];
		LANX = LANX*(dL/L_cell);	// Correction for length of ray
		// Ionization time
		double T_ion = 1.0e0*LANX/C/((S_0+1.e-30)*9.5234e48);
		
		double Delta_t = dt*3.15e13;
		double R_ion = Delta_t/T_ion;
		
		for(int nBin=0;nBin<FREQ_BIN_NUM;nBin++)
		{
			float dtau, dtauH, dtauHe, dampH, dampHe, absorb, A, B;
			
			// Hydrogen
			dtauH = sig[0][nBin]*NcolHI;
			
			/*double dtauH_new;
			if(Delta_t > T_ion)
			{
				double w = ray->flux[nBin]/S_0;
				double dtauH_critical = -log(1.0 - MIN(0.999,w*T_ion/Delta_t) );
				dtauH_new = MIN( dtauH, dtauH_critical );
			}
			dtauH = dtauH_new;*/
			dampH = exp(-dtauH);
			if(T_ion < Delta_t)
			{
				dampH = MAX(1.0 - T_ion/Delta_t, dampH);
				//dampH *= (1.0 - (1.0-exp(-R_ion))/R_ion);
			}
			
			// Helium
			dtauHe = sig[1][nBin]*NcolHeI;
			dampHe = exp(-dtauHe);
			
			// Number of absorbtions per second
			absorb = ray->flux[nBin]*(1.0 - dampH*dampHe);
			
			// Keep track of total flux
			ray->flux[nBin] *= dampH*dampHe;
			//ray->tau[nBin] += dtau;
			
			// Fraction absorbed by H, He
			dtau = dtauH + dtauHe;
			if(dtau < 1.e-10)
			{
				// simplify for dtau~0
				float temp_H = sig[0][nBin]*nH;
				float temp_He = sig[1][nBin]*nHe;
				A = temp_H/(temp_H+temp_He);
				B = temp_He/(temp_H+temp_He);
			}
			else
			{
				A = dtauH/dtau;
				B = dtauHe/dtau;
			}
			
			// Add total photon counts
			absorb	/= powf(L/DIMX,3)*3.086e24;	// Unit correction
			gamH	+= fc*A*absorb/nH;
			gamHe	+= fc*B*absorb/nHe;
			
			// AYAYAYAYYAYAYAYYA
			bool check = isnan(xGrid[ind]);
			if(check)
			{
				//printf("%e %e %e\n", I[0], I[1], I[2]);
				//printf("%d %d %d\n", I[0], I[1], I[2]);
			}
			//assert(!check);
			
			// Add the energy up CHECKXXX
			dE		+= fc*A*(energy[nBin]-13.6)*absorb/nH;
			dE		+= fc*B*(energy[nBin]-24.6)*absorb/nHe;
//			dE		+= fc*(energy[nBin]-13.6)*absorb/nGrid[ind];
		}
		
		// Update flux array
		atomicAdd(GamGrid + ind, gamH);
		atomicAdd(GamGrid + ind + nElements, gamHe);
		
		// Update Energy array
		atomicAdd(dEArray + ind, dE);
		
		/////////	Adjacent pixel correction //////////
		float ratio = xGrid[ind_c]/xGrid[ind];
		//float ratio = 1.0;
		float gamH_c = ratio*gamH*(1.0-fc)/fc;
		float gamHe_c = ratio*gamHe*(1.0-fc)/fc;
		float dE_c = ratio*dE*(1.0-fc)/fc;
	
		atomicAdd(GamGrid + ind_c, gamH_c);
		atomicAdd(GamGrid + ind_c + nElements, gamHe_c);
		atomicAdd(dEArray + ind_c, dE_c);
		/////////	Adjacent pixel correction //////////
		
		// Apply boundary conditions, if required
		float checkX[3];
		int checkI[3];
		memcpy(checkX, ray->position, 3*sizeof(float));
		memcpy(checkI, ray->gridloc, 3*sizeof(int));
		
		float X2[3];
		int I2[3];
		memcpy(X2, ray->position, 3*sizeof(float));
		memcpy(I2, ray->gridloc, 3*sizeof(int));
		
		if(PERIODIC == 1)
		{
			
			if(BoundaryCheck(X, I, DIMX))
			{
				BoundaryAdjust(X, I, vec, 0, DIMX);
			}
			
			if(0 && pixID == 10096)
			{
				printf("A %d %d %d %f %f %f \n", I2[0], I2[1], I2[2], X2[0], X2[1], X2[2]);
				printf("B %d %d %d %f %f %f \n", I[0], I[1], I[2], X[0], X[1], X[2]);
			}
		}
		
		nDom = domain.loc(I);
		
		// Terminate the ray
		if(	ray->R > Rmax || BoundaryCheck(X, I, DIMX))
		{
			ray->set_dom(-1);
			atomicSub(N, 1);
			
			for(int nBin=0; nBin<FREQ_BIN_NUM; nBin++)
			{
				atomicAdd(nfSback + nBin, ray->flux[nBin]); // CHECKXXX
			}
			
			return;
		}
		
		if(	nDom != domID )
		{
			rayFinish(ray, nDom, domain);
			atomicSub(N, 1);
			
			return;
		}
	}
	// Add up all the rays that don't terminate
	__syncthreads();
}

// This kernel splits the rays into the next HEALPix level until they split or end.
// PixList:	List of N pixels (in unique nested form)
// RayDat:	List of ray data in (R, tau_0, ..., tau_n) form
// N0:		Number of rays
// int is used because 2e9 is enough to get to HEALPix order 13
__global__ void raySplitKernel(	Ray *RayDat_init, Ray *RayDat, int *nRays, int N0,
								Ray *RayBuf, int* nBufLoc,
								const source * source_dev, Domain domain)
{
	// 2+1D grid of 2D blocks.
	// z dimension of grid is for particle ID
	// Blocks are 16x16 to fill the SM's in CC 3.5
	int blockID =	blockIdx.x + blockIdx.y * gridDim.x;
	int threadID = 	blockID * blockDim.x * blockDim.y
					+ threadIdx.y * blockDim.x + threadIdx.x;
	
	// Only computing Npix rays CHECKXXX
	
	if(threadID >= N0)
		return;
	
	Ray *ray = RayDat_init + threadID;
	
	// Terminated rays
	if(ray->get_dom() == -1)
	{
		return;
	}
	
	// Split rays
	int rayDom = ray->get_dom();
	if(rayDom == domain.get_id())
	{
		// Get a unique ID for the (first) ray
		int rayID = atomicAdd(nRays, 4);
		int partID = ray->get_part();
		int pixID, ord;
		ray->get_pix(&pixID, &ord);
		
		float origin[3];
		origin[0] = source_dev[partID].x;
		origin[1] = source_dev[partID].y;
		origin[2] = source_dev[partID].z;
	
		// Splitting into 4 rays
		for(int nSplit=0; nSplit<4; nSplit++)
		{
			Ray *ray_split = RayDat + (rayID+nSplit);
			
			int new_ID = 4*pixID + nSplit;
			int new_ord = ord + 1;
			int Nside = (1 << new_ord);
			float direction[3];
			fpix2vec_nest(Nside, new_ID, direction);
			
			ray_split->R = ray->R;
			ray_split->set_part(partID);
			ray_split->set_pix(new_ID, new_ord);
			
			float flux_init[FREQ_BIN_NUM];
			for(int nBin=0; nBin<FREQ_BIN_NUM; nBin++)
			{
				flux_init[nBin] = ray->flux[nBin]/4;
			}
			ray_split->set_flux(flux_init);
			
			ray_split->set_position(origin, ray->R, direction);
			
			// Apply boundary
			float * rayX = ray_split->position;
			int * rayI = ray_split->gridloc;
			
			float * checkX = ray->position;
			int * checkI = ray->gridloc;
			int check = -1;
			if(new_ID == check)
			{
				printf("Placing at %d %d %d %f %f %f\n", rayI[0], rayI[1], rayI[2], rayX[0], rayX[1], rayX[2]);
				printf("From %d %d %d %f %f %f\n", checkI[0], checkI[1], checkI[2], checkX[0], checkX[1], checkX[2]);
			}
			if(PERIODIC == 1)
			{
				if(BoundaryCheck(rayX, rayI, DIMX))
				{
					BoundaryAdjust(rayX, rayI, direction, 1, DIMX);
				}
			}
			if(new_ID == check)
			{
				printf("Now at %d %d %d %f %f %f\n", rayI[0], rayI[1], rayI[2], rayX[0], rayX[1], rayX[2]);
			}
			
			int splitDom = domain.loc(ray_split->gridloc);
			ray_split->set_dom(splitDom);
			
			// Move rays for different domains to the buffer
			if(splitDom != rayDom)
			{
				int nBufID = atomicAdd((nBufLoc + splitDom), 1);
				int pix, ord;
				ray_split->get_pix(&pix,&ord);
				RayBuf[splitDom*NUM_BUF + nBufID].copy_ray(*ray_split);
				//printf("Split ray in wrong domain! %d: %d -> %d, (%d, %d, %d) %f\n", new_ID, rayDom, splitDom, rayI[0], rayI[1], rayI[2], ray->R);
				ray_split->set_dom(-1);
			}
		}
		
		// Terminate old ray
		ray->set_dom(-1);
		return;
	}
	
	// Buffer rays
	for(int dom=0; dom<8; dom++)
	{
		if(ray->get_dom() == domain.get_id())
			continue;
		if(ray->get_dom() == dom)
		{
			// Conditional for testing REMOVEXXX
			int pixID, ord;
			ray->get_pix(&pixID, &ord);
			int Nside = (1 << ord);
			float direction[3];
			fpix2vec_nest(Nside, pixID, direction);
			//if(direction[0]<-0.75 || direction[1]<-0.75 || direction[2]<-0.75)
			if(1)
			{
				// Copy ray into buffer
				int nBufID = atomicAdd((nBufLoc + dom), 1);
				//printf("A %d\t%d\t%d\t%d\n", domain.get_id(), dom, nBufLoc[dom], dom*NUM_BUF+nBufID);
				int pix, ord;
				ray->get_pix(&pix,&ord);
			
				RayBuf[dom*NUM_BUF + nBufID].copy_ray(*ray);
				//ray->R = temp;
			}
			
			// Terminate old ray
			ray->set_dom(-1);
			return;
		}
	}
	printf("Couldn't find home for our ray!\n");
	ray->set_dom(-1);
}
