#include "ludcmp.cu"
#include "lubksb.cu"

__device__ void simpr(float* y, float* dydx, float* dfdx, float* dfdy,
	const float xs, const float htot, const int nstep, float* yout,
	void derivs(const float, float* , float*))
{
	int i,j,nn;
	float d,h,x;
	
	const int n = 5;

	float a[n*n];
	
	int indx[n];
	float del[n],ytemp[n];
	h=htot/nstep;
	for (i=0;i<n;i++) {
		for (j=0;j<n;j++) a[i*n+j] = -h*dfdy[i*n+j];
		++a[i*n+i];
	}
	ludcmp(a,indx,d);
	for (i=0;i<n;i++)
		yout[i]=h*(dydx[i]+h*dfdx[i]);
	lubksb(a,indx,yout);
	for (i=0;i<n;i++)
		ytemp[i]=y[i]+(del[i]=yout[i]);
	x=xs+h;
	derivs(x,ytemp,yout);
	for (nn=2;nn<=nstep;nn++) {
		for (i=0;i<n;i++)
			yout[i]=h*yout[i]-del[i];
		lubksb(a,indx,yout);
		for (i=0;i<n;i++) ytemp[i] += (del[i] += 2.0*yout[i]);
		x += h;
		derivs(x,ytemp,yout);
	}
	for (i=0;i<n;i++)
		yout[i]=h*yout[i]-del[i];
	lubksb(a,indx,yout);
	for (i=0;i<n;i++)
		yout[i] += ytemp[i];
}
