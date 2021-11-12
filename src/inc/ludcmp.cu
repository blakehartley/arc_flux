#include <cmath>
using namespace std;
// ][ -> *n+

__device__ void ludcmp(float* a, int* indx, float &d)
{
	const float TINY=1.0e-20;
	int i,imax,j,k;
	float big,dum,sum,temp;
	
	const int n = 5;
	float vv[n];
	d=1.0;
	for (i=0;i<n;i++) {
		big=0.0;
		for (j=0;j<n;j++)
			if ((temp=fabs(a[i*n+j])) > big) big=temp;
//		if (big == 0.0) nrerror("Singular matrix in routine ludcmp");
		vv[i]=1.0/big;
	}
	for (j=0;j<n;j++) {
		for (i=0;i<j;i++) {
			sum=a[i*n+j];
			for (k=0;k<i;k++) sum -= a[i*n+k]*a[k*n+j];
			a[i*n+j]=sum;
		}
		big=0.0;
		for (i=j;i<n;i++) {
			sum=a[i*n+j];
			for (k=0;k<j;k++) sum -= a[i*n+k]*a[k*n+j];
			a[i*n+j]=sum;
			if ((dum=vv[i]*fabs(sum)) >= big) {
				big=dum;
				imax=i;
			}
		}
		if (j != imax) {
			for (k=0;k<n;k++) {
				dum=a[imax*n+k];
				a[imax*n+k]=a[j*n+k];
				a[j*n+k]=dum;
			}
			d = -d;
			vv[imax]=vv[j];
		}
		indx[j]=imax;
		if (a[j*n+j] == 0.0) a[j*n+j]=TINY;
		if (j != n-1) {
			dum=1.0/(a[j*n+j]);
			for (i=j+1;i<n;i++) a[i*n+j] *= dum;
		}
	}
}
