// ][ -> *n+
__device__ void lubksb(float* a, int* indx, float* b)
{
	int i,ii=0,ip,j;
	float sum;
	
	int n = 5;
	for (i=0;i<n;i++) {
		ip=indx[i];
		sum=b[ip];
		b[ip]=b[i];
		if (ii != 0)
			for (j=ii-1;j<i;j++) sum -= a[i*n+j]*b[j];
		else if (sum != 0.0)
			ii=i+1;
		b[i]=sum;
	}
	for (i=n-1;i>=0;i--) {
		sum=b[i];
		for (j=i+1;j<n;j++) sum -= a[i*n+j]*b[j];
		b[i]=sum/a[i*n+i];
	}
}
