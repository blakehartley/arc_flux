double time(double);
double redshift(double);

// xarr must be monatomic increasing
/*float interp_f(float* yarr, float* xarr, float x, const int n)
{
	int i=n-2;
	for(; i>=0; i--)
		if(x > xarr[i])
			break;
	
	float yout = yarr[i];
	yout += (x - xarr[i])*(yarr[i+1] - yarr[i])/(xarr[i+1] - xarr[i]);
	
	printf("%e < %e < %e ==> %e < %e < %e\n", xarr[i], x, xarr[i+1], yarr[i], yout, yarr[i+1]);
	return yout;
}*/

void interpolate(float* f, float* x0, float* x, const int n)
{
	float fnew[n];
	int i=0, j=0;
	for(;i < n;)
	{
		if(x[i] <= x0[j])	// current i before current j; increase i
		{
			fnew[i] = 0;
			i++;
		}
		else if(x[i] > x0[j+1] && x[i] <= x0[n-1]) // current j range before current i; increase j
		{
			j++;
		}
		else if(x0[j] <= x[i] && x[i] <= x0[j+1])	// i in correct j range; move to next i
		{
			fnew[i] = (x[i]-x0[j])*(f[j+1]-f[j])/(x0[j+1]-x0[j])+f[j];
			i++;
			continue;
		}
		else if(x[i] >= x0[n-1])	// i is beyond j range
		{
			fnew[i] = 0;
			i++;
			continue;
		}
		else
		{
			cout << "Something broke!" << endl;
			cout << x0[j] << " - " << x[i] << " - " << x0[j+1] << endl;
		}
	}
	
	for(i=0; i<n; i++)
	{
		f[i] = fnew[i];
	}
}

float sig_H(float nu)
{
	float a0=6.3e-18;
	float nuih=13.6;
	float par2=(nu/nuih-1.0);
	float par;
	
	if(par2>=0)
		par = sqrt(par2);
	else
		par = 1.e-6;
	
	float sig;
	
	if(nu>nuih)
		sig = a0*pow(nuih/nu,4)*exp(4.0-(4.0*atan(par)/par))/(1.0-exp(-2.0*3.14159/par));
	else
		sig = 0;
	return sig;
}

float sig_He(float nu)
{
	float sig;
	
	if(nu>24.6)
		sig = sig_H(nu)*(6.53*(nu/24.6)-0.22);
	else if(nu>65.4)
		sig = sig_H(nu)*(37.0-19.1*pow(nu/65.4e0,-.76));
	else
		sig = 0;
	return sig;
}

void g_nu(float* g, float* nu, float nn, const int n)
{
	float nu_max = 60.0;
	float A = (pow(13.6, -1.0) - pow(nu_max, -1.0));
	//float Yp = 0.0;
	
	for(int i=0; i<n; i++)
	{
		if(nu[i]>nu_max || nu[i]<13.6)
			g[i] = 0;
		else
			g[i] = nu[i]*(1.0/A)*pow(nu[i],-2)*exp(-nn*(sig_H(nu[i])));
	}
	/*float nu_max=100.0;
	float A = 13.6*log(nu_max/13.6);
	for(int i=0; i<n; i++)
	{
		g[i] = nu[i]*(1.0/A)*(13.6/nu[i])*exp(-nn*sig_H(nu[i]));
		
		if(nu[i]>nu_max || nu[i]<13.6)
		{
			g[i] = 0.0;
		}
	}*/
}

float dtau(float dz, float z, float nu, float nb, float x_hi, float x_hei)
{
	float c=3.0e10;
	float Yp = 0.24;
	float xsig = x_hi*(1.-Yp)*sig_H(nu)+x_hei*Yp*sig_He(nu)/4.0;
	float dt = (3.15e13*(time(z-dz)-time(z)));
	return nb*xsig*c*dt;
}

void Source(	float* s, float ndot, float dz, float z0, float* nu,
				float nb, float x_hi, float x_hei, float R0, float nn, const int n)
{
	// Swapped variables to "double" to stop a precision error that was seizing the code
	double c		= 3.0e10;
	double ddz	= dz/64.0;
	double z1	= z0-dz;
	double dz1	= 0.0;
	double t0	= 0.0;
	
	//double dt0	= (3.15e13*(time(z0-dz)-time(z0)));
	
	for(int i=0; i<n; i++)
		s[i] = 0;
	
	while(z1<z0+ddz)
	{
		dz1 = dz1 + ddz;
		z1 = z1 + ddz;
		double dt = (3.15e13*(time(z1-ddz)-time(z1)));
		t0 = t0 + dt;
		double fac = (z1+1.0-dz1)/(z1+1.0);
		
		if(c*t0 > 3.086e24*R0/(z1+1.0))
		{
			float g[n], nu_p[n];
			g_nu(g, nu, nn, n);
			for(int i=0; i<n; i++)
				nu_p[i] = nu[i]/fac;
	
			interpolate(g, nu, nu_p, n);
	
			for(int i=0; i<n; i++)
			{
				double temp;
				temp = g[i]*exp(-dtau(dz1, z1, nu_p[i], nb, x_hi, x_hei));
				//temp = ndot*temp*dt*pow(fac,-3);
				temp = ndot*temp*c*dt*pow(fac,-3);
				s[i] += temp;
			}
		}
		
		// For outputting things
		for(int i=0; i<n; i++)
		{
			/*float g[1000];
			g_nu(g, nu, 0.0, 1000);
			printf("%e\t", g[i]*exp(-dtau(dz1, z1, nu[i], nb, x_hi, x_hei)));*/
			/*float g[1000];
			g_nu(g, nu, 0.0, 1000);
			printf("%e\t", g[i]);*/
			//printf("%e\t", s[i]);
			//printf("%e\t", dtau(dz1, z1, nu[i], nb, x_hi, x_hei));
			//printf("%e\t", sig_He(nu[i]));
			/*for(int i=0; i<n; i++)
				temp += s[i]*nu[i]*log(nu[1]/nu[0]);
			printf("Source = %e\n", temp);*/
		}
		//exit(0);
		// For outputting things
	}
}

// Absorbs for R0 worth of light travel/universal expansion
// Assumes the variables are constant
void Absorb(	float* J, float R0, float z0, float* nu, float nb,
				float x_hi, float x_hei, const int n)
{
	float c		= 3.0e10;
	
	float dt	= R0*3.086e24/c/3.15e13/(1.0+z0);
	float dz	= 0.6667*(1.+z0)*dt/time(z0);
	float fac	= (z0+1.0-dz)/(z0+1.0);
	//printf("%e\n", fac);
	
	float nu_p[n];
	for(int i=0; i<n; i++)
		nu_p[i] = nu[i]/fac;
	
	interpolate(J, nu, nu_p, n);
	for(int i=0; i<n; i++)
	{
		J[i] = J[i]*exp(-dtau(dz, z0, nu[i], nb, x_hi, x_hei))*pow(fac, 3);
	}
}

void Jnew(	float* J, float ndot, float dz, float z0, float* nu,
			float nb, float x_hi, float x_hei, float R0, float nn, const int n)
{
	float fac = (z0+1.0-dz)/(z0+1.0);
	float s[n];
	Source(s, ndot, dz, z0, nu, nb, x_hi, x_hei, R0, nn, n);
	
	// Shift the background back to remove radiation within R0
	//Absorb(s, 5.0, z0, nu, nb, x_hi, x_hei, n);
	
	float nu_p[n];
	for(int i=0; i<n; i++)
		nu_p[i] = nu[i]/fac;
		
	interpolate(J, nu, nu_p, n);
	
	for(int i=0; i<n; i++)
	{
		J[i] = J[i]*exp(-dtau(dz, z0, nu[i], nb, x_hi, x_hei));
		J[i] = J[i]*pow(fac, 3) + s[i];
	}
}

float lnint(float* J, float* nu, const int n)
{
	float sum = 0;
	for(int i=0; i<n; i++)
	{
		sum += J[i]*log(nu[1]/nu[0]);
	}
	return sum;
}
