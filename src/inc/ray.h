#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__ inline
#else
#define CUDA_CALLABLE_MEMBER
#endif

#define FREQ_NUM 4

typedef struct Ray
{
	// Particle Identifier
	int PID;
	
	// Unique Identifier scheme (to save memory)
	int UID;
	
	// Domain in which the ray is located
	int DOM;
	
	float R;
	float position[3];
	int gridloc[3];
	float flux[FREQ_NUM];
	
	// Constructor.
	CUDA_CALLABLE_MEMBER Ray();

	// Set Pixel/HEALPix level.
	CUDA_CALLABLE_MEMBER void set_pix(int p, int ord)
	{
		//int Nside2 = pow(4, ord);
		int Nside2 = 1 << (2*ord);
		if(p<0 || p>=12*Nside2)
		{
			printf("Invalid ray pixel: p=%d, ord=%d, DOM=%d, R=%f\n", p, ord, DOM, R);
			//printf("Invalid ray pixel: p=%d, ord=%d\n", p, ord);
		}
		UID = p + 4*Nside2;
	}
	
	// Get Pixel/HEALPix level.
	CUDA_CALLABLE_MEMBER void get_pix(int* pout, int* ordout)
	{
		int ord = (int) log2((float) (UID/4))/2;
		//int p = UID - 4*pow(4, ord);
		int p = UID - 4*(1 << 2*ord);
		*pout = p;
		*ordout = ord;
	}
	
	// Get Nside (hard to do 2^x in CUDA).
	CUDA_CALLABLE_MEMBER int get_Nside()
	{
		int ord = (int) log2((float) (UID/4))/2;
		return (1 << ord);
	}
	
	// Set particle ID.
	CUDA_CALLABLE_MEMBER void set_part(int particle)
	{
		PID = particle;
	}
	
	// Get particle ID.
	CUDA_CALLABLE_MEMBER int get_part()
	{
		return PID;
	}
	
	// Set ray position.
	CUDA_CALLABLE_MEMBER void set_position(float * x, float R0, float * vec)
	{
		position[0] = x[0] + R0 * vec[0];
		position[1] = x[1] + R0 * vec[1];
		position[2] = x[2] + R0 * vec[2];
		
		for(int i=0; i<3; i++)
		{
			gridloc[i] = static_cast<int>(position[i]);
		}
		/*gridloc[0] = int(position[0]);
		gridloc[1] = int(position[1]);
		gridloc[2] = int(position[2]);*/
	}
	
	// Set ray domain ID.
	CUDA_CALLABLE_MEMBER void set_dom(int domain)
	{
		DOM = domain;
	}
	
	// Get particle ID.
	CUDA_CALLABLE_MEMBER int get_dom()
	{
		return DOM;
	}
	
	// Set ray flux
	CUDA_CALLABLE_MEMBER void set_flux(float * flux_init)
	{
		for(int nBin=0; nBin<FREQ_NUM; nBin++)
		{
			flux[nBin] = flux_init[nBin];
		}
	}
	
	CUDA_CALLABLE_MEMBER void copy_ray(Ray ray0)
	{
		PID = ray0.get_part();
		
		int pix, ord;
		ray0.get_pix(&pix, &ord);
		this->set_pix(pix, ord);
		
		DOM = ray0.get_dom();
	
		R = ray0.R;
		
		position[0] = ray0.position[0];
		position[1] = ray0.position[1];
		position[2] = ray0.position[2];
		
		gridloc[0] = ray0.gridloc[0];
		gridloc[1] = ray0.gridloc[1];
		gridloc[2] = ray0.gridloc[2];
		
		for(int nBin=0; nBin<FREQ_NUM; nBin++)
		{
			flux[nBin] = ray0.flux[nBin];
		}
	}
} Ray;

CUDA_CALLABLE_MEMBER Ray :: Ray()
{
	PID = 0;
	UID = 4;
	DOM = -1;
	R = 0.0;
	
	position[0] = -1.0;
	position[1] = -1.0;
	position[2] = -1.0;
	gridloc[0] = static_cast<int>(position[0]);
	gridloc[1] = static_cast<int>(position[1]);
	gridloc[2] = static_cast<int>(position[2]);
	
	for(int bin=0; bin<FREQ_NUM; bin++)
	{
		flux[bin] = 0.0;
	}
}
