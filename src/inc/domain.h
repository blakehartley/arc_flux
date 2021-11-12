#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__ inline
#else
#define CUDA_CALLABLE_MEMBER
#endif

// Class for cubic domains
class Domain
{
		// Length of edge in grid units
		int dim;
		int DIM;
		
		// ID of the domain
		int id;
		int id3[3];
		
		// Lower limits
		int x0[3];
		
		// Upper limits
		int x1[3];
		
	public:
		CUDA_CALLABLE_MEMBER Domain(int ident, int dimension)
		{
			dim = dimension;
			DIM = dim*2;
			id = ident;
			
			id3[0] = id % 2;
			id3[1] = (id / 2) % 2;
			id3[2] = id / 4;
			
			for(int i=0; i<3; i++)
			{
				x0[i] = id3[i]*dim;
				x1[i] = x0[i] + dim;
			}
		}
		
		CUDA_CALLABLE_MEMBER int get_dim()
		{
			return dim;
		}
		
		// Return id of the domain
		CUDA_CALLABLE_MEMBER int get_id(void)
		{
			return id;
		}
		
		// Give the 3d id of the domain
		CUDA_CALLABLE_MEMBER void get_id3(int *out)
		{
			for(int i=0; i<3; i++)
			{
				out[i] = id3[i];
			}
		}
		
		CUDA_CALLABLE_MEMBER void get_x0(int* x)
		{
			for(int i=0; i<3; i++)
			{
				x[i] = x0[i];
			}
		}
		
		CUDA_CALLABLE_MEMBER void get_x1(int* x)
		{
			for(int i=0; i<3; i++)
			{
				x[i] = x1[i];
			}
		}
		
		CUDA_CALLABLE_MEMBER int loc(int* J)
		{
			/*int J[3];
			memcpy(J, I, 3*sizeof(int));*/
			
			// Boundary conditions
			// Terminate and add to the background if the ray reaches the boundary
			if(	J[0] < 0 || J[0] >= DIM ||
				J[1] < 0 || J[1] >= DIM ||
				J[2] < 0 || J[2] >= DIM )
			{
				return -1;
			}
			
			int i0 = J[0] / dim;
			int i1 = J[1] / dim;
			int i2 = J[2] / dim;
			
			return (4*i2 + 2*i1 + i0);
		}
};
