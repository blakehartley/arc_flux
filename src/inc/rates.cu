#define TTR_HI 157807.0
#define TTR_HeI 285335.0
#define TTR_HeII 631515.0
#define ERG_TO_EV 6.242e11

__device__ float rec_HII(float _t, int _case)
{
	float _a;
	float lam = 2.0*TTR_HI/_t;
	
	// Case A
	if(_case == 0)
	{
		_a = 1.269e-13*powf(lam, 1.503)/powf(1.0 + powf(lam/0.522, 0.470), 1.923);
	}
	else
	{
		_a = 2.753e-14*powf(lam, 1.500)/powf(1.0 + powf(lam/2.740, 0.407), 2.242);
	}
	
	return _a;
}

__device__ float rec_cool_HII(float _t, int _case)
{
	float _a;
	float lam = 2.0*TTR_HI/_t;
	
	// Case A
	if(_case == 0)
	{
		_a = _t*1.778e-29*powf(lam, 1.965)/powf(1.0 + powf(lam/0.541, 0.502), 2.697);
	}
	else
	{
		_a = _t*3.435e-30*powf(lam, 1.970)/powf(1.0 + powf(lam/2.250, 0.376), 3.720);
	}
	
	return _a*ERG_TO_EV;
}

__device__ float rec_HeII(float _t, int _case)
{
	float _a;
	float lam = 2.0*TTR_HeI/_t;
	
	if(_t <= 5.e3 || _t >= 5.e5)
		return 0.0;
	
	// Case A
	if(_case == 0)
	{
		_a = 3.0e-14*powf(lam, 0.654);
	}
	else
	{
		_a = 1.26e-14*powf(lam, 0.750);
	}
	
	return _a;
}

__device__ float rec_cool_HeII(float _t, int _case)
{
	float kb = 8.617e-5;
	
	float _a = (kb*_t)*rec_HeII(_t, _case);
	
	return _a;
}

__device__ float rec_HeIII(float _t, int _case)
{
	float _a;
	float lam = 2.0*TTR_HeII/_t;
	
	// Case A
	if(_case == 0)
	{
		_a = 2.0*1.269e-13*powf(lam, 1.503)/powf(1.0 + powf(lam/0.522, 0.470), 1.923);
	}
	else
	{
		_a = 2.0*2.753e-14*powf(lam, 1.500)/powf(1.0 + powf(lam/2.740, 0.407), 2.242);
	}
	
	return _a;
}

__device__ float rec_cool_HeIII(float _t, int _case)
{
	float _a;
	float lam = 2.0*TTR_HeII/_t;
	
	// Case A
	if(_case == 0)
	{
		_a = _t*8.0*1.778e-29*powf(lam, 1.965)/powf(1.0 + powf(lam/0.541, 0.502), 2.697);
	}
	else
	{
		_a = _t*8.0*3.435e-30*powf(lam, 1.970)/powf(1.0 + powf(lam/2.250, 0.376), 3.720);
	}
	
	return _a*ERG_TO_EV;
}

__device__ float col_HI(float _t)
{
	float lam = 2.0*TTR_HI/_t;
	
	if(_t < 1.e4 || _t > 1.e9)
		return 0.0;
	
	float _a = 21.11*powf(_t, -1.5)*expf(-lam/2.0);
	_a *= powf(lam, -1.089)/powf(1.0 + powf(lam/0.354, 0.874), 1.101);
	return _a;
}

__device__ float col_cool_HI(float _t)
{
	float kb = 8.617e-5;
	
	float _a = (kb*TTR_HI)*col_HI(_t);
	
	return _a;
}

__device__ float col_HeI(float _t)
{
	float lam = 2.0*TTR_HeI/_t;
	
	if(_t < 1.e4 || _t > 1.e9)
		return 0.0;
	
	float _a = 32.38*powf(_t, -1.5)*expf(-lam/2.0);
	_a *= powf(lam, -1.146)/powf(1.0 + powf(lam/0.416, 0.987), 1.056);
	return _a;
}

__device__ float col_cool_HeI(float _t)
{
	float kb = 8.617e-5;
	
	float _a = (kb*TTR_HeI)*col_HeI(_t);
	
	return _a;
}

__device__ float col_HeII(float _t)
{
	float lam = 2.0*TTR_HeII/_t;
	
	if(_t < 1.e4 || _t > 1.e9)
		return 0.0;
	
	float _a = 19.95*powf(_t, -1.5)*expf(-lam/2.0);
	_a *= powf(lam, -1.089)/powf(1.0 + powf(lam/0.553, 0.735), 1.275);
	return _a;
}

__device__ float col_cool_HeII(float _t)
{
	float kb = 8.617e-5;
	//float lam = 2.0*TTR_HeII/_t;
	
	float _a = (kb*TTR_HeII)*col_HeII(_t);
	
	return _a;
}

__device__ float colex_HI(float _t)
{
	float lam = 2.0*TTR_HI/_t;
	
	if(_t < 5.e3 || _t > 5.e5)
		return 0.0;
	
	float _a = 7.5e-19*expf(-0.375*lam)/(1.0+powf(_t/1.e5, 0.5));
	
	return _a*ERG_TO_EV;
}
