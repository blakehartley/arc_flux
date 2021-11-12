/* -----------------------------------------------------------------------------
*
*  Copyright (C) 1997-2016 Krzysztof M. Gorski, Eric Hivon, Martin Reinecke,
*                          Benjamin D. Wandelt, Anthony J. Banday,
*                          Matthias Bartelmann,
*                          Reza Ansari & Kenneth M. Ganga
*
*
*  This file is part of HEALPix.
*
*  HEALPix is free software; you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation; either version 2 of the License, or
*  (at your option) any later version.
*
*  HEALPix is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with HEALPix; if not, write to the Free Software
*  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
*
*  For more information about HEALPix see http://healpix.sourceforge.net
*
*---------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "chealpix.h"

__device__ static const int jrll[] = { 2,2,2,2,3,3,3,3,4,4,4,4 };
__device__ static const int jpll[] = { 1,3,5,7,0,2,4,6,1,3,5,7 };
__device__ static const double halfpi = 1.570796326794896619231321691639751442099;

__device__ static const short ctab[] = {
#define Z(a) a,a+1,a+256,a+257
#define Y(a) Z(a),Z(a+2),Z(a+512),Z(a+514)
#define X(a) Y(a),Y(a+4),Y(a+1024),Y(a+1028)
	X(0),X(8),X(2048),X(2056)
#undef X
#undef Y
#undef Z
};

__host__ __device__ long nside2npix(const long nside)
{
	return 12 * nside*nside;
}

__device__ __host__ static void nest2xyf(int nside, int pix, int *ix, int *iy, int *face_num)
{
	int npface_ = nside*nside, raw;
	*face_num = pix / npface_;
	pix &= (npface_ - 1);
	raw = (pix & 0x5555) | ((pix & 0x55550000) >> 15);
	*ix = ctab[raw & 0xff] | (ctab[raw >> 8] << 4);
	pix >>= 1;
	raw = (pix & 0x5555) | ((pix & 0x55550000) >> 15);
	*iy = ctab[raw & 0xff] | (ctab[raw >> 8] << 4);
}

__device__ __host__ static void pix2ang_nest_z_phi(int nside_, int pix, double *z, double *phi)
{
	int nl4 = nside_ * 4;
	int npix_ = 12 * nside_*nside_;
	double fact2_ = 4. / npix_;
	int face_num, ix, iy, nr, kshift;

	nest2xyf(nside_, pix, &ix, &iy, &face_num);
	int jr = (jrll[face_num] * nside_) - ix - iy - 1;

	if (jr<nside_)
	{
		nr = jr;
		*z = 1 - nr*nr*fact2_;
		kshift = 0;
	}
	else if (jr > 3 * nside_)
	{
		nr = nl4 - jr;
		*z = nr*nr*fact2_ - 1;
		kshift = 0;
	}
	else
	{
		double fact1_ = (nside_ << 1)*fact2_;
		nr = nside_;
		*z = (2 * nside_ - jr)*fact1_;
		kshift = (jr - nside_) & 1;
	}

	int jp = (jpll[face_num] * nr + ix - iy + 1 + kshift) / 2;
	if (jp>nl4) jp -= nl4;
	if (jp<1) jp += nl4;

	*phi = (jp - (kshift + 1)*0.5)*(halfpi / nr);
}

__device__ __host__ void pix2vec_nest(long nside, long ipix, double *vec)
{
	double z, phi;
	pix2ang_nest_z_phi(nside, ipix, &z, &phi);
	double stheta = sqrt((1. - z)*(1. + z));
	vec[0] = stheta*cos(phi);
	vec[1] = stheta*sin(phi);
	vec[2] = z;
}
