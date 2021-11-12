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
*----------------------------------------------------------------------------*/
/*
* chealpix.h
*/

#ifndef CHEALPIX_H
#define CHEALPIX_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

	/*! \defgroup chealpix HEALPix C interface
	All angles are in radian, all \a theta values are colatitudes, i.e. counted
	downwards from the North Pole. \a Nside can be any positive number for
	pixelisations in RING scheme; in NEST scheme, they must be powers of 2.
	The maximum \a Nside for the traditional interface is 8192; for the
	64bit interface it is 2^29.
	*/
	/*! \{ */

	/* -------------------- */
	/* Constant Definitions */
	/* -------------------- */

#ifndef HEALPIX_NULLVAL
#define HEALPIX_NULLVAL (-1.6375e30)
#endif /* HEALPIX_NULLVAL */

	/*! Returns \a 12*nside*nside. */
	__host__ __device__ long nside2npix(long nside);
	/*! Sets \a vec to the Cartesian vector pointing in the direction of the center
	of pixel \a ipix in NEST scheme at resolution \a nside. */
	__host__ __device__ void pix2vec_nest(long nside, long ipix, double *vec);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CHEALPIX_H */
