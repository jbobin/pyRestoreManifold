/*
 * cln.h - This file is part of MRS3D
 * Created on 16/05/11
 * Contributor : François Lanusse (francois.lanusse@gmail.com)
 *
 * Copyright 2012 CEA
 *
 * This software is a computer program whose purpose is to apply mutli-
 * resolution signal processing algorithms on spherical 3D data.
 *
 * This software is governed by the CeCILL  license under French law and
 * abiding by the rules of distribution of free software.  You can  use,
 * modify and/ or redistribute the software under the terms of the CeCILL
 * license as circulated by CEA, CNRS and INRIA at the following URL
 * "http://www.cecill.info".
 *
 * As a counterpart to the access to the source code and  rights to copy,
 * modify and redistribute granted by the license, users are provided only
 * with a limited warranty  and the software's author,  the holder of the
 * economic rights,  and the successive licensors  have only  limited
 * liability.
 *
 * In this respect, the user's attention is drawn to the risks associated
 * with loading,  using,  modifying and/or developing or reproducing the
 * software by the user in light of its specific status of free software,
 * that may mean  that it is complicated to manipulate,  and  that  also
 * therefore means  that it is reserved for developers  and  experienced
 * professionals having in-depth computer knowledge. Users are therefore
 * encouraged to load and test the software's suitability as regards their
 * requirements in conditions enabling the security of their systems and/or
 * data to be ensured and,  more generally, to use and operate it in the
 * same conditions as regards security.
 *
 * The fact that you are presently reading this means that you have had
 * knowledge of the CeCILL license and that you accept its terms.
 *
 */


#include <iostream>
#include <cmath>
#include "starlet2d.h"

Starlet2D::Starlet2D(int nx, int ny,int nz, int Nscales,int lh,int cscale)
{
    Nx = nx;
    Ny = ny;
    Nz = nz;
    J = Nscales;
    Lh = lh;
    Cscale = cscale;
}

// 1D Filtering routine with mirror border option
// Input, Output, Filter, Filter length

//def Exp_Sn(xref,v,theta):
//
//    m = len(xref)
//
//    # Projecting onto the xref,v plane
//
//    F = np.zeros((m,2))
//
//    F[:,0] = xref
//    F[:,1] = v
//    F = np.dot(F,RotMatrix(-theta))
//
//    return F[:,0]
//
//def Log_Sn(xref,x):
//
//    nX = np.shape(x)
//
//    m = nX[0]
//    t = nX[1]
//
//    G = np.zeros((t,))
//    Gv = np.zeros((m,t))
//
//    for r in range(t):
//
//        # Correct for permuations
//
//        Xout = dp(x[:,r])
//
//        a = np.sum(Xout*xref)/np.sqrt(np.sum(xref**2)*np.sum(Xout**2)) # Should have unit L2 norm
//        G[r] = np.arccos(a)  # Computing the angles
//        if a > 1:
//            a = 1
//        if a < -1:
//            a = -1
//
//        v = Xout - a*xref
//        Gv[:,r] = v / (1e-24 + np.linalg.norm(v))   # Unit vector in the tangent subspace
//
//    return G,Gv


//def Filtering_Sn(xref,x,h):
//
//    m = len(xref)
//
//    G,Gv = Log_Sn(xref,x)
//    vout = np.zeros((m,))
//    nh = np.sum(h)
//    Xout = dp(xref)
//    theta = 0.
//
//    for r in range(len(h)):
//        vout += h[r]*G[r]*Gv[:,r]
//
//    theta = np.linalg.norm(vout)/nh
//    vout = vout/(1e-12+np.linalg.norm(vout))
//
//    Xout = Exp_Sn(xref,vout,theta)
//
//    return Xout

void Starlet2D::RotateVector(double* x1, double* x2, double* xout, double Theta)
{
	int i = 0;

	for (i=0;i<Nz;i++)
    {
    	xout[i] = std::cos(Theta)*x1[i] + std::sin(Theta)*x2[i];
    }
}

// EXP SN

void Starlet2D::Exp_Sn(double* xref, double* v, double* xout, double Theta)
{
	RotateVector(xref, v, xout, Theta);
}

// LOG SN

double Starlet2D::Log_Sn(double* x1, double* x2, double* Gout)
{
	double a = 0;
	double n1 = 0;
	double n2 = 0;
	double nv = 0;
	double nn;
	int i = 0;

	for (i=0;i<Nz;i++)
    {
    	a += x1[i]*x2[i];
    	n1 += x1[i]*x1[i];
    	n2 += x2[i]*x2[i];
    }

    a = a/(1e-32 + std::sqrt(n1*n2));

    if (a < -1.){
      a = -1.;
    }

    if (a > 1.){
      a = 1.;
    }

    double aout = std::acos(a);

    for (i=0;i<Nz;i++)
    {
    	Gout[i] = x2[i] - a*x1[i];
    	nv += Gout[i]*Gout[i];
    }

    nn = 1e-32 + sqrt(nv);

    for (i=0;i<Nz;i++)
    {
    	Gout[i] = Gout[i]/nn;
    }

    return aout;
}

// FILTER 1D

void Starlet2D::filter_1d(double* xin,double* xout,double* xtemp1, double* xtemp2, double* pt_CS, double* Gtemp, double* Theta,int N,double* h,int scale) // FILTER 1D ON A COLUMN/ROW VECTOR
{
  int i = 0;
  int j = 0;
  int k = 0;

  int m2 = Lh/2;

  double val = 0;
  double valout = 0;

  int Lindix = 0;
  int Tindix = 0;
  double a = 0;

  for (i=0;i<N;i++)
  {
      valout = 0;

      for (k=0;k<Nz;k++)
	     {
	  	     xtemp1[k] = xin[k + Nz*i]; // The reference point
	     }

      for (j=0;j<Lh;j++)
      {
           Lindix = i + std::pow(2,scale)*(j - m2);

           Tindix = Lindix;

           if (Lindix < 0){
      	    	Lindix = -Tindix-1;
	  		   }

           if (Lindix > N-1){
	    		    Lindix = 2*N - Tindix - 1;
	  		   }

	  		   for (k=0;k<Nz;k++)
	  		   {
	  			    xtemp2[k] = xin[k + Nz*Lindix];
	  		   }

	  		   a = Log_Sn(xtemp1, xtemp2, Gtemp);

	  		   for (k=0;k<Nz;k++)
	  		   {

             if (j == 0){
	  				        xout[k + Nz*i] = h[j]*a*Gtemp[k];
	  			   }

	  			   if (j > 0){
	  				        xout[k + Nz*i] += h[j]*a*Gtemp[k];
	  			   }

	  		    }

	  		    valout += h[j]*a; // could be computed modulo pi
		     }

    // Normalizing the vector in the tangent space

    val = 0;

    for (k=0;k<Nz;k++)
	  	{
	  		val += xout[k + Nz*i]*xout[k + Nz*i];
	  	}

    valout = std::sqrt(val);


		for (k=0;k<Nz;k++)
	  	{
	  		xtemp2[k] = xout[k + Nz*i]/(1e-32 + valout);  // Normalization
	  	}

	  	Exp_Sn(xtemp1, xtemp2, Gtemp, valout); // Projecting back in Rn

	  	for (k=0;k<Nz;k++)
	  	{
	  		xout[k + Nz*i] = xtemp2[k];
	  		pt_CS[k + Nz*i] = Gtemp[k];
	  	}

	  	Theta[i] = valout; // Final angle value

    }

}


// FILTER 1D

void Starlet2D::filter_Ref_1d(double* xin,double* xref,double* xout,double* xtemp1, double* xtemp2, double* pt_CS, double* Gtemp, double* Theta,int N,double* h,int scale) // FILTER 1D ON A COLUMN/ROW VECTOR
{
  int i = 0;
  int j = 0;
  int k = 0;

  int m2 = Lh/2;

  double val = 0;
  double valout = 0;

  int Lindix = 0;
  int Tindix = 0;
  double a = 0;

  for (i=0;i<N;i++)
  {
      valout = 0;

      for (k=0;k<Nz;k++)
	     {
	  	     xtemp1[k] = xref[k + Nz*i]; // The reference point
	     }

      for (j=0;j<Lh;j++)
      {
           Lindix = i + std::pow(2,scale)*(j - m2);

           Tindix = Lindix;

           if (Lindix < 0){
      	    	Lindix = -Tindix-1;
	  		   }

           if (Lindix > N-1){
	    		    Lindix = 2*N - Tindix - 1;
	  		   }

	  		   for (k=0;k<Nz;k++)
	  		   {
	  			    xtemp2[k] = xin[k + Nz*Lindix];
	  		   }

	  		   a = Log_Sn(xtemp1, xtemp2, Gtemp);

	  		   for (k=0;k<Nz;k++)
	  		   {

             if (j == 0){
	  				        xout[k + Nz*i] = h[j]*a*Gtemp[k];
	  			   }

	  			   if (j > 0){
	  				        xout[k + Nz*i] += h[j]*a*Gtemp[k];
	  			   }

	  		    }

	  		    valout += h[j]*a; // could be computed modulo pi
		     }

    // Normalizing the vector in the tangent space

    val = 0;

    for (k=0;k<Nz;k++)
	  	{
	  		val += xout[k + Nz*i]*xout[k + Nz*i];
	  	}

    valout = std::sqrt(val);

		for (k=0;k<Nz;k++)
	  	{
	  		xtemp2[k] = xout[k + Nz*i]/(1e-32 + valout);  // Normalization
	  	}

	  	Exp_Sn(xtemp1, xtemp2, Gtemp, valout); // Projecting back in Rn

	  	for (k=0;k<Nz;k++)
	  	{
	  		xout[k + Nz*i] = xtemp2[k];
	  		pt_CS[k + Nz*i] = Gtemp[k];
	  	}

	  	Theta[i] = valout; // Final angle value

    }

}


// Forward starlet transform

//void Starlet2D::transform(double* In, double* Wt_Out, double* CR_Out, double* xIn, double* yIn, double* xOut, double* yOut, double* h)
//{
//  int i = 0;
//  int j = 0;
//  int k = 0;
//
//  for (j=0;j<J;j++) // Looping over the scales
//    {
//
//      for (i=0;i<Nx;i++)  // Looping over the rows
//	{
//	  for (k=0;k<Ny;k++)
//	    {
//	      yIn[k] = In[i + k*Nx];
//	    }
//
//	  filter_1d(yIn,yOut,Ny,h,j);
//
//	  for (k=0;k<Ny;k++)
//	    {
//	      CR_Out[i + k*Nx] = yOut[k];
//	    }
//
//	}
//
//      for (i=0;i<Ny;i++)  // Looping over the columns
//	{
//	  for (k=0;k<Nx;k++)
//	    {
//	      xIn[k] = CR_Out[k+i*Nx];
//	    }
//
//	  filter_1d(xIn,xOut,Nx,h,j);
//
//	  for (k=0;k<Nx;k++)
//	    {
//	      CR_Out[k+i*Nx] = xOut[k];
//	    }
//	}
//
//      // Updating the wavelet coefficients
//
//      for (i=0;i<Ny;i++)
//	{
//	  for (k=0;k<Nx;k++)
//	    {
//	      Wt_Out[k+i*Nx+j*Nx*Ny] = In[k + i*Nx] - CR_Out[k + i*Nx];
//	      In[k + i*Nx] =  CR_Out[k + i*Nx];
//	    }
//	}
//    }
//}
//
//
//void Starlet2D::reconstruct(double* CR, double* WT )
//{
//  int i=0;
//  int j=0;
//  int k=0;
//
//  for (j=0;j<J;j++)
//    {
//      for (i=0;i<Ny;i++)
//	{
//	  for (k=0;k<Nx;k++)
//	    {
//	      CR[k + i*Nx] += WT[k+i*Nx+j*Nx*Ny];
//	    }
//	}
//    }
//}
