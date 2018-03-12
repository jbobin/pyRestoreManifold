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

#ifndef STARLET2D_H
#define STARLET2D_H

#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include "omp.h"
#include "NumPyArrayData.h"
#include <cmath>

namespace bp = boost::python;
namespace np = boost::numpy;

class Starlet2D
{

public:
  Starlet2D(int Nx, int Ny, int Nz, int nscales,int lh,int cscale);

  void RotateVector(double* x1, double* x2, double* xout, double Theta);
  void Exp_Sn(double* xref, double* v, double* xout, double Theta);
  double Log_Sn(double* x1, double* x2, double* Gout);
  void filter_1d(double* xin,double* xout,double* xtemp1, double* xtemp2, double* pt_CS, double* Gtemp, double* Theta,int N,double* h,int scale);
  void filter_Ref_1d(double* xin,double* xref,double* xout,double* xtemp1, double* xtemp2, double* pt_CS, double* Gtemp, double* Theta,int N,double* h,int scale);

  // Testing LOG_SN

  np::ndarray filter2d(np::ndarray &In, np::ndarray &Filter){

  	// DEFINE THE INPUT VARIABLES

  	NumPyArrayData<double> In_data(In);
	  NumPyArrayData<double> F_data(Filter);

	// DEFINE THE OUTPUT VARIABLES

    np::ndarray Out = np::zeros(bp::make_tuple(Nz,Nx,Ny), np::dtype::get_builtin<double>());
    NumPyArrayData<double> Out_data(Out);

    double *pt_In = (double *) malloc(sizeof(double)*Nz*Nx*Ny); // The current approximation
    double *pt_App = (double *) malloc(sizeof(double)*Nz*Nx*Ny); // The running approximation

    double *pt_Row = (double *) malloc(sizeof(double)*Nx*Nz);
    double *pt_CS = (double *) malloc(sizeof(double)*Nx*Nz);    // One output of the filtering code
    double *pt_Out = (double *) malloc(sizeof(double)*Nx*Nz);   // One output of the filtering code

    double *pt_Temp1 = (double *) malloc(sizeof(double)*Nz);
    double *pt_Temp2 = (double *) malloc(sizeof(double)*Nz);
    double *pt_Gtemp = (double *) malloc(sizeof(double)*Nz);

    double *pt_F = (double *) malloc(sizeof(double)*Lh);
    double *pt_Theta = (double *) malloc(sizeof(double)*Nx);

    double a = 0;

    for (int x=0; x < Lh; x++)
	  {
	    pt_F[x] = F_data(x);
	  }
//
//
//
    // Feed the data

    for (int y=0;y<Ny;y++){
	   for (int k =0; k< Nz; k++){
		    for (int x=0; x < Nx; x++){
			    pt_In[k + x*Nz + y*Nx*Nz] = In_data(k,x,y);
			}
		}
    }

		// Filtering all columns

    	for (int y=0;y<Ny;y++){
    		  for (int k =0; k< Nz; k++){
		    	     for (int x=0; x < Nx; x++){
			    	         pt_Row[k + x*Nz] = pt_In[k + x*Nz + y*Nx*Nz];
				       }
			    }

			    // Filtering

			   filter_1d(pt_Row,pt_Out,pt_Temp1,pt_Temp2,pt_CS, pt_Gtemp,pt_Theta,Nx,pt_F,Cscale);

			   // Just collect the approximation

			   for (int k =0; k< Nz; k++){
		    	    for (int x=0; x < Nx; x++){
			    	        pt_App[k + x*Nz + y*Nx*Nz] = pt_CS[k + x*Nz];
				      }
			   }
    	}

    	// Filtering all the lines (BEWARE Nx = Ny !!!!!!)

    	for (int x=0; x < Nx; x++){
    		  for (int k =0; k< Nz; k++){
    			     for (int y=0;y<Ny;y++){
			    	         pt_Row[k + y*Nz] = pt_App[k + x*Nz + y*Nx*Nz];
				       }
		      }

			    // Filtering

			    filter_1d(pt_Row,pt_Out,pt_Temp1,pt_Temp2,pt_CS, pt_Gtemp,pt_Theta,Nx,pt_F,Cscale);

			    // Just collect the approximation

			    for (int k =0; k< Nz; k++){
    			   for (int y=0;y<Ny;y++){
			    	       pt_App[k + x*Nz + y*Nx*Nz] = pt_CS[k + y*Nz];
				     }
			    }
      }

    	// COMPUTE THE DISCREPANCY WITH THE APPROXIMATION IN THE TANGENT PLANE

    	for (int y=0;y<Ny;y++){
    		  for (int x=0;x<Nx;x++){
              for (int k=0;k<Nz;k++){
				            Out_data(k,x,y) = pt_App[k + x*Nz + y*Nx*Nz];
				       }
			     }
    	}

	return Out;

}
  // FORWARD 2D

    np::ndarray forward2d(np::ndarray &In, np::ndarray &Filter){

  	// DEFINE THE INPUT VARIABLES

  	NumPyArrayData<double> In_data(In);
	  NumPyArrayData<double> F_data(Filter);

	    // DEFINE THE OUTPUT VARIABLES

    np::ndarray Out = np::zeros(bp::make_tuple(Nz+1,Nx,Ny,J+1), np::dtype::get_builtin<double>());
    NumPyArrayData<double> Out_data(Out);

    double *pt_In = (double *) malloc(sizeof(double)*Nz*Nx*Ny); // The current approximation
    double *pt_App = (double *) malloc(sizeof(double)*Nz*Nx*Ny); // The running approximation

    double *pt_Row = (double *) malloc(sizeof(double)*Nx*Nz);
    double *pt_CS = (double *) malloc(sizeof(double)*Nx*Nz);    // One output of the filtering code
    double *pt_Out = (double *) malloc(sizeof(double)*Nx*Nz);   // One output of the filtering code

    double *pt_Temp1 = (double *) malloc(sizeof(double)*Nz);
    double *pt_Temp2 = (double *) malloc(sizeof(double)*Nz);
    double *pt_Gtemp = (double *) malloc(sizeof(double)*Nz);

    double *pt_F = (double *) malloc(sizeof(double)*Lh);
    double *pt_Theta = (double *) malloc(sizeof(double)*Nx);

    double a = 0;

    for (int x=0; x < Lh; x++)
	  {
	    pt_F[x] = F_data(x);
	  }
//
//
//
    // Feed the data

    for (int y=0;y<Ny;y++){
	   for (int k =0; k< Nz; k++){
		    for (int x=0; x < Nx; x++){
			    pt_In[k + x*Nz + y*Nx*Nz] = In_data(k,x,y);
			}
		}
    }

	// FILTERING ALL THE LINES

	for (int scale=0; scale < J; scale++){

		// Filtering all columns

    	for (int y=0;y<Ny;y++){
    		  for (int k =0; k< Nz; k++){
		    	     for (int x=0; x < Nx; x++){
			    	         pt_Row[k + x*Nz] = pt_In[k + x*Nz + y*Nx*Nz];
				       }
			    }

			    // Filtering

			   filter_1d(pt_Row,pt_Out,pt_Temp1,pt_Temp2,pt_CS, pt_Gtemp,pt_Theta,Nx,pt_F,scale);

			   // Just collect the approximation

			   for (int k =0; k< Nz; k++){
		    	    for (int x=0; x < Nx; x++){
			    	        pt_App[k + x*Nz + y*Nx*Nz] = pt_CS[k + x*Nz];
				      }
			   }
    	}

    	// Filtering all the lines (BEWARE Nx = Ny !!!!!!)

    	for (int x=0; x < Nx; x++){
    		  for (int k =0; k< Nz; k++){
    			     for (int y=0;y<Ny;y++){
			    	         pt_Row[k + y*Nz] = pt_App[k + x*Nz + y*Nx*Nz];
				       }
		      }

			    // Filtering

			    filter_1d(pt_Row,pt_Out,pt_Temp1,pt_Temp2,pt_CS, pt_Gtemp,pt_Theta,Nx,pt_F,scale);

			    // Just collect the approximation

			    for (int k =0; k< Nz; k++){
    			   for (int y=0;y<Ny;y++){
			    	       pt_App[k + x*Nz + y*Nx*Nz] = pt_CS[k + y*Nz];
				     }
			    }
      }

    	// COMPUTE THE DISCREPANCY WITH THE APPROXIMATION IN THE TANGENT PLANE

    	for (int y=0;y<Ny;y++){
    		  for (int x=0;x<Nx;x++){
  				      for (int k =0; k< Nz; k++){
			    	          pt_Temp1[k] = pt_In[k + x*Nz + y*Nx*Nz];
			    	          pt_Temp2[k] = pt_App[k + x*Nz + y*Nx*Nz];
				        }

				        a = Log_Sn(pt_Temp1, pt_Temp2, pt_Gtemp);

				        Out_data(Nz,x,y,scale) = a;  // angle delta

				        for (int k =0; k< Nz; k++){
					             Out_data(k,x,y,scale) = pt_Gtemp[k];
					             if (scale == J-1){
						                Out_data(k,x,y,J) = pt_App[k + x*Nz + y*Nx*Nz]; // Get the coarse scale
					             }
			    	           pt_In[k + x*Nz + y*Nx*Nz] = pt_App[k + x*Nz + y*Nx*Nz];  // pt_In becomes the new approximation
				        }
			     }
    	}

	}

	return Out;

 }


// FORWARD 2D

  np::ndarray forward2d_omp(np::ndarray &In, np::ndarray &Filter){

  	// DEFINE THE INPUT VARIABLES

  	NumPyArrayData<double> In_data(In);
	NumPyArrayData<double> F_data(Filter);

	// DEFINE THE OUTPUT VARIABLES

    np::ndarray Out = np::zeros(bp::make_tuple(Nz+1,Nx,Ny,J+1), np::dtype::get_builtin<double>());
    NumPyArrayData<double> Out_data(Out);

    double *pt_In = (double *) malloc(sizeof(double)*Nz*Nx*Ny); // The current approximation
    double *pt_App = (double *) malloc(sizeof(double)*Nz*Nx*Ny); // The running approximation
    double *pt_F = (double *) malloc(sizeof(double)*Lh);

    double a = 0;
    int yl = 0;
    int xl = 0;

    for (int x=0; x < Lh; x++)
	  {
	    pt_F[x] = F_data(x);
	  }
//
//
//
    // Feed the data

    for (int y=0;y<Ny;y++){
	   for (int k =0; k< Nz; k++){
		    for (int x=0; x < Nx; x++){
			    pt_In[k + x*Nz + y*Nx*Nz] = In_data(k,x,y);
			}
		}
    }

	// FILTERING ALL THE LINES

	for (int scale=0; scale < J; scale++){

		// Filtering all columns

		#pragma omp parallel for shared(pt_In,pt_App, pt_F, yl,scale)
    	for (yl=0;yl<Ny;yl++){

    		// DEFINE USEFUL VARIABLES

    		double *pt_Row = (double *) malloc(sizeof(double)*Nx*Nz);
    		double *pt_CS = (double *) malloc(sizeof(double)*Nx*Nz);    // One output of the filtering code
    		double *pt_Out = (double *) malloc(sizeof(double)*Nx*Nz);   // One output of the filtering code
    		double *pt_Temp1 = (double *) malloc(sizeof(double)*Nz);
    		double *pt_Temp2 = (double *) malloc(sizeof(double)*Nz);
    		double *pt_Gtemp = (double *) malloc(sizeof(double)*Nz);
    		double *pt_Theta = (double *) malloc(sizeof(double)*Nx);

    		// Get the data

    		for (int k =0; k< Nz; k++){
		    	for (int x=0; x < Nx; x++){
			    	pt_Row[k + x*Nz] = pt_In[k + x*Nz + yl*Nx*Nz];
				}
			}

			// Filtering

			filter_1d(pt_Row,pt_Out,pt_Temp1,pt_Temp2,pt_CS, pt_Gtemp,pt_Theta,Nx,pt_F,scale);

			// Just collect the approximation

			for (int k =0; k< Nz; k++){
		    	for (int x=0; x < Nx; x++){
			    	 pt_App[k + x*Nz + yl*Nx*Nz] = pt_CS[k + x*Nz];
				}
			}

			free(pt_Row);
			free(pt_CS);
			free(pt_Out);
			free(pt_Temp1);
			free(pt_Temp2);
			free(pt_Gtemp);
			free(pt_Theta);

    	}

    	// Filtering all the lines (BEWARE Nx = Ny !!!!!!)

    	#pragma omp parallel for shared(pt_In,pt_App, pt_F, xl,scale)
    	for (xl=0; xl < Nx; xl++){

    		double *pt_Row = (double *) malloc(sizeof(double)*Nx*Nz);
    		double *pt_CS = (double *) malloc(sizeof(double)*Nx*Nz);    // One output of the filtering code
    		double *pt_Out = (double *) malloc(sizeof(double)*Nx*Nz);   // One output of the filtering code
    		double *pt_Temp1 = (double *) malloc(sizeof(double)*Nz);
    		double *pt_Temp2 = (double *) malloc(sizeof(double)*Nz);
    		double *pt_Gtemp = (double *) malloc(sizeof(double)*Nz);
    		double *pt_Theta = (double *) malloc(sizeof(double)*Nx);

    		for (int k =0; k< Nz; k++){
    			for (int y=0;y<Ny;y++){
			    	pt_Row[k + y*Nz] = pt_App[k + xl*Nz + y*Nx*Nz];
				}
			}

			// Filtering

			filter_1d(pt_Row,pt_Out,pt_Temp1,pt_Temp2,pt_CS, pt_Gtemp,pt_Theta,Nx,pt_F,scale);

			// Just collect the approximation

			for (int k =0; k< Nz; k++){
    			for (int y=0;y<Ny;y++){
			    	 pt_App[k + xl*Nz + y*Nx*Nz] = pt_CS[k + y*Nz];
				}
			}

			free(pt_Row);
			free(pt_CS);
			free(pt_Out);
			free(pt_Temp1);
			free(pt_Temp2);
			free(pt_Gtemp);
			free(pt_Theta);

    	}

    	// COMPUTE THE DISCREPANCY WITH THE APPROXIMATION IN THE TANGENT PLANE

    	double *pt_Temp1 = (double *) malloc(sizeof(double)*Nz);
    	double *pt_Temp2 = (double *) malloc(sizeof(double)*Nz);
    	double *pt_Gtemp = (double *) malloc(sizeof(double)*Nz);

    	for (int y=0;y<Ny;y++){
    		for (int x=0;x<Nx;x++){
				for (int k =0; k< Nz; k++){
			    	 pt_Temp1[k] = pt_In[k + x*Nz + y*Nx*Nz];
			    	 pt_Temp2[k] = pt_App[k + x*Nz + y*Nx*Nz];
				}

				a = Log_Sn(pt_Temp1, pt_Temp2, pt_Gtemp);

				Out_data(Nz,x,y,scale) = a;  // angle delta

				for (int k =0; k< Nz; k++){
					Out_data(k,x,y,scale) = pt_Gtemp[k];
					if (scale == J-1){
						Out_data(k,x,y,J) = pt_App[k + x*Nz + y*Nx*Nz]; // Get the coarse scale
					}
			    	pt_In[k + x*Nz + y*Nx*Nz] = pt_App[k + x*Nz + y*Nx*Nz];  // pt_In becomes the new approximation
				}
			}
    	}

    free(pt_Temp1);
		free(pt_Temp2);
		free(pt_Gtemp);

	}

	return Out;

  free(pt_In);
  free(pt_App);
  free(pt_F);

  }



  // FORWARD 2D WITH REFERENCE SIGNAL V2 (MULTIRESOLUTION APPROXIMATION)

    np::ndarray forward2d_mr_ref_omp(np::ndarray &In, np::ndarray &Ref, np::ndarray &Filter){

    	// DEFINE THE INPUT VARIABLES

    NumPyArrayData<double> In_data(In);
    NumPyArrayData<double> Ref_data(Ref);
  	NumPyArrayData<double> F_data(Filter);

  	// DEFINE THE OUTPUT VARIABLES

      np::ndarray Out = np::zeros(bp::make_tuple(Nz+1,Nx,Ny,J+1), np::dtype::get_builtin<double>());
      NumPyArrayData<double> Out_data(Out);

      double *pt_In = (double *) malloc(sizeof(double)*Nz*Nx*Ny); // The current approximation
      double *pt_Ref = (double *) malloc(sizeof(double)*Nz*Nx*Ny*J); // The current approximation
      double *pt_App = (double *) malloc(sizeof(double)*Nz*Nx*Ny); // The running approximation
      double *pt_F = (double *) malloc(sizeof(double)*Lh);

      double a = 0;
      int yl = 0;
      int xl = 0;

      for (int x=0; x < Lh; x++)
  	  {
  	    pt_F[x] = F_data(x);
  	  }

      // Feed the data

      for (int y=0;y<Ny;y++){
  	   for (int k =0; k< Nz; k++){
  		    for (int x=0; x < Nx; x++){
  			    pt_In[k + x*Nz + y*Nx*Nz] = In_data(k,x,y);
            for (int s=0; s < J; s++){
    			    pt_Ref[k + x*Nz + y*Nx*Nz + s*Nx*Nz*Ny] = Ref_data(k,x,y,s);
            }
  			  }
  		  }
      }

  	// FILTERING ALL THE LINES

  	for (int scale=0; scale < J; scale++){

  		// Filtering all columns

  		#pragma omp parallel for shared(pt_In,pt_Ref,pt_App,pt_F,yl,scale)
      	for (yl=0;yl<Ny;yl++){

      		// DEFINE USEFUL VARIABLES

      		double *pt_Row = (double *) malloc(sizeof(double)*Nx*Nz);
          double *pt_Row_Ref = (double *) malloc(sizeof(double)*Nx*Nz);
      		double *pt_CS = (double *) malloc(sizeof(double)*Nx*Nz);
          double *pt_CS_Ref = (double *) malloc(sizeof(double)*Nx*Nz);    // One output of the filtering code
      		double *pt_Out = (double *) malloc(sizeof(double)*Nx*Nz);   // One output of the filtering code
      		double *pt_Temp1 = (double *) malloc(sizeof(double)*Nz);
      		double *pt_Temp2 = (double *) malloc(sizeof(double)*Nz);
      		double *pt_Gtemp = (double *) malloc(sizeof(double)*Nz);
      		double *pt_Theta = (double *) malloc(sizeof(double)*Nx);

      		// Get the data

      		for (int k =0; k< Nz; k++){
  		    	for (int x=0; x < Nx; x++){
  			    	pt_Row[k + x*Nz] = pt_In[k + x*Nz + yl*Nx*Nz];
              pt_Row_Ref[k + x*Nz] = pt_Ref[k + x*Nz + yl*Nx*Nz + scale*Nx*Nz*Ny];
  				}
  			}

  			// Filtering

  			filter_Ref_1d(pt_Row,pt_Row_Ref,pt_Out,pt_Temp1,pt_Temp2,pt_CS, pt_Gtemp,pt_Theta,Nx,pt_F,scale);  // With respect to the reference point

  			// Just collect the approximation

  			for (int k =0; k< Nz; k++){
  		    	for (int x=0; x < Nx; x++){
  			    	 pt_App[k + x*Nz + yl*Nx*Nz] = pt_CS[k + x*Nz];
  				}
  			}

  			free(pt_Row);
  			free(pt_CS);
  			free(pt_CS_Ref);
  			free(pt_Out);
  			free(pt_Temp1);
  			free(pt_Temp2);
  			free(pt_Gtemp);
  			free(pt_Theta);

      	}

      	// Filtering all the lines (BEWARE Nx = Ny !!!!!!)

      	#pragma omp parallel for shared(pt_In,pt_Ref,pt_App, pt_F, xl,scale)
      	for (xl=0; xl < Nx; xl++){

      		double *pt_Row = (double *) malloc(sizeof(double)*Nx*Nz);
      		double *pt_CS = (double *) malloc(sizeof(double)*Nx*Nz);    // One output of the filtering code
          double *pt_Row_Ref = (double *) malloc(sizeof(double)*Nx*Nz);
      		double *pt_CS_Ref = (double *) malloc(sizeof(double)*Nx*Nz);    // One output of the filtering code
      		double *pt_Out = (double *) malloc(sizeof(double)*Nx*Nz);   // One output of the filtering code
      		double *pt_Temp1 = (double *) malloc(sizeof(double)*Nz);
      		double *pt_Temp2 = (double *) malloc(sizeof(double)*Nz);
      		double *pt_Gtemp = (double *) malloc(sizeof(double)*Nz);
      		double *pt_Theta = (double *) malloc(sizeof(double)*Nx);

      		for (int k =0; k< Nz; k++){
      			for (int y=0;y<Ny;y++){
  			    	pt_Row[k + y*Nz] = pt_App[k + xl*Nz + y*Nx*Nz];
              pt_Row_Ref[k + y*Nz] = pt_Ref[k + xl*Nz + y*Nx*Nz + scale*Nx*Ny*Nz];
  				}
  			}

  			// Filtering

        filter_Ref_1d(pt_Row,pt_Row_Ref,pt_Out,pt_Temp1,pt_Temp2,pt_CS, pt_Gtemp,pt_Theta,Nx,pt_F,scale);  // With respect to the reference point

  			// Just collect the approximation

  			for (int k =0; k< Nz; k++){
      			for (int y=0;y<Ny;y++){
  			    	 pt_App[k + xl*Nz + y*Nx*Nz] = pt_CS[k + y*Nz];
  				}
  			}

  			free(pt_Row);
  			free(pt_CS);
  			free(pt_CS_Ref);
  			free(pt_Out);
  			free(pt_Temp1);
  			free(pt_Temp2);
  			free(pt_Gtemp);
  			free(pt_Theta);

      	}

      	// COMPUTE THE DISCREPANCY WITH THE APPROXIMATION IN THE TANGENT PLANE

      	double *pt_Temp1 = (double *) malloc(sizeof(double)*Nz);
      	double *pt_Temp2 = (double *) malloc(sizeof(double)*Nz);
      	double *pt_Gtemp = (double *) malloc(sizeof(double)*Nz);

        for (int y=0;y<Ny;y++){
      		for (int x=0;x<Nx;x++){
  				for (int k =0; k< Nz; k++){
  			    	 pt_Temp1[k] = pt_In[k + x*Nz + y*Nx*Nz];
  			    	 pt_Temp2[k] = pt_App[k + x*Nz + y*Nx*Nz];
  				}

  				a = Log_Sn(pt_Temp1, pt_Temp2, pt_Gtemp);

  				Out_data(Nz,x,y,scale) = a;  // angle delta

  				for (int k =0; k< Nz; k++){
  					Out_data(k,x,y,scale) = pt_Gtemp[k];
  					if (scale == J-1){
  						Out_data(k,x,y,J) = pt_App[k + x*Nz + y*Nx*Nz]; // Get the coarse scale
  					}
  			    	pt_In[k + x*Nz + y*Nx*Nz] = pt_App[k + x*Nz + y*Nx*Nz];  // pt_In becomes the new approximation
  				}
  			}
      	}

      	free(pt_Temp1);
  		free(pt_Temp2);
  		free(pt_Gtemp);

  	}

  	return Out;

    free(pt_In);
    free(pt_Ref);
    free(pt_App);
    free(pt_F);

    }

    // FORWARD 2D WITH REFERENCE SIGNAL

      np::ndarray forward2d_ref_omp(np::ndarray &In, np::ndarray &Ref, np::ndarray &Filter){

        // DEFINE THE INPUT VARIABLES

      NumPyArrayData<double> In_data(In);
      NumPyArrayData<double> Ref_data(Ref);
      NumPyArrayData<double> F_data(Filter);

      // DEFINE THE OUTPUT VARIABLES

        np::ndarray Out = np::zeros(bp::make_tuple(Nz+1,Nx,Ny,J+1), np::dtype::get_builtin<double>());
        NumPyArrayData<double> Out_data(Out);

        double *pt_In = (double *) malloc(sizeof(double)*Nz*Nx*Ny); // The current approximation
        double *pt_Ref = (double *) malloc(sizeof(double)*Nz*Nx*Ny); // The current approximation
        double *pt_App = (double *) malloc(sizeof(double)*Nz*Nx*Ny); // The running approximation
        double *pt_App_Ref = (double *) malloc(sizeof(double)*Nz*Nx*Ny); // The running approximation
        double *pt_F = (double *) malloc(sizeof(double)*Lh);

        double a = 0;
        int yl = 0;
        int xl = 0;

        for (int x=0; x < Lh; x++)
        {
          pt_F[x] = F_data(x);
        }
    //
    //
    //
        // Feed the data

        for (int y=0;y<Ny;y++){
         for (int k =0; k< Nz; k++){
            for (int x=0; x < Nx; x++){
              pt_In[k + x*Nz + y*Nx*Nz] = In_data(k,x,y);
          }
        }
        }

      // FILTERING ALL THE LINES

      for (int scale=0; scale < J; scale++){

        // Filtering all columns

        #pragma omp parallel for shared(pt_In,pt_Ref,pt_App,pt_App_Ref, pt_F, yl,scale)
          for (yl=0;yl<Ny;yl++){

            // DEFINE USEFUL VARIABLES

            double *pt_Row = (double *) malloc(sizeof(double)*Nx*Nz);
            double *pt_Row_Ref = (double *) malloc(sizeof(double)*Nx*Nz);
            double *pt_CS = (double *) malloc(sizeof(double)*Nx*Nz);
            double *pt_CS_Ref = (double *) malloc(sizeof(double)*Nx*Nz);    // One output of the filtering code
            double *pt_Out = (double *) malloc(sizeof(double)*Nx*Nz);   // One output of the filtering code
            double *pt_Temp1 = (double *) malloc(sizeof(double)*Nz);
            double *pt_Temp2 = (double *) malloc(sizeof(double)*Nz);
            double *pt_Gtemp = (double *) malloc(sizeof(double)*Nz);
            double *pt_Theta = (double *) malloc(sizeof(double)*Nx);

            // Get the data

            for (int k =0; k< Nz; k++){
              for (int x=0; x < Nx; x++){
                pt_Row[k + x*Nz] = pt_In[k + x*Nz + yl*Nx*Nz];
                pt_Row_Ref[k + x*Nz] = pt_Ref[k + x*Nz + yl*Nx*Nz];
            }
          }

          // Filtering

          filter_1d(pt_Row_Ref,pt_Out,pt_Temp1,pt_Temp2,pt_CS_Ref, pt_Gtemp,pt_Theta,Nx,pt_F,scale);  // With respect to the reference point
          filter_Ref_1d(pt_Row,pt_Ref,pt_Out,pt_Temp1,pt_Temp2,pt_CS, pt_Gtemp,pt_Theta,Nx,pt_F,scale);  // With respect to the reference point

          // Just collect the approximation

          for (int k =0; k< Nz; k++){
              for (int x=0; x < Nx; x++){
                 pt_App[k + x*Nz + yl*Nx*Nz] = pt_CS[k + x*Nz];
                 pt_App_Ref[k + x*Nz + yl*Nx*Nz] = pt_CS_Ref[k + x*Nz];
            }
          }

          free(pt_Row);
          free(pt_CS);
          free(pt_Row_Ref);
          free(pt_CS_Ref);
          free(pt_Out);
          free(pt_Temp1);
          free(pt_Temp2);
          free(pt_Gtemp);
          free(pt_Theta);

          }

          // Filtering all the lines (BEWARE Nx = Ny !!!!!!)

          #pragma omp parallel for shared(pt_In,pt_Ref,pt_App,pt_App_Ref, pt_F, xl,scale)
          for (xl=0; xl < Nx; xl++){

            double *pt_Row = (double *) malloc(sizeof(double)*Nx*Nz);
            double *pt_CS = (double *) malloc(sizeof(double)*Nx*Nz);    // One output of the filtering code
            double *pt_Row_Ref = (double *) malloc(sizeof(double)*Nx*Nz);
            double *pt_CS_Ref = (double *) malloc(sizeof(double)*Nx*Nz);    // One output of the filtering code
            double *pt_Out = (double *) malloc(sizeof(double)*Nx*Nz);   // One output of the filtering code
            double *pt_Temp1 = (double *) malloc(sizeof(double)*Nz);
            double *pt_Temp2 = (double *) malloc(sizeof(double)*Nz);
            double *pt_Gtemp = (double *) malloc(sizeof(double)*Nz);
            double *pt_Theta = (double *) malloc(sizeof(double)*Nx);

            for (int k =0; k< Nz; k++){
              for (int y=0;y<Ny;y++){
                pt_Row[k + y*Nz] = pt_App[k + xl*Nz + y*Nx*Nz];
                pt_Row_Ref[k + y*Nz] = pt_App_Ref[k + xl*Nz + y*Nx*Nz];
            }
          }

          // Filtering

          filter_1d(pt_Row_Ref,pt_Out,pt_Temp1,pt_Temp2,pt_CS_Ref, pt_Gtemp,pt_Theta,Nx,pt_F,scale);
          filter_Ref_1d(pt_Row,pt_Ref,pt_Out,pt_Temp1,pt_Temp2,pt_CS, pt_Gtemp,pt_Theta,Nx,pt_F,scale);  // With respect to the reference point

          // Just collect the approximation

          for (int k =0; k< Nz; k++){
              for (int y=0;y<Ny;y++){
                 pt_App[k + xl*Nz + y*Nx*Nz] = pt_CS[k + y*Nz];
                 pt_App_Ref[k + xl*Nz + y*Nx*Nz] = pt_CS_Ref[k + y*Nz];
            }
          }

          free(pt_Row);
          free(pt_CS);
          free(pt_Row_Ref);
          free(pt_CS_Ref);
          free(pt_Out);
          free(pt_Temp1);
          free(pt_Temp2);
          free(pt_Gtemp);
          free(pt_Theta);

          }

          // COMPUTE THE DISCREPANCY WITH THE APPROXIMATION IN THE TANGENT PLANE

          double *pt_Temp1 = (double *) malloc(sizeof(double)*Nz);
          double *pt_Temp2 = (double *) malloc(sizeof(double)*Nz);
          double *pt_Gtemp = (double *) malloc(sizeof(double)*Nz);

          for (int y=0;y<Ny;y++){
            for (int x=0;x<Nx;x++){
            for (int k =0; k< Nz; k++){
                 pt_Temp1[k] = pt_App_Ref[k + x*Nz + y*Nx*Nz];
                 pt_Temp2[k] = pt_App[k + x*Nz + y*Nx*Nz];
            }

            a = Log_Sn(pt_Temp1, pt_Temp2, pt_Gtemp);

            Out_data(Nz,x,y,scale) = a;  // angle delta

            for (int k =0; k< Nz; k++){
              Out_data(k,x,y,scale) = pt_Gtemp[k];
              if (scale == J-1){
                Out_data(k,x,y,J) = pt_App_Ref[k + x*Nz + y*Nx*Nz]; // Get the coarse scale
              }
                pt_In[k + x*Nz + y*Nx*Nz] = pt_App[k + x*Nz + y*Nx*Nz];  // pt_In becomes the new approximation
            }
          }
          }

          free(pt_Temp1);
        free(pt_Temp2);
        free(pt_Gtemp);

      }

      return Out;

      free(pt_In);
      free(pt_Ref);
      free(pt_App_Ref);
      free(pt_App);
      free(pt_F);

      }




// BACKWARD 2D

  np::ndarray backward2d_omp(np::ndarray &In){

  	// DEFINE THE INPUT VARIABLES

  	NumPyArrayData<double> In_data(In);

	// DEFINE THE OUTPUT VARIABLES

    np::ndarray Out = np::zeros(bp::make_tuple(Nz,Nx,Ny), np::dtype::get_builtin<double>());
    NumPyArrayData<double> Out_data(Out);

    double *pt_In = (double *) malloc(sizeof(double)*(Nz+1)*Nx*Ny*J); // The current approximation
    double *pt_App = (double *) malloc(sizeof(double)*Nz*Nx*Ny); // The running approximation
    int yl = 0;

    // Feed the data

    for (int y=0;y<Ny;y++){
	   for (int k =0; k< Nz+1; k++){
		    for (int x=0; x < Nx; x++){
		    	for (int sc=0;sc < J;sc++){
			    	pt_In[k + x*(Nz+1) + y*Nx*(Nz+1) + sc*Nx*Ny*(Nz+1)] = In_data(k,x,y,sc);  // Wavelets scales and vectors
		    	}
			}
		}
    }

    for (int y=0;y<Ny;y++){
	   for (int k =0; k< Nz; k++){
		    for (int x=0; x < Nx; x++){
			    pt_App[k + x*Nz + y*Nx*Nz] = In_data(k,x,y,J);  // Coarse scale
			}
		}
    }

	for (int sc=0; sc < J; sc++){ // DECREASE ORDER

		int scale = J-sc-1;

		// Filtering all columns

		#pragma omp parallel for shared(pt_In,pt_App,yl,scale)
    	for (yl=0;yl<Ny;yl++){
    		for (int x=0; x < Nx; x++){


    			double * nval = (double *) malloc(sizeof(double));
    			nval[0] = 0;

    			for (int k =0; k< Nz; k++){
    				pt_App[k + x*Nz + yl*Nx*Nz]  = pt_App[k + x*Nz + yl*Nx*Nz] - std::sin(pt_In[Nz + x*(Nz+1) + yl*Nx*(Nz+1) + scale*Nx*Ny*(Nz+1)])*pt_In[k + x*(Nz+1) + yl*Nx*(Nz+1) + scale*Nx*Ny*(Nz+1)];
    				nval[0] += pt_App[k + x*Nz + yl*Nx*Nz]*pt_App[k + x*Nz + yl*Nx*Nz];
				}

				for (int k =0; k< Nz; k++){
    				pt_App[k + x*Nz + yl*Nx*Nz]  = pt_App[k + x*Nz + yl*Nx*Nz]/(1e-12 + sqrt(nval[0]));  // Normalizing
				}

				free(nval);
			}
    	}
	}

	for (int x=0; x < Nx; x++){
		for (int y=0; y < Ny; y++){
			for (int k =0; k< Nz; k++){
			    	Out_data(k,x,y) = pt_App[k + x*Nz + y*Nx*Nz]; // Current approximation
			}
		}
	}

	return Out;

  free(pt_In);
  free(pt_App);

  }



  // BACKWARD 2D

    np::ndarray backward2d_omp_v2(np::ndarray &In){

    	// DEFINE THE INPUT VARIABLES

    	NumPyArrayData<double> In_data(In);

  	// DEFINE THE OUTPUT VARIABLES

      np::ndarray Out = np::zeros(bp::make_tuple(Nz,Nx,Ny), np::dtype::get_builtin<double>());
      NumPyArrayData<double> Out_data(Out);

      double *pt_In = (double *) malloc(sizeof(double)*(Nz+1)*Nx*Ny*J); // The current approximation
      double *pt_App = (double *) malloc(sizeof(double)*Nz*Nx*Ny); // The running approximation
      int yl = 0;

      // Feed the data

      for (int y=0;y<Ny;y++){
  	   for (int k =0; k< Nz+1; k++){
  		    for (int x=0; x < Nx; x++){
  		    	for (int sc=0;sc < J;sc++){
  			    	pt_In[k + x*(Nz+1) + y*Nx*(Nz+1) + sc*Nx*Ny*(Nz+1)] = In_data(k,x,y,sc);  // Wavelets scales and vectors
  		    	}
  			}
  		}
      }

      for (int y=0;y<Ny;y++){
  	   for (int k =0; k< Nz; k++){
  		    for (int x=0; x < Nx; x++){
  			    pt_App[k + x*Nz + y*Nx*Nz] = In_data(k,x,y,J);  // Coarse scale
  			}
  		}
      }

  	for (int sc=0; sc < J; sc++){ // DECREASE ORDER

  		int scale = J-sc-1;

  		// Filtering all columns

  		#pragma omp parallel for shared(pt_In,pt_App,yl,scale)
      	for (yl=0;yl<Ny;yl++){
      		for (int x=0; x < Nx; x++){

            double * pt_xref = (double *) malloc(sizeof(double)*Nz);
            double * pt_v = (double *) malloc(sizeof(double)*Nz);
            double * pt_xout = (double *) malloc(sizeof(double)*Nz);
            double * Theta = (double *) malloc(sizeof(double)); // The angle

            Theta[0] = -pt_In[Nz + x*(Nz+1) + yl*Nx*(Nz+1) + scale*Nx*Ny*(Nz+1)];

            for (int k =0; k< Nz; k++){
        				pt_xref[k]  = pt_App[k + x*Nz + yl*Nx*Nz];
                pt_v[k]  = pt_In[k + x*(Nz+1) + yl*Nx*(Nz+1) + scale*Nx*Ny*(Nz+1)];
    				}

             Exp_Sn(pt_xref, pt_v,  pt_xout, Theta[0]);

  				   for (int k =0; k< Nz; k++){
      				pt_App[k + x*Nz + yl*Nx*Nz]  = pt_xout[k];
  				   }

          free(Theta);
          free(pt_v);
          free(pt_xref);
          free(pt_xout);
  			}
      	}
  	}

  	for (int x=0; x < Nx; x++){
  		for (int y=0; y < Ny; y++){
  			for (int k =0; k< Nz; k++){
  			    	Out_data(k,x,y) = pt_App[k + x*Nz + y*Nx*Nz]; // Current approximation
  			}
  		}
  	}

  	return Out;

    free(pt_In);
    free(pt_App);

    }

//    void filter_1d(double* xin,double* xout,int N,double* h,int j);
//    void transform(double* In, double* Wt_Out, double* CR_Out, double* xIn, double* yIn, double* xOut, double* yOut, double* h);
//    np::ndarray transform_numpy(np::ndarray &In, np::ndarray &Filter){
//
//        // Objects to easily access the data in the arrays
//        NumPyArrayData<double> In_data(In);
//	NumPyArrayData<double> F_data(Filter);
//        np::ndarray Out = np::zeros(bp::make_tuple(Nz,Nx,Ny,J+1), np::dtype::get_builtin<double>());
//        NumPyArrayData<double> Out_data(Out);
//
//        double *pt_In = (double *) malloc(sizeof(double)*Nx*Ny);
//        double *pt_Out = (double *) malloc(sizeof(double)*Nx*Ny*(J+1));
//	double *pt_WT = (double *) malloc(sizeof(double)*Nx*Ny*J);
//	double *pt_CR = (double *) malloc(sizeof(double)*Nx*Ny);
//	double *pt_Row = (double *) malloc(sizeof(double)*Ny);
//	double *pt_Col = (double *) malloc(sizeof(double)*Nx);
//	double *pt_Ro = (double *) malloc(sizeof(double)*Ny);
//	double *pt_Co = (double *) malloc(sizeof(double)*Nx);
//	double *pt_F = (double *) malloc(sizeof(double)*Lh);
//
//	for (int x=0; x < Lh; x++)
//	  {
//	    pt_F[x] = F_data(x);
//	  }
//
//	for (int k =0; k< Nz; k++){
//
//	    for (int x=0; x < Nx; x++){
//	      for (int y =0; y< Ny; y++){
//		    pt_In[x + y*Nx] = In_data(k,x,y);
//		}
//	      }
//
//
//	    // Transform
//
//	    transform(pt_In, pt_WT, pt_CR, pt_Row, pt_Col, pt_Ro, pt_Co, pt_F);
//
//	    for (int x=0; x < Nx; x++) {
//	      for (int y =0; y< Ny; y++) {
//		for (int j =0; j< J; j++) {
//		   Out_data(k,x,y,j) = pt_WT[x + y*Nx + j*Nx*Ny];
//		}
//		Out_data(k,x,y,J) = pt_CR[x + y*Nx];
//	      }
//	    }
//
//	  }
//
//        free(pt_In);
//        free(pt_Out);
//	free(pt_WT);
//	free(pt_CR);
//	free(pt_Row);
//	free(pt_Col);
//	free(pt_Ro);
//	free(pt_Co);
//	free(pt_F);
//
//        return Out;
//    }
//
//    np::ndarray transform_omp_numpy(np::ndarray &In, np::ndarray &Filter){
//
//        // Objects to easily access the data in the arrays
//        NumPyArrayData<double> In_data(In);
//	NumPyArrayData<double> F_data(Filter);
//        np::ndarray Out = np::zeros(bp::make_tuple(Nz,Nx,Ny,J+1), np::dtype::get_builtin<double>());
//        NumPyArrayData<double> Out_data(Out);
//
//	double *pt_F = (double *) malloc(sizeof(double)*Lh);
//
//	for (int x=0; x < Lh; x++)
//	  {
//	    pt_F[x] = F_data(x);
//	  }
//
//
//	int k;
//
//#pragma omp parallel for shared(In_data,Out_data, pt_F, k)
//	for (k =0; k< Nz; k++){
//
//	    double *pt_In = (double *) malloc(sizeof(double)*Nx*Ny);
//	    double *pt_Out = (double *) malloc(sizeof(double)*Nx*Ny*(J+1));
//	    double *pt_WT = (double *) malloc(sizeof(double)*Nx*Ny*J);
//	    double *pt_CR = (double *) malloc(sizeof(double)*Nx*Ny);
//	    double *pt_Row = (double *) malloc(sizeof(double)*Ny);
//	    double *pt_Col = (double *) malloc(sizeof(double)*Nx);
//	    double *pt_Ro = (double *) malloc(sizeof(double)*Ny);
//	    double *pt_Co = (double *) malloc(sizeof(double)*Nx);
//
//	    for (int x=0; x < Nx; x++){
//	      for (int y =0; y< Ny; y++){
//		    pt_In[x + y*Nx] = In_data(k,x,y);
//		}
//	      }
//
//
//	    // Transform
//
//	    transform(pt_In, pt_WT, pt_CR, pt_Row, pt_Col, pt_Ro, pt_Co, pt_F);
//
//	    for (int x=0; x < Nx; x++) {
//	      for (int y =0; y< Ny; y++) {
//		for (int j =0; j< J; j++) {
//		   Out_data(k,x,y,j) = pt_WT[x + y*Nx + j*Nx*Ny];
//		}
//		Out_data(k,x,y,J) = pt_CR[x + y*Nx];
//	      }
//	    }
//
//	    free(pt_In);
//	    free(pt_Out);
//	    free(pt_WT);
//	    free(pt_CR);
//	    free(pt_Row);
//	    free(pt_Col);
//	    free(pt_Ro);
//	    free(pt_Co);
//
//	  }
//
//
//	free(pt_F);
//
//        return Out;
//    }
//
//
//    void reconstruct(double* CR, double* WT );
//    np::ndarray reconstruct_numpy(np::ndarray &In){
//
//        NumPyArrayData<double> In_data(In);
//        np::ndarray Out = np::zeros(bp::make_tuple(Nz,Nx,Ny), np::dtype::get_builtin<double>());
//        NumPyArrayData<double> Out_data(Out);
//
//        double *pt_WT = (double *) malloc (sizeof(double)*Nx*Ny*J);
//        double *pt_CR = (double *) malloc (sizeof(double)*Nx*Ny);
//	double *pt_Out = (double *) malloc (sizeof(double)*Nx*Ny);
//
//        for (int k =0; k< Nz; k++){
//
//	  for (int x=0; x < Nx; x++) {
//	      for (int y =0; y< Ny; y++) {
//		for (int j =0; j< J; j++) {
//		    pt_WT[x + y*Nx + j*Nx*Ny] = In_data(k,x,y,j);
//		}
//		pt_CR[x + y*Nx] = In_data(k,x,y,J);
//	      }
//	    }
//
//	    //Reconstruction
//
//	  reconstruct(pt_CR, pt_WT);
//
//	    for (int x=0; x < Nx; x++){
//	      for (int y =0; y< Ny; y++){
//		Out_data(k,x,y) = pt_CR[x + y*Nx];
//		}
//	      }
//
//	  }
//
//
//        return Out;
//    }
//
//    np::ndarray reconstruct_omp_numpy(np::ndarray &In){
//
//        NumPyArrayData<double> In_data(In);
//        np::ndarray Out = np::zeros(bp::make_tuple(Nz,Nx,Ny), np::dtype::get_builtin<double>());
//        NumPyArrayData<double> Out_data(Out);
//
//	int k;
//
//#pragma omp parallel for shared(In_data,Out_data)
//        for (k =0; k< Nz; k++){
//
//	  double *pt_WT = (double *) malloc (sizeof(double)*Nx*Ny*J);
//        double *pt_CR = (double *) malloc (sizeof(double)*Nx*Ny);
//	double *pt_Out = (double *) malloc (sizeof(double)*Nx*Ny);
//
//	  for (int x=0; x < Nx; x++) {
//	      for (int y =0; y< Ny; y++) {
//		for (int j =0; j< J; j++) {
//		    pt_WT[x + y*Nx + j*Nx*Ny] = In_data(k,x,y,j);
//		}
//		pt_CR[x + y*Nx] = In_data(k,x,y,J);
//	      }
//	    }
//
//	    //Reconstruction
//
//	  reconstruct(pt_CR, pt_WT);
//
//	    for (int x=0; x < Nx; x++){
//	      for (int y =0; y< Ny; y++){
//		Out_data(k,x,y) = pt_CR[x + y*Nx];
//		}
//	      }
//
//	    free(pt_CR);
//	    free(pt_WT);
//	    free(pt_Out);
//
//	  }
//
//
//        return Out;
//    }
//

private:

    int Nx, Ny, Nz, J,Lh,Cscale;
};

#endif // STARLET2D_H
