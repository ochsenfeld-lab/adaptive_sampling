#!/usr/bin/env python
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from adaptive_sampling import units

class TI:
    '''thermodynamic integration with the finite element method to get 2D PMFs
 
    args:
        equil_temp (double): temperature of free energy calculation in K
        outputfile (string): filename of output
    '''
    def __init__(self, 
        x: np.ndarray, 
        y: np.ndarray, 
        fx: np.ndarray, 
        fy: np.ndarray, 
        equil_temp: float=300.0, 
        outputfile: str='free_energy'
    ):
        
        out = open("FEM.out", "w")
        out.write("\n#######################################################\n")
        out.write("\tFEM integration of thermodynamic forces \n")
        out.write("#######################################################\n\n")
        out.write("Initialize spline functions.\n\n")
             
        self.RT      = units.kB_in_atomic * equil_temp * units.atomic_to_kJmol 
        self.outname = outputfile 
        
        # coordinates of bins
        minxi_1 = x.min()
        maxxi_1 = x.max()
        minxi_2 = y.min()
        maxxi_2 = y.max()
        
        dxi_2 = y[1]-y[0]
        self.x_bins = int((maxxi_2-minxi_2)/dxi_2 + 1)
        self.y_bins = int(x.shape[0]/self.x_bins)
        #x = x.reshape(self.y_bins, self.x_bins)
        #y = y.reshape(self.y_bins, self.x_bins)
        dxi_1 = x[1,0]-x[0,0]

        # control points
        self.dy = dxi_1 / 4
        self.dx = dxi_2 / 4
        self.minx = minxi_2
        self.maxx = maxxi_2
        self.miny = minxi_1
        self.maxy = maxxi_1

        self.x = np.arange(self.minx, self.maxx+self.dx, self.dx)
        if len(self.x) == 4*(self.x_bins-1)+2:
            self.x = self.x[:-1]

        self.y = np.arange(self.miny, self.maxy+self.dy, self.dy)
        if len(self.y) == 4*(self.y_bins-1)+2:
            self.y = self.y[:-1]

        self.lenx = len(self.x)
        self.leny = len(self.y)      
        self.xx, self.yy = np.meshgrid(self.x,self.y)
        
        # coefficient matrix
        self.alpha = np.full((int(self.x_bins*self.y_bins),), 0.0, dtype=np.float)
        out.write("%30s:\t%8d\n" % ("Number of coefficients", self.alpha.size))
        
        # linear interpolation of gradient to control points
        origD = [fx.reshape(self.y_bins, self.x_bins), fy.reshape(self.y_bins, self.x_bins)]
        D = [np.zeros(shape=(self.leny,self.lenx), dtype=np.float), np.zeros(shape=(self.leny,self.lenx), dtype=np.float)]
        for xy in range(2):
            D[xy][::4, ::4] = origD[xy]
            D[xy][::4,2::4] = 0.5*D[xy][::4,:-4:4] + 0.5*D[xy][::4,4::4] 
            D[xy][::4,1::4] = 0.5*D[xy][::4,2::4]  + 0.5*D[xy][::4,:-4:4]
            D[xy][::4,3::4] = 0.5*D[xy][::4,2::4]  + 0.5*D[xy][::4,4::4] 
            D[xy][2::4]     = 0.5*D[xy][:-4:4]     + 0.5*D[xy][4::4]
            D[xy][1::4]     = 0.5*D[xy][2::4]      + 0.5*D[xy][:-4:4]
            D[xy][3::4]     = 0.5*D[xy][2::4]      + 0.5*D[xy][4::4] 
        self.D = np.asarray(D) * units.atomic_to_kJmol
        self.D = self.D[:,:-1,:-1]
        out.write("%30s:\t%8d\n" % ("Elements in gradient matrix", self.D.size))
        
        # initialize B-spline functions 
        self.B = []
        self.gradB = []
        for center_y in x[:,0]:
            for center_x in y[0]:
                self.B.append(self.pyramid(self.x, self.y, dxi_2, dxi_1, center_x, center_y)) 
                self.gradB.append(self.gradPyr(self.x, self.y, dxi_2, dxi_1, center_x, center_y))
         
        self.B = np.asarray(self.B)
        self.gradB = np.asarray(self.gradB)
        self.gradB = self.gradB[:,:,:-1,:-1]

        out.write("%30s:\t%8d\n" % ("Elements in gradB matrix", self.gradB.size))
        out.close()

    #--------------------------------------------------------------------------------------
    def BFGS(self, maxiter=15000, ftol=1e-10, gtol=1e-5, error_function='rmsd'):
        '''BFGS minimization of error function

        args:
            maxiter		(int, 15000, maximum iterations of minimization)
            ftol        	(double, 1e-10, tolerance for error function)
            gtol		(double, 1e-5, tolerance for gradient)
            error_function	(string, rmsd, available is rmsd or power)

        returns:
            -
        '''        
        out = open("FEM.out", "a")
        if error_function == 'rmsd':
            out.write("\nerror = RMSD\n")
            self.error = self.error_rmsd
        else:
            out.write("\nerror = diff^2\n")
            self.error = self.error_power 
 
        options={
            'disp': True,
            'gtol': gtol,
            'maxiter': maxiter,
        }
        
        self.it = 0
        self.err0 = 0
        err = self.error(self.alpha) 
        
        out.write("\nStarting BFGS optimization of coefficents.\n")
        out.write("--------------------------------------------------------\n")
        out.write("%6s\t%14s\t%14s\t%14s\n" % ("Iter", "Error [kJ/mol]", "Change Error", "Wall Time [s]")) 
        out.write("--------------------------------------------------------\n")
        out.write("%6d\t%14.6f\t%14.6f\t%14.6f\n" % (self.it, err, 0.0, 0.0)) 
        out.close()

        self.start = time.perf_counter()
        result = opt.minimize(self.error, self.alpha, method='BFGS', tol=ftol, callback=self.BFGS_progress, options=options)
        
        self.alpha = result.x
        #self.get_F()

    #----------------------------------------------------------------------------------------
    def error_power(self, alpha):
        '''error function  
        '''
        a_gradB = np.zeros(shape=self.gradB.shape[1:])
        for ii, a in enumerate(alpha):
            a_gradB += a*self.gradB[ii]
        a_gradB_D = a_gradB - self.D
        return np.power(a_gradB_D, 2).sum()

    #--------------------------------------------------------------------------------------
    def error_rmsd(self, alpha):
        '''error function
        '''
        a_gradB = np.zeros(shape=self.gradB.shape[1:])
        for ii, a in enumerate(alpha):
            a_gradB += a*self.gradB[ii]
        a_gradB_D = a_gradB - self.D
        err = np.power(a_gradB_D, 2).sum(axis=0)
        err = err.mean()
        return np.sqrt(err)

    #--------------------------------------------------------------------------------------
    def BFGS_progress(self, alpha):
         '''callback function to display BFGS progress
         ''' 
         out = open("FEM.out", "a")
         self.it += 1
         err = self.error(alpha)
         out.write("%6d\t%14.6f\t%14.6f\t%14.6f\n" % (self.it, err, err-self.err0, time.perf_counter()-self.start))
         self.err0 = err

         self.alpha = alpha
         self.get_F()
         self.start = time.perf_counter()        
         
         out.close()

    #----------------------------------------------------------------------------------------
    def pyramid(self, x, y, dx, dy, cx, cy):
        '''pyramid function
        '''
        return_func = np.zeros(shape=(self.leny,self.lenx), dtype=np.float)
        for ii, val_y in enumerate(y):
            for jj, val_x in enumerate(x):
                if val_y >= (cy - dy) and val_y < cy:
                    if val_x >= (cx - dx) and val_x < cx:
                        return_func[ii, jj] = ((val_y - cy)/dy + 1)*((val_x - cx)/dx + 1)
                    if val_x < (cx + dx) and val_x >= cx:
                        return_func[ii, jj] = ((val_y - cy)/dy + 1)*((cx - val_x)/dx + 1)
                if val_y < (cy + dy) and val_y >= cy:
                    if val_x >= (cx - dx) and val_x < cx:
                        return_func[ii, jj] = ((cy - val_y)/dy + 1)*((val_x - cx)/dx + 1)
                    if val_x < (cx + dx) and val_x >= cx:
                        return_func[ii, jj] = ((cy - val_y)/dy + 1)*((cx - val_x)/dx + 1)
        #
        return return_func

    #----------------------------------------------------------------------------------------
    def gradPyr(self, x, y, dx, dy, cx, cy):
        '''gradients of pyramid function
        '''
        deriv_x = np.zeros(shape=(len(y),len(x)), dtype=np.float)
        deriv_y = np.zeros(shape=(len(y),len(x)), dtype=np.float)
        for ii, val_y in enumerate(y):
            for jj, val_x in enumerate(x):
                if val_y >= (cy - dy) and val_y < cy:
                    if val_x >= (cx - dx) and val_x < cx:
                        deriv_x[ii, jj] = ((val_y - cy)/dy + 1)*(1.0/dx)
                        deriv_y[ii, jj] = (1.0/dy)*((val_x - cx)/dx + 1)
                    if val_x < (cx + dx) and val_x >= cx:
                        deriv_x[ii, jj] = ((val_y - cy)/dy + 1)*(-1.0/dx)
                        deriv_y[ii, jj] = (1.0/dy)*((cx - val_x)/dx + 1)
                if val_y < (cy + dy) and val_y >= cy:
                    if val_x >= (cx - dx) and val_x < cx:
                        deriv_x[ii, jj] = ((cy - val_y)/dy + 1)*(1.0/dx)
                        deriv_y[ii, jj] = (-1.0/dy)*((val_x - cx)/dx + 1)
                    if val_x < (cx + dx) and val_x >= cx:
                        deriv_x[ii, jj] = ((cy - val_y)/dy + 1)*(-1.0/dx)
                        deriv_y[ii, jj] = (-1.0/dy)*((cx - val_x)/dx + 1)
        #
        return [deriv_x, deriv_y]

    #--------------------------------------------------------------------------------------
    def get_F(self):
        '''get free energy from optimized coefficients
        ''' 
        F_surface = np.zeros(shape=(self.leny,self.lenx), dtype=np.float64)
        fitted_grad_x = np.zeros(shape=(self.leny-1,self.lenx-1), dtype=np.float64)
        fitted_grad_y = np.zeros(shape=(self.leny-1,self.lenx-1), dtype=np.float64)

        for ii, a in enumerate(self.alpha):
            F_surface     += a*self.B[ii]
            fitted_grad_x += a*self.gradB[ii][0]
            fitted_grad_y += a*self.gradB[ii][1]
    
        prob_surface = np.exp(-F_surface/self.RT) 
        prob_surface /= prob_surface.sum()*self.dx*self.dy
        F_surface = -self.RT*np.log(prob_surface)       
 
        estim_err_x = np.abs(fitted_grad_x - self.D[0]) 
        estim_err_y = np.abs(fitted_grad_y - self.D[1]) 
         
        self.write_output(F_surface, prob_surface, estim_err_x, estim_err_y) 
        self.plot_F(F_surface, estim_err_x, estim_err_y)

    # -----------------------------------------------------------------------------------------------------
    def write_output(self, F_surface, prob_surface, estim_err_x, estim_err_y):
        '''write output of free energy calculations
        '''
        out = open(f"%s.dat" % (self.outname), "w")

        head = ("Xi1", "Xi1", "error x", "error y", "probability", "free energy [kJ/mol]")
        out.write("%14s\t%14s\t%14s\t%14s\t%14s\t%14s\n" % head)
        for i in range(self.leny-1):
            for j in range(self.lenx-1):
                row = (self.y[i], self.x[j], estim_err_x[i,j], estim_err_y[i,j], prob_surface[i,j], F_surface[i,j])
                out.write("%14.6f\t%14.6f\t%14.6f\t%14.6f\t%14.10f\t%14.10f\n" % row)

        out.close()
	
    #--------------------------------------------------------------------------------------
    def plot_F(self, A_surface, estim_err_x, estim_err_y):
        '''plot free energy surface 
        '''
        # Plotting
        plt.rcParams["figure.figsize"] = [8,14]
        fig, axs = plt.subplots(3)
        #
        im0 = axs[0].imshow(estim_err_x, origin='lower', extent=[self.xx.min(), self.xx.max(), self.yy.min(), self.yy.max()], zorder=1, interpolation='None', cmap='viridis', aspect='auto')
        cb0 = plt.colorbar(im0, ax=axs[0], fraction=0.044)
        cb0.outline.set_linewidth(2.5)
        cb0.ax.set_ylabel('Diff Gradient', fontsize=18)
        cb0.ax.tick_params(length=4,width=2.5,labelsize=18, pad=10, direction='in')
        #
        im1 = axs[1].imshow(estim_err_y, origin='lower', extent=[self.xx.min(), self.xx.max(), self.yy.min(), self.yy.max()], zorder=1, interpolation='None', cmap='viridis', aspect='auto')
        cb1 = plt.colorbar(im1, ax=axs[1], fraction=0.044)
        cb1.outline.set_linewidth(2.5)
        cb1.ax.set_ylabel('Diff Gradient', fontsize=18)
        cb1.ax.tick_params(length=4,width=2.5,labelsize=18, pad=10, direction='in')
        #
        CS = axs[2].contour(self.xx, self.yy, A_surface, colors='black', zorder=3)
        plt.clabel(CS, CS.levels, inline='true', fontsize=10, fmt="%5.3f")
        im2 = axs[2].imshow(A_surface, origin='lower', extent=[self.xx.min(), self.xx.max(), self.yy.min(), self.yy.max()], zorder=1, interpolation='None', cmap='viridis', aspect='auto')
        cb2 = plt.colorbar(im2, ax=axs[2], fraction=0.044)
        cb2.outline.set_linewidth(2.5)
        cb2.ax.set_ylabel('Free Energy', fontsize=18)
        cb2.ax.tick_params(length=4,width=2.5,labelsize=18, pad=10, direction='in')
        #
        for ax in axs:
            ax.set_ylim([self.miny, self.maxy])
            ax.set_xlim([self.minx, self.maxx])
            ax.set_xlabel(r'$\xi_1$', fontsize=20)
            ax.set_ylabel(r'$\xi_2$', fontsize=20)
            ax.spines['bottom'].set_linewidth('3')
            ax.spines['top'].set_linewidth('3')
            ax.spines['left'].set_linewidth('3')
            ax.spines['right'].set_linewidth('3')
            ax.tick_params(axis='y',length=6,width=3,labelsize=20, pad=10, direction='in')
            ax.tick_params(axis='x',length=6,width=3,labelsize=20, pad=10, direction='in')
        #
        plt.tight_layout()
        plt.savefig(f"%s.png" % (self.outname), dpi=400)
        plt.close()
