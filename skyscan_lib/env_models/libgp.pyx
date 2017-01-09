import numpy as np 
from libc.stdlib cimport malloc, free
from scipy.optimize import basinhopping




cdef np_vector(VectorXd base):
    return np.array(<double[:base.size()]>base.data())

cdef np_matrix(MatrixXd base):
    return np.array(<double[:base.rows(),:base.cols():1]>base.data())




cdef class GPModelOptimizer:
    cdef GaussianProcess_ptr model

    cpdef f(self, double[::1] x):
        self.model.covf().set_loghyper(&x[0])
        cdef double lik = self.model.log_likelihood()
        return -lik

    cpdef f_grad(self, x):
        cdef VectorXd ret = self.model.log_likelihood_gradient()
        return -np_vector(ret)

    cdef optimize(self, GaussianProcess_ptr model, kwargs):
        self.model = model
        bounds=kwargs['gp_hyper_bounds']
        minimizer_kwargs = {"method":"L-BFGS-B", "jac":self.f_grad, "bounds":bounds}
        niter = kwargs['gp_opt_iter']
        x0 = model.covf().get_loghyper()
        ret = basinhopping(self.f, np_vector(x0), minimizer_kwargs=minimizer_kwargs,niter=niter)
        print("global maximium: x_star = {}, f_star = {}".format(ret.x, -ret.fun))
        cdef double[:] x_star = ret.x
        model.covf().set_loghyper(&x_star[0])


cdef class CovfWrapper:
    cdef wrap(self, CovarianceFunction_ptr covf):
        self.covf = covf
        return self

    def __call__(self,double[:,::1] x1,double[:,::1] x2):
        cdef double[:,:] out = np.ndarray((x1.shape[0],x2.shape[0]),dtype=np.double)
        cdef size_t ndim = x1.shape[1]
        cdef size_t n1 = x1.shape[0]
        cdef size_t n2 = x2.shape[0]
        cdef size_t i,j
        cdef VectorXd vx1,vx2
        for i in range(n1):
            for j in range(n2):
                vx1 = <VectorXd> Map[VectorXd](&x1[i,0],ndim)
                vx2 = <VectorXd> Map[VectorXd](&x2[j,0],ndim)
                out[i,j] = self.covf.get(vx1,vx2)
        return out.base


cdef class GPWrapper:

    cdef wrap(self, GaussianProcess_ptr gp):
        self.gp = gp
        return self

    @property
    def K(self):
        return np_matrix(self.gp.K())

    @property
    def K_inv(self):
        return np_matrix(self.gp.K_inv())


    @property
    def X(self):
        cdef GaussianProcess_ptr gp = self.gp
        cdef size_t ss_size = gp.get_sampleset_size()
        cdef double y = np.nan
        out =  np.nan*np.ndarray((ss_size,gp.get_input_dim()))
        cdef double[:,::1] X = out
        for i in range(ss_size):
            gp.get_pattern(i,&X[i,0],&y)
        
        return out
    
    @property
    def y(self):
        cdef GaussianProcess_ptr gp = self.gp
        cdef double * x = <double *> malloc(gp.get_input_dim()*sizeof(double))      
        cdef size_t ss_size = gp.get_sampleset_size()
        out = np.nan*np.ndarray(ss_size)
        cdef double[:] y = out
        for i in range(ss_size):
            gp.get_pattern(i,x,&y[i])
        
        free(x)
        return out

    @property 
    def covf(self):
        return CovfWrapper().wrap(&self.gp.covf())

    @property
    def gaussian_noise_variance(self):
        return np.exp(2.*self.gp.covf().get_loghyper()[self.gp.covf().get_param_dim()-1])


cdef class GPModel:

    def __cinit__(self,int ndim, int nvar, string kernel_string, 
                  int discr_dim = 0, double discr_tol = 1e-1):
        self.ndim = ndim
        self.nvar = nvar
        self.discr_dim = discr_dim
        self.discr_tol = discr_tol
        self.gps=vector[GaussianProcess_ptr](nvar)
        for i in range(nvar):
            self.gps[i] = new GaussianProcess(ndim,kernel_string)
        self.rp = new RProp()

    def __dealloc__(self):
        for i in range(self.nvar):
            del self.gps[i]
        del self.rp

    cpdef update1(self, double[:,::1] x, double [:] y, int var_idx):
        cdef GaussianProcess_ptr gp
        gp = self.gps[var_idx]
        for i in range(x.shape[0]):
                gp.add_pattern(&x[i,0], y[i])
                
    cpdef update(self, double[:,::1] x, double [:,:] y):
        cdef GaussianProcess_ptr gp
        cdef VectorXd x1 = VectorXd.Zero(self.ndim)
        cdef VectorXd ref = VectorXd.Zero(self.ndim)
        cdef VectorXd tmp = VectorXd.Zero(self.ndim)
        cdef double y_dummy = 0
        cdef size_t n, idx
        cdef double k

        for ivar in range(self.nvar):
            gp = self.gps[ivar]
            
            # remove old stuff (at the beginning)
            ref.data()[self.discr_dim] = x[<int>x.base.shape[0]-1, self.discr_dim]
            n = gp.get_sampleset_size()
            idx=0
            while idx < n and self.discr_tol > 0:
                gp.get_pattern(idx, tmp.data(), &y_dummy)
                x1.data()[self.discr_dim] = tmp.data()[self.discr_dim]
                k = gp.covf().get(x1,ref)
                if k >= self.discr_tol or\
                   abs(ref.data()[self.discr_dim]- x1.data()[self.discr_dim]) < 60:
                    #print("x1={}, ref={} k={}".format(np_vector(x1), np_vector(ref), k))
                    break
                gp.remove_pattern(idx)
                n = n - 1

            for i in range(x.shape[0]):
                gp.add_pattern(&x[i,0], y[ivar,i])

    cpdef predict(self, double[:,::1] x, double [:,:,:] out):
        cdef GaussianProcess_ptr gp
        cdef double f = 0.
        cdef double var = 0.
        for ivar in range(self.nvar):
            gp = self.gps[ivar]
            for i in range(x.shape[0]):
                gp.predict(&x[i,0], &f, &var)
                out[ivar,0,i] = f
                out[ivar,1,i] = var
        
        return out

    cpdef optimize(self, kwargs=None):
        cdef GPModelOptimizer opt
        print("Optimizing model hyperparameters ...")

        if "gp_opt_python" not in kwargs:
            self.rp.init()
            n = kwargs['gp_opt_iter']
            verbose = kwargs['verbose']
            for i in range(self.nvar):
                self.rp.maximize(self.gps[i],n,verbose)
        else:            
            opt = GPModelOptimizer()
            for i in range(self.nvar):
                opt.optimize(self.gps[i],kwargs)

            del opt

    cpdef get_params(self, double[:,::1] values = None):
        cdef GaussianProcess_ptr gp
        cdef CovarianceFunction * covf
        cdef VectorXd vec
        cdef size_t pdim = self.get_params_dim()
        cdef size_t vdim = self.nvar

        if values is None:
            values = np.ndarray((vdim, pdim), dtype=np.double, order='C')

        for ivar in range(vdim):
            gp = self.gps[ivar]
            covf = &gp.covf()
            vec = covf.get_loghyper()
            for j in range(pdim):
                values[ivar,j] = vec[j]
        return values
    
    cpdef set_params(self, double[:,::1] values):
        cdef GaussianProcess_ptr gp
        cdef CovarianceFunction * covf

        for ivar in range(values.shape[0]):
            gp = self.gps[ivar]
            covf = &gp.covf()
            covf.set_loghyper(&values[ivar,0])
    

    cpdef get_params_dim(self):
        return self.gps[0].covf().get_param_dim()
    
    @property
    def X(self):
      return [m.X for m in self.models]
    
    @property
    def y(self):
       return [m.y for m in self.models]

    @property
    def models(self):
        return [GPWrapper().wrap(m) for m in self.gps]

cdef class EnvModel:
    def __cinit__(self, string kernel_string=string(b"CovSum(CovSEard, CovNoise)"), kwargs=None): 
                  
        self.wmodel = GPModel(4,3,kernel_string,
                              discr_tol=kwargs['gp_discr_tol'])
        self.params_set = False
        self.kwargs=kwargs

    def __dealloc__(self):
        pass

    cpdef predict_wind(self, double[:,::1] x, double [:,:,:] out = None):
        if out is None:
            out = np.nan*np.ndarray((3,2,x.shape[0]),dtype=np.double)
        return self.wmodel.predict(x,out).base

    cpdef update_wind(self, double[:, ::1] x, double[:,:] wind):
        self.wmodel.update(x,wind)
        if not self.params_set:
            self.optimize()

    cpdef optimize(self, dict kwargs=None):
        if kwargs is None:
            kwargs = self.kwargs
            print("Using default kwargs")
        self.wmodel.optimize(kwargs)

    cpdef get_params(self):
        cdef double[:,::1] params = self.wmodel.get_params()
        return params.base;

    cpdef set_params(self, double[:,::1] values):
        self.wmodel.set_params(values)
        self.params_set = True

    @property
    def X(self):
        return self.wmodel.X

    @property
    def y(self):
        return self.wmodel.y

    property wmodel:
        def __get__(self):
            return self.wmodel
        

