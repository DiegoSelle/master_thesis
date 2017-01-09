import numpy as np
import numpy.random as random
from libgp cimport EnvModel
cimport cython
from libcpp.string cimport string
import time

def test():
    cdef int n = 300
    cdef int npred = 3000
    cdef int niter = 30
    cdef dict kwargs = {'verbose':True, 'gp_opt_python':True, 
                        'gp_opt_iter':10, 
                        'gp_hyper_bounds':((-7,7),(-7,7),(-7,7),(-7,7),(-7,7),(-7,7))}

    x = 1000*random.ranf((n,4))
    y = np.ndarray((3,n),dtype=np.double)
    y[0] = np.sin(x[:,0])*np.sin(x[:,1])/(x[:,0]*x[:,1])+np.random.normal(0,0.1,n)
    y[1] = np.sin(x[:,1])*np.sin(x[:,2])/(x[:,1]*x[:,2])+np.random.normal(0,0.1,n)
    y[2] = np.sin(x[:,2])*np.sin(x[:,3])/(x[:,2]*x[:,3])+np.random.normal(0,0.1,n)
    ypred = np.zeros((3,2,npred))
    cdef EnvModel m = EnvModel("CovSum ( CovSEard, CovNoise)",kwargs)

    cdef double [:,::1] cx = x
    cdef double [:,:] cy = y
    cdef double [:,::1] cxpred = cx
    cdef double [:,:,::1] cypred = ypred

    print("With 1000 2 dimensional points ")

    begin = time.clock()
    m.update_wind(cx,cy)
    end = time.clock()
    print("Added points in {} seconds".format(end-begin))
    
    begin = time.clock()
    m.optimize(kwargs)
    end = time.clock()
    print("Optimised in {} seconds".format(end-begin))
    

    begin = time.clock()
    for _ in range(niter):
        cxpred = 1000*random.ranf((npred,4))
        m.predict_wind(cxpred,cypred)
    end = time.clock()
    print("Predicted points in {} seconds".format(end-begin))

if __name__ == "__main__":
    test()
