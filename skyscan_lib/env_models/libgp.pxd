from libcpp.string cimport string
from libcpp cimport bool
from libcpp.vector cimport vector
cimport cython

ctypedef GaussianProcess* GaussianProcess_ptr
ctypedef CovarianceFunction* CovarianceFunction_ptr

cdef extern from "libgp/gp.h" namespace "libgp":
    cdef cppclass GaussianProcess:
        GaussianProcess(size_t input_dim, string covf_def) except +
        GaussianProcess (const char * filename) except+
        void write(const char * filename)
        double f(const double * x)
        double var(const double * x)
        void predict(const double * x, double *f, double *var)
        void add_pattern(const double * x, double y)
        void get_pattern(size_t i, double *x, double *y)
        void remove_pattern(size_t i)
        bool set_y(size_t i, double y)
        size_t get_sampleset_size()
        void clear_sampleset()
        CovarianceFunction & covf()
        size_t get_input_dim()
        double log_likelihood()
        VectorXd log_likelihood_gradient()
        MatrixXd K()
        MatrixXd K_inv()

#NOT WORKING
#cdef extern from "libgp/gp_sparse.h" namespace "libgp":
#    cdef cppclass SparseGaussianProcess(GaussianProcess):
#        SparseGaussianProcess(size_t input_dim, string covf_def) except+
#        SparseGaussianProcess(const char * filename) except+
#        void compute()


cdef extern from "libgp/cov.h" namespace "libgp":
    cdef cppclass CovarianceFunction:
        CovarianceFunction() except +
        void set_loghyper(const double *)
        VectorXd get_loghyper()
        size_t get_param_dim()
        size_t get_input_dim()
        double get(VectorXd & x1, VectorXd & x2)

cdef extern from "libgp/rprop.h" namespace "libgp":
    cdef cppclass RProp:
        RProp()
        void init(double eps_stop, 
             double Delta0, 
             double Deltamin, double Deltamax, 
             double etaminus, double etaplus)
        void init()
        void maximize(GaussianProcess* gp, size_t n, bool verbose)
        void maximize(GaussianProcess* gp)


cdef extern from "Eigen/Dense" namespace "Eigen":
    cdef cppclass VectorXd:
        VectorXd()
        VectorXd(size_t size)
        @staticmethod
        const VectorXd & Zero(size_t size)
        size_t size()
        double * data()
        double & operator[](size_t idx)
        VectorXd & operator=(const VectorXd & other)

    cdef cppclass MatrixXd:
        MatrixXd()
        MatrixXd(size_t s1, size_t s2)
        @staticmethod
        const MatrixXd & Zero(size_t s1, size_t s2)
        size_t size()
        size_t rows()
        size_t cols()
        double * data()
        double & operator()(size_t, idx_rows, size_t idx_cols)
        MatrixXd & operator=(const MatrixXd & other)

    # Will only work for types with Scalar type double
    # How to import Scalar type from Map declaration ?
    # Not sure it is possible with Cython 
    # In the meantime this hack works fine
    cdef cppclass Map[T]:
        Map(double * ptr, size_t size)
        Map(double * ptr, size_t rows, size_t cols)

cdef class GPWrapper:
    cdef GaussianProcess_ptr gp
    cdef wrap(self,GaussianProcess_ptr gp)

cdef class CovfWrapper:
    cdef CovarianceFunction_ptr covf
    cdef wrap(self, CovarianceFunction_ptr covf)

@cython.final
cdef class GPModel:
    cdef size_t ndim
    cdef size_t nvar
    cdef size_t discr_dim
    cdef double discr_tol
    cdef vector[GaussianProcess_ptr] gps
    cdef RProp *rp
    cpdef update1(self, double[:,::1] x, double [:] y, int var_idx)
    cpdef update(self, double[:,::1] x, double [:,:] y)
    cpdef predict(self, double[:,::1] x, double [:,:,:] out)
    cpdef optimize(self, kwargs=?)
    cpdef set_params(self, double[:,::1] values)
    cpdef get_params(self, double[:,::1] values = ?)
    cpdef get_params_dim(self)

@cython.final
cdef class EnvModel:
    cdef GPModel wmodel
    cdef bool params_set
    cdef dict kwargs
    cpdef predict_wind(self, double[:,::1] x, double[:,:,:] out = ?)
    cpdef update_wind(self, double[:, ::1] x, double[:,:] wind)
    cpdef optimize(self, dict kwargs=?)
    cpdef get_params(self)
    cpdef set_params(self, double[:,::1] values)
