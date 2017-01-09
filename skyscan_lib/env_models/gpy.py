from __future__ import absolute_import, print_function, division
import time
import numpy as np
import multiprocessing as mp
import threading as thr
import GPy
import os


def timeprintit(f):
    class TpRecord:
        i = 1
        times = []

    if not hasattr(timeprintit, 'records'):
        timeprintit.records = dict()

    timeprintit.records[f.__name__] = TpRecord()

    def _timeprintit(*args, **kwargs):
        c1 = time.time()
        ret = f(*args, **kwargs)
        c2 = time.time()

        print("[{}][{}][{}] {} sec.".format(os.getpid(), f.__name__, timeprintit.records[f.__name__].i, (c2-c1)))

        timeprintit.records[f.__name__].i += 1
        return ret

    return _timeprintit

@timeprintit
def __opt(gp, args):
    gp.optimize_restarts(*args)


def _opt(gp, args):
    print("Optimize params ({} samples)".format(len(gp.X)))
    __opt(gp, args)
    return gp.optimizer_array


def _sync_opt(m):
    def _sync_opt_gp(opt):
        print("Sync opt")
        with m.gp_lock:
            m.apply_hparams(opt)
            m.opt_pending -= 1

    return _sync_opt_gp


class MapPredictor:

    GP = None
    X = None
    y = None
    gp_lock = None

    optimize_restarts = None
    optimize_every = None
    optimize_every_min = None
    optimize_last = None
    opt_pending = 0

    pool = None

    old_hparams = None
    eps_ch =None

    def __del__(self):
        self.stop()

    def stop(self):
        self.pool.close()
        self.pool.join()

    def __init__(self, dims, optimize_restarts=10, optimize_every=None, optimize_async=False,
                 max_samples=None, parameters_array=None):
        self.parameters_array = parameters_array
        self.optimize_restarts = optimize_restarts
        self.optimize_every = optimize_every
        self.optimize_every_min = optimize_every
        self.optimize_async = optimize_async
        self.optimize_last = 0
        self.eps_ch = 0.01

        self.gp_lock = thr.RLock()
        self.pool = mp.Pool(processes=1)

        dims = np.atleast_1d(dims).flatten()
        assert(len(dims) == 2)
        self.X = np.ndarray((0, dims[0]))
        self.y = np.ndarray((0, dims[1]))

        self.max_samples = max_samples
        if max_samples is None:
            self.max_samples = np.inf

    #@timeprintit
    def __call__(self, x):
        return self.predict(x)

    #@timeprintit
    def update_observations(self, X, y):
        assert(len(X) == len(y))
        with self.gp_lock:

            #print("X has shape {}, self {}".format(X.shape, self.X.shape))
            #print("y has shape {}, self {}".format(y.shape, self.y.shape))

            self.update_xy(X, y)

            if self.GP is None:
                self.GP = self._init_gp()

            self.GP.set_XY(self.X, self.y)

            if self.optimize_every is not None \
                    and len(self.X) - self.optimize_last >= self.optimize_every:
                self.optimize_hyperparameters(async=self.optimize_async)

    def optimize_hyperparameters(self, async=False):
        print("optimize_hyperparameters(async={})".format(async))

        with self.gp_lock:
            self.old_hparams = self.GP.param_array.copy()
            if not async:
                opt = _opt(self.GP.copy(), (self.optimize_restarts,))
                self.apply_hparams(opt)
                self.optimize_last = len(self.X)
                return

            if self.opt_pending > 0:
                print("Already pending")
                return

            self.opt_pending += 1
            self.optimize_last = len(self.X)
            self.pool.apply_async(_opt, args=(self.GP, (self.optimize_restarts,)), callback=_sync_opt(self))

    def apply_hparams(self, opt):
        with self.gp_lock:
            self.GP.optimizer_array = opt
            new_hparams = self.GP.param_array.copy()

            chrate = np.abs(1 - new_hparams/self.old_hparams)
            maxch = chrate.max()

            if self.optimize_every is not None \
                    and maxch < self.eps_ch:
                    self.optimize_every *= 2
            elif self.optimize_every >= 2*self.optimize_every_min:
                self.optimize_every /= 2

            print("Optimize every {}".format(self.optimize_every))


    def _init_gp(self):
        #kernel = GPy.kern.Matern32(input_dim=self.X[0].shape[0], ARD=True)
        kernel = GPy.kern.RBF(input_dim=self.X[0].shape[0], ARD=True)
        #kernel += GPy.kern.White(input_dim=self.X[0].shape[0])
        gp = GPy.models.GPRegression(self.X, self.y, kernel)
        gp.preferred_optimizer = 'bfgs'

        if self.parameters_array is not None:
            gp.param_array[:] = self.parameters_array

        return gp

    #@timeprintit
    def predict(self, x):
        with self.gp_lock:
            assert(self.GP is not None)

            y_pred, var = self.GP.predict(x)
        return y_pred, var

    def update_xy(self, X, y):

        self.X = np.concatenate((self.X, X))
        self.y = np.concatenate((self.y, y))

        if self.GP is None:
            return
        else:
            ker = self.GP.kern

        ker_min_val = -ker.K(X[0].reshape(1, -1), X[0].reshape(1, -1)).squeeze()

        lenX = len(self.X)
        mdist = np.full(lenX, np.inf)
        nz = 0

        for i in range(lenX-1):
            dist = -ker.K(self.X[i+1:], self.X[i].reshape(1, -1)).squeeze()
            dmin = dist.min()
            mdist[i] = dmin
            if dmin == ker_min_val:
                nz += 1

        ndel = nz + max(0, lenX-nz-self.max_samples)

        if ndel > 0:
            asort = np.argsort(mdist)
            del_rows = asort[0:ndel]
            self.X = np.delete(self.X, del_rows, axis=0)
            self.y = np.delete(self.y, del_rows, axis=0)
            self.optimize_last -= ndel
        print("{} samples in model (deleted {})".format(self.X.shape[0], ndel))

    def get_gp_parameters(self):
        return self.GP.param_array.copy()

    def set_gp_parameters(self, parameters):
        if parameters is None:
            return

        self.parameters_array = np.array(parameters)
        if self.GP is not None:
            self.GP.param_array[:] = parameters

    @property
    def covf(self):
        return self.GP.kern.K


    @property
    def gaussian_noise_variance(self):
        return self.GP['Gaussian_noise.variance'][0]

    @property
    def K(self):
        return self.GP.posterior._K

    @property
    def K_inv(self):
        return self.GP.posterior.woodbury_inv

class EnvModel:
    models = None

    def __init__(self,
                 optimize_restarts=10, optimize_every=1, optimize_async=False,
                 max_samples=300):
        n_dim = 4
        n_var = 3
        mp_args = ((n_dim, 1),
                   optimize_restarts, optimize_every, optimize_async,
                   max_samples)

        self.models = np.array([MapPredictor(*mp_args) for _ in range(n_var)])

    def predict_wind(self, X):
        # (component, value and error, coord)
        w_pred = np.array([w.predict(X) for w in self.models])

        return w_pred

        # (value and error, coord, component
        # return w_pred.swapaxes(0, 1).swapaxes(1, 2)

    def update_wind(self, X, w):
        X = np.atleast_2d(X)
        w = np.atleast_2d(w)
        w_t = w
        for i in range(len(self.models)):
            self.models[i]\
                .update_observations(X, w_t[i].reshape(len(w_t[i]), -1))

    def get_params(self):
        return np.array([m.get_gp_parameters() for m in self.models])

    def set_params(self, parameters):
        for (m, p) in zip(self.models, parameters):
            m.set_gp_parameters(p)
            m.optimize_every = None


def mappredictor_test():

    freq = 1.0/5
    Tmax = 20
    x = np.arange(start=0, stop=Tmax, step=1./(20*freq))
    noise = np.random.normal(0, 0.3, len(x))
    #noise = 0
    f = np.sin(2 * np.pi * freq * x) + noise
    from matplotlib import pyplot as plt

    for k in range(1):
        print("{}-th iteration".format(k))

        pred = None
        pred = MapPredictor((1, 1), optimize_restarts=10, optimize_every=50, var = None)


        for x_i, f_i in zip(x, f):
            pred.update_observations([[x_i]], [[f_i]])
            #time.sleep(0.1)

        pred.optimize_hyperparameters(process=True)
        pred.stop()
        p = pred.GP.plot()

        Xt = np.arange(start=0, stop=Tmax, step=1./(100*freq))
        plt.plot(Xt, np.sin(2 * np.pi * freq * Xt), 'r--')

        plt.ylim(-1.5, 1.5)

    plt.show(block=True)
    #plt.plot(x,f,'o')
    #plt.show(block=True)

    print("Stop")
    #input("Press Enter to continue...")


if __name__ == '__main__':
    mappredictor_test()




