import GPy
#GPy.plotting.change_plotting_library('plotly')
import numpy as np
from skyscan_lib.sim.mesonh_atmosphere import MesoNHAtmosphere
#from mesonh_atm.mesonh_atmosphere import MesoNHAtmosphere
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import scipy.optimize as sciopt
#from cloud import cloud,sample_variogram
import cloud
from GPy.kern import RBF
from skyscan_lib.env_models.libgp import GPModel
import random
from scipy.stats import skew
import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable

class myRBF(RBF):
    def __init__(self, input_dim, variance=1., lengthscale=None, ARD=False, active_dims=None, name='myrbf', useGPU=False, inv_l=False,phi_active_dim = 2):
        super(myRBF, self).__init__(input_dim, variance, lengthscale, ARD, active_dims, name, useGPU, inv_l)

    @override
    def _unscaled_dist(self, X, X2=None):
        """
        Compute the Euclidean distance between each row of X and X2, or between
        each pair of rows of X if X2 is None.
        """
        #X, = self._slice_X(X)
       if X2 is None:
            Xsq = np.sum(np.square(X),1)
            r2 = -2.*tdot(X) + (Xsq[:,None] + Xsq[None,:])
            util.diag.view(r2)[:,]= 0. # force diagnoal to be zero: sometime numerically a little negative
            r2 = np.clip(r2, 0, np.inf)
            return np.sqrt(r2)
        else:
            #X2, = self._slice_X(X2)
            X1sq = np.sum(np.square(X),1)
            X2sq = np.sum(np.square(X2),1)
            r2 = -2.*np.dot(X, X2.T) + X1sq[:,None] + X2sq[None,:]
            r2 = np.clip(r2, 0, np.inf)
            return np.sqrt(r2)

infile = 'results/plots_and_images/polar_analysis/Radial_trend/trends_data/rtrend_global_median_updraft_norm.npy'
rtrend_global_median_updraft_norm = np.load(infile)

def mad(data, axis=None):
    return np.nanmedian(np.abs(data - np.nanmedian(data, axis)), axis)/0.6745

def polar_cloud_norm(points_span,lwc_cloud,COM_2D_lwc_tz,z_interested,origin_xy):

    # Function to interpolate points of cloud1
    interpolate_points_cloud = RegularGridInterpolator(points=points_span,values=lwc_cloud,bounds_error=False,fill_value=0)


    # t,z,phi,r ---> [t,z,x,y] for interpolation: t in seconds, z in km, r in km, phi in degrees, x and y in km
    # Steps for x,y,z are in 0.01 km and steps in r are in 0.005 m
    polar_cloud = np.ndarray((len(points_span[0]),len(points_span[1]),360,151,4))
    polar_cloud[:] = np.NAN
    # Polar representation normalized to 100 for each radius of cloud1
    polar_cloud_norm = np.ndarray((len(points_span[0]),len(points_span[1]),360,151,4))
    polar_cloud_norm[:] = np.NAN
    tr = points_span[0]
    zr = points_span[1]
    for t in range(len(points_span[0])):
        for z in z_interested:
            COM_lwc_plane = COM_2D_lwc_tz[t,z]
            time = tr[t]
            z_value = zr[z]
            for r in range(0,151):
                phi = 0
                for step_phi in range(0,360):
                    if step_phi!=0:
                        phi = phi + 2*np.pi/360
                    x = 0.01*r/2* np.cos(phi)
                    y = 0.01*r/2* np.sin(phi)
                    polar_cloud[t,z,step_phi,r] = [time,z_value,x,y]

            polar_cloud[t,z,:,:,2] = 0.005 + 0.01*(COM_lwc_plane[0] + origin_xy[0]) + polar_cloud[t,z,:,:,2]
            polar_cloud[t,z,:,:,3] = 0.005 + 0.01*(COM_lwc_plane[1] + origin_xy[1]) + polar_cloud[t,z,:,:,3]


            polar_plane = polar_cloud[t,z]
            #zwind_polar_plane = atm.get_points(polar_plane,'WT','linear')
            #zwind_polar_plane=zwind_polar_plane.reshape(1,1,360,150)
            lwc_polar_plane = interpolate_points_cloud(polar_plane,"nearest")
            lwc_polar_plane = lwc_polar_plane.reshape(1,1,360,151)

            ####### Normalization to 100 ########
            # Basis for Normalization
            r_basis = 100
            for step_phi in range(0,360):
                try:
                    max_r = np.max(np.argwhere(lwc_polar_plane[0,0,step_phi]))
                    stepr = max_r/r_basis
                    for r_normal in range(0,151):
                        phi = step_phi*2*np.pi/360
                        x = 0.01*r_normal*stepr*np.cos(phi)/2
                        y = 0.01*r_normal*stepr*np.sin(phi)/2
                        polar_cloud_norm[t,z,step_phi,r_normal] = [time,z_value,x,y]
                except ValueError:
                    print('No values inside cloud at time = {} height = {},direction phi={}'.format(t,z,step_phi))

            polar_cloud_norm[t,z,:,:,2] = 0.005 + 0.01*(COM_lwc_plane[0] + origin_xy[0]) + polar_cloud_norm[t,z,:,:,2]
            polar_cloud_norm[t,z,:,:,3] = 0.005 + 0.01*(COM_lwc_plane[1] + origin_xy[1]) + polar_cloud_norm[t,z,:,:,3]

    return polar_cloud, polar_cloud_norm

##########################################################################################################
################ Function to normalize list of points, for example training inputs and test inputs #######
##########################################################################################################

def normalize(lwc_cloud_polar,points_cartesian,COM_2D_lwc_tz,tstart,zstart,origin_xy):
    # assuming in lwc_cloud_polar ndarray((tr,zr,360,151)) with lwc values and rstep = 0.005 km
    # and coordinates[t,z,phi,r],resolution of r in 5m and phi in degrees
    # tstart(seconds) and zstart(0.01m) are the time and height coordinates where the data of lwc_cloud_polar started
    # For indexation in z, it will be assumed that the resolution is 0.01 km(10m)
    # assuming points_cartesian as list of points [t,z,x,y] n by 4 and domain does not exceed lwc_cloud_polar
    points_polar_normal = np.ndarray((0,4))
    for point in points_cartesian:
        point_polar_normal = np.array([])
        trel = int(point[0] - tstart)
        zrel = int(np.round((point[1] - zstart)/0.01))
        COMx = COM_2D_lwc_tz[trel,zrel,0]
        COMy = COM_2D_lwc_tz[trel,zrel,1]
        dx = point[2] - (COMx + origin_xy[0])*0.01
        dy = point[3] - (COMy + origin_xy[1])*0.01
        phi = np.arctan2(dy,dx)/np.pi*180
        if phi < 0:
            phi = 360 + phi
        r = np.sqrt(dx**2 + dy**2)

        max_rs = np.array([])
        for step_phi in range(0,361):
            mod_phi = int(np.mod(step_phi,360))
            max_r = np.max(np.argwhere(lwc_cloud_polar[trel,zrel,mod_phi]))*0.005
            max_rs = np.append(max_rs,max_r)
            #print(max_rs.shape)
        interpolate_max_r= RegularGridInterpolator(points=np.arange(0,361).reshape(1,361),values=max_rs,bounds_error=False,fill_value=0)
        #print(phi)
        max_r = interpolate_max_r(np.array([phi]),'linear')
        r_norm = 100*r/max_r
        point_polar_normal = np.array([point[0],point[1],phi,r_norm])
        points_polar_normal = np.vstack((points_polar_normal,point_polar_normal))
    return points_polar_normal


#Old Data without advection
path = "/net/skyscanner/volume1/data/mesoNH/ARM_OneHour3600files_No_Horizontal_Wind/"
mfiles = [path+"U0K10.1.min{:02d}.{:03d}_diaKCL.nc".format(minute, second)
          for minute in range(1, 60)
          for second in range(1, 61)]
mtstep = 1
atm = MesoNHAtmosphere(mfiles, 1)

infile = 'results/plots_and_images/polar_analysis/Radial_trend/trends_data/rtrend_global_median_updraft_norm.npy'
rtrend_global_median_updraft_norm = np.load(infile)

lwc_data1=atm.data['RCT'][449:540,95:111,60:200,100:250]
zwind_data1=atm.data['WT'][449:540,95:111,60:200,100:250]
ids1,counter1,clouds1=cloud.cloud_segmentation(lwc_data1)
#ids1,counter1,clouds1=cloud_segmentation(lwc_data1)
clouds1=list(set(clouds1.values()))
length_point_clds = np.ndarray((0,1))
for each_cloud in clouds1:
    print(len(each_cloud.points))
    temp = len(each_cloud.points)
    length_point_clds = np.vstack((length_point_clds,temp))

cloud1 = clouds1[np.argmax(length_point_clds)]

del clouds1

cloud1.calculate_attributes(lwc_data1,zwind_data1)
lwc_cloud1 = np.zeros(lwc_data1.shape)
for point in cloud1.points:
    lwc_cloud1[point] = 1

#lwc_cloud1[lwc_cloud1==0]= 1e-9

xr =np.arange(0.005 + 60*0.01, 0.005 + 200*0.01,0.01)
yr= np.arange(0.005 + 100*0.01, 0.005 + 250*0.01,0.01)
all_Zs=atm.data["VLEV"][:,0,0]
zr = all_Zs[95:111]
#zr = np.arange(1.185,1.185 + 15*0.01,0.01)
tr = np.arange(449,540)
origin_xy = [60,100]
zspan = np.arange(0,16)
points_span = (tr,zr,xr,yr)

########## Important: Interpolator needs at least to coordinates for each
interpolate_points_cloud1 = RegularGridInterpolator(points=(tr,zr,xr,yr),values=lwc_cloud1,bounds_error=False,fill_value=0)

polar_cloud1,polar_cloud1_norm = polar_cloud_norm(points_span,lwc_cloud1,cloud1.COM_2D_lwc_tz,zspan,origin_xy)

del polar_cloud1_norm

lwc_cloud1_polar = interpolate_points_cloud1(polar_cloud1,'nearest')

M = np.array(np.meshgrid(tr,zr, xr, yr, indexing='ij'))
grid = np.stack(M, axis=-1)

del M
#zwind  = atm.get_points(grid, var='WT', method='nearest')
#lwc = atm.get_points(grid, var='WT', method='nearest')
plt.figure()
plt.imshow(lwc_cloud1[4,0].T,origin='lower',extent=[xr[0], xr[-1], yr[0], yr[-1]])
plt.title("Points inside of cloud, z=1.185 km,t=453s")
plt.xlim(xr[0], xr[-1])
plt.ylim(yr[0], yr[-1])


#####################################################################################
################################ GP test CS #########################################
#####################################################################################
all_rmse_static_cs = np.nan*np.ndarray((8,20,5,11))
all_rmse_dyn_cs = np.nan*np.ndarray((8,20,5,11))


all_r2_static_cs = np.nan*np.ndarray((8,20,5,11))
all_r2_dyn_cs = np.nan*np.ndarray((8,20,5,11))


all_std_pred_static_cs = np.nan*np.ndarray((8,20,5,9))
all_std_pred_dyn_cs = np.nan*np.ndarray((8,20,5,9))


all_std_pred_soundness_static_cs = np.nan*np.ndarray((8,20,5,9,3))
all_std_pred_soundness_dyn_cs = np.nan*np.ndarray((8,20,5,9,3))

all_len_train_data_static_cs = np.nan*np.ndarray(8)
all_len_train_data_dyn_cs = np.nan*np.ndarray(8)

all_hypers_static_cs = np.nan*np.ndarray((8,20,5,9,6)) ##### first 5 Models have hypers in log scale
all_hypers_dyn_cs = np.nan*np.ndarray((8,20,5,9,6))

all_error_stats_static_cs = np.nan*np.ndarray((8,20,5,11,4))
all_error_stats_dyn_cs = np.nan*np.ndarray((8,20,5,11,4))

all_vars_static_cs = np.nan*np.ndarray(8)
all_vars_dyn_cs = np.nan*np.ndarray(8)

zind_rel = np.arange(0,len(zspan),2)
for zi in np.arange(0,len(zind_rel)):
    ##########################################################################
    ############################## static test GP ############################
    ##########################################################################
    tstart_circle = 449
    tend_circle = 526
    radius_circle = 0.09

    drone1_circle = np.ndarray((0,4))
    xcenter = 1.1
    ycenter = 2.1
    for t in np.arange(tstart_circle,tend_circle):
        trel = t - tstart_circle
        xtemp = xcenter + radius_circle*np.cos(15/90*trel)
        ytemp = ycenter + radius_circle*np.sin(15/90*trel)
        temp = np.array([453,zr[zind_rel[zi]],xtemp,ytemp])
        incloud1 = interpolate_points_cloud1(temp,'nearest')
        if incloud1 == 1.0:
            drone1_circle = np.vstack((drone1_circle,temp))

    drone2_circle = np.ndarray((0,4))
    xcenter = 1.4
    ycenter = 2.06
    for t in np.arange(tstart_circle,tend_circle):
        trel = t - tstart_circle
        xtemp = xcenter + radius_circle*np.cos(15/90*trel)
        ytemp = ycenter + radius_circle*np.sin(15/90*trel)
        temp = np.array([453,zr[zind_rel[zi]],xtemp,ytemp])
        incloud1 = interpolate_points_cloud1(temp,'nearest')
        if incloud1 == 1.0:
            drone2_circle = np.vstack((drone2_circle,temp))

    drone3_circle = np.ndarray((0,4))
    xcenter = 1.23
    ycenter = 1.9
    for t in np.arange(tstart_circle,tend_circle):
        trel = t - tstart_circle
        xtemp = xcenter + radius_circle*np.cos(15/90*trel)
        ytemp = ycenter + radius_circle*np.sin(15/90*trel)
        temp = np.array([453,zr[zind_rel[zi]],xtemp,ytemp])
        incloud1 = interpolate_points_cloud1(temp,'nearest')
        if incloud1 == 1.0:
            drone3_circle = np.vstack((drone3_circle,temp))

    drone4_circle = np.ndarray((0,4))
    xcenter = 1.35
    ycenter = 1.9
    for t in np.arange(tstart_circle,tend_circle):
        trel = t - tstart_circle
        xtemp = xcenter + radius_circle*np.cos(15/90*trel)
        ytemp = ycenter + radius_circle*np.sin(15/90*trel)
        temp = np.array([453,zr[zind_rel[zi]],xtemp,ytemp])
        incloud1 = interpolate_points_cloud1(temp,'nearest')
        if incloud1 == 1.0:
            drone4_circle = np.vstack((drone4_circle,temp))

    drone5_circle = np.ndarray((0,4))
    xcenter = 1.35
    ycenter = 1.7
    for t in np.arange(tstart_circle,tend_circle):
        trel = t - tstart_circle
        xtemp = xcenter + radius_circle*np.cos(15/90*trel)
        ytemp = ycenter + radius_circle*np.sin(15/90*trel)
        temp = np.array([453,zr[zind_rel[zi]],xtemp,ytemp])
        incloud1 = interpolate_points_cloud1(temp,'nearest')
        if incloud1 == 1.0:
            drone5_circle = np.vstack((drone5_circle,temp))

    ################### Train and Test Data x,y,normalized GPs, does not change with noise and trials loops
    drones_train = np.vstack((drone1_circle,drone2_circle,drone3_circle,drone4_circle,drone5_circle))
    grid_unrolled = grid[4,zind_rel[zi]].reshape((-1,4))
    all_len_train_data_static_cs[zi] = len(drones_train)

    all_vars_static_cs[zi] = zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5].var()

    drones_train_normal = normalize(lwc_cloud1_polar,drones_train,cloud1.COM_2D_lwc_tz,449,1.185,origin_xy)
    grid_unrolled_normal = normalize(lwc_cloud1_polar,grid_unrolled,cloud1.COM_2D_lwc_tz,449,1.185,origin_xy)
    interpolate_rtrend = RegularGridInterpolator(points=np.arange(0,151).reshape(1,151),values=rtrend_global_median_updraft_norm,bounds_error=False,fill_value=0)
    COM =np.array([449,zr[zind_rel[zi]],(cloud1.COM_2D_lwc_tz[0,zind_rel[zi],0] + origin_xy[0])*0.01,(cloud1.COM_2D_lwc_tz[0,zind_rel[zi],1] + origin_xy[1])*0.01])
    ####### Trials to see effect of randomness in noise:
    #trials = np.arange(0,20)
    trials = np.arange(0,20)
    for i in trials:
        ############## Loop to see Effect if Noise variance
        noise = np.array([1e-3,0.1,0.25,0.5,0.75])
        #noise = np.array([1e-3])
        for j in range(len(noise)):
            #Training Data dependent on noise
            print('progress Static CS: Start of CS={},Trial={},Noise={}'.format(zi,i,j))
            time1 = datetime.datetime.now()
            zwind_train = atm.get_points(drones_train,'WT','linear')
            zwind_train = zwind_train.reshape((len(zwind_train),1))
            zwind_com = atm.get_points(COM,'WT','linear') + np.random.randn(1)*noise[j]
            zwind_train = zwind_train + np.random.randn(len(zwind_train),1)*noise[j]
            zwind_train_detrended = zwind_train - (zwind_com*interpolate_rtrend(drones_train_normal[:,3],'linear')).reshape((-1,1))

            rmse_trend = np.sqrt(np.mean((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5] -
            zwind_com*interpolate_rtrend(grid_unrolled_normal[:,3],'linear').reshape((140,150))[lwc_cloud1[4,zind_rel[zi]]>=1e-5])**2))

            explained_variance = np.sum((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5] -
            zwind_com*interpolate_rtrend(grid_unrolled_normal[:,3],'linear').reshape((140,150))[lwc_cloud1[4,zind_rel[zi]]>=1e-5])**2)
            total_variance = np.sum((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]- zwind_train.mean())**2)
            r2_trend = 1 - explained_variance/total_variance

            err_trend = (zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5] -
            zwind_com*interpolate_rtrend(grid_unrolled_normal[:,3],'linear').reshape((140,150))[lwc_cloud1[4,zind_rel[zi]]>=1e-5])
            err_stats_trend = np.array([np.mean(err_trend),np.median(err_trend),np.std(err_trend),skew(err_trend)])


            rmse_mean =np.sqrt(np.mean((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]- zwind_train.mean())**2))
            r2_mean = 0

            err_mean = zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]- zwind_train.mean()
            err_stats_mean = np.array([np.mean(err_mean),np.median(err_mean),np.std(err_mean),skew(err_mean)])
            ########################### skyscan.lib GP Model,Exponential x,y ########################
            M = GPModel(ndim=4,nvar=1, kernel_string=b"CovSum(CovExpArd, CovNoise)")
            M.update(drones_train,zwind_train.T)

            ## Values in log scale
            input_opt = {"gp_opt_python":True,'gp_opt_iter':10,'gp_hyper_bounds':((-10, 10), (-10, 10), (-10, 10), (-10, 10),
                             (-10, 10),
                             (-10, 10)),'gp_discr_tol':-1}
            M.optimize(input_opt)
            params1 = np.asarray(M.get_params()).reshape(6)

            predictions = np.ndarray((1,2,140*150))
            M.predict(grid_unrolled,predictions)

            mean_pred1 = predictions[0,0]
            var_pred1 = predictions[0,1]

            mean_pred1 = mean_pred1.reshape((140,150))
            var_pred1 = var_pred1.reshape((140,150))

            mean_std_pred_incloud11 = np.mean(np.sqrt(var_pred1[lwc_cloud1[4,zind_rel[zi]]>=1e-5]))
            rmse_incloud11 = np.sqrt(np.mean((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]-mean_pred1[lwc_cloud1[4,zind_rel[zi]]>=1e-5])**2))

            abs_err_incloud11 = np.abs(zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]-mean_pred1[lwc_cloud1[4,zind_rel[zi]]>=1e-5])
            std_pred_incloud11 = np.sqrt(var_pred1[lwc_cloud1[4,zind_rel[zi]]>=1e-5])
            test_std_soundness_incloud11 = (std_pred_incloud11-abs_err_incloud11)/std_pred_incloud11

            std_pred_soundness_incloud11 = np.array([np.percentile(test_std_soundness_incloud11,0.3),np.percentile(test_std_soundness_incloud11,5),
            np.percentile(test_std_soundness_incloud11,32)])


            explained_variance = np.sum((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5] -
                                mean_pred1[lwc_cloud1[4,zind_rel[zi]]>=1e-5])**2)
            total_variance = np.sum((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]- zwind_train.mean())**2)
            r2_incloud11 = 1 - explained_variance/total_variance

            err_incloud11 = zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]-mean_pred1[lwc_cloud1[4,zind_rel[zi]]>=1e-5]
            err_stats_incloud11 = np.array([np.mean(err_incloud11),np.median(err_incloud11),np.std(err_incloud11),skew(err_incloud11)])

            ########################### skyscan.lib GP Model,SE x,y ########################
            M2 = GPModel(ndim=4,nvar=1, kernel_string=b"CovSum(CovSEard, CovNoise)")
            M2.update(drones_train,zwind_train.T)

            ## Values in log scale
            input_opt = {"gp_opt_python":True,'gp_opt_iter':10,'gp_hyper_bounds':((-10, 10), (-10, 10), (-10, 10), (-10, 10),
                             (-10, 10),
                             (-10, 10)),'gp_discr_tol':-1}
            M2.optimize(input_opt)
            params2 = np.asarray(M2.get_params()).reshape(6)

            predictions2 = np.ndarray((1,2,140*150))
            M2.predict(grid_unrolled,predictions2)

            mean_pred2 = predictions2[0,0]
            var_pred2 = predictions2[0,1]

            mean_pred2 = mean_pred2.reshape((140,150))
            var_pred2 = var_pred2.reshape((140,150))

            mean_std_pred_incloud12 = np.mean(np.sqrt(var_pred2[lwc_cloud1[4,zind_rel[zi]]>=1e-5]))
            rmse_incloud12 = np.sqrt(np.mean((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]-mean_pred2[lwc_cloud1[4,zind_rel[zi]]>=1e-5])**2))

            abs_err_incloud12 = np.abs(zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]-mean_pred2[lwc_cloud1[4,zind_rel[zi]]>=1e-5])
            std_pred_incloud12 = np.sqrt(var_pred2[lwc_cloud1[4,zind_rel[zi]]>=1e-5])
            test_std_soundness_incloud12 = (std_pred_incloud12-abs_err_incloud12)/std_pred_incloud12

            std_pred_soundness_incloud12 = np.array([np.percentile(test_std_soundness_incloud12,0.3),np.percentile(test_std_soundness_incloud12,5),
            np.percentile(test_std_soundness_incloud12,32)])


            explained_variance = np.sum((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5] -
                                mean_pred2[lwc_cloud1[4,zind_rel[zi]]>=1e-5])**2)
            total_variance = np.sum((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]- zwind_train.mean())**2)
            r2_incloud12 = 1 - explained_variance/total_variance

            err_incloud12 = zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]-mean_pred2[lwc_cloud1[4,zind_rel[zi]]>=1e-5]
            err_stats_incloud12 = np.array([np.mean(err_incloud12),np.median(err_incloud12),np.std(err_incloud12),skew(err_incloud12)])
            ###################### libgp, normalized radius and corrected phi,not optimized,variograms hypers #####################

            M3 = GPModel(ndim=4,nvar=1, kernel_string=b"CovSum(CovExpArdPhi, CovNoise)")
            #M3 = GPModel(ndim=4,nvar=1, kernel_string="CovExpArdPhi")
            lt = np.log(51.026823559394558)
            lz = np.log(0.01*13.054954891415182)
            l_phi = np.log(23.025993674634258)
            lr = np.log(40.199201579845884)
            sigma2 = 0.5*np.log(0.84069334459964384)
            noise_var =0.5*np.log(noise[j]**2)

            params = np.array([lt,lz,l_phi,lr,sigma2,noise_var]).reshape(1,-1)
            M3.set_params(params)
            M3.update(drones_train_normal,zwind_train_detrended.T)
            params3 = params.reshape(6)
            predictions3 =np.nan*np.ndarray((1,2,140*150))
            M3.predict(grid_unrolled_normal,predictions3)

            mean_pred3 = predictions3[0,0].reshape(-1,1)
            var_pred3 = predictions3[0,1].reshape(-1,1)

            mean_pred3 = mean_pred3 + zwind_com*interpolate_rtrend(grid_unrolled_normal[:,3],'linear').reshape((-1,1))

            mean_pred3 = mean_pred3.reshape((140,150))
            var_pred3 = var_pred3.reshape((140,150))

            mean_std_pred_incloud13 = np.mean(np.sqrt(var_pred3[lwc_cloud1[4,zind_rel[zi]]>=1e-5]))
            rmse_incloud13 = np.sqrt(np.mean((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]-mean_pred3[lwc_cloud1[4,zind_rel[zi]]>=1e-5])**2))

            abs_err_incloud13 = np.abs(zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]-mean_pred3[lwc_cloud1[4,zind_rel[zi]]>=1e-5])
            std_pred_incloud13 = np.sqrt(var_pred3[lwc_cloud1[4,zind_rel[zi]]>=1e-5])
            test_std_soundness_incloud13 = (std_pred_incloud13-abs_err_incloud13)/std_pred_incloud13

            std_pred_soundness_incloud13 = np.array([np.percentile(test_std_soundness_incloud13,0.3),np.percentile(test_std_soundness_incloud13,5),
            np.percentile(test_std_soundness_incloud13,32)])


            explained_variance = np.sum((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5] -
                                mean_pred3[lwc_cloud1[4,zind_rel[zi]]>=1e-5])**2)
            total_variance = np.sum((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]- zwind_train.mean())**2)
            r2_incloud13 = 1 - explained_variance/total_variance

            err_incloud13 = zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]-mean_pred3[lwc_cloud1[4,zind_rel[zi]]>=1e-5]
            err_stats_incloud13 = np.array([np.mean(err_incloud13),np.median(err_incloud13),np.std(err_incloud13),skew(err_incloud13)])
            ###################### libgp, normalized radius and corrected phi,optimized #####################
            M4 = GPModel(ndim=4,nvar=1, kernel_string=b"CovSum(CovExpArdPhi, CovNoise)")
            M4.update(drones_train_normal,zwind_train_detrended.T)

            ########## Start optimization at random numbers to avoid optimization to all zeros
            #lt = np.random.randn(1)*3 + 3
            #lz = np.random.randn(1)*3 + 3
            #l_phi = np.random.randn(1)*3 + 3
            #lr = np.random.randn(1)*3 + 3
            #sigma2 = np.random.randn(1)*3 + 3
            #noise_var = np.random.randn(1)*3 + 3

            #params = np.array([lt,lz,l_phi,lr,sigma2,noise_var]).reshape(1,-1)
            #M4.set_params(params)

            ## Values in log scale
            input_opt = {"gp_opt_python":True,'gp_opt_iter':10,'gp_hyper_bounds':((-10, 10), (-10, 10), (-10, 10), (-10, 10),
                             (-10, 10),
                             (-10, 10)),'gp_discr_tol':-1}
            M4.optimize(input_opt)
            params4 = np.asarray(M4.get_params()).reshape(6)
            predictions4 = np.ndarray((1,2,140*150))

            M4.predict(grid_unrolled_normal,predictions4)

            mean_pred4 = predictions4[0,0].reshape(-1,1)
            var_pred4 = predictions4[0,1].reshape(-1,1)

            mean_pred4 = mean_pred4 + zwind_com*interpolate_rtrend(grid_unrolled_normal[:,3],'linear').reshape((-1,1))

            mean_pred4 = mean_pred4.reshape((140,150))
            var_pred4 = var_pred4.reshape((140,150))

            mean_std_pred_incloud14 = np.mean(np.sqrt(var_pred4[lwc_cloud1[4,zind_rel[zi]]>=1e-5]))
            rmse_incloud14 = np.sqrt(np.mean((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]-mean_pred4[lwc_cloud1[4,zind_rel[zi]]>=1e-5])**2))

            abs_err_incloud14 = np.abs(zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]-mean_pred4[lwc_cloud1[4,zind_rel[zi]]>=1e-5])
            std_pred_incloud14 = np.sqrt(var_pred4[lwc_cloud1[4,zind_rel[zi]]>=1e-5])
            test_std_soundness_incloud14 = (std_pred_incloud14-abs_err_incloud14)/std_pred_incloud14

            std_pred_soundness_incloud14 = np.array([np.percentile(test_std_soundness_incloud14,0.3),np.percentile(test_std_soundness_incloud14,5),
            np.percentile(test_std_soundness_incloud14,32)])


            explained_variance = np.sum((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5] -
                                mean_pred4[lwc_cloud1[4,zind_rel[zi]]>=1e-5])**2)
            total_variance = np.sum((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]- zwind_train.mean())**2)
            r2_incloud14 = 1 - explained_variance/total_variance

            err_incloud14 = zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]-mean_pred4[lwc_cloud1[4,zind_rel[zi]]>=1e-5]
            err_stats_incloud14 = np.array([np.mean(err_incloud14),np.median(err_incloud14),np.std(err_incloud14),skew(err_incloud14)])

            ################### libgp, normalized radius and corrected phi,optimized, starting from variogram hypers #####################
            M5 = GPModel(ndim=4,nvar=1, kernel_string=b"CovSum(CovExpArdPhi, CovNoise)")
            M5.update(drones_train_normal,zwind_train_detrended.T)

            ## Values in log scale
            input_opt = {"gp_opt_python":True,'gp_opt_iter':10,'gp_hyper_bounds':((-10, 10), (-10, 10), (-10, 10), (-10, 10),
                             (-10, 10),
                             (-10, 10)),'gp_discr_tol':-1}

            lt = np.log(51.026823559394558)
            lz = np.log(0.01*13.054954891415182)
            l_phi = np.log(23.025993674634258)
            lr = np.log(40.199201579845884)
            sigma2 = 0.5*np.log(0.84069334459964384)
            noise_var =0.5*np.log(noise[j]**2)

            params = np.array([lt,lz,l_phi,lr,sigma2,noise_var]).reshape(1,-1)
            M5.set_params(params)

            M5.optimize(input_opt)
            params5 = np.asarray(M5.get_params()).reshape(6)
            predictions5 = np.ndarray((1,2,140*150))

            M5.predict(grid_unrolled_normal,predictions5)

            mean_pred5 = predictions5[0,0].reshape(-1,1)
            var_pred5 = predictions5[0,1].reshape(-1,1)

            mean_pred5 = mean_pred5 + zwind_com*interpolate_rtrend(grid_unrolled_normal[:,3],'linear').reshape((-1,1))

            mean_pred5 = mean_pred5.reshape((140,150))
            var_pred5 = var_pred5.reshape((140,150))

            mean_std_pred_incloud15 = np.mean(np.sqrt(var_pred5[lwc_cloud1[4,zind_rel[zi]]>=1e-5]))
            rmse_incloud15 = np.sqrt(np.mean((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]-mean_pred5[lwc_cloud1[4,zind_rel[zi]]>=1e-5])**2))

            abs_err_incloud15 = np.abs(zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]-mean_pred5[lwc_cloud1[4,zind_rel[zi]]>=1e-5])
            std_pred_incloud15 = np.sqrt(var_pred5[lwc_cloud1[4,zind_rel[zi]]>=1e-5])
            test_std_soundness_incloud15 = (std_pred_incloud15-abs_err_incloud15)/std_pred_incloud15

            std_pred_soundness_incloud15 = np.array([np.percentile(test_std_soundness_incloud15,0.3),np.percentile(test_std_soundness_incloud15,5),
            np.percentile(test_std_soundness_incloud15,32)])


            explained_variance = np.sum((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5] -
                                mean_pred5[lwc_cloud1[4,zind_rel[zi]]>=1e-5])**2)
            total_variance = np.sum((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]- zwind_train.mean())**2)
            r2_incloud15 = 1 - explained_variance/total_variance


            err_incloud15 = zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]-mean_pred5[lwc_cloud1[4,zind_rel[zi]]>=1e-5]
            err_stats_incloud15 = np.array([np.mean(err_incloud15),np.median(err_incloud15),np.std(err_incloud15),skew(err_incloud15)])
            ############## These two tests serve to validate results of libgp

            ###################### GPy, normalized radius and uncorrected phi,optimized #####################
            try:
                k1 = GPy.kern.Exponential(input_dim=2,active_dims=[2,3],variance=[0.7],lengthscale = [20.3,40.2] ,ARD=True)
                k2 = GPy.kern.RBF(input_dim=2,active_dims=[2,3],variance=[0.7],lengthscale = [20.3,40.2] ,ARD=True)
                kernel = k1
                m = GPy.models.GPRegression(drones_train_normal,zwind_train_detrended,kernel)
                m.optimize_restarts(num_restarts = 10)
                params6 = np.nan * np.ndarray((6))
                params6[[4,2,3,5]] = m.param_array

                predictions6 = m.predict(grid_unrolled_normal)

                mean_pred6 = predictions6[0]
                var_pred6 = predictions6[1]

                mean_pred6 = mean_pred6 + zwind_com*interpolate_rtrend(grid_unrolled_normal[:,3],'linear').reshape((-1,1))
                mean_pred6 = mean_pred6.reshape((140,150))
                var_pred6 = var_pred6.reshape((140,150))

                mean_std_pred_incloud16 = np.mean(np.sqrt(var_pred6[lwc_cloud1[4,zind_rel[zi]]>=1e-5]))
                rmse_incloud16 = np.sqrt(np.mean((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]-mean_pred6[lwc_cloud1[4,zind_rel[zi]]>=1e-5])**2))

                abs_err_incloud16 = np.abs(zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]-mean_pred6[lwc_cloud1[4,zind_rel[zi]]>=1e-5])
                std_pred_incloud16 = np.sqrt(var_pred6[lwc_cloud1[4,zind_rel[zi]]>=1e-5])
                test_std_soundness_incloud16 = (std_pred_incloud16-abs_err_incloud16)/std_pred_incloud16

                std_pred_soundness_incloud16 = np.array([np.percentile(test_std_soundness_incloud16,0.3),np.percentile(test_std_soundness_incloud16,5),
                np.percentile(test_std_soundness_incloud16,32)])


                explained_variance = np.sum((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5] -
                                    mean_pred6[lwc_cloud1[4,zind_rel[zi]]>=1e-5])**2)
                total_variance = np.sum((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]- zwind_train.mean())**2)
                r2_incloud16 = 1 - explained_variance/total_variance

                err_incloud16 = zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]-mean_pred6[lwc_cloud1[4,zind_rel[zi]]>=1e-5]
                err_stats_incloud16 = np.array([np.mean(err_incloud16),np.median(err_incloud16),np.std(err_incloud16),skew(err_incloud16)])
                ###########################  GPy,Exponential x,y ########################
                kernel = GPy.kern.Exponential(input_dim=2,active_dims=[2,3],ARD=True)

                m2 = GPy.models.GPRegression(drones_train,zwind_train,kernel)
                m2.optimize_restarts(num_restarts = 10)

                params7 = np.nan * np.ndarray((6))
                params7[[4,2,3,5]] = m2.param_array

                predictions7 = m2.predict(grid_unrolled)

                mean_pred7 = predictions7[0]
                var_pred7 = predictions7[1]

                mean_pred7 = mean_pred7.reshape((140,150))
                var_pred7 = var_pred7.reshape((140,150))

                mean_std_pred_incloud17 = np.mean(np.sqrt(var_pred7[lwc_cloud1[4,zind_rel[zi]]>=1e-5]))
                rmse_incloud17 = np.sqrt(np.mean((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]-mean_pred7[lwc_cloud1[4,zind_rel[zi]]>=1e-5])**2))

                abs_err_incloud17 = np.abs(zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]-mean_pred7[lwc_cloud1[4,zind_rel[zi]]>=1e-5])
                std_pred_incloud17 = np.sqrt(var_pred7[lwc_cloud1[4,zind_rel[zi]]>=1e-5])
                test_std_soundness_incloud17 = (std_pred_incloud17-abs_err_incloud17)/std_pred_incloud17

                std_pred_soundness_incloud17 = np.array([np.percentile(test_std_soundness_incloud17,0.3),np.percentile(test_std_soundness_incloud17,5),
                np.percentile(test_std_soundness_incloud17,32)])


                explained_variance = np.sum((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5] -
                                    mean_pred7[lwc_cloud1[4,zind_rel[zi]]>=1e-5])**2)
                total_variance = np.sum((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]- zwind_train.mean())**2)
                r2_incloud17 = 1 - explained_variance/total_variance

                err_incloud17 = zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]-mean_pred7[lwc_cloud1[4,zind_rel[zi]]>=1e-5]
                err_stats_incloud17 = np.array([np.mean(err_incloud17),np.median(err_incloud17),np.std(err_incloud17),skew(err_incloud17)])
            except KeyboardInterrupt:
                print('KeyboardInterrupt')
                break
            except:
                print('Something went wrong with LinAlg, please continue')

            ########################################### libgp, SE, xy detrended
            M6 = GPModel(ndim=4,nvar=1, kernel_string=b"CovSum(CovSEard, CovNoise)")
            M6.update(drones_train,zwind_train_detrended.T)

            ## Values in log scale
            input_opt = {"gp_opt_python":True,'gp_opt_iter':10,'gp_hyper_bounds':((-10, 10), (-10, 10), (-10, 10), (-10, 10),
                             (-10, 10),
                             (-10, 10)),'gp_discr_tol':-1}

            M6.optimize(input_opt)
            params8 = np.asarray(M6.get_params()).reshape(6)
            predictions8 = np.ndarray((1,2,140*150))

            M6.predict(grid_unrolled,predictions8)

            mean_pred8 = predictions8[0,0].reshape(-1,1)
            var_pred8 = predictions8[0,1].reshape(-1,1)

            mean_pred8 = mean_pred8 + zwind_com*interpolate_rtrend(grid_unrolled_normal[:,3],'linear').reshape((-1,1))

            mean_pred8 = mean_pred8.reshape((140,150))
            var_pred8 = var_pred8.reshape((140,150))

            mean_std_pred_incloud18 = np.mean(np.sqrt(var_pred8[lwc_cloud1[4,zind_rel[zi]]>=1e-5]))
            rmse_incloud18 = np.sqrt(np.mean((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]-mean_pred8[lwc_cloud1[4,zind_rel[zi]]>=1e-5])**2))

            abs_err_incloud18 = np.abs(zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]-mean_pred8[lwc_cloud1[4,zind_rel[zi]]>=1e-5])
            std_pred_incloud18 = np.sqrt(var_pred8[lwc_cloud1[4,zind_rel[zi]]>=1e-5])
            test_std_soundness_incloud18 = (std_pred_incloud18-abs_err_incloud18)/std_pred_incloud18

            std_pred_soundness_incloud18 = np.array([np.percentile(test_std_soundness_incloud18,0.3),np.percentile(test_std_soundness_incloud18,5),
            np.percentile(test_std_soundness_incloud18,32)])


            explained_variance = np.sum((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5] -
                                mean_pred8[lwc_cloud1[4,zind_rel[zi]]>=1e-5])**2)
            total_variance = np.sum((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]- zwind_train.mean())**2)
            r2_incloud18 = 1 - explained_variance/total_variance


            err_incloud18 = zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]-mean_pred8[lwc_cloud1[4,zind_rel[zi]]>=1e-5]
            err_stats_incloud18 = np.array([np.mean(err_incloud18),np.median(err_incloud18),np.std(err_incloud18),skew(err_incloud18)])
            ########################################### libgp Exp xy, detrended data
            M7 = GPModel(ndim=4,nvar=1, kernel_string=b"CovSum(CovExpArd, CovNoise)")
            M7.update(drones_train,zwind_train_detrended.T)

            ## Values in log scale
            input_opt = {"gp_opt_python":True,'gp_opt_iter':10,'gp_hyper_bounds':((-10, 10), (-10, 10), (-10, 10), (-10, 10),
                             (-10, 10),
                             (-10, 10)),'gp_discr_tol':-1}

            M7.optimize(input_opt)
            params9 = np.asarray(M7.get_params()).reshape(6)
            predictions9 = np.ndarray((1,2,140*150))

            M7.predict(grid_unrolled,predictions9)

            mean_pred9 = predictions9[0,0].reshape(-1,1)
            var_pred9 = predictions9[0,1].reshape(-1,1)

            mean_pred9 = mean_pred9 + zwind_com*interpolate_rtrend(grid_unrolled_normal[:,3],'linear').reshape((-1,1))

            mean_pred9 = mean_pred9.reshape((140,150))
            var_pred9 = var_pred9.reshape((140,150))

            mean_std_pred_incloud19 = np.mean(np.sqrt(var_pred9[lwc_cloud1[4,zind_rel[zi]]>=1e-5]))
            rmse_incloud19 = np.sqrt(np.mean((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]-mean_pred9[lwc_cloud1[4,zind_rel[zi]]>=1e-5])**2))

            abs_err_incloud19 = np.abs(zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]-mean_pred9[lwc_cloud1[4,zind_rel[zi]]>=1e-5])
            std_pred_incloud19 = np.sqrt(var_pred9[lwc_cloud1[4,zind_rel[zi]]>=1e-5])
            test_std_soundness_incloud19 = (std_pred_incloud19-abs_err_incloud19)/std_pred_incloud19

            std_pred_soundness_incloud19 = np.array([np.percentile(test_std_soundness_incloud19,0.3),np.percentile(test_std_soundness_incloud19,5),
            np.percentile(test_std_soundness_incloud19,32)])


            explained_variance = np.sum((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5] -
                                mean_pred9[lwc_cloud1[4,zind_rel[zi]]>=1e-5])**2)
            total_variance = np.sum((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]- zwind_train.mean())**2)
            r2_incloud19 = 1 - explained_variance/total_variance


            err_incloud19 = zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]-mean_pred9[lwc_cloud1[4,zind_rel[zi]]>=1e-5]
            err_stats_incloud19 = np.array([np.mean(err_incloud19),np.median(err_incloud19),np.std(err_incloud19),skew(err_incloud19)])

            ############################# saving_results
            all_rmse_static_cs[zi,i,j] =np.array([rmse_incloud11, rmse_incloud12, rmse_incloud13, rmse_incloud14, rmse_incloud15, rmse_incloud16,
            rmse_incloud17,rmse_trend, rmse_mean,rmse_incloud18,rmse_incloud19 ])


            all_r2_static_cs[zi,i,j] = np.array([r2_incloud11,r2_incloud12,r2_incloud13,r2_incloud14,r2_incloud15,r2_incloud16,r2_incloud17, r2_trend, r2_mean,
            r2_incloud18,r2_incloud19])


            all_std_pred_static_cs[zi,i,j] = np.array([mean_std_pred_incloud11, mean_std_pred_incloud12, mean_std_pred_incloud13, mean_std_pred_incloud14,
            mean_std_pred_incloud15, mean_std_pred_incloud16, mean_std_pred_incloud17,mean_std_pred_incloud18,mean_std_pred_incloud19])

            all_std_pred_soundness_static_cs[zi,i,j] = np.array([std_pred_soundness_incloud11, std_pred_soundness_incloud12, std_pred_soundness_incloud13,
            std_pred_soundness_incloud14, std_pred_soundness_incloud15, std_pred_soundness_incloud16, std_pred_soundness_incloud17,
            std_pred_soundness_incloud18,std_pred_soundness_incloud19])

            all_hypers_static_cs[zi,i,j] = np.array([params1,params2,params3,params4,params5,params6,params7,params8,params9])

            all_error_stats_static_cs[zi,i,j] = np.array([err_stats_incloud11,err_stats_incloud12, err_stats_incloud13, err_stats_incloud14,
            err_stats_incloud15, err_stats_incloud16,err_stats_incloud17,err_stats_trend,err_stats_mean,err_stats_incloud18,err_stats_incloud19])

            outfile = '/home/dselle/Skyscanner/data_exploration/results/TestGP/dump_results_tests_cs_gp2.npz'
            np.savez(outfile, all_rmse_static_cs=all_rmse_static_cs, all_r2_static_cs=all_r2_static_cs,all_std_pred_static_cs=all_std_pred_static_cs,
            all_std_pred_soundness_static_cs=all_std_pred_soundness_static_cs, all_len_train_data_static_cs = all_len_train_data_static_cs,
            all_hypers_static_cs = all_hypers_static_cs, all_error_stats_static_cs = all_error_stats_static_cs, all_vars_static_cs = all_vars_static_cs,
            all_rmse_dyn_cs=all_rmse_dyn_cs,all_r2_dyn_cs=all_r2_dyn_cs,all_std_pred_dyn_cs = all_std_pred_dyn_cs,
            all_std_pred_soundness_dyn_cs = all_std_pred_soundness_dyn_cs, all_len_train_data_dyn_cs = all_len_train_data_dyn_cs,
            all_hypers_dyn_cs = all_hypers_dyn_cs, all_error_stats_dyn_cs=all_error_stats_dyn_cs, all_vars_dyn_cs=all_vars_dyn_cs)

            print('Total progress:{}%'.format((80*(zi)+4*(i)+(j+1))/1360*100))
            time2 = datetime.datetime.now()
            print(time2-time1)
    ##########################################################################
    ###################### dynamic test GP single CS #########################
    ##########################################################################


    tstart_circle = 449
    tend_circle = 526
    radius_circle = 0.09

    drone1_circle = np.ndarray((0,4))
    xcenter = 1.1
    ycenter = 2.1
    for t in np.arange(tstart_circle,tend_circle):
        trel = t - tstart_circle
        xtemp = xcenter + radius_circle*np.cos(15/90*trel)
        ytemp = ycenter + radius_circle*np.sin(15/90*trel)
        temp = np.array([t,zr[zind_rel[zi]],xtemp,ytemp])
        incloud1 = interpolate_points_cloud1(temp,'nearest')
        if incloud1 == 1.0:
            drone1_circle = np.vstack((drone1_circle,temp))

    drone2_circle = np.ndarray((0,4))
    xcenter = 1.4
    ycenter = 2.06
    for t in np.arange(tstart_circle,tend_circle):
        trel = t - tstart_circle
        xtemp = xcenter + radius_circle*np.cos(15/90*trel)
        ytemp = ycenter + radius_circle*np.sin(15/90*trel)
        temp = np.array([t,zr[zind_rel[zi]],xtemp,ytemp])
        incloud1 = interpolate_points_cloud1(temp,'nearest')
        if incloud1 == 1.0:
            drone2_circle = np.vstack((drone2_circle,temp))

    drone3_circle = np.ndarray((0,4))
    xcenter = 1.23
    ycenter = 1.9
    for t in np.arange(tstart_circle,tend_circle):
        trel = t - tstart_circle
        xtemp = xcenter + radius_circle*np.cos(15/90*trel)
        ytemp = ycenter + radius_circle*np.sin(15/90*trel)
        temp = np.array([t,zr[zind_rel[zi]],xtemp,ytemp])
        incloud1 = interpolate_points_cloud1(temp,'nearest')
        if incloud1 == 1.0:
            drone3_circle = np.vstack((drone3_circle,temp))

    drone4_circle = np.ndarray((0,4))
    xcenter = 1.35
    ycenter = 1.9
    for t in np.arange(tstart_circle,tend_circle):
        trel = t - tstart_circle
        xtemp = xcenter + radius_circle*np.cos(15/90*trel)
        ytemp = ycenter + radius_circle*np.sin(15/90*trel)
        temp = np.array([t,zr[zind_rel[zi]],xtemp,ytemp])
        incloud1 = interpolate_points_cloud1(temp,'nearest')
        if incloud1 == 1.0:
            drone4_circle = np.vstack((drone4_circle,temp))

    drone5_circle = np.ndarray((0,4))
    xcenter = 1.35
    ycenter = 1.7
    for t in np.arange(tstart_circle,tend_circle):
        trel = t - tstart_circle
        xtemp = xcenter + radius_circle*np.cos(15/90*trel)
        ytemp = ycenter + radius_circle*np.sin(15/90*trel)
        temp = np.array([t,zr[zind_rel[zi]],xtemp,ytemp])
        incloud1 = interpolate_points_cloud1(temp,'nearest')
        if incloud1 == 1.0:
            drone5_circle = np.vstack((drone5_circle,temp))


    ################### Train and Test Data not dependent on noise ################
    drones_train_dyn_cs = np.vstack((drone1_circle,drone2_circle,drone3_circle,drone4_circle,drone5_circle))

    drones_train_dyn_cs = drones_train_dyn_cs[drones_train_dyn_cs[:,0]!=480]

    all_len_train_data_dyn_cs[zi] = len(drones_train_dyn_cs)

    all_vars_dyn_cs[zi] = zwind_data1[0:76,zind_rel[zi]][lwc_cloud1[0:76,zind_rel[zi]]>=1e-5].var()

    drones_train_normal_dyn_cs = normalize(lwc_cloud1_polar,drones_train_dyn_cs,cloud1.COM_2D_lwc_tz,449,1.185,origin_xy)
    grid_unrolled_dyn_cs = grid[76,zind_rel[zi]].reshape((-1,4))
    grid_unrolled_normal_dyn_cs = normalize(lwc_cloud1_polar,grid_unrolled_dyn_cs,cloud1.COM_2D_lwc_tz,449,1.185,origin_xy)
    interpolate_rtrend = RegularGridInterpolator(points=np.arange(0,151).reshape(1,151),values=rtrend_global_median_updraft_norm,bounds_error=False,fill_value=0)

    COM =np.array([449,zr[zind_rel[zi]],(cloud1.COM_2D_lwc_tz[0,zind_rel[zi],0] + origin_xy[0])*0.01,(cloud1.COM_2D_lwc_tz[0,zind_rel[zi],1] + origin_xy[1])*0.01])

    ####### Trials to see effect of randomness in noise:
    #trials = np.arange(0,20)
    trials = np.arange(0,20)
    for i in trials:
        noise = np.array([1e-3,0.1,0.25,0.5,0.75])
        #noise = np.array([1e-3])
        ################## See the effect of noise
        for j in range(len(noise)):
            print('progress Dyn CS: CS={},Trial={},Noise={}'.format(zi,i,j))
            time1 = datetime.datetime.now()
            zwind_com = atm.get_points(COM,'WT','linear') + np.random.randn(1)*noise[j]

            zwind_train_dyn_cs = atm.get_points(drones_train_dyn_cs,'WT','linear')
            zwind_train_dyn_cs = zwind_train_dyn_cs.reshape((len(zwind_train_dyn_cs),1))
            zwind_train_dyn_cs = zwind_train_dyn_cs + np.random.randn(len(zwind_train_dyn_cs),1)*noise[j]

            zwind_train_detrended_dyn_cs = zwind_train_dyn_cs - (zwind_com*interpolate_rtrend(drones_train_normal_dyn_cs[:,3],'linear')).reshape((-1,1))

            rmse_trend_dyn_cs = np.sqrt(np.mean((zwind_data1[76,zind_rel[zi]][lwc_cloud1[76,zind_rel[zi]]>=1e-5] -
            zwind_com*interpolate_rtrend(grid_unrolled_normal_dyn_cs[:,3],'linear').reshape((140,150))[lwc_cloud1[76,zind_rel[zi]]>=1e-5])**2))

            explained_variance = np.sum((zwind_data1[76,zind_rel[zi]][lwc_cloud1[76,zind_rel[zi]]>=1e-5] -
            zwind_com*interpolate_rtrend(grid_unrolled_normal_dyn_cs[:,3],'linear').reshape((140,150))[lwc_cloud1[76,zind_rel[zi]]>=1e-5])**2)
            total_variance = np.sum((zwind_data1[76,zind_rel[zi]][lwc_cloud1[76,zind_rel[zi]]>=1e-5]- zwind_train_dyn_cs.mean())**2)
            r2_trend_dyn_cs = 1 - explained_variance/total_variance

            err_trend_dyn_cs = (zwind_data1[76,zind_rel[zi]][lwc_cloud1[76,zind_rel[zi]]>=1e-5]-
            zwind_com*interpolate_rtrend(grid_unrolled_normal_dyn_cs[:,3],'linear').reshape((140,150))[lwc_cloud1[76,zind_rel[zi]]>=1e-5])
            err_stats_trend_dyn_cs = np.array([np.mean(err_trend_dyn_cs),np.median(err_trend_dyn_cs),np.std(err_trend_dyn_cs),skew(err_trend_dyn_cs)])

            rmse_mean_dyn_cs =np.sqrt(np.mean((zwind_data1[76,zind_rel[zi]][lwc_cloud1[76,zind_rel[zi]]>=1e-5]- zwind_train_dyn_cs.mean())**2))
            r2_mean_dyn_cs = 0

            err_mean_dyn_cs = zwind_data1[76,zind_rel[zi]][lwc_cloud1[76,zind_rel[zi]]>=1e-5]- zwind_train_dyn_cs.mean()
            err_stats_mean_dyn_cs = np.array([np.mean(err_mean_dyn_cs),np.median(err_trend_dyn_cs),np.std(err_trend_dyn_cs),skew(err_trend_dyn_cs)])
            ############################################## libgp Exponential t,x,y, #####################################################
            M_dyn_cs = GPModel(ndim=4,nvar=1, kernel_string=b"CovSum(CovExpArd, CovNoise)")
            #M3 = GPModel(ndim=4,nvar=1, kernel_string="CovExpArdPhi")
            #drones_train_normal_dyn_cs_wo_t = drones_train_normal_dyn_cs.copy()
            #drones_train_normal_dyn_cs_wo_t[:,0] = 1
            #drones_train_normal_dyn_cs_wo_phi = drones_train_normal_dyn_cs.copy()
            #drones_train_normal_dyn_cs_wo_phi[:,2] = 1
            M_dyn_cs.update(drones_train_dyn_cs,zwind_train_dyn_cs.T)

            ## Values in log scale
            input_opt = {"gp_opt_python":True,'gp_opt_iter':10,'gp_hyper_bounds':((-10, 10), (-10, 10), (-10, 10), (-10, 10),
                             (-10, 10),
                             (-10, 10)),'gp_discr_tol':-1}
            M_dyn_cs.optimize(input_opt)

            predictions_dyn_cs = np.ndarray((1,2,140*150))
            params1_dyn_cs = np.asarray(M_dyn_cs.get_params()).reshape(6)
            #grid_unrolled_normal_dyn_cs_wo_t = grid_unrolled_normal_dyn_cs.copy()
            #grid_unrolled_normal_dyn_cs_wo_t[:,0] =1
            #grid_unrolled_normal_dyn_cs_wo_phi = grid_unrolled_normal_dyn_cs.copy()
            #grid_unrolled_normal_dyn_cs_wo_phi[:,2] = 1
            M_dyn_cs.predict(grid_unrolled_dyn_cs,predictions_dyn_cs)

            mean_pred_dyn_cs = predictions_dyn_cs[0,0].reshape(-1,1)
            var_pred_dyn_cs = predictions_dyn_cs[0,1].reshape(-1,1)

            mean_pred_dyn_cs = mean_pred_dyn_cs.reshape((140,150))
            var_pred_dyn_cs = var_pred_dyn_cs.reshape((140,150))

            mean_std_pred_incloud11_dyn_cs = np.mean(np.sqrt(var_pred_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5]))
            rmse_incloud11_dyn_cs = np.sqrt(np.mean((zwind_data1[76,zind_rel[zi]][lwc_cloud1[76,zind_rel[zi]]>=1e-5]-mean_pred_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5])**2))

            abs_err_incloud11_dyn_cs = np.abs(zwind_data1[76,zind_rel[zi]][lwc_cloud1[76,zind_rel[zi]]>=1e-5]-mean_pred_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5])
            std_pred_incloud11_dyn_cs = np.sqrt(var_pred_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5])
            test_std_soundness_incloud11_dyn_cs = (std_pred_incloud11_dyn_cs-abs_err_incloud11_dyn_cs)/std_pred_incloud11_dyn_cs

            std_pred_soundness_incloud11_dyn_cs = np.array([np.percentile(test_std_soundness_incloud11_dyn_cs,0.3),np.percentile(test_std_soundness_incloud11_dyn_cs,5),
            np.percentile(test_std_soundness_incloud11_dyn_cs,32)])


            explained_variance = np.sum((zwind_data1[76,zind_rel[zi]][lwc_cloud1[76,zind_rel[zi]]>=1e-5] -
                                mean_pred_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5])**2)
            total_variance = np.sum((zwind_data1[76,zind_rel[zi]][lwc_cloud1[76,zind_rel[zi]]>=1e-5]- zwind_train_dyn_cs.mean())**2)
            r2_incloud11_dyn_cs = 1 - explained_variance/total_variance

            err_incloud11_dyn_cs = zwind_data1[76,zind_rel[zi]][lwc_cloud1[76,zind_rel[zi]]>=1e-5]-mean_pred_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5]
            err_stats_incloud11_dyn_cs = np.array([np.mean(err_incloud11_dyn_cs),np.median(err_incloud11_dyn_cs),np.std(err_incloud11_dyn_cs),skew(err_incloud11_dyn_cs)])

            ############################################## libgp RBF t,x,y, #####################################################
            M2_dyn_cs = GPModel(ndim=4,nvar=1, kernel_string=b"CovSum(CovSEard, CovNoise)")
            #M3 = GPModel(ndim=4,nvar=1, kernel_string="CovExpArdPhi")
            #drones_train_normal_dyn_cs_wo_t = drones_train_normal_dyn_cs.copy()
            #drones_train_normal_dyn_cs_wo_t[:,0] = 1
            #drones_train_normal_dyn_cs_wo_phi = drones_train_normal_dyn_cs.copy()
            #drones_train_normal_dyn_cs_wo_phi[:,2] = 1
            M2_dyn_cs.update(drones_train_dyn_cs,zwind_train_dyn_cs.T)

            ## Values in log scale
            input_opt = {"gp_opt_python":True,'gp_opt_iter':10,'gp_hyper_bounds':((-10, 10), (-10, 10), (-10, 10), (-10, 10),
                             (-10, 10),
                             (-10, 10)),'gp_discr_tol':-1}
            M2_dyn_cs.optimize(input_opt)

            params2_dyn_cs = np.asarray(M2_dyn_cs.get_params()).reshape(6)
            predictions2_dyn_cs = np.ndarray((1,2,140*150))

            #grid_unrolled_normal_dyn_cs_wo_t = grid_unrolled_normal_dyn_cs.copy()
            #grid_unrolled_normal_dyn_cs_wo_t[:,0] =1
            #grid_unrolled_normal_dyn_cs_wo_phi = grid_unrolled_normal_dyn_cs.copy()
            #grid_unrolled_normal_dyn_cs_wo_phi[:,2] = 1
            M2_dyn_cs.predict(grid_unrolled_dyn_cs,predictions2_dyn_cs)

            mean_pred2_dyn_cs = predictions2_dyn_cs[0,0].reshape(-1,1)
            var_pred2_dyn_cs = predictions2_dyn_cs[0,1].reshape(-1,1)

            mean_pred2_dyn_cs = mean_pred2_dyn_cs.reshape((140,150))
            var_pred2_dyn_cs = var_pred2_dyn_cs.reshape((140,150))

            mean_std_pred_incloud12_dyn_cs = np.mean(np.sqrt(var_pred2_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5]))
            rmse_incloud12_dyn_cs = np.sqrt(np.mean((zwind_data1[76,zind_rel[zi]][lwc_cloud1[76,zind_rel[zi]]>=1e-5]-mean_pred2_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5])**2))

            abs_err_incloud12_dyn_cs = np.abs(zwind_data1[76,zind_rel[zi]][lwc_cloud1[76,zind_rel[zi]]>=1e-5]-mean_pred2_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5])
            std_pred_incloud12_dyn_cs = np.sqrt(var_pred2_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5])
            test_std_soundness_incloud12_dyn_cs = (std_pred_incloud12_dyn_cs-abs_err_incloud12_dyn_cs)/std_pred_incloud12_dyn_cs

            std_pred_soundness_incloud12_dyn_cs = np.array([np.percentile(test_std_soundness_incloud12_dyn_cs,0.3),np.percentile(test_std_soundness_incloud12_dyn_cs,5),
            np.percentile(test_std_soundness_incloud12_dyn_cs,32)])


            explained_variance = np.sum((zwind_data1[76,zind_rel[zi]][lwc_cloud1[76,zind_rel[zi]]>=1e-5] -
                                mean_pred2_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5])**2)
            total_variance = np.sum((zwind_data1[76,zind_rel[zi]][lwc_cloud1[76,zind_rel[zi]]>=1e-5]- zwind_train_dyn_cs.mean())**2)
            r2_incloud12_dyn_cs = 1 - explained_variance/total_variance

            err_incloud12_dyn_cs = zwind_data1[76,zind_rel[zi]][lwc_cloud1[76,zind_rel[zi]]>=1e-5]-mean_pred2_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5]
            err_stats_incloud12_dyn_cs = np.array([np.mean(err_incloud12_dyn_cs),np.median(err_incloud12_dyn_cs),np.std(err_incloud12_dyn_cs),skew(err_incloud12_dyn_cs)])
            ############################################## libgp normalized, corrected phi, variogram hypers #####################################################
            M3_dyn_cs = GPModel(ndim=4,nvar=1, kernel_string=b"CovSum(CovExpArdPhi, CovNoise)")

            lt = np.log(51.026823559394558)
            lz = np.log(0.01*13.054954891415182)
            l_phi = np.log(23.025993674634258)
            lr = np.log(40.199201579845884)
            sigma2 = 0.5*np.log(0.84069334459964384)
            noise_var =0.5*np.log(noise[j]**2)

            params = np.array([lt,lz,l_phi,lr,sigma2,noise_var]).reshape(1,-1)
            params3_dyn_cs = params.reshape(6)
            M3_dyn_cs.set_params(params)
            M3_dyn_cs.update(drones_train_normal_dyn_cs,zwind_train_detrended_dyn_cs.T)

            predictions3_dyn_cs =np.nan*np.ndarray((1,2,140*150))
            M3_dyn_cs.predict(grid_unrolled_normal_dyn_cs,predictions3_dyn_cs)

            mean_pred3_dyn_cs = predictions3_dyn_cs[0,0].reshape(-1,1)
            var_pred3_dyn_cs = predictions3_dyn_cs[0,1].reshape(-1,1)

            mean_pred3_dyn_cs = mean_pred3_dyn_cs + (zwind_com*interpolate_rtrend(grid_unrolled_normal_dyn_cs[:,3],'linear')).reshape((-1,1))

            mean_pred3_dyn_cs = mean_pred3_dyn_cs.reshape((140,150))
            var_pred3_dyn_cs = var_pred3_dyn_cs.reshape((140,150))

            mean_std_pred_incloud13_dyn_cs = np.mean(np.sqrt(var_pred3_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5]))
            rmse_incloud13_dyn_cs = np.sqrt(np.mean((zwind_data1[76,zind_rel[zi]][lwc_cloud1[76,zind_rel[zi]]>=1e-5]-mean_pred3_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5])**2))

            abs_err_incloud13_dyn_cs = np.abs(zwind_data1[76,zind_rel[zi]][lwc_cloud1[76,zind_rel[zi]]>=1e-5]-mean_pred3_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5])
            std_pred_incloud13_dyn_cs = np.sqrt(var_pred3_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5])
            test_std_soundness_incloud13_dyn_cs = (std_pred_incloud13_dyn_cs-abs_err_incloud13_dyn_cs)/std_pred_incloud13_dyn_cs

            std_pred_soundness_incloud13_dyn_cs = np.array([np.percentile(test_std_soundness_incloud13_dyn_cs,0.3),np.percentile(test_std_soundness_incloud13_dyn_cs,5),
            np.percentile(test_std_soundness_incloud13_dyn_cs,32)])


            explained_variance = np.sum((zwind_data1[76,zind_rel[zi]][lwc_cloud1[76,zind_rel[zi]]>=1e-5] -
                                mean_pred3_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5])**2)
            total_variance = np.sum((zwind_data1[76,zind_rel[zi]][lwc_cloud1[76,zind_rel[zi]]>=1e-5]- zwind_train_dyn_cs.mean())**2)
            r2_incloud13_dyn_cs = 1 - explained_variance/total_variance

            err_incloud13_dyn_cs = zwind_data1[76,zind_rel[zi]][lwc_cloud1[76,zind_rel[zi]]>=1e-5]-mean_pred3_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5]
            err_stats_incloud13_dyn_cs = np.array([np.mean(err_incloud13_dyn_cs),np.median(err_incloud13_dyn_cs),np.std(err_incloud13_dyn_cs),skew(err_incloud13_dyn_cs)])
            ############################################## libgp normalized, corrected phi,optimized #####################################################
            M4_dyn_cs = GPModel(ndim=4,nvar=1, kernel_string=b"CovSum(CovExpArdPhi, CovNoise)")
            #M3 = GPModel(ndim=4,nvar=1, kernel_string="CovExpArdPhi")
            #drones_train_normal_dyn_cs_wo_t = drones_train_normal_dyn_cs.copy()
            #drones_train_normal_dyn_cs_wo_t[:,0] = 1
            #drones_train_normal_dyn_cs_wo_phi = drones_train_normal_dyn_cs.copy()
            #drones_train_normal_dyn_cs_wo_phi[:,2] = 1
            M4_dyn_cs.update(drones_train_normal_dyn_cs,zwind_train_detrended_dyn_cs.T)
            ########## Start optimization at random numbers to avoid optimization to all zeros
            #lt = np.random.randn(1)*3 + 3
            #lz = np.random.randn(1)*3 + 3
            #l_phi = np.random.randn(1)*3 + 3
            #lr = np.random.randn(1)*3 + 3
            #sigma2 = np.random.randn(1)*3 + 3
            #noise_var = np.random.randn(1)*3 + 3

            #params = np.array([lt,lz,l_phi,lr,sigma2,noise_var]).reshape(1,-1)
            #M4_dyn_cs.set_params(params)

            ## Values in log scale
            input_opt = {"gp_opt_python":True,'gp_opt_iter':10,'gp_hyper_bounds':((-10, 10), (-10, 10), (-10, 10), (-10, 10),
                             (-10, 10),
                             (-10, 10)),'gp_discr_tol':-1}
            M4_dyn_cs.optimize(input_opt)
            params4_dyn_cs = np.asarray(M4_dyn_cs.get_params()).reshape(6)

            predictions4_dyn_cs = np.ndarray((1,2,140*150))

            #grid_unrolled_normal_dyn_cs_wo_t = grid_unrolled_normal_dyn_cs.copy()
            #grid_unrolled_normal_dyn_cs_wo_t[:,0] =1
            #grid_unrolled_normal_dyn_cs_wo_phi = grid_unrolled_normal_dyn_cs.copy()
            #grid_unrolled_normal_dyn_cs_wo_phi[:,2] = 1
            M4_dyn_cs.predict(grid_unrolled_normal_dyn_cs,predictions4_dyn_cs)

            mean_pred4_dyn_cs = predictions4_dyn_cs[0,0].reshape(-1,1)
            var_pred4_dyn_cs = predictions4_dyn_cs[0,1].reshape(-1,1)

            mean_pred4_dyn_cs = mean_pred4_dyn_cs + (zwind_com*interpolate_rtrend(grid_unrolled_normal_dyn_cs[:,3],'linear')).reshape((-1,1))

            mean_pred4_dyn_cs = mean_pred4_dyn_cs.reshape((140,150))
            var_pred4_dyn_cs = var_pred4_dyn_cs.reshape((140,150))

            mean_std_pred_incloud14_dyn_cs = np.mean(np.sqrt(var_pred4_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5]))
            rmse_incloud14_dyn_cs = np.sqrt(np.mean((zwind_data1[76,zind_rel[zi]][lwc_cloud1[76,zind_rel[zi]]>=1e-5]-mean_pred4_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5])**2))

            abs_err_incloud14_dyn_cs = np.abs(zwind_data1[76,zind_rel[zi]][lwc_cloud1[76,zind_rel[zi]]>=1e-5]-mean_pred4_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5])
            std_pred_incloud14_dyn_cs = np.sqrt(var_pred4_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5])
            test_std_soundness_incloud14_dyn_cs = (std_pred_incloud14_dyn_cs-abs_err_incloud14_dyn_cs)/std_pred_incloud14_dyn_cs

            std_pred_soundness_incloud14_dyn_cs = np.array([np.percentile(test_std_soundness_incloud14_dyn_cs,0.3),np.percentile(test_std_soundness_incloud14_dyn_cs,5),
            np.percentile(test_std_soundness_incloud14_dyn_cs,32)])


            explained_variance = np.sum((zwind_data1[76,zind_rel[zi]][lwc_cloud1[76,zind_rel[zi]]>=1e-5] -
                                mean_pred4_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5])**2)
            total_variance = np.sum((zwind_data1[76,zind_rel[zi]][lwc_cloud1[76,zind_rel[zi]]>=1e-5]- zwind_train_dyn_cs.mean())**2)
            r2_incloud14_dyn_cs = 1 - explained_variance/total_variance

            err_incloud14_dyn_cs = zwind_data1[76,zind_rel[zi]][lwc_cloud1[76,zind_rel[zi]]>=1e-5]-mean_pred4_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5]
            err_stats_incloud14_dyn_cs = np.array([np.mean(err_incloud14_dyn_cs),np.median(err_incloud14_dyn_cs),np.std(err_incloud14_dyn_cs),skew(err_incloud14_dyn_cs)])
            ############################################## libgp normalized, corrected phi,optimized starting from variogram hypers #####################################################
            M5_dyn_cs = GPModel(ndim=4,nvar=1, kernel_string=b"CovSum(CovExpArdPhi, CovNoise)")
            #M3 = GPModel(ndim=4,nvar=1, kernel_string="CovExpArdPhi")
            #drones_train_normal_dyn_cs_wo_t = drones_train_normal_dyn_cs.copy()
            #drones_train_normal_dyn_cs_wo_t[:,0] = 1
            #drones_train_normal_dyn_cs_wo_phi = drones_train_normal_dyn_cs.copy()
            #drones_train_normal_dyn_cs_wo_phi[:,2] = 1
            M5_dyn_cs.update(drones_train_normal_dyn_cs,zwind_train_detrended_dyn_cs.T)

            lt = np.log(51.026823559394558)
            lz = np.log(0.01*13.054954891415182)
            l_phi = np.log(23.025993674634258)
            lr = np.log(40.199201579845884)
            sigma2 = 0.5*np.log(0.84069334459964384)
            noise_var =0.5*np.log(noise[j]**2)

            params = np.array([lt,lz,l_phi,lr,sigma2,noise_var]).reshape(1,-1)
            M5_dyn_cs.set_params(params)

            ## Values in log scale
            input_opt = {"gp_opt_python":True,'gp_opt_iter':10,'gp_hyper_bounds':((-10, 10), (-10, 10), (-10, 10), (-10, 10),
                             (-10, 10),
                             (-10, 10)),'gp_discr_tol':-1}
            M5_dyn_cs.optimize(input_opt)

            params5_dyn_cs = np.asarray(M5_dyn_cs.get_params()).reshape(6)
            predictions5_dyn_cs = np.ndarray((1,2,140*150))

            #grid_unrolled_normal_dyn_cs_wo_t = grid_unrolled_normal_dyn_cs.copy()
            #grid_unrolled_normal_dyn_cs_wo_t[:,0] =1
            #grid_unrolled_normal_dyn_cs_wo_phi = grid_unrolled_normal_dyn_cs.copy()
            #grid_unrolled_normal_dyn_cs_wo_phi[:,2] = 1
            M5_dyn_cs.predict(grid_unrolled_normal_dyn_cs,predictions5_dyn_cs)

            mean_pred5_dyn_cs = predictions5_dyn_cs[0,0].reshape(-1,1)
            var_pred5_dyn_cs = predictions5_dyn_cs[0,1].reshape(-1,1)

            mean_pred5_dyn_cs = mean_pred5_dyn_cs + (zwind_com*interpolate_rtrend(grid_unrolled_normal_dyn_cs[:,3],'linear')).reshape((-1,1))

            mean_pred5_dyn_cs = mean_pred5_dyn_cs.reshape((140,150))
            var_pred5_dyn_cs = var_pred5_dyn_cs.reshape((140,150))

            mean_std_pred_incloud15_dyn_cs = np.mean(np.sqrt(var_pred5_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5]))
            rmse_incloud15_dyn_cs = np.sqrt(np.mean((zwind_data1[76,zind_rel[zi]][lwc_cloud1[76,zind_rel[zi]]>=1e-5]-mean_pred5_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5])**2))

            abs_err_incloud15_dyn_cs = np.abs(zwind_data1[76,zind_rel[zi]][lwc_cloud1[76,zind_rel[zi]]>=1e-5]-mean_pred5_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5])
            std_pred_incloud15_dyn_cs = np.sqrt(var_pred5_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5])
            test_std_soundness_incloud15_dyn_cs = (std_pred_incloud15_dyn_cs-abs_err_incloud15_dyn_cs)/std_pred_incloud15_dyn_cs

            std_pred_soundness_incloud15_dyn_cs = np.array([np.percentile(test_std_soundness_incloud15_dyn_cs,0.3),np.percentile(test_std_soundness_incloud15_dyn_cs,5),
            np.percentile(test_std_soundness_incloud15_dyn_cs,32)])


            explained_variance = np.sum((zwind_data1[76,zind_rel[zi]][lwc_cloud1[76,zind_rel[zi]]>=1e-5] -
                                mean_pred5_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5])**2)
            total_variance = np.sum((zwind_data1[76,zind_rel[zi]][lwc_cloud1[76,zind_rel[zi]]>=1e-5]- zwind_train_dyn_cs.mean())**2)
            r2_incloud15_dyn_cs = 1 - explained_variance/total_variance

            err_incloud15_dyn_cs = zwind_data1[76,zind_rel[zi]][lwc_cloud1[76,zind_rel[zi]]>=1e-5]-mean_pred5_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5]
            err_stats_incloud15_dyn_cs = np.array([np.mean(err_incloud15_dyn_cs),np.median(err_incloud15_dyn_cs),np.std(err_incloud15_dyn_cs),skew(err_incloud15_dyn_cs)])

            ########################## GPy implementation normalized,Exponential uncorrected phi ############################
            try:
                kernel = GPy.kern.Exponential(input_dim=3,active_dims=[0,2,3],ARD=True)

                m_dyn_cs = GPy.models.GPRegression(drones_train_normal_dyn_cs,zwind_train_detrended_dyn_cs,kernel)
                m_dyn_cs.optimize_restarts(num_restarts = 10)

                params6_dyn_cs = np.nan*np.ndarray(6)
                params6_dyn_cs[[4,0,2,3,5]] = m_dyn_cs.param_array
                predictions6_dyn_cs = m_dyn_cs.predict(grid_unrolled_normal_dyn_cs)

                mean_pred6_dyn_cs = predictions6_dyn_cs[0]
                var_pred6_dyn_cs = predictions6_dyn_cs[1]

                mean_pred6_dyn_cs = mean_pred6_dyn_cs + (zwind_com*interpolate_rtrend(grid_unrolled_normal_dyn_cs[:,3],'linear')).reshape((-1,1))

                mean_pred6_dyn_cs = mean_pred6_dyn_cs.reshape((140,150))
                var_pred6_dyn_cs = var_pred6_dyn_cs.reshape((140,150))

                mean_std_pred_incloud16_dyn_cs = np.mean(np.sqrt(var_pred6_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5]))
                rmse_incloud16_dyn_cs = np.sqrt(np.mean((zwind_data1[76,zind_rel[zi]][lwc_cloud1[76,zind_rel[zi]]>=1e-5]-mean_pred6_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5])**2))

                abs_err_incloud16_dyn_cs = np.abs(zwind_data1[76,zind_rel[zi]][lwc_cloud1[76,zind_rel[zi]]>=1e-5]-mean_pred6_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5])
                std_pred_incloud16_dyn_cs = np.sqrt(var_pred6_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5])
                test_std_soundness_incloud16_dyn_cs = (std_pred_incloud16_dyn_cs-abs_err_incloud16_dyn_cs)/std_pred_incloud16_dyn_cs

                std_pred_soundness_incloud16_dyn_cs = np.array([np.percentile(test_std_soundness_incloud16_dyn_cs,0.3),np.percentile(test_std_soundness_incloud16_dyn_cs,5),
                np.percentile(test_std_soundness_incloud16_dyn_cs,32)])


                explained_variance = np.sum((zwind_data1[76,zind_rel[zi]][lwc_cloud1[76,zind_rel[zi]]>=1e-5] -
                                    mean_pred6_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5])**2)
                total_variance = np.sum((zwind_data1[76,zind_rel[zi]][lwc_cloud1[76,zind_rel[zi]]>=1e-5]- zwind_train_dyn_cs.mean())**2)
                r2_incloud16_dyn_cs = 1 - explained_variance/total_variance

                err_incloud16_dyn_cs = zwind_data1[76,zind_rel[zi]][lwc_cloud1[76,zind_rel[zi]]>=1e-5]-mean_pred6_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5]
                err_stats_incloud16_dyn_cs = np.array([np.mean(err_incloud16_dyn_cs),np.median(err_incloud16_dyn_cs),np.std(err_incloud16_dyn_cs),skew(err_incloud16_dyn_cs)])
                ########################## GPy implementation RBF, t,x,y ############################
                kernel = GPy.kern.RBF(input_dim=3,active_dims=[0,2,3],ARD=True)

                m2_dyn_cs = GPy.models.GPRegression(drones_train_dyn_cs,zwind_train_dyn_cs,kernel)
                m2_dyn_cs.optimize_restarts(num_restarts = 10)

                params7_dyn_cs = np.nan*np.ndarray(6)
                params7_dyn_cs[[4,0,2,3,5]] = m2_dyn_cs.param_array

                predictions7_dyn_cs = m2_dyn_cs.predict(grid_unrolled_dyn_cs)

                mean_pred7_dyn_cs = predictions7_dyn_cs[0]
                var_pred7_dyn_cs = predictions7_dyn_cs[1]

                mean_pred7_dyn_cs = mean_pred7_dyn_cs.reshape((140,150))
                var_pred7_dyn_cs = var_pred7_dyn_cs.reshape((140,150))

                mean_std_pred_incloud17_dyn_cs = np.mean(np.sqrt(var_pred7_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5]))
                rmse_incloud17_dyn_cs = np.sqrt(np.mean((zwind_data1[76,zind_rel[zi]][lwc_cloud1[76,zind_rel[zi]]>=1e-5]-mean_pred7_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5])**2))

                abs_err_incloud17_dyn_cs = np.abs(zwind_data1[76,zind_rel[zi]][lwc_cloud1[76,zind_rel[zi]]>=1e-5]-mean_pred7_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5])
                std_pred_incloud17_dyn_cs = np.sqrt(var_pred7_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5])
                test_std_soundness_incloud17_dyn_cs = (std_pred_incloud17_dyn_cs-abs_err_incloud17_dyn_cs)/std_pred_incloud17_dyn_cs

                std_pred_soundness_incloud17_dyn_cs = np.array([np.percentile(test_std_soundness_incloud17_dyn_cs,0.3),np.percentile(test_std_soundness_incloud17_dyn_cs,5),
                np.percentile(test_std_soundness_incloud17_dyn_cs,32)])


                explained_variance = np.sum((zwind_data1[76,zind_rel[zi]][lwc_cloud1[76,zind_rel[zi]]>=1e-5] -
                                    mean_pred7_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5])**2)
                total_variance = np.sum((zwind_data1[76,zind_rel[zi]][lwc_cloud1[76,zind_rel[zi]]>=1e-5]- zwind_train_dyn_cs.mean())**2)
                r2_incloud17_dyn_cs = 1 - explained_variance/total_variance

                err_incloud17_dyn_cs = zwind_data1[76,zind_rel[zi]][lwc_cloud1[76,zind_rel[zi]]>=1e-5]-mean_pred7_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5]
                err_stats_incloud17_dyn_cs = np.array([np.mean(err_incloud17_dyn_cs),np.median(err_incloud17_dyn_cs),np.std(err_incloud17_dyn_cs),skew(err_incloud17_dyn_cs)])

            except KeyboardInterrupt:
                print('KeyboardInterrupt')
                break
            except:
                print('Something went wrong with LinAlg, please continue')
            ########################################### libgp, SE, xy detrended
            M6_dyn_cs = GPModel(ndim=4,nvar=1, kernel_string=b"CovSum(CovSEard, CovNoise)")
            M6_dyn_cs.update(drones_train_dyn_cs,zwind_train_detrended_dyn_cs.T)

            ## Values in log scale
            input_opt = {"gp_opt_python":True,'gp_opt_iter':10,'gp_hyper_bounds':((-10, 10), (-10, 10), (-10, 10), (-10, 10),
                             (-10, 10),
                             (-10, 10)),'gp_discr_tol':-1}

            M6_dyn_cs.optimize(input_opt)
            params8_dyn_cs = np.asarray(M6_dyn_cs.get_params()).reshape(6)
            predictions8_dyn_cs = np.ndarray((1,2,140*150))

            M6_dyn_cs.predict(grid_unrolled_dyn_cs,predictions8_dyn_cs)

            mean_pred8_dyn_cs = predictions8_dyn_cs[0,0].reshape(-1,1)
            var_pred8_dyn_cs = predictions8_dyn_cs[0,1].reshape(-1,1)

            mean_pred8_dyn_cs = mean_pred8_dyn_cs + zwind_com*interpolate_rtrend(grid_unrolled_normal_dyn_cs[:,3],'linear').reshape((-1,1))

            mean_pred8_dyn_cs = mean_pred8_dyn_cs.reshape((140,150))
            var_pred8_dyn_cs = var_pred8_dyn_cs.reshape((140,150))

            mean_std_pred_incloud18_dyn_cs = np.mean(np.sqrt(var_pred8_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5]))
            rmse_incloud18_dyn_cs = np.sqrt(np.mean((zwind_data1[76,zind_rel[zi]][lwc_cloud1[76,zind_rel[zi]]>=1e-5]-mean_pred8_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5])**2))

            abs_err_incloud18_dyn_cs = np.abs(zwind_data1[76,zind_rel[zi]][lwc_cloud1[76,zind_rel[zi]]>=1e-5]-mean_pred8_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5])
            std_pred_incloud18_dyn_cs = np.sqrt(var_pred8_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5])
            test_std_soundness_incloud18_dyn_cs = (std_pred_incloud18_dyn_cs-abs_err_incloud18_dyn_cs)/std_pred_incloud18_dyn_cs

            std_pred_soundness_incloud18_dyn_cs = np.array([np.percentile(test_std_soundness_incloud18_dyn_cs,0.3),np.percentile(test_std_soundness_incloud18_dyn_cs,5),
            np.percentile(test_std_soundness_incloud18_dyn_cs,32)])


            explained_variance = np.sum((zwind_data1[76,zind_rel[zi]][lwc_cloud1[76,zind_rel[zi]]>=1e-5] -
                                mean_pred8_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5])**2)
            total_variance = np.sum((zwind_data1[76,zind_rel[zi]][lwc_cloud1[76,zind_rel[zi]]>=1e-5]- zwind_train_dyn_cs.mean())**2)
            r2_incloud18_dyn_cs = 1 - explained_variance/total_variance


            err_incloud18_dyn_cs = zwind_data1[76,zind_rel[zi]][lwc_cloud1[76,zind_rel[zi]]>=1e-5]-mean_pred8_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5]
            err_stats_incloud18_dyn_cs = np.array([np.mean(err_incloud18_dyn_cs),np.median(err_incloud18_dyn_cs),np.std(err_incloud18_dyn_cs),skew(err_incloud18_dyn_cs)])
            ########################################### libgp Exp xy, detrended data
            M7_dyn_cs = GPModel(ndim=4,nvar=1, kernel_string=b"CovSum(CovExpArd, CovNoise)")
            M7_dyn_cs.update(drones_train_dyn_cs,zwind_train_detrended_dyn_cs.T)

            ## Values in log scale
            input_opt = {"gp_opt_python":True,'gp_opt_iter':10,'gp_hyper_bounds':((-10, 10), (-10, 10), (-10, 10), (-10, 10),
                             (-10, 10),
                             (-10, 10)),'gp_discr_tol':-1}

            M7_dyn_cs.optimize(input_opt)
            params9_dyn_cs = np.asarray(M7_dyn_cs.get_params()).reshape(6)
            predictions9_dyn_cs = np.ndarray((1,2,140*150))

            M7_dyn_cs.predict(grid_unrolled_dyn_cs,predictions9_dyn_cs)

            mean_pred9_dyn_cs = predictions9_dyn_cs[0,0].reshape(-1,1)
            var_pred9_dyn_cs = predictions9_dyn_cs[0,1].reshape(-1,1)

            mean_pred9_dyn_cs = mean_pred9_dyn_cs + zwind_com*interpolate_rtrend(grid_unrolled_normal_dyn_cs[:,3],'linear').reshape((-1,1))

            mean_pred9_dyn_cs = mean_pred9_dyn_cs.reshape((140,150))
            var_pred9_dyn_cs = var_pred9_dyn_cs.reshape((140,150))

            mean_std_pred_incloud19_dyn_cs = np.mean(np.sqrt(var_pred9_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5]))
            rmse_incloud19_dyn_cs = np.sqrt(np.mean((zwind_data1[76,zind_rel[zi]][lwc_cloud1[76,zind_rel[zi]]>=1e-5]-mean_pred9_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5])**2))

            abs_err_incloud19_dyn_cs = np.abs(zwind_data1[76,zind_rel[zi]][lwc_cloud1[76,zind_rel[zi]]>=1e-5]-mean_pred9_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5])
            std_pred_incloud19_dyn_cs = np.sqrt(var_pred9_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5])
            test_std_soundness_incloud19_dyn_cs = (std_pred_incloud19_dyn_cs-abs_err_incloud19_dyn_cs)/std_pred_incloud19_dyn_cs

            std_pred_soundness_incloud19_dyn_cs = np.array([np.percentile(test_std_soundness_incloud19_dyn_cs,0.3),np.percentile(test_std_soundness_incloud19_dyn_cs,5),
            np.percentile(test_std_soundness_incloud19_dyn_cs,32)])


            explained_variance = np.sum((zwind_data1[76,zind_rel[zi]][lwc_cloud1[76,zind_rel[zi]]>=1e-5] -
                                mean_pred9_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5])**2)
            total_variance = np.sum((zwind_data1[76,zind_rel[zi]][lwc_cloud1[76,zind_rel[zi]]>=1e-5]- zwind_train_dyn_cs.mean())**2)
            r2_incloud19_dyn_cs = 1 - explained_variance/total_variance


            err_incloud19_dyn_cs = zwind_data1[76,zind_rel[zi]][lwc_cloud1[76,zind_rel[zi]]>=1e-5]-mean_pred9_dyn_cs[lwc_cloud1[76,zind_rel[zi]]>=1e-5]
            err_stats_incloud19_dyn_cs = np.array([np.mean(err_incloud19_dyn_cs),np.median(err_incloud19_dyn_cs),np.std(err_incloud19_dyn_cs),skew(err_incloud19_dyn_cs)])


            ###################################### Saving results ############################################

            all_rmse_dyn_cs[zi,i,j] =np.array([rmse_incloud11_dyn_cs, rmse_incloud12_dyn_cs, rmse_incloud13_dyn_cs, rmse_incloud14_dyn_cs,
            rmse_incloud15_dyn_cs, rmse_incloud16_dyn_cs, rmse_incloud17_dyn_cs,rmse_trend_dyn_cs, rmse_mean_dyn_cs,rmse_incloud18_dyn_cs,rmse_incloud19_dyn_cs ])


            all_r2_dyn_cs[zi,i,j] = np.array([r2_incloud11_dyn_cs,r2_incloud12_dyn_cs,r2_incloud13_dyn_cs,r2_incloud14_dyn_cs,
            r2_incloud15_dyn_cs, r2_incloud16_dyn_cs,r2_incloud17_dyn_cs, r2_trend_dyn_cs, r2_mean_dyn_cs,r2_incloud18_dyn_cs,r2_incloud19_dyn_cs])


            all_std_pred_dyn_cs[zi,i,j] = np.array([mean_std_pred_incloud11_dyn_cs, mean_std_pred_incloud12_dyn_cs, mean_std_pred_incloud13_dyn_cs,
            mean_std_pred_incloud14_dyn_cs, mean_std_pred_incloud15_dyn_cs, mean_std_pred_incloud16_dyn_cs, mean_std_pred_incloud17_dyn_cs,
            mean_std_pred_incloud18_dyn_cs,mean_std_pred_incloud19_dyn_cs])

            all_std_pred_soundness_dyn_cs[zi,i,j] = np.array([std_pred_soundness_incloud11_dyn_cs, std_pred_soundness_incloud12_dyn_cs, std_pred_soundness_incloud13_dyn_cs,
            std_pred_soundness_incloud14_dyn_cs, std_pred_soundness_incloud15_dyn_cs, std_pred_soundness_incloud16_dyn_cs, std_pred_soundness_incloud17_dyn_cs,
            std_pred_soundness_incloud18_dyn_cs,std_pred_soundness_incloud19_dyn_cs])


            all_hypers_dyn_cs[zi,i,j] = np.array([params1_dyn_cs,params2_dyn_cs,params3_dyn_cs,params4_dyn_cs,params5_dyn_cs,params6_dyn_cs,params7_dyn_cs,
            params8_dyn_cs,params9_dyn_cs])

            all_error_stats_dyn_cs[zi,i,j] = np.array([err_stats_incloud11_dyn_cs,err_stats_incloud12_dyn_cs, err_stats_incloud13_dyn_cs,
            err_stats_incloud14_dyn_cs, err_stats_incloud15_dyn_cs, err_stats_incloud16_dyn_cs,err_stats_incloud17_dyn_cs,err_stats_trend_dyn_cs,err_stats_mean_dyn_cs,
            err_stats_incloud18_dyn_cs,err_stats_incloud19_dyn_cs])

            outfile = '/home/dselle/Skyscanner/data_exploration/results/TestGP/dump_results_tests_cs_gp2.npz'
            np.savez(outfile, all_rmse_static_cs=all_rmse_static_cs, all_r2_static_cs=all_r2_static_cs,all_std_pred_static_cs=all_std_pred_static_cs,
            all_std_pred_soundness_static_cs=all_std_pred_soundness_static_cs, all_len_train_data_static_cs = all_len_train_data_static_cs,
            all_hypers_static_cs = all_hypers_static_cs, all_error_stats_static_cs = all_error_stats_static_cs, all_vars_static_cs = all_vars_static_cs,
            all_rmse_dyn_cs=all_rmse_dyn_cs,all_r2_dyn_cs=all_r2_dyn_cs,all_std_pred_dyn_cs = all_std_pred_dyn_cs,
            all_std_pred_soundness_dyn_cs = all_std_pred_soundness_dyn_cs, all_len_train_data_dyn_cs = all_len_train_data_dyn_cs,
            all_hypers_dyn_cs = all_hypers_dyn_cs, all_error_stats_dyn_cs=all_error_stats_dyn_cs, all_vars_dyn_cs=all_vars_dyn_cs)

            print('Total progress:{}%'.format((80*(2*zi+1)+4*(i)+(j+1))/1360*100))

            time2 = datetime.datetime.now()
            print(time2-time1)
########################
#### End of CS loop ####
########################

#####################################################################################
################# Plots for Statics CS, 1 trial, 1 noise ############################
#####################################################################################
zi = 2
tstart_circle = 449
tend_circle = 526
radius_circle = 0.09

drone1_circle = np.ndarray((0,4))
xcenter = 1.1
ycenter = 2.1
for t in np.arange(tstart_circle,tend_circle):
    trel = t - tstart_circle
    xtemp = xcenter + radius_circle*np.cos(15/90*trel)
    ytemp = ycenter + radius_circle*np.sin(15/90*trel)
    temp = np.array([453,zr[zind_rel[zi]],xtemp,ytemp])
    incloud1 = interpolate_points_cloud1(temp,'nearest')
    if incloud1 == 1.0:
        drone1_circle = np.vstack((drone1_circle,temp))

drone2_circle = np.ndarray((0,4))
xcenter = 1.4
ycenter = 2.06
for t in np.arange(tstart_circle,tend_circle):
    trel = t - tstart_circle
    xtemp = xcenter + radius_circle*np.cos(15/90*trel)
    ytemp = ycenter + radius_circle*np.sin(15/90*trel)
    temp = np.array([453,zr[zind_rel[zi]],xtemp,ytemp])
    incloud1 = interpolate_points_cloud1(temp,'nearest')
    if incloud1 == 1.0:
        drone2_circle = np.vstack((drone2_circle,temp))

drone3_circle = np.ndarray((0,4))
xcenter = 1.23
ycenter = 1.9
for t in np.arange(tstart_circle,tend_circle):
    trel = t - tstart_circle
    xtemp = xcenter + radius_circle*np.cos(15/90*trel)
    ytemp = ycenter + radius_circle*np.sin(15/90*trel)
    temp = np.array([453,zr[zind_rel[zi]],xtemp,ytemp])
    incloud1 = interpolate_points_cloud1(temp,'nearest')
    if incloud1 == 1.0:
        drone3_circle = np.vstack((drone3_circle,temp))

drone4_circle = np.ndarray((0,4))
xcenter = 1.35
ycenter = 1.9
for t in np.arange(tstart_circle,tend_circle):
    trel = t - tstart_circle
    xtemp = xcenter + radius_circle*np.cos(15/90*trel)
    ytemp = ycenter + radius_circle*np.sin(15/90*trel)
    temp = np.array([453,zr[zind_rel[zi]],xtemp,ytemp])
    incloud1 = interpolate_points_cloud1(temp,'nearest')
    if incloud1 == 1.0:
        drone4_circle = np.vstack((drone4_circle,temp))

drone5_circle = np.ndarray((0,4))
xcenter = 1.35
ycenter = 1.7
for t in np.arange(tstart_circle,tend_circle):
    trel = t - tstart_circle
    xtemp = xcenter + radius_circle*np.cos(15/90*trel)
    ytemp = ycenter + radius_circle*np.sin(15/90*trel)
    temp = np.array([453,zr[zind_rel[zi]],xtemp,ytemp])
    incloud1 = interpolate_points_cloud1(temp,'nearest')
    if incloud1 == 1.0:
        drone5_circle = np.vstack((drone5_circle,temp))

################### Train and Test Data x,y,normalized GPs, does not change with noise and trials loops
drones_train = np.vstack((drone1_circle,drone2_circle,drone3_circle,drone4_circle,drone5_circle))
grid_unrolled = grid[4,zind_rel[zi]].reshape((-1,4))
all_len_train_data_static_cs[zi] = len(drones_train)

all_vars_static_cs[zi] = zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5].var()

drones_train_normal = normalize(lwc_cloud1_polar,drones_train,cloud1.COM_2D_lwc_tz,449,1.185,origin_xy)
grid_unrolled_normal = normalize(lwc_cloud1_polar,grid_unrolled,cloud1.COM_2D_lwc_tz,449,1.185,origin_xy)
interpolate_rtrend = RegularGridInterpolator(points=np.arange(0,151).reshape(1,151),values=rtrend_global_median_updraft_norm,bounds_error=False,fill_value=0)
COM =np.array([449,zr[zind_rel[zi]],(cloud1.COM_2D_lwc_tz[0,zind_rel[zi],0] + origin_xy[0])*0.01,(cloud1.COM_2D_lwc_tz[0,zind_rel[zi],1] + origin_xy[1])*0.01])
####### Trials to see effect of randomness in noise:
#trials = np.arange(0,20)
#trials = np.arange(0,20)
#for i in trials:
############## Loop to see Effect if Noise variance
#noise = np.array([1e-3,0.1,0.25,0.5,0.75])
font = {'size'   : 22}

plt.rc('font', **font)

noise = np.array([0.75])
for j in range(len(noise)):
    #Training Data dependent on noise
    print('progress Static CS: Start of CS={},Trial={},Noise={}'.format(zi,i,j))
    time1 = datetime.datetime.now()
    zwind_train = atm.get_points(drones_train,'WT','linear')
    zwind_train = zwind_train.reshape((len(zwind_train),1))
    zwind_com = atm.get_points(COM,'WT','linear') + np.random.randn(1)*noise[j]
    zwind_train = zwind_train + np.random.randn(len(zwind_train),1)*noise[j]
    zwind_train_detrended = zwind_train - (zwind_com*interpolate_rtrend(drones_train_normal[:,3],'linear')).reshape((-1,1))

    rmse_trend = np.sqrt(np.mean((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5] -
    zwind_com*interpolate_rtrend(grid_unrolled_normal[:,3],'linear').reshape((140,150))[lwc_cloud1[4,zind_rel[zi]]>=1e-5])**2))

    explained_variance = np.sum((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5] -
    zwind_com*interpolate_rtrend(grid_unrolled_normal[:,3],'linear').reshape((140,150))[lwc_cloud1[4,zind_rel[zi]]>=1e-5])**2)
    total_variance = np.sum((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]- zwind_train.mean())**2)
    r2_trend = 1 - explained_variance/total_variance

    err_trend = (zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5] -
    zwind_com*interpolate_rtrend(grid_unrolled_normal[:,3],'linear').reshape((140,150))[lwc_cloud1[4,zind_rel[zi]]>=1e-5])
    err_stats_trend = np.array([np.mean(err_trend),np.median(err_trend),np.std(err_trend),skew(err_trend)])


    rmse_mean =np.sqrt(np.mean((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]- zwind_train.mean())**2))
    r2_mean = 0

    err_mean = zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]- zwind_train.mean()
    err_stats_mean = np.array([np.mean(err_mean),np.median(err_mean),np.std(err_mean),skew(err_mean)])

    trend_pred = zwind_com*interpolate_rtrend(grid_unrolled_normal[:,3],'linear').reshape((140,150))
    err_trend_total = zwind_data1[4,zind_rel[zi]] - zwind_com*interpolate_rtrend(grid_unrolled_normal[:,3],'linear').reshape((140,150))

    vmin = np.min(zwind_data1[4,zind_rel[zi],:,30:])
    vmax = np.max(zwind_data1[4,zind_rel[zi],:,30:])
    ####### Plotting Results of trend

    plt.figure()
    plt.imshow(lwc_cloud1[36,0].T,origin='lower',extent=[xr[0], xr[-1], yr[0], yr[-1]])
    plt.title("Points inside of cloud, z=1.185 km,t={}s".format(grid[36,0,0,0,0]))
    plt.xlim(xr[0], xr[-1])
    plt.ylim(yr[0], yr[-1])
    plt.plot(drones_train[:,2],drones_train[:,3],'k.',alpha=0.3)
    plt.xlabel('x coordinate(km)')
    plt.ylabel('y coordinate(km)')

    f, (ax1, ax2, ax3) = plt.subplots(1, 3)

    f.suptitle('Trend, Static CS:{} km, noise_std:{} m/s,rmse in cloud:{} m/s,zwind_com:{} m/s'.format(np.round(float(zr[zind_rel[zi]]),3),noise[j],
    np.round(rmse_trend,3),np.round(float(zwind_com),3)))

    im1 = ax1.imshow(zwind_data1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],vmin=vmin,vmax=vmax)
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    cbar1 = plt.colorbar(im1, cax=cax1)
    ax1.contour(lwc_cloud1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
    ax1.set_xlim(xr[0], xr[-1])
    ax1.set_ylim(yr[30], yr[-1])
    ax1.plot(drones_train[:,2],drones_train[:,3],'k.',alpha=0.3)
    ax1.set_xlabel('x coordinate (km)')
    ax1.set_ylabel('y coordinate(km)')
    ax1.set_title('Ground truth')

    im2 = ax2.imshow(trend_pred[:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],vmin=vmin,vmax=vmax)
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.1)
    cbar2 = plt.colorbar(im2, cax=cax2)
    ax2.contour(lwc_cloud1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
    ax2.set_xlim(xr[0], xr[-1])
    ax2.set_ylim(yr[30], yr[-1])
    ax2.plot(drones_train[:,2],drones_train[:,3],'k.',alpha=0.3)
    ax2.set_xlabel('x coordinate (km)')
    ax2.set_ylabel('y coordinate(km)')
    ax2.set_title('Prediction of trend')

    im3 = ax3.imshow(err_trend_total[:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],vmin=-1.5,vmax=1.5)
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="5%", pad=0.1)
    cbar3 = plt.colorbar(im3, cax=cax3)
    ax3.contour(lwc_cloud1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
    ax3.set_xlim(xr[0], xr[-1])
    ax3.set_ylim(yr[30], yr[-1])
    ax3.plot(drones_train[:,2],drones_train[:,3],'k.',alpha=0.3)
    ax3.set_xlabel('x coordinate (km)')
    ax3.set_ylabel('y coordinate(km)')
    ax3.set_title('Prediction error of trend')

    ########################### skyscan.lib GP Model,Exponential x,y ########################
    M = GPModel(ndim=4,nvar=1, kernel_string=b"CovSum(CovExpArd, CovNoise)")
    M.update(drones_train,zwind_train.T)

    ## Values in log scale
    input_opt = {"gp_opt_python":True,'gp_opt_iter':10,'gp_hyper_bounds':((-10, 10), (-10, 10), (-10, 10), (-10, 10),
                     (-10, 10),
                     (-10, 10)),'gp_discr_tol':-1}
    M.optimize(input_opt)
    params1 = np.asarray(M.get_params()).reshape(6)

    predictions = np.ndarray((1,2,140*150))
    M.predict(grid_unrolled,predictions)

    mean_pred1 = predictions[0,0]
    var_pred1 = predictions[0,1]

    mean_pred1 = mean_pred1.reshape((140,150))
    var_pred1 = var_pred1.reshape((140,150))

    mean_std_pred_incloud11 = np.mean(np.sqrt(var_pred1[lwc_cloud1[4,zind_rel[zi]]>=1e-5]))
    rmse_incloud11 = np.sqrt(np.mean((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]-mean_pred1[lwc_cloud1[4,zind_rel[zi]]>=1e-5])**2))

    abs_err_incloud11 = np.abs(zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]-mean_pred1[lwc_cloud1[4,zind_rel[zi]]>=1e-5])
    std_pred_incloud11 = np.sqrt(var_pred1[lwc_cloud1[4,zind_rel[zi]]>=1e-5])
    test_std_soundness_incloud11 = (std_pred_incloud11-abs_err_incloud11)/std_pred_incloud11

    std_pred_soundness_incloud11 = np.array([np.percentile(test_std_soundness_incloud11,0.3),np.percentile(test_std_soundness_incloud11,5),
    np.percentile(test_std_soundness_incloud11,32)])


    explained_variance = np.sum((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5] -
                        mean_pred1[lwc_cloud1[4,zind_rel[zi]]>=1e-5])**2)
    total_variance = np.sum((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]- zwind_train.mean())**2)
    r2_incloud11 = 1 - explained_variance/total_variance

    err_incloud11 = zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]-mean_pred1[lwc_cloud1[4,zind_rel[zi]]>=1e-5]
    err_stats_incloud11 = np.array([np.mean(err_incloud11),np.median(err_incloud11),np.std(err_incloud11),skew(err_incloud11)])


    err_incloud11_total = zwind_data1[4,zind_rel[zi]]-mean_pred1

    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4,sharey=True)
    f.suptitle('Exp., Static CS:{}km, noise_std:{}m/s,rmse in cloud:{}m/s'.format(np.round(float(zr[zind_rel[zi]]),3),noise[j],np.round(rmse_incloud11,3)))
    im1 = ax1.imshow(zwind_data1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],vmin=vmin,vmax=vmax)
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cbar1 = plt.colorbar(im1, cax=cax1)
    ax1.contour(lwc_cloud1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
    ax1.set_xlim(xr[0], xr[-1])
    ax1.set_ylim(yr[30], yr[-1])
    ax1.plot(drones_train[:,2],drones_train[:,3],'k.',alpha=0.3)
    ax1.set_xlabel('x coordinate (km)')
    ax1.set_ylabel('y coordinate(km)')
    ax1.set_title('Ground truth')

    im2 = ax2.imshow(mean_pred1[:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],vmin=vmin,vmax=vmax)
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    cbar2 = plt.colorbar(im2, cax=cax2)
    ax2.contour(lwc_cloud1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
    ax2.set_xlim(xr[0], xr[-1])
    ax2.set_ylim(yr[30], yr[-1])
    ax2.plot(drones_train[:,2],drones_train[:,3],'k.',alpha=0.3)
    ax2.set_xlabel('x coordinate (km)')
    ax2.set_ylabel('y coordinate(km)')
    ax2.set_title('Predicted mean $y_{\star}$')

    im3 = ax3.imshow(np.sqrt(var_pred1[:,30:].T),origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],vmin=0,vmax=2)
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="5%", pad=0.05)
    cbar3 = plt.colorbar(im3, cax=cax3)
    ax3.contour(lwc_cloud1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
    ax3.set_xlim(xr[0], xr[-1])
    ax3.set_ylim(yr[30], yr[-1])
    ax3.plot(drones_train[:,2],drones_train[:,3],'k.',alpha=0.3)
    ax3.set_xlabel('x coordinate (km)')
    ax3.set_ylabel('y coordinate(km)')
    ax3.set_title('Predicted $\sqrt{V[y_{\star}]}$')

    im4 = ax4.imshow(err_incloud11_total[:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],vmin=-1.5,vmax=1.5)
    divider4 = make_axes_locatable(ax4)
    cax4 = divider4.append_axes("right", size="5%", pad=0.05)
    cbar4 = plt.colorbar(im4, cax=cax4)
    ax4.contour(lwc_cloud1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
    ax4.set_xlim(xr[0], xr[-1])
    ax4.set_ylim(yr[30], yr[-1])
    ax4.plot(drones_train[:,2],drones_train[:,3],'k.',alpha=0.3)
    ax4.set_xlabel('x coordinate (km)')
    ax4.set_ylabel('y coordinate(km)')
    ax4.set_title('Prediction error')
    ########################### skyscan.lib GP Model,SE x,y ########################
    M2 = GPModel(ndim=4,nvar=1, kernel_string=b"CovSum(CovSEard, CovNoise)")
    M2.update(drones_train,zwind_train.T)

    ## Values in log scale
    input_opt = {"gp_opt_python":True,'gp_opt_iter':10,'gp_hyper_bounds':((-10, 10), (-10, 10), (-10, 10), (-10, 10),
                     (-10, 10),
                     (-10, 10)),'gp_discr_tol':-1}
    M2.optimize(input_opt)
    params2 = np.asarray(M2.get_params()).reshape(6)

    predictions2 = np.ndarray((1,2,140*150))
    M2.predict(grid_unrolled,predictions2)

    mean_pred2 = predictions2[0,0]
    var_pred2 = predictions2[0,1]

    mean_pred2 = mean_pred2.reshape((140,150))
    var_pred2 = var_pred2.reshape((140,150))

    mean_std_pred_incloud12 = np.mean(np.sqrt(var_pred2[lwc_cloud1[4,zind_rel[zi]]>=1e-5]))
    rmse_incloud12 = np.sqrt(np.mean((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]-mean_pred2[lwc_cloud1[4,zind_rel[zi]]>=1e-5])**2))

    abs_err_incloud12 = np.abs(zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]-mean_pred2[lwc_cloud1[4,zind_rel[zi]]>=1e-5])
    std_pred_incloud12 = np.sqrt(var_pred2[lwc_cloud1[4,zind_rel[zi]]>=1e-5])
    test_std_soundness_incloud12 = (std_pred_incloud12-abs_err_incloud12)/std_pred_incloud12

    std_pred_soundness_incloud12 = np.array([np.percentile(test_std_soundness_incloud12,0.3),np.percentile(test_std_soundness_incloud12,5),
    np.percentile(test_std_soundness_incloud12,32)])


    explained_variance = np.sum((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5] -
                        mean_pred2[lwc_cloud1[4,zind_rel[zi]]>=1e-5])**2)
    total_variance = np.sum((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]- zwind_train.mean())**2)
    r2_incloud12 = 1 - explained_variance/total_variance

    err_incloud12 = zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]-mean_pred2[lwc_cloud1[4,zind_rel[zi]]>=1e-5]
    err_stats_incloud12 = np.array([np.mean(err_incloud12),np.median(err_incloud12),np.std(err_incloud12),skew(err_incloud12)])

    err_incloud12_total = zwind_data1[4,zind_rel[zi]]-mean_pred2

    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4,sharey=True)
    f.suptitle('Sq. Exp., Static CS:{}km, noise_std:{}m/s,rmse in cloud:{}m/s'.format(np.round(float(zr[zind_rel[zi]]),3),noise[j],np.round(rmse_incloud12,3)))
    im1 = ax1.imshow(zwind_data1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],vmin=vmin,vmax=vmax)
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cbar1 = plt.colorbar(im1, cax=cax1)
    ax1.contour(lwc_cloud1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
    ax1.set_xlim(xr[0], xr[-1])
    ax1.set_ylim(yr[30], yr[-1])
    ax1.plot(drones_train[:,2],drones_train[:,3],'k.',alpha=0.3)
    ax1.set_xlabel('x coordinate (km)')
    ax1.set_ylabel('y coordinate(km)')
    ax1.set_title('Ground truth')

    im2 = ax2.imshow(mean_pred2[:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],vmin=vmin,vmax=vmax)
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    cbar2 = plt.colorbar(im2, cax=cax2)
    ax2.contour(lwc_cloud1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
    ax2.set_xlim(xr[0], xr[-1])
    ax2.set_ylim(yr[30], yr[-1])
    ax2.plot(drones_train[:,2],drones_train[:,3],'k.',alpha=0.3)
    ax2.set_xlabel('x coordinate (km)')
    ax2.set_ylabel('y coordinate(km)')
    ax2.set_title('Predicted mean $y_{\star}$')

    im3 = ax3.imshow(np.sqrt(var_pred2[:,30:].T),origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],vmin=0,vmax=2)
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="5%", pad=0.05)
    cbar3 = plt.colorbar(im3, cax=cax3)
    ax3.contour(lwc_cloud1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
    ax3.set_xlim(xr[0], xr[-1])
    ax3.set_ylim(yr[30], yr[-1])
    ax3.plot(drones_train[:,2],drones_train[:,3],'k.',alpha=0.3)
    ax3.set_xlabel('x coordinate (km)')
    ax3.set_ylabel('y coordinate(km)')
    ax3.set_title('Predicted $\sqrt{V[y_{\star}]}$')

    im4 = ax4.imshow(err_incloud12_total[:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],vmin=-1.5,vmax=1.5)
    divider4 = make_axes_locatable(ax4)
    cax4 = divider4.append_axes("right", size="5%", pad=0.05)
    cbar4 = plt.colorbar(im4, cax=cax4)
    ax4.contour(lwc_cloud1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
    ax4.set_xlim(xr[0], xr[-1])
    ax4.set_ylim(yr[30], yr[-1])
    ax4.plot(drones_train[:,2],drones_train[:,3],'k.',alpha=0.3)
    ax4.set_xlabel('x coordinate (km)')
    ax4.set_ylabel('y coordinate(km)')
    ax4.set_title('Prediction error')








    ###################### libgp, normalized radius and corrected phi,not optimized,variograms hypers #####################

    M3 = GPModel(ndim=4,nvar=1, kernel_string=b"CovSum(CovExpArdPhi, CovNoise)")
    #M3 = GPModel(ndim=4,nvar=1, kernel_string="CovExpArdPhi")
    lt = np.log(51.026823559394558)
    lz = np.log(0.01*13.054954891415182)
    l_phi = np.log(23.025993674634258)
    lr = np.log(40.199201579845884)
    sigma2 = 0.5*np.log(0.84069334459964384)
    noise_var =0.5*np.log(noise[j]**2)

    params = np.array([lt,lz,l_phi,lr,sigma2,noise_var]).reshape(1,-1)
    M3.set_params(params)
    M3.update(drones_train_normal,zwind_train_detrended.T)
    params3 = params.reshape(6)
    predictions3 =np.nan*np.ndarray((1,2,140*150))
    M3.predict(grid_unrolled_normal,predictions3)

    mean_pred3 = predictions3[0,0].reshape(-1,1)
    var_pred3 = predictions3[0,1].reshape(-1,1)

    mean_pred3 = mean_pred3 + zwind_com*interpolate_rtrend(grid_unrolled_normal[:,3],'linear').reshape((-1,1))

    mean_pred3 = mean_pred3.reshape((140,150))
    var_pred3 = var_pred3.reshape((140,150))

    mean_std_pred_incloud13 = np.mean(np.sqrt(var_pred3[lwc_cloud1[4,zind_rel[zi]]>=1e-5]))
    rmse_incloud13 = np.sqrt(np.mean((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]-mean_pred3[lwc_cloud1[4,zind_rel[zi]]>=1e-5])**2))

    abs_err_incloud13 = np.abs(zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]-mean_pred3[lwc_cloud1[4,zind_rel[zi]]>=1e-5])
    std_pred_incloud13 = np.sqrt(var_pred3[lwc_cloud1[4,zind_rel[zi]]>=1e-5])
    test_std_soundness_incloud13 = (std_pred_incloud13-abs_err_incloud13)/std_pred_incloud13

    std_pred_soundness_incloud13 = np.array([np.percentile(test_std_soundness_incloud13,0.3),np.percentile(test_std_soundness_incloud13,5),
    np.percentile(test_std_soundness_incloud13,32)])


    explained_variance = np.sum((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5] -
                        mean_pred3[lwc_cloud1[4,zind_rel[zi]]>=1e-5])**2)
    total_variance = np.sum((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]- zwind_train.mean())**2)
    r2_incloud13 = 1 - explained_variance/total_variance

    err_incloud13 = zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]-mean_pred3[lwc_cloud1[4,zind_rel[zi]]>=1e-5]
    err_stats_incloud13 = np.array([np.mean(err_incloud13),np.median(err_incloud13),np.std(err_incloud13),skew(err_incloud13)])

    err_incloud13_total = zwind_data1[4,zind_rel[zi]]-mean_pred3

    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4,sharey=True)
    f.suptitle('Exp. Norm1, Static CS:{}km, noise_std:{}m/s,rmse in cloud:{}m/s'.format(np.round(float(zr[zind_rel[zi]]),3),noise[j],np.round(rmse_incloud13,3)))
    im1 = ax1.imshow(zwind_data1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],vmin=vmin,vmax=vmax)
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cbar1 = plt.colorbar(im1, cax=cax1)
    ax1.contour(lwc_cloud1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
    ax1.set_xlim(xr[0], xr[-1])
    ax1.set_ylim(yr[30], yr[-1])
    ax1.plot(drones_train[:,2],drones_train[:,3],'k.',alpha=0.3)
    ax1.set_xlabel('x coordinate (km)')
    ax1.set_ylabel('y coordinate(km)')
    ax1.set_title('Ground truth')

    im2 = ax2.imshow(mean_pred3[:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],vmin=vmin,vmax=vmax)
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    cbar2 = plt.colorbar(im2, cax=cax2)
    ax2.contour(lwc_cloud1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
    ax2.set_xlim(xr[0], xr[-1])
    ax2.set_ylim(yr[30], yr[-1])
    ax2.plot(drones_train[:,2],drones_train[:,3],'k.',alpha=0.3)
    ax2.set_xlabel('x coordinate (km)')
    ax2.set_ylabel('y coordinate(km)')
    ax2.set_title('Predicted mean $y_{\star}$')

    im3 = ax3.imshow(np.sqrt(var_pred3[:,30:].T),origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],vmin=0,vmax=2)
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="5%", pad=0.05)
    cbar3 = plt.colorbar(im3, cax=cax3)
    ax3.contour(lwc_cloud1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
    ax3.set_xlim(xr[0], xr[-1])
    ax3.set_ylim(yr[30], yr[-1])
    ax3.plot(drones_train[:,2],drones_train[:,3],'k.',alpha=0.3)
    ax3.set_xlabel('x coordinate (km)')
    ax3.set_ylabel('y coordinate(km)')
    ax3.set_title('Predicted $\sqrt{V[y_{\star}]}$')

    im4 = ax4.imshow(err_incloud13_total[:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],vmin=-1.5,vmax=1.5)
    divider4 = make_axes_locatable(ax4)
    cax4 = divider4.append_axes("right", size="5%", pad=0.05)
    cbar4 = plt.colorbar(im4, cax=cax4)
    ax4.contour(lwc_cloud1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
    ax4.set_xlim(xr[0], xr[-1])
    ax4.set_ylim(yr[30], yr[-1])
    ax4.plot(drones_train[:,2],drones_train[:,3],'k.',alpha=0.3)
    ax4.set_xlabel('x coordinate (km)')
    ax4.set_ylabel('y coordinate(km)')
    ax4.set_title('Prediction error')





    ###################### libgp, normalized radius and corrected phi,optimized #####################
    M4 = GPModel(ndim=4,nvar=1, kernel_string=b"CovSum(CovExpArdPhi, CovNoise)")
    M4.update(drones_train_normal,zwind_train_detrended.T)

    ########## Start optimization at random numbers to avoid optimization to all zeros
    #lt = np.random.randn(1)*3 + 3
    #lz = np.random.randn(1)*3 + 3
    #l_phi = np.random.randn(1)*3 + 3
    #lr = np.random.randn(1)*3 + 3
    #sigma2 = np.random.randn(1)*3 + 3
    #noise_var = np.random.randn(1)*3 + 3

    #params = np.array([lt,lz,l_phi,lr,sigma2,noise_var]).reshape(1,-1)
    #M4.set_params(params)

    ## Values in log scale
    input_opt = {"gp_opt_python":True,'gp_opt_iter':10,'gp_hyper_bounds':((-10, 10), (-10, 10), (-10, 10), (-10, 10),
                     (-10, 10),
                     (-10, 10)),'gp_discr_tol':-1}
    M4.optimize(input_opt)
    params4 = np.asarray(M4.get_params()).reshape(6)
    predictions4 = np.ndarray((1,2,140*150))

    M4.predict(grid_unrolled_normal,predictions4)

    mean_pred4 = predictions4[0,0].reshape(-1,1)
    var_pred4 = predictions4[0,1].reshape(-1,1)

    mean_pred4 = mean_pred4 + zwind_com*interpolate_rtrend(grid_unrolled_normal[:,3],'linear').reshape((-1,1))

    mean_pred4 = mean_pred4.reshape((140,150))
    var_pred4 = var_pred4.reshape((140,150))

    mean_std_pred_incloud14 = np.mean(np.sqrt(var_pred4[lwc_cloud1[4,zind_rel[zi]]>=1e-5]))
    rmse_incloud14 = np.sqrt(np.mean((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]-mean_pred4[lwc_cloud1[4,zind_rel[zi]]>=1e-5])**2))

    abs_err_incloud14 = np.abs(zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]-mean_pred4[lwc_cloud1[4,zind_rel[zi]]>=1e-5])
    std_pred_incloud14 = np.sqrt(var_pred4[lwc_cloud1[4,zind_rel[zi]]>=1e-5])
    test_std_soundness_incloud14 = (std_pred_incloud14-abs_err_incloud14)/std_pred_incloud14

    std_pred_soundness_incloud14 = np.array([np.percentile(test_std_soundness_incloud14,0.3),np.percentile(test_std_soundness_incloud14,5),
    np.percentile(test_std_soundness_incloud14,32)])


    explained_variance = np.sum((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5] -
                        mean_pred4[lwc_cloud1[4,zind_rel[zi]]>=1e-5])**2)
    total_variance = np.sum((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]- zwind_train.mean())**2)
    r2_incloud14 = 1 - explained_variance/total_variance

    err_incloud14 = zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]-mean_pred4[lwc_cloud1[4,zind_rel[zi]]>=1e-5]
    err_stats_incloud14 = np.array([np.mean(err_incloud14),np.median(err_incloud14),np.std(err_incloud14),skew(err_incloud14)])

    err_incloud14_total = zwind_data1[4,zind_rel[zi]]-mean_pred4

    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4,sharey=True)

    f.suptitle('Exp. Norm2, Static CS:{}km, noise_std:{}m/s,rmse in cloud:{}m/s'.format(np.round(float(zr[zind_rel[zi]]),3),noise[j],np.round(rmse_incloud14,3)))

    im1 = ax1.imshow(zwind_data1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],vmin=vmin,vmax=vmax)
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cbar1 = plt.colorbar(im1, cax=cax1)
    ax1.contour(lwc_cloud1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
    ax1.set_xlim(xr[0], xr[-1])
    ax1.set_ylim(yr[30], yr[-1])
    ax1.plot(drones_train[:,2],drones_train[:,3],'k.',alpha=0.3)
    ax1.set_xlabel('x coordinate (km)')
    ax1.set_ylabel('y coordinate(km)')
    ax1.set_title('Ground truth')

    im2 = ax2.imshow(mean_pred4[:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],vmin=vmin,vmax=vmax)
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    cbar2 = plt.colorbar(im2, cax=cax2)
    ax2.contour(lwc_cloud1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
    ax2.set_xlim(xr[0], xr[-1])
    ax2.set_ylim(yr[30], yr[-1])
    ax2.plot(drones_train[:,2],drones_train[:,3],'k.',alpha=0.3)
    ax2.set_xlabel('x coordinate (km)')
    ax2.set_ylabel('y coordinate(km)')
    ax2.set_title('Predicted mean $y_{\star}$')

    im3 = ax3.imshow(np.sqrt(var_pred4[:,30:].T),origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],vmin=0,vmax=2)
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="5%", pad=0.05)
    cbar3 = plt.colorbar(im3, cax=cax3)
    ax3.contour(lwc_cloud1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
    ax3.set_xlim(xr[0], xr[-1])
    ax3.set_ylim(yr[30], yr[-1])
    ax3.plot(drones_train[:,2],drones_train[:,3],'k.',alpha=0.3)
    ax3.set_xlabel('x coordinate (km)')
    ax3.set_ylabel('y coordinate(km)')
    ax3.set_title('Predicted $\sqrt{V[y_{\star}]}$')

    im4 = ax4.imshow(err_incloud14_total[:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],vmin=-1.5,vmax=1.5)
    divider4 = make_axes_locatable(ax4)
    cax4 = divider4.append_axes("right", size="5%", pad=0.05)
    cbar4 = plt.colorbar(im4, cax=cax4)
    ax4.contour(lwc_cloud1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
    ax4.set_xlim(xr[0], xr[-1])
    ax4.set_ylim(yr[30], yr[-1])
    ax4.plot(drones_train[:,2],drones_train[:,3],'k.',alpha=0.3)
    ax4.set_xlabel('x coordinate (km)')
    ax4.set_ylabel('y coordinate(km)')
    ax4.set_title('Prediction error')


    ################### libgp, normalized radius and corrected phi,optimized, starting from variogram hypers #####################
    M5 = GPModel(ndim=4,nvar=1, kernel_string=b"CovSum(CovExpArdPhi, CovNoise)")
    M5.update(drones_train_normal,zwind_train_detrended.T)

    ## Values in log scale
    input_opt = {"gp_opt_python":True,'gp_opt_iter':10,'gp_hyper_bounds':((-10, 10), (-10, 10), (-10, 10), (-10, 10),
                     (-10, 10),
                     (-10, 10)),'gp_discr_tol':-1}

    lt = np.log(51.026823559394558)
    lz = np.log(0.01*13.054954891415182)
    l_phi = np.log(23.025993674634258)
    lr = np.log(40.199201579845884)
    sigma2 = 0.5*np.log(0.84069334459964384)
    noise_var =0.5*np.log(noise[j]**2)

    params = np.array([lt,lz,l_phi,lr,sigma2,noise_var]).reshape(1,-1)
    M5.set_params(params)

    M5.optimize(input_opt)
    params5 = np.asarray(M5.get_params()).reshape(6)
    predictions5 = np.ndarray((1,2,140*150))

    M5.predict(grid_unrolled_normal,predictions5)

    mean_pred5 = predictions5[0,0].reshape(-1,1)
    var_pred5 = predictions5[0,1].reshape(-1,1)

    mean_pred5 = mean_pred5 + zwind_com*interpolate_rtrend(grid_unrolled_normal[:,3],'linear').reshape((-1,1))

    mean_pred5 = mean_pred5.reshape((140,150))
    var_pred5 = var_pred5.reshape((140,150))

    mean_std_pred_incloud15 = np.mean(np.sqrt(var_pred5[lwc_cloud1[4,zind_rel[zi]]>=1e-5]))
    rmse_incloud15 = np.sqrt(np.mean((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]-mean_pred5[lwc_cloud1[4,zind_rel[zi]]>=1e-5])**2))

    abs_err_incloud15 = np.abs(zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]-mean_pred5[lwc_cloud1[4,zind_rel[zi]]>=1e-5])
    std_pred_incloud15 = np.sqrt(var_pred5[lwc_cloud1[4,zind_rel[zi]]>=1e-5])
    test_std_soundness_incloud15 = (std_pred_incloud15-abs_err_incloud15)/std_pred_incloud15

    std_pred_soundness_incloud15 = np.array([np.percentile(test_std_soundness_incloud15,0.3),np.percentile(test_std_soundness_incloud15,5),
    np.percentile(test_std_soundness_incloud15,32)])


    explained_variance = np.sum((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5] -
                        mean_pred5[lwc_cloud1[4,zind_rel[zi]]>=1e-5])**2)
    total_variance = np.sum((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]- zwind_train.mean())**2)
    r2_incloud15 = 1 - explained_variance/total_variance


    err_incloud15 = zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]-mean_pred5[lwc_cloud1[4,zind_rel[zi]]>=1e-5]
    err_stats_incloud15 = np.array([np.mean(err_incloud15),np.median(err_incloud15),np.std(err_incloud15),skew(err_incloud15)])

    err_incloud15_total = zwind_data1[4,zind_rel[zi]]-mean_pred5

    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4,sharey=True)

    f.suptitle('Exp. Norm3, Static CS:{}km, noise_std:{}m/s,rmse in cloud:{}m/s'.format(np.round(float(zr[zind_rel[zi]]),3),noise[j],np.round(rmse_incloud15,3)))
    im1 = ax1.imshow(zwind_data1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],vmin=vmin,vmax=vmax)
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cbar1 = plt.colorbar(im1, cax=cax1)
    ax1.contour(lwc_cloud1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
    ax1.set_xlim(xr[0], xr[-1])
    ax1.set_ylim(yr[30], yr[-1])
    ax1.plot(drones_train[:,2],drones_train[:,3],'k.',alpha=0.3)
    ax1.set_xlabel('x coordinate (km)')
    ax1.set_ylabel('y coordinate(km)')
    ax1.set_title('Ground truth')

    im2 = ax2.imshow(mean_pred5[:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],vmin=vmin,vmax=vmax)
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    cbar2 = plt.colorbar(im2, cax=cax2)
    ax2.contour(lwc_cloud1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
    ax2.set_xlim(xr[0], xr[-1])
    ax2.set_ylim(yr[30], yr[-1])
    ax2.plot(drones_train[:,2],drones_train[:,3],'k.',alpha=0.3)
    ax2.set_xlabel('x coordinate (km)')
    ax2.set_ylabel('y coordinate(km)')
    ax2.set_title('Predicted mean $y_{\star}$')

    im3 = ax3.imshow(np.sqrt(var_pred5[:,30:].T),origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],vmin=0,vmax=2)
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="5%", pad=0.05)
    cbar3 = plt.colorbar(im3, cax=cax3)
    ax3.contour(lwc_cloud1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
    ax3.set_xlim(xr[0], xr[-1])
    ax3.set_ylim(yr[30], yr[-1])
    ax3.plot(drones_train[:,2],drones_train[:,3],'k.',alpha=0.3)
    ax3.set_xlabel('x coordinate (km)')
    ax3.set_ylabel('y coordinate(km)')
    ax3.set_title('Predicted $\sqrt{V[y_{\star}]}$')

    im4 = ax4.imshow(err_incloud15_total[:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],vmin=-1.5,vmax=1.5)
    divider4 = make_axes_locatable(ax4)
    cax4 = divider4.append_axes("right", size="5%", pad=0.05)
    cbar4 = plt.colorbar(im4, cax=cax4)
    ax4.contour(lwc_cloud1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
    ax4.set_xlim(xr[0], xr[-1])
    ax4.set_ylim(yr[30], yr[-1])
    ax4.plot(drones_train[:,2],drones_train[:,3],'k.',alpha=0.3)
    ax4.set_xlabel('x coordinate (km)')
    ax4.set_ylabel('y coordinate(km)')
    ax4.set_title('Prediction error')

    ########################################### libgp SE xy, detrended data
    M6 = GPModel(ndim=4,nvar=1, kernel_string=b"CovSum(CovSEard, CovNoise)")
    M6.update(drones_train,zwind_train_detrended.T)

    ## Values in log scale
    input_opt = {"gp_opt_python":True,'gp_opt_iter':10,'gp_hyper_bounds':((-10, 10), (-10, 10), (-10, 10), (-10, 10),
                     (-10, 10),
                     (-10, 10)),'gp_discr_tol':-1}



    M6.optimize(input_opt)
    params8 = np.asarray(M6.get_params()).reshape(6)
    predictions8 = np.ndarray((1,2,140*150))

    M6.predict(grid_unrolled,predictions8)

    mean_pred8 = predictions8[0,0].reshape(-1,1)
    var_pred8 = predictions8[0,1].reshape(-1,1)

    mean_pred8 = mean_pred8 + zwind_com*interpolate_rtrend(grid_unrolled_normal[:,3],'linear').reshape((-1,1))

    mean_pred8 = mean_pred8.reshape((140,150))
    var_pred8 = var_pred8.reshape((140,150))

    mean_std_pred_incloud18 = np.mean(np.sqrt(var_pred8[lwc_cloud1[4,zind_rel[zi]]>=1e-5]))
    rmse_incloud18 = np.sqrt(np.mean((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]-mean_pred8[lwc_cloud1[4,zind_rel[zi]]>=1e-5])**2))

    abs_err_incloud18 = np.abs(zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]-mean_pred8[lwc_cloud1[4,zind_rel[zi]]>=1e-5])
    std_pred_incloud18 = np.sqrt(var_pred8[lwc_cloud1[4,zind_rel[zi]]>=1e-5])
    test_std_soundness_incloud18 = (std_pred_incloud18-abs_err_incloud18)/std_pred_incloud18

    std_pred_soundness_incloud18 = np.array([np.percentile(test_std_soundness_incloud18,0.3),np.percentile(test_std_soundness_incloud18,5),
    np.percentile(test_std_soundness_incloud18,32)])


    explained_variance = np.sum((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5] -
                        mean_pred8[lwc_cloud1[4,zind_rel[zi]]>=1e-5])**2)
    total_variance = np.sum((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]- zwind_train.mean())**2)
    r2_incloud18 = 1 - explained_variance/total_variance


    err_incloud18 = zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]-mean_pred8[lwc_cloud1[4,zind_rel[zi]]>=1e-5]
    err_stats_incloud18 = np.array([np.mean(err_incloud18),np.median(err_incloud18),np.std(err_incloud18),skew(err_incloud18)])

    err_incloud18_total = zwind_data1[4,zind_rel[zi]]-mean_pred8

    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4,sharey=True)

    f.suptitle('Sq. Exp detrended, Static CS:{}km, noise_std:{}m/s,rmse in cloud:{}m/s'.format(np.round(float(zr[zind_rel[zi]]),3),noise[j],np.round(rmse_incloud18,3)))
    im1 = ax1.imshow(zwind_data1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],vmin=vmin,vmax=vmax)
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cbar1 = plt.colorbar(im1, cax=cax1)
    ax1.contour(lwc_cloud1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
    ax1.set_xlim(xr[0], xr[-1])
    ax1.set_ylim(yr[30], yr[-1])
    ax1.plot(drones_train[:,2],drones_train[:,3],'k.',alpha=0.3)
    ax1.set_xlabel('x coordinate (km)')
    ax1.set_ylabel('y coordinate(km)')
    ax1.set_title('Ground truth')

    im2 = ax2.imshow(mean_pred8[:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],vmin=vmin,vmax=vmax)
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    cbar2 = plt.colorbar(im2, cax=cax2)
    ax2.contour(lwc_cloud1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
    ax2.set_xlim(xr[0], xr[-1])
    ax2.set_ylim(yr[30], yr[-1])
    ax2.plot(drones_train[:,2],drones_train[:,3],'k.',alpha=0.3)
    ax2.set_xlabel('x coordinate (km)')
    ax2.set_ylabel('y coordinate(km)')
    ax2.set_title('Predicted mean $y_{\star}$')

    im3 = ax3.imshow(np.sqrt(var_pred8[:,30:].T),origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],vmin=0,vmax=2)
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="5%", pad=0.05)
    cbar3 = plt.colorbar(im3, cax=cax3)
    ax3.contour(lwc_cloud1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
    ax3.set_xlim(xr[0], xr[-1])
    ax3.set_ylim(yr[30], yr[-1])
    ax3.plot(drones_train[:,2],drones_train[:,3],'k.',alpha=0.3)
    ax3.set_xlabel('x coordinate (km)')
    ax3.set_ylabel('y coordinate(km)')
    ax3.set_title('Predicted $\sqrt{V[y_{\star}]}$')

    im4 = ax4.imshow(err_incloud18_total[:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],vmin=-1.5,vmax=1.5)
    divider4 = make_axes_locatable(ax4)
    cax4 = divider4.append_axes("right", size="5%", pad=0.05)
    cbar4 = plt.colorbar(im4, cax=cax4)
    ax4.contour(lwc_cloud1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
    ax4.set_xlim(xr[0], xr[-1])
    ax4.set_ylim(yr[30], yr[-1])
    ax4.plot(drones_train[:,2],drones_train[:,3],'k.',alpha=0.3)
    ax4.set_xlabel('x coordinate (km)')
    ax4.set_ylabel('y coordinate(km)')
    ax4.set_title('Prediction error')

    ########################################### libgp Exp xy, detrended data
    M7 = GPModel(ndim=4,nvar=1, kernel_string=b"CovSum(CovExpArd, CovNoise)")
    M7.update(drones_train,zwind_train_detrended.T)

    ## Values in log scale
    input_opt = {"gp_opt_python":True,'gp_opt_iter':10,'gp_hyper_bounds':((-10, 10), (-10, 10), (-10, 10), (-10, 10),
                     (-10, 10),
                     (-10, 10)),'gp_discr_tol':-1}



    M7.optimize(input_opt)
    params9 = np.asarray(M7.get_params()).reshape(6)
    predictions9 = np.ndarray((1,2,140*150))

    M7.predict(grid_unrolled,predictions9)

    mean_pred9 = predictions9[0,0].reshape(-1,1)
    var_pred9 = predictions9[0,1].reshape(-1,1)

    mean_pred9 = mean_pred9 + zwind_com*interpolate_rtrend(grid_unrolled_normal[:,3],'linear').reshape((-1,1))

    mean_pred9 = mean_pred9.reshape((140,150))
    var_pred9 = var_pred9.reshape((140,150))

    mean_std_pred_incloud19 = np.mean(np.sqrt(var_pred9[lwc_cloud1[4,zind_rel[zi]]>=1e-5]))
    rmse_incloud19 = np.sqrt(np.mean((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]-mean_pred9[lwc_cloud1[4,zind_rel[zi]]>=1e-5])**2))

    abs_err_incloud19 = np.abs(zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]-mean_pred9[lwc_cloud1[4,zind_rel[zi]]>=1e-5])
    std_pred_incloud19 = np.sqrt(var_pred9[lwc_cloud1[4,zind_rel[zi]]>=1e-5])
    test_std_soundness_incloud19 = (std_pred_incloud19-abs_err_incloud19)/std_pred_incloud19

    std_pred_soundness_incloud19 = np.array([np.percentile(test_std_soundness_incloud19,0.3),np.percentile(test_std_soundness_incloud19,5),
    np.percentile(test_std_soundness_incloud19,32)])


    explained_variance = np.sum((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5] -
                        mean_pred9[lwc_cloud1[4,zind_rel[zi]]>=1e-5])**2)
    total_variance = np.sum((zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]- zwind_train.mean())**2)
    r2_incloud19 = 1 - explained_variance/total_variance


    err_incloud19 = zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5]-mean_pred9[lwc_cloud1[4,zind_rel[zi]]>=1e-5]
    err_stats_incloud19 = np.array([np.mean(err_incloud19),np.median(err_incloud19),np.std(err_incloud19),skew(err_incloud19)])

    err_incloud19_total = zwind_data1[4,zind_rel[zi]]-mean_pred9

    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4,sharey=True)

    f.suptitle('Exp. detrended, Static CS:{}km, noise_std:{} m/s,rmse in cloud:{} m/s'.format(np.round(float(zr[zind_rel[zi]]),3),noise[j],np.round(rmse_incloud19,3)))
    im1 = ax1.imshow(zwind_data1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],vmin=vmin,vmax=vmax)
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cbar1 = plt.colorbar(im1, cax=cax1)
    ax1.contour(lwc_cloud1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
    ax1.set_xlim(xr[0], xr[-1])
    ax1.set_ylim(yr[30], yr[-1])
    ax1.plot(drones_train[:,2],drones_train[:,3],'k.',alpha=0.3)
    ax1.set_xlabel('x coordinate (km)')
    ax1.set_ylabel('y coordinate(km)')
    ax1.set_title('Ground truth')

    im2 = ax2.imshow(mean_pred9[:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],vmin=vmin,vmax=vmax)
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    cbar2 = plt.colorbar(im2, cax=cax2)
    ax2.contour(lwc_cloud1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
    ax2.set_xlim(xr[0], xr[-1])
    ax2.set_ylim(yr[30], yr[-1])
    ax2.plot(drones_train[:,2],drones_train[:,3],'k.',alpha=0.3)
    ax2.set_xlabel('x coordinate (km)')
    ax2.set_ylabel('y coordinate(km)')
    ax2.set_title('Predicted mean $y_{\star}$')

    im3 = ax3.imshow(np.sqrt(var_pred9[:,30:].T),origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],vmin=0,vmax=2)
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="5%", pad=0.05)
    cbar3 = plt.colorbar(im3, cax=cax3)
    ax3.contour(lwc_cloud1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
    ax3.set_xlim(xr[0], xr[-1])
    ax3.set_ylim(yr[30], yr[-1])
    ax3.plot(drones_train[:,2],drones_train[:,3],'k.',alpha=0.3)
    ax3.set_xlabel('x coordinate (km)')
    ax3.set_ylabel('y coordinate(km)')
    ax3.set_title('Predicted $\sqrt{V[y_{\star}]}$')

    im4 = ax4.imshow(err_incloud19_total[:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],vmin=-1.5,vmax=1.5)
    divider4 = make_axes_locatable(ax4)
    cax4 = divider4.append_axes("right", size="5%", pad=0.05)
    cbar4 = plt.colorbar(im4, cax=cax4)
    ax4.contour(lwc_cloud1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
    ax4.set_xlim(xr[0], xr[-1])
    ax4.set_ylim(yr[30], yr[-1])
    ax4.plot(drones_train[:,2],drones_train[:,3],'k.',alpha=0.3)
    ax4.set_xlabel('x coordinate (km)')
    ax4.set_ylabel('y coordinate(km)')
    ax4.set_title('Prediction error')



    ############################################# Comparison SE xy,trend,libgp expnorm1
    font = {'size'   : 15}

    plt.rc('font', **font)


    f, axarr = plt.subplots(3, 4,sharey=True)

    f.suptitle('Static CS:{}km, noise_std:{}m/s,rmse Sq.Exp:{}m/s,rmse trend:{}m/s,rmse ExpNorm1:{}m/s'.format(np.round(float(zr[zind_rel[zi]]),3),noise[j],
    np.round(rmse_incloud12,3),np.round(rmse_trend,3),np.round(rmse_incloud13,3)))

    im1 = axarr[0,0].imshow(zwind_data1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],vmin=vmin,vmax=vmax)
    divider1 = make_axes_locatable(axarr[0,0])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cbar1 = plt.colorbar(im1, cax=cax1)
    axarr[0,0].contour(lwc_cloud1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
    axarr[0,0].set_xlim(xr[0], xr[-1])
    axarr[0,0].set_ylim(yr[30], yr[-1])
    axarr[0,0].plot(drones_train[:,2],drones_train[:,3],'k.',alpha=0.3)
    axarr[0,0].set_xlabel('x coordinate (km)')
    axarr[0,0].set_ylabel('y coordinate(km)')
    axarr[0,0].set_title('Ground truth')

    im2 = axarr[0,1].imshow(mean_pred2[:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],vmin=vmin,vmax=vmax)
    divider2 = make_axes_locatable(axarr[0,1])
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    cbar2 = plt.colorbar(im2, cax=cax2)
    axarr[0,1].contour(lwc_cloud1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
    axarr[0,1].set_xlim(xr[0], xr[-1])
    axarr[0,1].set_ylim(yr[30], yr[-1])
    axarr[0,1].plot(drones_train[:,2],drones_train[:,3],'k.',alpha=0.3)
    axarr[0,1].set_xlabel('x coordinate (km)')
    axarr[0,1].set_ylabel('y coordinate(km)')
    axarr[0,1].set_title('Predicted mean $y_{\star}$')

    im3 = axarr[0,2].imshow(np.sqrt(var_pred2[:,30:].T),origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],vmin=0,vmax=2)
    divider3 = make_axes_locatable(axarr[0,2])
    cax3 = divider3.append_axes("right", size="5%", pad=0.05)
    cbar3 = plt.colorbar(im3, cax=cax3)
    axarr[0,2].contour(lwc_cloud1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
    axarr[0,2].set_xlim(xr[0], xr[-1])
    axarr[0,2].set_ylim(yr[30], yr[-1])
    axarr[0,2].plot(drones_train[:,2],drones_train[:,3],'k.',alpha=0.3)
    axarr[0,2].set_xlabel('x coordinate (km)')
    axarr[0,2].set_ylabel('y coordinate(km)')
    axarr[0,2].set_title('Predicted $\sqrt{V[y_{\star}]}$')

    im4 = axarr[0,3].imshow(err_incloud12_total[:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],vmin=-1.5,vmax=1.5)
    divider4 = make_axes_locatable(axarr[0,3])
    cax4 = divider4.append_axes("right", size="5%", pad=0.05)
    cbar4 = plt.colorbar(im4, cax=cax4)
    axarr[0,3].contour(lwc_cloud1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
    axarr[0,3].set_xlim(xr[0], xr[-1])
    axarr[0,3].set_ylim(yr[30], yr[-1])
    axarr[0,3].plot(drones_train[:,2],drones_train[:,3],'k.',alpha=0.3)
    axarr[0,3].set_xlabel('x coordinate (km)')
    axarr[0,3].set_ylabel('y coordinate(km)')
    axarr[0,3].set_title('Prediction error')

    im1 = axarr[1,0].imshow(zwind_data1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],vmin=vmin,vmax=vmax)
    divider1 = make_axes_locatable(axarr[1,0])
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    cbar1 = plt.colorbar(im1, cax=cax1)
    axarr[1,0].contour(lwc_cloud1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
    axarr[1,0].set_xlim(xr[0], xr[-1])
    axarr[1,0].set_ylim(yr[30], yr[-1])
    axarr[1,0].plot(drones_train[:,2],drones_train[:,3],'k.',alpha=0.3)
    axarr[1,0].set_xlabel('x coordinate (km)')
    axarr[1,0].set_ylabel('y coordinate(km)')
    axarr[1,0].set_title('Ground truth')

    im2 = axarr[1,1].imshow(trend_pred[:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],vmin=vmin,vmax=vmax)
    divider2 = make_axes_locatable(axarr[1,1])
    cax2 = divider2.append_axes("right", size="5%", pad=0.1)
    cbar2 = plt.colorbar(im2, cax=cax2)
    axarr[1,1].contour(lwc_cloud1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
    axarr[1,1].set_xlim(xr[0], xr[-1])
    axarr[1,1].set_ylim(yr[30], yr[-1])
    axarr[1,1].plot(drones_train[:,2],drones_train[:,3],'k.',alpha=0.3)
    axarr[1,1].set_xlabel('x coordinate (km)')
    axarr[1,1].set_ylabel('y coordinate(km)')
    axarr[1,1].set_title('Prediction of trend')

    im3 = axarr[1,2].imshow(np.zeros((140,120)).T,extent=[xr[0], xr[-1], yr[30], yr[-1]],vmin=0,vmax=2)
    axarr[1,2].contour(lwc_cloud1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
    divider3 = make_axes_locatable(axarr[1,2])
    cax3 = divider3.append_axes("right", size="5%", pad=0.1)
    cbar3 = plt.colorbar(im3, cax=cax3)
    axarr[1,2].set_xlim(xr[0], xr[-1])
    axarr[1,2].set_ylim(yr[30], yr[-1])
    axarr[1,2].plot(drones_train[:,2],drones_train[:,3],'k.',alpha=0.3)
    axarr[1,2].set_title('Predicted std does not apply')
    axarr[1,2].set_xlabel('x coordinate (km)')
    axarr[1,2].set_ylabel('y coordinate(km)')

    im4 = axarr[1,3].imshow(err_trend_total[:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],vmin=-1.5,vmax=1.5)
    divider4 = make_axes_locatable(axarr[1,3])
    cax4 = divider4.append_axes("right", size="5%", pad=0.1)
    cbar4 = plt.colorbar(im4, cax=cax4)
    axarr[1,3].contour(lwc_cloud1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
    axarr[1,3].set_xlim(xr[0], xr[-1])
    axarr[1,3].set_ylim(yr[30], yr[-1])
    axarr[1,3].plot(drones_train[:,2],drones_train[:,3],'k.',alpha=0.3)
    axarr[1,3].set_xlabel('x coordinate (km)')
    axarr[1,3].set_ylabel('y coordinate(km)')
    axarr[1,3].set_title('Prediction error of trend')

    im1 = axarr[2,0].imshow(zwind_data1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],vmin=vmin,vmax=vmax)
    divider1 = make_axes_locatable(axarr[2,0])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cbar1 = plt.colorbar(im1, cax=cax1)
    axarr[2,0].contour(lwc_cloud1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
    axarr[2,0].set_xlim(xr[0], xr[-1])
    axarr[2,0].set_ylim(yr[30], yr[-1])
    axarr[2,0].plot(drones_train[:,2],drones_train[:,3],'k.',alpha=0.3)
    axarr[2,0].set_xlabel('x coordinate (km)')
    axarr[2,0].set_ylabel('y coordinate(km)')
    axarr[2,0].set_title('Ground truth')

    im2 = axarr[2,1].imshow(mean_pred3[:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],vmin=vmin,vmax=vmax)
    divider2 = make_axes_locatable(axarr[2,1])
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    cbar2 = plt.colorbar(im2, cax=cax2)
    axarr[2,1].contour(lwc_cloud1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
    axarr[2,1].set_xlim(xr[0], xr[-1])
    axarr[2,1].set_ylim(yr[30], yr[-1])
    axarr[2,1].plot(drones_train[:,2],drones_train[:,3],'k.',alpha=0.3)
    axarr[2,1].set_xlabel('x coordinate (km)')
    axarr[2,1].set_ylabel('y coordinate(km)')
    axarr[2,1].set_title('Predicted mean $y_{\star}$')

    im3 = axarr[2,2].imshow(np.sqrt(var_pred3[:,30:].T),origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],vmin=0,vmax=2)
    divider3 = make_axes_locatable(axarr[2,2])
    cax3 = divider3.append_axes("right", size="5%", pad=0.05)
    cbar3 = plt.colorbar(im3, cax=cax3)
    axarr[2,2].contour(lwc_cloud1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
    axarr[2,2].set_xlim(xr[0], xr[-1])
    axarr[2,2].set_ylim(yr[30], yr[-1])
    axarr[2,2].plot(drones_train[:,2],drones_train[:,3],'k.',alpha=0.3)
    axarr[2,2].set_xlabel('x coordinate (km)')
    axarr[2,2].set_ylabel('y coordinate(km)')
    axarr[2,2].set_title('Predicted $\sqrt{V[y_{\star}]}$')

    im4 = axarr[2,3].imshow(err_incloud13_total[:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],vmin=-1.5,vmax=1.5)
    divider4 = make_axes_locatable(axarr[2,3])
    cax4 = divider4.append_axes("right", size="5%", pad=0.05)
    cbar4 = plt.colorbar(im4, cax=cax4)
    axarr[2,3].contour(lwc_cloud1[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
    axarr[2,3].set_xlim(xr[0], xr[-1])
    axarr[2,3].set_ylim(yr[30], yr[-1])
    axarr[2,3].plot(drones_train[:,2],drones_train[:,3],'k.',alpha=0.3)
    axarr[2,3].set_xlabel('x coordinate (km)')
    axarr[2,3].set_ylabel('y coordinate(km)')
    axarr[2,3].set_title('Prediction error')




#################################################################################################
###################################### dynamic test GP entire cloud #############################
#################################################################################################
all_rmse_dyn_whole_cl = np.nan*np.ndarray((20,5,11))

all_r2_dyn_whole_cl = np.nan*np.ndarray((20,5,11))

all_std_pred_dyn_whole_cl = np.nan*np.ndarray((20,5,9))

all_std_pred_soundness_dyn_whole_cl = np.nan*np.ndarray((20,5,9,3))

all_hypers_dyn_whole_cl = np.nan*np.ndarray((20,5,9,6))

all_error_stats_dyn_whole_cl = np.nan*np.ndarray((20,5,11,4))

tstart_helix = 449
tend_helix = 526
zstart_helix = 1.185
radius_circle = 0.09

drone1_helix = np.ndarray((0,4))
xcenter = 1.1
ycenter = 2.1
for t in np.arange(tstart_helix,tend_helix):
    trel = t - tstart_helix
    xtemp = xcenter + radius_circle*np.cos(10/90*trel)
    ytemp = ycenter + radius_circle*np.sin(10/90*trel)
    ztemp = zstart_helix + trel*0.002
    temp = np.array([t,ztemp,xtemp,ytemp])
    incloud1 = interpolate_points_cloud1(temp,'nearest')
    if incloud1 == 1.0:
        drone1_helix = np.vstack((drone1_helix,temp))

drone2_helix = np.ndarray((0,4))
xcenter = 1.4
ycenter = 2.06
for t in np.arange(tstart_helix,tend_helix):
    trel = t - tstart_helix
    xtemp = xcenter + radius_circle*np.cos(10/90*trel)
    ytemp = ycenter + radius_circle*np.sin(10/90*trel)
    ztemp = zstart_helix + trel*0.002
    temp = np.array([t,ztemp,xtemp,ytemp])
    incloud1 = interpolate_points_cloud1(temp,'nearest')
    if incloud1 == 1.0:
        drone2_helix = np.vstack((drone2_helix,temp))

drone3_helix = np.ndarray((0,4))
xcenter = 1.23
ycenter = 1.9
for t in np.arange(tstart_helix,tend_helix):
    trel = t - tstart_helix
    xtemp = xcenter + radius_circle*np.cos(15/90*trel)
    ytemp = ycenter + radius_circle*np.sin(15/90*trel)
    ztemp = zstart_helix + trel*0.002
    temp = np.array([t,ztemp,xtemp,ytemp])
    incloud1 = interpolate_points_cloud1(temp,'nearest')
    if incloud1 == 1.0:
        drone3_helix = np.vstack((drone3_helix,temp))

drone4_helix = np.ndarray((0,4))
xcenter = 1.35
ycenter = 1.9
for t in np.arange(tstart_helix,tend_helix):
    trel = t - tstart_helix
    xtemp = xcenter + radius_circle*np.cos(15/90*trel)
    ytemp = ycenter + radius_circle*np.sin(15/90*trel)
    ztemp = zstart_helix + trel*0.002
    temp = np.array([t,ztemp,xtemp,ytemp])
    incloud1 = interpolate_points_cloud1(temp,'nearest')
    if incloud1 == 1.0:
        drone4_helix = np.vstack((drone4_helix,temp))

drone5_helix = np.ndarray((0,4))
xcenter = 1.35
ycenter = 1.7
for t in np.arange(tstart_helix,tend_helix):
    trel = t - tstart_helix
    xtemp = xcenter + radius_circle*np.cos(15/90*trel)
    ytemp = ycenter + radius_circle*np.sin(15/90*trel)
    ztemp = zstart_helix + trel*0.002
    temp = np.array([t,ztemp,xtemp,ytemp])
    incloud1 = interpolate_points_cloud1(temp,'nearest')
    if incloud1 == 1.0:
        drone5_helix = np.vstack((drone5_helix,temp))


################### Train and Test Data GPs, not dependent on noise ################
drones_train_dyn_whole_cl = np.vstack((drone1_helix,drone2_helix,drone3_helix,drone4_helix,drone5_helix))
drones_train_dyn_whole_cl = drones_train_dyn_whole_cl[drones_train_dyn_whole_cl[:,0]!=480]
grid_unrolled_dyn_whole_cl= grid[76].reshape((-1,4))

drones_train_normal_dyn_whole_cl = normalize(lwc_cloud1_polar,drones_train_dyn_whole_cl,cloud1.COM_2D_lwc_tz,449,1.185,origin_xy)
grid_unrolled_normal_dyn_whole_cl = normalize(lwc_cloud1_polar,grid_unrolled_dyn_whole_cl,cloud1.COM_2D_lwc_tz,449,1.185,origin_xy)


#trials = np.arange(0,20)
trials = np.arange(0,20)
for i in trials:
    noise = np.array([0.01,0.1,0.25,0.5,0.75])
    #noise = np.array([1e-3])
    for j in range(len(noise)):
        #try:
        print('progress Whole Cl.:Trial={},Noise={}'.format(i,j))

        time1 = datetime.datetime.now()
        zwind_train_dyn_whole_cl = atm.get_points(drones_train_dyn_whole_cl,'WT','linear')
        zwind_train_dyn_whole_cl = zwind_train_dyn_whole_cl.reshape((len(zwind_train_dyn_whole_cl),1))
        zwind_train_dyn_whole_cl = zwind_train_dyn_whole_cl + np.random.randn(len(zwind_train_dyn_whole_cl),1)*noise[j]

        interpolate_rtrend = RegularGridInterpolator(points=np.arange(0,151).reshape(1,151),values=rtrend_global_median_updraft_norm,bounds_error=False,fill_value=0)

        COM =np.array([449,1.185,(cloud1.COM_2D_lwc_tz[0,0,0] + origin_xy[0])*0.01,(cloud1.COM_2D_lwc_tz[0,0,1] + origin_xy[1])*0.01])
        zwind_com = atm.get_points(COM,'WT','linear') + np.random.randn(1)*noise[j]


        zwind_train_detrended_dyn_whole_cl = zwind_train_dyn_whole_cl - (zwind_com*interpolate_rtrend(drones_train_normal_dyn_whole_cl[:,3],'linear')).reshape((-1,1))

        rmse_trend_dyn_whole_cl = np.sqrt(np.mean((zwind_data1[76][lwc_cloud1[76]>=1e-5] -
        zwind_com*interpolate_rtrend(grid_unrolled_normal_dyn_whole_cl[:,3],'linear').reshape((16,140,150))[lwc_cloud1[76]>=1e-5])**2))

        explained_variance = np.sum((zwind_data1[76][lwc_cloud1[76]>=1e-5] -
        zwind_com*interpolate_rtrend(grid_unrolled_normal_dyn_whole_cl[:,3],'linear').reshape((16,140,150))[lwc_cloud1[76]>=1e-5])**2)
        total_variance = np.sum((zwind_data1[76][lwc_cloud1[76]>=1e-5]- zwind_train_dyn_whole_cl.mean())**2)
        r2_trend_dyn_whole_cl = 1 - explained_variance/total_variance

        err_trend_dyn_whole_cl = (zwind_data1[76][lwc_cloud1[76]>=1e-5]-
        zwind_com*interpolate_rtrend(grid_unrolled_normal_dyn_whole_cl[:,3],'linear').reshape((16,140,150))[lwc_cloud1[76]>=1e-5])
        err_stats_trend_dyn_whole_cl = np.array([np.mean(err_trend_dyn_whole_cl),np.median(err_trend_dyn_whole_cl),np.std(err_trend_dyn_whole_cl),
        skew(err_trend_dyn_whole_cl)])

        rmse_mean_dyn_whole_cl =np.sqrt(np.mean((zwind_data1[76][lwc_cloud1[76]>=1e-5]- zwind_train_dyn_whole_cl.mean())**2))
        r2_mean_dyn_whole_cl = 0

        err_mean_dyn_whole_cl = zwind_data1[76][lwc_cloud1[76]>=1e-5]- zwind_train_dyn_whole_cl.mean()
        err_stats_mean_dyn_whole_cl = np.array([np.mean(err_mean_dyn_whole_cl),np.median(err_mean_dyn_whole_cl),np.std(err_mean_dyn_whole_cl),skew(err_mean_dyn_whole_cl)])
        ############################################## libgp Exponential t,x,y, #####################################################
        M_dyn_whole_cl = GPModel(ndim=4,nvar=1, kernel_string=b"CovSum(CovExpArd, CovNoise)")
        #M3 = GPModel(ndim=4,nvar=1, kernel_string="CovExpArdPhi")
        #drones_train_normal_dyn_cs_wo_t = drones_train_normal_dyn_cs.copy()
        #drones_train_normal_dyn_cs_wo_t[:,0] = 1
        #drones_train_normal_dyn_cs_wo_phi = drones_train_normal_dyn_cs.copy()
        #drones_train_normal_dyn_cs_wo_phi[:,2] = 1
        M_dyn_whole_cl.update(drones_train_dyn_whole_cl,zwind_train_dyn_whole_cl.T)

        ## Values in log scale
        input_opt = {"gp_opt_python":True,'gp_opt_iter':10,'gp_hyper_bounds':((-10, 10), (-10, 10), (-10, 10), (-10, 10),
                         (-10, 10),
                         (-10, 10)),'gp_discr_tol':-1}
        M_dyn_whole_cl.optimize(input_opt)
        params1_dyn_whole_cl = np.asarray(M_dyn_whole_cl.get_params()).reshape(6)
        predictions_dyn_whole_cl = np.ndarray((1,2,16*140*150))

        #grid_unrolled_normal_dyn_cs_wo_t = grid_unrolled_normal_dyn_cs.copy()
        #grid_unrolled_normal_dyn_cs_wo_t[:,0] =1
        #grid_unrolled_normal_dyn_cs_wo_phi = grid_unrolled_normal_dyn_cs.copy()
        #grid_unrolled_normal_dyn_cs_wo_phi[:,2] = 1
        M_dyn_whole_cl.predict(grid_unrolled_dyn_whole_cl,predictions_dyn_whole_cl)

        mean_pred_dyn_whole_cl = predictions_dyn_whole_cl[0,0].reshape(-1,1)
        var_pred_dyn_whole_cl = predictions_dyn_whole_cl[0,1].reshape(-1,1)

        mean_pred_dyn_whole_cl = mean_pred_dyn_whole_cl.reshape((16,140,150))
        var_pred_dyn_whole_cl = var_pred_dyn_whole_cl.reshape((16,140,150))

        mean_std_pred_incloud11_dyn_whole_cl = np.mean(np.sqrt(var_pred_dyn_whole_cl[lwc_cloud1[76]>=1e-5]))
        rmse_incloud11_dyn_whole_cl = np.sqrt(np.mean((zwind_data1[76][lwc_cloud1[76]>=1e-5]-mean_pred_dyn_whole_cl[lwc_cloud1[76]>=1e-5])**2))


        abs_err_incloud11_dyn_whole_cl = np.abs(zwind_data1[76][lwc_cloud1[76]>=1e-5]-mean_pred_dyn_whole_cl[lwc_cloud1[76]>=1e-5])
        std_pred_incloud11_dyn_whole_cl = np.sqrt(var_pred_dyn_whole_cl[lwc_cloud1[76]>=1e-5])
        test_std_soundness_incloud11_dyn_whole_cl = (std_pred_incloud11_dyn_whole_cl-abs_err_incloud11_dyn_whole_cl)/std_pred_incloud11_dyn_whole_cl

        std_pred_soundness_incloud11_dyn_whole_cl = np.array([np.percentile(test_std_soundness_incloud11_dyn_whole_cl,0.3),np.percentile(test_std_soundness_incloud11_dyn_whole_cl,5),
        np.percentile(test_std_soundness_incloud11_dyn_whole_cl,32)])


        explained_variance = np.sum((zwind_data1[76][lwc_cloud1[76]>=1e-5] -
                            mean_pred_dyn_whole_cl[lwc_cloud1[76]>=1e-5])**2)
        total_variance = np.sum((zwind_data1[76][lwc_cloud1[76]>=1e-5]- zwind_train_dyn_whole_cl.mean())**2)
        r2_incloud11_dyn_whole_cl = 1 - explained_variance/total_variance

        err_incloud11_dyn_whole_cl = zwind_data1[76][lwc_cloud1[76]>=1e-5]-mean_pred_dyn_whole_cl[lwc_cloud1[76]>=1e-5]
        err_stats_incloud11_dyn_whole_cl = np.array([np.mean(err_incloud11_dyn_whole_cl),np.median(err_incloud11_dyn_whole_cl),
        np.std(err_incloud11_dyn_whole_cl),skew(err_incloud11_dyn_whole_cl)])
        ############################################## libgp RBF t,x,y, #####################################################
        M2_dyn_whole_cl = GPModel(ndim=4,nvar=1, kernel_string=b"CovSum(CovSEard, CovNoise)")
        #M3 = GPModel(ndim=4,nvar=1, kernel_string="CovExpArdPhi")
        #drones_train_normal_dyn_cs_wo_t = drones_train_normal_dyn_cs.copy()
        #drones_train_normal_dyn_cs_wo_t[:,0] = 1
        #drones_train_normal_dyn_cs_wo_phi = drones_train_normal_dyn_cs.copy()
        #drones_train_normal_dyn_cs_wo_phi[:,2] = 1
        M2_dyn_whole_cl.update(drones_train_dyn_whole_cl,zwind_train_dyn_whole_cl.T)

        ## Values in log scale
        input_opt = {"gp_opt_python":True,'gp_opt_iter':10,'gp_hyper_bounds':((-10, 10), (-10, 10), (-10, 10), (-10, 10),
                         (-10, 10),
                         (-10, 10)),'gp_discr_tol':-1}
        M2_dyn_whole_cl.optimize(input_opt)
        params2_dyn_whole_cl = np.asarray(M2_dyn_whole_cl.get_params()).reshape(6)

        predictions2_dyn_whole_cl = np.ndarray((1,2,16*140*150))

        #grid_unrolled_normal_dyn_cs_wo_t = grid_unrolled_normal_dyn_cs.copy()
        #grid_unrolled_normal_dyn_cs_wo_t[:,0] =1
        #grid_unrolled_normal_dyn_cs_wo_phi = grid_unrolled_normal_dyn_cs.copy()
        #grid_unrolled_normal_dyn_cs_wo_phi[:,2] = 1
        M2_dyn_whole_cl.predict(grid_unrolled_dyn_whole_cl,predictions2_dyn_whole_cl)

        mean_pred2_dyn_whole_cl = predictions2_dyn_whole_cl[0,0].reshape(-1,1)
        var_pred2_dyn_whole_cl = predictions2_dyn_whole_cl[0,1].reshape(-1,1)

        mean_pred2_dyn_whole_cl = mean_pred2_dyn_whole_cl.reshape((16,140,150))
        var_pred2_dyn_whole_cl = var_pred2_dyn_whole_cl.reshape((16,140,150))

        mean_std_pred_incloud12_dyn_whole_cl = np.mean(np.sqrt(var_pred2_dyn_whole_cl[lwc_cloud1[76]>=1e-5]))
        rmse_incloud12_dyn_whole_cl = np.sqrt(np.mean((zwind_data1[76][lwc_cloud1[76]>=1e-5]-mean_pred2_dyn_whole_cl[lwc_cloud1[76]>=1e-5])**2))

        abs_err_incloud12_dyn_whole_cl = np.abs(zwind_data1[76][lwc_cloud1[76]>=1e-5]-mean_pred2_dyn_whole_cl[lwc_cloud1[76]>=1e-5])
        std_pred_incloud12_dyn_whole_cl = np.sqrt(var_pred2_dyn_whole_cl[lwc_cloud1[76]>=1e-5])
        test_std_soundness_incloud12_dyn_whole_cl = (std_pred_incloud12_dyn_whole_cl-abs_err_incloud12_dyn_whole_cl)/std_pred_incloud12_dyn_whole_cl

        std_pred_soundness_incloud12_dyn_whole_cl = np.array([np.percentile(test_std_soundness_incloud12_dyn_whole_cl,0.3),np.percentile(test_std_soundness_incloud12_dyn_whole_cl,5),
        np.percentile(test_std_soundness_incloud12_dyn_whole_cl,32)])


        explained_variance = np.sum((zwind_data1[76][lwc_cloud1[76]>=1e-5] -
                            mean_pred2_dyn_whole_cl[lwc_cloud1[76]>=1e-5])**2)
        total_variance = np.sum((zwind_data1[76][lwc_cloud1[76]>=1e-5]- zwind_train_dyn_whole_cl.mean())**2)
        r2_incloud12_dyn_whole_cl = 1 - explained_variance/total_variance

        err_incloud12_dyn_whole_cl = zwind_data1[76][lwc_cloud1[76]>=1e-5]-mean_pred2_dyn_whole_cl[lwc_cloud1[76]>=1e-5]
        err_stats_incloud12_dyn_whole_cl = np.array([np.mean(err_incloud12_dyn_whole_cl),np.median(err_incloud12_dyn_whole_cl),
        np.std(err_incloud12_dyn_whole_cl),skew(err_incloud12_dyn_whole_cl)])
        ############################################## libgp normalized, corrected phi, variogram hypers #####################################################
        M3_dyn_whole_cl = GPModel(ndim=4,nvar=1, kernel_string=b"CovSum(CovExpArdPhi, CovNoise)")

        lt = np.log(51.026823559394558)
        lz = np.log(0.01*13.054954891415182)
        l_phi = np.log(23.025993674634258)
        lr = np.log(40.199201579845884)
        sigma2 = 0.5*np.log(0.84069334459964384)
        noise_var =0.5*np.log(noise[j]**2)

        params = np.array([lt,lz,l_phi,lr,sigma2,noise_var]).reshape(1,-1)
        params3_dyn_whole_cl = params.reshape(6)

        M3_dyn_whole_cl.set_params(params)
        M3_dyn_whole_cl.update(drones_train_normal_dyn_whole_cl,zwind_train_detrended_dyn_whole_cl.T)

        predictions3_dyn_whole_cl =np.nan*np.ndarray((1,2,16*140*150))
        M3_dyn_whole_cl.predict(grid_unrolled_normal_dyn_whole_cl,predictions3_dyn_whole_cl)

        mean_pred3_dyn_whole_cl = predictions3_dyn_whole_cl[0,0].reshape(-1,1)
        var_pred3_dyn_whole_cl = predictions3_dyn_whole_cl[0,1].reshape(-1,1)

        mean_pred3_dyn_whole_cl = mean_pred3_dyn_whole_cl + (zwind_com*interpolate_rtrend(grid_unrolled_normal_dyn_whole_cl[:,3],'linear')).reshape((-1,1))

        mean_pred3_dyn_whole_cl = mean_pred3_dyn_whole_cl.reshape((16,140,150))
        var_pred3_dyn_whole_cl = var_pred3_dyn_whole_cl.reshape((16,140,150))

        mean_std_pred_incloud13_dyn_whole_cl = np.mean(np.sqrt(var_pred3_dyn_whole_cl[lwc_cloud1[76]>=1e-5]))
        rmse_incloud13_dyn_whole_cl = np.sqrt(np.mean((zwind_data1[76][lwc_cloud1[76]>=1e-5]-mean_pred3_dyn_whole_cl[lwc_cloud1[76]>=1e-5])**2))

        abs_err_incloud13_dyn_whole_cl = np.abs(zwind_data1[76][lwc_cloud1[76]>=1e-5]-mean_pred3_dyn_whole_cl[lwc_cloud1[76]>=1e-5])
        std_pred_incloud13_dyn_whole_cl = np.sqrt(var_pred3_dyn_whole_cl[lwc_cloud1[76]>=1e-5])
        test_std_soundness_incloud13_dyn_whole_cl = (std_pred_incloud13_dyn_whole_cl-abs_err_incloud13_dyn_whole_cl)/std_pred_incloud13_dyn_whole_cl

        std_pred_soundness_incloud13_dyn_whole_cl = np.array([np.percentile(test_std_soundness_incloud13_dyn_whole_cl,0.3),np.percentile(test_std_soundness_incloud13_dyn_whole_cl,5),
        np.percentile(test_std_soundness_incloud13_dyn_whole_cl,32)])


        explained_variance = np.sum((zwind_data1[76][lwc_cloud1[76]>=1e-5] -
                            mean_pred3_dyn_whole_cl[lwc_cloud1[76]>=1e-5])**2)
        total_variance = np.sum((zwind_data1[76][lwc_cloud1[76]>=1e-5]- zwind_train_dyn_whole_cl.mean())**2)
        r2_incloud13_dyn_whole_cl = 1 - explained_variance/total_variance

        err_incloud13_dyn_whole_cl = zwind_data1[76][lwc_cloud1[76]>=1e-5]-mean_pred3_dyn_whole_cl[lwc_cloud1[76]>=1e-5]
        err_stats_incloud13_dyn_whole_cl = np.array([np.mean(err_incloud13_dyn_whole_cl),np.median(err_incloud13_dyn_whole_cl),
        np.std(err_incloud13_dyn_whole_cl),skew(err_incloud13_dyn_whole_cl)])
        ############################################## libgp normalized, corrected phi,optimized #####################################################
        M4_dyn_whole_cl = GPModel(ndim=4,nvar=1, kernel_string=b"CovSum(CovExpArdPhi, CovNoise)")
        #M3 = GPModel(ndim=4,nvar=1, kernel_string="CovExpArdPhi")
        #drones_train_normal_dyn_cs_wo_t = drones_train_normal_dyn_cs.copy()
        #drones_train_normal_dyn_cs_wo_t[:,0] = 1
        #drones_train_normal_dyn_cs_wo_phi = drones_train_normal_dyn_cs.copy()
        #drones_train_normal_dyn_cs_wo_phi[:,2] = 1
        M4_dyn_whole_cl.update(drones_train_normal_dyn_whole_cl,zwind_train_detrended_dyn_whole_cl.T)

        ######## Start at some random number to avoid problem of not optimizing
        #lt = np.random.randn(1)*3 + 3
        #lz = np.random.randn(1)*3 + 3
        #l_phi = np.random.randn(1)*3 + 3
        #lr = np.random.randn(1)*3 + 3
        #sigma2 = np.random.randn(1)*3 + 3
        #noise_var = np.random.randn(1)*3 + 3

        #params = np.array([lt,lz,l_phi,lr,sigma2,noise_var]).reshape(1,-1)
        #M4_dyn_whole_cl.set_params(params)

        ## Values in log scale
        input_opt = {"gp_opt_python":True,'gp_opt_iter':10,'gp_hyper_bounds':((-10, 10), (-10, 10), (-10, 10), (-10, 10),
                         (-10, 10),
                         (-10, 10)),'gp_discr_tol':-1}
        M4_dyn_whole_cl.optimize(input_opt)

        params4_dyn_whole_cl = np.asarray(M4_dyn_whole_cl.get_params()).reshape(6)
        predictions4_dyn_whole_cl = np.ndarray((1,2,16*140*150))
        #grid_unrolled_normal_dyn_cs_wo_t = grid_unrolled_normal_dyn_cs.copy()
        #grid_unrolled_normal_dyn_cs_wo_t[:,0] =1
        #grid_unrolled_normal_dyn_cs_wo_phi = grid_unrolled_normal_dyn_cs.copy()
        #grid_unrolled_normal_dyn_cs_wo_phi[:,2] = 1
        M4_dyn_whole_cl.predict(grid_unrolled_normal_dyn_whole_cl,predictions4_dyn_whole_cl)

        mean_pred4_dyn_whole_cl = predictions4_dyn_whole_cl[0,0].reshape(-1,1)
        var_pred4_dyn_whole_cl = predictions4_dyn_whole_cl[0,1].reshape(-1,1)

        mean_pred4_dyn_whole_cl = mean_pred4_dyn_whole_cl + (zwind_com*interpolate_rtrend(grid_unrolled_normal_dyn_whole_cl[:,3],'linear')).reshape((-1,1))

        mean_pred4_dyn_whole_cl = mean_pred4_dyn_whole_cl.reshape((16,140,150))
        var_pred4_dyn_whole_cl = var_pred4_dyn_whole_cl.reshape((16,140,150))

        mean_std_pred_incloud14_dyn_whole_cl = np.mean(np.sqrt(var_pred4_dyn_whole_cl[lwc_cloud1[76]>=1e-5]))
        rmse_incloud14_dyn_whole_cl = np.sqrt(np.mean((zwind_data1[76][lwc_cloud1[76]>=1e-5]-mean_pred4_dyn_whole_cl[lwc_cloud1[76]>=1e-5])**2))

        abs_err_incloud14_dyn_whole_cl = np.abs(zwind_data1[76][lwc_cloud1[76]>=1e-5]-mean_pred4_dyn_whole_cl[lwc_cloud1[76]>=1e-5])
        std_pred_incloud14_dyn_whole_cl = np.sqrt(var_pred4_dyn_whole_cl[lwc_cloud1[76]>=1e-5])
        test_std_soundness_incloud14_dyn_whole_cl = (std_pred_incloud14_dyn_whole_cl-abs_err_incloud14_dyn_whole_cl)/std_pred_incloud14_dyn_whole_cl

        std_pred_soundness_incloud14_dyn_whole_cl = np.array([np.percentile(test_std_soundness_incloud14_dyn_whole_cl,0.3),np.percentile(test_std_soundness_incloud14_dyn_whole_cl,5),
        np.percentile(test_std_soundness_incloud14_dyn_whole_cl,32)])


        explained_variance = np.sum((zwind_data1[76][lwc_cloud1[76]>=1e-5] -
                            mean_pred4_dyn_whole_cl[lwc_cloud1[76]>=1e-5])**2)
        total_variance = np.sum((zwind_data1[76][lwc_cloud1[76]>=1e-5]- zwind_train_dyn_whole_cl.mean())**2)
        r2_incloud14_dyn_whole_cl = 1 - explained_variance/total_variance

        err_incloud14_dyn_whole_cl = zwind_data1[76][lwc_cloud1[76]>=1e-5]-mean_pred4_dyn_whole_cl[lwc_cloud1[76]>=1e-5]
        err_stats_incloud14_dyn_whole_cl = np.array([np.mean(err_incloud14_dyn_whole_cl),np.median(err_incloud14_dyn_whole_cl),
        np.std(err_incloud14_dyn_whole_cl),skew(err_incloud14_dyn_whole_cl)])
        ############################################## libgp normalized, corrected phi,optimized starting from variogram hypers #####################################################
        M5_dyn_whole_cl = GPModel(ndim=4,nvar=1, kernel_string=b"CovSum(CovExpArdPhi, CovNoise)")
        #M3 = GPModel(ndim=4,nvar=1, kernel_string="CovExpArdPhi")
        #drones_train_normal_dyn_cs_wo_t = drones_train_normal_dyn_cs.copy()
        #drones_train_normal_dyn_cs_wo_t[:,0] = 1
        #drones_train_normal_dyn_cs_wo_phi = drones_train_normal_dyn_cs.copy()
        #drones_train_normal_dyn_cs_wo_phi[:,2] = 1
        M5_dyn_whole_cl.update(drones_train_normal_dyn_whole_cl,zwind_train_detrended_dyn_whole_cl.T)

        lt = np.log(51.026823559394558)
        lz = np.log(0.01*13.054954891415182)
        l_phi = np.log(23.025993674634258)
        lr = np.log(40.199201579845884)
        sigma2 = 0.5*np.log(0.84069334459964384)
        noise_var =0.5*np.log(noise[j]**2)

        params = np.array([lt,lz,l_phi,lr,sigma2,noise_var]).reshape(1,-1)
        M5_dyn_whole_cl.set_params(params)

        ## Values in log scale
        input_opt = {"gp_opt_python":True,'gp_opt_iter':10,'gp_hyper_bounds':((-10, 10), (-10, 10), (-10, 10), (-10, 10),
                         (-10, 10),
                         (-10, 10)),'gp_discr_tol':-1}
        M5_dyn_whole_cl.optimize(input_opt)
        params5_dyn_whole_cl = np.asarray(M5_dyn_whole_cl.get_params()).reshape(6)

        predictions5_dyn_whole_cl = np.ndarray((1,2,16*140*150))

        #grid_unrolled_normal_dyn_cs_wo_t = grid_unrolled_normal_dyn_cs.copy()
        #grid_unrolled_normal_dyn_cs_wo_t[:,0] =1
        #grid_unrolled_normal_dyn_cs_wo_phi = grid_unrolled_normal_dyn_cs.copy()
        #grid_unrolled_normal_dyn_cs_wo_phi[:,2] = 1
        M5_dyn_whole_cl.predict(grid_unrolled_normal_dyn_whole_cl,predictions5_dyn_whole_cl)

        mean_pred5_dyn_whole_cl = predictions5_dyn_whole_cl[0,0].reshape(-1,1)
        var_pred5_dyn_whole_cl = predictions5_dyn_whole_cl[0,1].reshape(-1,1)

        mean_pred5_dyn_whole_cl = mean_pred5_dyn_whole_cl + (zwind_com*interpolate_rtrend(grid_unrolled_normal_dyn_whole_cl[:,3],'linear')).reshape((-1,1))

        mean_pred5_dyn_whole_cl = mean_pred5_dyn_whole_cl.reshape((16,140,150))
        var_pred5_dyn_whole_cl = var_pred5_dyn_whole_cl.reshape((16,140,150))

        mean_std_pred_incloud15_dyn_whole_cl = np.mean(np.sqrt(var_pred5_dyn_whole_cl[lwc_cloud1[76]>=1e-5]))
        rmse_incloud15_dyn_whole_cl = np.sqrt(np.mean((zwind_data1[76][lwc_cloud1[76]>=1e-5]-mean_pred5_dyn_whole_cl[lwc_cloud1[76]>=1e-5])**2))

        abs_err_incloud15_dyn_whole_cl = np.abs(zwind_data1[76][lwc_cloud1[76]>=1e-5]-mean_pred5_dyn_whole_cl[lwc_cloud1[76]>=1e-5])
        std_pred_incloud15_dyn_whole_cl = np.sqrt(var_pred5_dyn_whole_cl[lwc_cloud1[76]>=1e-5])
        test_std_soundness_incloud15_dyn_whole_cl = (std_pred_incloud15_dyn_whole_cl-abs_err_incloud15_dyn_whole_cl)/std_pred_incloud15_dyn_whole_cl

        std_pred_soundness_incloud15_dyn_whole_cl = np.array([np.percentile(test_std_soundness_incloud15_dyn_whole_cl,0.3),np.percentile(test_std_soundness_incloud15_dyn_whole_cl,5),
        np.percentile(test_std_soundness_incloud15_dyn_whole_cl,32)])


        explained_variance = np.sum((zwind_data1[76][lwc_cloud1[76]>=1e-5] -
                            mean_pred5_dyn_whole_cl[lwc_cloud1[76]>=1e-5])**2)
        total_variance = np.sum((zwind_data1[76][lwc_cloud1[76]>=1e-5]- zwind_train_dyn_whole_cl.mean())**2)
        r2_incloud15_dyn_whole_cl = 1 - explained_variance/total_variance

        err_incloud15_dyn_whole_cl = zwind_data1[76][lwc_cloud1[76]>=1e-5]-mean_pred5_dyn_whole_cl[lwc_cloud1[76]>=1e-5]
        err_stats_incloud15_dyn_whole_cl = np.array([np.mean(err_incloud15_dyn_whole_cl),np.median(err_incloud15_dyn_whole_cl),
        np.std(err_incloud15_dyn_whole_cl),skew(err_incloud15_dyn_whole_cl)])
        ########################## GPy implementation normalized,Exponential uncorrected phi ############################
        try:
            kernel = GPy.kern.Exponential(input_dim=4,ARD=True)

            m_dyn_whole_cl = GPy.models.GPRegression(drones_train_normal_dyn_whole_cl,zwind_train_detrended_dyn_whole_cl,kernel)
            m_dyn_whole_cl.optimize_restarts(num_restarts = 10)

            params6_dyn_whole_cl = np.nan*np.ndarray(6)
            params6_dyn_whole_cl[[4,0,1,2,3,5]] = m_dyn_whole_cl.param_array

            predictions6_dyn_whole_cl = m_dyn_whole_cl.predict(grid_unrolled_normal_dyn_whole_cl)

            mean_pred6_dyn_whole_cl = predictions6_dyn_whole_cl[0]
            var_pred6_dyn_whole_cl = predictions6_dyn_whole_cl[1]

            mean_pred6_dyn_whole_cl = mean_pred6_dyn_whole_cl + (zwind_com*interpolate_rtrend(grid_unrolled_normal_dyn_whole_cl[:,3],'linear')).reshape((-1,1))

            mean_pred6_dyn_whole_cl = mean_pred6_dyn_whole_cl.reshape((16,140,150))
            var_pred6_dyn_whole_cl = var_pred6_dyn_whole_cl.reshape((16,140,150))

            mean_std_pred_incloud16_dyn_whole_cl = np.mean(np.sqrt(var_pred6_dyn_whole_cl[lwc_cloud1[76]>=1e-5]))
            rmse_incloud16_dyn_whole_cl = np.sqrt(np.mean((zwind_data1[76][lwc_cloud1[76]>=1e-5]-mean_pred6_dyn_whole_cl[lwc_cloud1[76]>=1e-5])**2))

            abs_err_incloud16_dyn_whole_cl = np.abs(zwind_data1[76][lwc_cloud1[76]>=1e-5]-mean_pred6_dyn_whole_cl[lwc_cloud1[76]>=1e-5])
            std_pred_incloud16_dyn_whole_cl = np.sqrt(var_pred6_dyn_whole_cl[lwc_cloud1[76]>=1e-5])
            test_std_soundness_incloud16_dyn_whole_cl = (std_pred_incloud16_dyn_whole_cl-abs_err_incloud16_dyn_whole_cl)/std_pred_incloud16_dyn_whole_cl

            std_pred_soundness_incloud16_dyn_whole_cl = np.array([np.percentile(test_std_soundness_incloud16_dyn_whole_cl,0.3),np.percentile(test_std_soundness_incloud16_dyn_whole_cl,5),
            np.percentile(test_std_soundness_incloud16_dyn_whole_cl,32)])


            explained_variance = np.sum((zwind_data1[76][lwc_cloud1[76]>=1e-5] -
                                mean_pred6_dyn_whole_cl[lwc_cloud1[76]>=1e-5])**2)
            total_variance = np.sum((zwind_data1[76][lwc_cloud1[76]>=1e-5]- zwind_train_dyn_whole_cl.mean())**2)
            r2_incloud16_dyn_whole_cl = 1 - explained_variance/total_variance

            err_incloud16_dyn_whole_cl = zwind_data1[76][lwc_cloud1[76]>=1e-5]-mean_pred6_dyn_whole_cl[lwc_cloud1[76]>=1e-5]
            err_stats_incloud16_dyn_whole_cl = np.array([np.mean(err_incloud16_dyn_whole_cl),np.median(err_incloud16_dyn_whole_cl),
            np.std(err_incloud16_dyn_whole_cl),skew(err_incloud16_dyn_whole_cl)])

            ########################## GPy implementation RBF, t,x,y ############################
            kernel = GPy.kern.RBF(input_dim=4,ARD=True)

            m2_dyn_whole_cl = GPy.models.GPRegression(drones_train_dyn_whole_cl,zwind_train_dyn_whole_cl,kernel)
            m2_dyn_whole_cl.optimize_restarts(num_restarts = 10)

            params7_dyn_whole_cl = np.nan*np.ndarray(6)
            params7_dyn_whole_cl[[4,0,1,2,3,5]] = m2_dyn_whole_cl.param_array

            predictions7_dyn_whole_cl = m2_dyn_whole_cl.predict(grid_unrolled_dyn_whole_cl)

            mean_pred7_dyn_whole_cl = predictions7_dyn_whole_cl[0]
            var_pred7_dyn_whole_cl = predictions7_dyn_whole_cl[1]

            mean_pred7_dyn_whole_cl = mean_pred7_dyn_whole_cl.reshape((16,140,150))
            var_pred7_dyn_whole_cl = var_pred7_dyn_whole_cl.reshape((16,140,150))

            mean_std_pred_incloud17_dyn_whole_cl = np.mean(np.sqrt(var_pred7_dyn_whole_cl[lwc_cloud1[76]>=1e-5]))
            rmse_incloud17_dyn_whole_cl = np.sqrt(np.mean((zwind_data1[76][lwc_cloud1[76]>=1e-5]-mean_pred7_dyn_whole_cl[lwc_cloud1[76]>=1e-5])**2))

            abs_err_incloud17_dyn_whole_cl = np.abs(zwind_data1[76][lwc_cloud1[76]>=1e-5]-mean_pred7_dyn_whole_cl[lwc_cloud1[76]>=1e-5])
            std_pred_incloud17_dyn_whole_cl = np.sqrt(var_pred7_dyn_whole_cl[lwc_cloud1[76]>=1e-5])
            test_std_soundness_incloud17_dyn_whole_cl = (std_pred_incloud17_dyn_whole_cl-abs_err_incloud17_dyn_whole_cl)/std_pred_incloud17_dyn_whole_cl

            std_pred_soundness_incloud17_dyn_whole_cl = np.array([np.percentile(test_std_soundness_incloud17_dyn_whole_cl,0.3),np.percentile(test_std_soundness_incloud17_dyn_whole_cl,5),
            np.percentile(test_std_soundness_incloud17_dyn_whole_cl,32)])


            explained_variance = np.sum((zwind_data1[76][lwc_cloud1[76]>=1e-5] -
                                mean_pred7_dyn_whole_cl[lwc_cloud1[76]>=1e-5])**2)
            total_variance = np.sum((zwind_data1[76][lwc_cloud1[76]>=1e-5]- zwind_train_dyn_whole_cl.mean())**2)
            r2_incloud17_dyn_whole_cl = 1 - explained_variance/total_variance

            err_incloud17_dyn_whole_cl = zwind_data1[76][lwc_cloud1[76]>=1e-5]-mean_pred7_dyn_whole_cl[lwc_cloud1[76]>=1e-5]
            err_stats_incloud17_dyn_whole_cl = np.array([np.mean(err_incloud17_dyn_whole_cl),np.median(err_incloud17_dyn_whole_cl),
            np.std(err_incloud17_dyn_whole_cl),skew(err_incloud17_dyn_whole_cl)])

        except KeyboardInterrupt:
            print('KeyboardInterrupt')
            break
        except:
            print('Something went wrong with LinAlg,I think, please continue')
        ########################################### libgp, SE, xy detrended
        M6_dyn_whole_cl = GPModel(ndim=4,nvar=1, kernel_string=b"CovSum(CovSEard, CovNoise)")
        M6_dyn_whole_cl.update(drones_train_dyn_whole_cl,zwind_train_detrended_dyn_whole_cl.T)

        ## Values in log scale
        input_opt = {"gp_opt_python":True,'gp_opt_iter':10,'gp_hyper_bounds':((-10, 10), (-10, 10), (-10, 10), (-10, 10),
                         (-10, 10),
                         (-10, 10)),'gp_discr_tol':-1}

        M6_dyn_whole_cl.optimize(input_opt)
        params8_dyn_whole_cl = np.asarray(M6_dyn_whole_cl.get_params()).reshape(6)
        predictions8_dyn_whole_cl = np.ndarray((1,2,16*140*150))

        M6_dyn_whole_cl.predict(grid_unrolled_dyn_whole_cl,predictions8_dyn_whole_cl)

        mean_pred8_dyn_whole_cl = predictions8_dyn_whole_cl[0,0].reshape(-1,1)
        var_pred8_dyn_whole_cl = predictions8_dyn_whole_cl[0,1].reshape(-1,1)

        mean_pred8_dyn_whole_cl = mean_pred8_dyn_whole_cl + zwind_com*interpolate_rtrend(grid_unrolled_normal_dyn_whole_cl[:,3],'linear').reshape((-1,1))

        mean_pred8_dyn_whole_cl = mean_pred8_dyn_whole_cl.reshape((16,140,150))
        var_pred8_dyn_whole_cl = var_pred8_dyn_whole_cl.reshape((16,140,150))

        mean_std_pred_incloud18_dyn_whole_cl = np.mean(np.sqrt(var_pred8_dyn_whole_cl[lwc_cloud1[76]>=1e-5]))
        rmse_incloud18_dyn_whole_cl = np.sqrt(np.mean((zwind_data1[76][lwc_cloud1[76]>=1e-5]-mean_pred8_dyn_whole_cl[lwc_cloud1[76]>=1e-5])**2))

        abs_err_incloud18_dyn_whole_cl = np.abs(zwind_data1[76][lwc_cloud1[76]>=1e-5]-mean_pred8_dyn_whole_cl[lwc_cloud1[76]>=1e-5])
        std_pred_incloud18_dyn_whole_cl = np.sqrt(var_pred8_dyn_whole_cl[lwc_cloud1[76]>=1e-5])
        test_std_soundness_incloud18_dyn_whole_cl = (std_pred_incloud18_dyn_whole_cl-abs_err_incloud18_dyn_whole_cl)/std_pred_incloud18_dyn_whole_cl

        std_pred_soundness_incloud18_dyn_whole_cl = np.array([np.percentile(test_std_soundness_incloud18_dyn_whole_cl,0.3),np.percentile(test_std_soundness_incloud18_dyn_whole_cl,5),
        np.percentile(test_std_soundness_incloud18_dyn_whole_cl,32)])


        explained_variance = np.sum((zwind_data1[76][lwc_cloud1[76]>=1e-5] -
                            mean_pred8_dyn_whole_cl[lwc_cloud1[76]>=1e-5])**2)
        total_variance = np.sum((zwind_data1[76][lwc_cloud1[76]>=1e-5]- zwind_train_dyn_whole_cl.mean())**2)
        r2_incloud18_dyn_whole_cl = 1 - explained_variance/total_variance


        err_incloud18_dyn_whole_cl = zwind_data1[76][lwc_cloud1[76]>=1e-5]-mean_pred8_dyn_whole_cl[lwc_cloud1[76]>=1e-5]
        err_stats_incloud18_dyn_whole_cl = np.array([np.mean(err_incloud18_dyn_whole_cl),np.median(err_incloud18_dyn_whole_cl),np.std(err_incloud18_dyn_whole_cl),
        skew(err_incloud18_dyn_whole_cl)])
        ########################################### libgp Exp xy, detrended data
        M7_dyn_whole_cl = GPModel(ndim=4,nvar=1, kernel_string=b"CovSum(CovExpArd, CovNoise)")
        M7_dyn_whole_cl.update(drones_train_dyn_whole_cl,zwind_train_detrended_dyn_whole_cl.T)

        ## Values in log scale
        input_opt = {"gp_opt_python":True,'gp_opt_iter':10,'gp_hyper_bounds':((-10, 10), (-10, 10), (-10, 10), (-10, 10),
                         (-10, 10),
                         (-10, 10)),'gp_discr_tol':-1}

        M7_dyn_whole_cl.optimize(input_opt)
        params9_dyn_whole_cl = np.asarray(M7_dyn_whole_cl.get_params()).reshape(6)
        predictions9_dyn_whole_cl = np.ndarray((1,2,16*140*150))

        M7_dyn_whole_cl.predict(grid_unrolled_dyn_whole_cl,predictions9_dyn_whole_cl)

        mean_pred9_dyn_whole_cl = predictions9_dyn_whole_cl[0,0].reshape(-1,1)
        var_pred9_dyn_whole_cl = predictions9_dyn_whole_cl[0,1].reshape(-1,1)

        mean_pred9_dyn_whole_cl = mean_pred9_dyn_whole_cl + zwind_com*interpolate_rtrend(grid_unrolled_normal_dyn_whole_cl[:,3],'linear').reshape((-1,1))

        mean_pred9_dyn_whole_cl = mean_pred9_dyn_whole_cl.reshape((16,140,150))
        var_pred9_dyn_whole_cl = var_pred9_dyn_whole_cl.reshape((16,140,150))

        mean_std_pred_incloud19_dyn_whole_cl = np.mean(np.sqrt(var_pred9_dyn_whole_cl[lwc_cloud1[76]>=1e-5]))
        rmse_incloud19_dyn_whole_cl = np.sqrt(np.mean((zwind_data1[76][lwc_cloud1[76]>=1e-5]-mean_pred9_dyn_whole_cl[lwc_cloud1[76]>=1e-5])**2))

        abs_err_incloud19_dyn_whole_cl = np.abs(zwind_data1[76][lwc_cloud1[76]>=1e-5]-mean_pred9_dyn_whole_cl[lwc_cloud1[76]>=1e-5])
        std_pred_incloud19_dyn_whole_cl = np.sqrt(var_pred9_dyn_whole_cl[lwc_cloud1[76]>=1e-5])
        test_std_soundness_incloud19_dyn_whole_cl = (std_pred_incloud19_dyn_whole_cl-abs_err_incloud19_dyn_whole_cl)/std_pred_incloud19_dyn_whole_cl

        std_pred_soundness_incloud19_dyn_whole_cl = np.array([np.percentile(test_std_soundness_incloud19_dyn_whole_cl,0.3),np.percentile(test_std_soundness_incloud19_dyn_whole_cl,5),
        np.percentile(test_std_soundness_incloud19_dyn_whole_cl,32)])


        explained_variance = np.sum((zwind_data1[76][lwc_cloud1[76]>=1e-5] -
                            mean_pred9_dyn_whole_cl[lwc_cloud1[76]>=1e-5])**2)
        total_variance = np.sum((zwind_data1[76][lwc_cloud1[76]>=1e-5]- zwind_train_dyn_whole_cl.mean())**2)
        r2_incloud19_dyn_whole_cl = 1 - explained_variance/total_variance


        err_incloud19_dyn_whole_cl = zwind_data1[76][lwc_cloud1[76]>=1e-5]-mean_pred9_dyn_whole_cl[lwc_cloud1[76]>=1e-5]
        err_stats_incloud19_dyn_whole_cl = np.array([np.mean(err_incloud19_dyn_whole_cl),np.median(err_incloud19_dyn_whole_cl),
        np.std(err_incloud19_dyn_whole_cl),skew(err_incloud19_dyn_whole_cl)])




        ################################ Saving Results #######################

        all_rmse_dyn_whole_cl[i,j] = np.array([rmse_incloud11_dyn_whole_cl, rmse_incloud12_dyn_whole_cl, rmse_incloud13_dyn_whole_cl, rmse_incloud14_dyn_whole_cl,
        rmse_incloud15_dyn_whole_cl, rmse_incloud16_dyn_whole_cl, rmse_incloud17_dyn_whole_cl,rmse_trend_dyn_whole_cl, rmse_mean_dyn_whole_cl,
        rmse_incloud18_dyn_whole_cl, rmse_incloud19_dyn_whole_cl ])


        all_r2_dyn_whole_cl[i,j] = np.array([r2_incloud11_dyn_whole_cl,r2_incloud12_dyn_whole_cl,r2_incloud13_dyn_whole_cl,r2_incloud14_dyn_whole_cl,
        r2_incloud15_dyn_whole_cl, r2_incloud16_dyn_whole_cl,r2_incloud17_dyn_whole_cl, r2_trend_dyn_whole_cl, r2_mean_dyn_whole_cl,
        r2_incloud18_dyn_whole_cl,r2_incloud19_dyn_whole_cl])


        all_std_pred_dyn_whole_cl[i,j] = np.array([mean_std_pred_incloud11_dyn_whole_cl, mean_std_pred_incloud12_dyn_whole_cl, mean_std_pred_incloud13_dyn_whole_cl,
        mean_std_pred_incloud14_dyn_whole_cl, mean_std_pred_incloud15_dyn_whole_cl, mean_std_pred_incloud16_dyn_whole_cl, mean_std_pred_incloud17_dyn_whole_cl,
        mean_std_pred_incloud18_dyn_whole_cl, mean_std_pred_incloud19_dyn_whole_cl])

        all_std_pred_soundness_dyn_whole_cl[i,j] = np.array([std_pred_soundness_incloud11_dyn_whole_cl, std_pred_soundness_incloud12_dyn_whole_cl,
        std_pred_soundness_incloud13_dyn_whole_cl,std_pred_soundness_incloud14_dyn_whole_cl, std_pred_soundness_incloud15_dyn_whole_cl, std_pred_soundness_incloud16_dyn_whole_cl,
        std_pred_soundness_incloud17_dyn_whole_cl, std_pred_soundness_incloud18_dyn_whole_cl,std_pred_soundness_incloud19_dyn_whole_cl])

        all_hypers_dyn_whole_cl[i,j] = np.array([params1_dyn_whole_cl,params2_dyn_whole_cl,params3_dyn_whole_cl,params4_dyn_whole_cl,
        params5_dyn_whole_cl,params6_dyn_whole_cl,params7_dyn_whole_cl,params8_dyn_whole_cl,params9_dyn_whole_cl])

        all_error_stats_dyn_whole_cl[i,j] = np.array([err_stats_incloud11_dyn_whole_cl,err_stats_incloud12_dyn_whole_cl, err_stats_incloud13_dyn_whole_cl,
        err_stats_incloud14_dyn_whole_cl, err_stats_incloud15_dyn_whole_cl, err_stats_incloud16_dyn_whole_cl,err_stats_incloud17_dyn_whole_cl,
        err_stats_trend_dyn_whole_cl,err_stats_mean_dyn_whole_cl,err_stats_incloud18_dyn_whole_cl,err_stats_incloud19_dyn_whole_cl])

        outfile = '/home/dselle/Skyscanner/data_exploration/results/TestGP/dump_results_test_whole_cl_gp2.npz'
        np.savez(outfile, all_rmse_dyn_whole_cl = all_rmse_dyn_whole_cl, all_r2_dyn_whole_cl = all_r2_dyn_whole_cl, all_std_pred_dyn_whole_cl = all_std_pred_dyn_whole_cl,
        all_std_pred_soundness_dyn_whole_cl = all_std_pred_soundness_dyn_whole_cl,all_hypers_dyn_whole_cl=all_hypers_dyn_whole_cl,
        all_error_stats_dyn_whole_cl=all_error_stats_dyn_whole_cl)

        time2 = datetime.datetime.now()
        print('Total progress:{}%'.format((1280+4*(i)+(j+1))/1360*100))
        print(time2-time1)
        #except ValueError:
            #print('Something happened')




######################################## Load results summary all experiments #####################
### Most recent file with all desired analysis of Cross-section
outfile = "/home/dselle/Skyscanner/data_exploration/results/TestGP/dump_results_tests_cs_gp2.npz"
all_arrays_cs = np.load(outfile)

all_rmse_static_cs = all_arrays_cs['all_rmse_static_cs']
all_r2_static_cs = all_arrays_cs['all_r2_static_cs']
all_std_pred_static_cs = all_arrays_cs['all_std_pred_static_cs']
all_std_pred_soundness_static_cs=all_arrays_cs['all_std_pred_soundness_static_cs']
all_len_train_data_static_cs = all_arrays_cs['all_len_train_data_static_cs']
all_hypers_static_cs = all_arrays_cs['all_hypers_static_cs']
all_error_stats_static_cs = all_arrays_cs['all_error_stats_static_cs']
all_vars_static_cs = all_arrays_cs['all_vars_static_cs']


all_rmse_dyn_cs = all_arrays_cs['all_rmse_dyn_cs']
all_r2_dyn_cs = all_arrays_cs['all_r2_dyn_cs']
all_std_pred_dyn_cs = all_arrays_cs['all_std_pred_dyn_cs']
all_std_pred_soundness_dyn_cs = all_arrays_cs['all_std_pred_soundness_dyn_cs']
all_len_train_data_dyn_cs =all_arrays_cs['all_len_train_data_dyn_cs']
all_hypers_dyn_cs = all_arrays_cs['all_hypers_dyn_cs']
all_error_stats_dyn_cs = all_arrays_cs['all_error_stats_dyn_cs']
all_vars_dyn_cs = all_arrays_cs['all_vars_dyn_cs']

### Most recent file with all desired analysis of the Whole Cloud
outfile = '/home/dselle/Skyscanner/data_exploration/results/TestGP/dump_results_test_whole_cl_gp2.npz'

all_arrays_whole_cl = np.load(outfile)

all_rmse_dyn_whole_cl = all_arrays_whole_cl['all_rmse_dyn_whole_cl']
all_r2_dyn_whole_cl = all_arrays_whole_cl['all_r2_dyn_whole_cl']
all_std_pred_dyn_whole_cl = all_arrays_whole_cl['all_std_pred_dyn_whole_cl']
all_std_pred_soundness_dyn_whole_cl = all_arrays_whole_cl['all_std_pred_soundness_dyn_whole_cl']
all_hypers_dyn_whole_cl = all_arrays_whole_cl['all_hypers_dyn_whole_cl']
all_error_stats_dyn_whole_cl = all_arrays_whole_cl['all_error_stats_dyn_whole_cl']


################## Plotting results
all_rmse_static_cs_median = np.nanmedian(all_rmse_static_cs,axis=(0,1))
all_rmse_dyn_cs_median = np.nanmedian(all_rmse_dyn_cs,axis=(0,1))
all_rmse_dyn_whole_cl_median = np.nanmedian(all_rmse_dyn_whole_cl,axis=(0))

all_rmse_static_cs_mad = mad(all_rmse_static_cs,axis=(0,1))
all_rmse_dyn_cs_mad = mad(all_rmse_dyn_cs,axis=(0,1))
all_rmse_dyn_whole_cl_mad = mad(all_rmse_dyn_whole_cl,axis=(0))

all_std_pred_static_cs_median = np.nanmedian(all_std_pred_static_cs,axis=(0,1))
all_std_pred_dyn_cs_median = np.nanmedian(all_std_pred_dyn_cs,axis=(0,1))
all_std_pred_dyn_whole_cl_median = np.nanmedian(all_std_pred_dyn_whole_cl,axis=(0))


all_std_pred_static_cs_mad = mad(all_std_pred_static_cs,axis=(0,1))
all_std_pred_dyn_cs_mad = mad(all_std_pred_dyn_cs, axis=(0,1))
all_std_pred_dyn_whole_cl_mad = mad(all_std_pred_dyn_whole_cl,axis=(0))


################## plotting RMSE and Variance
font = {'size'   : 16}

plt.rc('font', **font)


k = 4
noise = [0.001,0.1,0.25,0.5,0.75]

ind = np.arange(1,14,5)  # the x locations for the groups
width = 0.40
# plotting RMSE and Variance
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

### Exp x,y
rmse_incloud11_median = all_rmse_static_cs_median[k,0]
rmse_incloud11_dyn_cs_median = all_rmse_dyn_cs_median[k,0]
rmse_incloud11_dyn_whole_cl_median = all_rmse_dyn_whole_cl_median[k,0]

ExpRmseMedian = (rmse_incloud11_median,rmse_incloud11_dyn_cs_median,rmse_incloud11_dyn_whole_cl_median)

rmse_incloud11_mad = all_rmse_static_cs_mad[k,0]
rmse_incloud11_dyn_cs_mad = all_rmse_dyn_cs_mad[k,0]
rmse_incloud11_dyn_whole_cl_mad = all_rmse_dyn_whole_cl_mad[k,0]

ExpRmseMad = (rmse_incloud11_mad,rmse_incloud11_dyn_cs_mad,rmse_incloud11_dyn_whole_cl_mad)
rects1 = ax1.bar(ind, ExpRmseMedian, width, color='r',yerr = ExpRmseMad)

### SE x,y
rmse_incloud12_median = all_rmse_static_cs_median[k,1]
rmse_incloud12_dyn_cs_median = all_rmse_dyn_cs_median[k,1]
rmse_incloud12_dyn_whole_cl_median = all_rmse_dyn_whole_cl_median[k,1]

SErmseMedian = (rmse_incloud12_median,rmse_incloud12_dyn_cs_median,rmse_incloud12_dyn_whole_cl_median)

rmse_incloud12_mad = all_rmse_static_cs_mad[k,1]
rmse_incloud12_dyn_cs_mad = all_rmse_dyn_cs_mad[k,1]
rmse_incloud12_dyn_whole_cl_mad = all_rmse_dyn_whole_cl_mad[k,1]

SErmseMad = (rmse_incloud12_mad,rmse_incloud12_dyn_cs_mad,rmse_incloud12_dyn_whole_cl_mad)

rects2 = ax1.bar(ind + width, SErmseMedian, width, color='y',yerr = SErmseMad)

#### ExpArdPhi 1
rmse_incloud13_median = all_rmse_static_cs_median[k,2]
rmse_incloud13_dyn_cs_median = all_rmse_dyn_cs_median[k,2]
rmse_incloud13_dyn_whole_cl_median = all_rmse_dyn_whole_cl_median[k,2]

ExpNorm1RmseMedian = (rmse_incloud13_median,rmse_incloud13_dyn_cs_median, rmse_incloud13_dyn_whole_cl_median)

rmse_incloud13_mad = all_rmse_static_cs_mad[k,2]
rmse_incloud13_dyn_cs_mad = all_rmse_dyn_cs_mad[k,2]
rmse_incloud11_dyn_whole_cl_mad = all_rmse_dyn_whole_cl_mad[k,2]

ExpNorm1RmseMad = (rmse_incloud13_mad,rmse_incloud13_dyn_cs_mad,rmse_incloud11_dyn_whole_cl_mad)

rects3 = ax1.bar(ind + 2*width, ExpNorm1RmseMedian, width, color='b',yerr = ExpNorm1RmseMad )

#### ExpArdPhi 2
rmse_incloud14_median = all_rmse_static_cs_median[k,3]
rmse_incloud14_dyn_cs_median = all_rmse_dyn_cs_median[k,3]
rmse_incloud14_dyn_whole_cl_median = all_rmse_dyn_whole_cl_median[k,3]

ExpNorm2RmseMedian = (rmse_incloud14_median,rmse_incloud14_dyn_cs_median,rmse_incloud14_dyn_whole_cl_median)

rmse_incloud14_mad = all_rmse_static_cs_mad[k,3]
rmse_incloud14_dyn_cs_mad = all_rmse_dyn_cs_mad[k,3]
rmse_incloud14_dyn_whole_cl_mad = all_rmse_dyn_whole_cl_mad[k,3]


ExpNorm2RmseMad = (rmse_incloud14_mad,rmse_incloud14_dyn_cs_mad, rmse_incloud14_dyn_whole_cl_mad)
rects4 = ax1.bar(ind + 3*width, ExpNorm2RmseMedian, width, color='g',yerr = ExpNorm2RmseMad)

#### ExpArdPhi 3
rmse_incloud15_median = all_rmse_static_cs_median[k,4]
rmse_incloud15_dyn_cs_median = all_rmse_dyn_cs_median[k,4]
rmse_incloud15_dyn_whole_cl_median = all_rmse_dyn_whole_cl_median[k,4]

ExpNorm3RmseMedian = (rmse_incloud15_median,rmse_incloud15_dyn_cs_median,rmse_incloud15_dyn_whole_cl_median)

rmse_incloud15_mad = all_rmse_static_cs_mad[k,4]
rmse_incloud15_dyn_cs_mad = all_rmse_dyn_cs_mad[k,4]
rmse_incloud15_dyn_whole_cl_mad = all_rmse_dyn_whole_cl_mad[k,4]

ExpNorm3RmseMad = (rmse_incloud15_mad,rmse_incloud15_dyn_cs_mad,rmse_incloud15_dyn_whole_cl_mad)

rects5 = ax1.bar(ind + 4*width, ExpNorm3RmseMedian, width, color='k',yerr = ExpNorm3RmseMad)
####### Sq. Exp. Detrend
rmse_incloud18_median = all_rmse_static_cs_median[k,9]
rmse_incloud18_dyn_cs_median = all_rmse_dyn_cs_median[k,9]
rmse_incloud18_dyn_whole_cl_median = all_rmse_dyn_whole_cl_median[k,9]

SEdetrRmseMedian = (rmse_incloud18_median,rmse_incloud18_dyn_cs_median,rmse_incloud18_dyn_whole_cl_median)

rmse_incloud18_mad = all_rmse_static_cs_mad[k,9]
rmse_incloud18_dyn_cs_mad = all_rmse_dyn_cs_mad[k,9]
rmse_incloud18_dyn_whole_cl_mad = all_rmse_dyn_whole_cl_mad[k,9]

SEdetrRmseMad = (rmse_incloud18_mad,rmse_incloud18_dyn_cs_mad,rmse_incloud18_dyn_whole_cl_mad)

rects6 = ax1.bar(ind + 5*width, SEdetrRmseMedian, width, color='#e67e22',yerr = SEdetrRmseMad)

###### Exp. Detrend
rmse_incloud19_median = all_rmse_static_cs_median[k,10]
rmse_incloud19_dyn_cs_median = all_rmse_dyn_cs_median[k,10]
rmse_incloud19_dyn_whole_cl_median = all_rmse_dyn_whole_cl_median[k,10]

ExpDetrRmseMedian = (rmse_incloud19_median,rmse_incloud19_dyn_cs_median,rmse_incloud19_dyn_whole_cl_median)

rmse_incloud19_mad = all_rmse_static_cs_mad[k,10]
rmse_incloud19_dyn_cs_mad = all_rmse_dyn_cs_mad[k,10]
rmse_incloud19_dyn_whole_cl_mad = all_rmse_dyn_whole_cl_mad[k,10]

ExpDetrRmseMad = (rmse_incloud19_mad,rmse_incloud19_dyn_cs_mad,rmse_incloud19_dyn_whole_cl_mad)

rects7 = ax1.bar(ind + 6*width, ExpDetrRmseMedian , width, color='#7d3c98',yerr = ExpDetrRmseMad)

#### Trend
rmse_trend_median = all_rmse_static_cs_median[k,7]
rmse_trend_dyn_cs_median = all_rmse_dyn_cs_median[k,7]
rmse_trend_dyn_whole_cl_median = all_rmse_dyn_whole_cl_median[k,7]

TrendRmseMedian = (rmse_trend_median,rmse_trend_dyn_cs_median,rmse_trend_dyn_whole_cl_median)

rmse_trend_mad = all_rmse_static_cs_mad[k,7]
rmse_trend_dyn_cs_mad = all_rmse_dyn_cs_mad[k,7]
rmse_trend_dyn_whole_cl_mad = all_rmse_dyn_whole_cl_mad[k,7]

TrendRmseMad = (rmse_trend_mad,rmse_trend_dyn_cs_mad,rmse_trend_dyn_whole_cl_mad)

rects8 = ax1.bar(ind + 7*width, TrendRmseMedian, width, color='c',yerr = TrendRmseMad)

### Zwind Train Mean
rmse_mean_median = all_rmse_static_cs_median[k,8]
rmse_mean_dyn_cs_median = all_rmse_dyn_cs_median[k,8]
rmse_mean_dyn_whole_cl_median = all_rmse_dyn_whole_cl_median[k,8]

MeanRmseMedian = (rmse_mean_median,rmse_mean_dyn_cs_median,rmse_mean_dyn_whole_cl_median)

rmse_mean_mad = all_rmse_static_cs_mad[k,8]
rmse_mean_dyn_cs_mad = all_rmse_dyn_cs_mad[k,8]
rmse_mean_dyn_whole_cl_mad = all_rmse_dyn_whole_cl_mad[k,8]

MeanRmseMad = (rmse_mean_mad,rmse_mean_dyn_cs_mad,rmse_mean_dyn_whole_cl_mad)

rects9 = ax1.bar(ind + 8*width, MeanRmseMedian, width, color='m',yerr = MeanRmseMad)

# add some text for labels, title and axes ticks
ax1.set_ylabel('RMSE')
ax1.set_title('RMSE Cloud1 Experiments, noise_std ={} m/s'.format(float(noise[k])))
ax1.set_xticks(ind + 4.5*width)
ax1.set_xticklabels(('Static CS', 'Dyn. CS','Dyn. Whole Cl.'))
ax1.set_ylim(0,2.5)
ax1.legend((rects1[0], rects2[0],rects3[0],rects4[0],rects5[0],rects6[0],rects7[0],rects8[0],rects9[0]), ('Exp.', 'Sq. Exp','Exp. norm1','Exp. norm2',
'Exp. norm3','Sq.Exp Detr.','Exp. Detr.','Trend','Mean'))

############### Predicted Standard Deviation
### Exp x,y
mean_std_pred_incloud11_median = all_std_pred_static_cs_median[k,0]
mean_std_pred_incloud11_dyn_cs_median = all_std_pred_dyn_cs_median[k,0]
mean_std_pred_incloud11_dyn_whole_cl_median = all_std_pred_dyn_whole_cl_median[k,0]

ExpStdPredMedian = (mean_std_pred_incloud11_median,mean_std_pred_incloud11_dyn_cs_median,mean_std_pred_incloud11_dyn_whole_cl_median)

mean_std_pred_incloud11_mad = all_std_pred_static_cs_mad[k,0]
mean_std_pred_incloud11_dyn_cs_mad = all_std_pred_dyn_cs_mad[k,0]
mean_std_pred_incloud11_dyn_whole_cl_mad = all_std_pred_dyn_whole_cl_mad[k,0]

ExpStdPredMad = (mean_std_pred_incloud11_mad,mean_std_pred_incloud11_dyn_cs_mad,mean_std_pred_incloud11_dyn_whole_cl_mad)
rects1 = ax2.bar(ind, ExpStdPredMedian, width, color='r',yerr = ExpStdPredMad)

### SE x,y
mean_std_pred_incloud12_median = all_std_pred_static_cs_median[k,1]
mean_std_incloud12_dyn_cs_median = all_std_pred_dyn_cs_median[k,1]
mean_std_pred_incloud12_dyn_whole_cl_median = all_std_pred_dyn_whole_cl_median[k,1]


SEstdPredMedian = (mean_std_pred_incloud12_median,mean_std_incloud12_dyn_cs_median,mean_std_pred_incloud12_dyn_whole_cl_median)

mean_std_pred_incloud12_mad = all_std_pred_static_cs_mad[k,1]
mean_std_pred_incloud12_dyn_cs_mad = all_std_pred_dyn_cs_mad[k,1]
mean_std_pred_incloud12_dyn_whole_cl_mad = all_std_pred_dyn_whole_cl_mad[k,1]


SEstdPredMad = (mean_std_pred_incloud12_mad,mean_std_pred_incloud12_dyn_cs_mad,mean_std_pred_incloud12_dyn_whole_cl_mad)
rects2 = ax2.bar(ind+width, SEstdPredMedian, width, color='y',yerr = SEstdPredMad)

### ExpArdPhi 1

mean_std_pred_incloud13_median = all_std_pred_static_cs_median[k,2]
mean_std_incloud13_dyn_cs_median = all_std_pred_dyn_cs_median[k,2]
mean_std_pred_incloud13_dyn_whole_cl_median = all_std_pred_dyn_whole_cl_median[k,2]

ExpNorm1StdPredMedian = (mean_std_pred_incloud13_median,mean_std_incloud13_dyn_cs_median,mean_std_pred_incloud13_dyn_whole_cl_median )

mean_std_pred_incloud13_mad = all_std_pred_static_cs_mad[k,2]
mean_std_pred_incloud13_dyn_cs_mad = all_std_pred_dyn_cs_mad[k,2]
mean_std_pred_incloud13_dyn_whole_cl_mad = all_std_pred_dyn_whole_cl_mad[k,2]


ExpNorm1StdPredMad = (mean_std_pred_incloud13_mad,mean_std_pred_incloud13_dyn_cs_mad,mean_std_pred_incloud13_dyn_whole_cl_mad)
rects3 = ax2.bar(ind+ 2*width, ExpNorm1StdPredMedian, width, color='b',yerr = ExpNorm1StdPredMad)

### ExpArdPhi 2

mean_std_pred_incloud14_median = all_std_pred_static_cs_median[k,3]
mean_std_incloud14_dyn_cs_median = all_std_pred_dyn_cs_median[k,3]
mean_std_pred_incloud14_dyn_whole_cl_median = all_std_pred_dyn_whole_cl_median[k,3]

ExpNorm2StdPredMedian = (mean_std_pred_incloud14_median,mean_std_incloud14_dyn_cs_median,mean_std_pred_incloud14_dyn_whole_cl_median)

mean_std_pred_incloud14_mad = all_std_pred_static_cs_mad[k,3]
mean_std_pred_incloud14_dyn_cs_mad = all_std_pred_dyn_cs_mad[k,3]
mean_std_pred_incloud14_dyn_whole_cl_mad = all_std_pred_dyn_whole_cl_mad[k,3]

ExpNorm2StdPredMad = (mean_std_pred_incloud14_mad,mean_std_pred_incloud14_dyn_cs_mad,mean_std_pred_incloud14_dyn_whole_cl_mad)
rects4 = ax2.bar(ind+ 3*width, ExpNorm2StdPredMedian, width, color='g',yerr = ExpNorm2StdPredMad)

### ExpArdPhi 3

mean_std_pred_incloud15_median = all_std_pred_static_cs_median[k,4]
mean_std_incloud15_dyn_cs_median = all_std_pred_dyn_cs_median[k,4]
mean_std_pred_incloud15_dyn_whole_cl_median = all_std_pred_dyn_whole_cl_median[k,4]

ExpNorm3StdPredMedian = (mean_std_pred_incloud15_median,mean_std_incloud15_dyn_cs_median,mean_std_pred_incloud15_dyn_whole_cl_median)

mean_std_pred_incloud15_mad = all_std_pred_static_cs_mad[k,4]
mean_std_pred_incloud15_dyn_cs_mad = all_std_pred_dyn_cs_mad[k,4]
mean_std_pred_incloud15_dyn_whole_cl_mad = all_std_pred_dyn_whole_cl_mad[k,4]

ExpNorm3StdPredMad = (mean_std_pred_incloud15_mad,mean_std_pred_incloud15_dyn_cs_mad,mean_std_pred_incloud15_dyn_whole_cl_mad)
rects5 = ax2.bar(ind+ 4*width, ExpNorm3StdPredMedian, width, color='k',yerr = ExpNorm3StdPredMad)

##### Sq. Exp Detrended
mean_std_pred_incloud18_median = all_std_pred_static_cs_median[k,7]
mean_std_incloud18_dyn_cs_median = all_std_pred_dyn_cs_median[k,7]
mean_std_pred_incloud18_dyn_whole_cl_median = all_std_pred_dyn_whole_cl_median[k,7]

SEdetrStdPredMedian = (mean_std_pred_incloud18_median,mean_std_incloud18_dyn_cs_median,mean_std_pred_incloud18_dyn_whole_cl_median)

mean_std_pred_incloud18_mad = all_std_pred_static_cs_mad[k,7]
mean_std_pred_incloud18_dyn_cs_mad = all_std_pred_dyn_cs_mad[k,7]
mean_std_pred_incloud18_dyn_whole_cl_mad = all_std_pred_dyn_whole_cl_mad[k,7]

SEdetrStdPredMad = (mean_std_pred_incloud18_mad,mean_std_pred_incloud18_dyn_cs_mad,mean_std_pred_incloud18_dyn_whole_cl_mad)
rects6 = ax2.bar(ind+ 5*width, SEdetrStdPredMedian, width, color='#e67e22',yerr = SEdetrStdPredMad)
###### Exp. Detrend
mean_std_pred_incloud19_median = all_std_pred_static_cs_median[k,8]
mean_std_incloud19_dyn_cs_median = all_std_pred_dyn_cs_median[k,8]
mean_std_pred_incloud19_dyn_whole_cl_median = all_std_pred_dyn_whole_cl_median[k,8]

ExpDetrStdPredMedian = (mean_std_pred_incloud19_median,mean_std_incloud19_dyn_cs_median,mean_std_pred_incloud19_dyn_whole_cl_median)

mean_std_pred_incloud19_mad = all_std_pred_static_cs_mad[k,8]
mean_std_pred_incloud19_dyn_cs_mad = all_std_pred_dyn_cs_mad[k,8]
mean_std_pred_incloud19_dyn_whole_cl_mad = all_std_pred_dyn_whole_cl_mad[k,8]

ExpDetrStdPredMad = (mean_std_pred_incloud19_mad,mean_std_pred_incloud19_dyn_cs_mad,mean_std_pred_incloud19_dyn_whole_cl_mad)
rects7 = ax2.bar(ind+ 6*width, ExpDetrStdPredMedian, width, color='#7d3c98',yerr = ExpDetrStdPredMad)


# add some text for labels, title and axes ticks
ax2.set_ylabel('Predicted Standard Deviation')
ax2.set_title('Predicted Standard Deviation Cloud1 Experiments, noise_std ={} m/s'.format(float(noise[k])))
ax2.set_xticks(ind + 3.5*width)
ax2.set_xticklabels(('Static CS', 'Dyn. CS','Dyn. Whole Cl.'))
ax2.set_ylim(0,2.5)

ax2.legend((rects1[0], rects2[0],rects3[0],rects4[0],rects5[0],rects6[0],rects7[0]), ('Exp.', 'Sq. Exp','Exp. norm1','Exp. norm2',
'Exp. norm3','Sq.Exp Detr.','Exp. Detr.'))


############### Hyperparameters Distribution
all_hypers_dyn_cs[:,:,:,:5,:4] = np.exp(all_hypers_dyn_cs[:,:,:,:5,:4])
all_hypers_dyn_cs[:,:,:,:5,4:6] = np.sqrt(np.exp(2*all_hypers_dyn_cs[:,:,:,:5,4:6]))

all_hypers_static_cs[:,:,:,:5,:4] = np.exp(all_hypers_static_cs[:,:,:,:5,:4])
all_hypers_static_cs[:,:,:,:5,4:6] = np.sqrt(np.exp(2*all_hypers_static_cs[:,:,:,:5,4:6]))

all_hypers_dyn_whole_cl[:,:,:5,:4] = np.exp(all_hypers_dyn_whole_cl[:,:,:5,:4])
all_hypers_dyn_whole_cl[:,:,:5,4:6] = np.sqrt(np.exp(2*all_hypers_dyn_whole_cl[:,:,:5,4:6]))

all_hypers_dyn_whole_cl_median = np.nanmedian(all_hypers_dyn_whole_cl,axis = 0)

all_hypers_dyn_cs_median = np.nanmedian(all_hypers_dyn_cs,axis=(0,1))
all_hypers_dyn_cs_mad = mad(all_hypers_dyn_cs,axis=(0,1))

all_hypers_static_cs_median = np.nanmedian(all_hypers_static_cs,axis=(0,1))
all_hypers_static_cs_mad = mad(all_hypers_static_cs,axis=(0,1))

#################### Plotting Hyperparameters for each model
k = 1
noise = [0.1,0.25,0.5,0.75]

models = [0,1,2,3,4]
models_name = ['Exp.', 'Sq. Exp','Exp. norm1','Exp. norm2','Exp. norm3']
model = 4
ind = np.arange(1,7,3)  # the x locations for the groups
width = 0.30

fig, ax1 = plt.subplots()

## t lengthscale
if model ==0:
    scaling_t = 0.001
else:
    scaling_t = 1
Mod_lt_dyn_cs_median = all_hypers_dyn_cs_median[k,model,0]*scaling_t

ModLtMedian = (0,Mod_lt_dyn_cs_median)

Mod_lt_dyn_cs_mad = all_hypers_dyn_cs_mad[k,model,0]*scaling_t

ModLtMad = (0,Mod_lt_dyn_cs_mad)
rects1 = ax1.bar(ind, ModLtMedian, width, color='r',yerr = ModLtMad)

# No Zs yet, results of whole cloud not out

# lx
Mod_lx_dyn_cs_median = all_hypers_dyn_cs_median[k,model,2]
Mod_lx_static_cs_median = all_hypers_static_cs_median[k,model,2]

ModLxMedian = (Mod_lx_static_cs_median,Mod_lx_dyn_cs_median)

Mod_lx_dyn_cs_mad = all_hypers_dyn_cs_mad[k,model,2]
Mod_lx_static_cs_mad = all_hypers_static_cs_mad[k,model,2]

ModLxMad = (Mod_lx_static_cs_mad,Mod_lx_dyn_cs_mad)
rects2 = ax1.bar(ind + width, ModLxMedian, width, color='y',yerr = ModLxMad)

# y lengthscale
Mod_ly_dyn_cs_median = all_hypers_dyn_cs_median[k,model,3]
Mod_ly_static_cs_median = all_hypers_static_cs_median[k,model,3]

ModLyMedian = (Mod_ly_static_cs_median,Mod_ly_dyn_cs_median)

Mod_ly_dyn_cs_mad = all_hypers_dyn_cs_mad[k,model,3]
Mod_ly_static_cs_mad = all_hypers_static_cs_mad[k,model,3]

ModLyMad = (Mod_ly_static_cs_mad,Mod_ly_dyn_cs_mad)
rects3 = ax1.bar(ind+2*width, ModLyMedian, width, color='b',yerr = ModLyMad)

if model>=2:
    factor_s = 100
else:
    factor_s = 1
# Process Variance
Mod_sf_dyn_cs_median = all_hypers_dyn_cs_median[k,model,4]*factor_s
Mod_sf_static_cs_median = all_hypers_static_cs_median[k,model,4]*factor_s

ModSfMedian = (Mod_sf_static_cs_median,Mod_sf_dyn_cs_median)

Mod_sf_dyn_cs_mad = all_hypers_dyn_cs_mad[k,model,4]*factor_s
Mod_sf_static_cs_mad = all_hypers_static_cs_mad[k,model,4]*factor_s

ModSfMad = (Mod_sf_static_cs_mad,Mod_sf_dyn_cs_mad)
rects4 = ax1.bar(ind+3*width, ModSfMedian, width, color='g',yerr = ModSfMad)

# Noise Standard Deviation
Mod_sn_dyn_cs_median = all_hypers_dyn_cs_median[k,model,5]*factor_s
Mod_sn_static_cs_median = all_hypers_static_cs_median[k,model,5]*factor_s

ModSnMedian = (Mod_sn_static_cs_median,Mod_sn_dyn_cs_median)

Mod_sn_dyn_cs_mad = all_hypers_dyn_cs_mad[k,model,5]*factor_s
Mod_sn_static_cs_mad = all_hypers_static_cs_mad[k,model,5]*factor_s

ModSnMad = (Mod_sn_static_cs_mad,Mod_sn_dyn_cs_mad)
rects5 = ax1.bar(ind+4*width, ModSnMedian, width, color='k',yerr = ModSnMad)
# add some text for labels, title and axes ticks
ax1.set_ylabel('Model Hyperparameters')
ax1.set_title('Hyperparameters Cloud1 Experiments, Model={}, noise_std ={} m/s'.format(models_name[model],noise[k]))
ax1.set_xticks(ind + 3*width)
ax1.set_xticklabels(('Static CS', 'Dyn. CS'))

if model==0:
    ax1.legend((rects1[0], rects2[0],rects3[0],rects4[0],rects5[0]), ('$l_t(ms)$', '$l_x(km)$','$l_y(km)$','$\sigma_f(-)$','$\sigma_n(m/s)$'))
elif model ==1:
    ax1.legend((rects1[0], rects2[0],rects3[0],rects4[0],rects5[0]), ('$l_t(s)$', '$l_x(km)$','$l_y(km)$','$\sigma_f(-)$','$\sigma_n(m/s)$'))
else:
    ax1.legend((rects1[0], rects2[0],rects3[0],rects4[0],rects5[0]), ('$l_t(s)$', '$l_{\phi(deg)}$','$l_r(\%)$','$100*\sigma_f(-)$','$100*\sigma_n(m/s)$'))

################ Error Stats

all_error_stats_static_cs_median = np.nanmedian(all_error_stats_static_cs,axis=(0,1))
all_error_stats_dyn_cs_median = np.nanmedian(all_error_stats_dyn_cs,axis=(0,1))
all_error_stats_dyn_whole_cl_median = np.nanmedian(all_error_stats_dyn_whole_cl,axis=(0))

all_error_stats_static_cs_mad = mad(all_error_stats_static_cs,axis=(0,1))
all_error_stats_dyn_cs_mad = mad(all_error_stats_dyn_cs,axis=(0,1))
all_error_stats_dyn_whole_cl_mad = mad(all_error_stats_dyn_whole_cl,axis=(0))

#############################################################################################
###### Hyperparameter Distribution Exp cart detrended, best model, Noise 0.25 ###############
#############################################################################################
all_hypers_dyn_cs_exp = all_hypers_dyn_cs.copy()
all_hypers_dyn_cs_exp[:,:,:,8,:4] = np.exp(all_hypers_dyn_cs[:,:,:,8,:4])
all_hypers_dyn_cs_exp[:,:,:,8,4:6] = np.sqrt(np.exp(2*all_hypers_dyn_cs[:,:,:,8,4:6]))
all_hypers_dyn_cs_exp_median = np.nanmedian(all_hypers_dyn_cs_exp,axis=(0,1))


all_hypers_static_cs_exp = all_hypers_static_cs.copy()
all_hypers_static_cs_exp[:,:,:,8,:4] = np.exp(all_hypers_static_cs[:,:,:,8,:4])
all_hypers_static_cs_exp[:,:,:,8,4:6] = np.sqrt(np.exp(2*all_hypers_static_cs[:,:,:,8,4:6]))
all_hypers_static_cs_exp_median = np.nanmedian(all_hypers_static_cs_exp,axis=(0,1))

all_hypers_dyn_whole_cl_exp = all_hypers_dyn_whole_cl.copy()
all_hypers_dyn_whole_cl_exp[:,:,8,:4] = np.exp(all_hypers_dyn_whole_cl[:,:,8,:4])
all_hypers_dyn_whole_cl_exp[:,:,8,4:6] = np.sqrt(np.exp(2*all_hypers_dyn_whole_cl[:,:,8,4:6]))
all_hypers_dyn_whole_cl_exp_median = np.nanmedian(all_hypers_dyn_whole_cl_exp,axis = 0)


######### Static CS,
font = {'size'   : 52}

plt.rc('font', **font)

noise = np.array([0.001,0.1,0.25,0.5,0.75])
k = 2

plt.figure()
plt.hist(all_hypers_static_cs_exp[:,:,k,8,4].reshape(-1),bins=15)
#plt.title('Process Standard Deviation $\sigma_f$ Exp. cart. detrended, Static CS noise_std={}'.format(float(noise[k])))
plt.title('Process Standard Deviation $\sigma_f$')
plt.xlabel('Values(m/s)')
plt.ylabel('Count')

plt.figure()
plt.hist(all_hypers_static_cs_exp[:,:,k,8,5].reshape(-1),bins=15)
#plt.title('Process Noise Standard Deviation $\sigma_n$ Exp. cart. detrended, Static CS noise_std={}'.format(float(noise[k])))
plt.title('Process Noise Standard Deviation $\sigma_n$')
plt.xlabel('Values(m/s)')
plt.ylabel('Count')

plt.figure()
plt.hist(all_hypers_static_cs_exp[:,:,k,8,2].reshape(-1),bins=15)
#plt.title('Lengthscale $l_x$ Exp. cart. detrended, Static CS noise_std={}'.format(float(noise[k])))
plt.title('Lengthscale $l_x$')
plt.xlabel('Values(km)')
plt.ylabel('Count')

plt.figure()
plt.hist(all_hypers_static_cs_exp[:,:,k,8,3].reshape(-1),bins=15)
#plt.title('Lengthscale $l_y$ Exp. cart. detrended, Static CS noise_std={}'.format(float(noise[k])))
plt.title('Lengthscale $l_y$')
plt.xlabel('Values(km)')
plt.ylabel('Count')

########### Dyn CS
plt.figure()
plt.hist(all_hypers_dyn_cs_exp[:,:,k,8,4].reshape(-1),bins=15)
#plt.title('Process Standard Deviation $\sigma_f$ Exp. cart. detrended, Dyn. CS noise_std={}'.format(float(noise[k])))
plt.title('Process Standard Deviation $\sigma_f$')
plt.xlabel('Values(m/s)')
plt.ylabel('Count')


plt.figure()
plt.hist(all_hypers_dyn_cs_exp[:,:,k,8,5].reshape(-1),bins=15)
#plt.title('Process Noise Standard Deviation $\sigma_n$ Exp. cart. detrended,  Dyn. CS noise_std={}'.format(float(noise[k])))
plt.title('Process Noise Standard Deviation $\sigma_n$')
plt.xlabel('Values(m/s)')
plt.ylabel('Count')

plt.figure()
plt.hist(all_hypers_dyn_cs_exp[:,:,k,8,0].reshape(-1),bins=15)
#plt.title('Lengthscale $l_t$ Exp. cart. detrended,  Dyn. CS noise_std={}'.format(float(noise[k])))
plt.title('Lengthscale $l_t$')
plt.xlabel('Values(s)')
plt.ylabel('Count')

plt.figure()
plt.hist(all_hypers_dyn_cs_exp[:,:,k,8,2].reshape(-1),bins=15)
#plt.title('Lengthscale $l_x$ Exp. cart. detrended,  Dyn. CS noise_std={}'.format(float(noise[k])))
plt.title('Lengthscale $l_x$')
plt.xlabel('Values(km)')
plt.ylabel('Count')

plt.figure()
plt.hist(all_hypers_dyn_cs_exp[:,:,k,8,3].reshape(-1),bins=15)
#plt.title('Lengthscale $l_y$ Exp. cart. detrended,  Dyn. CS noise_std={}'.format(float(noise[k])))
plt.title('Lengthscale $l_y$')
plt.xlabel('Values(km)')
plt.ylabel('Count')

########### Whole Cl.
plt.figure()
plt.hist(all_hypers_dyn_whole_cl_exp[:,k,8,4].reshape(-1),bins=10)
#plt.title('Process Standard Deviation $\sigma_f$ Exp. cart. detrended, Dyn. whole cl. noise_std={}'.format(float(noise[k])))
plt.title('Process Standard Deviation $\sigma_f$')
plt.xlabel('Values(m/s)')
plt.ylabel('Count')

plt.figure()
plt.hist(all_hypers_dyn_whole_cl_exp[:,k,8,5].reshape(-1),bins=10)
#plt.title('Process Noise Standard Deviation $\sigma_n$ Exp. cart. detrended,  Dyn. whole cl. noise_std={}'.format(float(noise[k])))
plt.title('Process Noise Standard Deviation $\sigma_n$')
plt.xlabel('Values(m/s)')
plt.ylabel('Count')

plt.figure()
plt.hist(all_hypers_dyn_whole_cl_exp[:,k,8,0].reshape(-1),bins=10)
#plt.title('Lengthscale $l_t$ Exp. cart. detrended,  Dyn. whole cl. noise_std={}'.format(float(noise[k])))
plt.title('Lengthscale $l_t$')
plt.xlabel('Values(s)')
plt.ylabel('Count')

plt.figure()
plt.hist(all_hypers_dyn_whole_cl_exp[:,k,8,1].reshape(-1),bins=10)
#plt.title('Lengthscale $l_z$ Exp. cart. detrended,  Dyn. whole cl. noise_std={}'.format(float(noise[k])))
plt.title('Lengthscale $l_z$')
plt.xlabel('Values(km)')
plt.ylabel('Count')

plt.figure()
plt.hist(all_hypers_dyn_whole_cl_exp[:,k,8,2].reshape(-1),bins=10)
#plt.title('Lengthscale $l_x$ Exp. cart. detrended,  Dyn. whole cl. noise_std={}'.format(float(noise[k])))
plt.title('Lengthscale $l_x$')
plt.xlabel('Values(km)')
plt.ylabel('Count')

plt.figure()
plt.hist(all_hypers_dyn_whole_cl_exp[:,k,8,3].reshape(-1),bins=10)
#plt.title('Lengthscale $l_y$ Exp. cart. detrended,  Dyn. whole cl. noise_std={}'.format(float(noise[k])))
plt.title('Lengthscale $l_y$')
plt.xlabel('Values(km)')
plt.ylabel('Count')
