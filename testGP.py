import GPy #used previously to verify integrity of libgp
from mesonh_atm.mesonh_atmosphere import MesoNHAtmosphere
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import scipy.optimize as sciopt
import modules.cloud as ModCloud
# Inhouse Python Interface for C++ Gaussian Process library, done before my time, last build, no more access
# rights to private git repository
from skyscan_lib.env_models.libgp import GPModel
import random
from scipy.stats import skew
import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable


infile = 'results/plots_and_images/polar_analysis/Radial_trend/trends_data/rtrend_global_median_updraft_norm.npy'
rtrend_global_median_updraft_norm = np.load(infile)

def mad(data, axis=None):
    return np.nanmedian(np.abs(data - np.nanmedian(data, axis)), axis)/0.6745


#Data without advection
path = "/net/skyscanner/volume1/data/mesoNH/ARM_OneHour3600files_No_Horizontal_Wind/"
mfiles = [path+"U0K10.1.min{:02d}.{:03d}_diaKCL.nc".format(minute, second)
          for minute in range(1, 60)
          for second in range(1, 61)]
mtstep = 1
atm = MesoNHAtmosphere(mfiles, 1)

# Loading median trend
infile = 'results/plots_and_images/polar_analysis/Radial_trend/trends_data/rtrend_global_median_updraft_norm.npy'
rtrend_global_median_updraft_norm = np.load(infile)

# Example Data of two variables with the coordinates of a rough bounding box of a cloud
# RCT = liquid water content, WT = vertical wind
lwc_data = atm.data['RCT'][449:540,95:111,60:200,100:250]
zwind_data = atm.data['WT'][449:540,95:111,60:200,100:250]
ids,counter,clouds = ModCloud.cloud_segmentation(lwc_data)

clouds=list(set(clouds.values()))
length_point_clds = np.ndarray((0,1))
for each_cloud in clouds:
    print(len(each_cloud.points))
    temp = len(each_cloud.points)
    length_point_clds = np.vstack((length_point_clds,temp))

cloud = clouds[np.argmax(length_point_clds)]

cloud.calculate_attributes(lwc_data,zwind_data)
lwc_cloud = np.zeros(lwc_data.shape)
for point in cloud.points:
    lwc_cloud[point] = 1

# Range of the rough bounding box
xr =np.arange(0.005 + 60*0.01, 0.005 + 200*0.01,0.01)
yr= np.arange(0.005 + 100*0.01, 0.005 + 250*0.01,0.01)
all_Zs=atm.data["VLEV"][:,0,0]
zr = all_Zs[95:111]
tr = np.arange(449,540)
origin_xy = [60,100]
zspan = np.arange(0,16)
points_span = (tr,zr,xr,yr)


interpolate_points_cloud = RegularGridInterpolator(points=(tr,zr,xr,yr),values=lwc_cloud,bounds_error=False,fill_value=0)

polar_cloud,polar_cloud_norm = polar_cloud_norm(points_span,lwc_cloud,cloud.COM_2D_lwc_tz,zspan,origin_xy)
lwc_cloud_polar = interpolate_points_cloud(polar_cloud,'nearest')

M = np.array(np.meshgrid(tr,zr, xr, yr, indexing='ij'))
grid = np.stack(M, axis=-1)


########################################################################################
############## Plots for Static Cross-section, 1 trial, 1 noise level ##################
########################################################################################

############## Only one instance of the different models will be analyzed, thus permiting
############## Some visualizations. For the aggregate results of several trials and noise
############## check manuscript, chapter 4.3


# Creating the coordinates of the circular trajectories for a given cross-section

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
    incloud = interpolate_points_cloud(temp,'nearest')
    if incloud == 1.0:
        drone1_circle = np.vstack((drone1_circle,temp))

drone2_circle = np.ndarray((0,4))
xcenter = 1.4
ycenter = 2.06
for t in np.arange(tstart_circle,tend_circle):
    trel = t - tstart_circle
    xtemp = xcenter + radius_circle*np.cos(15/90*trel)
    ytemp = ycenter + radius_circle*np.sin(15/90*trel)
    temp = np.array([453,zr[zind_rel[zi]],xtemp,ytemp])
    incloud = interpolate_points_cloud(temp,'nearest')
    if incloud == 1.0:
        drone2_circle = np.vstack((drone2_circle,temp))

drone3_circle = np.ndarray((0,4))
xcenter = 1.23
ycenter = 1.9
for t in np.arange(tstart_circle,tend_circle):
    trel = t - tstart_circle
    xtemp = xcenter + radius_circle*np.cos(15/90*trel)
    ytemp = ycenter + radius_circle*np.sin(15/90*trel)
    temp = np.array([453,zr[zind_rel[zi]],xtemp,ytemp])
    incloud = interpolate_points_cloud(temp,'nearest')
    if incloud == 1.0:
        drone3_circle = np.vstack((drone3_circle,temp))

drone4_circle = np.ndarray((0,4))
xcenter = 1.35
ycenter = 1.9
for t in np.arange(tstart_circle,tend_circle):
    trel = t - tstart_circle
    xtemp = xcenter + radius_circle*np.cos(15/90*trel)
    ytemp = ycenter + radius_circle*np.sin(15/90*trel)
    temp = np.array([453,zr[zind_rel[zi]],xtemp,ytemp])
    incloud = interpolate_points_cloud(temp,'nearest')
    if incloud == 1.0:
        drone4_circle = np.vstack((drone4_circle,temp))

drone5_circle = np.ndarray((0,4))
xcenter = 1.35
ycenter = 1.7
for t in np.arange(tstart_circle,tend_circle):
    trel = t - tstart_circle
    xtemp = xcenter + radius_circle*np.cos(15/90*trel)
    ytemp = ycenter + radius_circle*np.sin(15/90*trel)
    temp = np.array([453,zr[zind_rel[zi]],xtemp,ytemp])
    incloud = interpolate_points_cloud(temp,'nearest')
    if incloud == 1.0:
        drone5_circle = np.vstack((drone5_circle,temp))

################### Train and Test Data for cartesian and polar normalized GPs
drones_train = np.vstack((drone1_circle,drone2_circle,drone3_circle,drone4_circle,drone5_circle))
grid_unrolled = grid[4,zind_rel[zi]].reshape((-1,4))
all_len_train_data_static_cs[zi] = len(drones_train)

all_vars_static_cs[zi] = zwind_data[4,zind_rel[zi]][lwc_cloud[4,zind_rel[zi]]>=1e-5].var()

# The drone trajectories in normalized polar coordinates
drones_train_normal = ModCloud.normalize(lwc_cloud_polar,drones_train,cloud.COM_2D_lwc_tz,449,1.185,origin_xy)
grid_unrolled_normal = ModCloud.normalize(lwc_cloud_polar,grid_unrolled,cloud.COM_2D_lwc_tz,449,1.185,origin_xy)

# For detrending
interpolate_rtrend = RegularGridInterpolator(points=np.arange(0,151).reshape(1,151),values=rtrend_global_median_updraft_norm,bounds_error=False,fill_value=0)
COM =np.array([449,zr[zind_rel[zi]],(cloud.COM_2D_lwc_tz[0,zind_rel[zi],0] + origin_xy[0])*0.01,(cloud.COM_2D_lwc_tz[0,zind_rel[zi],1] + origin_xy[1])*0.01])

font = {'size'   : 22}

plt.rc('font', **font)

noise = 0.25

#Training Data contaminated with white Gaussian Noise
zwind_train = atm.get_points(drones_train,'WT','linear')
zwind_train = zwind_train.reshape((len(zwind_train),1))
# Measure Wind at center of cross-section beforehand to scale trend
zwind_com = atm.get_points(COM,'WT','linear') + np.random.randn(1)*noise
zwind_train = zwind_train + np.random.randn(len(zwind_train),1)*noise

#Detrending by scaling global median trend by measured vertical wind at the center of cross-section
zwind_train_detrended = zwind_train - (zwind_com*interpolate_rtrend(drones_train_normal[:,3],'linear')).reshape((-1,1))

#Calculating performance metrics of the trend
rmse_trend = np.sqrt(np.mean((zwind_data[4,zind_rel[zi]][lwc_cloud[4,zind_rel[zi]]>=1e-5] -
zwind_com*interpolate_rtrend(grid_unrolled_normal[:,3],'linear').reshape((140,150))[lwc_cloud[4,zind_rel[zi]]>=1e-5])**2))

explained_variance = np.sum((zwind_data[4,zind_rel[zi]][lwc_cloud[4,zind_rel[zi]]>=1e-5] -
zwind_com*interpolate_rtrend(grid_unrolled_normal[:,3],'linear').reshape((140,150))[lwc_cloud[4,zind_rel[zi]]>=1e-5])**2)
total_variance = np.sum((zwind_data[4,zind_rel[zi]][lwc_cloud[4,zind_rel[zi]]>=1e-5]- zwind_train.mean())**2)
r2_trend = 1 - explained_variance/total_variance

error_trend = (zwind_data[4,zind_rel[zi]][lwc_cloud[4,zind_rel[zi]]>=1e-5] -
zwind_com*interpolate_rtrend(grid_unrolled_normal[:,3],'linear').reshape((140,150))[lwc_cloud[4,zind_rel[zi]]>=1e-5])
error_stats_trend = np.array([np.mean(error_trend),np.median(error_trend),np.std(error_trend),skew(error_trend)])

trend_pred = zwind_com*interpolate_rtrend(grid_unrolled_normal[:,3],'linear').reshape((140,150))
error_trend_total = zwind_data[4,zind_rel[zi]] - zwind_com*interpolate_rtrend(grid_unrolled_normal[:,3],'linear').reshape((140,150))

# Values to set coloring in later plots
vmin = np.min(zwind_data[4,zind_rel[zi],:,30:])
vmax = np.max(zwind_data[4,zind_rel[zi],:,30:])

########################### skyscan.lib GP Model Squared Exponential with GP Noise,
###########################         cartesian coordinates,vertical wind as is.
M_cart_SE = GPModel(ndim=4,nvar=1, kernel_string=b"CovSum(CovSEard, CovNoise)")
M_cart_SE.update(drones_train,zwind_train.T)

## Values in log scale
input_opt = {"gp_opt_python":True,'gp_opt_iter':10,'gp_hyper_bounds':((-10, 10), (-10, 10), (-10, 10), (-10, 10),
                 (-10, 10),
                 (-10, 10)),'gp_discr_tol':-1}
M_cart_SE.optimize(input_opt)
params_cart_SE = np.asarray(M2.get_params()).reshape(6)

predictions_cart_SE = np.ndarray((1,2,140*150))
M_cart_SE.predict(grid_unrolled,predictions_cart_SE)

mean_pred_cart_SE = predictions_cart_SE[0,0]
var_pred_cart_SE = predictions_cart_SE[0,1]

mean_pred_cart_SE = mean_pred_cart_SE.reshape((140,150))
var_pred_cart_SE = var_pred_cart_SE.reshape((140,150))

mean_std_pred_incloud_cart_SE = np.mean(np.sqrt(var_pred_cart_SE[lwc_cloud[4,zind_rel[zi]]>=1e-5]))
rmse_incloud_cart_SE = np.sqrt(np.mean((zwind_data[4,zind_rel[zi]][lwc_cloud[4,zind_rel[zi]]>=1e-5]-mean_pred_cart_SE[lwc_cloud[4,zind_rel[zi]]>=1e-5])**2))

abs_error_incloud_cart_SE = np.abs(zwind_data[4,zind_rel[zi]][lwc_cloud[4,zind_rel[zi]]>=1e-5]-mean_pred_cart_SE[lwc_cloud[4,zind_rel[zi]]>=1e-5])
std_pred_incloud_cart_SE = np.sqrt(var_pred_cart_SE[lwc_cloud[4,zind_rel[zi]]>=1e-5])
test_std_soundness_incloud_cart_SE = (std_pred_incloud_cart_SE-abs_error_incloud_cart_SE)/std_pred_incloud_cart_SE

std_pred_soundness_incloud_cart_SE = np.array([np.percentile(test_std_soundness_incloud_cart_SE,0.3),np.percentile(test_std_soundness_incloud_cart_SE,5),
np.percentile(test_std_soundness_incloud_cart_SE,32)])


explained_variance = np.sum((zwind_data[4,zind_rel[zi]][lwc_cloud[4,zind_rel[zi]]>=1e-5] -
                    mean_pred_cart_SE[lwc_cloud[4,zind_rel[zi]]>=1e-5])**2)
total_variance = np.sum((zwind_data[4,zind_rel[zi]][lwc_cloud[4,zind_rel[zi]]>=1e-5]- zwind_train.mean())**2)
r2_incloud_cart_SE = 1 - explained_variance/total_variance

error_incloud_cart_SE = zwind_data[4,zind_rel[zi]][lwc_cloud[4,zind_rel[zi]]>=1e-5]-mean_pred_cart_SE[lwc_cloud[4,zind_rel[zi]]>=1e-5]
error_stats_incloud_cart_SE = np.array([np.mean(error_incloud_cart_SE),np.median(error_incloud_cart_SE),np.std(error_incloud_cart_SE),skew(error_incloud_cart_SE)])

error_incloud_cart_SE_total = zwind_data[4,zind_rel[zi]]-mean_pred_cart_SE
###################### libgp, normalized radius and corrected phi, exponential cov with added GP noise,
###################### not optimized,variograms hypers, vertical wind detrended

M_norm_exp_detrended = GPModel(ndim=4,nvar=1, kernel_string=b"CovSum(CovExpArdPhi, CovNoise)")

lt = np.log(51.026823559394558)
lz = np.log(0.01*13.054954891415182)
l_phi = np.log(23.025993674634258)
lr = np.log(40.199201579845884)
sigma2 = 0.5*np.log(0.84069334459964384)
noise_var =0.5*np.log(noise**2) # assuming sensor noise model is known in advance

params = np.array([lt,lz,l_phi,lr,sigma2,noise_var]).reshape(1,-1)
M_norm_exp_detrended.set_params(params)
M_norm_exp_detrended.update(drones_train_normal,zwind_train_detrended.T)
params_norm_exp_detrended = params.reshape(6)
predictions_norm_exp_detrended =np.nan*np.ndarray((1,2,140*150))
M_norm_exp_detrended.predict(grid_unrolled_normal,predictions_norm_exp_detrended)

mean_pred_norm_exp_detrended = predictions_norm_exp_detrended[0,0].reshape(-1,1)
var_pred_norm_exp_detrended = predictions_norm_exp_detrended[0,1].reshape(-1,1)

mean_pred_norm_exp_detrended = mean_pred_norm_exp_detrended + zwind_com*interpolate_rtrend(grid_unrolled_normal[:,3],'linear').reshape((-1,1))

mean_pred_norm_exp_detrended = mean_pred_norm_exp_detrended.reshape((140,150))
var_pred_norm_exp_detrended = var_pred_norm_exp_detrended.reshape((140,150))

mean_std_pred_incloud_norm_exp_detrended = np.mean(np.sqrt(var_pred_norm_exp_detrended[lwc_cloud[4,zind_rel[zi]]>=1e-5]))
rmse_incloud_norm_exp_detrended = np.sqrt(np.mean((zwind_data[4,zind_rel[zi]][lwc_cloud[4,zind_rel[zi]]>=1e-5]-mean_pred_norm_exp_detrended[lwc_cloud[4,zind_rel[zi]]>=1e-5])**2))

abs_error_incloud_norm_exp_detrended = np.abs(zwind_data[4,zind_rel[zi]][lwc_cloud[4,zind_rel[zi]]>=1e-5]-mean_pred_norm_exp_detrended[lwc_cloud[4,zind_rel[zi]]>=1e-5])
std_pred_incloud_norm_exp_detrended = np.sqrt(var_pred_norm_exp_detrended[lwc_cloud[4,zind_rel[zi]]>=1e-5])
test_std_soundness_incloud_norm_exp_detrended = (std_pred_incloud_norm_exp_detrended-abs_error_incloud_norm_exp_detrended)/std_pred_incloud_norm_exp_detrended

std_pred_soundness_incloud_norm_exp_detrended = np.array([np.percentile(test_std_soundness_incloud_norm_exp_detrended,0.3),np.percentile(test_std_soundness_incloud_norm_exp_detrended,5),
np.percentile(test_std_soundness_incloud_norm_exp_detrended,32)])


explained_variance = np.sum((zwind_data[4,zind_rel[zi]][lwc_cloud[4,zind_rel[zi]]>=1e-5] -
                    mean_pred_norm_exp_detrended[lwc_cloud[4,zind_rel[zi]]>=1e-5])**2)
total_variance = np.sum((zwind_data[4,zind_rel[zi]][lwc_cloud[4,zind_rel[zi]]>=1e-5]- zwind_train.mean())**2)
r2_incloud_norm_exp_detrended = 1 - explained_variance/total_variance

error_incloud_norm_exp_detrended = zwind_data[4,zind_rel[zi]][lwc_cloud[4,zind_rel[zi]]>=1e-5]-mean_pred_norm_exp_detrended[lwc_cloud[4,zind_rel[zi]]>=1e-5]
error_stats_incloud_norm_exp_detrended = np.array([np.mean(error_incloud_norm_exp_detrended),np.median(error_incloud_norm_exp_detrended),np.std(error_incloud_norm_exp_detrended),skew(error_incloud_norm_exp_detrended)])

error_incloud_norm_exp_detrended_total = zwind_data[4,zind_rel[zi]]-mean_pred_norm_exp_detrended

############################################# Comparison SE xy,trend,libgp expnorm1
font = {'size'   : 15}

plt.rc('font', **font)

### Three rows of subplots to compare the two alternatives of GP Models and the trend
f, axarr = plt.subplots(3, 4,sharey=True)

f.suptitle('Static CS:{}km, noise_std:{}m/s,rmse Sq.Exp:{}m/s,rmse trend:{}m/s,rmse ExpNorm1:{}m/s'.format(np.round(float(zr[zind_rel[zi]]),3),noise,
np.round(rmse_incloud2,3),np.round(rmse_trend,3),np.round(rmse_incloud3,3)))

######### Row of plots of the off-the-shelf GP with SE kernel, vertical wind as is and cartesian coordinates
######### as inputs for the training and test set.
im1 = axarr[0,0].imshow(zwind_data[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],vmin=vmin,vmax=vmax)
divider1 = make_axes_locatable(axarr[0,0])
cax1 = divider1.append_axes("right", size="5%", pad=0.05)
cbar1 = plt.colorbar(im1, cax=cax1)
axarr[0,0].contour(lwc_cloud[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
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
axarr[0,1].contour(lwc_cloud[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
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
axarr[0,2].contour(lwc_cloud[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
axarr[0,2].set_xlim(xr[0], xr[-1])
axarr[0,2].set_ylim(yr[30], yr[-1])
axarr[0,2].plot(drones_train[:,2],drones_train[:,3],'k.',alpha=0.3)
axarr[0,2].set_xlabel('x coordinate (km)')
axarr[0,2].set_ylabel('y coordinate(km)')
axarr[0,2].set_title('Predicted $\sqrt{V[y_{\star}]}$')

im4 = axarr[0,3].imshow(error_incloud2_total[:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],vmin=-1.5,vmax=1.5)
divider4 = make_axes_locatable(axarr[0,3])
cax4 = divider4.append_axes("right", size="5%", pad=0.05)
cbar4 = plt.colorbar(im4, cax=cax4)
axarr[0,3].contour(lwc_cloud[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
axarr[0,3].set_xlim(xr[0], xr[-1])
axarr[0,3].set_ylim(yr[30], yr[-1])
axarr[0,3].plot(drones_train[:,2],drones_train[:,3],'k.',alpha=0.3)
axarr[0,3].set_xlabel('x coordinate (km)')
axarr[0,3].set_ylabel('y coordinate(km)')
axarr[0,3].set_title('Prediction error')

######### Row of plots of the trend

im1 = axarr[1,0].imshow(zwind_data[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],vmin=vmin,vmax=vmax)
divider1 = make_axes_locatable(axarr[1,0])
cax1 = divider1.append_axes("right", size="5%", pad=0.1)
cbar1 = plt.colorbar(im1, cax=cax1)
axarr[1,0].contour(lwc_cloud[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
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
axarr[1,1].contour(lwc_cloud[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
axarr[1,1].set_xlim(xr[0], xr[-1])
axarr[1,1].set_ylim(yr[30], yr[-1])
axarr[1,1].plot(drones_train[:,2],drones_train[:,3],'k.',alpha=0.3)
axarr[1,1].set_xlabel('x coordinate (km)')
axarr[1,1].set_ylabel('y coordinate(km)')
axarr[1,1].set_title('Prediction of trend')

im3 = axarr[1,2].imshow(np.zeros((140,120)).T,extent=[xr[0], xr[-1], yr[30], yr[-1]],vmin=0,vmax=2)
axarr[1,2].contour(lwc_cloud[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
divider3 = make_axes_locatable(axarr[1,2])
cax3 = divider3.append_axes("right", size="5%", pad=0.1)
cbar3 = plt.colorbar(im3, cax=cax3)
axarr[1,2].set_xlim(xr[0], xr[-1])
axarr[1,2].set_ylim(yr[30], yr[-1])
axarr[1,2].plot(drones_train[:,2],drones_train[:,3],'k.',alpha=0.3)
axarr[1,2].set_title('Predicted std does not apply')
axarr[1,2].set_xlabel('x coordinate (km)')
axarr[1,2].set_ylabel('y coordinate(km)')

im4 = axarr[1,3].imshow(error_trend_total[:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],vmin=-1.5,vmax=1.5)
divider4 = make_axes_locatable(axarr[1,3])
cax4 = divider4.append_axes("right", size="5%", pad=0.1)
cbar4 = plt.colorbar(im4, cax=cax4)
axarr[1,3].contour(lwc_cloud[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
axarr[1,3].set_xlim(xr[0], xr[-1])
axarr[1,3].set_ylim(yr[30], yr[-1])
axarr[1,3].plot(drones_train[:,2],drones_train[:,3],'k.',alpha=0.3)
axarr[1,3].set_xlabel('x coordinate (km)')
axarr[1,3].set_ylabel('y coordinate(km)')
axarr[1,3].set_title('Prediction error of trend')

######### Row of plots of the new GP with SE kernel, vertical wind detrended and normalized polar coordinates
######### as inputs for the training and test set.

im1 = axarr[2,0].imshow(zwind_data[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],vmin=vmin,vmax=vmax)
divider1 = make_axes_locatable(axarr[2,0])
cax1 = divider1.append_axes("right", size="5%", pad=0.05)
cbar1 = plt.colorbar(im1, cax=cax1)
axarr[2,0].contour(lwc_cloud[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
axarr[2,0].set_xlim(xr[0], xr[-1])
axarr[2,0].set_ylim(yr[30], yr[-1])
axarr[2,0].plot(drones_train[:,2],drones_train[:,3],'k.',alpha=0.3)
axarr[2,0].set_xlabel('x coordinate (km)')
axarr[2,0].set_ylabel('y coordinate(km)')
axarr[2,0].set_title('Ground truth')

im2 = axarr[2,1].imshow(mean_pred_norm_exp_detrended[:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],vmin=vmin,vmax=vmax)
divider2 = make_axes_locatable(axarr[2,1])
cax2 = divider2.append_axes("right", size="5%", pad=0.05)
cbar2 = plt.colorbar(im2, cax=cax2)
axarr[2,1].contour(lwc_cloud[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
axarr[2,1].set_xlim(xr[0], xr[-1])
axarr[2,1].set_ylim(yr[30], yr[-1])
axarr[2,1].plot(drones_train[:,2],drones_train[:,3],'k.',alpha=0.3)
axarr[2,1].set_xlabel('x coordinate (km)')
axarr[2,1].set_ylabel('y coordinate(km)')
axarr[2,1].set_title('Predicted mean $y_{\star}$')

im3 = axarr[2,2].imshow(np.sqrt(var_pred_norm_exp_detrended[:,30:].T),origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],vmin=0,vmax=2)
divider3 = make_axes_locatable(axarr[2,2])
cax3 = divider3.append_axes("right", size="5%", pad=0.05)
cbar3 = plt.colorbar(im3, cax=cax3)
axarr[2,2].contour(lwc_cloud[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
axarr[2,2].set_xlim(xr[0], xr[-1])
axarr[2,2].set_ylim(yr[30], yr[-1])
axarr[2,2].plot(drones_train[:,2],drones_train[:,3],'k.',alpha=0.3)
axarr[2,2].set_xlabel('x coordinate (km)')
axarr[2,2].set_ylabel('y coordinate(km)')
axarr[2,2].set_title('Predicted $\sqrt{V[y_{\star}]}$')

im4 = axarr[2,3].imshow(error_incloud_norm_exp_detrended_total[:,0:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],vmin=-1.5,vmax=1.5)
divider4 = make_axes_locatable(axarr[2,3])
cax4 = divider4.append_axes("right", size="5%", pad=0.05)
cbar4 = plt.colorbar(im4, cax=cax4)
axarr[2,3].contour(lwc_cloud[4,zind_rel[zi],:,30:].T,origin='lower',extent=[xr[0], xr[-1], yr[30], yr[-1]],alpha=0.7,cmap='Greys')
axarr[2,3].set_xlim(xr[0], xr[-1])
axarr[2,3].set_ylim(yr[30], yr[-1])
axarr[2,3].plot(drones_train[:,2],drones_train[:,3],'k.',alpha=0.3)
axarr[2,3].set_xlabel('x coordinate (km)')
axarr[2,3].set_ylabel('y coordinate(km)')
axarr[2,3].set_title('Prediction error')
