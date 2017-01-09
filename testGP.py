import GPy
from mesonh_atm.mesonh_atmosphere import MesoNHAtmosphere
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import scipy.optimize as sciopt
import modules.cloud as ModCloud
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

# Example coordinates of a rough bounding box of a cloud
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
    incloud = interpolate_points_cloud1(temp,'nearest')
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
    incloud = interpolate_points_cloud1(temp,'nearest')
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
    incloud = interpolate_points_cloud1(temp,'nearest')
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
    incloud = interpolate_points_cloud1(temp,'nearest')
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
    incloud = interpolate_points_cloud1(temp,'nearest')
    if incloud == 1.0:
        drone5_circle = np.vstack((drone5_circle,temp))

################### Train and Test Data x,y,normalized GPs, does not change with noise and trials loops
drones_train = np.vstack((drone1_circle,drone2_circle,drone3_circle,drone4_circle,drone5_circle))
grid_unrolled = grid[4,zind_rel[zi]].reshape((-1,4))
all_len_train_data_static_cs[zi] = len(drones_train)

all_vars_static_cs[zi] = zwind_data1[4,zind_rel[zi]][lwc_cloud1[4,zind_rel[zi]]>=1e-5].var()

drones_train_normal = ModCloud.normalize(lwc_cloud_polar,drones_train,cloud1.COM_2D_lwc_tz,449,1.185,origin_xy)
grid_unrolled_normal = ModCloud.normalize(lwc_cloud1_polar,grid_unrolled,cloud1.COM_2D_lwc_tz,449,1.185,origin_xy)
interpolate_rtrend = RegularGridInterpolator(points=np.arange(0,151).reshape(1,151),values=rtrend_global_median_updraft_norm,bounds_error=False,fill_value=0)
COM =np.array([449,zr[zind_rel[zi]],(cloud1.COM_2D_lwc_tz[0,zind_rel[zi],0] + origin_xy[0])*0.01,(cloud1.COM_2D_lwc_tz[0,zind_rel[zi],1] + origin_xy[1])*0.01])

font = {'size'   : 22}

plt.rc('font', **font)

noise = 0.25

    #Training Data dependent on noise
    time1 = datetime.datetime.now()
    zwind_train = atm.get_points(drones_train,'WT','linear')
    zwind_train = zwind_train.reshape((len(zwind_train),1))
    zwind_com = atm.get_points(COM,'WT','linear') + np.random.randn(1)*noise
    zwind_train = zwind_train + np.random.randn(len(zwind_train),1)*noise
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
