import numpy as np
from mesonh_atm.mesonh_atmosphere import MesoNHAtmosphere
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import scipy.optimize as sciopt
import modules.cloud as ModCloud
import pickle

font = {'size'   : 26}

plt.rc('font', **font)

#Data without advection
path = "/net/skyscanner/volume1/data/mesoNH/ARM_OneHour3600files_No_Horizontal_Wind/"
mfiles = [path+"U0K10.1.min{:02d}.{:03d}_diaKCL.nc".format(minute, second)
          for minute in range(1, 60)
          for second in range(1, 61)]
mtstep = 1
atm = MesoNHAtmosphere(mfiles, 1)

infile = 'results/plots_and_images/polar_analysis/Radial_trend/trends_data/rtrend_global_median_updraft_norm.npy'
rtrend_global_median_updraft_norm = np.load(infile)

######################################################
############ Variograms Cloud example ################
######################################################

######### For loop to calculate t,r,phi Variograms at different Cross-sections
# Due to RAM issues, the 4 dimensions could not be analyzed together
rmse_csec = np.array([])
r2_csec = np.array([])
variance_csec = np.array([])
#Cross-sections to analyze
z_index_wanted = np.arange(90,125,5)
variograms_cloud_trphi = {}

for z_index in z_index_wanted:

    lwc_data=atm.data['RCT'][455:605,z_index:(z_index+2),60:200,100:250]
    zwind_data=atm.data['WT'][455:605,z_index:(z_index+2),60:200,100:250]

    ids,counter,clouds = ModCloud.cloud_segmentation(lwc_data)
    clouds=list(set(clouds.values()))
    length_point_clds = np.ndarray((0,1))
    for each_cloud in clouds:
        print(len(each_cloud.points))
        temp = len(each_cloud.points)
        length_point_clds = np.vstack((length_point_clds,temp))
    # Get the cloud with the most points in the bounding box
    cloud = clouds[np.argmax(length_point_clds)]
    print('calculating attributes')
    cloud.calculate_attributes(lwc_data,zwind_data)

    #Binarized hypercube with the geometry of the cloud
    print('Creating lwc_cloud')
    lwc_cloud = np.zeros(lwc_data.shape)
    for point in cloud.points:
        lwc_cloud[point] = 1


    # Coordinates to interpolate values inside of cloud
    xr =np.arange(0.005 + 60*0.01, 0.005 + 200*0.01,0.01)
    yr= np.arange(0.005 + 100*0.01, 0.005 + 250*0.01,0.01)
    all_Zs=atm.data["VLEV"][:,0,0]
    zr = all_Zs[z_index:z_index+2]
    tr = np.arange(455,605)
    zspan = np.arange(0,1)

    points_span = (tr,zr,xr,yr)
    origin_xy = [60,100]
    polar_cloud,polar_cloud_norm = ModCloud.polar_cloud_norm(points_span,lwc_cloud,cloud.COM_2D_lwc_tz,zspan,origin_xy)

    # Some time points generated errors
    times_wanted = np.concatenate((np.arange(0,25),np.arange(26,85),np.arange(86,145),np.arange(146,150)))
    zwind_cloud_polar_norm = atm.get_points(polar_cloud_norm[times_wanted,0],'WT','linear')
    zwind_cloud_polar_norm = zwind_cloud_polar_norm.reshape((147,1,360,151))
    zwind_cloud_polar_norm_detrend = zwind_cloud_polar_norm.copy()

    # Measuring Wind at the center of the cross-sections at different time points to
    # scale the trend
    zwind_com2d_lwc_t = np.array([])
    for t in range(tr[0],tr[len(tr)-1],20):
        trel = t - tr[0]
        COMx = cloud.COM_2D_lwc_tz[trel,0,0]
        COMy = cloud.COM_2D_lwc_tz[trel,0,1]

        zwind_com2d_lwc_temp = atm.get_points([t,float(zr[0]),0.005 + (origin_xy[0] +COMx)*0.01,0.005 + (origin_xy[1] +COMy)*0.01],'WT','linear')
        zwind_com2d_lwc_t = np.append(zwind_com2d_lwc_t,zwind_com2d_lwc_temp)

    zwind_com_lwc_mean_plane = np.mean(zwind_com2d_lwc_t)
    zwind_com_lwc_median_plane = np.median(zwind_com2d_lwc_t)

    # Transform cloud information to polar coordinates
    interpolate_points_cloud = RegularGridInterpolator(points=points_span,values=lwc_cloud,bounds_error=False,fill_value=0)
    lwc_cloud_polar = interpolate_points_cloud(polar_cloud_norm[times_wanted,0],'nearest')
    lwc_cloud_polar = lwc_cloud_polar.reshape((147,1,360,151))

    ########### Detrending cross-section to later compute variograms
    # that are more stationary
    for r in range(0,151):
        zwind_cloud_polar_norm_detrend[:,:,:,r] = zwind_cloud_polar_norm[:,:,:,r] - zwind_com_lwc_mean_plane*rtrend_global_median_updraft_norm[r]
    ## R**2 By definition, handle with care, R**2 technically entirely valid for
    # linear models, still useful to compare

    explained_variance = np.sum(zwind_cloud_polar_norm_detrend[lwc_cloud_polar>=1e-5]**2)
    total_variance = zwind_cloud_polar_norm[lwc_cloud_polar>=1e-5].var() * np.count_nonzero(zwind_cloud_polar_norm[lwc_cloud_polar>=1e-5])
    rsquared2 = 1 - explained_variance/total_variance
    r2_csec = np.append(r2_csec,rsquared2)

    # Visualize Residuals of trend. Unbiased residuals resulted in variograms with more stationarity.
    plt.figure()
    plt.hist(zwind_cloud_polar_norm_detrend[lwc_cloud_polar>=1e-5].reshape(-1),bins=30)
    plt.xlabel('Residuals(m/s)')
    plt.ylabel('Count')
    plt.title('Residuals of Trend,z={} km, Zwind at center = {} m/s, $R^2$ of trend = {}'.format(np.round(float(zr[0]),3),
                np.round(zwind_com_lwc_mean_plane,2),np.round(rsquared2,2)))
    rmse =np.sqrt(np.sum(zwind_cloud_polar_norm_detrend[lwc_cloud_polar>=1e-5]**2)/np.count_nonzero(zwind_cloud_polar_norm_detrend[lwc_cloud_polar>=1e-5]))
    rmse_csec = np.append(rmse_cs,rmse)

    variograms_polar_norm= ModCloud.sample_variogram(zwind_cloud_polar_norm_detrend,'classical',lwc_cloud_polar)

    plt.figure()
    plt.title("Cloud,z={} km, Zwind at center = {} m/s, $R^2$ of trend = {}".format(np.round(float(zr[0]),3),
                np.round(zwind_com_lwc_mean_plane,2),np.round(rsquared2,2)))
    plt.xlabel(r"$h_r(\%),h_{\varphi}(degrees),h_t(s)$")
    plt.ylabel("$\hat{\gamma}(|h_i|)$")
    plt.plot(np.append(0,variograms_polar_norm["xvariogram_hat"][:181,0]),'-o',label=r"$\hat{\gamma}(|h_{\varphi}|)$")
    plt.plot(np.append(0,variograms_polar_norm["yvariogram_hat"][:100,0]),'-o',label="$\hat{\gamma}(|h_r|)$")
    plt.plot(np.append(0,variograms_polar_norm["tvariogram_hat"][:,0]),'-o',label="$\hat{\gamma}(|h_t|)$")
    plt.axhline(zwind_cloud_polar_norm_detrend[lwc_cloud_polar>=1e-5].var())
    plt.legend()
    variograms_cloud_trphi['variograms_cloud_trphi_zs{}'.format(z_index)] = variograms_polar_norm
    variance_csec = np.append(variance_csec,zwind_cloud_polar_norm_detrend[lwc_cloud_polar>=1e-5].var())

variograms_cloud_trphi['rmse_csec'] = rmse_csec
variograms_cloud_trphi['r2_csec'] = r2_csec
variograms_cloud_trphi['zsec'] = z_index_wanted
variograms_cloud_trphi['variance_csec'] =variance_csec
outfile = '/home/dselle/Skyscanner/data_exploration/results/plots_and_images/polar_analysis/Detrended_normalized_radial_variograms/cloud/various_cs_t150/variograms_cloud_trphi.pkl'
with open(outfile, 'wb') as output:
    pickle.dump(variograms_cloud_trphi, output, pickle.HIGHEST_PROTOCOL)

######### For loop to calculate z,r,phi Variograms at 5 seconds span
################# Due To RAM issues, the 4 dimensions could not be analyzed together

rmse_frozen_cl = np.array([])
r2_frozen_cl = np.array([])
variance_frozen_cl = np.array([])
t_index_wanted = [455,475,495,515,535]
variograms_cloud_zrphi = {}
for t_index in t_index_wanted:

    lwc_data=atm.data['RCT'][t_index:(t_index+5),90:123,60:200,100:250]
    zwind_data=atm.data['WT'][t_index:(t_index+5),90:123,60:200,100:250]

    ids,counter,clouds=ModCloud.cloud_segmentation(lwc_data)
    clouds = list(set(clouds.values()))
    length_point_clds = np.ndarray((0,1))
    for each_cloud in clouds:
        print(len(each_cloud.points))
        temp = len(each_cloud.points)
        length_point_clds = np.vstack((length_point_clds,temp))

    # Pick cloud with the most points inside of it
    cloud = clouds[np.argmax(length_point_clds)]
    print('calculating attributes')
    cloud.calculate_attributes(lwc_data,zwind_data)

    #Create binarized hypercube with geometry of the cloud
    print('Creating lwc_cloud')
    lwc_cloud = np.zeros(lwc_data.shape)
    for point in cloud.points:
        lwc_cloud[point] = 1


    # Coordinates to interpolate values inside of cloud
    xr =np.arange(0.005 + 60*0.01, 0.005 + 200*0.01,0.01)
    yr= np.arange(0.005 + 100*0.01, 0.005 + 250*0.01,0.01)
    all_Zs=atm.data["VLEV"][:,0,0]
    zr = all_Zs[90:123]
    tr = np.arange(t_index,t_index+5)
    zspan = np.arange(0,33)


    points_span = (tr,zr,xr,yr)
    origin_xy = [60,100]
    polar_cloud,polar_cloud_norm = ModCloud.polar_cloud_norm(points_span,lwc_cloud,cloud.COM_2D_lwc_tz,zspan,origin_xy)

    times_wanted = np.arange(0,5)
    zwind_cloud_polar_norm = atm.get_points(polar_cloud_norm[times_wanted],'WT','linear')
    zwind_cloud_polar_norm = zwind_cloud_polar_norm.reshape((5,33,360,151))
    zwind_cloud_polar_norm_detrend = zwind_cloud_polar_norm.copy()

    # Get wind at the center of some of the cross-sections available
    # to scale the trend
    zwind_com2d_lwc_z = np.array([])
    for z in range(zspan[0],zspan[len(zspan)-1],10):
        COMx = cloud.COM_2D_lwc_tz[0,z,0]
        COMy = cloud.COM_2D_lwc_tz[0,z,1]

        zwind_com2d_lwc_temp = atm.get_points([tr[0],zr[z],0.005 + (origin_xy[0] +COMx)*0.01,0.005 + (origin_xy[1] +COMy)*0.01],'WT','linear')
        zwind_com2d_lwc_z = np.append(zwind_com2d_lwc_z,zwind_com2d_lwc_temp)

    zwind_com_lwc_mean_cloud = np.mean(zwind_com2d_lwc_z)
    zwind_com_lwc_median_cloud = np.median(zwind_com2d_lwc_z)

    # Get the geometry information of the cloud in polar form
    interpolate_points_cloud = RegularGridInterpolator(points=points_span,values=lwc_cloud,bounds_error=False,fill_value=0)
    lwc_cloud_polar = interpolate_points_cloud(polar_cloud_norm[times_wanted],'nearest')
    lwc_cloud_polar = lwc_cloud_polar.reshape((5,33,360,151))

    ########### Detrending the whole cloud
    for r in range(0,151):
        zwind_cloud_polar_norm_detrend[:,:,:,r] = zwind_cloud_polar_norm[:,:,:,r] - zwind_com_lwc_mean_cloud*rtrend_global_median_updraft_norm[r]
    ## R**2 By definition, take this metric with a grain of salt, but still useful to compare.
    explained_variance = np.sum(zwind_cloud_polar_norm_detrend[lwc_cloud_polar>=1e-5]**2)
    total_variance = zwind_cloud_polar_norm[lwc_cloud_polar>=1e-5].var() * np.count_nonzero(zwind_cloud_polar_norm[lwc_cloud_polar>=1e-5])
    rsquared2 = 1 - explained_variance/total_variance
    r2_frozen_cl = np.append(r2_frozen_cl,rsquared2)

    # Visualizing residuals. Unbiased residuals resulted in variograms with more stationarity.
    plt.figure()
    plt.hist(zwind_cloud_polar_norm_detrend[lwc_cloud_polar>=1e-5].reshape(-1),bins=30)
    plt.xlabel('Residuals(m/s)')
    plt.ylabel('Count')
    plt.title('Residuals of Trend,t={}...{},Zwind at center = {} m/s, $R^2$ of trend = {}'.format(t_index,t_index+5,
                np.round(zwind_com_lwc_mean_cloud,2),np.round(rsquared2,2)))
    rmse =np.sqrt(np.sum(zwind_cloud_polar_norm_detrend[lwc_cloud_polar>=1e-5]**2)/np.count_nonzero(zwind_cloud_polar_norm_detrend[lwc_cloud_polar>=1e-5]))
    rmse_frozen_cl = np.append(rmse_frozen_cl,rmse)

    variograms_polar_norm= ModCloud.sample_variogram(zwind_cloud_polar_norm_detrend,'classical',lwc_cloud_polar)

    plt.figure()
    plt.title("Cloud,t={}...{}, Zwind at center = {} m/s, $R^2$ of trend = {}".format(t_index,t_index+5,
                np.round(zwind_com_lwc_mean_cloud,2),np.round(rsquared2,2)))
    plt.xlabel(r"$h_r(\%),h_{\varphi}(degrees),h_z(10m)$")
    plt.ylabel("$\hat{\gamma}(|h_i|)$")
    plt.plot(np.append(0,variograms_polar_norm["xvariogram_hat"][:181,0]),'-o',label=r"$\hat{\gamma}(|h_{\varphi}|)$")
    plt.plot(np.append(0,variograms_polar_norm["yvariogram_hat"][:100,0]),'-o',label="$\hat{\gamma}(|h_r|)$")
    plt.plot(np.append(0,variograms_polar_norm["zvariogram_hat"][:,0]),'-o',label="$\hat{\gamma}(|h_z|)$")
    plt.axhline(zwind_cloud_polar_norm_detrend[lwc_cloud_polar>=1e-5].var())
    plt.legend()
    variograms_cloud_zrphi['variograms_cloud_zrphi_ts{}'.format(t_index)] = variograms_polar_norm
    variance_frozen_cl = np.append(variance_frozen_cl,zwind_cloud_polar_norm_detrend[lwc_cloud_polar>=1e-5].var())

variograms_cloud_zrphi['rmse_frozen_cl'] = rmse_frozen_cl
variograms_cloud_zrphi['r2_frozen_cl'] = r2_frozen_cl
variograms_cloud_zrphi['ts'] = t_index_wanted
variograms_cloud_zrphi['variance_frozen_cl'] = variance_frozen_cl
outfile = '/home/dselle/Skyscanner/data_exploration/results/plots_and_images/polar_analysis/Detrended_normalized_radial_variograms/cloud/entire_cloud_t5_33cs/variograms_cloud_zxy.pkl'
with open(outfile, 'wb') as output:
    pickle.dump(variograms_cloud_zrphi, output, pickle.HIGHEST_PROTOCOL)


##############################################
######## Fitting the variograms ##############
##############################################

outfile = '/home/dselle/Skyscanner/data_exploration/results/plots_and_images/polar_analysis/Detrended_normalized_radial_variograms/cloud/various_cs_t150/variograms_cloud_trphi.pkl'
with open(outfile, 'rb') as input:
    print('Opening file')
    variograms_cloud_trphi = pickle.load(input)


outfile = '/home/dselle/Skyscanner/data_exploration/results/plots_and_images/polar_analysis/Detrended_normalized_radial_variograms/cloud/entire_cloud_t5_33cs/variograms_cloud_zrphi.pkl'
with open(outfile, 'rb') as input:
    print('Opening file')
    variograms_cloud_zrphi = pickle.load(input)

############ Use trphi which contains t,phi,r variograms, to get distribution of hyperparameter in the t direction

zsec = variograms_cloud_trphi['zsec']
tvariograms_fit_cloud = np.ndarray((4,len(zsec),3)) # (nmodels,nvariograms,[params, cost])
for i in range(len(zsec)):
    variograms_at_z = variograms_cloud_trphi['variograms_cloud_trphi_zs{}'.format(zs[i])]
    tvariogram_hat = variograms_at_z['tvariogram_hat']
    ht = np.arange(1,len(tvariogram_hat)+1)
    models = ('gaussian','exponential','matern_simp32','matern_simp52')
    for j in range(len(models)):
        params,cost,opt_object = ModCloud.fit_variogram(tvariogram_hat,ht,models[j])
        tvariograms_fit_cloud[j,i] = [params[0],params[1],cost]

###################### fitting z,phi,r variograms
ts = variograms_cloud_zrphi['ts']
z_r_phi_variograms_fit_cloud = np.ndarray((3,4,len(ts),3)) # (ndim[z,phi,r],nmodels,nvariograms,[params, cost])
for i in range(len(ts)):
    variograms_at_t = variograms_cloud_zrphi['variograms_cloud_zrphi_ts{}'.format(ts[i])]
    #Function sample_variogram was developed for cartesian coordinates
    variogram_names = ('zvariogram_hat','xvariogram_hat','yvariogram_hat')
    for j in range(len(variogram_names)):
        name = variogram_names[j]
        variogram_hat = variograms_at_t[name]
        if name == 'zvariogram_hat':
            h = np.arange(1,len(variogram_hat)+1)
        elif name == 'xvariogram_hat':
            # corresponds to phi
            h = np.arange(1,181)
            variogram_hat = variogram_hat[:180]
        else:
            # corresponds to r
            h = np.arange(1,101)
            variogram_hat = variogram_hat[:100]
        models = ('gaussian','exponential','matern_simp32','matern_simp52')
        for k in range(len(models)):
            params,cost,opt_object = ModCloud.fit_variogram(variogram_hat,h,models[k])
            z_r_phi_variograms_fit_cloud[j,k,i] = [params[0],params[1],cost]


# In the case of computing the variograms on more clouds, the results
# Can easily be concatenated.
all_variograms_fit_t = tvariograms_fit_cloud
all_variograms_fit_z_r_phi = z_r_phi_variograms_fit_cloud

############### Exponential Model has the best performance in normalized polar coordinates:
min_model = np.argmin(np.sum(all_variograms_fit_t,axis=1)[:,2]+np.sum(all_variograms_fit_z_r_phi,axis=(0,2))[:,2])

font = {'size'   : 52}

plt.rc('font', **font)

# Viewing the hyperparameters distribution of the exponential model
plt.figure()
plt.hist(all_variograms_fit_t[1,:,0],bins=10)
plt.title('Process Standard Deviation $\sigma_t$ Exponential t-variogram')
plt.xlabel('Values')
plt.ylabel('Count')

plt.figure()
plt.hist(all_variograms_fit_t[1,:,1],bins=10)
plt.title('Lengthscale $l_t$ Exponential t-variogram')
plt.xlabel('Values(s)')
plt.ylabel('Count')

plt.figure()
plt.hist(all_variograms_fit_z_r_phi[0,1,:,0],bins=10)
plt.title('Process Standard Deviation $\sigma_z$ Exponential z-variogram')
plt.xlabel('Values')
plt.ylabel('Count')

plt.figure()
plt.hist(all_variograms_fit_z_r_phi[0,1,:,1],bins=10)
plt.title('Lengthscale $l_z$ Exponential z-variogram')
plt.xlabel('Values(10m)')
plt.ylabel('Count')

plt.figure()
plt.hist(all_variograms_fit_z_r_phi[1,1,:,0],bins=10)
plt.title(r'Process Standard Deviation $\sigma_{\varphi}$ Exponential $\varphi$-variogram')
plt.xlabel('Values')
plt.ylabel('Count')

plt.figure()
plt.hist(all_variograms_fit_z_r_phi[1,1,:,1],bins=10)
plt.title(r'Lengthscale $l_{\varphi}$ Exponential $\varphi$-variogram')
plt.xlabel('Values(degrees)')
plt.ylabel('Count')

plt.figure()
plt.hist(all_variograms_fit_z_r_phi[2,1,:,0],bins=10)
plt.title('Process Standard Deviation $\sigma_r$ Exponential r-variogram')
plt.xlabel('Values')
plt.ylabel('Count')

plt.figure()
plt.hist(all_variograms_fit_z_r_phi[2,1,:,1],bins=10)
plt.title('Lengthscale $l_r$ Exponential r-variogram')
plt.xlabel('Values(%)')
plt.ylabel('Count')



#### Visualizing the exponential model in all directions with median hyperparameters.
sigmat,lt = np.median(all_variograms_fit_t[1,:,0]), np.median(all_variograms_fit_t[1,:,1])
h = np.arange(0,180)
gamma_h_modelt=(sigmat**2)*(1-np.exp(-(h/lt)))

sigmaz,lz = np.median(all_variograms_fit_z_r_phi[0,1,:,0]), np.median(all_variograms_fit_z_r_phi[0,1,:,1])
h = np.arange(0,21)
gamma_h_modelz=(sigmaz**2)*(1-np.exp(-(h/lz)))

sigma_phi,l_phi = np.median(all_variograms_fit_z_r_phi[1,1,:,0]), np.median(all_variograms_fit_z_r_phi[1,1,:,1])
h = np.arange(0,181)
gamma_h_modelphi=(sigma_phi**2)*(1-np.exp(-(h/l_phi)))

sigmar,lr = np.median(all_variograms_fit_z_r_phi[2,1,:,0]), np.median(all_variograms_fit_z_r_phi[2,1,:,1])
h = np.arange(0,151)
gamma_h_modelr=(sigmar**2)*(1-np.exp(-(h/lr)))

font = {'size'   : 26}

plt.rc('font', **font)

plt.figure()
plt.title('Exponential Variograms with median hyperparameters')
plt.plot(gamma_h_modelt,label='Exponential $\gamma(|h_t|)$',linewidth=3)
plt.plot(10*np.arange(0,21),gamma_h_modelz,label='Exponential $\gamma(|h_z|)$',linewidth=3)
plt.plot(gamma_h_modelphi, label=r'Exponential $\gamma(|h_{\varphi}|)$',linewidth=3)
plt.plot(gamma_h_modelr, label='Exponential $\gamma(|h_r|)$',linewidth=3)
plt.xlabel(r'$h_t(s),h_z(m),h_{\varphi}(degree),h_r(\%)$')
plt.ylabel('$\gamma(h_i)$')
plt.legend()
