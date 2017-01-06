import numpy as np
from skyscan_lib.sim.mesonh_atmosphere import MesoNHAtmosphere
#from mesonh_atm.mesonh_atmosphere import MesoNHAtmosphere
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import scipy.optimize as sciopt
#from cloud import cloud,sample_variogram
import cloud
import pickle
font = {'size'   : 26}

plt.rc('font', **font)



#############
### todos:###
#############
# change extracting_cube to add variograms functions in cloud.py


#Old Data without advection
path = "/net/skyscanner/volume1/data/mesoNH/ARM_OneHour3600files_No_Horizontal_Wind/"
mfiles = [path+"U0K10.1.min{:02d}.{:03d}_diaKCL.nc".format(minute, second)
          for minute in range(1, 60)
          for second in range(1, 61)]
mtstep = 1
atm = MesoNHAtmosphere(mfiles, 1)

infile = 'results/plots_and_images/polar_analysis/Radial_trend/trends_data/rtrend_global_median_updraft_norm.npy'
rtrend_global_median_updraft_norm = np.load(infile)
###########################################################
############ Function to transform to Polar ###############
############ representation, including      ###############
############        normalization           ###############
###########################################################
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

################################################################
########################### cloud1 #############################
################################################################

#############################################################################
#### Do not run any of the cloud loops again, results are stored ############
#############################################################################

######### For loop to calculate t,r,phi Variograms at different Cross-sections
################# Due To RAM issues, better run this for loop alone
rmse_cs = np.array([])
r2_cs = np.array([])
vars_cs = np.array([])
z_index_wanted = np.arange(90,125,5)
variograms_cloud1_txy = {}
for z_index in z_index_wanted:

    lwc_data1=atm.data['RCT'][455:605,z_index:(z_index+2),60:200,100:250]
    zwind_data1=atm.data['WT'][455:605,z_index:(z_index+2),60:200,100:250]

    ids1,counter1,clouds1=cloud.cloud_segmentation(lwc_data1)
    #ids1,counter1,clouds1=cloud_segmentation(lwc_data1)
    clouds1=list(set(clouds1.values()))
    length_point_clds = np.ndarray((0,1))
    for each_cloud in clouds1:
        print(len(each_cloud.points))
        temp = len(each_cloud.points)
        length_point_clds = np.vstack((length_point_clds,temp))

    cloud1 = clouds1[np.argmax(length_point_clds)]
    print('calculating attributes')
    cloud1.calculate_attributes(lwc_data1,zwind_data1)

    print('Creating lwc_cloud1')
    lwc_cloud1 = np.zeros(lwc_data1.shape)
    for point in cloud1.points:
        lwc_cloud1[point] = 1


    # Coordinates to interpolate values inside of cloud1
    xr =np.arange(0.005 + 60*0.01, 0.005 + 200*0.01,0.01)
    yr= np.arange(0.005 + 100*0.01, 0.005 + 250*0.01,0.01)
    all_Zs=atm.data["VLEV"][:,0,0]
    #zr = np.array([all_Zs[z_index]])
    zr = all_Zs[z_index:z_index+2]
    #zr = np.arange(1.185,1.185 + 15*0.01,0.01)
    tr = np.arange(455,605)
    zspan = np.arange(0,1)
    #tr = 449
    #M = np.array(np.meshgrid(tr,zr, xr, yr, indexing='ij'))
    #grid = np.stack(M, axis=-1)


    points_span = (tr,zr,xr,yr)
    origin_xy = [60,100]
    polar_cloud1,polar_cloud1_norm = polar_cloud_norm(points_span,lwc_cloud1,cloud1.COM_2D_lwc_tz,zspan,origin_xy)

    times_wanted = np.concatenate((np.arange(0,25),np.arange(26,85),np.arange(86,145),np.arange(146,150)))
    zwind_cloud1_polar_norm = atm.get_points(polar_cloud1_norm[times_wanted,0],'WT','linear')
    zwind_cloud1_polar_norm = zwind_cloud1_polar_norm.reshape((147,1,360,151))
    zwind_cloud1_polar_norm_detrend = zwind_cloud1_polar_norm.copy()

    zwind_com2d_lwc_t = np.array([])
    for t in range(tr[0],tr[len(tr)-1],20):
        trel = t - tr[0]
        COMx = cloud1.COM_2D_lwc_tz[trel,0,0]
        COMy = cloud1.COM_2D_lwc_tz[trel,0,1]

        zwind_com2d_lwc_temp = atm.get_points([t,float(zr[0]),0.005 + (origin_xy[0] +COMx)*0.01,0.005 + (origin_xy[1] +COMy)*0.01],'WT','linear')
        zwind_com2d_lwc_t = np.append(zwind_com2d_lwc_t,zwind_com2d_lwc_temp)

    zwind_com_lwc_mean_plane = np.mean(zwind_com2d_lwc_t)
    zwind_com_lwc_median_plane = np.median(zwind_com2d_lwc_t)

    interpolate_points_cloud = RegularGridInterpolator(points=points_span,values=lwc_cloud1,bounds_error=False,fill_value=0)
    lwc_cloud1_polar = interpolate_points_cloud(polar_cloud1_norm[times_wanted,0],'nearest')
    lwc_cloud1_polar = lwc_cloud1_polar.reshape((147,1,360,151))

    #rsquared2_zwind_com = np.array([])
    #for zwind_com_lwc_mean_plane in np.arange(3,4.5,0.1):
    ########### Using one cross-section to detrend the whole cloud
    for r in range(0,151):
        zwind_cloud1_polar_norm_detrend[:,:,:,r] = zwind_cloud1_polar_norm[:,:,:,r] - zwind_com_lwc_mean_plane*rtrend_global_median_updraft_norm[r]
    ## R**2 By definition
    explained_variance = np.sum(zwind_cloud1_polar_norm_detrend[lwc_cloud1_polar>=1e-5]**2)
    total_variance = zwind_cloud1_polar_norm[lwc_cloud1_polar>=1e-5].var() * np.count_nonzero(zwind_cloud1_polar_norm[lwc_cloud1_polar>=1e-5])
    rsquared2 = 1 - explained_variance/total_variance
    r2_cs = np.append(r2_cs,rsquared2)
    #rsquared2_zwind_com = np.append(rsquared2_zwind_com,rsquared2)

    plt.figure()
    plt.hist(zwind_cloud1_polar_norm_detrend[lwc_cloud1_polar>=1e-5].reshape(-1),bins=30)
    plt.xlabel('Residuals(m/s)')
    plt.ylabel('Count')
    plt.title('Residuals of Trend,z={} km, Zwind at center = {} m/s, $R^2$ of trend = {}'.format(np.round(float(zr[0]),3),
                np.round(zwind_com_lwc_mean_plane,2),np.round(rsquared2,2)))
    rmse =np.sqrt(np.sum(zwind_cloud1_polar_norm_detrend[lwc_cloud1_polar>=1e-5]**2)/np.count_nonzero(zwind_cloud1_polar_norm_detrend[lwc_cloud1_polar>=1e-5]))
    rmse_cs = np.append(rmse_cs,rmse)

    variograms1_polar_norm= cloud.sample_variogram(zwind_cloud1_polar_norm_detrend,'classical',lwc_cloud1_polar)

    plt.figure()
    plt.title("Cloud1,z={} km, Zwind at center = {} m/s, $R^2$ of trend = {}".format(np.round(float(zr[0]),3),
                np.round(zwind_com_lwc_mean_plane,2),np.round(rsquared2,2)))
    plt.xlabel(r"$h_r(\%),h_{\varphi}(degrees),h_t(s)$")
    plt.ylabel("$\hat{\gamma}(|h_i|)$")
    plt.plot(np.append(0,variograms1_polar_norm["xvariogram_hat"][:181,0]),'-o',label=r"$\hat{\gamma}(|h_{\varphi}|)$")
    plt.plot(np.append(0,variograms1_polar_norm["yvariogram_hat"][:100,0]),'-o',label="$\hat{\gamma}(|h_r|)$")
    plt.plot(np.append(0,variograms1_polar_norm["tvariogram_hat"][:,0]),'-o',label="$\hat{\gamma}(|h_t|)$")
    plt.axhline(zwind_cloud1_polar_norm_detrend[lwc_cloud1_polar>=1e-5].var())
    plt.legend()
    variograms_cloud1_txy['variograms_cloud1_txy_zs{}'.format(z_index)] = variograms1_polar_norm
    vars_cs = np.append(vars_cs,zwind_cloud1_polar_norm_detrend[lwc_cloud1_polar>=1e-5].var())

variograms_cloud1_txy['rmse_cs'] = rmse_cs
variograms_cloud1_txy['r2_cs'] = r2_cs
variograms_cloud1_txy['zs'] = z_index_wanted
variograms_cloud1_txy['vars_cs'] =vars_cs
outfile = '/home/dselle/Skyscanner/data_exploration/results/plots_and_images/polar_analysis/Detrended_normalized_radial_variograms/cloud1/various_cs_t150/variograms_cloud1_txy.pkl'
with open(outfile, 'wb') as output:
    pickle.dump(variograms_cloud1_txy, output, pickle.HIGHEST_PROTOCOL)

######### For loop to calculate z,r,phi Variograms at 5 seconds span
################# Due To RAM issues, better run this for loop alone

rmse_cs = np.array([])
r2_cs = np.array([])
vars_cs = np.array([])
t_index_wanted = [455,475,495,515,535]
variograms_cloud1_zxy = {}
for t_index in t_index_wanted:

    lwc_data1=atm.data['RCT'][t_index:(t_index+5),90:123,60:200,100:250]
    zwind_data1=atm.data['WT'][t_index:(t_index+5),90:123,60:200,100:250]

    ids1,counter1,clouds1=cloud.cloud_segmentation(lwc_data1)
    #ids1,counter1,clouds1=cloud_segmentation(lwc_data1)
    clouds1=list(set(clouds1.values()))
    length_point_clds = np.ndarray((0,1))
    for each_cloud in clouds1:
        print(len(each_cloud.points))
        temp = len(each_cloud.points)
        length_point_clds = np.vstack((length_point_clds,temp))

    cloud1 = clouds1[np.argmax(length_point_clds)]
    print('calculating attributes')
    cloud1.calculate_attributes(lwc_data1,zwind_data1)
    print('Creating lwc_cloud1')
    lwc_cloud1 = np.zeros(lwc_data1.shape)
    for point in cloud1.points:
        lwc_cloud1[point] = 1


    # Coordinates to interpolate values inside of cloud1
    xr =np.arange(0.005 + 60*0.01, 0.005 + 200*0.01,0.01)
    yr= np.arange(0.005 + 100*0.01, 0.005 + 250*0.01,0.01)
    all_Zs=atm.data["VLEV"][:,0,0]
    zr = all_Zs[90:123]
    #zr = np.arange(1.185,1.185 + 15*0.01,0.01)
    tr = np.arange(t_index,t_index+5)
    zspan = np.arange(0,33)
    #tr = 449
    #M = np.array(np.meshgrid(tr,zr, xr, yr, indexing='ij'))
    #grid = np.stack(M, axis=-1)


    points_span = (tr,zr,xr,yr)
    origin_xy = [60,100]
    polar_cloud1,polar_cloud1_norm = polar_cloud_norm(points_span,lwc_cloud1,cloud1.COM_2D_lwc_tz,zspan,origin_xy)

    #times_wanted = np.concatenate((np.arange(0,25),np.arange(26,85),np.arange(86,145),np.arange(146,150)))
    times_wanted = np.arange(0,5)
    zwind_cloud1_polar_norm = atm.get_points(polar_cloud1_norm[times_wanted],'WT','linear')
    zwind_cloud1_polar_norm = zwind_cloud1_polar_norm.reshape((5,33,360,151))
    zwind_cloud1_polar_norm_detrend = zwind_cloud1_polar_norm.copy()

    zwind_com2d_lwc_z = np.array([])
    for z in range(zspan[0],zspan[len(zspan)-1],10):
        COMx = cloud1.COM_2D_lwc_tz[0,z,0]
        COMy = cloud1.COM_2D_lwc_tz[0,z,1]

        zwind_com2d_lwc_temp = atm.get_points([tr[0],zr[z],0.005 + (origin_xy[0] +COMx)*0.01,0.005 + (origin_xy[1] +COMy)*0.01],'WT','linear')
        zwind_com2d_lwc_z = np.append(zwind_com2d_lwc_z,zwind_com2d_lwc_temp)

    zwind_com_lwc_mean_cloud = np.mean(zwind_com2d_lwc_z)
    zwind_com_lwc_median_cloud = np.median(zwind_com2d_lwc_z)

    interpolate_points_cloud = RegularGridInterpolator(points=points_span,values=lwc_cloud1,bounds_error=False,fill_value=0)
    lwc_cloud1_polar = interpolate_points_cloud(polar_cloud1_norm[times_wanted],'nearest')
    lwc_cloud1_polar = lwc_cloud1_polar.reshape((5,33,360,151))

    #rsquared2_zwind_com = np.array([])
    #for zwind_com_lwc_mean_plane in np.arange(3,4.5,0.1):
    ########### Using one cross-section to detrend the whole cloud
    for r in range(0,151):
        zwind_cloud1_polar_norm_detrend[:,:,:,r] = zwind_cloud1_polar_norm[:,:,:,r] - zwind_com_lwc_mean_cloud*rtrend_global_median_updraft_norm[r]
    ## R**2 By definition
    explained_variance = np.sum(zwind_cloud1_polar_norm_detrend[lwc_cloud1_polar>=1e-5]**2)
    total_variance = zwind_cloud1_polar_norm[lwc_cloud1_polar>=1e-5].var() * np.count_nonzero(zwind_cloud1_polar_norm[lwc_cloud1_polar>=1e-5])
    rsquared2 = 1 - explained_variance/total_variance
    r2_cs = np.append(r2_cs,rsquared2)
    #rsquared2_zwind_com = np.append(rsquared2_zwind_com,rsquared2)

    plt.figure()
    plt.hist(zwind_cloud1_polar_norm_detrend[lwc_cloud1_polar>=1e-5].reshape(-1),bins=30)
    plt.xlabel('Residuals(m/s)')
    plt.ylabel('Count')
    plt.title('Residuals of Trend,t={}...{},Zwind at center = {} m/s, $R^2$ of trend = {}'.format(t_index,t_index+5,
                np.round(zwind_com_lwc_mean_cloud,2),np.round(rsquared2,2)))
    rmse =np.sqrt(np.sum(zwind_cloud1_polar_norm_detrend[lwc_cloud1_polar>=1e-5]**2)/np.count_nonzero(zwind_cloud1_polar_norm_detrend[lwc_cloud1_polar>=1e-5]))
    rmse_cs = np.append(rmse_cs,rmse)

    variograms1_polar_norm= cloud.sample_variogram(zwind_cloud1_polar_norm_detrend,'classical',lwc_cloud1_polar)

    plt.figure()
    plt.title("Cloud1,t={}...{}, Zwind at center = {} m/s, $R^2$ of trend = {}".format(t_index,t_index+5,
                np.round(zwind_com_lwc_mean_cloud,2),np.round(rsquared2,2)))
    plt.xlabel(r"$h_r(\%),h_{\varphi}(degrees),h_z(10m)$")
    plt.ylabel("$\hat{\gamma}(|h_i|)$")
    plt.plot(np.append(0,variograms1_polar_norm["xvariogram_hat"][:181,0]),'-o',label=r"$\hat{\gamma}(|h_{\varphi}|)$")
    plt.plot(np.append(0,variograms1_polar_norm["yvariogram_hat"][:100,0]),'-o',label="$\hat{\gamma}(|h_r|)$")
    plt.plot(np.append(0,variograms1_polar_norm["zvariogram_hat"][:,0]),'-o',label="$\hat{\gamma}(|h_z|)$")
    plt.axhline(zwind_cloud1_polar_norm_detrend[lwc_cloud1_polar>=1e-5].var())
    plt.legend()
    variograms_cloud1_zxy['variograms_cloud1_zxy_ts{}'.format(t_index)] = variograms1_polar_norm
    vars_cs = np.append(vars_cs,zwind_cloud1_polar_norm_detrend[lwc_cloud1_polar>=1e-5].var())

variograms_cloud1_zxy['rmse_cs'] = rmse_cs
variograms_cloud1_zxy['r2_cs'] = r2_cs
variograms_cloud1_zxy['ts'] = t_index_wanted
variograms_cloud1_zxy['vars_cs'] = vars_cs
outfile = '/home/dselle/Skyscanner/data_exploration/results/plots_and_images/polar_analysis/Detrended_normalized_radial_variograms/cloud1/entire_cloud_t5_33cs/variograms_cloud1_zxy.pkl'
with open(outfile, 'wb') as output:
    pickle.dump(variograms_cloud1_zxy, output, pickle.HIGHEST_PROTOCOL)

################################################################
########################### cloud2 #############################
################################################################

######### For loop to calculate t,r,phi Variograms at different Cross-sections
################# Due To RAM issues, better run this for loop alone

rmse_cs = np.array([])
r2_cs = np.array([])
vars_cs = np.array([])
z_index_wanted = np.arange(90,125,5)
variograms_cloud2_txy = {}

for z_index in z_index_wanted:
    lwc_data2 = atm.data['RCT'][935:1085,z_index:(z_index+2),125:250,270:370]
    zwind_data2 = atm.data['WT'][935:1085,z_index:(z_index+2),125:250,270:370]

    ids2,counter2,clouds2=cloud.cloud_segmentation(lwc_data2)
    clouds2=list(set(clouds2.values()))
    length_point_clds = np.ndarray((0,1))
    for each_cloud in clouds2:
        print(len(each_cloud.points))
        temp = len(each_cloud.points)
        length_point_clds = np.vstack((length_point_clds,temp))

    cloud2 = clouds2[np.argmax(length_point_clds)]

    cloud2.calculate_attributes(lwc_data2,zwind_data2)


    lwc_cloud2 = np.zeros(lwc_data2.shape)
    for point in cloud2.points:
        lwc_cloud2[point] = 1


    # Coordinates to interpolate values inside of cloud1
    xr =np.arange(0.005 + 125*0.01, 0.005 + 250*0.01,0.01)
    yr= np.arange(0.005 + 270*0.01, 0.005 + 370*0.01,0.01)
    #zr = np.arange(1.135,1.135+30*0.01,0.01)
    all_Zs=atm.data["VLEV"][:,0,0]
    zr = all_Zs[z_index:z_index+2]
    tr = np.arange(935,1085)
    zspan = np.arange(0,1)

    points_span = (tr,zr,xr,yr)
    origin_xy = [125,270]

    polar_cloud2,polar_cloud2_norm = polar_cloud_norm(points_span,lwc_cloud2,cloud2.COM_2D_lwc_tz,zspan,origin_xy)

    times_wanted = np.concatenate((np.arange(0,25),np.arange(26,85),np.arange(86,145),np.arange(146,150)))
    zwind_cloud2_polar_norm = atm.get_points(polar_cloud2_norm[times_wanted,0],'WT','linear')
    zwind_cloud2_polar_norm = zwind_cloud2_polar_norm.reshape((147,1,360,151))
    zwind_cloud2_polar_norm_detrend = zwind_cloud2_polar_norm.copy()

    zwind_com2d_lwc_t = np.array([])
    for t in range(tr[0],tr[len(tr)-1],20):
        trel = t - tr[0]
        COMx = cloud2.COM_2D_lwc_tz[trel,0,0]
        COMy = cloud2.COM_2D_lwc_tz[trel,0,1]

        zwind_com2d_lwc_temp = atm.get_points([t,float(zr[0]),0.005 + (origin_xy[0]+COMx)*0.01,0.005 + (origin_xy[1]+COMy)*0.01],'WT','linear')
        zwind_com2d_lwc_t = np.append(zwind_com2d_lwc_t,zwind_com2d_lwc_temp)

    zwind_com_lwc_mean_plane = np.mean(zwind_com2d_lwc_t)
    zwind_com_lwc_median_plane = np.median(zwind_com2d_lwc_t)

    interpolate_points_cloud = RegularGridInterpolator(points=points_span,values=lwc_cloud2,bounds_error=False,fill_value=0)
    lwc_cloud2_polar = interpolate_points_cloud(polar_cloud2_norm[times_wanted,0],'nearest')
    lwc_cloud2_polar = lwc_cloud2_polar.reshape((147,1,360,151))

    #rsquared2_zwind_com = np.array([])
    #for zwind_com_lwc_mean_plane in np.arange(3,4.5,0.1):
    ########### Using one cross-section to detrend the whole cloud
    for r in range(0,151):
        zwind_cloud2_polar_norm_detrend[:,:,:,r] = zwind_cloud2_polar_norm[:,:,:,r] - zwind_com_lwc_mean_plane*rtrend_global_median_updraft_norm[r]
    ## R**2 By definition
    explained_variance = np.sum(zwind_cloud2_polar_norm_detrend[lwc_cloud2_polar>=1e-5]**2)
    total_variance = zwind_cloud2_polar_norm[lwc_cloud2_polar>=1e-5].var() * np.count_nonzero(zwind_cloud2_polar_norm[lwc_cloud2_polar>=1e-5])
    rsquared2 = 1 - explained_variance/total_variance
    r2_cs = np.append(r2_cs,rsquared2)
    #rsquared2_zwind_com = np.append(rsquared2_zwind_com,rsquared2)

    plt.figure()
    plt.hist(zwind_cloud2_polar_norm_detrend[lwc_cloud2_polar>=1e-5].reshape(-1),bins=30)
    plt.xlabel('Residuals(m/s)')
    plt.ylabel('Count')
    plt.title('Residuals of Trend, z={} km, Zwind at center = {} m/s, $R^2$ of trend = {}'.format(np.round(float(zr[0]),3),
                np.round(zwind_com_lwc_mean_plane,2),np.round(rsquared2,2)))
    rmse =np.sqrt(np.sum(zwind_cloud2_polar_norm_detrend[lwc_cloud2_polar>=1e-5]**2)/np.count_nonzero(zwind_cloud2_polar_norm_detrend[lwc_cloud2_polar>=1e-5]))
    rmse_cs = np.append(rmse_cs,rmse)

    variograms2_polar_norm= cloud.sample_variogram(zwind_cloud2_polar_norm_detrend,'classical',lwc_cloud2_polar)

    plt.figure()
    plt.title("Cloud2, z={} km, Zwind at center = {} m/s, $R^2$ of trend = {}".format(np.round(float(zr[0]),3),
                np.round(zwind_com_lwc_mean_plane,2),np.round(rsquared2,2)))
    plt.xlabel(r"$h_r(\%),h_{\varphi}(degrees),h_t(s)$")
    plt.ylabel("$\hat{\gamma}(|h_i|)$")
    plt.plot(np.append(0,variograms2_polar_norm["xvariogram_hat"][:181,0]),'-o',label=r"$\hat{\gamma}(|h_{\varphi}|)$")
    plt.plot(np.append(0,variograms2_polar_norm["yvariogram_hat"][:100,0]),'-o',label="$\hat{\gamma}(|h_r|)$")
    plt.plot(np.append(0,variograms2_polar_norm["tvariogram_hat"][:,0]),'-o',label="$\hat{\gamma}(|h_t|)$")
    plt.axhline(zwind_cloud2_polar_norm_detrend[lwc_cloud2_polar>=1e-5].var())
    plt.legend()
    variograms_cloud2_txy['variograms_cloud2_txy_zs{}'.format(z_index)] = variograms2_polar_norm
    vars_cs = np.append(vars_cs,zwind_cloud2_polar_norm_detrend[lwc_cloud2_polar>=1e-5].var())

variograms_cloud2_txy['rmse_cs'] = rmse_cs
variograms_cloud2_txy['r2_cs'] = r2_cs
variograms_cloud2_txy['zs'] = z_index_wanted
variograms_cloud2_txy['vars_cs'] = vars_cs
outfile = '/home/dselle/Skyscanner/data_exploration/results/plots_and_images/polar_analysis/Detrended_normalized_radial_variograms/cloud2/various_cs_t150/variograms_cloud2_txy.pkl'
with open(outfile, 'wb') as output:
    pickle.dump(variograms_cloud2_txy, output, pickle.HIGHEST_PROTOCOL)


######### For loop to calculate z,r,phi Variograms at 5 seconds span
################# Due To RAM issues, better run this for loop alone

rmse_cs = np.array([])
r2_cs = np.array([])
vars_cs = np.array([])
t_index_wanted = [935,955,975,995,1015]
variograms_cloud2_zxy = {}
for t_index in t_index_wanted:
    lwc_data2 = atm.data['RCT'][t_index:(t_index+5),90:123,125:250,270:370]
    zwind_data2 = atm.data['WT'][t_index:(t_index+5),90:123,125:250,270:370]
    ids2,counter2,clouds2=cloud.cloud_segmentation(lwc_data2)

    clouds2=list(set(clouds2.values()))
    length_point_clds = np.ndarray((0,1))
    for each_cloud in clouds2:
        print(len(each_cloud.points))
        temp = len(each_cloud.points)
        length_point_clds = np.vstack((length_point_clds,temp))

    cloud2= clouds2[np.argmax(length_point_clds)]
    print('calculating attributes')
    cloud2.calculate_attributes(lwc_data2,zwind_data2)
    print('Creating lwc_cloud2')
    lwc_cloud2 = np.zeros(lwc_data2.shape)
    for point in cloud2.points:
        lwc_cloud2[point] = 1


    # Coordinates to interpolate values inside of cloud1
    xr =np.arange(0.005 + 125*0.01, 0.005 + 250*0.01,0.01)
    yr= np.arange(0.005 + 270*0.01, 0.005 + 370*0.01,0.01)
    all_Zs=atm.data["VLEV"][:,0,0]
    zr = all_Zs[90:123]
    #zr = np.arange(1.185,1.185 + 15*0.01,0.01)
    tr = np.arange(t_index,t_index+5)
    zspan = np.arange(0,33)
    #tr = 449
    #M = np.array(np.meshgrid(tr,zr, xr, yr, indexing='ij'))
    #grid = np.stack(M, axis=-1)


    points_span = (tr,zr,xr,yr)
    origin_xy = [125,270]
    polar_cloud2,polar_cloud2_norm = polar_cloud_norm(points_span,lwc_cloud2,cloud2.COM_2D_lwc_tz,zspan,origin_xy)

    #times_wanted = np.concatenate((np.arange(0,25),np.arange(26,85),np.arange(86,145),np.arange(146,150)))
    times_wanted = np.arange(0,5)
    zwind_cloud2_polar_norm = atm.get_points(polar_cloud2_norm[times_wanted],'WT','linear')
    zwind_cloud2_polar_norm = zwind_cloud2_polar_norm.reshape((5,33,360,151))
    zwind_cloud2_polar_norm_detrend = zwind_cloud2_polar_norm.copy()

    zwind_com2d_lwc_z = np.array([])
    for z in range(zspan[0],zspan[len(zspan)-1],10):
        COMx = cloud2.COM_2D_lwc_tz[0,z,0]
        COMy = cloud2.COM_2D_lwc_tz[0,z,1]

        zwind_com2d_lwc_temp = atm.get_points([tr[0],float(zr[z]),0.005 + (origin_xy[0] +COMx)*0.01,0.005 + (origin_xy[1] +COMy)*0.01],'WT','linear')
        zwind_com2d_lwc_z = np.append(zwind_com2d_lwc_z,zwind_com2d_lwc_temp)

    zwind_com_lwc_mean_cloud = np.mean(zwind_com2d_lwc_z)
    zwind_com_lwc_median_cloud = np.median(zwind_com2d_lwc_z)

    interpolate_points_cloud = RegularGridInterpolator(points=points_span,values=lwc_cloud2,bounds_error=False,fill_value=0)
    lwc_cloud2_polar = interpolate_points_cloud(polar_cloud2_norm[times_wanted],'nearest')
    lwc_cloud2_polar = lwc_cloud2_polar.reshape((5,33,360,151))

    #rsquared2_zwind_com = np.array([])
    #for zwind_com_lwc_mean_plane in np.arange(3,4.5,0.1):
    ########### Using one cross-section to detrend the whole cloud
    for r in range(0,151):
        zwind_cloud2_polar_norm_detrend[:,:,:,r] = zwind_cloud2_polar_norm[:,:,:,r] - zwind_com_lwc_mean_cloud*rtrend_global_median_updraft_norm[r]
    ## R**2 By definition
    explained_variance = np.sum(zwind_cloud2_polar_norm_detrend[lwc_cloud2_polar>=1e-5]**2)
    total_variance = zwind_cloud2_polar_norm[lwc_cloud2_polar>=1e-5].var() * np.count_nonzero(zwind_cloud2_polar_norm[lwc_cloud2_polar>=1e-5])
    rsquared2 = 1 - explained_variance/total_variance
    r2_cs = np.append(r2_cs,rsquared2)
    #rsquared2_zwind_com = np.append(rsquared2_zwind_com,rsquared2)

    plt.figure()
    plt.hist(zwind_cloud2_polar_norm_detrend[lwc_cloud2_polar>=1e-5].reshape(-1),bins=30)
    plt.xlabel('Residuals(m/s)')
    plt.ylabel('Count')
    plt.title('Residuals of Trend,t={}...{},Zwind at center = {} m/s, $R^2$ of trend = {}'.format(t_index,t_index+5,
                np.round(zwind_com_lwc_mean_cloud,2),np.round(rsquared2,2)))
    rmse =np.sqrt(np.sum(zwind_cloud2_polar_norm_detrend[lwc_cloud2_polar>=1e-5]**2)/np.count_nonzero(zwind_cloud2_polar_norm_detrend[lwc_cloud2_polar>=1e-5]))
    rmse_cs = np.append(rmse_cs,rmse)

    variograms2_polar_norm= cloud.sample_variogram(zwind_cloud2_polar_norm_detrend,'classical',lwc_cloud2_polar)

    plt.figure()
    plt.title("Cloud2,t={}...{}, Zwind at center = {} m/s, $R^2$ of trend = {}".format(t_index,t_index+5,
                np.round(zwind_com_lwc_mean_cloud,2),np.round(rsquared2,2)))
    plt.xlabel(r"$h_r(\%),h_{\varphi}(degrees),h_z(10m)$")
    plt.ylabel("$\hat{\gamma}(|h_i|)$")
    plt.plot(np.append(0,variograms2_polar_norm["xvariogram_hat"][:181,0]),'-o',label=r"$\hat{\gamma}(|h_{\varphi}|)$")
    plt.plot(np.append(0,variograms2_polar_norm["yvariogram_hat"][:100,0]),'-o',label="$\hat{\gamma}(|h_r|)$")
    plt.plot(np.append(0,variograms2_polar_norm["zvariogram_hat"][:,0]),'-o',label="$\hat{\gamma}(|h_z|)$")
    plt.axhline(zwind_cloud2_polar_norm_detrend[lwc_cloud2_polar>=1e-5].var())
    plt.legend()
    variograms_cloud2_zxy['variograms_cloud2_zxy_ts{}'.format(t_index)] = variograms2_polar_norm
    vars_cs = np.append(vars_cs,zwind_cloud2_polar_norm_detrend[lwc_cloud2_polar>=1e-5].var())

variograms_cloud2_zxy['rmse_cs'] = rmse_cs
variograms_cloud2_zxy['r2_cs'] = r2_cs
variograms_cloud2_zxy['ts'] = t_index_wanted
variograms_cloud2_zxy['vars_cs'] = vars_cs
outfile = '/home/dselle/Skyscanner/data_exploration/results/plots_and_images/polar_analysis/Detrended_normalized_radial_variograms/cloud2/entire_cloud_t5_33cs/variograms_cloud2_zxy.pkl'
with open(outfile, 'wb') as output:
    pickle.dump(variograms_cloud2_zxy, output, pickle.HIGHEST_PROTOCOL)


################################################################
########################### cloud3 #############################
################################################################

######### For loop to calculate t,r,phi Variograms at different Cross-sections
################# Due To RAM issues, better run this for loop alone
rmse_cs = np.array([])
r2_cs = np.array([])
z_index_wanted = np.arange(90,126,5)
variograms_cloud3_txy = {}
vars_cs = np.array([])
for z_index in z_index_wanted:

    lwc_data3 = atm.data['RCT'][1535:1685,z_index:(z_index+2),0:115,50:200]
    zwind_data3 = atm.data['WT'][1535:1685,z_index:(z_index+2),0:115,50:200]

    ids3,counter3,clouds3=cloud.cloud_segmentation(lwc_data3)
    clouds3=list(set(clouds3.values()))
    length_point_clds = np.ndarray((0,1))
    for each_cloud in clouds3:
        print(len(each_cloud.points))
        temp = len(each_cloud.points)
        length_point_clds = np.vstack((length_point_clds,temp))

    cloud3 = clouds3[np.argmax(length_point_clds)]

    cloud3.calculate_attributes(lwc_data3,zwind_data3)


    lwc_cloud3 = np.zeros(lwc_data3.shape)
    for point in cloud3.points:
        lwc_cloud3[point] = 1

    # Coordinates to interpolate values inside of cloud1
    xr =np.arange(0.005 + 0*0.01, 0.005 + 114*0.01,0.01)
    yr= np.arange(0.005 + 50*0.01, 0.005 + 200*0.01,0.01)
    #zr = np.arange(1.135,1.135+30*0.01,0.01)
    all_Zs=atm.data["VLEV"][:,0,0]
    zr = all_Zs[z_index:z_index+2]
    tr = np.arange(1535,1685)
    zspan = np.arange(0,1)

    points_span = (tr,zr,xr,yr)
    origin_xy = [0,50]

    polar_cloud3,polar_cloud3_norm = polar_cloud_norm(points_span,lwc_cloud3,cloud3.COM_2D_lwc_tz,zspan,origin_xy)

    times_wanted = np.concatenate((np.arange(0,25),np.arange(26,85),np.arange(87,145)))
    zwind_cloud3_polar_norm = atm.get_points(polar_cloud3_norm[times_wanted,0],'WT','linear')
    zwind_cloud3_polar_norm = zwind_cloud3_polar_norm.reshape((142,1,360,151))
    zwind_cloud3_polar_norm_detrend = zwind_cloud3_polar_norm.copy()

    zwind_com2d_lwc_t = np.array([])
    for t in range(tr[0],tr[len(tr)-1],20):
        trel = t - tr[0]
        COMx = cloud3.COM_2D_lwc_tz[trel,0,0]
        COMy = cloud3.COM_2D_lwc_tz[trel,0,1]

        zwind_com2d_lwc_temp = atm.get_points([t,float(zr[0]),0.005 + (origin_xy[0]+COMx)*0.01,0.005 + (origin_xy[1]+COMy)*0.01],'WT','linear')
        zwind_com2d_lwc_t = np.append(zwind_com2d_lwc_t,zwind_com2d_lwc_temp)

    zwind_com_lwc_mean_plane = np.mean(zwind_com2d_lwc_t)
    zwind_com_lwc_median_plane = np.median(zwind_com2d_lwc_t)

    interpolate_points_cloud = RegularGridInterpolator(points=points_span,values=lwc_cloud3,bounds_error=False,fill_value=0)
    lwc_cloud3_polar = interpolate_points_cloud(polar_cloud3_norm[times_wanted,0],'nearest')
    lwc_cloud3_polar = lwc_cloud3_polar.reshape((142,1,360,151))

    #rsquared2_zwind_com = np.array([])
    #for zwind_com_lwc_mean_plane in np.arange(3,4.5,0.1):
    ########### Using one cross-section to detrend the whole cloud
    for r in range(0,151):
        zwind_cloud3_polar_norm_detrend[:,:,:,r] = zwind_cloud3_polar_norm[:,:,:,r] - zwind_com_lwc_mean_plane*rtrend_global_median_updraft_norm[r]
    ## R**2 By definition
    explained_variance = np.sum(zwind_cloud3_polar_norm_detrend[lwc_cloud3_polar>=1e-5]**2)
    total_variance = zwind_cloud3_polar_norm[lwc_cloud3_polar>=1e-5].var() * np.count_nonzero(zwind_cloud3_polar_norm[lwc_cloud3_polar>=1e-5])
    rsquared2 = 1 - explained_variance/total_variance
    r2_cs = np.append(r2_cs,rsquared2)
    #rsquared2_zwind_com = np.append(rsquared2_zwind_com,rsquared2)

    plt.figure()
    plt.hist(zwind_cloud3_polar_norm_detrend[lwc_cloud3_polar>=1e-5].reshape(-1),bins=30)
    plt.xlabel('Residuals(m/s)')
    plt.ylabel('Count')
    plt.title('Residuals of Trend, z={} km, Zwind at center = {} m/s, $R^2$ of trend = {}'.format(np.round(float(zr[0]),3),
                np.round(zwind_com_lwc_mean_plane,2),np.round(rsquared2,2)))
    rmse =np.sqrt(np.sum(zwind_cloud3_polar_norm_detrend[lwc_cloud3_polar>=1e-5]**2)/np.count_nonzero(zwind_cloud3_polar_norm_detrend[lwc_cloud3_polar>=1e-5]))
    rmse_cs = np.append(rmse_cs,rmse)

    variograms3_polar_norm= cloud.sample_variogram(zwind_cloud3_polar_norm_detrend,'classical',lwc_cloud3_polar)

    plt.figure()
    plt.title("Cloud3, z={} km, Zwind at center = {} m/s, $R^2$ of trend = {}".format(np.round(float(zr[0]),3),
                np.round(zwind_com_lwc_mean_plane,2),np.round(rsquared2,2)))
    plt.xlabel(r"$h_r(\%),h_{\varphi}(degrees),h_t(s)$")
    plt.ylabel("$\hat{\gamma}(|h_i|)$")
    plt.plot(np.append(0,variograms3_polar_norm["xvariogram_hat"][:181,0]),'-o',label=r"$\hat{\gamma}(|h_{\varphi}|)$")
    plt.plot(np.append(0,variograms3_polar_norm["yvariogram_hat"][:100,0]),'-o',label="$\hat{\gamma}(|h_r|)$")
    plt.plot(np.append(0,variograms3_polar_norm["tvariogram_hat"][:,0]),'-o',label="$\hat{\gamma}(|h_t|)$")
    plt.axhline(zwind_cloud3_polar_norm_detrend[lwc_cloud3_polar>=1e-5].var())
    plt.legend()
    variograms_cloud3_txy['variograms_cloud3_txy_zs{}'.format(z_index)] = variograms3_polar_norm
    vars_cs = np.append(vars_cs,zwind_cloud3_polar_norm_detrend[lwc_cloud3_polar>=1e-5].var())

variograms_cloud3_txy['rmse_cs'] = rmse_cs
variograms_cloud3_txy['r2_cs'] = r2_cs
variograms_cloud3_txy['zs'] = z_index_wanted
variograms_cloud3_txy['vars_cs'] = vars_cs
outfile = '/home/dselle/Skyscanner/data_exploration/results/plots_and_images/polar_analysis/Detrended_normalized_radial_variograms/cloud3/various_cs_t150/variograms_cloud3_txy.pkl'
with open(outfile, 'wb') as output:
    pickle.dump(variograms_cloud3_txy, output, pickle.HIGHEST_PROTOCOL)

######### For loop to calculate z,r,phi Variograms at 5 seconds span
################# Due To RAM issues, better run this for loop alone

rmse_cs = np.array([])
r2_cs = np.array([])
vars_cs = np.array([])
t_index_wanted = [1535,1555,1575,1595,1615]
variograms_cloud3_zxy = {}
for t_index in t_index_wanted:
    lwc_data3 = atm.data['RCT'][t_index:(t_index+5),90:125,0:115,50:200]
    zwind_data3 = atm.data['WT'][t_index:(t_index+5),90:125,0:115,50:200]

    ids3,counter3,clouds3=cloud.cloud_segmentation(lwc_data3)

    clouds3=list(set(clouds3.values()))
    length_point_clds = np.ndarray((0,1))
    for each_cloud in clouds3:
        print(len(each_cloud.points))
        temp = len(each_cloud.points)
        length_point_clds = np.vstack((length_point_clds,temp))

    cloud3 = clouds3[np.argmax(length_point_clds)]
    print('calculating attributes')
    cloud3.calculate_attributes(lwc_data3,zwind_data3)
    print('Creating lwc_cloud3')
    lwc_cloud3 = np.zeros(lwc_data3.shape)
    for point in cloud3.points:
        lwc_cloud3[point] = 1


    # Coordinates to interpolate values inside of cloud3
    xr =np.arange(0.005 + 0*0.01, 0.005 + 114*0.01,0.01)
    yr= np.arange(0.005 + 50*0.01, 0.005 + 200*0.01,0.01)
    #zr = np.arange(1.135,1.135+30*0.01,0.01)
    all_Zs=atm.data["VLEV"][:,0,0]
    zr = all_Zs[90:125]
    tr = np.arange(t_index,(t_index+5))
    zspan = np.arange(0,35)

    points_span = (tr,zr,xr,yr)
    origin_xy = [0,50]

    polar_cloud3,polar_cloud3_norm = polar_cloud_norm(points_span,lwc_cloud3,cloud3.COM_2D_lwc_tz,zspan,origin_xy)

    #times_wanted = np.concatenate((np.arange(0,25),np.arange(26,85),np.arange(86,145),np.arange(146,150)))
    times_wanted = np.arange(0,5)
    zwind_cloud3_polar_norm = atm.get_points(polar_cloud3_norm[times_wanted],'WT','linear')
    zwind_cloud3_polar_norm = zwind_cloud3_polar_norm.reshape((5,35,360,151))
    zwind_cloud3_polar_norm_detrend = zwind_cloud3_polar_norm.copy()

    zwind_com2d_lwc_z = np.array([])
    for z in range(zspan[0],zspan[len(zspan)-1],10):
        COMx = cloud3.COM_2D_lwc_tz[0,z,0]
        COMy = cloud3.COM_2D_lwc_tz[0,z,1]

        zwind_com2d_lwc_temp = atm.get_points([tr[0],float(zr[z]),0.005 + (origin_xy[0] +COMx)*0.01,0.005 + (origin_xy[1] +COMy)*0.01],'WT','linear')
        zwind_com2d_lwc_z = np.append(zwind_com2d_lwc_z,zwind_com2d_lwc_temp)

    zwind_com_lwc_mean_cloud = np.mean(zwind_com2d_lwc_z)
    zwind_com_lwc_median_cloud = np.median(zwind_com2d_lwc_z)

    interpolate_points_cloud = RegularGridInterpolator(points=points_span,values=lwc_cloud3,bounds_error=False,fill_value=0)
    lwc_cloud3_polar = interpolate_points_cloud(polar_cloud3_norm[times_wanted],'nearest')
    lwc_cloud3_polar = lwc_cloud3_polar.reshape((5,35,360,151))

    #rsquared2_zwind_com = np.array([])
    #for zwind_com_lwc_mean_plane in np.arange(3,4.5,0.1):
    ########### Using one cross-section to detrend the whole cloud
    for r in range(0,151):
        zwind_cloud3_polar_norm_detrend[:,:,:,r] = zwind_cloud3_polar_norm[:,:,:,r] - zwind_com_lwc_mean_cloud*rtrend_global_median_updraft_norm[r]
    ## R**2 By definition
    explained_variance = np.sum(zwind_cloud3_polar_norm_detrend[lwc_cloud3_polar>=1e-5]**2)
    total_variance = zwind_cloud3_polar_norm[lwc_cloud3_polar>=1e-5].var() * np.count_nonzero(zwind_cloud3_polar_norm[lwc_cloud3_polar>=1e-5])
    rsquared2 = 1 - explained_variance/total_variance
    r2_cs = np.append(r2_cs,rsquared2)
    #rsquared2_zwind_com = np.append(rsquared2_zwind_com,rsquared2)

    plt.figure()
    plt.hist(zwind_cloud3_polar_norm_detrend[lwc_cloud3_polar>=1e-5].reshape(-1),bins=30)
    plt.xlabel('Residuals(m/s)')
    plt.ylabel('Count')
    plt.title('Residuals of Trend,t={}...{},Zwind at center = {} m/s, $R^2$ of trend = {}'.format(t_index,t_index+5,
                np.round(zwind_com_lwc_mean_cloud,2),np.round(rsquared2,2)))
    rmse =np.sqrt(np.sum(zwind_cloud3_polar_norm_detrend[lwc_cloud3_polar>=1e-5]**2)/np.count_nonzero(zwind_cloud3_polar_norm_detrend[lwc_cloud3_polar>=1e-5]))
    rmse_cs = np.append(rmse_cs,rmse)

    variograms3_polar_norm= cloud.sample_variogram(zwind_cloud3_polar_norm_detrend,'classical',lwc_cloud3_polar)

    plt.figure()
    plt.title("Cloud3,t={}...{}, Zwind at center = {} m/s, $R^2$ of trend = {}".format(t_index,t_index+5,
                np.round(zwind_com_lwc_mean_cloud,2),np.round(rsquared2,2)))
    plt.xlabel(r"$h_r(\%),h_{\varphi}(degrees),h_z(10m)$")
    plt.ylabel("$\hat{\gamma}(|h_i|)$")
    plt.plot(np.append(0,variograms3_polar_norm["xvariogram_hat"][:181,0]),'-o',label=r"$\hat{\gamma}(|h_{\varphi}|)$")
    plt.plot(np.append(0,variograms3_polar_norm["yvariogram_hat"][:100,0]),'-o',label="$\hat{\gamma}(|h_r|)$")
    plt.plot(np.append(0,variograms3_polar_norm["zvariogram_hat"][:,0]),'-o',label="$\hat{\gamma}(|h_z|)$")
    plt.axhline(zwind_cloud3_polar_norm_detrend[lwc_cloud3_polar>=1e-5].var())
    plt.legend()
    variograms_cloud3_zxy['variograms_cloud3_zxy_ts{}'.format(t_index)] = variograms3_polar_norm
    vars_cs = np.append(vars_cs,zwind_cloud3_polar_norm_detrend[lwc_cloud3_polar>=1e-5].var())


variograms_cloud3_zxy['rmse_cs'] = rmse_cs
variograms_cloud3_zxy['r2_cs'] = r2_cs
variograms_cloud3_zxy['ts'] = t_index_wanted
variograms_cloud3_zxy['vars_cs'] = vars_cs
outfile = '/home/dselle/Skyscanner/data_exploration/results/plots_and_images/polar_analysis/Detrended_normalized_radial_variograms/cloud3/entire_cloud_t5_33cs/variograms_cloud3_zxy.pkl'
with open(outfile, 'wb') as output:
    pickle.dump(variograms_cloud3_zxy, output, pickle.HIGHEST_PROTOCOL)


################################################################
########################### cloud4 #############################
################################################################


rmse_cs = np.array([])
r2_cs = np.array([])
z_index_wanted = np.arange(90,103,3)
variograms_cloud4_txy = {}
vars_cs = np.array([])
for z_index in z_index_wanted:

    lwc_data4 = atm.data['RCT'][2795:2945,z_index:(z_index+2),310:400,0:125]
    zwind_data4 = atm.data['WT'][2795:2945,z_index:(z_index+2),310:400,0:125]

    ids4,counter4,clouds4=cloud.cloud_segmentation(lwc_data4)
    clouds4=list(set(clouds4.values()))
    length_point_clds = np.ndarray((0,1))
    for each_cloud in clouds4:
        print(len(each_cloud.points))
        temp = len(each_cloud.points)
        length_point_clds = np.vstack((length_point_clds,temp))

    cloud4 = clouds4[np.argmax(length_point_clds)]

    cloud4.calculate_attributes(lwc_data4,zwind_data4)


    lwc_cloud4 = np.zeros(lwc_data4.shape)
    for point in cloud4.points:
        lwc_cloud4[point] = 1

    # Coordinates to interpolate values inside of cloud1
    xr =np.arange(0.005 + 310*0.01, 0.005 + 400*0.01,0.01)
    yr= np.arange(0.005 + 0*0.01, 0.005 + 125*0.01,0.01)
    #zr = np.arange(1.135,1.135+30*0.01,0.01)
    all_Zs=atm.data["VLEV"][:,0,0]
    zr = all_Zs[z_index:z_index+2]
    tr = np.arange(2795,2945)
    zspan = np.arange(0,1)

    points_span = (tr,zr,xr,yr)
    origin_xy = [310,0]

    polar_cloud4,polar_cloud4_norm = polar_cloud_norm(points_span,lwc_cloud4,cloud4.COM_2D_lwc_tz,zspan,origin_xy)

    times_wanted = np.concatenate((np.arange(0,25),np.arange(28,85),np.arange(86,145),np.arange(147,150)))
    zwind_cloud4_polar_norm = atm.get_points(polar_cloud4_norm[times_wanted,0],'WT','linear')
    zwind_cloud4_polar_norm = zwind_cloud4_polar_norm.reshape((144,1,360,151))
    zwind_cloud4_polar_norm_detrend = zwind_cloud4_polar_norm.copy()

    zwind_com2d_lwc_t = np.array([])
    for t in range(tr[0],tr[len(tr)-1],20):
        trel = t - tr[0]
        COMx = cloud4.COM_2D_lwc_tz[trel,0,0]
        COMy = cloud4.COM_2D_lwc_tz[trel,0,1]

        zwind_com2d_lwc_temp = atm.get_points([t,float(zr[0]),0.005 + (origin_xy[0]+COMx)*0.01,0.005 + (origin_xy[1]+COMy)*0.01],'WT','linear')
        zwind_com2d_lwc_t = np.append(zwind_com2d_lwc_t,zwind_com2d_lwc_temp)

    zwind_com_lwc_mean_plane = np.mean(zwind_com2d_lwc_t)
    zwind_com_lwc_median_plane = np.median(zwind_com2d_lwc_t)

    interpolate_points_cloud = RegularGridInterpolator(points=points_span,values=lwc_cloud4,bounds_error=False,fill_value=0)
    lwc_cloud4_polar = interpolate_points_cloud(polar_cloud4_norm[times_wanted,0],'nearest')
    lwc_cloud4_polar = lwc_cloud4_polar.reshape((144,1,360,151))

    #rsquared2_zwind_com = np.array([])
    #for zwind_com_lwc_mean_plane in np.arange(3,4.5,0.1):
    ########### Using one cross-section to detrend the whole cloud
    for r in range(0,151):
        zwind_cloud4_polar_norm_detrend[:,:,:,r] = zwind_cloud4_polar_norm[:,:,:,r] - zwind_com_lwc_mean_plane*rtrend_global_median_updraft_norm[r]
    ## R**2 By definition
    explained_variance = np.sum(zwind_cloud4_polar_norm_detrend[lwc_cloud4_polar>=1e-5]**2)
    total_variance = zwind_cloud4_polar_norm[lwc_cloud4_polar>=1e-5].var() * np.count_nonzero(zwind_cloud4_polar_norm[lwc_cloud4_polar>=1e-5])
    rsquared2 = 1 - explained_variance/total_variance
    r2_cs = np.append(r2_cs,rsquared2)
    #rsquared2_zwind_com = np.append(rsquared2_zwind_com,rsquared2)

    plt.figure()
    plt.hist(zwind_cloud4_polar_norm_detrend[lwc_cloud4_polar>=1e-5].reshape(-1),bins=30)
    plt.xlabel('Residuals(m/s)')
    plt.ylabel('Count')
    plt.title('Residuals of Trend, z={} km, Zwind at center = {} m/s, $R^2$ of trend = {}'.format(np.round(float(zr[0]),3),
                np.round(zwind_com_lwc_mean_plane,2),np.round(rsquared2,2)))
    rmse =np.sqrt(np.sum(zwind_cloud4_polar_norm_detrend[lwc_cloud4_polar>=1e-5]**2)/np.count_nonzero(zwind_cloud4_polar_norm_detrend[lwc_cloud4_polar>=1e-5]))
    rmse_cs = np.append(rmse_cs,rmse)

    variograms4_polar_norm= cloud.sample_variogram(zwind_cloud4_polar_norm_detrend,'classical',lwc_cloud4_polar)

    plt.figure()
    plt.title("Cloud4, z={} km, Zwind at center = {} m/s, $R^2$ of trend = {}".format(np.round(float(zr[0]),3),
                np.round(zwind_com_lwc_mean_plane,2),np.round(rsquared2,2)))
    plt.xlabel(r"$h_r(\%),h_{\varphi}(degrees),h_t(s)$")
    plt.ylabel("$\hat{\gamma}(|h_i|)$")
    plt.plot(np.append(0,variograms4_polar_norm["xvariogram_hat"][:181,0]),'-o',label=r"$\hat{\gamma}(|h_{\varphi}|)$")
    plt.plot(np.append(0,variograms4_polar_norm["yvariogram_hat"][:100,0]),'-o',label="$\hat{\gamma}(|h_r|)$")
    plt.plot(np.append(0,variograms4_polar_norm["tvariogram_hat"][:,0]),'-o',label="$\hat{\gamma}(|h_t|)$")
    plt.axhline(zwind_cloud4_polar_norm_detrend[lwc_cloud4_polar>=1e-5].var())
    plt.legend()
    variograms_cloud4_txy['variograms_cloud4_txy_zs{}'.format(z_index)] = variograms4_polar_norm
    vars_cs = np.append(vars_cs,zwind_cloud4_polar_norm_detrend[lwc_cloud4_polar>=1e-5].var())
variograms_cloud4_txy['rmse_cs'] = rmse_cs
variograms_cloud4_txy['r2_cs'] = r2_cs
variograms_cloud4_txy['zs'] = z_index_wanted
variograms_cloud4_txy['vars_cs'] = vars_cs
outfile = '/home/dselle/Skyscanner/data_exploration/results/plots_and_images/polar_analysis/Detrended_normalized_radial_variograms/cloud4/various_cs_t150/variograms_cloud4_txy.pkl'
with open(outfile, 'wb') as output:
    pickle.dump(variograms_cloud4_txy, output, pickle.HIGHEST_PROTOCOL)



######### For loop to calculate z,r,phi Variograms at 5 seconds span
################# Due To RAM issues, better run this for loop alone

rmse_cs = np.array([])
r2_cs = np.array([])
vars_cs = np.array([])
t_index_wanted = [2795,2815,2835,2855]
variograms_cloud4_zxy = {}
for t_index in t_index_wanted:
    lwc_data4 = atm.data['RCT'][t_index:(t_index+5),90:110,310:400,0:125]
    zwind_data4 = atm.data['WT'][t_index:(t_index+5),90:110,310:400,0:125]

    ids4,counter4,clouds4=cloud.cloud_segmentation(lwc_data4)

    clouds4=list(set(clouds4.values()))
    length_point_clds = np.ndarray((0,1))
    for each_cloud in clouds4:
        print(len(each_cloud.points))
        temp = len(each_cloud.points)
        length_point_clds = np.vstack((length_point_clds,temp))

    cloud4 = clouds4[np.argmax(length_point_clds)]
    print('calculating attributes')
    cloud4.calculate_attributes(lwc_data4,zwind_data4)
    print('Creating lwc_cloud4')
    lwc_cloud4 = np.zeros(lwc_data4.shape)
    for point in cloud4.points:
        lwc_cloud4[point] = 1


    # Coordinates to interpolate values inside of cloud1
    xr =np.arange(0.005 + 310*0.01, 0.005 + 400*0.01,0.01)
    yr= np.arange(0.005 + 0*0.01, 0.005 + 125*0.01,0.01)
    #zr = np.arange(1.135,1.135+30*0.01,0.01)
    all_Zs=atm.data["VLEV"][:,0,0]
    zr = all_Zs[90:110]
    tr = np.arange(t_index,(t_index+5))
    zspan = np.arange(0,20)

    points_span = (tr,zr,xr,yr)
    origin_xy = [310,0]

    polar_cloud4,polar_cloud4_norm = polar_cloud_norm(points_span,lwc_cloud4,cloud4.COM_2D_lwc_tz,zspan,origin_xy)

    #times_wanted = np.concatenate((np.arange(0,25),np.arange(26,85),np.arange(86,145),np.arange(146,150)))
    times_wanted = np.arange(0,5)
    zwind_cloud4_polar_norm = atm.get_points(polar_cloud4_norm[times_wanted],'WT','linear')
    zwind_cloud4_polar_norm = zwind_cloud4_polar_norm.reshape((5,20,360,151))
    #zwind_cloud4_polar_norm = zwind_cloud4_polar_norm + np.random.randn(5,20,360,151)*0.25
    zwind_cloud4_polar_norm_detrend = zwind_cloud4_polar_norm.copy()

    zwind_com2d_lwc_z = np.array([])
    for z in range(zspan[0],zspan[len(zspan)-1],6):
        COMx = cloud4.COM_2D_lwc_tz[0,z,0]
        COMy = cloud4.COM_2D_lwc_tz[0,z,1]

        zwind_com2d_lwc_temp = atm.get_points([tr[0],float(zr[z]),0.005 + (origin_xy[0] +COMx)*0.01,0.005 + (origin_xy[1] +COMy)*0.01],'WT','linear')
        zwind_com2d_lwc_z = np.append(zwind_com2d_lwc_z,zwind_com2d_lwc_temp)

    zwind_com_lwc_mean_cloud = np.mean(zwind_com2d_lwc_z)
    zwind_com_lwc_median_cloud = np.median(zwind_com2d_lwc_z)

    interpolate_points_cloud = RegularGridInterpolator(points=points_span,values=lwc_cloud4,bounds_error=False,fill_value=0)
    lwc_cloud4_polar = interpolate_points_cloud(polar_cloud4_norm[times_wanted],'nearest')
    lwc_cloud4_polar = lwc_cloud4_polar.reshape((5,20,360,151))

    #rsquared2_zwind_com = np.array([])
    #for zwind_com_lwc_mean_plane in np.arange(3,4.5,0.1):
    ########### Using one cross-section to detrend the whole cloud
    for r in range(0,151):
        zwind_cloud4_polar_norm_detrend[:,:,:,r] = zwind_cloud4_polar_norm[:,:,:,r] - zwind_com_lwc_mean_cloud*rtrend_global_median_updraft_norm[r]
    ## R**2 By definition
    explained_variance = np.sum(zwind_cloud4_polar_norm_detrend[lwc_cloud4_polar>=1e-5]**2)
    total_variance = zwind_cloud4_polar_norm[lwc_cloud4_polar>=1e-5].var() * np.count_nonzero(zwind_cloud4_polar_norm[lwc_cloud4_polar>=1e-5])
    rsquared2 = 1 - explained_variance/total_variance
    r2_cs = np.append(r2_cs,rsquared2)
    #rsquared2_zwind_com = np.append(rsquared2_zwind_com,rsquared2)

    plt.figure()
    plt.hist(zwind_cloud4_polar_norm_detrend[lwc_cloud4_polar>=1e-5].reshape(-1),bins=30)
    plt.xlabel('Residuals(m/s)')
    plt.ylabel('Count')
    plt.title('Residuals of Trend,t={}...{},Zwind at center = {} m/s, $R^2$ of trend = {}'.format(t_index,t_index+5,
                np.round(zwind_com_lwc_mean_cloud,2),np.round(rsquared2,2)))
    rmse =np.sqrt(np.sum(zwind_cloud4_polar_norm_detrend[lwc_cloud4_polar>=1e-5]**2)/np.count_nonzero(zwind_cloud4_polar_norm_detrend[lwc_cloud4_polar>=1e-5]))
    rmse_cs = np.append(rmse_cs,rmse)

    variograms4_polar_norm= cloud.sample_variogram(zwind_cloud4_polar_norm_detrend,'classical',lwc_cloud4_polar)

    plt.figure()
    plt.title("Cloud4,t={}...{}, Zwind at center = {} m/s, $R^2$ of trend = {}".format(t_index,t_index+5,
                np.round(zwind_com_lwc_mean_cloud,2),np.round(rsquared2,2)))
    plt.xlabel(r"$h_r(\%),h_{\varphi}(degrees),h_z(10m)$")
    plt.ylabel("$\hat{\gamma}(|h_i|)$")
    plt.plot(np.append(0,variograms4_polar_norm["xvariogram_hat"][:181,0]),'-o',label=r"$\hat{\gamma}(|h_{\varphi}|)$")
    plt.plot(np.append(0,variograms4_polar_norm["yvariogram_hat"][:100,0]),'-o',label="$\hat{\gamma}(|h_r|)$")
    plt.plot(np.append(0,variograms4_polar_norm["zvariogram_hat"][:,0]),'-o',label="$\hat{\gamma}(|h_z|)$")
    plt.axhline(zwind_cloud4_polar_norm_detrend[lwc_cloud4_polar>=1e-5].var())
    plt.legend()
    variograms_cloud4_zxy['variograms_cloud4_zxy_ts{}'.format(t_index)] = variograms4_polar_norm
    vars_cs = np.append(vars_cs,zwind_cloud4_polar_norm_detrend[lwc_cloud4_polar>=1e-5].var())

variograms_cloud4_zxy['rmse_cs'] = rmse_cs
variograms_cloud4_zxy['r2_cs'] = r2_cs
variograms_cloud4_zxy['ts'] = t_index_wanted
variograms_cloud4_zxy['vars_cs'] = vars_cs
outfile = '/home/dselle/Skyscanner/data_exploration/results/plots_and_images/polar_analysis/Detrended_normalized_radial_variograms/cloud4/entire_cloud_t5_33cs/variograms_cloud4_zxy.pkl'
with open(outfile, 'wb') as output:
    pickle.dump(variograms_cloud4_zxy, output, pickle.HIGHEST_PROTOCOL)


################################################################
########################### cloud5 #############################
################################################################

rmse_cs = np.array([])
r2_cs = np.array([])
z_index_wanted = np.arange(95,110,3)
variograms_cloud5_txy = {}
vars_cs = np.array([])
for z_index in z_index_wanted:

    lwc_data5 = atm.data['RCT'][3395:3540,z_index:(z_index+2),0:200,200:400]
    zwind_data5 = atm.data['WT'][3395:3540,z_index:(z_index+2),0:200,200:400]

    ids5,counter5,clouds5=cloud.cloud_segmentation(lwc_data5)
    clouds5=list(set(clouds5.values()))
    length_point_clds = np.ndarray((0,1))
    for each_cloud in clouds5:
        print(len(each_cloud.points))
        temp = len(each_cloud.points)
        length_point_clds = np.vstack((length_point_clds,temp))

    cloud5 = clouds5[np.argmax(length_point_clds)]

    cloud5.calculate_attributes(lwc_data5,zwind_data5)


    lwc_cloud5 = np.zeros(lwc_data5.shape)
    for point in cloud5.points:
        lwc_cloud5[point] = 1

    # Coordinates to interpolate values inside of cloud5
    #zr = np.arange(1.135,1.135+30*0.01,0.01)
    xr =np.arange(0.005 + 0*0.01, 0.005 + 200*0.01,0.01)
    yr= np.arange(0.005 + 200*0.01, 0.005 + 400*0.01,0.01)

    all_Zs=atm.data["VLEV"][:,0,0]
    zr = all_Zs[z_index:z_index+2]
    tr = np.arange(3395,3540)


    zspan = np.arange(0,1)
    points_span = (tr,zr,xr,yr)
    origin_xy = [0,200]

    polar_cloud5,polar_cloud5_norm = polar_cloud_norm(points_span,lwc_cloud5,cloud5.COM_2D_lwc_tz,zspan,origin_xy)

    times_wanted = np.concatenate((np.arange(0,25),np.arange(26,85),np.arange(86,145)))
    zwind_cloud5_polar_norm = atm.get_points(polar_cloud5_norm[times_wanted,0],'WT','linear')
    zwind_cloud5_polar_norm = zwind_cloud5_polar_norm.reshape((143,1,360,151))
    zwind_cloud5_polar_norm_detrend = zwind_cloud5_polar_norm.copy()

    zwind_com2d_lwc_t = np.array([])
    for t in range(tr[0],tr[len(tr)-1],20):
        trel = t - tr[0]
        COMx = cloud5.COM_2D_lwc_tz[trel,0,0]
        COMy = cloud5.COM_2D_lwc_tz[trel,0,1]

        zwind_com2d_lwc_temp = atm.get_points([t,float(zr[0]),0.005 + (origin_xy[0]+COMx)*0.01,0.005 + (origin_xy[1]+COMy)*0.01],'WT','linear')
        zwind_com2d_lwc_t = np.append(zwind_com2d_lwc_t,zwind_com2d_lwc_temp)

    zwind_com_lwc_mean_plane = np.mean(zwind_com2d_lwc_t)
    zwind_com_lwc_median_plane = np.median(zwind_com2d_lwc_t)

    interpolate_points_cloud = RegularGridInterpolator(points=points_span,values=lwc_cloud5,bounds_error=False,fill_value=0)
    lwc_cloud5_polar = interpolate_points_cloud(polar_cloud5_norm[times_wanted,0],'nearest')
    lwc_cloud5_polar = lwc_cloud5_polar.reshape((143,1,360,151))

    #rsquared2_zwind_com = np.array([])
    #for zwind_com_lwc_mean_plane in np.arange(3,4.5,0.1):
    ########### Using one cross-section to detrend the whole cloud
    for r in range(0,151):
        zwind_cloud5_polar_norm_detrend[:,:,:,r] = zwind_cloud5_polar_norm[:,:,:,r] - zwind_com_lwc_mean_plane*rtrend_global_median_updraft_norm[r]
    ## R**2 By definition
    explained_variance = np.sum(zwind_cloud5_polar_norm_detrend[lwc_cloud5_polar>=1e-5]**2)
    total_variance = zwind_cloud5_polar_norm[lwc_cloud5_polar>=1e-5].var() * np.count_nonzero(zwind_cloud5_polar_norm[lwc_cloud5_polar>=1e-5])
    rsquared2 = 1 - explained_variance/total_variance
    r2_cs = np.append(r2_cs,rsquared2)
    #rsquared2_zwind_com = np.append(rsquared2_zwind_com,rsquared2)

    plt.figure()
    plt.hist(zwind_cloud5_polar_norm_detrend[lwc_cloud5_polar>=1e-5].reshape(-1),bins=30)
    plt.xlabel('Residuals(m/s)')
    plt.ylabel('Count')
    plt.title('Residuals of Trend, z={} km, Zwind at center = {} m/s, R^2 of trend = {}'.format(np.round(float(zr[0]),3),
                np.round(zwind_com_lwc_mean_plane,2),np.round(rsquared2,2)))
    rmse =np.sqrt(np.sum(zwind_cloud5_polar_norm_detrend[lwc_cloud5_polar>=1e-5]**2)/np.count_nonzero(zwind_cloud5_polar_norm_detrend[lwc_cloud5_polar>=1e-5]))
    rmse_cs = np.append(rmse_cs,rmse)

    variograms5_polar_norm= cloud.sample_variogram(zwind_cloud5_polar_norm_detrend,'classical',lwc_cloud5_polar)

    plt.figure()
    plt.title("Cloud5, z={} km, Zwind at center = {} m/s, $R^2$ of trend = {}".format(np.round(float(zr[0]),3),
                np.round(zwind_com_lwc_mean_plane,2),np.round(rsquared2,2)))
    plt.xlabel(r"$h_r(\%),h_{\varphi}(degrees),h_t(s)$")
    plt.ylabel("$\hat{\gamma}(|h_i|)$")
    plt.plot(np.append(0,variograms5_polar_norm["xvariogram_hat"][:181,0]),'-o',label=r"$\hat{\gamma}(|h_{\varphi}|)$")
    plt.plot(np.append(0,variograms5_polar_norm["yvariogram_hat"][:100,0]),'-o',label="$\hat{\gamma}(|h_r|)$")
    plt.plot(np.append(0,variograms5_polar_norm["tvariogram_hat"][:,0]),'-o',label="$\hat{\gamma}(|h_t|)$")
    plt.axhline(zwind_cloud5_polar_norm_detrend[lwc_cloud5_polar>=1e-5].var())
    plt.legend()
    variograms_cloud5_txy['variograms_cloud5_txy_zs{}'.format(z_index)] = variograms5_polar_norm
    vars_cs = np.append(vars_cs,zwind_cloud5_polar_norm_detrend[lwc_cloud5_polar>=1e-5].var())



variograms_cloud5_txy['rmse_cs'] = rmse_cs
variograms_cloud5_txy['r2_cs'] = r2_cs
variograms_cloud5_txy['zs'] = z_index_wanted
variograms_cloud5_txy['vars_cs'] = vars_cs
outfile = '/home/dselle/Skyscanner/data_exploration/results/plots_and_images/polar_analysis/Detrended_normalized_radial_variograms/cloud5/various_cs_t150/variograms_cloud5_txy.pkl'
with open(outfile, 'wb') as output:
    pickle.dump(variograms_cloud5_txy, output, pickle.HIGHEST_PROTOCOL)

############################################
######### For loop to calculate z,r,phi Variograms at 5 seconds span
################# Due To RAM issues, better run this for loop alone

rmse_cs = np.array([])
r2_cs = np.array([])
vars_cs = np.array([])
t_index_wanted = [3395,3415,3435,3455,3475]
variograms_cloud5_zxy = {}
for t_index in t_index_wanted:
    lwc_data5 = atm.data['RCT'][t_index:(t_index+5),95:111,0:200,200:400]
    zwind_data5 = atm.data['WT'][t_index:(t_index+5),95:111,0:200,200:400]

    ids5,counter5,clouds5=cloud.cloud_segmentation(lwc_data5)

    clouds5=list(set(clouds5.values()))
    length_point_clds = np.ndarray((0,1))
    for each_cloud in clouds5:
        print(len(each_cloud.points))
        temp = len(each_cloud.points)
        length_point_clds = np.vstack((length_point_clds,temp))

    cloud5 = clouds5[np.argmax(length_point_clds)]
    print('calculating attributes')
    cloud5.calculate_attributes(lwc_data5,zwind_data5)
    print('Creating lwc_cloud5')
    lwc_cloud5 = np.zeros(lwc_data5.shape)
    for point in cloud5.points:
        lwc_cloud5[point] = 1


    # Coordinates to interpolate values inside of cloud1
    xr =np.arange(0.005 + 0*0.01, 0.005 + 200*0.01,0.01)
    yr= np.arange(0.005 + 200*0.01, 0.005 + 400*0.01,0.01)
    #zr = np.arange(1.135,1.135+30*0.01,0.01)
    all_Zs=atm.data["VLEV"][:,0,0]
    zr = all_Zs[95:111]
    tr = np.arange(t_index,(t_index+5))
    zspan = np.arange(0,16)

    points_span = (tr,zr,xr,yr)
    origin_xy = [0,200]

    polar_cloud5,polar_cloud5_norm = polar_cloud_norm(points_span,lwc_cloud5,cloud5.COM_2D_lwc_tz,zspan,origin_xy)

    #times_wanted = np.concatenate((np.arange(0,25),np.arange(26,85),np.arange(86,145),np.arange(146,150)))
    times_wanted = np.arange(0,5)
    zwind_cloud5_polar_norm = atm.get_points(polar_cloud5_norm[times_wanted],'WT','linear')
    zwind_cloud5_polar_norm = zwind_cloud5_polar_norm.reshape((5,16,360,151))
    #zwind_cloud5_polar_norm = zwind_cloud5_polar_norm + np.random.randn(5,28,360,151)*0.25
    zwind_cloud5_polar_norm_detrend = zwind_cloud5_polar_norm.copy()

    zwind_com2d_lwc_z = np.array([])
    for z in range(zspan[0],zspan[len(zspan)-1],5):
        COMx = cloud5.COM_2D_lwc_tz[0,z,0]
        COMy = cloud5.COM_2D_lwc_tz[0,z,1]

        zwind_com2d_lwc_temp = atm.get_points([tr[0],float(zr[z]),0.005 + (origin_xy[0] +COMx)*0.01,0.005 + (origin_xy[1] +COMy)*0.01],'WT','linear')
        zwind_com2d_lwc_z = np.append(zwind_com2d_lwc_z,zwind_com2d_lwc_temp)

    zwind_com_lwc_mean_cloud = np.mean(zwind_com2d_lwc_z)
    zwind_com_lwc_median_cloud = np.median(zwind_com2d_lwc_z)

    interpolate_points_cloud = RegularGridInterpolator(points=points_span,values=lwc_cloud5,bounds_error=False,fill_value=0)
    lwc_cloud5_polar = interpolate_points_cloud(polar_cloud5_norm[times_wanted],'nearest')
    lwc_cloud5_polar = lwc_cloud5_polar.reshape((5,16,360,151))

    #rsquared2_zwind_com = np.array([])
    #for zwind_com_lwc_mean_plane in np.arange(3,4.5,0.1):
    ########### Using one cross-section to detrend the whole cloud
    for r in range(0,151):
        zwind_cloud5_polar_norm_detrend[:,:,:,r] = zwind_cloud5_polar_norm[:,:,:,r] - zwind_com_lwc_mean_cloud*rtrend_global_median_updraft_norm[r]
    ## R**2 By definition
    explained_variance = np.sum(zwind_cloud5_polar_norm_detrend[lwc_cloud5_polar>=1e-5]**2)
    total_variance = zwind_cloud5_polar_norm[lwc_cloud5_polar>=1e-5].var() * np.count_nonzero(zwind_cloud5_polar_norm[lwc_cloud5_polar>=1e-5])
    rsquared2 = 1 - explained_variance/total_variance
    r2_cs = np.append(r2_cs,rsquared2)
    #rsquared2_zwind_com = np.append(rsquared2_zwind_com,rsquared2)

    plt.figure()
    plt.hist(zwind_cloud5_polar_norm_detrend[lwc_cloud5_polar>=1e-5].reshape(-1),bins=30)
    plt.xlabel('Residuals(m/s)')
    plt.ylabel('Count')
    plt.title('Residuals of Trend,t={}...{},Zwind at center = {} m/s, $R^2$ of trend = {}'.format(t_index,t_index+5,
                np.round(zwind_com_lwc_mean_cloud,2),np.round(rsquared2,2)))
    rmse =np.sqrt(np.sum(zwind_cloud5_polar_norm_detrend[lwc_cloud5_polar>=1e-5]**2)/np.count_nonzero(zwind_cloud5_polar_norm_detrend[lwc_cloud5_polar>=1e-5]))
    rmse_cs = np.append(rmse_cs,rmse)

    variograms5_polar_norm= cloud.sample_variogram(zwind_cloud5_polar_norm_detrend,'classical',lwc_cloud5_polar)

    plt.figure()
    plt.title("Cloud5,t={}...{}, Zwind at center = {} m/s, $R^2$ of trend = {}".format(t_index,t_index+5,
                np.round(zwind_com_lwc_mean_cloud,2),np.round(rsquared2,2)))
    plt.xlabel(r"$h_r(\%),h_{\varphi}(degrees),h_z(10m)$")
    plt.ylabel("$\hat{\gamma}(|h_i|)$")
    plt.plot(np.append(0,variograms5_polar_norm["xvariogram_hat"][:181,0]),'-o',label=r"$\hat{\gamma}(|h_{\varphi}|)$")
    plt.plot(np.append(0,variograms5_polar_norm["yvariogram_hat"][:100,0]),'-o',label="$\hat{\gamma}(|h_r|)$")
    plt.plot(np.append(0,variograms5_polar_norm["zvariogram_hat"][:,0]),'-o',label="$\hat{\gamma}(|h_z|)$")
    plt.axhline(zwind_cloud5_polar_norm_detrend[lwc_cloud5_polar>=1e-5].var())
    plt.legend()

    variograms_cloud5_zxy['variograms_cloud5_zxy_ts{}'.format(t_index)] = variograms5_polar_norm
    vars_cs = np.append(vars_cs,zwind_cloud5_polar_norm_detrend[lwc_cloud5_polar>=1e-5].var())

variograms_cloud5_zxy['rmse_cs'] = rmse_cs
variograms_cloud5_zxy['r2_cs'] = r2_cs
variograms_cloud5_zxy['ts'] = t_index_wanted
variograms_cloud5_zxy['vars_cs'] = vars_cs
outfile = '/home/dselle/Skyscanner/data_exploration/results/plots_and_images/polar_analysis/Detrended_normalized_radial_variograms/cloud5/entire_cloud_t5_33cs/variograms_cloud5_zxy.pkl'
with open(outfile, 'wb') as output:
    pickle.dump(variograms_cloud5_zxy, output, pickle.HIGHEST_PROTOCOL)

##############################################
######## Fitting the variograms ##############
##############################################

outfile = '/home/dselle/Skyscanner/data_exploration/results/plots_and_images/polar_analysis/Detrended_normalized_radial_variograms/cloud1/various_cs_t150/variograms_cloud1_txy.pkl'
with open(outfile, 'rb') as input:
    print('Opening file')
    variograms_cloud1_txy = pickle.load(input)

outfile = '/home/dselle/Skyscanner/data_exploration/results/plots_and_images/polar_analysis/Detrended_normalized_radial_variograms/cloud2/various_cs_t150/variograms_cloud2_txy.pkl'
with open(outfile, 'rb') as input:
    print('Opening file')
    variograms_cloud2_txy = pickle.load(input)

outfile = '/home/dselle/Skyscanner/data_exploration/results/plots_and_images/polar_analysis/Detrended_normalized_radial_variograms/cloud3/various_cs_t150/variograms_cloud3_txy.pkl'
with open(outfile, 'rb') as input:
    print('Opening file')
    variograms_cloud3_txy = pickle.load(input)

outfile = '/home/dselle/Skyscanner/data_exploration/results/plots_and_images/polar_analysis/Detrended_normalized_radial_variograms/cloud4/various_cs_t150/variograms_cloud4_txy.pkl'
with open(outfile, 'rb') as input:
    print('Opening file')
    variograms_cloud4_txy = pickle.load(input)

outfile = '/home/dselle/Skyscanner/data_exploration/results/plots_and_images/polar_analysis/Detrended_normalized_radial_variograms/cloud5/various_cs_t150/variograms_cloud5_txy.pkl'
with open(outfile, 'rb') as input:
    print('Opening file')
    variograms_cloud5_txy = pickle.load(input)

outfile = '/home/dselle/Skyscanner/data_exploration/results/plots_and_images/polar_analysis/Detrended_normalized_radial_variograms/cloud1/entire_cloud_t5_33cs/variograms_cloud1_zxy.pkl'
with open(outfile, 'rb') as input:
    print('Opening file')
    variograms_cloud1_zxy = pickle.load(input)

outfile = '/home/dselle/Skyscanner/data_exploration/results/plots_and_images/polar_analysis/Detrended_normalized_radial_variograms/cloud2/entire_cloud_t5_33cs/variograms_cloud2_zxy.pkl'
with open(outfile, 'rb') as input:
    print('Opening file')
    variograms_cloud2_zxy = pickle.load(input)

outfile = '/home/dselle/Skyscanner/data_exploration/results/plots_and_images/polar_analysis/Detrended_normalized_radial_variograms/cloud3/entire_cloud_t5_33cs/variograms_cloud3_zxy.pkl'
with open(outfile, 'rb') as input:
    print('Opening file')
    variograms_cloud3_zxy = pickle.load(input)

outfile = '/home/dselle/Skyscanner/data_exploration/results/plots_and_images/polar_analysis/Detrended_normalized_radial_variograms/cloud4/entire_cloud_t5_33cs/variograms_cloud4_zxy.pkl'
with open(outfile, 'rb') as input:
    print('Opening file')
    variograms_cloud4_zxy = pickle.load(input)

outfile = '/home/dselle/Skyscanner/data_exploration/results/plots_and_images/polar_analysis/Detrended_normalized_radial_variograms/cloud5/entire_cloud_t5_33cs/variograms_cloud5_zxy.pkl'
with open(outfile, 'rb') as input:
    print('Opening file')
    variograms_cloud5_zxy = pickle.load(input)


############ Use txy which contains t,phi,r variograms, to get distribution of hyperparameter in the t direction

zs = variograms_cloud1_txy['zs']
tvariograms_fit_cloud1 = np.ndarray((4,len(zs),3)) # (nmodels,nvariograms,[params, cost])
for i in range(len(zs)):
    variograms_at_z = variograms_cloud1_txy['variograms_cloud1_txy_zs{}'.format(zs[i])]
    tvariogram_hat = variograms_at_z['tvariogram_hat']
    ht = np.arange(1,len(tvariogram_hat)+1)
    models = ('gaussian','exponential','matern_simp32','matern_simp52')
    for j in range(len(models)):
        params,cost,opt_object = cloud.fit_variogram(tvariogram_hat,ht,models[j])
        tvariograms_fit_cloud1[j,i] = [params[0],params[1],cost]

zs = variograms_cloud2_txy['zs']
tvariograms_fit_cloud2 = np.ndarray((4,len(zs),3)) # (nmodels,nvariograms,[params, cost])
for i in range(len(zs)):
    variograms_at_z = variograms_cloud2_txy['variograms_cloud2_txy_zs{}'.format(zs[i])]
    tvariogram_hat = variograms_at_z['tvariogram_hat']
    ht = np.arange(1,len(tvariogram_hat)+1)
    models = ('gaussian','exponential','matern_simp32','matern_simp52')
    for j in range(len(models)):
        params,cost,opt_object = cloud.fit_variogram(tvariogram_hat,ht,models[j])
        tvariograms_fit_cloud2[j,i] = [params[0],params[1],cost]

zs = variograms_cloud3_txy['zs']
tvariograms_fit_cloud3 = np.ndarray((4,len(zs),3)) # (nmodels,nvariograms,[params, cost])
for i in range(len(zs)):
    variograms_at_z = variograms_cloud3_txy['variograms_cloud3_txy_zs{}'.format(zs[i])]
    tvariogram_hat = variograms_at_z['tvariogram_hat']
    ht = np.arange(1,len(tvariogram_hat)+1)
    models = ('gaussian','exponential','matern_simp32','matern_simp52')
    for j in range(len(models)):
        params,cost,opt_object = cloud.fit_variogram(tvariogram_hat,ht,models[j])
        tvariograms_fit_cloud3[j,i] = [params[0],params[1],cost]

zs = variograms_cloud4_txy['zs']
tvariograms_fit_cloud4 = np.ndarray((4,len(zs),3)) # (nmodels,nvariograms,[params, cost])
for i in range(len(zs)):
    variograms_at_z = variograms_cloud4_txy['variograms_cloud4_txy_zs{}'.format(zs[i])]
    tvariogram_hat = variograms_at_z['tvariogram_hat']
    ht = np.arange(1,len(tvariogram_hat)+1)
    models = ('gaussian','exponential','matern_simp32','matern_simp52')
    for j in range(len(models)):
        params,cost,opt_object = cloud.fit_variogram(tvariogram_hat,ht,models[j])
        tvariograms_fit_cloud4[j,i] = [params[0],params[1],cost]

zs = variograms_cloud5_txy['zs']
tvariograms_fit_cloud5 = np.ndarray((4,len(zs),3)) # (nmodels,nvariograms,[params, cost])
for i in range(len(zs)):
    variograms_at_z = variograms_cloud5_txy['variograms_cloud5_txy_zs{}'.format(zs[i])]
    tvariogram_hat = variograms_at_z['tvariogram_hat']
    ht = np.arange(1,len(tvariogram_hat)+1)
    models = ('gaussian','exponential','matern_simp32','matern_simp52')
    for j in range(len(models)):
        params,cost,opt_object = cloud.fit_variogram(tvariogram_hat,ht,models[j])
        tvariograms_fit_cloud5[j,i] = [params[0],params[1],cost]

###################### fitting z,phi,r variograms
ts = variograms_cloud1_zxy['ts']
z_r_phi_variograms_fit_cloud1 = np.ndarray((3,4,len(ts),3)) # (ndim[z,phi,r],nmodels,nvariograms,[params, cost])
for i in range(len(ts)):
    variograms_at_t = variograms_cloud1_zxy['variograms_cloud1_zxy_ts{}'.format(ts[i])]
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
            params,cost,opt_object = cloud.fit_variogram(variogram_hat,h,models[k])
            z_r_phi_variograms_fit_cloud1[j,k,i] = [params[0],params[1],cost]


ts = variograms_cloud2_zxy['ts']
z_r_phi_variograms_fit_cloud2 = np.ndarray((3,4,len(ts),3)) # (ndim[z,phi,r],nmodels,nvariograms,[params, cost])
for i in range(len(ts)):
    variograms_at_t = variograms_cloud2_zxy['variograms_cloud2_zxy_ts{}'.format(ts[i])]
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
            params,cost,opt_object = cloud.fit_variogram(variogram_hat,h,models[k])
            z_r_phi_variograms_fit_cloud2[j,k,i] = [params[0],params[1],cost]


ts = variograms_cloud3_zxy['ts']
z_r_phi_variograms_fit_cloud3 = np.ndarray((3,4,len(ts),3)) # (ndim[z,phi,r],nmodels,nvariograms,[params, cost])
for i in range(len(ts)):
    variograms_at_t = variograms_cloud3_zxy['variograms_cloud3_zxy_ts{}'.format(ts[i])]
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
            params,cost,opt_object = cloud.fit_variogram(variogram_hat,h,models[k])
            z_r_phi_variograms_fit_cloud3[j,k,i] = [params[0],params[1],cost]

ts = variograms_cloud4_zxy['ts']
z_r_phi_variograms_fit_cloud4 = np.ndarray((3,4,len(ts),3)) # (ndim[z,phi,r],nmodels,nvariograms,[params, cost])
for i in range(len(ts)):
    variograms_at_t = variograms_cloud4_zxy['variograms_cloud4_zxy_ts{}'.format(ts[i])]
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
            params,cost,opt_object = cloud.fit_variogram(variogram_hat,h,models[k])
            z_r_phi_variograms_fit_cloud4[j,k,i] = [params[0],params[1],cost]

ts = variograms_cloud5_zxy['ts']
z_r_phi_variograms_fit_cloud5 = np.ndarray((3,4,len(ts),3)) # (ndim[z,phi,r],nmodels,nvariograms,[params, cost])
for i in range(len(ts)):
    variograms_at_t = variograms_cloud5_zxy['variograms_cloud5_zxy_ts{}'.format(ts[i])]
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
            params,cost,opt_object = cloud.fit_variogram(variogram_hat,h,models[k])
            z_r_phi_variograms_fit_cloud5[j,k,i] = [params[0],params[1],cost]



all_variograms_fit_t = np.concatenate((tvariograms_fit_cloud1,tvariograms_fit_cloud2,tvariograms_fit_cloud3,tvariograms_fit_cloud4,tvariograms_fit_cloud5),axis=1)
all_variograms_fit_z_r_phi = np.concatenate((z_r_phi_variograms_fit_cloud1,z_r_phi_variograms_fit_cloud2,z_r_phi_variograms_fit_cloud3,z_r_phi_variograms_fit_cloud4,z_r_phi_variograms_fit_cloud5),axis=2)

############### Exponential Model has the best performance:
min_model = np.argmin(np.sum(all_variograms_fit_t,axis=1)[:,2]+np.sum(all_variograms_fit_z_r_phi,axis=(0,2))[:,2])

font = {'size'   : 52}

plt.rc('font', **font)

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
