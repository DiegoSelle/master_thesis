import numpy as np
from mesonh_atm.mesonh_atmosphere import MesoNHAtmosphere
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import RegularGridInterpolator
import scipy.optimize as sciopt
import modules.cloud as ModCloud



#Old Data without advection
path = "/net/skyscanner/volume1/data/mesoNH/ARM_OneHour3600files_No_Horizontal_Wind/"
mfiles = [path+"U0K10.1.min{:02d}.{:03d}_diaKCL.nc".format(minute, second)
          for minute in range(1, 60)
          for second in range(1, 61)]
mtstep = 1
atm = MesoNHAtmosphere(mfiles, 1)

font = {'size'   : 26}

plt.rc('font', **font)

############################################################
#################### cloud example #########################
############################################################
lwc_data=atm.data['RCT'][449:455,75:125,60:200,110:250]
zwind_data=atm.data['WT'][449:455,75:125,60:200,110:250]
ids,counter,clouds=cloud.cloud_segmentation(lwc_data)

clouds=list(set(clouds.values()))
length_point_clds = np.ndarray((0,1))
for each_cloud in clouds:
    print(len(each_cloud.points))
    temp = len(each_cloud.points)
    length_point_clds = np.vstack((length_point_clds,temp))

cloud = clouds[np.argmax(length_point_clds)]

cloud1.calculate_attributes(lwc_data,zwind_data)
lwc_cloud1 = np.zeros(lwc_data.shape)
for point in cloud1.points:
    lwc_cloud1[point] = 1


# Coordinates to interpolate values inside of cloud
xr =np.arange(0.005 + 60*0.01, 0.005 + 200*0.01,0.01)
yr= np.arange(0.005 + 110*0.01, 0.005 + 250*0.01,0.01)
zr = np.arange(0.985,0.985+50*0.01,0.01)
tr = np.arange(449,455)
zspan = np.arange(10,50)
#tr = 449
#M = np.array(np.meshgrid(tr,zr, xr, yr, indexing='ij'))
#grid = np.stack(M, axis=-1)


points_span = (tr,zr,xr,yr)
origin_xy = [60,110]
polar_cloud1,polar_cloud1_norm = polar_cloud_norm(points_span,lwc_cloud1,cloud1.COM_2D_lwc_tz,zspan,origin_xy)

zwind_cloud1_polar_norm = atm.get_points(polar_cloud1_norm[:,10:48],'WT','linear')


interpolate_points_cloud = RegularGridInterpolator(points=points_span,values=lwc_cloud1,bounds_error=False,fill_value=0)
lwc_cloud1_polar = interpolate_points_cloud(polar_cloud1_norm[:,10:48],'nearest')
#lwc_cloud1_polar[lwc_cloud1_polar>=0.75]=1
#lwc_cloud1_polar[lwc_cloud1_polar<0.75]=0

plt.figure()
plt.contour(zwind_cloud1_polar_norm[0,15].T,origin='lower')
cbar=plt.colorbar()
cbar.set_label('m/s')
plt.contour(lwc_cloud1_polar[0,15].T,origin='lower',alpha=0.6,cmap='Greys')
plt.xlabel(r'$\varphi(Degrees)$')
plt.ylabel('Normalized Radius(%)')
plt.title('Zwind(m/s) normalized cloud1, z={}km'.format(np.round(float(zr[15]),3)))


plt.figure()
plt.imshow(lwc_cloud1_polar[0,15].T,origin='lower')
plt.xlabel('Phi(Degrees)')
plt.ylabel('Normalized Radius(%)')
cbar=plt.colorbar()
cbar.set_label('In-out of Cloud')
plt.title('Cloud1 Normalized')




########### trivial median Detrending ###############
rtrend_cloud1 = np.ndarray((6,38,360,151))
rtrend_cloud1[:] = np.NAN
for phi in range(0,360):
    trend_temp= zwind_cloud1_polar_norm[:,:,phi,:][lwc_cloud1_polar[:,:,phi,:]==1]
    rtrend_cloud1[:,:,phi,:][lwc_cloud1_polar[:,:,phi,:]==1] = trend_temp

rtrend_cloud1[:,:,:,101:151] = zwind_cloud1_polar_norm[:,:,:,101:151]
rtrend_cloud1_median = np.nanmedian(rtrend_cloud1,axis=(0,1,2))

radii = np.tile(np.arange(0,151),82080)
rtrend_cloud1_wo_nan = rtrend_cloud1.copy()
rtrend_cloud1_wo_nan[np.isnan(rtrend_cloud1_wo_nan)]=-3
prod_shape = np.prod(rtrend_cloud1_wo_nan.shape)/rtrend_cloud1_wo_nan.shape[3]

plt.figure()
plt.hist2d(radii,rtrend_cloud1_wo_nan.reshape(-1),bins=[151,20],vmin = 1e-3,
            weights= 100./(prod_shape)*np.ones_like(rtrend_cloud1_wo_nan.reshape(-1)),
            alpha=0.7)

plt.plot(rtrend_cloud1_median,'k-',label='Median Zwind')
plt.xlabel('Normalized Radius(%)')
plt.ylabel('Zwind(m/s)')
plt.title('Updraft Trend Cloud1')
cbar=plt.colorbar()
cbar.set_label('Frequency(%)')

########### visualizing the trend with updraft normalization ###############

rtrend_cloud1_updraft_norm = rtrend_cloud1.copy()
for t in range(rtrend_cloud1.shape[0]):
    for z in range(rtrend_cloud1.shape[1]):
        temp = rtrend_cloud1[t,z]
        temp = temp/temp[0,0]
        temp[temp>4]=np.NAN
        temp[temp<-2]= np.NAN
        rtrend_cloud1_updraft_norm[t,z]=temp


rtrend_cloud1_median_updraft_norm = np.nanmedian(rtrend_cloud1_updraft_norm,axis=(0,1,2))

rtrend_cloud1_wo_nan_updraft_norm = rtrend_cloud1_updraft_norm.copy()
rtrend_cloud1_wo_nan_updraft_norm[np.isnan(rtrend_cloud1_wo_nan_updraft_norm)]=-3
prod_shape = np.prod(rtrend_cloud1_wo_nan_updraft_norm.shape)/rtrend_cloud1_wo_nan_updraft_norm.shape[3]
radii = np.tile(np.arange(0,151),int(prod_shape))

plt.figure()
plt.hist2d(radii,rtrend_cloud1_wo_nan_updraft_norm.reshape(-1),bins=[151,20],vmin = 1e-3,
            weights= 100./(prod_shape)*np.ones_like(rtrend_cloud1_wo_nan_updraft_norm.reshape(-1)),
            alpha=0.7)

plt.plot(rtrend_cloud1_median_updraft_norm,'k-',label='Median Zwind')
plt.xlabel('Normalized Radius(%)')
plt.ylabel('Normalized Zwind(-)')
plt.title('Zwind Trend Cloud1')
cbar=plt.colorbar()
cbar.set_label('Frequency(%)')

output_file = 'results/plots_and_images/polar_analysis/Radial_trend/trends_data/rtrends_cloud.npz'
np.savez(output_file,rtrend_cloud1=rtrend_cloud1,rtrend_cloud1_updraft_norm=rtrend_cloud1_updraft_norm)


###########################################
########### fitting trend #################
###########################################

output_file1 = 'results/plots_and_images/polar_analysis/Radial_trend/trends_data/rtrends_cloud1.npz'
output_file2 = 'results/plots_and_images/polar_analysis/Radial_trend/trends_data/rtrends_cloud2.npz'
output_file3 = 'results/plots_and_images/polar_analysis/Radial_trend/trends_data/rtrends_cloud3.npz'
output_file4 = 'results/plots_and_images/polar_analysis/Radial_trend/trends_data/rtrends_cloud4.npz'
output_file5 = 'results/plots_and_images/polar_analysis/Radial_trend/trends_data/rtrends_cloud5.npz'

files_cloud1 = np.load(output_file1)
files_cloud2 = np.load(output_file2)
files_cloud3 = np.load(output_file3)
files_cloud4 = np.load(output_file4)
files_cloud5 = np.load(output_file5)

rtrend_cloud1_updraft_norm = files_cloud1['rtrend_cloud1_updraft_norm']
rtrend_cloud2_updraft_norm = files_cloud2['rtrend_cloud2_updraft_norm']
rtrend_cloud3_updraft_norm = files_cloud3['rtrend_cloud3_updraft_norm']
rtrend_cloud4_updraft_norm = files_cloud4['rtrend_cloud4_updraft_norm']
rtrend_cloud5_updraft_norm = files_cloud5['rtrend_cloud5_updraft_norm']

rtrend_global_updraft_norm = np.concatenate((rtrend_cloud1_updraft_norm,rtrend_cloud2_updraft_norm,
                                            rtrend_cloud3_updraft_norm,rtrend_cloud4_updraft_norm,
                                            rtrend_cloud5_updraft_norm),axis=1)

rtrend_global_median_updraft_norm = np.nanmedian(rtrend_global_updraft_norm,axis=(0,1,2))

rtrend_global_updraft_norm_wo_nan = rtrend_global_updraft_norm.copy()
rtrend_global_updraft_norm_wo_nan[np.isnan(rtrend_global_updraft_norm_wo_nan)]=-3
prod_shape = int(np.prod(rtrend_global_updraft_norm_wo_nan.shape)/rtrend_global_updraft_norm_wo_nan.shape[3])
radii = np.tile(np.arange(0,151),prod_shape)

plt.figure()
plt.hist2d(radii,rtrend_global_updraft_norm_wo_nan.reshape(-1),bins=[151,20],vmin = 1e-3,
            weights= 100./(prod_shape)*np.ones_like(rtrend_global_updraft_norm_wo_nan.reshape(-1)),
            alpha=0.7)

plt.plot(rtrend_global_median_updraft_norm,'k-',label='Median Zwind')
plt.xlabel('Normalized Radius(%)')
plt.ylabel('Zwind(-)')
plt.title('Zwind Trend Global')
cbar=plt.colorbar()
cbar.set_label('Frequency(%)')

outfile = 'results/plots_and_images/polar_analysis/Radial_trend/trends_data/rtrend_global_median_updraft_norm.npy'
np.save(outfile,rtrend_global_median_updraft_norm)


r = np.arange(0,151)

plsq1 = fit_radial_trend(rtrend_global_median_updraft_norm,r,"generalized_logistic")
plsq2 = fit_radial_trend(rtrend_global_median_updraft_norm,r,"polynomial")
plsq3 = fit_radial_trend(rtrend_global_median_updraft_norm,r,"inverse_quadratic")
plsq4 = fit_radial_trend(rtrend_global_median_updraft_norm,r,"rbf")
plsq5 = fit_radial_trend(rtrend_global_median_updraft_norm,r,"tanh")

A,B,K,Q,nu = plsq1['x']
theta0,theta1,theta2,theta3,theta4,theta5 = plsq2['x']
a,b,c = plsq3['x']
a2,b2,c2 = plsq4['x']
a3,b3,c3,d3 = plsq5['x']

predict1 = (A+(K-A)/(1+Q*np.exp(-B*r))**(1/nu))
predict2 = (theta0 + theta1*r + theta2*r**2 + theta3*r**3 + theta4*r**4 +
                    theta5*r**5)
predict3 = a + b/(1+c*r**2)
predict4 =  a2 + b2*np.exp(-c2*r**2)
predict5 =  a3 + b3*np.tanh(c3*r+d3)

plt.figure()
plt.plot(rtrend_global_median_updraft_norm,'k-o',label='Median Zwind',markersize=6)
plt.plot(predict1,'g-',label='Generalized Logistic',linewidth=3)
plt.plot(predict2,'r-',label='5th Order Polynomial',linewidth=3)
plt.plot(predict3,'b-',label='Inverse Quadratic',linewidth=3)
plt.plot(predict4,'c-',label='Gaussian-Like RBF',linewidth=3)
plt.plot(predict5,'y-',label='tanh',linewidth=3)
plt.legend()
plt.title('Fitting Median Global Trend of Zwind')
plt.xlabel('Normalized Radius(%)')
plt.ylabel('Normalized Zwind(-)')
