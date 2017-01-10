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
# Example Data of two variables with the coordinates of a rough bounding box of a cloud
# RCT = liquid water content, WT = vertical wind
lwc_data=atm.data['RCT'][449:455,75:125,60:200,110:250]
zwind_data=atm.data['WT'][449:455,75:125,60:200,110:250]
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


# Coordinates to interpolate values inside of cloud, i.e. rough bounding box
xr =np.arange(0.005 + 60*0.01, 0.005 + 200*0.01,0.01)
yr= np.arange(0.005 + 110*0.01, 0.005 + 250*0.01,0.01)
zr = np.arange(0.985,0.985+50*0.01,0.01)
tr = np.arange(449,455)
zspan = np.arange(10,50)


points_span = (tr,zr,xr,yr)
origin_xy = [60,110]
polar_cloud,polar_cloud_norm = ModCloud.polar_cloud_norm(points_span,lwc_cloud,cloud.COM_2D_lwc_tz,zspan,origin_xy)

zwind_cloud_polar_norm = atm.get_points(polar_cloud_norm[:,10:48],'WT','linear')


interpolate_points_cloud = RegularGridInterpolator(points=points_span,values=lwc_cloud,bounds_error=False,fill_value=0)
lwc_cloud_polar = interpolate_points_cloud(polar_cloud_norm[:,10:48],'nearest')


plt.figure()
plt.contour(zwind_cloud_polar_norm[0,15].T,origin='lower')
cbar=plt.colorbar()
cbar.set_label('m/s')
plt.contour(lwc_cloud_polar[0,15].T,origin='lower',alpha=0.6,cmap='Greys')
plt.xlabel(r'$\varphi(Degrees)$')
plt.ylabel('Normalized Radius(%)')
plt.title('Zwind(m/s) normalized Cloud, z={}km'.format(np.round(float(zr[15]),3)))


plt.figure()
plt.imshow(lwc_cloud_polar[0,15].T,origin='lower')
plt.xlabel('Phi(Degrees)')
plt.ylabel('Normalized Radius(%)')
cbar=plt.colorbar()
cbar.set_label('In-out of Cloud')
plt.title('Cloud Normalized')

########### Calculation Median trend ###############
rtrend_cloud = np.ndarray((6,38,360,151))
rtrend_cloud[:] = np.NAN
for phi in range(0,360):
    trend_temp= zwind_cloud_polar_norm[:,:,phi,:][lwc_cloud_polar[:,:,phi,:]==1]
    rtrend_cloud[:,:,phi,:][lwc_cloud_polar[:,:,phi,:]==1] = trend_temp

rtrend_cloud[:,:,:,101:151] = zwind_cloud_polar_norm[:,:,:,101:151]
rtrend_cloud_median = np.nanmedian(rtrend_cloud,axis=(0,1,2))

radii = np.tile(np.arange(0,151),82080)
rtrend_cloud_wo_nan = rtrend_cloud.copy()
rtrend_cloud_wo_nan[np.isnan(rtrend_cloud_wo_nan)]=-3
prod_shape = np.prod(rtrend_cloud_wo_nan.shape)/rtrend_cloud_wo_nan.shape[3]

plt.figure()
plt.hist2d(radii,rtrend_cloud_wo_nan.reshape(-1),bins=[151,20],vmin = 1e-3,
            weights= 100./(prod_shape)*np.ones_like(rtrend_cloud_wo_nan.reshape(-1)),
            alpha=0.7)

plt.plot(rtrend_cloud_median,'k-',label='Median Zwind')
plt.xlabel('Normalized Radius(%)')
plt.ylabel('Zwind(m/s)')
plt.title('Updraft Trend Cloud')
cbar=plt.colorbar()
cbar.set_label('Frequency(%)')

########### visualizing the trend with updraft normalization ###############

rtrend_cloud_updraft_norm = rtrend_cloud.copy()
for t in range(rtrend_cloud.shape[0]):
    for z in range(rtrend_cloud.shape[1]):
        temp = rtrend_cloud[t,z]
        temp = temp/temp[0,0]
        temp[temp>4]=np.NAN
        temp[temp<-2]= np.NAN
        rtrend_cloud_updraft_norm[t,z]=temp


rtrend_cloud_median_updraft_norm = np.nanmedian(rtrend_cloud_updraft_norm,axis=(0,1,2))

rtrend_cloud_wo_nan_updraft_norm = rtrend_cloud_updraft_norm.copy()
rtrend_cloud_wo_nan_updraft_norm[np.isnan(rtrend_cloud_wo_nan_updraft_norm)]=-3
prod_shape = np.prod(rtrend_cloud_wo_nan_updraft_norm.shape)/rtrend_cloud_wo_nan_updraft_norm.shape[3]
radii = np.tile(np.arange(0,151),int(prod_shape))

plt.figure()
plt.hist2d(radii,rtrend_cloud_wo_nan_updraft_norm.reshape(-1),bins=[151,20],vmin = 1e-3,
            weights= 100./(prod_shape)*np.ones_like(rtrend_cloud_wo_nan_updraft_norm.reshape(-1)),
            alpha=0.7)

plt.plot(rtrend_cloud_median_updraft_norm,'k-',label='Median Zwind')
plt.xlabel('Normalized Radius(%)')
plt.ylabel('Normalized Zwind(-)')
plt.title('Zwind Trend Cloud')
cbar=plt.colorbar()
cbar.set_label('Frequency(%)')

output_file = 'results/plots_and_images/polar_analysis/Radial_trend/trends_data/rtrends_cloud.npz'
np.savez(output_file,rtrend_cloud=rtrend_cloud,rtrend_cloud_updraft_norm=rtrend_cloud_updraft_norm)


###########################################
########### fitting trend #################
###########################################

# Example to fit analytical models to the median radial trends obtained from the example cloud
# This analysis can be easily repeated with more clouds by concatenating the trends of several clouds
output_file = 'results/plots_and_images/polar_analysis/Radial_trend/trends_data/rtrends_cloud.npz'

files_cloud = np.load(output_file)

rtrend_cloud_updraft_norm = files_cloud['rtrend_cloud_updraft_norm']

rtrend_global_updraft_norm = rtrend_cloud_updraft_norm

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

# Saving global trend
outfile = 'results/plots_and_images/polar_analysis/Radial_trend/trends_data/rtrend_global_median_updraft_norm.npy'
np.save(outfile,rtrend_global_median_updraft_norm)


### Curve fitting optimization
r = np.arange(0,151)

plsq1 = ModCloud.fit_radial_trend(rtrend_global_median_updraft_norm,r,"generalized_logistic")
plsq2 = ModCloud.fit_radial_trend(rtrend_global_median_updraft_norm,r,"polynomial")
plsq3 = ModCloud.fit_radial_trend(rtrend_global_median_updraft_norm,r,"inverse_quadratic")
plsq4 = ModCloud.fit_radial_trend(rtrend_global_median_updraft_norm,r,"rbf")
plsq5 = ModCloud.fit_radial_trend(rtrend_global_median_updraft_norm,r,"tanh")

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

# visualizing fit
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
