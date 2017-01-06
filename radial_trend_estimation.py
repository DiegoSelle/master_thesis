import numpy as np
from skyscan_lib.sim.mesonh_atmosphere import MesoNHAtmosphere
#from mesonh_atm.mesonh_atmosphere import MesoNHAtmosphere
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import RegularGridInterpolator
import scipy.optimize as sciopt
#from cloud import cloud,sample_variogram
import cloud


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

font = {'size'   : 26}

plt.rc('font', **font)

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
lwc_data1=atm.data['RCT'][449:455,75:125,60:200,110:250]
zwind_data1=atm.data['WT'][449:455,75:125,60:200,110:250]
ids1,counter1,clouds1=cloud.cloud_segmentation(lwc_data1)
#ids1,counter1,clouds1=cloud_segmentation(lwc_data1)
clouds1=list(set(clouds1.values()))
length_point_clds = np.ndarray((0,1))
for each_cloud in clouds1:
    print(len(each_cloud.points))
    temp = len(each_cloud.points)
    length_point_clds = np.vstack((length_point_clds,temp))

cloud1 = clouds1[np.argmax(length_point_clds)]

cloud1.calculate_attributes(lwc_data1,zwind_data1)
lwc_cloud1 = np.zeros(lwc_data1.shape)
for point in cloud1.points:
    lwc_cloud1[point] = 1


# Coordinates to interpolate values inside of cloud1
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

output_file = 'results/plots_and_images/polar_analysis/Radial_trend/trends_data/rtrends_cloud1.npz'
np.savez(output_file,rtrend_cloud1=rtrend_cloud1,rtrend_cloud1_updraft_norm=rtrend_cloud1_updraft_norm)

################################################################
########################### cloud2 #############################
################################################################
lwc_data2 = atm.data['RCT'][929:935,75:125,125:250,270:370]
zwind_data2 = atm.data['WT'][929:935,75:125,125:250,270:370]
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
zr = np.arange(0.985,0.985+50*0.01,0.01)
tr = np.arange(929,935)
zspan = np.arange(15,50)

points_span = (tr,zr,xr,yr)
origin_xy = [125,270]

polar_cloud2,polar_cloud2_norm = polar_cloud_norm(points_span,lwc_cloud2,cloud2.COM_2D_lwc_tz,zspan,origin_xy)



zwind_cloud2_polar_norm = atm.get_points(polar_cloud2_norm[:,15:48],'WT','linear')

interpolate_points_cloud = RegularGridInterpolator(points=points_span,values=lwc_cloud2,bounds_error=False,fill_value=0)
lwc_cloud2_polar = interpolate_points_cloud(polar_cloud2_norm[:,15:48],'nearest')
#lwc_cloud1_polar[lwc_cloud1_polar>=0.75]=1
#lwc_cloud1_polar[lwc_cloud1_polar<0.75]=0


########### trivial median Detrending ###############
rtrend_cloud2 = np.ndarray((6,33,360,151))
rtrend_cloud2[:] = np.NAN
for phi in range(0,360):
    trend_temp= zwind_cloud2_polar_norm[:,:,phi,:][lwc_cloud2_polar[:,:,phi,:]==1]
    rtrend_cloud2[:,:,phi,:][lwc_cloud2_polar[:,:,phi,:]==1] = trend_temp

rtrend_cloud2[:,:,:,101:151] = zwind_cloud2_polar_norm[:,:,:,101:151]
rtrend_cloud2_median = np.nanmedian(rtrend_cloud2,axis=(0,1,2))

radii = np.tile(np.arange(0,151),71280)
rtrend_cloud2_wo_nan = rtrend_cloud2.copy()
rtrend_cloud2_wo_nan[np.isnan(rtrend_cloud2_wo_nan)]=-3
prod_shape = np.prod(rtrend_cloud2_wo_nan.shape)/rtrend_cloud2_wo_nan.shape[3]

plt.figure()
plt.hist2d(radii,rtrend_cloud2_wo_nan.reshape(-1),bins=[151,20],vmin = 1e-3,
            weights= 100./(prod_shape)*np.ones_like(rtrend_cloud2_wo_nan.reshape(-1)),
            alpha=0.7)

plt.plot(rtrend_cloud2_median,'k-',label='Median Zwind')
plt.xlabel('Normalized Radius(%)')
plt.ylabel('Zwind(m/s)')
plt.title('Zwind Trend Cloud2')
cbar=plt.colorbar()
cbar.set_label('Frequency(%)')
########### visualizing the trend with updraft normalization ###############

rtrend_cloud2_updraft_norm = rtrend_cloud2.copy()
for t in range(rtrend_cloud2.shape[0]):
    for z in range(rtrend_cloud2.shape[1]):
        temp = rtrend_cloud2[t,z]
        temp = temp/temp[0,0]
        temp[temp>4]=np.NAN
        temp[temp<-2]= np.NAN
        rtrend_cloud2_updraft_norm[t,z]=temp


rtrend_cloud2_median_updraft_norm = np.nanmedian(rtrend_cloud2_updraft_norm,axis=(0,1,2))

rtrend_cloud2_wo_nan_updraft_norm = rtrend_cloud2_updraft_norm.copy()
rtrend_cloud2_wo_nan_updraft_norm[np.isnan(rtrend_cloud2_wo_nan_updraft_norm)]=-3
prod_shape = np.prod(rtrend_cloud2_wo_nan_updraft_norm.shape)/rtrend_cloud2_wo_nan_updraft_norm.shape[3]
radii = np.tile(np.arange(0,151),int(prod_shape))

plt.figure()
plt.hist2d(radii,rtrend_cloud2_wo_nan_updraft_norm.reshape(-1),bins=[151,20],vmin = 1e-3,
            weights= 100./(prod_shape)*np.ones_like(rtrend_cloud2_wo_nan_updraft_norm.reshape(-1)),
            alpha=0.7)

plt.plot(rtrend_cloud2_median_updraft_norm,'k-',label='Median Zwind')
plt.xlabel('Normalized Radius(%)')
plt.ylabel('Normalized Zwind(-)')
plt.title('Zwind Trend Cloud2')
cbar=plt.colorbar()
cbar.set_label('Frequency(%)')

output_file = 'results/plots_and_images/polar_analysis/Radial_trend/trends_data/rtrends_cloud2.npz'
np.savez(output_file,rtrend_cloud2=rtrend_cloud2,rtrend_cloud2_updraft_norm=rtrend_cloud2_updraft_norm)


################################################################
########################### cloud3 #############################
################################################################
lwc_data3 = atm.data['RCT'][1529:1535,75:125,0:115,50:200]
zwind_data3 = atm.data['WT'][1529:1535,75:125,0:115,50:200]
ids3,counter3,clouds3 = cloud.cloud_segmentation(lwc_data3)
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
# Something strange is happening with the last value of xr -> 114 instead of 115
xr =np.arange(0.005 + 0*0.01, 0.005 + 114*0.01,0.01)
yr= np.arange(0.005 + 50*0.01, 0.005 + 200*0.01,0.01)
zr = np.arange(0.985,0.985+50*0.01,0.01)
tr = np.arange(1529,1535)
zspan = np.arange(15,50)

points_span = (tr,zr,xr,yr)
origin_xy = [0,50]

polar_cloud3,polar_cloud3_norm = polar_cloud_norm(points_span,lwc_cloud3,cloud3.COM_2D_lwc_tz,zspan,origin_xy)


zwind_cloud3_polar_norm = atm.get_points(polar_cloud3_norm[:,15:50],'WT','linear')

interpolate_points_cloud = RegularGridInterpolator(points=points_span,values=lwc_cloud3,bounds_error=False,fill_value=0)
lwc_cloud3_polar = interpolate_points_cloud(polar_cloud3_norm[:,15:50],'nearest')
#lwc_cloud1_polar[lwc_cloud1_polar>=0.75]=1
#lwc_cloud1_polar[lwc_cloud1_polar<0.75]=0


########### trivial median Detrending ###############
rtrend_cloud3 = np.ndarray((6,35,360,151))
rtrend_cloud3[:] = np.NAN
for phi in range(0,360):
    trend_temp= zwind_cloud3_polar_norm[:,:,phi,:][lwc_cloud3_polar[:,:,phi,:]==1]
    rtrend_cloud3[:,:,phi,:][lwc_cloud3_polar[:,:,phi,:]==1] = trend_temp

rtrend_cloud3[:,:,:,101:151] = zwind_cloud3_polar_norm[:,:,:,101:151]
rtrend_cloud3_median = np.nanmedian(rtrend_cloud3,axis=(0,1,2))

rtrend_cloud3_wo_nan = rtrend_cloud3.copy()
rtrend_cloud3_wo_nan[np.isnan(rtrend_cloud3_wo_nan)]=-3
prod_shape = np.prod(rtrend_cloud3_wo_nan.shape)/rtrend_cloud3_wo_nan.shape[3]
radii = np.tile(np.arange(0,151),int(prod_shape))

plt.figure()
plt.hist2d(radii,rtrend_cloud3_wo_nan.reshape(-1),bins=[151,20],vmin = 1e-3,
            weights= 100./(prod_shape)*np.ones_like(rtrend_cloud3_wo_nan.reshape(-1)),
            alpha=0.7)

plt.plot(rtrend_cloud3_median,'k-',label='Median Zwind')
plt.xlabel('Normalized Radius(%)')
plt.ylabel('Zwind(m/s)')
plt.title('Zwind Trend Cloud3')
cbar=plt.colorbar()
cbar.set_label('Frequency(%)')

########### visualizing the trend with updraft normalization ###############
rtrend_cloud3_updraft_norm = rtrend_cloud3.copy()
for t in range(rtrend_cloud3.shape[0]):
    for z in range(rtrend_cloud3.shape[1]):
        temp = rtrend_cloud3[t,z]
        temp = temp/temp[0,0]
        temp[temp>3]=np.NAN
        temp[temp<-2]= np.NAN
        rtrend_cloud3_updraft_norm[t,z]=temp


rtrend_cloud3_median_updraft_norm = np.nanmedian(rtrend_cloud3_updraft_norm,axis=(0,1,2))

rtrend_cloud3_wo_nan_updraft_norm = rtrend_cloud3_updraft_norm.copy()
rtrend_cloud3_wo_nan_updraft_norm[np.isnan(rtrend_cloud3_wo_nan_updraft_norm)]=-3
prod_shape = int(np.prod(rtrend_cloud3_wo_nan_updraft_norm.shape)/rtrend_cloud3_wo_nan_updraft_norm.shape[3])
radii = np.tile(np.arange(0,151),prod_shape)

plt.figure()
plt.hist2d(radii,rtrend_cloud3_wo_nan_updraft_norm.reshape(-1),bins=[151,20],vmin = 1e-3,
            weights= 100./(prod_shape)*np.ones_like(rtrend_cloud3_wo_nan_updraft_norm.reshape(-1)),
            alpha=0.7)

plt.plot(rtrend_cloud3_median_updraft_norm,'k-',label='Median Zwind')
plt.xlabel('Normalized Radius(%)')
plt.ylabel('Normalized Zwind(-)')
plt.title('Zwind Trend Cloud3')
cbar=plt.colorbar()
cbar.set_label('Frequency(%)')

output_file = 'results/plots_and_images/polar_analysis/Radial_trend/trends_data/rtrends_cloud3.npz'
np.savez(output_file,rtrend_cloud3=rtrend_cloud3,rtrend_cloud3_updraft_norm=rtrend_cloud3_updraft_norm)
################################################################
########################### cloud4 #############################
################################################################
lwc_data4 = atm.data['RCT'][2789:2795,75:125,310:400,0:125]
zwind_data4 = atm.data['WT'][2789:2795,75:125,310:400,0:125]
ids4,counter4,clouds4 = cloud.cloud_segmentation(lwc_data4)
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

xr =np.arange(0.005 + 310*0.01, 0.005 + 400*0.01,0.01)
yr= np.arange(0.005 + 0*0.01, 0.005 + 125*0.01,0.01)
zr = np.arange(0.985,0.985+50*0.01,0.01)
tr = np.arange(2789,2795)
zspan = np.arange(15,35)

points_span = (tr,zr,xr,yr)
origin_xy = [310,0]

polar_cloud4,polar_cloud4_norm = polar_cloud_norm(points_span,lwc_cloud4,cloud4.COM_2D_lwc_tz,zspan,origin_xy)


zwind_cloud4_polar_norm = atm.get_points(polar_cloud4_norm[:,15:35],'WT','linear')

interpolate_points_cloud = RegularGridInterpolator(points=points_span,values=lwc_cloud4,bounds_error=False,fill_value=0)
lwc_cloud4_polar = interpolate_points_cloud(polar_cloud4_norm[:,15:35],'nearest')
#lwc_cloud1_polar[lwc_cloud1_polar>=0.75]=1
#lwc_cloud1_polar[lwc_cloud1_polar<0.75]=0


########### trivial median Detrending ###############
rtrend_cloud4 = np.ndarray((6,20,360,151))
rtrend_cloud4[:] = np.NAN
for phi in range(0,360):
    trend_temp= zwind_cloud4_polar_norm[:,:,phi,:][lwc_cloud4_polar[:,:,phi,:]==1]
    rtrend_cloud4[:,:,phi,:][lwc_cloud4_polar[:,:,phi,:]==1] = trend_temp

rtrend_cloud4[:,:,:,101:151] = zwind_cloud4_polar_norm[:,:,:,101:151]
rtrend_cloud4_median = np.nanmedian(rtrend_cloud4,axis=(0,1,2))

rtrend_cloud4_wo_nan = rtrend_cloud4.copy()
rtrend_cloud4_wo_nan[np.isnan(rtrend_cloud4_wo_nan)]=-3
prod_shape = np.prod(rtrend_cloud4_wo_nan.shape)/rtrend_cloud4_wo_nan.shape[3]
radii = np.tile(np.arange(0,151),prod_shape)

plt.figure()
plt.hist2d(radii,rtrend_cloud4_wo_nan.reshape(-1),bins=[151,20],vmin = 1e-3,
            weights= 100./(prod_shape)*np.ones_like(rtrend_cloud4_wo_nan.reshape(-1)),
            alpha=0.7)

plt.plot(rtrend_cloud4_median,'k-',label='Median Zwind')
plt.xlabel('Normalized Radius(%)')
plt.ylabel('Zwind(m/s)')
plt.title('Zwind Trend Cloud4')
cbar=plt.colorbar()
cbar.set_label('Frequency(%)')

########### visualizing the trend with updraft normalization ###############

rtrend_cloud4_updraft_norm = rtrend_cloud4.copy()
for t in range(rtrend_cloud4.shape[0]):
    for z in range(rtrend_cloud4.shape[1]):
        temp = rtrend_cloud4[t,z]
        temp = temp/temp[0,0]
        temp[temp>4]=np.NAN
        temp[temp<-2]= np.NAN
        rtrend_cloud4_updraft_norm[t,z]=temp


rtrend_cloud4_median_updraft_norm = np.nanmedian(rtrend_cloud4_updraft_norm,axis=(0,1,2))

rtrend_cloud4_wo_nan_updraft_norm = rtrend_cloud4_updraft_norm.copy()
rtrend_cloud4_wo_nan_updraft_norm[np.isnan(rtrend_cloud4_wo_nan_updraft_norm)]=-3
prod_shape = int(np.prod(rtrend_cloud4_wo_nan_updraft_norm.shape)/rtrend_cloud4_wo_nan_updraft_norm.shape[3])
radii = np.tile(np.arange(0,151),prod_shape)

plt.figure()
plt.hist2d(radii,rtrend_cloud4_wo_nan_updraft_norm.reshape(-1),bins=[151,20],vmin = 1e-3,
            weights= 100./(prod_shape)*np.ones_like(rtrend_cloud4_wo_nan_updraft_norm.reshape(-1)),
            alpha=0.7)

plt.plot(rtrend_cloud4_median_updraft_norm,'k-',label='Median Zwind')
plt.xlabel('Normalized Radius(%)')
plt.ylabel('Normalized Zwind(-)')
plt.title('Zwind Trend Cloud4')
cbar=plt.colorbar()
cbar.set_label('Frequency(%)')

output_file = 'results/plots_and_images/polar_analysis/Radial_trend/trends_data/rtrends_cloud4.npz'
np.savez(output_file,rtrend_cloud4=rtrend_cloud4,rtrend_cloud4_updraft_norm=rtrend_cloud4_updraft_norm)

################################################################
########################### cloud5 #############################
################################################################

lwc_data5 = atm.data['RCT'][3389:3395,75:125,0:200,200:400]
zwind_data5 = atm.data['WT'][3389:3395,75:125,0:200,200:400]
ids5,counter5,clouds5 = cloud.cloud_segmentation(lwc_data5)
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

xr =np.arange(0.005 + 0*0.01, 0.005 + 200*0.01,0.01)
yr= np.arange(0.005 + 200*0.01, 0.005 + 400*0.01,0.01)
zr = np.arange(0.985,0.985+50*0.01,0.01)
tr = np.arange(3389,3395)
zspan = np.arange(18,46)

points_span = (tr,zr,xr,yr)
origin_xy = [0,200]

polar_cloud5,polar_cloud5_norm = polar_cloud_norm(points_span,lwc_cloud5,cloud5.COM_2D_lwc_tz,zspan,origin_xy)

zwind_cloud5_polar_norm = atm.get_points(polar_cloud5_norm[:,18:46],'WT','linear')

interpolate_points_cloud = RegularGridInterpolator(points=points_span,values=lwc_cloud5,bounds_error=False,fill_value=0)
lwc_cloud5_polar = interpolate_points_cloud(polar_cloud5_norm[:,18:46],'nearest')
#lwc_cloud1_polar[lwc_cloud1_polar>=0.75]=1
#lwc_cloud1_polar[lwc_cloud1_polar<0.75]=0


########### visualizing the trend ###############
rtrend_cloud5 = np.ndarray((6,28,360,151))
rtrend_cloud5[:] = np.NAN
for phi in range(0,360):
    trend_temp= zwind_cloud5_polar_norm[:,:,phi,:][lwc_cloud5_polar[:,:,phi,:]==1]
    rtrend_cloud5[:,:,phi,:][lwc_cloud5_polar[:,:,phi,:]==1] = trend_temp

rtrend_cloud5[:,:,:,101:151] = zwind_cloud5_polar_norm[:,:,:,101:151]

rtrend_cloud5_median = np.nanmedian(rtrend_cloud5,axis=(0,1,2))

rtrend_cloud5_wo_nan = rtrend_cloud5.copy()
rtrend_cloud5_wo_nan[np.isnan(rtrend_cloud5_wo_nan)]=-3
prod_shape = np.prod(rtrend_cloud5_wo_nan.shape)/rtrend_cloud5_wo_nan.shape[3]
radii = np.tile(np.arange(0,151),prod_shape)

plt.figure()
plt.hist2d(radii,rtrend_cloud5_wo_nan.reshape(-1),bins=[151,20],vmin = 1e-3,
            weights= 100./(prod_shape)*np.ones_like(rtrend_cloud5_wo_nan.reshape(-1)),
            alpha=0.7)

plt.plot(rtrend_cloud5_median,'k-',label='Median Zwind')
plt.xlabel('Normalized Radius(%)')
plt.ylabel('Zwind(m/s)')
plt.title('Zwind Trend Cloud5')
cbar=plt.colorbar()
cbar.set_label('Frequency(%)')

########### visualizing the trend with updraft normalization ###############

rtrend_cloud5_updraft_norm = rtrend_cloud5.copy()
for t in range(rtrend_cloud5.shape[0]):
    for z in range(rtrend_cloud5.shape[1]):
        temp = rtrend_cloud5[t,z]
        temp = temp/temp[0,0]
        temp[temp>4]=np.NAN
        temp[temp<-2]= np.NAN
        rtrend_cloud5_updraft_norm[t,z]=temp


rtrend_cloud5_median_updraft_norm = np.nanmedian(rtrend_cloud5_updraft_norm,axis=(0,1,2))

rtrend_cloud5_wo_nan_updraft_norm = rtrend_cloud5_updraft_norm.copy()
rtrend_cloud5_wo_nan_updraft_norm[np.isnan(rtrend_cloud5_wo_nan_updraft_norm)]=-3
prod_shape = np.prod(rtrend_cloud5_wo_nan_updraft_norm.shape)/rtrend_cloud5_wo_nan_updraft_norm.shape[3]
radii = np.tile(np.arange(0,151),prod_shape)

plt.figure()
plt.hist2d(radii,rtrend_cloud5_wo_nan_updraft_norm.reshape(-1),bins=[151,20],vmin = 1e-3,
            weights= 100./(prod_shape)*np.ones_like(rtrend_cloud5_wo_nan_updraft_norm.reshape(-1)),
            alpha=0.7)

plt.plot(rtrend_cloud5_median_updraft_norm,'k-',label='Median Zwind')
plt.xlabel('Normalized Radius(%)')
plt.ylabel('Normalized Zwind(-)')
plt.title('Zwind Trend Cloud5')
cbar=plt.colorbar()
cbar.set_label('Frequency(%)')

output_file = 'results/plots_and_images/polar_analysis/Radial_trend/trends_data/rtrends_cloud5.npz'
np.savez(output_file,rtrend_cloud5=rtrend_cloud5,rtrend_cloud5_updraft_norm=rtrend_cloud5_updraft_norm)

##################################################################
########### Summary all clouds and fitting trend #################
##################################################################

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

def fit_radial_trend(rtrend_global_median,r,model):
    if model == 'generalized_logistic':
        # params = [A,B,K,Q,nu]
        p0 = [1,0.01,-0.2,1,0.5]
        def residuals(p,rtrend_global_median,r):
            A,B,K,Q,nu = p
            err = (A+(K-A)/(1+Q*np.exp(-B*r))**(1/nu))-rtrend_global_median
            return err
        p_wlsq = sciopt.least_squares(residuals, p0, args=(rtrend_global_median,r))
        return p_wlsq
    if model == 'polynomial':
        # regularized linear regression with fifth order polynomial
        p0 = [1,0,0,0,0,0]
        def residuals(p,rtrend_global_median,r):
            theta0,theta1,theta2,theta3,theta4,theta5 = p
            err = (theta0 + theta1*r + theta2*r**2 + theta3*r**3 + theta4*r**4 +
                    theta5*r**5) - rtrend_global_median
            return err
        p_wlsq = sciopt.least_squares(residuals, p0, args=(rtrend_global_median,r))
        return p_wlsq
    if model == 'inverse_quadratic':
        # params
        p0 = [-0.1,1.1,0.001]
        def residuals(p,rtrend_global_median,r):
            a,b,c = p
            err = a + b/(1+c*r**2) - rtrend_global_median
            return err
        p_wlsq = sciopt.least_squares(residuals, p0, args=(rtrend_global_median,r))
        return p_wlsq
    if model == 'rbf':
        # Gaussian-like Radial Basis Function
        p0 = [-0.1,1,0.01]
        def residuals(p,rtrend_global_median,r):
            a,b,c = p
            err = a + b*np.exp(-c*r**2) - rtrend_global_median
            return err
        p_wlsq = sciopt.least_squares(residuals, p0, args=(rtrend_global_median,r))
        return p_wlsq
    if model == 'tanh':
        # Tangens hyperbolicus logistic-like function
        p0 = [0.5,-0.6,0.05,-4]
        def residuals(p,rtrend_global_median,r):
            a,b,c,d = p
            err = a + b*np.tanh(c*r+d) - rtrend_global_median
            return err
        p_wlsq = sciopt.least_squares(residuals, p0, args=(rtrend_global_median,r))
        return p_wlsq

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
