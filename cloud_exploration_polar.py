import numpy as np
from mesonh_atm.mesonh_atmosphere import MesoNHAtmosphere
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import modules.cloud as ModCloud



#Data without advection
path = "/net/skyscanner/volume1/data/mesoNH/ARM_OneHour3600files_No_Horizontal_Wind/"
mfiles = [path+"U0K10.1.min{:02d}.{:03d}_diaKCL.nc".format(minute, second)
          for minute in range(1, 60)
          for second in range(1, 61)]
mtstep = 1
atm = MesoNHAtmosphere(mfiles, 1)

#The height of each cross-section of the simulation
all_Zs=atm.data["VLEV"][:,0,0]

#picking the LWC variable, calculating density of points with LWC>=0.00001 kg/kg with regards to height,
#i.e. that are part of clouds, for t=400

indices=atm.data["RCT"][400]
lwc_densities_z=np.array([])
for z_index in range(indices.shape[0]):
    lwc_plane=atm.data["RCT"][400,z_index]
    count_lwc=np.count_nonzero(lwc_plane>=1e-5)
    lwc_density=count_lwc/np.prod(lwc_plane.shape)
    lwc_densities_z=np.append(lwc_densities_z,lwc_density)

plt.figure()
plt.plot(lwc_densities_z)
plt.title('Cloud Density Per Cross-section, whole domain,t=400s')
plt.xlabel('Z Index')
plt.ylabel('Proportion of Points that are cloud')

plt.figure()
plt.plot(np.diff(atm.data["VLEV"][:,0,0]))
plt.title('Resolution of Height at given Z index')
plt.xlabel('Z-index')
plt.ylabel('Resolution(km)')

#######################################################################
########## Cloud exploration example, polar coordinates ###############
#######################################################################

# RCT is the name of the 'Liquid Water Content' Variable. WT is the
# name of the vertical wind component.
lwc_data=atm.data['RCT'][449:455,85:123,60:200,110:250]
zwind_data=atm.data['WT'][449:455,85:123,60:200,110:250]
ids,counter,clouds=ModCloud.cloud_segmentation(lwc_data)

# Get the cloud with the biggest amount of points inside the rough bounding box
clouds=list(set(clouds.values()))
length_point_clds = np.ndarray((0,1))
for each_cloud in clouds:
    print(len(each_cloud.points))
    temp = len(each_cloud.points)
    length_point_clds = np.vstack((length_point_clds,temp))

cloud = clouds[np.argmax(length_point_clds)]
cloud.calculate_attributes(lwc_data,zwind_data)

# Creating binarized cloud geometry with initial structure of lwc_data,
# which contains all clouds
lwc_cloud = np.zeros(lwc_data.shape)
for point in cloud.points:
    lwc_cloud[point] = 1

#Example coordinates of bounding box of a cloud
xr =np.arange(0.005 + 60*0.01, 0.005 + 200*0.01,0.01)
yr= np.arange(0.005 + 110*0.01, 0.005 + 250*0.01,0.01)
zr = all_Zs[85:123]
tr = np.arange(449,455)
zspan = np.arange(0,38)
points_span = (tr,zr,xr,yr)
origin_xy = [60,110]

########## Coordinates Transformation ########
polar_cloud,polar_cloud_norm = ModCloud.polar_cloud_norm(points_span,lwc_cloud,cloud.COM_2D_lwc_tz,zspan,origin_xy)


# Function to interpolate points of cloud
interpolate_points_cloud = RegularGridInterpolator(points=(tr,zr,xr,yr),values=lwc_cloud,bounds_error=False,fill_value=0)

zwind_cloud_polar_norm = atm.get_points(polar_cloud_norm,'WT','linear')
zwind_cloud_polar = atm.get_points(polar_cloud,'WT','linear')

lwc_cloud_polar_norm = interpolate_points_cloud(polar_cloud_norm,"nearest")
lwc_cloud_polar= interpolate_points_cloud(polar_cloud,"nearest")

#################################################################################
##### Visualizing a cross-section in polar coordinates, normalized and not ######
#################################################################################

plt.figure()
plt.title("Zwind Cloud,z_index=90, relative t=0")
plt.imshow(zwind_cloud_polar[0,5].T,origin='lower')
plt.xlabel('phi')
plt.ylabel('radius')
cbar=plt.colorbar()
cbar.set_label('m/s')


plt.figure()
plt.title("Zwind Cloud,z_index=90, relative t=0")
plt.imshow(zwind_cloud_polar_norm[0,5].T,origin='lower')
plt.xlabel('phi')
plt.ylabel('radius(%)')
cbar=plt.colorbar()
cbar.set_label('m/s')


plt.figure()
plt.title("Zwind Cloud,z_index=90, relative t=0")
plt.contour(zwind_cloud_polar[0,5].T,origin='lower')
plt.contour(lwc_cloud_polar[0,5].T,origin='lower',linestyles='dashed',alpha=0.3)
plt.xlabel('phi')
plt.ylabel('radius')
cbar=plt.colorbar()
cbar.set_label('m/s')


plt.figure()
plt.title("LWC Cloud,z_index=90, relative t=0")
plt.xlabel('phi')
plt.ylabel('radius')
plt.imshow(lwc_cloud_polar_norm[0,5].T,origin='lower')
cbar=plt.colorbar()
cbar.set_label('kg/kg')

# Visualizing radial trend of wind vertical component
plt.figure()
plt.title("Zwind Cloud,z_index=90, relative t=0")
plt.xlabel('r in 10m')
plt.ylabel('Zwind in m/s')
for phi in range(0,360,20):
    if phi<120:
        plt.plot(zwind_cloud_polar_norm[0,5,phi,:][zwind_cloud_polar_norm[0,0,phi,:]>0.5],'-o',label='phi={}'.format(phi))
    if phi>=120 and phi <240:
        plt.plot(zwind_cloud_polar_norm[0,5,phi,:][zwind_cloud_polar_norm[0,0,phi,:]>0.5],'-x',label='phi={}'.format(phi))
    if phi >=240:
        plt.plot(zwind_cloud_polar_norm[0,0,phi,:][zwind_cloud_polar_norm[0,0,phi,:]>0.5],'-+',label='phi={}'.format(phi))
plt.legend()
