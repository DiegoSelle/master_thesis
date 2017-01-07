import numpy as np
from mesonh_atm.mesonh_atmosphere import MesoNHAtmosphere
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
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


#######################################################################
########################### cloud example #############################
#######################################################################


lwc_data=atm.data['RCT'][449:599,75:125,60:200,110:250]
zwind_data=atm.data['WT'][449:599,75:125,60:200,110:250]
ids,counter,clouds=ModCloud.cloud_segmentation(lwc_data)

clouds=list(set(clouds.values()))
length_point_clds = np.ndarray((0,1))
for each_cloud in clouds:
    print(len(each_cloud.points))
    temp = len(each_cloud.points)
    length_point_clds = np.vstack((length_point_clds,temp))

# Get cloud with the biggest amount of points in the bounding box
cloud = clouds[np.argmax(length_point_clds)]
cloud.calculate_attributes(lwc_data,zwind_data)

lwc_cloud = np.zeros(lwc_data.shape)
for point in cloud.points:
    lwc_cloud[point] = 1

#Coordinates of the rough bounding box of the example cloud
xr = np.arange(0.005 + 60*0.01, 0.005 + 200*0.01,0.01)
yr = np.arange(0.005 + 110*0.01, 0.005 + 250*0.01,0.01)
all_Zs = atm.data["VLEV"][:,0,0]
zr = all_Zs[75:125]
tr = np.arange(449,599)
origin_xy = [60,110]
zspan = np.arange(0,16)

# Plotting three different cross-sections including the center of geometry COG and the center of masses
# of the vertical wind and liquid water content
plt.figure()
plt.xlabel("x coordinate(km)")
plt.ylabel("y coordinate(km)")
plt.contour(zwind_data[0,15].T,origin="lower",label='zwind',extent=[xr[0], xr[-1], yr[0], yr[-1]],linewidths=2)
cbar=plt.colorbar()
cbar.set_label('m/s')
plt.contour(lwc_cloud[0,15].T,V=[0,1],origin='lower',extent=[xr[0], xr[-1], yr[0], yr[-1]],alpha=0.6,cmap='Greys')

COG_2D = cloud.COG_2D_tz[0,15]*0.01 + np.array([0.005 + origin_xy[0]*0.01,0.005 + origin_xy[1]*0.01])
plt.plot(COG_2D[0],COG_2D[1],'ro',markersize=8,label='COG 2D')

COM_2D_zwind = cloud.COM_2D_zwind_tz[0,15]*0.01 + np.array([0.005 + origin_xy[0]*0.01,0.005 + origin_xy[1]*0.01])
plt.plot(COM_2D_zwind[0],COM_2D_zwind[1],'gx',markersize=8, label='COM 2D zwind')

COM_2D_lwc = cloud.COM_2D_lwc_tz[0,15]*0.01 + np.array([0.005 + origin_xy[0]*0.01,0.005 + origin_xy[1]*0.01])
plt.plot(COM_2D_lwc[0],COM_2D_lwc[1],'b>',markersize=8, label='COM 2D lwc')

plt.title("Zwind Cross-section Cloud Example, z={}km, t={}s".format(np.round(float(zr[15]),3),tr[0]))
plt.xlim(xr[0], xr[-1])
plt.ylim(yr[0], yr[-1])
plt.legend()


plt.figure()
plt.xlabel("x coordinate(km)")
plt.ylabel("y coordinate(km)")
plt.contour(zwind_data[0,19].T,origin="lower",label='zwind',extent=[xr[0], xr[-1], yr[0], yr[-1]],linewidths=2)
cbar=plt.colorbar()
cbar.set_label('m/s')
plt.contour(lwc_cloud[0,19].T,V=[0,1],origin='lower',extent=[xr[0], xr[-1], yr[0], yr[-1]],alpha=0.6,cmap='Greys')

COG_2D = cloud.COG_2D_tz[0,19]*0.01 + np.array([0.005 + origin_xy[0]*0.01,0.005 + origin_xy[1]*0.01])
plt.plot(COG_2D[0],COG_2D[1],'ro',markersize=8,label='COG 2D')

COM_2D_zwind = cloud.COM_2D_zwind_tz[0,19]*0.01 + np.array([0.005 + origin_xy[0]*0.01,0.005 + origin_xy[1]*0.01])
plt.plot(COM_2D_zwind[0],COM_2D_zwind[1],'gx',markersize=8, label='COM 2D zwind')

COM_2D_lwc = cloud.COM_2D_lwc_tz[0,19]*0.01 + np.array([0.005 + origin_xy[0]*0.01,0.005 + origin_xy[1]*0.01])
plt.plot(COM_2D_lwc[0],COM_2D_lwc[1],'b>',markersize=8, label='COM 2D lwc')

plt.title("Zwind Cross-section Cloud Example, z={}km, t={}s".format(np.round(float(zr[19]),3),tr[0]))
plt.xlim(xr[0], xr[-1])
plt.ylim(yr[0], yr[-1])
plt.legend()


plt.figure()
plt.xlabel("x coordinate(km)")
plt.ylabel("y coordinate(km)")
plt.contour(zwind_data[0,30].T,origin="lower",label='zwind',extent=[xr[0], xr[-1], yr[0], yr[-1]],linewidths=2)
cbar=plt.colorbar()
cbar.set_label('m/s')
plt.contour(lwc_cloud[0,30].T,V=[0,1],origin='lower',extent=[xr[0], xr[-1], yr[0], yr[-1]],alpha=0.6,cmap='Greys')

COG_2D = cloud.COG_2D_tz[0,30]*0.01 + np.array([0.005 + origin_xy[0]*0.01,0.005 + origin_xy[1]*0.01])
plt.plot(COG_2D[0],COG_2D[1],'ro',markersize=8,label='COG 2D')

COM_2D_zwind = cloud.COM_2D_zwind_tz[0,30]*0.01 + np.array([0.005 + origin_xy[0]*0.01,0.005 + origin_xy[1]*0.01])
plt.plot(COM_2D_zwind[0],COM_2D_zwind[1],'gx',markersize=8, label='COM 2D zwind')

COM_2D_lwc = cloud.COM_2D_lwc_tz[0,30]*0.01 + np.array([0.005 + origin_xy[0]*0.01,0.005 + origin_xy[1]*0.01])
plt.plot(COM_2D_lwc[0],COM_2D_lwc[1],'b>',markersize=8, label='COM 2D lwc')

plt.title("Zwind Cross-section Cloud, z={}km, t={}s".format(np.round(float(zr[30]),3),tr[0]))
plt.xlim(xr[0], xr[-1])
plt.ylim(yr[0], yr[-1])
plt.legend()

# Center of masses and Geometry, for each cross-section
plt.figure()
plt.xlabel("z coordinate(km)")
plt.ylabel("y coordinate(km)")
plt.plot(zr,cloud.COG_2D_tz[0,:,1]*0.01 + 0.005 + origin_xy[1]*0.01,label='COG 2D',linewidth=3)
plt.plot(zr,cloud.COM_2D_lwc_tz[0,:,1]*0.01 + 0.005 + origin_xy[1]*0.01, label='COM 2D lwc',linewidth=3)
plt.plot(zr,cloud.COM_2D_zwind_tz[0,:,1]*0.01 + 0.005 + origin_xy[1]*0.01, label='COM 2D zwind',linewidth=3)
plt.legend()
plt.title('Center of masses and geometry Cloud, t = {}s'.format(tr[0]))

plt.figure()
plt.xlabel("z coordinate(km)")
plt.ylabel("x coordinate(km)")
plt.plot(zr,cloud.COG_2D_tz[0,:,0]*0.01 + 0.005 + origin_xy[1]*0.01,label='COG 2D',linewidth=3)
plt.plot(zr,cloud.COM_2D_lwc_tz[0,:,0]*0.01 + 0.005 + origin_xy[1]*0.01, label='COM 2D lwc',linewidth=3)
plt.plot(zr,cloud.COM_2D_zwind_tz[0,:,0]*0.01 + 0.005 + origin_xy[1]*0.01, label='COM 2D zwind',linewidth=3)
plt.legend()
plt.title('Center of masses and geometry Cloud, t = {}s'.format(tr[0]))

plt.figure()
plt.xlabel("z coordinate(km)")
plt.ylabel("Surface(100$m^2$)")
plt.plot(zr,cloud.area_cs_tz[0],linewidth=3)
plt.title('Surface Area of Cloud, t={}s'.format(tr[0]))


plt.figure()
plt.xlabel("time(s)")
plt.ylabel("Volume(1000 $m^3$)")
plt.plot(tr,cloud.volumen_t,linewidth=3)
plt.title('Volume of Cloud')

####### Visualizing max vertical wind as a function of z
zwind_maxz = np.ndarray((0,1))
for z in range(int(cloud.zmin_t[0]),int(cloud.zmax_t[0])+1):
    zwind_max = np.max(zwind_data[0,z][lwc_cloud[0,z]>0])
    zwind_maxz = np.vstack((zwind_maxz,zwind_max))

####### Visualizing mean vertical wind as a function of z
zwind_meanz = np.ndarray((0,1))
for z in range(int(cloud.zmin_t[0]),int(cloud.zmax_t[0])+1):
    zwind_mean = np.mean(zwind_data[0,z][lwc_cloud[0,z]>0])
    zwind_meanz = np.vstack((zwind_meanz,zwind_mean))


plt.figure()
plt.xlabel("z coordinate(km)")
plt.ylabel("Max zwind(m/s)")
plt.plot(zr[4:],zwind_maxz,linewidth=3)
plt.title('Max Zwind per z cross-section Cloud,t={}s'.format(tr[0]))

plt.figure()
plt.xlabel("z coordinate(km)")
plt.ylabel("Mean Zwind (m/s)")
plt.plot(zr[4:],zwind_meanz,linewidth=3)
plt.title('Mean Zwind per z cross-section Cloud,t={}s'.format(tr[0]))
################# Variance behaviour of vertical wind in dependence of z
zwind_varz = np.ndarray((0,1))
for z in range(int(cloud.zmin_t[0]),int(cloud.zmax_t[0])+1):
    zwind_var = zwind_data[0,z][lwc_cloud[0,z]>0].var()
    zwind_varz = np.vstack((zwind_varz,zwind_var))

plt.figure()
plt.xlabel("z coordinate(km)")
plt.ylabel("Variance Zwind")
plt.plot(zr[4:],zwind_varz,linewidth=3)
plt.title('Mean Zwind per z cross-section Cloud,t={}s'.format(tr[0]))


##########################################
############# Variogram Analysis #########
##########################################

##############################################################
##### creating moving bounding box that follows center #######
##############################################################
xbound_max = int(np.max(cloud.xsize_t))
ybound_max = int(np.max(cloud.ysize_t))
zbound_max = int(np.max(cloud.zsize_t))
xcenter_t = (cloud.xmin_t + cloud.xmax_t)/2
ycenter_t = (cloud.ymin_t + cloud.ymax_t)/2
zcenter_t = (cloud.zmin_t + cloud.zmax_t)/2

zwind_hypercube = np.ndarray((cloud.tmax+1-cloud.tmin,zbound_max,xbound_max,ybound_max))
cloud_hypercube = np.ndarray((cloud.tmax+1-cloud.tmin,zbound_max,xbound_max,ybound_max))
total_size = lwc_cloud.shape

### Make this an attribute maybe ?
for time in range(cloud.tmin,cloud.tmax+1):
    xmin = int(np.ceil(xcenter_t[time] - xbound_max/2))
    xmax = int(np.ceil(xcenter_t[time] + xbound_max/2))
    ymin = int(np.ceil(ycenter_t[time] - ybound_max/2))
    ymax = int(np.ceil(ycenter_t[time] + ybound_max/2))
    zmin = int(np.ceil(zcenter_t[time] - zbound_max/2))
    zmax = int(np.ceil(zcenter_t[time] + zbound_max/2))

    if xmin < 0:
        xmax = xmax - xmin
        xmin = 0
    if ymin < 0:
        ymax = ymax - ymin
        ymin = 0
    if zmin < 0:
        zmax = zmax - zmin
        zmin = 0

    if xmax > total_size[2]:
        xmin = xmin - (xmax-total_size[2])
        xmax = total_size[2]
    if ymax > total_size[3]:
        ymin = ymin - (ymax-total_size[3])
        ymax = total_size[3]
    if zmax > total_size[1]:
        zmin = zmin - (zmax-total_size[1])
        zmax = total_size[1]

    zwind_hypercube[time] = zwind_data[time,zmin:zmax,xmin:xmax,ymin:ymax]
    cloud_hypercube[time] = lwc_cloud[time,zmin:zmax,xmin:xmax,ymin:ymax]


###########################################
##### Variogram analysis 3D and time ######
###########################################
variograms_3Dt = cloud.sample_variogram(zwind_hypercube,'classical',cloud_hypercube)

plt.figure()
plt.title("Cloud")
plt.xlabel("$h_z,h_x,h_y(10m),h_t(s)$")
plt.ylabel("$\hat{\gamma}(|h_i|)$")
plt.plot(np.append(0,variograms_3Dt["tvariogram_hat"][:,0]),'-o',label="$\hat{\gamma}(|h_t|)$")
plt.plot(np.append(0,variograms_3Dt["zvariogram_hat"][:,0]),'-o',label="$\hat{\gamma}(|h_z|)$")
plt.plot(np.append(0,variograms_3Dt["xvariogram_hat"][:,0]),'-o',label="$\hat{\gamma}(|h_x|)$")
plt.plot(np.append(0,variograms_3Dt["yvariogram_hat"][:,0]),'-o',label="$\hat{\gamma}(|h_y|)$")
plt.axhline(zwind_hypercube[cloud_hypercube>0].var())
plt.legend()


###########################################
##### Variogram analysis 2D and time ######
###########################################

# Take only one z-cross-section
zwind_hypercube2Dt = zwind_hypercube[:,15:16]
cloud_hypercube2Dt = cloud_hypercube[:,15:16]

variograms_2Dt = cloud.sample_variogram(zwind_hypercube2Dt,'classical',cloud_hypercube2Dt)

plt.figure()
plt.title("Cloud, z={}km".format(np.round(float(zr[15]),3)))
plt.xlabel("$h_x,h_y(10m),h_t(s)$")
plt.ylabel("$\hat{\gamma}(|h_i|)$")
plt.plot(np.append(0,variograms_2Dt["tvariogram_hat"][:,0]),'-o',label="$\hat{\gamma}(|h_t|)$")
plt.plot(np.append(0,variograms_2Dt["xvariogram_hat"][:,0]),'-o',label="$\hat{\gamma}(|h_x|)$")
plt.plot(np.append(0,variograms_2Dt["yvariogram_hat"][:,0]),'-o',label="$\hat{\gamma}(|h_y|)$")
plt.axhline(zwind_hypercube2Dt[cloud_hypercube2Dt>0].var())
plt.legend()

##################################
##### Variogram analysis 2D ######
##################################

# Frozen z-cross-section of the bounding box
zwind_hypercube2D = zwind_hypercube[0:1,15:16]
cloud_hypercube2D = cloud_hypercube[0:1,15:16]

variograms_2D = cloud.sample_variogram(zwind_hypercube2D,'classical',cloud_hypercube2D)

plt.figure()
plt.title("Cloud, z={}km, t={}s".format(np.round(float(zr[15]),3),tr[0]))
plt.xlabel("$h_x,h_y(10m)$")
plt.ylabel("$\hat{\gamma}(|h_i|)$")
plt.plot(np.append(0,variograms_2D["xvariogram_hat"][:,0]),'-o',label="$\hat{\gamma}(|h_x|)$")
plt.plot(np.append(0,variograms_2D["yvariogram_hat"][:,0]),'-o',label="$\hat{\gamma}(|h_y|)$")
plt.axhline(zwind_hypercube2D[cloud_hypercube2D>0].var())
plt.legend()
