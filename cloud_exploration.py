import numpy as np
from skyscan_lib.sim.mesonh_atmosphere import MesoNHAtmosphere
#from mesonh_atm.mesonh_atmosphere import MesoNHAtmosphere
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

cloud = clouds[np.argmax(length_point_clds)]

cloud.calculate_attributes(lwc_data,zwind_data)
lwc_cloud = np.zeros(lwc_data.shape)
for point in cloud.points:
    lwc_cloud[point] = 1

#Coordinates of the example cloud
xr = np.arange(0.005 + 60*0.01, 0.005 + 200*0.01,0.01)
yr = np.arange(0.005 + 110*0.01, 0.005 + 250*0.01,0.01)
all_Zs = atm.data["VLEV"][:,0,0]
zr = all_Zs[75:125]
#zr = np.arange(1.185,1.185 + 15*0.01,0.01)
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
plt.contour(lwc_cloud1[0,15].T,V=[0,1],origin='lower',extent=[xr[0], xr[-1], yr[0], yr[-1]],alpha=0.6,cmap='Greys')

COG_2D = cloud.COG_2D_tz[0,15]*0.01 + np.array([0.005 + origin_xy[0]*0.01,0.005 + origin_xy[1]*0.01])
plt.plot(COG_2D[0],COG_2D[1],'ro',markersize=8,label='COG 2D')

COM_2D_zwind = cloud.COM_2D_zwind_tz[0,15]*0.01 + np.array([0.005 + origin_xy[0]*0.01,0.005 + origin_xy[1]*0.01])
plt.plot(COM_2D_zwind[0],COM_2D_zwind[1],'gx',markersize=8, label='COM 2D zwind')

COM_2D_lwc = cloud.COM_2D_lwc_tz[0,15]*0.01 + np.array([0.005 + origin_xy[0]*0.01,0.005 + origin_xy[1]*0.01])
plt.plot(COM_2D_lwc[0],COM_2D_lwc[1],'b>',markersize=8, label='COM 2D lwc')
plt.title("Zwind Cross-section Cloud Example, z={}km, t={}s".format(np.round(float(zr[15]),3),tr[0]))
#plt.title("Zwind Cross-section, relative z={}, relative t={}".format(zindex, 0+offset))
plt.xlim(xr[0], xr[-1])
plt.ylim(yr[0], yr[-1])
plt.legend()



plt.figure()
plt.xlabel("x coordinate(km)")
plt.ylabel("y coordinate(km)")
plt.contour(zwind_data[0,19].T,origin="lower",label='zwind',extent=[xr[0], xr[-1], yr[0], yr[-1]],linewidths=2)
cbar=plt.colorbar()
cbar.set_label('m/s')
plt.contour(lwc_cloud1[0,19].T,V=[0,1],origin='lower',extent=[xr[0], xr[-1], yr[0], yr[-1]],alpha=0.6,cmap='Greys')

COG_2D = cloud1.COG_2D_tz[0,19]*0.01 + np.array([0.005 + origin_xy[0]*0.01,0.005 + origin_xy[1]*0.01])
plt.plot(COG_2D[0],COG_2D[1],'ro',markersize=8,label='COG 2D')

COM_2D_zwind = cloud1.COM_2D_zwind_tz[0,19]*0.01 + np.array([0.005 + origin_xy[0]*0.01,0.005 + origin_xy[1]*0.01])
plt.plot(COM_2D_zwind[0],COM_2D_zwind[1],'gx',markersize=8, label='COM 2D zwind')

COM_2D_lwc = cloud1.COM_2D_lwc_tz[0,19]*0.01 + np.array([0.005 + origin_xy[0]*0.01,0.005 + origin_xy[1]*0.01])
plt.plot(COM_2D_lwc[0],COM_2D_lwc[1],'b>',markersize=8, label='COM 2D lwc')
plt.title("Zwind Cross-section Cloud Example, z={}km, t={}s".format(np.round(float(zr[19]),3),tr[0]))
#plt.title("Zwind Cross-section, relative z={}, relative t={}".format(zindex, 0+offset))
plt.xlim(xr[0], xr[-1])
plt.ylim(yr[0], yr[-1])
plt.legend()


plt.figure()
plt.xlabel("x coordinate(km)")
plt.ylabel("y coordinate(km)")
plt.contour(zwind_data[0,30].T,origin="lower",label='zwind',extent=[xr[0], xr[-1], yr[0], yr[-1]],linewidths=2)
cbar=plt.colorbar()
cbar.set_label('m/s')
plt.contour(lwc_cloud1[0,30].T,V=[0,1],origin='lower',extent=[xr[0], xr[-1], yr[0], yr[-1]],alpha=0.6,cmap='Greys')

COG_2D = cloud1.COG_2D_tz[0,30]*0.01 + np.array([0.005 + origin_xy[0]*0.01,0.005 + origin_xy[1]*0.01])
plt.plot(COG_2D[0],COG_2D[1],'ro',markersize=8,label='COG 2D')

COM_2D_zwind = cloud1.COM_2D_zwind_tz[0,30]*0.01 + np.array([0.005 + origin_xy[0]*0.01,0.005 + origin_xy[1]*0.01])
plt.plot(COM_2D_zwind[0],COM_2D_zwind[1],'gx',markersize=8, label='COM 2D zwind')

COM_2D_lwc = cloud1.COM_2D_lwc_tz[0,30]*0.01 + np.array([0.005 + origin_xy[0]*0.01,0.005 + origin_xy[1]*0.01])
plt.plot(COM_2D_lwc[0],COM_2D_lwc[1],'b>',markersize=8, label='COM 2D lwc')
plt.title("Zwind Cross-section cloud1, z={}km, t={}s".format(np.round(float(zr[30]),3),tr[0]))
#plt.title("Zwind Cross-section, relative z={}, relative t={}".format(zindex, 0+offset))
plt.xlim(xr[0], xr[-1])
plt.ylim(yr[0], yr[-1])
plt.legend()

# Center of masses and Geometry
plt.figure()
plt.xlabel("z coordinate(km)")
plt.ylabel("y coordinate(km)")
plt.plot(zr,cloud1.COG_2D_tz[0,:,1]*0.01 + 0.005 + origin_xy[1]*0.01,label='COG 2D',linewidth=3)
plt.plot(zr,cloud1.COM_2D_lwc_tz[0,:,1]*0.01 + 0.005 + origin_xy[1]*0.01, label='COM 2D lwc',linewidth=3)
plt.plot(zr,cloud1.COM_2D_zwind_tz[0,:,1]*0.01 + 0.005 + origin_xy[1]*0.01, label='COM 2D zwind',linewidth=3)
plt.legend()
plt.title('Center of masses and geometry cloud1, t = {}s'.format(tr[0]))

plt.figure()
plt.xlabel("z coordinate(km)")
plt.ylabel("x coordinate(km)")
plt.plot(zr,cloud1.COG_2D_tz[0,:,0]*0.01 + 0.005 + origin_xy[1]*0.01,label='COG 2D',linewidth=3)
plt.plot(zr,cloud1.COM_2D_lwc_tz[0,:,0]*0.01 + 0.005 + origin_xy[1]*0.01, label='COM 2D lwc',linewidth=3)
plt.plot(zr,cloud1.COM_2D_zwind_tz[0,:,0]*0.01 + 0.005 + origin_xy[1]*0.01, label='COM 2D zwind',linewidth=3)
plt.legend()
plt.title('Center of masses and geometry cloud1, t = {}s'.format(tr[0]))

plt.figure()
plt.xlabel("z coordinate(km)")
plt.ylabel("Surface(100$m^2$)")
plt.plot(zr,cloud1.area_cs_tz[0],linewidth=3)
plt.title('Surface Area of cloud1, t={}s'.format(tr[0]))


plt.figure()
plt.xlabel("time(s)")
plt.ylabel("Volume(1000 $m^3$)")
plt.plot(tr,cloud1.volumen_t,linewidth=3)
plt.title('Volume of cloud1')

######### Create attribute and method for these two in time -> no time
zwind1_maxz = np.ndarray((0,1))
for z in range(int(cloud1.zmin_t[0]),int(cloud1.zmax_t[0])+1):
    zwind_max = np.max(zwind_data[0,z][lwc_cloud1[0,z]>0])
    zwind1_maxz = np.vstack((zwind1_maxz,zwind_max))

zwind1_meanz = np.ndarray((0,1))
for z in range(int(cloud1.zmin_t[0]),int(cloud1.zmax_t[0])+1):
    zwind_mean = np.mean(zwind_data[0,z][lwc_cloud1[0,z]>0])
    zwind1_meanz = np.vstack((zwind1_meanz,zwind_mean))


plt.figure()
plt.xlabel("z coordinate(km)")
plt.ylabel("Max zwind(m/s)")
plt.plot(zr[4:],zwind1_maxz,linewidth=3)
plt.title('Max Zwind per z cross-section Cloud1,t={}s'.format(tr[0]))

plt.figure()
plt.xlabel("z coordinate(km)")
plt.ylabel("Mean Zwind (m/s)")
plt.plot(zr[4:],zwind1_meanz,linewidth=3)
plt.title('Mean Zwind per z cross-section Cloud1,t={}s'.format(tr[0]))
################# Variance behaviour in z, maybe include later, if there is time
zwind1_varz = np.ndarray((0,1))
for z in range(int(cloud1.zmin_t[0]),int(cloud1.zmax_t[0])+1):
    zwind1_var = zwind_data[0,z][lwc_cloud1[0,z]>0].var()
    zwind1_varz = np.vstack((zwind1_varz,zwind1_var))

plt.figure()
plt.xlabel("z coordinate(km)")
plt.ylabel("Variance Zwind")
plt.plot(zr[4:],zwind1_varz,linewidth=3)
plt.title('Mean Zwind per z cross-section Cloud1,t={}s'.format(tr[0]))


##########################################
############# Variogram Analysis #########
##########################################

#################################
##### moving bounding box #######
################################
xbound_max = int(np.max(cloud1.xsize_t))
ybound_max = int(np.max(cloud1.ysize_t))
zbound_max = int(np.max(cloud1.zsize_t))
xcenter_t = (cloud1.xmin_t + cloud1.xmax_t)/2
ycenter_t = (cloud1.ymin_t + cloud1.ymax_t)/2
zcenter_t = (cloud1.zmin_t + cloud1.zmax_t)/2

zwind1_hypercube = np.ndarray((cloud1.tmax+1-cloud1.tmin,zbound_max,xbound_max,ybound_max))
cloud1_hypercube = np.ndarray((cloud1.tmax+1-cloud1.tmin,zbound_max,xbound_max,ybound_max))
total_size = lwc_cloud1.shape

### Make this an attribute maybe ?
for time in range(cloud1.tmin,cloud1.tmax+1):
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

    zwind1_hypercube[time] = zwind_data[time,zmin:zmax,xmin:xmax,ymin:ymax]
    cloud1_hypercube[time] = lwc_cloud1[time,zmin:zmax,xmin:xmax,ymin:ymax]

variograms1 = cloud.sample_variogram(zwind1_hypercube,'classical',cloud1_hypercube)

plt.figure()
plt.title("Cloud1")
plt.xlabel("$h_z,h_x,h_y(10m),h_t(s)$")
plt.ylabel("$\hat{\gamma}(|h_i|)$")
plt.plot(np.append(0,variograms1["tvariogram_hat"][:,0]),'-o',label="$\hat{\gamma}(|h_t|)$")
plt.plot(np.append(0,variograms1["zvariogram_hat"][:,0]),'-o',label="$\hat{\gamma}(|h_z|)$")
plt.plot(np.append(0,variograms1["xvariogram_hat"][:,0]),'-o',label="$\hat{\gamma}(|h_x|)$")
plt.plot(np.append(0,variograms1["yvariogram_hat"][:,0]),'-o',label="$\hat{\gamma}(|h_y|)$")
plt.axhline(zwind1_hypercube[cloud1_hypercube>0].var())
plt.legend()
#plt.savefig("variograms_cloud1.png")

###########################################
##### Variogram analysis 2D and time ######
###########################################

zwind1_hypercube2dt = zwind1_hypercube[:,15:16]
cloud1_hypercube2dt = cloud1_hypercube[:,15:16]

variograms1_2dt = cloud.sample_variogram(zwind1_hypercube2dt,'classical',cloud1_hypercube2dt)

plt.figure()
plt.title("Cloud1, z={}km".format(np.round(float(zr[15]),3)))
plt.xlabel("$h_x,h_y(10m),h_t(s)$")
plt.ylabel("$\hat{\gamma}(|h_i|)$")
plt.plot(np.append(0,variograms1_2dt["tvariogram_hat"][:,0]),'-o',label="$\hat{\gamma}(|h_t|)$")
plt.plot(np.append(0,variograms1_2dt["xvariogram_hat"][:,0]),'-o',label="$\hat{\gamma}(|h_x|)$")
plt.plot(np.append(0,variograms1_2dt["yvariogram_hat"][:,0]),'-o',label="$\hat{\gamma}(|h_y|)$")
plt.axhline(zwind1_hypercube2dt[cloud1_hypercube2dt>0].var())
plt.legend()

##################################
##### Variogram analysis 2D ######
##################################

zwind1_hypercube2d = zwind1_hypercube[0:1,15:16]
cloud1_hypercube2d = cloud1_hypercube[0:1,15:16]

variograms1_2d = cloud.sample_variogram(zwind1_hypercube2d,'classical',cloud1_hypercube2d)

plt.figure()
plt.title("Cloud1, z={}km, t={}s".format(np.round(float(zr[15]),3),tr[0]))
plt.xlabel("$h_x,h_y(10m)$")
plt.ylabel("$\hat{\gamma}(|h_i|)$")
plt.plot(np.append(0,variograms1_2d["xvariogram_hat"][:,0]),'-o',label="$\hat{\gamma}(|h_x|)$")
plt.plot(np.append(0,variograms1_2d["yvariogram_hat"][:,0]),'-o',label="$\hat{\gamma}(|h_y|)$")
plt.axhline(zwind1_hypercube2d[cloud1_hypercube2d>0].var())
plt.legend()
