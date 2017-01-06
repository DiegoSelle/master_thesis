import numpy as np
import scipy.optimize as sciopt
import scipy.special as scispec

# Cloud object with different geometrical information such as
# Centers of gravity 2D and 3D, bounding box, etc. for the relevant variables of vertical wind and
# liquid water content. The cloud object is created by segmenting the data.
class cloud(object):
    def __init__(self,id):
        self.id = id
        self.merged = set({id})

        # Bounding Box during whole timespan
        self.tmin = -np.inf
        self.tmax = np.inf
        self.xmin = -np.inf
        self.xmax = np.inf
        self.ymin = -np.inf
        self.ymax = np.inf
        self.zmin = -np.inf
        self.zmax = np.inf

        # Bounding Box for each time
        self.xmin_t = np.ndarray((0,1))
        self.xmax_t = np.ndarray((0,1))
        self.ymin_t = np.ndarray((0,1))
        self.ymax_t = np.ndarray((0,1))
        self.zmin_t = np.ndarray((0,1))
        self.zmax_t = np.ndarray((0,1))
        self.xsize_t = np.ndarray((0,1))
        self.ysize_t = np.ndarray((0,1))
        self.zsize_t = np.ndarray((0,1))

        self.volumen_t = np.ndarray((0,1))
        # Center of Mass LWC
        self.COM_3D_lwc_t = np.ndarray((0,3))
        #Center of Geometry LWC
        self.COG_3D_t = np.ndarray((0,3))
        self.COM_2D_lwc_tz = np.ndarray((0,0,2))
        self.COG_2D_tz = np.ndarray((0,0,2))
        self.COM_2D_zwind_tz = np.ndarray((0,0,2))
        self.area_cs_tz = np.ndarray((0,0,1))
        self.zwind_model = []
        self.points = []
        self.recalculate = False
    def merge(self,other_cloud):
        self.points.extend(other_cloud.points)
        self.merged.update(other_cloud.merged)
        self.recalculate = True
        # Status= Recalculate
    def add_point(self,point):
        self.points.append(point)
        self.recalculate = True
    def _bounding_box(self):
        points_cloud=np.array(self.points)
        self.tmin = np.min(points_cloud[:,0])
        self.tmax = np.max(points_cloud[:,0])
        self.zmin = np.min(points_cloud[:,1])
        self.zmax = np.max(points_cloud[:,1])
        self.xmin = np.min(points_cloud[:,2])
        self.xmax = np.max(points_cloud[:,2])
        self.ymin = np.min(points_cloud[:,0])
        self.ymax = np.max(points_cloud[:,0])
    def _bounding_box_t(self):
        self.xmin_t = np.ndarray((0,1))
        self.xmax_t = np.ndarray((0,1))
        self.ymin_t = np.ndarray((0,1))
        self.ymax_t = np.ndarray((0,1))
        self.zmin_t = np.ndarray((0,1))
        self.zmax_t = np.ndarray((0,1))
        self.xsize_t = np.ndarray((0,1))
        self.ysize_t = np.ndarray((0,1))
        self.zsize_t = np.ndarray((0,1))


        points_cloud = np.array(self.points)
        for time in range(int(self.tmin),int(self.tmax+1)):
            temp_t = points_cloud == time
            x_time = points_cloud[temp_t[:,0]][:,2]
            y_time = points_cloud[temp_t[:,0]][:,3]
            z_time = points_cloud[temp_t[:,0]][:,1]
            xmin_time = np.min(x_time)
            xmax_time = np.max(x_time)
            ymin_time = np.min(y_time)
            ymax_time = np.max(y_time)
            zmin_time = np.min(z_time)
            zmax_time = np.max(z_time)
            xsize_time = xmax_time - xmin_time
            ysize_time = ymax_time - ymin_time
            zsize_time = zmax_time - zmin_time

            self.xmin_t = np.vstack((self.xmin_t,xmin_time))
            self.xmax_t = np.vstack((self.xmax_t,xmax_time))
            self.ymin_t = np.vstack((self.ymin_t,ymin_time))
            self.ymax_t = np.vstack((self.ymax_t,ymax_time))
            self.zmin_t = np.vstack((self.zmin_t,zmin_time))
            self.zmax_t = np.vstack((self.zmax_t,zmax_time))
            self.xsize_t = np.vstack((self.xsize_t,xsize_time))
            self.ysize_t = np.vstack((self.ysize_t,ysize_time))
            self.zsize_t = np.vstack((self.zsize_t,zsize_time))
    def _com3d_lwc_t(self,lwc_data):
        #lwc_data is the data used in cloud_segmentation()
        self.COM_3D_lwc_t = np.ndarray((0,3))

        if not lwc_data.any():
            print("lwc_data is empty")
        else:
            points_cloud = np.array(self.points)
            for time in range(int(self.tmin),int(self.tmax+1)):
                temp_t = points_cloud == time
                points_time = points_cloud[temp_t[:,0]]
                lwc_values_t = np.ndarray((0,1))
                for point in points_time:
                    lwc_values_t=np.vstack((lwc_values_t,lwc_data[tuple(point)]))
                temp_com = points_time*lwc_values_t
                xc = np.sum(temp_com[:,2])/np.sum(lwc_values_t)
                zc = np.sum(temp_com[:,1])/np.sum(lwc_values_t)
                yc = np.sum(temp_com[:,3])/np.sum(lwc_values_t)
                self.COM_3D_lwc_t = np.vstack((self.COM_3D_lwc_t,[zc,xc,yc]))
    def _cog3d_cloud_t(self):
        self.COG_3D_t = np.ndarray((0,3))

        points_cloud = np.array(self.points)
        for time in range(int(self.tmin),int(self.tmax+1)):
            temp_t = points_cloud == time
            x_time = points_cloud[temp_t[:,0]][:,2]
            y_time = points_cloud[temp_t[:,0]][:,3]
            z_time = points_cloud[temp_t[:,0]][:,1]
            xc = x_time.mean()
            zc = z_time.mean()
            yc = y_time.mean()
            self.COG_3D_t = np.vstack((self.COG_3D_t,[zc,xc,yc]))
    def _com2d_lwc_tz(self,lwc_data):
        #lwc_data is the data used in cloud_segmentation()
        if not lwc_data.any():
            print("lwc_data is empty")
        else:
            points_cloud = np.array(self.points)
            self.COM_2D_lwc_tz= np.ndarray((int(self.tmax)+1,int(self.zmax)+1,2))
            self.COM_2D_lwc_tz[:] = np.NAN
            for time in range(int(self.tmin),int(self.tmax+1)):
                for z in range(int(self.zmin_t[time]),int(self.zmax_t[time]+1)):
                    temp_t = points_cloud == time
                    points_time = points_cloud[temp_t[:,0]]
                    temp_z = points_time == z
                    points_time_z = points_time[temp_z[:,1]]
                    lwc_values_tz = np.ndarray((0,1))
                    for point in points_time_z:
                        lwc_values_tz=np.vstack((lwc_values_tz,lwc_data[tuple(point)]))
                    temp_com = lwc_values_tz*points_time_z
                    xc = np.sum(temp_com[:,2])/(np.sum(lwc_values_tz))
                    yc = np.sum(temp_com[:,3])/(np.sum(lwc_values_tz))
                    self.COM_2D_lwc_tz[time,z] = [xc, yc]
    def _cog2d_cloud_tz(self):
        points_cloud = np.array(self.points)
        self.COG_2D_tz = np.ndarray((int(self.tmax)+1,int(self.zmax)+1,2))
        self.COG_2D_tz[:] = np.NAN
        for time in range(int(self.tmin),int(self.tmax+1)):
            for z in range(int(self.zmin_t[time]),int(self.zmax_t[time]+1)):
                temp_t = points_cloud == time
                points_time = points_cloud[temp_t[:,0]]
                temp_z = points_time == z
                points_time_z = points_time[temp_z[:,1]]
                xc = points_time_z[:,2].mean()
                yc = points_time_z[:,3].mean()
                self.COG_2D_tz[time,z] = [xc,yc]
    def _com2d_zwind_tz(self,zwind_data):
        #zwind_data has to have same indexing as lwc_data used for cloud_segmentation
        if not zwind_data.any():
            print("zwind_data is empty")
        else:
            points_cloud = np.array(self.points)
            self.COM_2D_zwind_tz= np.ndarray((int(self.tmax)+1,int(self.zmax)+1,2))
            self.COM_2D_zwind_tz[:] = np.NAN
            for time in range(int(self.tmin),int(self.tmax+1)):
                for z in range(int(self.zmin_t[time]),int(self.zmax_t[time]+1)):
                    temp_t = points_cloud == time
                    points_time = points_cloud[temp_t[:,0]]
                    temp_z = points_time == z
                    points_time_z = points_time[temp_z[:,1]]
                    zwind_values_tz = np.ndarray((0,1))
                    for point in points_time_z:
                        zwind_values_tz=np.vstack((zwind_values_tz,zwind_data[tuple(point)]))
                    temp_com = zwind_values_tz*points_time_z
                    xc = (np.sum(temp_com[:,2]))/(np.sum(zwind_values_tz))
                    yc = (np.sum(temp_com[:,3]))/(np.sum(zwind_values_tz))
                    self.COM_2D_zwind_tz[time,z] = [xc, yc]
    def _volumen_t(self):
        self.volumen_t = np.ndarray((0,1))

        points_cloud = np.array(self.points)
        for time in range(int(self.tmin),int(self.tmax+1)):
            temp_t = points_cloud == time
            points_time = points_cloud[temp_t[:,0]]
            self.volumen_t = np.vstack((self.volumen_t,len(points_time)))
    def _area_cs_tz(self):
        points_cloud = np.array(self.points)
        self.area_cs_tz = np.ndarray((int(self.tmax)+1,int(self.zmax)+1,1))
        self.area_cs_tz[:] = np.NAN
        for time in range(int(self.tmin),int(self.tmax+1)):
            for z in range(int(self.zmin_t[time]),int(self.zmax_t[time]+1)):
                temp_t = points_cloud == time
                points_time = points_cloud[temp_t[:,0]]
                temp_z = points_time == z
                points_time_z = points_time[temp_z[:,1]]
                self.area_cs_tz[time,z] = len(points_time_z)
    def calculate_attributes(self,lwc_data=np.array([]),zwind_data=np.array([])):
        if self.recalculate:
            self._bounding_box()
            self._bounding_box_t()
            self._volumen_t()
            self._area_cs_tz()
            self._com3d_lwc_t(lwc_data)
            self._com2d_lwc_tz(lwc_data)
            self._cog3d_cloud_t()
            self._cog2d_cloud_tz()
            self._com2d_zwind_tz(zwind_data)
        else:
            print("Cloud Attributes are up to Date")
        self.recalculate=False

# Sub-function for segmenting clouds, which is based on an exhaustive neighbor check of the
# data.
def check_neighbours(ids,counter,clouds,current_index):
    temp_ids=(ids[tuple(current_index+[0,1,1,1])],
        ids[tuple(current_index+[1,0,1,1])],
        ids[tuple(current_index+[1,1,0,1])],
        ids[tuple(current_index+[1,1,1,0])])

    unique_temp_ids=np.unique(temp_ids)
    current_index=tuple(current_index)
    if len(unique_temp_ids)==1:
        if unique_temp_ids[0]==0:
            counter=counter+1
            ids[tuple(np.array(current_index)+[1,1,1,1])]=counter
            c=cloud(counter)
            clouds[counter]=c
            c.add_point(current_index)
        else:
            ids[tuple(np.array(current_index)+[1,1,1,1])]=unique_temp_ids[0]
            c=clouds[unique_temp_ids[0]]
            c.add_point(current_index)

    elif len(unique_temp_ids)==2 and unique_temp_ids[0]==0:
        c=clouds[unique_temp_ids[1]]
        ids[tuple(np.array(current_index)+[1,1,1,1])]=unique_temp_ids[1]
        c.add_point(current_index)
    else:
        if unique_temp_ids[0]==0:
            unique_temp_ids=unique_temp_ids[1:]
        c=clouds[unique_temp_ids[0]]
        ids[tuple(np.array(current_index)+[1,1,1,1])]=unique_temp_ids[0]
        c.add_point(current_index)
        for i in range(1,len(unique_temp_ids)):
            if c.id != clouds[unique_temp_ids[i]].id:
                #print("merging {}/{}<--{}/{}".format(unique_temp_ids[0],c.id,unique_temp_ids[i],clouds[unique_temp_ids[i]].id))
                c.merge(clouds[unique_temp_ids[i]])
                for j in c.merged:
                    clouds[j]=c
    return counter

def cloud_segmentation(lwc_data1):
    lwc_data = lwc_data1.copy()
    lwc_data[lwc_data>=1e-5]=1
    lwc_data[lwc_data<1e-5]=0
    size_data=np.array(lwc_data.shape)
    print(size_data)
    ids=np.zeros(size_data+1)
    counter=0
    clouds={}
    for t in range(size_data[0]):
        for z in range(size_data[1]):
            for x in range(size_data[2]):
                for y in range(size_data[3]):
                    if lwc_data[t,z,x,y]==0:
                        ids[t,z,x,y]=0
                    else:
                        current_index=np.array([t,z,x,y])
                        counter=check_neighbours(ids,counter,clouds,current_index)
    return ids,counter,clouds

toy_cloud=np.array([[0,1,0,0,0,0],[0,1,0,1,1,1],[0,1,1,1,0,0],[0,0,0,0,0,0],[0,0,0,0,1,1],[0,0,0,0,1,1]])


###################################################################
#################### sample variogram function ####################
###################################################################

def sample_variogram(var_hypercube,method,lwc_hypercube=np.array([])):
    if len(var_hypercube.shape)!=4:
        return "var_hypercube does not have correct shape"
    else:
        tvariogram_hat=np.array([])
        zvariogram_hat=np.array([])
        xvariogram_hat=np.array([])
        yvariogram_hat=np.array([])
        for tindex in range((var_hypercube.shape[0]-1)):
            if method=="classical":
                if not lwc_hypercube.any():
                    tgamma_h=(var_hypercube[(tindex+1):,:,:,:]-var_hypercube[0:-(tindex+1),:,:,:])**2
                    tNh=np.prod(tgamma_h.shape)
                    tgamma_h=np.sum(tgamma_h)/(2*tNh)
                elif lwc_hypercube.any():
                    if lwc_hypercube.shape==var_hypercube.shape:
                        cloud=lwc_hypercube>=1e-5
                        tcloud_h=np.logical_and(cloud[(tindex+1):,:,:,:],cloud[0:-(tindex+1),:,:,:])
                        tgamma_h=(var_hypercube[(tindex+1):,:,:,:]-var_hypercube[0:-(tindex+1),:,:,:])**2
                        tNh=np.count_nonzero(tcloud_h)
                        tgamma_h=np.sum(tgamma_h[tcloud_h])/(2*tNh)
                    else:
                        return "lwc_hypercube and var_hypercube do not have the same dimensions"
            elif method=="robust":
                if not lwc_hypercube.any():
                    tgamma_h=(np.abs(var_hypercube[(tindex+1):,:,:,:]-var_hypercube[0:-(tindex+1),:,:,:]))**0.5
                    tNh=np.prod(tgamma_h.shape)
                    tgamma_h=0.5*((np.sum(tgamma_h)/tNh)**4)/(0.457+0.494/tNh)
                elif lwc_hypercube.any():
                    if lwc_hypercube.shape==var_hypercube.shape:
                        cloud=lwc_hypercube>=1e-5
                        tcloud_h=np.logical_and(cloud[(tindex+1):,:,:,:],cloud[0:-(tindex+1),:,:,:])
                        tgamma_h=(np.abs(var_hypercube[(tindex+1):,:,:,:]-var_hypercube[0:-(tindex+1),:,:,:]))**0.5
                        tNh=np.count_nonzero(tcloud_h)
                        tgamma_h=0.5*((np.sum(tgamma_h[tcloud_h])/tNh)**4)/(0.457+0.494/tNh)
                    else:
                        return "lwc_hypercube and var_hypercube do not have the same dimensions"
            else:
                return "Method is not supported"
            if tindex==0:
                tvariogram_hat=np.array([[tgamma_h,tNh]])
            else:
                tvariogram_hat=np.append(tvariogram_hat,np.array([[tgamma_h,tNh]]),axis=0)
        for zindex in range((var_hypercube.shape[1]-1)):
            if method=="classical":
                if not lwc_hypercube.any():
                    zgamma_h=(var_hypercube[:,(zindex+1):,:,:]-var_hypercube[:,0:-(zindex+1),:,:])**2
                    zNh=np.prod(zgamma_h.shape)
                    zgamma_h=np.sum(zgamma_h)/(2*zNh)
                elif lwc_hypercube.any():
                    if lwc_hypercube.shape==var_hypercube.shape:
                        cloud=lwc_hypercube>=1e-5
                        zcloud_h=np.logical_and(cloud[:,(zindex+1):,:,:],cloud[:,0:-(zindex+1),:,:])
                        zgamma_h=(var_hypercube[:,(zindex+1):,:,:]-var_hypercube[:,0:-(zindex+1),:,:])**2
                        zNh=np.count_nonzero(zcloud_h)
                        zgamma_h=np.sum(zgamma_h[zcloud_h])/(2*zNh)
                    else:
                        return "lwc_hypercube and var_hypercube do not have the same dimensions"
            elif method=="robust":
                if not lwc_hypercube.any():
                    zgamma_h=(np.abs(var_hypercube[:,(zindex+1):,:,:]-var_hypercube[:,0:-(zindex+1),:,:]))**0.5
                    zNh=np.prod(zgamma_h.shape)
                    zgamma_h=0.5*((np.sum(zgamma_h)/zNh)**4)/(0.457+0.494/zNh)
                elif lwc_hypercube.any():
                    if lwc_hypercube.shape==var_hypercube.shape:
                        cloud=lwc_hypercube>=1e-5
                        zcloud_h=np.logical_and(cloud[:,(zindex+1):,:,:],cloud[:,0:-(zindex+1),:,:])
                        zgamma_h=(np.abs(var_hypercube[:,(zindex+1):,:,:]-var_hypercube[:,0:-(zindex+1),:,:]))**0.5
                        zNh=np.count_nonzero(zcloud_h)
                        zgamma_h=0.5*((np.sum(zgamma_h[zcloud_h])/zNh)**4)/(0.457+0.494/zNh)
                    else:
                        return "lwc_hypercube and var_hypercube do not have the same dimensions"
            else:
                return "Method is not supported"
            if zindex==0:
                zvariogram_hat=np.array([[zgamma_h,zNh]])
            else:
                zvariogram_hat=np.append(zvariogram_hat,np.array([[zgamma_h,zNh]]),axis=0)

        for xindex in range((var_hypercube.shape[2]-1)):
            if method=="classical":
                if not lwc_hypercube.any():
                    xgamma_h=(var_hypercube[:,:,(xindex+1):,:]-var_hypercube[:,:,0:-(xindex+1),:])**2
                    xNh=np.prod(xgamma_h.shape)
                    xgamma_h=np.sum(xgamma_h)/(2*xNh)
                elif lwc_hypercube.any():
                    if lwc_hypercube.shape==var_hypercube.shape:
                        cloud=lwc_hypercube>=1e-5
                        xcloud_h=np.logical_and(cloud[:,:,(xindex+1):,:],cloud[:,:,0:-(xindex+1),:])
                        xgamma_h=(var_hypercube[:,:,(xindex+1):,:]-var_hypercube[:,:,0:-(xindex+1),:])**2
                        xNh=np.count_nonzero(xcloud_h)
                        xgamma_h=np.sum(xgamma_h[xcloud_h])/(2*xNh)
                    else:
                        return "lwc_hypercube and var_hypercube do not have the same dimensions"
            elif method=="robust":
                if not lwc_hypercube.any():
                    xgamma_h=(np.abs(var_hypercube[:,:,(xindex+1):,:]-var_hypercube[:,:,0:-(xindex+1),:]))**0.5
                    xNh=np.prod(xgamma_h.shape)
                    xgamma_h=0.5*((np.sum(xgamma_h)/xNh)**4)/(0.457+0.494/xNh)
                if lwc_hypercube.any():
                    if lwc_hypercube.shape==var_hypercube.shape:
                        cloud=lwc_hypercube>=1e-5
                        xcloud_h=np.logical_and(cloud[:,:,(xindex+1):,:],cloud[:,:,0:-(xindex+1),:])
                        xgamma_h=(np.abs(var_hypercube[:,:,(xindex+1):,:]-var_hypercube[:,:,0:-(xindex+1),:]))**0.5
                        xNh=np.count_nonzero(xcloud_h)
                        xgamma_h=0.5*((np.sum(xgamma_h[xcloud_h])/xNh)**4)/(0.457+0.494/xNh)
                    else:
                        return "lwc_hypercube and var_hypercube do not have the same dimensions"
            else:
                return "Method is not supported"
            if xindex==0:
                xvariogram_hat=np.array([[xgamma_h,xNh]])
            else:
                xvariogram_hat=np.append(xvariogram_hat,np.array([[xgamma_h,xNh]]),axis=0)
        for yindex in range((var_hypercube.shape[3]-1)):
            if method=="classical":
                if not lwc_hypercube.any():
                    ygamma_h=(var_hypercube[:,:,:,(yindex+1):]-var_hypercube[:,:,:,0:-(yindex+1)])**2
                    yNh=np.prod(ygamma_h.shape)
                    ygamma_h=np.sum(ygamma_h)/(2*yNh)
                if lwc_hypercube.any():
                    if lwc_hypercube.shape==var_hypercube.shape:
                        cloud=lwc_hypercube>=1e-5
                        ycloud_h=np.logical_and(cloud[:,:,:,(yindex+1):],cloud[:,:,:,0:-(yindex+1)])
                        ygamma_h=(var_hypercube[:,:,:,(yindex+1):]-var_hypercube[:,:,:,0:-(yindex+1)])**2
                        yNh=np.count_nonzero(ycloud_h)
                        ygamma_h=np.sum(ygamma_h[ycloud_h])/(2*yNh)
                    else:
                        return "lwc_hypercube and var_hypercube do not have the same dimensions"
            elif method=="robust":
                if not lwc_hypercube.any():
                    ygamma_h=(np.abs(var_hypercube[:,:,:,(yindex+1):]-var_hypercube[:,:,:,0:-(yindex+1)]))**0.5
                    yNh=np.prod(ygamma_h.shape)
                    ygamma_h=0.5*((np.sum(ygamma_h)/yNh)**4)/(0.457+0.494/yNh)
                if lwc_hypercube.any():
                    if lwc_hypercube.shape==var_hypercube.shape:
                        cloud=lwc_hypercube>=1e-5
                        ycloud_h=np.logical_and(cloud[:,:,:,(yindex+1):],cloud[:,:,:,0:-(yindex+1)])
                        ygamma_h=(np.abs(var_hypercube[:,:,:,(yindex+1):]-var_hypercube[:,:,:,0:-(yindex+1)]))**0.5
                        yNh=np.count_nonzero(ycloud_h)
                        ygamma_h=0.5*((np.sum(ygamma_h[ycloud_h])/yNh)**4)/(0.457+0.494/yNh)
                    else:
                        return "lwc_hypercube and var_hypercube do not have the same dimensions"
            else:
                return "Method is not supported"
            if yindex==0:
                yvariogram_hat=np.array([[ygamma_h,yNh]])
            else:
                yvariogram_hat=np.append(yvariogram_hat,np.array([[ygamma_h,yNh]]),axis=0)

    return {"tvariogram_hat":tvariogram_hat,"zvariogram_hat":zvariogram_hat,"xvariogram_hat":xvariogram_hat,"yvariogram_hat":yvariogram_hat}

###############################################################################################
########################### Function to fit variogram model ###################################
###############################################################################################

def fit_variogram(variogram_hat,h,model):
    # do check of whether h conforms with variogram_hat
    if model=="gaussian":
        p0=[0.5,30]
        Nh=variogram_hat[:,1]
        gamma_h_hat=variogram_hat[:,0]
        bounds=[1e-9,np.inf]
        def residuals(p,gamma_h_hat,h,Nh):
            sigma,l=p
            gamma_h_model=(sigma**2)*(1-np.exp(-0.5*(h/l)**2))
            err = (Nh**0.5)*(gamma_h_hat/gamma_h_model-1)
            return err
        p_wlsq = sciopt.least_squares(residuals, p0, args=(gamma_h_hat, h,Nh),bounds=bounds)
        #p_wlsq1 = sciopt.leastsq(residuals, p0, args=(gamma_h_hat, h,Nh),bounds=bounds)
        return p_wlsq["x"],p_wlsq["cost"],p_wlsq
    elif model=="exponential":
        p0=[0.5,30]
        Nh=variogram_hat[:,1]
        gamma_h_hat=variogram_hat[:,0]
        bounds=[1e-9,np.inf]
        def residuals(p,gamma_h_hat,h,Nh):
            sigma,l=p
            gamma_h_model=(sigma**2)*(1-np.exp(-(h/l)))
            err = (Nh**0.5)*(gamma_h_hat/gamma_h_model-1)
            return err
        p_wlsq = sciopt.least_squares(residuals, p0, args=(gamma_h_hat, h,Nh),bounds=bounds)
        return p_wlsq["x"],p_wlsq["cost"],p_wlsq
    elif model=="matern_simp32":
        #nu=3/2
        p0=[0.5,30]
        Nh=variogram_hat[:,1]
        gamma_h_hat=variogram_hat[:,0]
        bounds=[1e-9,np.inf]
        def residuals(p,gamma_h_hat,h,Nh):
            sigma,l=p
            gamma_h_model=(sigma**2)*(1-(1+(3**0.5)*h/l)*np.exp(-((3**0.5)*h/l)))
            err = (Nh**0.5)*(gamma_h_hat/gamma_h_model-1)
            return err
        p_wlsq = sciopt.least_squares(residuals, p0, args=(gamma_h_hat, h,Nh),bounds=bounds)
        return p_wlsq["x"],p_wlsq["cost"],p_wlsq
    elif model=="matern_simp52":
        #nu=5/2
        p0=[0.5,30]
        Nh=variogram_hat[:,1]
        gamma_h_hat=variogram_hat[:,0]
        bounds=[1e-9,np.inf]
        def residuals(p,gamma_h_hat,h,Nh):
            sigma,l=p
            gamma_h_model=(sigma**2)*(1-(1+(5**0.5)*h/l+(5/3)*(h/l)**2)*np.exp(-((5**0.5)*h/l)))
            err = (Nh**0.5)*(gamma_h_hat/gamma_h_model-1)
            return err
        p_wlsq = sciopt.least_squares(residuals, p0, args=(gamma_h_hat, h,Nh), bounds=bounds)
        return p_wlsq["x"],p_wlsq["cost"],p_wlsq
    elif model=="matern_general":
        p0=[0.5,30,0.5]
        Nh=variogram_hat[:,1]
        gamma_h_hat=variogram_hat[:,0]
        bounds=[1e-9,np.inf]
        def residuals(p,gamma_h_hat,h,Nh):
            sigma,l,nu=p
            gamma_h_model=(sigma**2)*(1-(2**(1-nu))/(scispec.gamma(nu))*(((2**0.5)*nu*h/l)**nu)*scispec.kv(nu,((2**0.5)*nu*h/l)))
            err = (Nh**0.5)*(gamma_h_hat/gamma_h_model-1)
            return err
        p_wlsq = sciopt.least_squares(residuals, p0, args=(gamma_h_hat, h,Nh),bounds=bounds)
        return p_wlsq["x"],p_wlsq["cost"],p_wlsq
    else:
        print("Variogram model not supported")


##################################################################
############### Function to transform clouds       ###############
############### to polar representation, including ###############
###############        normalization               ###############
##################################################################
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


##########################################################################################################
################ Function to normalize list of points, for example training inputs and test inputs #######
##########################################################################################################

def normalize(lwc_cloud_polar,points_cartesian,COM_2D_lwc_tz,tstart,zstart,origin_xy):
    # assuming in lwc_cloud_polar ndarray((tr,zr,360,151)) with lwc values and rstep = 0.005 km
    # and coordinates[t,z,phi,r],resolution of r in 5m and phi in degrees
    # tstart(seconds) and zstart(0.01m) are the time and height coordinates where the data of lwc_cloud_polar started
    # For indexation in z, it will be assumed that the resolution is 0.01 km(10m)
    # assuming points_cartesian as list of points [t,z,x,y] n by 4 and domain does not exceed lwc_cloud_polar
    points_polar_normal = np.ndarray((0,4))
    for point in points_cartesian:
        point_polar_normal = np.array([])
        trel = int(point[0] - tstart)
        zrel = int(np.round((point[1] - zstart)/0.01))
        COMx = COM_2D_lwc_tz[trel,zrel,0]
        COMy = COM_2D_lwc_tz[trel,zrel,1]
        dx = point[2] - (COMx + origin_xy[0])*0.01
        dy = point[3] - (COMy + origin_xy[1])*0.01
        phi = np.arctan2(dy,dx)/np.pi*180
        if phi < 0:
            phi = 360 + phi
        r = np.sqrt(dx**2 + dy**2)

        max_rs = np.array([])
        for step_phi in range(0,361):
            mod_phi = int(np.mod(step_phi,360))
            max_r = np.max(np.argwhere(lwc_cloud_polar[trel,zrel,mod_phi]))*0.005
            max_rs = np.append(max_rs,max_r)
            #print(max_rs.shape)
        interpolate_max_r= RegularGridInterpolator(points=np.arange(0,361).reshape(1,361),values=max_rs,bounds_error=False,fill_value=0)
        #print(phi)
        max_r = interpolate_max_r(np.array([phi]),'linear')
        r_norm = 100*r/max_r
        point_polar_normal = np.array([point[0],point[1],phi,r_norm])
        points_polar_normal = np.vstack((points_polar_normal,point_polar_normal))
    return points_polar_normal
