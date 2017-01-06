import numpy as np
import matplotlib.pyplot as plt
from mesonh_atm.mesonh_atmosphere import MesoNHAtmosphere
import matplotlib.animation as animation
plt.rcParams['animation.mencoder_path'] = '/usr/bin/mencoder'

#Data without advection
path = "/net/skyscanner/volume1/data/mesoNH/ARM_OneHour3600files_No_Horizontal_Wind/"
mfiles = [path+"U0K10.1.min{:02d}.{:03d}_diaKCL.nc".format(minute, second)
          for minute in range(1, 60)
          for second in range(1, 61)]
mtstep = 1
atm = MesoNHAtmosphere(mfiles, 1)

#Different Data with advection
path2="/net/skyscanner/volume1/data/mesoNH/Data_5min_at_second/"
mfiles2 = [path2+"min{:02d}/SK1mn.1.min{:02d}.{:03d}_diaKCL.nc".format(minute,minute, second)
          for minute in range(3, 6)
          for second in range(1, 61) if minute!=3 or second!=5]
atm2 = MesoNHAtmosphere(mfiles2, 1)

path3="/net/skyscanner/volume1/data/mesoNH/Z_June2016_U10_Hour06_Sec/"
mfiles3 = [path3+"min{:02d}/hr06_m{:02d}_{:03d}_diaKCL.nc".format(minute,minute, second)
          for minute in range(1, 6)
          for second in range(1, 61)]

atm3 = MesoNHAtmosphere(mfiles3, 1)

path4="/net/skyscanner/volume1/data/mesoNH/Z_June2016_U00_Hour06_Sec/"
mfiles4 = [path4+"min{:02d}/h6_U0_m{:02d}_{:03d}_diaKCL.nc".format(minute,minute, second)
          for minute in range(1, 6)
          for second in range(1, 61)]

          #hr06_m01_001_diaKCL.nc
atm4 = MesoNHAtmosphere(mfiles4, 1)

###################################################################
#################### example Animation ############################
###################################################################
zindex = 65
xindex=slice(400)
yindex=slice(400)
tframes=299

fig = plt.figure("my_name")
fig.clf()
title = None

var = "RCT"
rmin=np.min(atm.data[var][:tframes,zindex,xindex,yindex])
rmax=np.max(atm.data[var][:tframes,zindex,xindex,yindex])

im = None
d = None
cbar = None
p=None
offset=0

def fun_anim_init():
    global d, im, cbar, title,p
    fig.clf()
    title = plt.title("RCT Time Animation, Zindex={}, t={}".format(zindex, 0+offset))
    plt.xlabel("Y Index")
    plt.ylabel("X Index")

    d = atm.data[var][0+offset, zindex,xindex,yindex]
    #d[d > rmin] = rmax
    im = plt.imshow(d, origin="lower", vmin=rmin, vmax=rmax, cmap="viridis")
    p = plt.plot(300,200, "ro", markersize=6)
    plt.xlim(0,400)
    plt.ylim(0,400)
    cbar = plt.colorbar()
    cbar.set_label("kg/kg")


def fun_anim(i):
    global d, im, title
    d[:] = atm.data[var][i+offset, zindex,xindex,yindex]
    #d[d > rmin] = rmax
    im.set_data(d)
    title.set_text("RCT Time Animation, Zindex={}, t={}".format(zindex, i+offset))
    return im, title

fun_anim_init()
ani = animation.FuncAnimation(fig, fun_anim, frames=tframes,
                              init_func=None,
                              interval=50, blit=False)


ani.save('WT_animation_new_simu_with_advection.mp4',writer="mencoder")
plt.show()
