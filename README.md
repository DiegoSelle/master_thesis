## Repository of Diego Selle's Master's Thesis

This is a repository with sample code of the Master's Thesis of Diego Selle.
It was written in Python 3.5. The code will be thoroughly discussed and the context of
the master's thesis will be discussed.

-Developer: diego.selle@laas.fr, diego.selle@gmx.de

## Skyscanner Project:

The presented thesis was part of the Skyscanner project:
https://www.laas.fr/projects/skyscanner/

It is a joint venture of five research instititions in Toulouse that aims at
the study and experimentation of a fleet of mini-drones that cooperates to enable the
adaptive sampling of cumulus-type clouds. The goal is to analyze the cloud's behaviour over the
timespan of an hour, which corresponds to the expected lifetime of a cloud.


## PDF File of the Master's thesis done with LaTex

[Master's Thesis](http://bit.ly/2hXzuco)


## Code

### Requirements for Python 3.5:

+numpy>=1.10
+scipy>=0.17.1
+netCDF4>=1.2.4
+matplotlib >= 1.5.1
+GPy
+libgp

Other requirements:
+Mencoder (For the animations)
+Increase limit of simultaneous open files-> '''~$ ulimit -Sn 4096'''

###Modules:

####mesonh_atm.mesonh_atmosphere

####modules.cloud :


###Implementation Scripts

1) animation_clouds.py

Animate cross-sections of a given variable wrt time.  

2) cloud_exploration.py

3) cloud_exploration_polar.py
