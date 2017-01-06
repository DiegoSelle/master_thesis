## Repository of Diego Selle's Master's Thesis

This is a repository with sample code of the Master's Thesis of Diego Selle.
It was written in Python 3.5. The code will be thoroughly discussed and the context of
the master's thesis will also be presented.

-Developer: diego.selle@laas.fr, diego.selle@gmx.de

## Skyscanner Project:

The presented thesis was part of the Skyscanner project:
https://www.laas.fr/projects/skyscanner/

It is a joint venture of [five research instititions](https://www.laas.fr/projects/skyscanner/skyscanner-team)
in Toulouse that aims at
the study and experimentation of a fleet of mini-drones that cooperates to enable the
adaptive sampling of cumulus-type clouds. The goal is to analyze the cloud's behavior over the
timespan of an hour, which corresponds to the typical lifetime of a cloud.

In my particular case, I was part of the RIS Team at LAAS-CNRS under the supervision of
Simon Lacroix.

## PDF File of the Master's thesis done with LaTex

[Master's Thesis](http://bit.ly/2hXzuco)


## Code

### Requirements for Python 3.5:

+ numpy>=1.10
+ scipy>=0.17.1
+ netCDF4>=1.2.4
+ matplotlib >= 1.5.1
+ [GPy](https://github.com/SheffieldML/GPy) >= 1.0.7
+ [libgp](https://github.com/mblum/libgp)

Other requirements:
+ Mencoder (For the animations)
+ Increase limit of simultaneous open files-> `~$ ulimit -Sn 4096`

###Modules:

####mesonh_atm.mesonh_atmosphere

####modules.cloud :


###Implementation Scripts

1. animation_clouds.py

Animate cross-sections of a given variable wrt time, e.g. Vertical Wind

![alt text][logo]

[logo]: https://github.com/DiegoSelle/master_thesis/blob/master/example_results/example_animation.gif

2.  cloud_exploration.py

![alt text][logo]

[logo]: https://github.com/DiegoSelle/master_thesis/blob/master/example_results/cloud1_cs1.png

![alt text][logo]

[logo]:https://github.com/DiegoSelle/master_thesis/blob/master/example_results/variograms_cloud1_new.png

3.  cloud_exploration_polar.py

![alt text][logo]

[logo]:
