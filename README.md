## Repository of Diego Selle's Master's Thesis

This is a repository with sample code of the master's thesis of Diego Selle.
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
[Simon Lacroix](https://scholar.google.com/citations?user=7cgLDwUAAAAJ&hl=en&oi=ao).

## PDF File of the Master's thesis done with LaTex

[Master's Thesis](http://bit.ly/2hXzuco)


## Goal of the Thesis
The goal of the thesis was to determine prior knowledge of the stochastic process inside clouds,
in particular of the vertical wind. This would help improve the mapping via a Gaussian Process Regression(GPR),
for it would permit initial mappings of the wind field without the need for computationally
expensive hyperparameter optimizations at each sampling step to determine the GPR model.

This goal was achieved by inferring the hyperparameter distributions through offline variograms
(See chapter 2.3 of the thesis manuscript for the theory)
and also by determining rough trends of the behavior of the vertical wind field inside clouds.
Both of these analysis were done using realistic atmospheric simulations available to us
thanks to the contribution of the CNRM team.

## Data

Given its size (~3 TB) and also due to legal reasons, the data used during
my master's thesis will not be published. Nonetheless, some example code and its results
will be presented in this repository.

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
Includes inhouse developed modules prior to my time at RIS to access and
index the atmospheric simulation data.

####modules.cloud :
Functions and objects related to the analysis I needed to do during my thesis.
These include a cloud class whose objects are generated after a segmentation algorithm,
which gives the points that are part of the same cloud. This in turn permits the
computation of important geometrical information such as center of masses, volume or bounding boxes.

Moreover, it includes a function to compute variograms in 4 different directions and another function to fit
the variograms with models typical in literature such as the Matern models or the Squared Exponentials.

Furthermore, it was deemed necessary to analyze the data in its polar form and to this end functions
were implemented.

###Implementation Scripts

1. animation_clouds.py

Animate cross-sections of a given variable wrt time, e.g. Vertical Wind

![alt text][animation]

[animation]: https://github.com/DiegoSelle/master_thesis/blob/master/example_results/example_animation.gif

This script was particularly useful for finding bounding boxes for clouds when analyzing
the liquid water content variable.

2.  cloud_exploration.py

This script served to visualize clouds in its Cartesian grid-form. The following example pertains to
a z-cross-section of the vertical wind field:

![alt text][cross-section]

[cross-section]: https://github.com/DiegoSelle/master_thesis/blob/master/example_results/cloud1_cs1.png

Further visualizations of other geometric aspects of the cloud were created and can be reviewed in chapter
4.1 of the manuscript.

Moreover, the first variograms were calculated using the bounding box information of the cloud. The results are
akin to the following:

![alt text][variogram_cart]

[variogram_cart]:https://github.com/DiegoSelle/master_thesis/blob/master/example_results/variograms_cloud1_new.png

This clearly shows that the xy plane has similar values of the wind field near the edges of the cloud,
suggesting that a polar representation may be more adequate to the problem at hand. This idea can be directly
verified in the plot of the cross-section.

3.  cloud_exploration_polar.py

![alt text][polar_cs]

[polar_cs]:https://github.com/DiegoSelle/master_thesis/blob/master/example_results/cloud1_zwind_normalized.png

4. radial_trend_estimation.py

![alt text][radial_trend]

[radial_trend]:https://github.com/DiegoSelle/master_thesis/blob/master/example_results/radial_trend_cloud1_incl_out.png

5. radial_variograms.py

![alt text][variogram_rad]

[variogram_rad]:https://github.com/DiegoSelle/master_thesis/blob/master/example_results/variogram_cloud1_z115_zwc_432_t150.png

![alt text][variogram_fit]

[variogram_fit]: https://github.com/DiegoSelle/master_thesis/blob/master/example_results/median_exp_variograms_all_clouds.png

6. testGP.py

![alt text][testGP]

[testGP]: https://github.com/DiegoSelle/master_thesis/blob/master/example_results/comp_noise2.png
