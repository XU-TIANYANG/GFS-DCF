# GFS-DCF
Matlab implementation of ICCV2019 paper "Joint Group Feature Selection and Discriminative Filter Learning for Robust Visual Object Tracking"

## [Download the Paper](https://www.researchgate.net/publication/334849529_Joint_Group_Feature_Selection_and_Discriminative_Filter_Learning_for_Robust_Visual_Object_Tracking)
>@article{xu2019joint,
  title={Joint Group Feature Selection and Discriminative Filter Learning for Robust Visual Object Tracking},
  author={Xu, Tianyang and Feng, Zhen-Hua and Wu, Xiao-Jun and Kittler, Josef},
  journal={arXiv preprint arXiv:1907.13242},
  year={2019}}

![image](https://github.com/XU-TIANYANG/GFS-DCF/blob/master/Fig.jpg)

Dependencies:
MatConvNet, PDollar Toolbox. 
Please download the latest MatConvNet (http://www.vlfeat.org/matconvnet/) 
in './tracker_exter/matconvnet' 
(Set 'opts.enableGpu = true' in 'matconvnet/matlab/vl_compilenn.m')

Installation:
Run install.m file to compile the libraries and download networks.

Demo:
Run demo.m 

Raw Results:
[OTB2015(OTB2013)](https://github.com/XU-TIANYANG/cakes/raw/master/GFSDCF_OTB_results.zip)
[TrackingNet Test](https://github.com/XU-TIANYANG/cakes/raw/master/GFSDCF_TrackingNet_results.zip)









