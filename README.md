# Coarse-to-Fine CloudMatch

Obs: This source code was produced for research and experimentation on rigid and non-rigid registration. It is not completly functional or documented.

These are the experiments and how to run them:

## Errors from KITTI sequence 

Assuming the following directory organization:

```
-- sequence #
 |-- 0
   |-- 0.txt
 |-- 1
   |-- 0.txt
   |-- 1.txt
   |-- groundlesss.txt
 |...
 ...
```

where, each individual object `0.txt`, `1.xt` is gotten following the KITTI GT (bounding boxes and position). `groundless.txt` is the entire point cloud with the groud removed.

The next step is to generate the transformation matrix for each onject. This can be done by calling:

```
python3 python/evaluate_rigid.py --mode=<tech name>
```

, where eval can be `[svr, filterreg, gmmtree, c2ICP, pmICP, trICP, SHOT_RANSAC]`. For `fitlerreg` and `gmmmtree`, in this specific case, we rely in a python implementation.

Calculated the transformations, the next step is to estimate the average error for each object. This is done by calling the same script, but with the flag `python3 python/evaluate_rigid.py --eval`.

## Estimation between 2 cloud files

To estimate the estimate the transformation between to distinct cloud and also visualize the result there are scripts at `./script/`, in special `seq.sh`. Depending on the technique desired, the `CONFIG.json` has to be updating in the field `KITTITECH`. 

## Segmenting objects from KITTI

To produce the expected segmentation use 

```
./build/apps/app/app -cubes -t <TRACKLET> -dir <DIR> -start <START> -end <END> -config <CONFIG> -seg-config <SEGMENTATION-CONFIG> -result-file <RESULT-FILE>
```

- `TRACKLET` - KITTI sequence `tracklet_labels.xml` file;
- `DIR` - KITTI sequence "velodyne_points" directory, it contains the `.bin` files;
- `START` - start file number, it is expected that files are in order;
- `END` - end file number, it is expected that files are in order;
- `CONFIG` and `SEGMENTATION-CONFIG`- explained below, with `GT_SEG` set to the segmentation option;
- `RESULT-FILE` - is ignored.

## Configuration files

### Config

```
{
    "normal_radius": 0.25, # Normal estimation search radius
    "keypointTECH": "UNIFORM", # Keypoints selection technique [HARRIS, UNIFORM]
    "keypoint_method": "HARRIS", # If HARRIS the feature can be [HARRIS, TOMASI, CURVATURE] 
    "keypoint_threshold": 1e-7, 
    "keypoint_radius": 0.1,


    "descriptorTECH": "NONE", # [SHOT, NONE] if SHOT the feature matching is used, otherwise local ICP
    "descriptor_radius": 0.4,

    "matchTECH": "SKIP", # Explained below
    "match_radius": 0.3, # Used for maximum distance on non-rigid matching 
    "match_icp": 1.2, # Rigid body maximum distance


    "patch_radius": 0.3, # PF non-rigid patch size
    "patch_min": 0.15, # Not used
    "patchmatch_levels": 2, # Not used

    # ICP parameters
    "ransac_max_iter": 10,
    "icp_max_iter": 5,
    "icp_transformation_eps": 0.000001,

    # PF parameters
    "number_init": 3,
    "sigma_s": 2,
    "sigma_r": 0.015,
    "sigma_d": 0.7,    
    "radius_pf": 0.15,

    # Sampling for uniform sampling, used for CloudMatch, PF, etc
    "uniform_radius": 0.6,

    # CPD Parameters
    "cpd_beta": 2,
    "cpd_lambda": 1.5,
    "cpd_max_iterations": 10,
    "cpd_tolerance": 0.001,

    "use_icp": true,
    "motion_bb": 0.0,
    "use_GT": false,
    "KITTITECH": "c2ICP",
    "focus_object": "All"
}
```

`matchTECH` controls whether to refine the registration using non-rigig estimation, it can be:
- `SKIP` - skips this steps and only consider rigid transformation;
- `NONE` - uses PF filter;
- `CPD` - non-rigid registration;
- `OPTIMIZATION` - Uses registration minimization followig approach proposed by Dewan.

### Segmentation Config

```
{
    # Not used
    "plane_iter": 100, 
    "plane_dist": 0.1,
    "outlier_threshold": 1.0,
    "outlier_k": 30,

    # KITTI ground removal params (used to removed ground)
    "ground_slope": 1.0,
    "ground_dist": 0.5,
    "ground_size": 2,
    
    # Not used
    "cluster_dist": 0.5,
    "cluster_angle": 1.0,
    "max_cluster": 10000,
    "min_cluster": 200,

    "segment_mode": "GT", # Explained below
    "segmented_dir": "/home/gustavo/fitler/segmented/91/" #Directory where sementation result will be stored.
}
```

`segment_mode` can be:
- `SEGMENT` - Used PCL segmentation (not tested thoroughly);
- `GROUND` - Registration between clouds, but removing ground before (not tested thoroughly);
- `GT` - Normal comparison
- `GT_SEG`- Normal comparison but saves each object and scene with no ground at `segmented_dr`.

# Non-Rigid Registration

## All cloud

``` 
./build/apps/app/app -vec -f1 <CLOUD1> -f2 <CLOUD2> -config <CONFIG> -seg-config <SEGMENTATION-CONFIG> -result-file <RESULT-FILE>
```

## Iteractive

``` 
./build/apps/app/app -ivec -f1 <CLOUD1> -f2 <CLOUD2> -config <CONFIG> -seg-config <SEGMENTATION-CONFIG> -result-file <RESULT-FILE>
```

### Selecting keypoints

In the iteractive mode it is possible to select keypoints by pressing \<shift> and clicking in the desired point.

### Checking vectors

In the resulting cloud, after executing (either iteractive or not), it is possible to check the vector values. By just pressing \<shift> and clicking in the desired point, the value is printed in the terminal.

### General note

In this example, the idea is to focus on the person walking in the cloud, in special the movement of the limbs. By calling, `patch` the `matchTECH` parameter is ignored.

## MIT Sequence

The script has details about the parameters.
http://people.csail.mit.edu/drdaniel/mesh_animation/#data

``` 
python3 python/mesh_eval.py
```

## KITTI Sequence

The `scripts/seq.sh` has also a template to perform this call. As before, it expected objects segmented.

```
./build/apps/app/app -cubes -t <TRACKLET> -dir <DIR> -start <START> -end <END> -config <CONFIG> -seg-config <SEGMENTATION-CONFIG> -result-file <RESULT-FILE>
```

- `TRACKLET` - KITTI sequence `tracklet_labels.xml` file;
- `DIR` - KITTI sequence "velodyne_points" directory, it contains the `.bin` files;
- `START` - start file number, it is expected that files are in order;
- `END` - end file number, it is expected that files are in order;
- `CONFIG` and `SEGMENTATION-CONFIG`- explained above;
- `RESULT-FILE` - directory where file containing sequence metrics is saved. The metric is the average distance between each point in the first cloud to the closest one, in the second.
