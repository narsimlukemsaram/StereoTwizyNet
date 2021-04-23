# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.

@page dwx_stereo_disparity_sample Stereo Disparity Sample
@tableofcontents

@section dwx_stereo_disparity_description Description

The Stereo Disparity sample demonstrates the stereo pipeline.

The sample reads frames from two rectified stereo videos. It then runs the
frames through the stereo pipeline and displays a confidence map and final stereo
output.

@section dwx_stereo_disparity_running Running the Sample

The command line for the sample is:

    ./sample_stereo_disparity --video0=[path/to/the/left/video]
                              --video1=[path/to/the/right/video]
                              --level=[0|1|2|3] 
                              --single_side=[0|1]

where

    --video0=[path/to/the/left/video]
        Is the path to the video recorded from the left camera.
        Default value: path/to/data/samples/stereo/left_1.h264

    --video1=[path/to/the/right/video]
        Is the path to the video recorded from the right camera.
        Default value: path/to/data/samples/stereo/right_1.h264

    --level=[0|1|2|3] 
        Defines the pyramid level to display the disparity, depends on the number of levels.
        Default value: 1

    --single_side=[0|1]
        If `--single_side` is 0, the sample computes left and right stereo images
        and performs complete stereo pipeline. Otherwise, it computes only the left
        image and approximates occlusions by thresholding the confidence map.
        Default value: 0

It is possible to use keyboard input to change parameters at runtime:

    0-6: changes the level of refinement (0 no refinement)
    Q,W: changes the gain to the color code for visualizaion
    O  : toggles occlusions
    K  : infills occlusions (only if on)
    +,-: changes invalidy threshold (appears as white pixels)
    I  : toggle horizontal infill of invalidity pixels

@section dwx_stereo_disparity_output Output

The sample creates a window and displays:

- Top: Anaglyph of left and right image
- Bottom: Stereo images

The stereo output is color coded for clarity and some pixels are masked if they
are occluded or invalid. 

![stereo disparity](sample_stereo_disparity.png)

@section dwx_stereo_disparity_more Additional information

For more details see @ref stereo_mainsection . 
