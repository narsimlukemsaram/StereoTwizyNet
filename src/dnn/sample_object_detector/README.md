# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.

@page dwx_object_detector_sample Basic Object Detector Sample
@tableofcontents

@section dwx_object_detector_description Description

The Object Detector sample streams a H.264 or RAW video and runs DNN inference on each frame to
detect objects using NVIDIA<sup>&reg;</sup> TensorRT<sup>&tm;</sup> model.

The interpretation of the output of a network depends on the network design. In this sample,
2 output blobs (with `coverage` and `bboxes` as blob names) are interpreted as coverage and bounding boxes.

For a more sophisticated, higher-resolution, multi-class object detector, see @ref dwx_object_drivenet_sample .

@section dwx_object_detector_sample_running Running the Sample

The Object Detector sample, `sample_object_detector`, accepts the following optional parameters. If none are specified, the sample performs detections on a supplied pre-recorded video.

    ./sample_object_detector --input-type=[video|camera]
                             --video=[path/to/video]
                             --camera-type=[camera]
                             --csi-port=[a|b|c]
                             --camera-index=[0|1|2|3]
                             --slave=[0,1]
                             --tensorRT_model=[path/to/TensorRT/model]

Where:

    --input-type=[video|camera] 
            Defines if the input is from live camera or from a recorded video. 
            Live camera is only supported on On NVIDIA DRIVE platform.
            Default value: video
    
    --video=[path/to/video]
            Is the absolute or relative path of a raw or h264 recording.
            Only applicable if --input-type=video
            Default value: path/to/data/samples/sfm/triangulation/video_0.h264.

    --camera-type=[camera] 
            Is either an `ov10640` camera or a supported AR0231 `RCCB` sensor.
            Only applicable if --input-type=camera.
            Default value: ar0231-rccb-bae-sf3324

    --csi-port=[a|b|c]
            Is the port where the camera is connected to.
            Only applicable if --input-type=camera.
            Default value: a

    --camera-index=[0|1|2|3] 
            Indicates the camera index on the given port.
            Default value: 0

    --slave=[0|1]
            Setting this parameter to 1 when running the sample on Tegra B allows to access a camera that 
            is being used on Tegra A. Only applicable if --input-type=camera.
            Default value: 0

    --tensorRT_model=[path/to/TensorRT/model]
            Specifies the path to the TensorRT model file.
            The loaded network is expected to have a coverage output blob named "coverage" and a bounding box output blob named "bboxes".
            Default value: path/to/data/samples/detector/<gpu-architecture>/tensorRT_model.bin where <gpu-architecture> can be either Pascal or Volta.

@note This sample loads its DataConditioner parameters from DNN metadata. This metadata
can be provided to DNN module by placing the json file in the same directory as the model file
with json extension; i.e. `TensorRT_model_file.json`.
For an example of a DNN metadata file, see:

    data/samples/detector/pascal/tensorRT_model.bin.json

@subsection dwx_object_detector_sample_examples Examples

#### To run the sample on a video with a custom TensorRT network

    ./sample_object_detector --input-type=video --video=<video file.h264/raw> --tensorRT_model=<TensorRT model file>

#### To run the sample on a camera on NVIDIA DRIVE platforms with a custom TensorRT network

    ./sample_object_detector --input-type=camera --camera-type=<rccb_camera_type> --csi-port=<csi port> --camera-index=<camera idx on csi port> --tensorRT_model=<TensorRT model file>

where `<rccb_camera_type>` is a supported `RCCB` sensor.
See @ref supported_sensors for the list of supported cameras for each platform.

@section dwx_object_detector_sample_output Output

The sample creates a window, displays a video, and overlays bounding boxes for detected cars.
The yellow bounding box identifies the region that was given as an input to the DNN.

![Car detector on a H.264 stream](sample_object_detector.png)

@section dwx_object_detector_sample_more Additional Information

For more information, see @ref object_description1.
