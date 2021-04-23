# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.


# EXPERIMENTAL SAMPLE
# project(sample_experimental_template C CXX)

# PUBLIC SAMPLE
project(StereoPilotNet C CXX)

#-------------------------------------------------------------------------------
# Project files
#-------------------------------------------------------------------------------
set(PUBLIC_DOCS
    README.md
)

set(SOURCES
    InputHandler.hpp
    StereoRectifier.hpp
    StereoDisparity.hpp    
    FreeSpaceDetector.hpp
    DriveNet.hpp
    Disparity2Depth.hpp
    StereoPilotNet.hpp
    main.cpp
)

set(LIBRARIES
    dw_samples_framework
    dw_samples_stereo_common
    ${Driveworks_LIBRARIES}
)

#-------------------------------------------------------------------------------
# Final target
#-------------------------------------------------------------------------------
cuda_add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBRARIES})
set_property(TARGET ${PROJECT_NAME} PROPERTY FOLDER "Samples")

# ------------------------------------------------------------------------------
# Install target - UNCOMMENT FOR INSTALLATION
# ------------------------------------------------------------------------------
sdk_add_sample(${PROJECT_NAME})
