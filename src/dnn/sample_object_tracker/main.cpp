/////////////////////////////////////////////////////////////////////////////////////////
// This code contains NVIDIA Confidential Information and is disclosed
// under the Mutual Non-Disclosure Agreement.
//
// Notice
// ALL NVIDIA DESIGN SPECIFICATIONS AND CODE ("MATERIALS") ARE PROVIDED "AS IS" NVIDIA MAKES
// NO REPRESENTATIONS, WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO
// THE MATERIALS, AND EXPRESSLY DISCLAIMS ANY IMPLIED WARRANTIES OF NONINFRINGEMENT,
// MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
//
// NVIDIA Corporation assumes no responsibility for the consequences of use of such
// information or for any infringement of patents or other rights of third parties that may
// result from its use. No license is granted by implication or otherwise under any patent
// or patent rights of NVIDIA Corporation. No third party distribution is allowed unless
// expressly authorized by NVIDIA.  Details are subject to change without notice.
// This code supersedes and replaces all information previously supplied.
// NVIDIA Corporation products are not authorized for use as critical
// components in life support devices or systems without express written approval of
// NVIDIA Corporation.
//
// Copyright (c) 2015-2018 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and proprietary
// rights in and to this software and related documentation and any modifications thereto.
// Any use, reproduction, disclosure or distribution of this software and related
// documentation without an express license agreement from NVIDIA Corporation is
// strictly prohibited.
//
/////////////////////////////////////////////////////////////////////////////////////////

// Samples
#include <framework/DriveWorksSample.hpp>
#include <framework/SimpleStreamer.hpp>
#include <framework/SimpleCamera.hpp>
#include <framework/Checks.hpp>

// Core
#include <dw/core/Context.h>
#include <dw/core/Logger.h>
#include <dw/core/VersionCurrent.h>

// HAL
#include <dw/sensors/Sensors.h>

// DNN
#include <dw/dnn/DNN.h>

// Renderer
#include <dw/renderer/RenderEngine.h>

// Tracker
#include <dw/features/Features.h>
#include <dw/features/BoxTracker2D.h>

using namespace dw_samples::common;

class ObjectTrackerApp : public DriveWorksSample
{
private:
    // ------------------------------------------------
    // Driveworks Context and SAL
    // ------------------------------------------------
    dwContextHandle_t m_sdk = DW_NULL_HANDLE;
    dwSALHandle_t m_sal     = DW_NULL_HANDLE;

    // ------------------------------------------------
    // DNN
    // ------------------------------------------------
    typedef std::pair<dwRectf,float32_t> BBoxConf;
    static constexpr float32_t COVERAGE_THRESHOLD = 0.6f;
    const uint32_t m_maxDetections = 1000U;
    const float32_t m_nonMaxSuppressionOverlapThreshold = 0.5;

    dwDNNHandle_t m_dnn = DW_NULL_HANDLE;
    dwDataConditionerHandle_t m_dataConditioner = DW_NULL_HANDLE;
    std::vector<dwBox2D> m_detectedBoxList;
    std::vector<dwRectf> m_detectedBoxListFloat;
    float32_t *m_dnnInputDevice;
    float32_t *m_dnnOutputsDevice[2];
    std::unique_ptr<float32_t[]> m_dnnOutputsHost[2];

    uint32_t m_cvgIdx;
    uint32_t m_bboxIdx;
    dwBlobSize m_networkInputDimensions;
    dwBlobSize m_networkOutputDimensions[2];

    uint32_t m_totalSizeInput;
    uint32_t m_totalSizesOutput[2];
    dwRect m_detectionRegion;

    // ------------------------------------------------
    // Feature Tracker
    // ------------------------------------------------
    uint32_t m_maxFeatureCount;
    uint32_t m_historyCapacity;
    dwFeatureTrackerHandle_t m_featureTracker = DW_NULL_HANDLE;
    dwFeatureListHandle_t m_featureList       = DW_NULL_HANDLE;
    dwPyramidHandle_t m_pyramidPrevious       = DW_NULL_HANDLE;
    dwPyramidHandle_t m_pyramidCurrent        = DW_NULL_HANDLE;
    std::unique_ptr<uint8_t[]> m_featureDatabase;
    dwFeatureListPointers m_featureData;
    uint32_t* m_d_validFeatureCount;
    uint32_t* m_d_validFeatureIndexes;
    uint32_t* m_d_invalidFeatureCount;
    uint32_t* m_d_invalidFeatureIndexes;
    uint8_t *m_featureMask;

    // ------------------------------------------------
    // Box Tracker
    // ------------------------------------------------
    dwBoxTracker2DHandle_t m_boxTracker;
    std::vector<float32_t> m_previousFeatureLocations;
    std::vector<float32_t> m_currentFeatureLocations;
    std::vector<dwFeatureStatus> m_featureStatuses;
    const dwTrackedBox2D * m_trackedBoxes = nullptr;
    size_t m_numTrackedBoxes  = 0;
    std::vector<dwRectf> m_trackedBoxListFloat;

    // ------------------------------------------------
    // Renderer
    // ------------------------------------------------
    dwRenderEngineHandle_t m_renderEngine = DW_NULL_HANDLE;
    dwImageHandle_t m_imageRGBA;
    std::unique_ptr<SimpleImageStreamer<>> m_streamerCUDA2GL;
    cudaStream_t m_cudaStream = 0;
    const dwVector4f m_colorLightBlue{DW_RENDERER_COLOR_LIGHTBLUE[0],
                                      DW_RENDERER_COLOR_LIGHTBLUE[1],
                                      DW_RENDERER_COLOR_LIGHTBLUE[2],
                                      DW_RENDERER_COLOR_LIGHTBLUE[3]};
    const dwVector4f m_colorYellow{DW_RENDERER_COLOR_YELLOW[0],
                                   DW_RENDERER_COLOR_YELLOW[1],
                                   DW_RENDERER_COLOR_YELLOW[2],
                                   DW_RENDERER_COLOR_YELLOW[3]};

    // ------------------------------------------------
    // Camera
    // ------------------------------------------------
    std::unique_ptr<SimpleCamera> m_camera;
    dwImageGL* m_imgGl;
    dwImageProperties m_rcbProperties;

    // image width and height
    uint32_t m_imageWidth;
    uint32_t m_imageHeight;
    bool m_isRaw;

public:
    /// -----------------------------
    /// Initialize application
    /// -----------------------------
    ObjectTrackerApp(const ProgramArguments& args)
        : DriveWorksSample(args) {}

    /// -----------------------------
    /// Initialize modules
    /// -----------------------------
    bool onInitialize() override
    {
        // -----------------------------------------
        // Initialize DriveWorks SDK context and SAL
        // -----------------------------------------
        {
            // initialize logger to print verbose message on console in color
            CHECK_DW_ERROR(dwLogger_initialize(getConsoleLoggerCallback(true)));
            CHECK_DW_ERROR(dwLogger_setLogLevel(DW_LOG_VERBOSE));

            // initialize SDK context, using data folder
            dwContextParameters sdkParams = {};
            sdkParams.dataPath            = DataPath::get_cstr();

#ifdef VIBRANTE
            sdkParams.eglDisplay = getEGLDisplay();
#endif

            CHECK_DW_ERROR(dwInitialize(&m_sdk, DW_VERSION, &sdkParams));
            CHECK_DW_ERROR(dwSAL_initialize(&m_sal, m_sdk));
        }

        //------------------------------------------------------------------------------
        // initialize Sensors
        //------------------------------------------------------------------------------
        {
            dwSensorParams params;
            {
#ifdef VIBRANTE
                if (getArgument("input-type").compare("camera") == 0)
                {
                    std::string parameterString = "camera-type=" + getArgument("camera-type");
                    parameterString += ",csi-port=" + getArgument("csi-port");
                    parameterString += ",slave=" + getArgument("slave");
                    parameterString += ",serialize=false,output-format=raw,camera-count=4";
                    std::string cameraMask[4] = {"0001", "0010", "0100", "1000"};
                    uint32_t cameraIdx        = std::stoi(getArgument("camera-index"));
                    if (cameraIdx < 0 || cameraIdx > 3)
                    {
                        std::cerr << "Error: camera index must be 0, 1, 2 or 3" << std::endl;
                        return false;
                    }
                    parameterString += ",camera-mask=" + cameraMask[cameraIdx];

                    params.parameters = parameterString.c_str();
                    params.protocol   = "camera.gmsl";

                    m_camera.reset(new RawSimpleCamera(params, m_sal, m_sdk, m_cudaStream, DW_CAMERA_OUTPUT_NATIVE_PROCESSED));
                    m_isRaw = true;
                }
                else
#endif
                {
                    std::string parameterString = getArgs().parameterString();
                    params.parameters           = parameterString.c_str();
                    params.protocol             = "camera.virtual";

                    std::string videoFormat = getArgument("video");
                    std::size_t found       = videoFormat.find_last_of(".");

                    if (videoFormat.substr(found + 1).compare("h264") == 0)
                    {
                        m_camera.reset(new SimpleCamera(params, m_sal, m_sdk));
                        dwImageProperties outputProperties = m_camera->getOutputProperties();
                        outputProperties.type              = DW_IMAGE_CUDA;
                        m_camera->setOutputProperties(outputProperties);
                        m_isRaw = false;
                    }
                    else
                    {
                        m_camera.reset(new RawSimpleCamera(params, m_sal, m_sdk, m_cudaStream,
                                                           DW_CAMERA_OUTPUT_NATIVE_PROCESSED));
                    }
                }
            }

            if (m_camera == nullptr)
            {
                logError("Camera could not be created\n");
                return false;
            }

#ifdef VIBRANTE
            if (getArgument("input-type").compare("camera") == 0)
            {
                dwCameraRawFormat rawFormat = m_camera->getCameraProperties().rawFormat;
                if (rawFormat != DW_CAMERA_RAW_FORMAT_RCCB &&
                    rawFormat != DW_CAMERA_RAW_FORMAT_BCCR &&
                    rawFormat != DW_CAMERA_RAW_FORMAT_CRBC &&
                    rawFormat != DW_CAMERA_RAW_FORMAT_CBRC)
                {
                    logError("Camera not supported, see documentation\n");
                    return false;
                }
            }
#endif

            std::cout << "Camera image with " << m_camera->getCameraProperties().resolution.x << "x"
                      << m_camera->getCameraProperties().resolution.y << " at "
                      << m_camera->getCameraProperties().framerate << " FPS" << std::endl;

            dwImageProperties displayProperties = m_camera->getOutputProperties();
            displayProperties.format            = DW_IMAGE_FORMAT_RGBA_UINT8;

            CHECK_DW_ERROR(dwImage_create(&m_imageRGBA, displayProperties, m_sdk));

            m_streamerCUDA2GL.reset(new SimpleImageStreamer<>(displayProperties, DW_IMAGE_GL, 1000, m_sdk));

            m_rcbProperties = m_camera->getOutputProperties();

            m_imageWidth  = displayProperties.width;
            m_imageHeight = displayProperties.height;
        }

        //------------------------------------------------------------------------------
        // initialize Renderer
        //------------------------------------------------------------------------------
        {
            // Setup render engine
            dwRenderEngineParams params{};
            CHECK_DW_ERROR(dwRenderEngine_initDefaultParams(&params, getWindowWidth(), getWindowHeight()));
            params.defaultTile.lineWidth = 0.2f;
            params.defaultTile.font      = DW_RENDER_ENGINE_FONT_VERDANA_20;
            CHECK_DW_ERROR(dwRenderEngine_initialize(&m_renderEngine, &params, m_sdk));
        }

        //------------------------------------------------------------------------------
        // initialize DNN
        //------------------------------------------------------------------------------
        {
            // If not specified, load the correct network based on platform
            std::string tensorRTModel = getArgument("tensorRT_model");
            if (tensorRTModel.empty()) {
                tensorRTModel = DataPath::get() + "/samples/detector/";
                tensorRTModel += getPlatformPrefix();
                tensorRTModel += "/tensorRT_model.bin";
            }

            // Initialize DNN from a TensorRT file
            CHECK_DW_ERROR(dwDNN_initializeTensorRTFromFile(&m_dnn, tensorRTModel.c_str(), m_sdk));

            CHECK_DW_ERROR(dwDNN_setCUDAStream(m_cudaStream, m_dnn));

            // Get input and output dimensions
            CHECK_DW_ERROR(dwDNN_getInputSize(&m_networkInputDimensions, 0U, m_dnn));
            CHECK_DW_ERROR(dwDNN_getOutputSize(&m_networkOutputDimensions[0], 0U, m_dnn));
            CHECK_DW_ERROR(dwDNN_getOutputSize(&m_networkOutputDimensions[1], 1U, m_dnn));

            auto getTotalSize = [] (const dwBlobSize &blobSize) {
                return blobSize.channels * blobSize.height * blobSize.width;
            };

            // Calculate total size needed to store input and output
            m_totalSizeInput = getTotalSize(m_networkInputDimensions);
            m_totalSizesOutput[0] = getTotalSize(m_networkOutputDimensions[0]);
            m_totalSizesOutput[1] = getTotalSize(m_networkOutputDimensions[1]);

            // Get coverage and bounding box blob indices
            const char *coverageBlobName = "coverage";
            const char *boundingBoxBlobName = "bboxes";
            CHECK_DW_ERROR(dwDNN_getOutputIndex(&m_cvgIdx, coverageBlobName, m_dnn));
            CHECK_DW_ERROR(dwDNN_getOutputIndex(&m_bboxIdx, boundingBoxBlobName, m_dnn));

            // Allocate GPU memory
            CHECK_CUDA_ERROR(cudaMalloc((void **)&m_dnnInputDevice, sizeof(float32_t) * m_totalSizeInput));
            CHECK_CUDA_ERROR(cudaMalloc((void **)&m_dnnOutputsDevice[0],
                             sizeof(float32_t) * m_totalSizesOutput[0]));
            CHECK_CUDA_ERROR(cudaMalloc((void **)&m_dnnOutputsDevice[1],
                             sizeof(float32_t) * m_totalSizesOutput[1]));
            // Allocate CPU memory for reading the output of DNN
            m_dnnOutputsHost[0].reset(new float32_t[m_totalSizesOutput[0]]);
            m_dnnOutputsHost[1].reset(new float32_t[m_totalSizesOutput[1]]);

            // Get metadata from DNN module
            // DNN loads metadata automatically from json file stored next to the dnn model,
            // with the same name but additional .json extension if present.
            // Otherwise, the metadata will be filled with default values and the dataconditioner parameters
            // should be filled manually.
            dwDNNMetaData metadata;
            CHECK_DW_ERROR(dwDNN_getMetaData(&metadata, m_dnn));

            // Initialie data conditioner
            CHECK_DW_ERROR(dwDataConditioner_initialize(&m_dataConditioner, &m_networkInputDimensions,
                                                        &metadata.dataConditionerParams, m_cudaStream,
                                                        m_sdk));

            // Reserve space for detected objects
            m_detectedBoxList.reserve(m_maxDetections);
            m_detectedBoxListFloat.reserve(m_maxDetections);

            // Detection region
            m_detectionRegion.width = std::min(static_cast<uint32_t>(m_networkInputDimensions.width),
                                               m_imageWidth);
            m_detectionRegion.height = std::min(static_cast<uint32_t>(m_networkInputDimensions.height),
                                                m_imageHeight);
            m_detectionRegion.x = (m_imageWidth - m_detectionRegion.width) / 2;
            m_detectionRegion.y = (m_imageHeight - m_detectionRegion.height) / 2;
       }

        //------------------------------------------------------------------------------
        // Initialize Feature Tracker
        //------------------------------------------------------------------------------
        {
            m_maxFeatureCount = 4000;
            m_historyCapacity = 10;
            dwFeatureTrackerConfig featureTrackerConfig{};
            dwFeatureTracker_initDefaultParams(&featureTrackerConfig);
            featureTrackerConfig.cellSize                   = 32;
            featureTrackerConfig.numEvenDistributionPerCell = 5;
            featureTrackerConfig.imageWidth                 = m_imageWidth;
            featureTrackerConfig.imageHeight                = m_imageHeight;
            featureTrackerConfig.detectorScoreThreshold     = 0.0004f;
            featureTrackerConfig.windowSizeLK               = 8;
            featureTrackerConfig.iterationsLK               = 10;
            featureTrackerConfig.detectorScoreThreshold     = 0.01f;
            featureTrackerConfig.detectorDetailThreshold    = 0.5f;
            featureTrackerConfig.maxFeatureCount            = m_maxFeatureCount;
            CHECK_DW_ERROR(dwFeatureTracker_initialize(&m_featureTracker, &featureTrackerConfig, m_cudaStream,
                                                       m_sdk));

            // Tracker pyramid init
            dwPyramidConfig pyramidConfig{};
            pyramidConfig.width      = m_imageWidth;
            pyramidConfig.height     = m_imageHeight;
            pyramidConfig.levelCount = 6;
            pyramidConfig.dataType   = DW_TYPE_UINT8;
            CHECK_DW_ERROR(dwPyramid_initialize(&m_pyramidPrevious, &pyramidConfig, m_cudaStream, m_sdk));
            CHECK_DW_ERROR(dwPyramid_initialize(&m_pyramidCurrent, &pyramidConfig, m_cudaStream, m_sdk));
            CHECK_DW_ERROR(dwFeatureList_initialize(&m_featureList, featureTrackerConfig.maxFeatureCount,
                                                    m_historyCapacity, m_imageWidth, m_imageHeight,
                                                    m_cudaStream, m_sdk));

            void* tempDatabase;
            size_t featureDataSize;
            CHECK_DW_ERROR(dwFeatureList_getDataBasePointer(&tempDatabase, &featureDataSize, m_featureList));

            m_featureDatabase.reset(new uint8_t[featureDataSize]);
            CHECK_DW_ERROR(dwFeatureList_getDataPointers(&m_featureData, m_featureDatabase.get(), m_featureList));

            CHECK_CUDA_ERROR(cudaMalloc(&m_d_validFeatureCount, sizeof(uint32_t)));
            CHECK_CUDA_ERROR(cudaMalloc(&m_d_validFeatureIndexes,
                                        featureTrackerConfig.maxFeatureCount * sizeof(uint32_t)));
            CHECK_CUDA_ERROR(cudaMalloc(&m_d_invalidFeatureCount, sizeof(uint32_t)));
            CHECK_CUDA_ERROR(cudaMalloc(&m_d_invalidFeatureIndexes,
                                        featureTrackerConfig.maxFeatureCount * sizeof(uint32_t)));

            // Set up mask. Apply feature tracking only to half of the image
            size_t pitch = 0;
            CHECK_CUDA_ERROR(cudaMallocPitch(&m_featureMask, &pitch, m_imageWidth, m_imageHeight));
            CHECK_CUDA_ERROR(cudaMemset(m_featureMask, 255, pitch * m_imageHeight));
            CHECK_CUDA_ERROR(cudaMemset(m_featureMask, 0, pitch * m_imageHeight / 2));

            CHECK_DW_ERROR(dwFeatureTracker_setMask(m_featureMask, pitch, m_imageWidth, m_imageHeight,
                                                    m_featureTracker));
        }

        //------------------------------------------------------------------------------
        // Initialize Box Tracker
        //------------------------------------------------------------------------------
        {
            dwBoxTracker2DParams params{};
            dwBoxTracker2D_initParams(&params);
            params.maxBoxImageScale = 0.5f;
            params.minBoxImageScale = 0.005f;
            params.similarityThreshold = 0.2f;
            params.groupThreshold = 2.0f;
            params.maxBoxCount = m_maxDetections;
            CHECK_DW_ERROR(dwBoxTracker2D_initialize(&m_boxTracker, &params, m_imageWidth, m_imageHeight,
                                                     m_sdk));
            // Reserve for storing feature locations and statuses in CPU
            m_currentFeatureLocations.reserve(2 * m_maxFeatureCount);
            m_previousFeatureLocations.reserve(2 * m_maxFeatureCount);
            m_featureStatuses.reserve(2 * m_maxFeatureCount);
            m_trackedBoxListFloat.reserve(m_maxDetections);
        }

        return true;
    }

    ///------------------------------------------------------------------------------
    /// Main processing of the sample
    ///     - collect sensor frame
    ///     - run detection and tracking
    ///------------------------------------------------------------------------------
    void onProcess() override
    {
        // read from camera
        dwImageCUDA* yuvImage = nullptr;
        getNextFrame(&yuvImage, &m_imgGl);
        std::this_thread::yield();
        while (yuvImage == nullptr)
        {
            onReset();

            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            getNextFrame(&yuvImage, &m_imgGl);
        }

        // Run data conditioner to prepare input for the network
        dwImageCUDA *rgbaImage;
        CHECK_DW_ERROR(dwImage_getCUDA(&rgbaImage, m_imageRGBA));
        CHECK_DW_ERROR(dwDataConditioner_prepareData(m_dnnInputDevice, &rgbaImage, 1, &m_detectionRegion,
                                                     cudaAddressModeClamp, m_dataConditioner));
        // Run DNN on the output of data conditioner
        CHECK_DW_ERROR(dwDNN_infer(m_dnnOutputsDevice, &m_dnnInputDevice, m_dnn));

        // Copy output back
        CHECK_CUDA_ERROR(cudaMemcpy(m_dnnOutputsHost[0].get(), m_dnnOutputsDevice[0],
                         sizeof(float32_t) * m_totalSizesOutput[0], cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(m_dnnOutputsHost[1].get(), m_dnnOutputsDevice[1],
                         sizeof(float32_t) * m_totalSizesOutput[1], cudaMemcpyDeviceToHost));

        // Interpret output blobs to extract detected boxes
        interpretOutput(m_dnnOutputsHost[m_cvgIdx].get(), m_dnnOutputsHost[m_bboxIdx].get(),
                        &m_detectionRegion);

        // Track objects
        runTracker(yuvImage);
    }

    ///------------------------------------------------------------------------------
    /// Render sample output on screen
    ///     - render video
    ///     - render boxes with labels
    ///------------------------------------------------------------------------------
    void onRender() override
    {
        CHECK_DW_ERROR(dwRenderEngine_setTile(0, m_renderEngine));
        CHECK_DW_ERROR(dwRenderEngine_resetTile(m_renderEngine));

        dwVector2f range{};
        range.x = m_imgGl->prop.width;
        range.y = m_imgGl->prop.height;
        CHECK_DW_ERROR(dwRenderEngine_setCoordinateRange2D(range, m_renderEngine));
        CHECK_DW_ERROR(dwRenderEngine_renderImage2D(m_imgGl, {0.0f, 0.0f, range.x, range.y}, m_renderEngine));

        // Render detection region
        dwRectf detectionRegionFloat;
        detectionRegionFloat.x = m_detectionRegion.x;
        detectionRegionFloat.y = m_detectionRegion.y;
        detectionRegionFloat.width = m_detectionRegion.width;
        detectionRegionFloat.height = m_detectionRegion.height;

        CHECK_DW_ERROR(dwRenderEngine_setColor(DW_RENDER_ENGINE_COLOR_YELLOW, m_renderEngine));
        CHECK_DW_ERROR(dwRenderEngine_render(DW_RENDER_ENGINE_PRIMITIVE_TYPE_BOXES_2D, &detectionRegionFloat,
                                             sizeof(dwRectf), 0, 1, m_renderEngine));


        CHECK_DW_ERROR(dwRenderEngine_setColor(DW_RENDER_ENGINE_COLOR_RED, m_renderEngine));
        CHECK_DW_ERROR(dwRenderEngine_setLineWidth(2.0f, m_renderEngine));
        CHECK_DW_ERROR(dwRenderEngine_render(DW_RENDER_ENGINE_PRIMITIVE_TYPE_BOXES_2D,
                                             m_trackedBoxListFloat.data(), sizeof(dwRectf), 0,
                                             m_trackedBoxListFloat.size(), m_renderEngine));

        CHECK_DW_ERROR(dwRenderEngine_setPointSize(2.0f, m_renderEngine));
        // Draw tracked features that belong to detected objects
        for (uint32_t boxIdx = 0U; boxIdx < m_numTrackedBoxes; ++boxIdx) {
            const dwTrackedBox2D &trackedBox = m_trackedBoxes[boxIdx];
            dwRenderEngine_render(DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_2D,
                                  trackedBox.featureLocations, sizeof(dwVector2f),
                                  0, trackedBox.nFeatures, m_renderEngine);
            // Render box id
            dwVector2f pos{static_cast<float32_t>(trackedBox.box.x),
                           static_cast<float32_t>(trackedBox.box.y)};
            dwRenderEngine_renderText2D(std::to_string(trackedBox.id).c_str(), pos, m_renderEngine);
        }
    }

    ///------------------------------------------------------------------------------
    /// Free up used memory here
    ///------------------------------------------------------------------------------
    void onRelease() override
    {
        // Free GPU memory
        if (m_dnnOutputsDevice[0]) {
            CHECK_CUDA_ERROR(cudaFree(m_dnnOutputsDevice[0]));
        }
        if (m_dnnOutputsDevice[1]) {
            CHECK_CUDA_ERROR(cudaFree(m_dnnOutputsDevice[1]));
        }
        if (m_featureMask) {
            CHECK_CUDA_ERROR(cudaFree(m_featureMask));
        }
        if (m_imageRGBA) {
            CHECK_DW_ERROR(dwImage_destroy(&m_imageRGBA));
        }
        if (m_d_validFeatureCount) {
            cudaFree(m_d_validFeatureCount);
        }
        if (m_d_validFeatureIndexes) {
            cudaFree(m_d_validFeatureIndexes);
        }
        if (m_d_invalidFeatureCount) {
            cudaFree(m_d_invalidFeatureCount);
        }
        if (m_d_invalidFeatureIndexes) {
            cudaFree(m_d_invalidFeatureIndexes);
        }

        // Release box tracker
        CHECK_DW_ERROR(dwBoxTracker2D_release(&m_boxTracker));
        // Release feature tracker and list
        CHECK_DW_ERROR(dwFeatureList_release(&m_featureList));
        CHECK_DW_ERROR(dwFeatureTracker_release(&m_featureTracker));
        // Release pyramids
        CHECK_DW_ERROR(dwPyramid_release(&m_pyramidCurrent));
        CHECK_DW_ERROR(dwPyramid_release(&m_pyramidPrevious));
        // Release detector
        CHECK_DW_ERROR(dwDNN_release(&m_dnn));
        // Release data conditioner
        CHECK_DW_ERROR(dwDataConditioner_release(&m_dataConditioner));
        // Release render engine
        CHECK_DW_ERROR(dwRenderEngine_release(&m_renderEngine));
        // Release camera
        m_camera.reset();

        // Release SDK
        CHECK_DW_ERROR(dwSAL_release(&m_sal));
        CHECK_DW_ERROR(dwRelease(&m_sdk));
    }

    ///------------------------------------------------------------------------------
    /// Reset tracker and detector
    ///------------------------------------------------------------------------------
    void onReset() override
    {
        CHECK_DW_ERROR(dwDNN_reset(m_dnn));
        CHECK_DW_ERROR(dwDataConditioner_reset(m_dataConditioner));
        CHECK_DW_ERROR(dwFeatureList_reset(m_featureList));
        CHECK_DW_ERROR(dwFeatureTracker_reset(m_featureTracker));
        CHECK_DW_ERROR(dwBoxTracker2D_reset(m_boxTracker));
    }

    ///------------------------------------------------------------------------------
    /// Change renderer properties when main rendering window is resized
    ///------------------------------------------------------------------------------
    void onResizeWindow(int width, int height) override
    {
        {
            CHECK_DW_ERROR(dwRenderEngine_reset(m_renderEngine));
            dwRectf rect;
            rect.width  = width;
            rect.height = height;
            rect.x      = 0;
            rect.y      = 0;
            CHECK_DW_ERROR(dwRenderEngine_setBounds(rect, m_renderEngine));
        }
    }

private:

    //------------------------------------------------------------------------------
    void getNextFrame(dwImageCUDA** nextFrameCUDA, dwImageGL** nextFrameGL)
    {
        dwImageHandle_t nextFrame = m_camera->readFrame();
        if (nextFrame == nullptr) {
            m_camera->resetCamera();
        } else {
            dwImage_getCUDA(nextFrameCUDA, nextFrame);
            CHECK_DW_ERROR(dwImage_copyConvert(m_imageRGBA, nextFrame, m_sdk));
            dwImageHandle_t frameGL = m_streamerCUDA2GL->post(m_imageRGBA);
            dwImage_getGL(nextFrameGL, frameGL);
        }
    }

    //------------------------------------------------------------------------------
    void interpretOutput(const float32_t *outConf, const float32_t *outBBox, const dwRect *const roi)
    {
        // Clear detection list
        m_detectedBoxList.clear();
        m_detectedBoxListFloat.clear();

        uint32_t numBBoxes = 0U;
        uint16_t gridH    = m_networkOutputDimensions[0].height;
        uint16_t gridW    = m_networkOutputDimensions[0].width;
        uint16_t cellSize = m_networkInputDimensions.height / gridH;
        uint32_t gridSize = gridH * gridW;

        for (uint16_t gridY = 0U; gridY < gridH; ++gridY) {
            const float32_t *outConfRow = &outConf[gridY * gridW];
            for (uint16_t gridX = 0U; gridX < gridW; ++gridX) {
                float32_t conf = outConfRow[gridX];
                if (conf > COVERAGE_THRESHOLD && numBBoxes < m_maxDetections) {
                    // This is a detection!
                    float32_t imageX = (float32_t)gridX * (float32_t)cellSize;
                    float32_t imageY = (float32_t)gridY * (float32_t)cellSize;
                    uint32_t offset  = gridY * gridW + gridX;

                    float32_t boxX1;
                    float32_t boxY1;
                    float32_t boxX2;
                    float32_t boxY2;

                    dwDataConditioner_outputPositionToInput(&boxX1, &boxY1, outBBox[offset] + imageX,
                                                            outBBox[gridSize + offset] + imageY, roi,
                                                            m_dataConditioner);
                    dwDataConditioner_outputPositionToInput(&boxX2, &boxY2,
                                                            outBBox[gridSize * 2 + offset] + imageX,
                                                            outBBox[gridSize * 3 + offset] + imageY, roi,
                                                            m_dataConditioner);
                    dwRectf bboxFloat{boxX1, boxY1, boxX2 - boxX1, boxY2 - boxY1};
                    dwBox2D bbox;
                    bbox.width = static_cast<int32_t>(std::round(bboxFloat.width));
                    bbox.height = static_cast<int32_t>(std::round(bboxFloat.height));
                    bbox.x = static_cast<int32_t>(std::round(bboxFloat.x));
                    bbox.y = static_cast<int32_t>(std::round(bboxFloat.y));

                    m_detectedBoxList.push_back(bbox);
                    m_detectedBoxListFloat.push_back(bboxFloat);
                    numBBoxes++;
                }
            }
        }
    }

    //------------------------------------------------------------------------------
    float32_t overlap(const dwRectf &boxA, const dwRectf &boxB)
    {

        int32_t overlapWidth = std::min(boxA.x + boxA.width,
                                        boxB.x + boxB.width) -
                               std::max(boxA.x, boxB.x);
        int32_t overlapHeight = std::min(boxA.y + boxA.height,
                                         boxB.y + boxB.height) -
                                std::max(boxA.y, boxB.y);

        return (overlapWidth < 0 || overlapHeight < 0) ? 0.0f : (overlapWidth * overlapHeight);
    }

    //------------------------------------------------------------------------------
    void runTracker(const dwImageCUDA* image)
    {
        // add candidates to box tracker
        CHECK_DW_ERROR(dwBoxTracker2D_add(m_detectedBoxList.data(), m_detectedBoxList.size(), m_boxTracker));

        // track features
        uint32_t featureCount = trackFeatures(image);

        // If this is not the first frame, update the features
        if (getFrameIndex() != 0) {
            // update box features
            CHECK_DW_ERROR(dwBoxTracker2D_updateFeatures(m_previousFeatureLocations.data(),
                                                         m_featureStatuses.data(),
                                                         featureCount, m_boxTracker));
        }

        // Run box tracker
        CHECK_DW_ERROR(dwBoxTracker2D_track(m_currentFeatureLocations.data(), m_featureStatuses.data(),
                                            m_previousFeatureLocations.data(), m_boxTracker));

        // Get tracked boxes
        CHECK_DW_ERROR(dwBoxTracker2D_get(&m_trackedBoxes, &m_numTrackedBoxes, m_boxTracker));

        // Extract boxes from tracked object list
        m_trackedBoxListFloat.clear();
        for (uint32_t tIdx = 0U; tIdx < m_numTrackedBoxes; ++tIdx) {
            const dwBox2D &box = m_trackedBoxes[tIdx].box;
            dwRectf rectf;
            rectf.x = static_cast<float32_t>(box.x);
            rectf.y = static_cast<float32_t>(box.y);
            rectf.width = static_cast<float32_t>(box.width);
            rectf.height = static_cast<float32_t>(box.height);
            m_trackedBoxListFloat.push_back(rectf);
        }
    }

    // ------------------------------------------------
    // Feature tracking
    // ------------------------------------------------
    uint32_t trackFeatures(const dwImageCUDA* image)
    {
        void* featureDatabaseDevice;
        size_t featureDataSize;
        dwFeatureListPointers featureListDevice;

        CHECK_DW_ERROR(dwFeatureList_getDataBasePointer(&featureDatabaseDevice, &featureDataSize,
                                                        m_featureList));
        CHECK_DW_ERROR(dwFeatureList_getDataPointers(&featureListDevice, featureDatabaseDevice,
                                                     m_featureList));

        std::swap(m_pyramidCurrent, m_pyramidPrevious);

        // slice 3-d image into one for feature computation/tracking
        dwImageCUDA planeY{};
        if (m_isRaw)
        {
            CHECK_DW_ERROR(dwImageCUDA_getPlaneAsImage(&planeY, image, 1)); //RCB (clear channel)
        }
        else
        {
            CHECK_DW_ERROR(dwImageCUDA_getPlaneAsImage(&planeY, image, 0)); //Yuv (Y: luminance channel)
        }

        // build pyramid
        CHECK_DW_ERROR(dwPyramid_build(&planeY, m_pyramidCurrent));

        // track features
        CHECK_DW_ERROR(dwFeatureTracker_trackFeatures(m_featureList, m_pyramidPrevious, m_pyramidCurrent, 0,
                                                      m_featureTracker));

        //Get feature info to CPU
        CHECK_CUDA_ERROR(cudaMemcpy(m_featureDatabase.get(), featureDatabaseDevice,
                                    featureDataSize, cudaMemcpyDeviceToHost));

        // Update feature locations after tracking
        uint32_t featureCount = updateFeatureLocationsStatuses();

        // apply proximity filters to make features uniformly distributed
        CHECK_DW_ERROR(dwFeatureList_proximityFilter(m_featureList));

        // make features compact
        CHECK_DW_ERROR(dwFeatureList_selectValid(m_d_validFeatureCount, m_d_validFeatureIndexes,
                                                 m_d_invalidFeatureCount, m_d_invalidFeatureIndexes,
                                                 m_featureList));
        CHECK_DW_ERROR(dwFeatureList_compact(m_featureList, m_d_validFeatureCount, m_d_validFeatureIndexes,
                                             m_d_invalidFeatureCount, m_d_invalidFeatureIndexes));

        // detect new features
        CHECK_DW_ERROR(dwFeatureTracker_detectNewFeatures(m_featureList, m_pyramidCurrent, m_featureTracker));

        return featureCount;
    }

    // ------------------------------------------------
    uint32_t updateFeatureLocationsStatuses()
    {
        uint32_t currentTimeIdx;
        CHECK_DW_ERROR(dwFeatureList_getCurrentTimeIdx(&currentTimeIdx, m_featureList));
        uint32_t previousTimeIdx = (currentTimeIdx + 1) % m_historyCapacity;
        // Get previous locations and update box tracker
        dwVector2f *preLocations = &m_featureData.locationHistory[previousTimeIdx * m_maxFeatureCount];
        dwVector2f *curLocations = &m_featureData.locationHistory[currentTimeIdx * m_maxFeatureCount];
        uint32_t newSize = std::min(m_maxFeatureCount, *m_featureData.featureCount);

        m_previousFeatureLocations.clear();
        m_currentFeatureLocations.clear();
        m_featureStatuses.clear();
        for (uint32_t featureIdx = 0; featureIdx < newSize; featureIdx++) {
            m_previousFeatureLocations.push_back(preLocations[featureIdx].x);
            m_previousFeatureLocations.push_back(preLocations[featureIdx].y);
            m_currentFeatureLocations.push_back(curLocations[featureIdx].x);
            m_currentFeatureLocations.push_back(curLocations[featureIdx].y);

            m_featureStatuses.push_back(m_featureData.statuses[featureIdx]);
        }
        return newSize;
    }

    // ------------------------------------------------
    std::string getPlatformPrefix()
    {
        static const int32_t CUDA_VOLTA_COMPUTE_CAPABILITY = 7;

        std::string path;
        int32_t currentGPU;
        cudaDeviceProp gpuProp{};

        CHECK_DW_ERROR(dwContext_getGPUDeviceCurrent(&currentGPU, m_sdk));
        CHECK_DW_ERROR(dwContext_getGPUProperties(&gpuProp, currentGPU, m_sdk));

        path = "pascal";
        if (gpuProp.major == CUDA_VOLTA_COMPUTE_CAPABILITY)
            path = "volta";

        return path;
    }

};

int main(int argc, const char** argv)
{
    // -------------------
    // define all arguments used by the application
    ProgramArguments args(argc, argv,
                          {
#ifdef VIBRANTE
                              ProgramArguments::Option_t("camera-type", "ar0231-rccb-bae-sf3324", "camera gmsl type (see sample_sensors_info for all available camera types on this platform)"),
                              ProgramArguments::Option_t("csi-port", "a", "input port"),
                              ProgramArguments::Option_t("camera-index", "0", "camera index within the csi-port 0-3"),
                              ProgramArguments::Option_t("slave", "0", "activate slave mode for Tegra B"),
                              ProgramArguments::Option_t("input-type", "video", "input type either video or camera"),
#endif
                              ProgramArguments::Option_t("video", (DataPath::get() + "/samples/sfm/triangulation/video_0.h264").c_str(), "path to video"),
                              ProgramArguments::Option_t("tensorRT_model", "", (std::string("path to TensorRT model file. By default: ") + DataPath::get() + "/samples/detector/<gpu-architecture>/tensorRT_model.bin").c_str())},
                          "Object Tracker sample which detects and tracks cars.");

    ObjectTrackerApp app(args);
    app.initializeWindow("Object Tracker", 1280, 800, args.enabled("offscreen"));

    if (!args.enabled("offscreen"))
        app.setProcessRate(30);

    return app.run();
}
