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

using namespace dw_samples::common;

class ObjectDetectorApp : public DriveWorksSample
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
    static constexpr float32_t COVERAGE_THRESHOLD = 0.3f;
    const uint32_t m_maxDetections = 1000U;
    const float32_t m_nonMaxSuppressionOverlapThreshold = 0.5;

    dwDNNHandle_t m_dnn = DW_NULL_HANDLE;
    dwDataConditionerHandle_t m_dataConditioner = DW_NULL_HANDLE;
    std::vector<dwRectf> m_detectedBoxList;
    std::vector<BBoxConf> m_bboxConfList;
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
    ObjectDetectorApp(const ProgramArguments& args)
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

                    m_camera.reset(new RawSimpleCamera(DW_IMAGE_FORMAT_RGBA_UINT8,
                                                       params, m_sal, m_sdk, m_cudaStream,
                                                       DW_CAMERA_OUTPUT_NATIVE_PROCESSED));
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
                        outputProperties.format            = DW_IMAGE_FORMAT_RGBA_UINT8;
                        m_camera->setOutputProperties(outputProperties);
                        m_isRaw = false;
                    }
                    else
                    {
                        m_camera.reset(new RawSimpleCamera(DW_IMAGE_FORMAT_RGBA_UINT8,
                                                           params, m_sal, m_sdk, m_cudaStream,
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

            m_streamerCUDA2GL.reset(new SimpleImageStreamer<>(displayProperties, DW_IMAGE_GL, 1000, m_sdk));

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
            m_bboxConfList.reserve(m_maxDetections);

            // Detection region
            m_detectionRegion.width = std::min(static_cast<uint32_t>(m_networkInputDimensions.width),
                                               m_imageWidth);
            m_detectionRegion.height = std::min(static_cast<uint32_t>(m_networkInputDimensions.height),
                                                m_imageHeight);
            m_detectionRegion.x = (m_imageWidth - m_detectionRegion.width) / 2;
            m_detectionRegion.y = (m_imageHeight - m_detectionRegion.height) / 2;
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
        dwImageCUDA* rgbaImage = nullptr;
        getNextFrame(&rgbaImage, &m_imgGl);
        std::this_thread::yield();
        while (rgbaImage == nullptr)
        {
            onReset();

            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            getNextFrame(&rgbaImage, &m_imgGl);
        }

        // Run data conditioner to prepare input for the network
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

        CHECK_DW_ERROR(dwRenderEngine_setColor(DW_RENDER_ENGINE_COLOR_GREEN, m_renderEngine));
        CHECK_DW_ERROR(dwRenderEngine_setLineWidth(2.0f, m_renderEngine));
        CHECK_DW_ERROR(dwRenderEngine_render(DW_RENDER_ENGINE_PRIMITIVE_TYPE_BOXES_2D,
                                             m_detectedBoxList.data(), sizeof(dwRectf), 0,
                                             m_detectedBoxList.size(), m_renderEngine));
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
        dwImageHandle_t imageRGBA = m_camera->readFrame();
        if (imageRGBA == nullptr) {
            m_camera->resetCamera();
        } else {
            dwImage_getCUDA(nextFrameCUDA, imageRGBA);
            dwImageHandle_t frameGL = m_streamerCUDA2GL->post(imageRGBA);
            dwImage_getGL(nextFrameGL, frameGL);
        }
    }

    //------------------------------------------------------------------------------
    void nonMaxSuppression(float32_t overlapThresh)
    {
        // Sort boxes based on confidence
        std::sort(m_bboxConfList.begin(), m_bboxConfList.end(),
                  [](std::pair<dwRectf, float32_t> elem1,
                     std::pair<dwRectf, float32_t> elem2) -> bool {
            return elem1.second < elem2.second;
        });

        auto getArea = [] (const dwRectf &box) {
            return box.width * box.height;
        };

        for (auto objItr = m_bboxConfList.begin(); objItr != m_bboxConfList.end(); ++objItr) {
            const auto& objA = *objItr;
            bool keepObj = true;
            for (auto next = objItr + 1; next != m_bboxConfList.end(); ++next) {
                const auto& objB = *next;
                const dwRectf &objABox = objA.first;
                const dwRectf &objBBox = objB.first;

                float32_t objARight = objABox.x + objABox.width;
                float32_t objABottom = objABox.y + objABox.height;
                float32_t objBRight = objBBox.x + objBBox.width;
                float32_t objBBottom = objBBox.y + objBBox.height;

                float32_t ovl =  overlap(objABox, objBBox)
                    / std::min( getArea(objABox), getArea(objBBox) );

                bool is_new_box_inside_old_box = (objBBox.x > objABox.x) &&
                                                 (objBRight < objARight) &&
                                                 (objBBox.y > objABox.y) &&
                                                 (objBBottom < objABottom);

                if (ovl > overlapThresh || is_new_box_inside_old_box)
                    keepObj = false;
            }
            if (keepObj)
                m_detectedBoxList.push_back(objA.first);
        }
    }

    //------------------------------------------------------------------------------
    void interpretOutput(const float32_t *outConf, const float32_t *outBBox, const dwRect *const roi)
    {
        // Clear detection list
        m_detectedBoxList.clear();
        m_bboxConfList.clear();

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
                    dwRectf bbox;
                    bbox.width  = boxX2 - boxX1;
                    bbox.height = boxY2 - boxY1;
                    bbox.x = boxX1;
                    bbox.y = boxY1;

                    m_bboxConfList.push_back(std::make_pair(bbox, conf));
                    numBBoxes++;
                }
            }
        }

        // Merge overlapping bounding boxes by non-maximum suppression
        nonMaxSuppression(m_nonMaxSuppressionOverlapThreshold);
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
                          "Object Detector sample which detects cars.");

    ObjectDetectorApp app(args);
    app.initializeWindow("Object Detector", 1280, 800, args.enabled("offscreen"));

    if (!args.enabled("offscreen"))
        app.setProcessRate(30);

    return app.run();
}
