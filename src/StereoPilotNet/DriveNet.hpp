#ifndef DRIVE_NET_HPP
#define DRIVE_NET_HPP

// Samples
#include <framework/DriveWorksSample.hpp>
#include <framework/SimpleStreamer.hpp>
#include <framework/SampleFramework.hpp>
#include <framework/SimpleCamera.hpp>
#include <framework/Checks.hpp>

// Core
#include <dw/core/Context.h>
#include <dw/core/Logger.h>
#include <dw/core/VersionCurrent.h>

// HAL
#include <dw/sensors/Sensors.h>

// DriveNet
#include <dw/dnn/DriveNet.h>
#include <dw/objectperception/camera/ObjectDetector.h>

// tracker
#include "DriveNetBoxTracker.hpp"


class DriveNet
{
    private:
        // ------------------------------------------------
        // Driveworks Context, SAL and render engine
        // ------------------------------------------------
        dwContextHandle_t m_context           = DW_NULL_HANDLE;
        
        // ------------------------------------------------
        // DriveNet
        // ------------------------------------------------
        uint32_t m_numImages          = 1U;
        dwDriveNetHandle_t m_driveNet = DW_NULL_HANDLE;
        dwDriveNetParams m_driveNetParams{};
        const dwDriveNetClass* m_driveNetClasses = nullptr;
        uint32_t m_numDriveNetClasses            = 0;
        const uint32_t m_maxProposalsPerClass    = 1000U;
        const uint32_t m_maxClustersPerClass     = 400U;
        // Detector
        dwObjectDetectorParams m_detectorParams{};
        dwObjectDetectorHandle_t m_driveNetDetector = DW_NULL_HANDLE;
        // Clustering
        dwObjectClusteringHandle_t* m_objectClusteringHandles = nullptr;
        
        dwRectf m_detectorROIs[2];

        cudaStream_t m_cudaStream = 0;
        
        dwImageProperties m_rcbProperties;

        /// The maximum number of output objects for a given bound output.
        static constexpr uint32_t MAX_OBJECT_OUTPUT_COUNT = 1000;

        dwObjectHandleList m_detectorOutput[DW_OBJECT_MAX_CLASSES];
        dwObjectHandleList m_clustererOutput[DW_OBJECT_MAX_CLASSES];

        std::unique_ptr<dwObjectHandle_t[]> m_detectorOutputObjects[DW_OBJECT_MAX_CLASSES];
        std::unique_ptr<dwObjectHandle_t[]> m_clustererOutputObjects[DW_OBJECT_MAX_CLASSES];
        ProgramArguments m_args;

        const dwImageCUDA* m_detectorInputImages[2];
        bool initDriveNet();
        void processResults();
        const std::string& getArgument(const char* name) const;


    public:
        // Labels of each class
        std::vector<std::string> m_classLabels;
        // Vectors of boxes and class label ids
        std::vector<std::vector<dwRectf>> m_dnnBoxList;
        std::vector<std::vector<dwBox2D>> m_dnnBox4track;
        std::vector<std::vector<std::string>> m_dnnLabelList;
        std::vector<std::vector<const char*>> m_dnnLabelListPtr;
        DriveNetBoxTracker* driveNetBoxTracker;

        void setContext(dwContextHandle_t  _m_context)
        {
            m_context =_m_context;
        }


        void setArgsParam(ProgramArguments stPar)
        {
            //inputArgs = stPar;    
            m_args  = stPar;             
        }

        void setImageProperties(dwImageProperties _m_cameraImageProperties)
        {
            //m_cameraImageProperties = _m_cameraImageProperties;
            m_rcbProperties = _m_cameraImageProperties;
        }

        /// Initialize input source and rectifier object
        bool Initialize( );             
        void Process(dwImageCUDA*);    
        void Reset();

        DriveNet();
        ~DriveNet();
};

bool DriveNet::Initialize()
{
    driveNetBoxTracker = new DriveNetBoxTracker();
    driveNetBoxTracker->setContext(m_context);
    //driveNetBoxTracker->setArgsParam(m_args);    
    driveNetBoxTracker->setImageProperties(m_rcbProperties);   
    driveNetBoxTracker->Initialize();
    return initDriveNet();
}

void DriveNet::Process(dwImageCUDA* rcbImage)
{
   
    // detect objects and get the results
    m_detectorInputImages[0] = rcbImage;
    m_detectorInputImages[1] = rcbImage;
    CHECK_DW_ERROR(dwObjectDetector_processDeviceAsync(m_driveNetDetector));
    processResults();
    

    driveNetBoxTracker->Process(rcbImage, m_classLabels , m_dnnBox4track, m_dnnLabelListPtr);
}

bool DriveNet::initDriveNet()
{
    {
        // Check if foveal should be enabled
        bool fovealEnabled = getArgument("enableFoveal").compare("1") == 0;
        // If foveal is enabled, there will be two images: full resolution image and a cropped region
        // of the same image
        m_numImages = fovealEnabled ? 2U : 1U;

        // Initialize DriveNet network
        CHECK_DW_ERROR(dwDriveNet_initDefaultParams(&m_driveNetParams));
        // Set up max number of proposals and clusters
        m_driveNetParams.maxClustersPerClass  = m_maxClustersPerClass;
        m_driveNetParams.maxProposalsPerClass = m_maxProposalsPerClass;
        m_driveNetParams.networkModel         = DW_DRIVENET_MODEL_FRONT;
        m_driveNetParams.batchSize            = fovealEnabled ? DW_DRIVENET_BATCH_SIZE_2 : DW_DRIVENET_BATCH_SIZE_1;
        //m_driveNetParams.batchSize            = DW_DRIVENET_BATCH_SIZE_1;

        // Get precision from command line
        std::string precisionArg = getArgument("precision");
        if (precisionArg.compare("fp32") == 0)
        {
            m_driveNetParams.networkPrecision = DW_PRECISION_FP32;
        }
        else if (precisionArg.compare("fp16") == 0)
        {
            m_driveNetParams.networkPrecision = DW_PRECISION_FP16;
        }
        else if (precisionArg.compare("int8") == 0)
        {
            m_driveNetParams.networkPrecision = DW_PRECISION_INT8;
        }

        // Check if network should run on DLA
        bool dla = getArgument("dla").compare("1") == 0;

        // Check if this platform supports DLA
        if (dla)
        {
            int32_t dlaEngineCount = 0;
            CHECK_DW_ERROR(dwContext_getDLAEngineCount(&dlaEngineCount, m_context));
            if (!dlaEngineCount)
            {
                throw std::runtime_error("No DLA Engine available on this platform.");
            }

            // Check which DLA engine should DriveNet run on
            int32_t dlaEngineNo = std::atoi(getArgument("dlaEngineNo").c_str());
            CHECK_DW_ERROR(dwContext_selectDLAEngine(dlaEngineNo, m_context));

            m_driveNetParams.processorType = DW_PROCESSOR_TYPE_DLA;
            // DLA supports only FP16 precision.
            m_driveNetParams.networkPrecision = DW_PRECISION_FP16;
        }

        CHECK_DW_ERROR(dwDriveNet_initialize(&m_driveNet, &m_objectClusteringHandles, &m_driveNetClasses,
                                                &m_numDriveNetClasses, &m_driveNetParams, m_context));

        // Initialize Object Detector from DriveNet
        CHECK_DW_ERROR(dwObjectDetector_initDefaultParams(&m_detectorParams));
        // Enable fusing objects from different ROIs
        m_detectorParams.enableFuseObjects = true;
        // Two images will be given as input. Each image is a region on the image received from camera.
        m_detectorParams.maxNumImages = m_numImages;
        CHECK_DW_ERROR(dwObjectDetector_initializeFromDriveNet(&m_driveNetDetector, &m_detectorParams,
                                                                m_driveNet, m_context));
        CHECK_DW_ERROR(dwObjectDetector_setCUDAStream(m_cudaStream, m_driveNetDetector));

        // since our input images might have a different aspect ratio as the input to drivenet
        // we setup the ROI such that the crop happens from the top of the image
        float32_t aspectRatio = 1.0f;
        {
            dwBlobSize inputBlob;
            CHECK_DW_ERROR(dwDriveNet_getInputBlobsize(&inputBlob, m_driveNet));

            aspectRatio = static_cast<float32_t>(inputBlob.height) / static_cast<float32_t>(inputBlob.width);
        }

        // 1st image is a full resolution image as it comes out from the RawPipeline (cropped to DriveNet aspect ratio)
        dwRect fullROI;
        {
            fullROI = {0, 0, static_cast<int32_t>(m_rcbProperties.width),
                        static_cast<int32_t>(m_rcbProperties.width * aspectRatio)};
            dwTransformation2D transformation = {{1.0f, 0.0f, 0.0f,
                                                    0.0f, 1.0f, 0.0f,
                                                    0.0f, 0.0f, 1.0f}};

            CHECK_DW_ERROR(dwObjectDetector_setROI(0, &fullROI, &transformation, m_driveNetDetector));
        }

        // 2nd image is a cropped out region within the 1/4-3/4 of the original image in the center
        if (fovealEnabled)
        {
            dwRect ROI = {fullROI.width / 4, fullROI.height / 4,
                            fullROI.width / 2, fullROI.height / 2};
            dwTransformation2D transformation = {{1.0f, 0.0f, 0.0f,
                                                    0.0f, 1.0f, 0.0f,
                                                    0.0f, 0.0f, 1.0f}};

            CHECK_DW_ERROR(dwObjectDetector_setROI(1, &ROI, &transformation, m_driveNetDetector));
        }

        // fill out member structure according to the ROIs
        for (uint32_t roiIdx = 0U; roiIdx < m_numImages; ++roiIdx)
        {
            CHECK_DW_ERROR(dwObjectDetector_getROI(&m_detectorParams.ROIs[roiIdx],
                                                    &m_detectorParams.transformations[roiIdx], roiIdx, m_driveNetDetector));
            m_detectorROIs[roiIdx].x      = m_detectorParams.ROIs[roiIdx].x;
            m_detectorROIs[roiIdx].y      = m_detectorParams.ROIs[roiIdx].y;
            m_detectorROIs[roiIdx].width  = m_detectorParams.ROIs[roiIdx].width;
            m_detectorROIs[roiIdx].height = m_detectorParams.ROIs[roiIdx].height;
        }

        CHECK_DW_ERROR(dwObjectDetector_bindInput(m_detectorInputImages, m_numImages, m_driveNetDetector));

        for (uint32_t classIdx = 0; classIdx < m_numDriveNetClasses; ++classIdx)
        {
            m_detectorOutputObjects[classIdx].reset(new dwObjectHandle_t[MAX_OBJECT_OUTPUT_COUNT]);
            m_clustererOutputObjects[classIdx].reset(new dwObjectHandle_t[MAX_OBJECT_OUTPUT_COUNT]);

            // Initialize each object handle
            for (uint32_t objIdx = 0U; objIdx < MAX_OBJECT_OUTPUT_COUNT; ++objIdx)
            {
                dwObjectData objectData{};
                dwObjectDataCamera objectDataCamera{};
                CHECK_DW_ERROR(dwObject_createCamera(&m_detectorOutputObjects[classIdx][objIdx],
                                                        &objectData,
                                                        &objectDataCamera));
                CHECK_DW_ERROR(dwObject_createCamera(&m_clustererOutputObjects[classIdx][objIdx],
                                                        &objectData,
                                                        &objectDataCamera));
            }

            m_detectorOutput[classIdx].count    = 0;
            m_detectorOutput[classIdx].objects  = m_detectorOutputObjects[classIdx].get();
            m_detectorOutput[classIdx].maxCount = MAX_OBJECT_OUTPUT_COUNT;
            m_clustererOutput[classIdx].count   = 0;
            m_clustererOutput[classIdx].objects = m_clustererOutputObjects[classIdx].get();
            m_clustererOutput[classIdx].maxCount = MAX_OBJECT_OUTPUT_COUNT;

            CHECK_DW_ERROR(dwObjectDetector_bindOutput(&m_detectorOutput[classIdx], 0,
                                                        classIdx, m_driveNetDetector));

            CHECK_DW_ERROR(dwObjectClustering_bindInput(&m_detectorOutput[classIdx],
                                                        m_objectClusteringHandles[classIdx]));

            CHECK_DW_ERROR(dwObjectClustering_bindOutput(&m_clustererOutput[classIdx],
                                                            m_objectClusteringHandles[classIdx]));
        }

        // Initialize box list
        m_dnnBoxList.resize(m_numDriveNetClasses);
        m_dnnBox4track.resize(m_numDriveNetClasses);
        m_dnnLabelList.resize(m_numDriveNetClasses);
        m_dnnLabelListPtr.resize(m_numDriveNetClasses);

        // Get which label name for each class id
        m_classLabels.resize(m_numDriveNetClasses);
        for (uint32_t classIdx = 0U; classIdx < m_numDriveNetClasses; ++classIdx)
        {
            const char* classLabel;
            CHECK_DW_ERROR(dwDriveNet_getClassLabel(&classLabel, classIdx, m_driveNet));
            m_classLabels[classIdx] = classLabel;

            // Reserve label and box lists
            m_dnnBoxList[classIdx].reserve(m_maxClustersPerClass);
            m_dnnBox4track[classIdx].reserve(m_maxClustersPerClass);
            m_dnnLabelList[classIdx].reserve(m_maxClustersPerClass);
            m_dnnLabelListPtr[classIdx].reserve(m_maxClustersPerClass);
        }
    }

    return true;
}

void DriveNet::processResults()
{
    CHECK_DW_ERROR(dwObjectDetector_processHost(m_driveNetDetector));

    // for each detection class, we do
    for (uint32_t classIdx = 0U; classIdx < m_classLabels.size(); ++classIdx)
    {
        CHECK_DW_ERROR(dwObjectClustering_process(m_objectClusteringHandles[classIdx]));

        // Get outputs of object clustering
        m_dnnLabelListPtr[classIdx].clear();
        m_dnnLabelList[classIdx].clear();
        m_dnnBoxList[classIdx].clear();
        m_dnnBox4track[classIdx].clear();

        dwObjectHandleList clusters = m_clustererOutput[classIdx];
        for (uint32_t objIdx = 0U; objIdx < clusters.count; ++objIdx)
        {
            dwObjectHandle_t obj = clusters.objects[objIdx];
            dwObjectDataCamera objCameraData{};
            dwObject_getDataCamera(&objCameraData, 0, obj);
            
            m_dnnBoxList[classIdx].push_back(objCameraData.box2D);
            dwBox2D tempBox;
            tempBox.height = objCameraData.box2D.height;
            tempBox.width = objCameraData.box2D.width;
            tempBox.x = objCameraData.box2D.x;
            tempBox.y = objCameraData.box2D.y;
            m_dnnBox4track[classIdx].push_back(tempBox);

            std::string box_annotation = m_classLabels[classIdx];
            // This operation is safe because m_dnnLabelList is allocated using `reserve` at initialization
            // and it is not going to reallocate
            m_dnnLabelList[classIdx].push_back(box_annotation);
            m_dnnLabelListPtr[classIdx].push_back(m_dnnLabelList[classIdx].back().c_str());
        }
    }


}

DriveNet::DriveNet()
{
}

const std::string& DriveNet::getArgument(const char* name) const
{
    return m_args.get(name);
}

DriveNet::~DriveNet()
{
    // Release clustering
    for (uint32_t clsIdx = 0U; clsIdx < m_numDriveNetClasses; ++clsIdx)
    {
        CHECK_DW_ERROR(dwObjectClustering_release(&m_objectClusteringHandles[clsIdx]));
    }

    // Release detector
    CHECK_DW_ERROR(dwObjectDetector_release(&m_driveNetDetector));
    // Release drivenet
    CHECK_DW_ERROR(dwDriveNet_release(&m_driveNet));
    // Release object handles for detector and clusterer
    for (uint32_t classIdx = 0U; classIdx < m_numDriveNetClasses; ++classIdx)
    {
        for (uint32_t objIdx = 0U; objIdx < MAX_OBJECT_OUTPUT_COUNT; ++objIdx)
        {
            CHECK_DW_ERROR(dwObject_destroy(&m_detectorOutputObjects[classIdx][objIdx]));
            CHECK_DW_ERROR(dwObject_destroy(&m_clustererOutputObjects[classIdx][objIdx]));
        }
    }
    if (m_context != DW_NULL_HANDLE)
    {
        CHECK_DW_ERROR(dwRelease(&m_context));
    }
    delete(driveNetBoxTracker);

}




#endif