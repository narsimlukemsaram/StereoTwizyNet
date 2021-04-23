#ifndef DRIVE_NET_BOX_TRACKER_HPP
#define DRIVE_NET_BOX_TRACKER_HPP

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

// Tracker
#include <dw/features/Features.h>
#include <dw/features/BoxTracker2D.h>


class DriveNetBoxTracker
{
    private:
        // ------------------------------------------------
        // Driveworks Context and SAL
        // ------------------------------------------------
        dwContextHandle_t m_sdk = DW_NULL_HANDLE;
        const uint32_t m_maxDetections = 1000U;       

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
        //dwBoxTracker2DHandle_t m_boxTracker;
        std::vector<dwBoxTracker2DHandle_t> m_boxTracker;
        std::vector<float32_t> m_previousFeatureLocations;
        //std::vector<std::vector<float32_t>> m_previousFeatureLocations;
        std::vector<float32_t> m_currentFeatureLocations;
        //std::vector<std::vector<float32_t>> m_currentFeatureLocations;
        std::vector<dwFeatureStatus> m_featureStatuses;        
        cudaStream_t m_cudaStream = 0;
        
        
        dwImageProperties m_rcbProperties;

        // image width and height
        uint32_t m_imageWidth;
        uint32_t m_imageHeight;
        bool m_isRaw;

        bool initDriveNetBT();
        void runTracker(const dwImageCUDA*);
        uint32_t trackFeatures(const dwImageCUDA* );
        void processResults();
        uint32_t updateFeatureLocationsStatuses();
        const std::string& getArgument(const char* name) const;

        bool _first_frame = true;
        dwImageCUDA* planeY, *inputImage;
        dwImageHandle_t planeImage, inputImageHandle;
        dwImageProperties imagePropsRUINT8;


    public:
        // Labels of each class
        std::vector<std::string> m_classLabels;
        // Vectors of boxes and class label ids
        std::vector<std::vector<dwRectf>> m_dnnBoxList;
        std::vector<std::vector<std::string>> m_dnnLabelList;
        std::vector<std::vector<const char*>> m_dnnLabelListPtr;
        std::vector<const dwTrackedBox2D *> m_trackedBoxes ;
        std::vector<size_t> m_numTrackedBoxes;
        //std::vector<dwRectf> m_trackedBoxListFloat;
        std::vector<std::vector<dwRectf>> m_trackedBoxListFloat;

        void setContext(dwContextHandle_t  _m_context)
        {
            m_sdk =_m_context;
        }

        void setImageProperties(dwImageProperties _m_cameraImageProperties)
        {
            //m_cameraImageProperties = _m_cameraImageProperties;
            m_rcbProperties = _m_cameraImageProperties;
            m_imageWidth    = m_rcbProperties.width;
            m_imageHeight   = m_rcbProperties.height; 
        }
        /// Initialize input source and rectifier object
        bool Initialize( );             
        void Process(dwImageCUDA*, std::vector<std::string>, std::vector<std::vector<dwBox2D>>, std::vector<std::vector<const char*>>);    
        void Reset();

        DriveNetBoxTracker();
        ~DriveNetBoxTracker();
};

bool DriveNetBoxTracker::Initialize()
{    

    return initDriveNetBT();

}


bool DriveNetBoxTracker::initDriveNetBT()
{
    //------------------------------------------------------------------------------
    // Initialize Feature Tracker
    //------------------------------------------------------------------------------
    {
        m_maxFeatureCount = 4000;
        m_historyCapacity = 10;
        dwFeatureTrackerConfig featureTrackerConfig{};
        dwFeatureTracker_initDefaultParams(&featureTrackerConfig);
        featureTrackerConfig.cellSize                   = 32;//
        featureTrackerConfig.numEvenDistributionPerCell = 5;//
        featureTrackerConfig.imageWidth                 = m_imageWidth;//
        featureTrackerConfig.imageHeight                = m_imageHeight;//
        featureTrackerConfig.detectorScoreThreshold     = 0.1f;//0.0004f
        featureTrackerConfig.windowSizeLK               = 8;//
        featureTrackerConfig.iterationsLK               = 10;//
        featureTrackerConfig.detectorDetailThreshold    = 0.5f;//0.5f 0.005f
        featureTrackerConfig.maxFeatureCount            = m_maxFeatureCount;

        {
            CHECK_DW_ERROR(dwFeatureTracker_initialize(&m_featureTracker, &featureTrackerConfig, m_cudaStream,
                                                    m_sdk));
        }

        // Tracker pyramid init
        dwPyramidConfig pyramidConfig{};
        pyramidConfig.width      = m_imageWidth;
        pyramidConfig.height     = m_imageHeight;
        pyramidConfig.levelCount = 10;
        pyramidConfig.dataType   = DW_TYPE_UINT8;
        {
            CHECK_DW_ERROR(dwPyramid_initialize(&m_pyramidPrevious, &pyramidConfig, m_cudaStream, m_sdk));
        }
        {
            CHECK_DW_ERROR(dwPyramid_initialize(&m_pyramidCurrent, &pyramidConfig, m_cudaStream, m_sdk));
        }
        {
            CHECK_DW_ERROR(dwFeatureList_initialize(&m_featureList, featureTrackerConfig.maxFeatureCount,
                                                m_historyCapacity, m_imageWidth, m_imageHeight,
                                                m_cudaStream, m_sdk));
        }

        void* tempDatabase;
        size_t featureDataSize;
        {
            CHECK_DW_ERROR(dwFeatureList_getDataBasePointer(&tempDatabase, &featureDataSize, m_featureList));
        }

        {
            m_featureDatabase.reset(new uint8_t[featureDataSize]);
        }
        {
            CHECK_DW_ERROR(dwFeatureList_getDataPointers(&m_featureData, m_featureDatabase.get(), m_featureList));
        }
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
        CHECK_CUDA_ERROR(cudaMemset(m_featureMask, 0, pitch * m_imageHeight /4));

        CHECK_DW_ERROR(dwFeatureTracker_setMask(m_featureMask, pitch, m_imageWidth, m_imageHeight,
                                                m_featureTracker));
    }
    //------------------------------------------------------------------------------
    // Initialize Box Tracker
    //------------------------------------------------------------------------------

    {
        // need to upgrade
        dwBoxTracker2DParams params{};
        dwBoxTracker2D_initParams(&params);
        params.maxBoxImageScale = .9f;//.9f
        params.minBoxImageScale = 0.0005f;//0.0005f
        params.similarityThreshold = 0;//1.002f .0000002f
        params.groupThreshold = 0.01f;//0.00001f
        //params.confThreshDiscard = 0.01f;
        params.maxBoxCount = m_maxDetections;
       
        //params.minMatchOverlap = .01f;

        //CHECK_DW_ERROR(dwBoxTracker2D_initialize(&m_boxTracker, &params, m_imageWidth, m_imageHeight,
        //                                                m_sdk));
        m_boxTracker.resize(5);
        for( int numofclass = 0; numofclass <5; ++numofclass) 
        {
            {
                CHECK_DW_ERROR(dwBoxTracker2D_initialize(&m_boxTracker[numofclass], &params, m_imageWidth, m_imageHeight,
                                                        m_sdk));
            }
            {
                CHECK_DW_ERROR(dwBoxTracker2D_enablePriorityTracking(true,m_boxTracker[numofclass]));    
            }                                        
        }
        // Reserve for storing feature locations and statuses in CPU

        m_currentFeatureLocations.reserve(2 * m_maxFeatureCount);
        m_previousFeatureLocations.reserve(2 * m_maxFeatureCount);
        m_featureStatuses.reserve(2 * m_maxFeatureCount);
        m_trackedBoxListFloat.resize(5);
        m_trackedBoxes.resize(5) ;
        m_numTrackedBoxes.resize(5);
        for(int i =0 ; i<5; ++i)
        {
             m_trackedBoxListFloat[i].reserve(m_maxDetections);
        }
    }

    imagePropsRUINT8 = m_rcbProperties;
    imagePropsRUINT8.type = DW_IMAGE_CUDA;
    imagePropsRUINT8.memoryLayout = DW_IMAGE_MEMORY_TYPE_PITCH ;//DW_IMAGE_MEMORY_TYPE_BLOCK DW_IMAGE_MEMORY_TYPE_PITCH
    imagePropsRUINT8.format = DW_IMAGE_FORMAT_R_UINT8;//DW_IMAGE_FORMAT_RGB_UINT8_PLANAR;
    imagePropsRUINT8.width = m_rcbProperties.width;
    imagePropsRUINT8.height = m_rcbProperties.height;

    CHECK_DW_ERROR(dwImage_create(&planeImage, imagePropsRUINT8, m_sdk)); 
    CHECK_DW_ERROR(dwImage_create(&inputImageHandle, m_rcbProperties, m_sdk)); 

    return true;
}


void DriveNetBoxTracker::Process(dwImageCUDA* rgbaImage, std::vector<std::string> _m_classLabels ,
                                     std::vector<std::vector<dwBox2D>> _m_dnnBox4track,
                                     std::vector<std::vector<const char*>> _m_dnnLabelListPtr)
{

   
    std::vector<std::vector<const char*>> anm = _m_dnnLabelListPtr;

    for (size_t classIdx = 0; classIdx < _m_classLabels.size(); classIdx++)
    {
        //std::cout<<"Tracker inside _m_dnnBox4track["<<classIdx<<"].size() = "<<_m_dnnBox4track[classIdx].size()<<"\n";

        //if (&_m_dnnBox4track[classIdx][0] != nullptr)
        for(int nboxes =0;nboxes<(int)_m_dnnBox4track[classIdx].size();++nboxes)
        {
            
            // add candidates to box tracker
            //CHECK_DW_ERROR(dwBoxTracker2D_add(_m_dnnBox4track[classIdx].data(),_m_dnnBox4track[classIdx].size(), m_boxTracker[classIdx]));
            //CHECK_DW_ERROR(dwBoxTracker2D_add(_m_detectedBoxList[classIdx].data(), _m_detectedBoxList[classIdx].size(), m_boxTracker[classIdx]));
            CHECK_DW_ERROR(dwBoxTracker2D_add(&_m_dnnBox4track[classIdx][nboxes],1, m_boxTracker[classIdx]));
        }
    }
        

    // track features
    uint32_t featureCount = trackFeatures(rgbaImage);

    for (size_t classIdx = 0; classIdx < _m_classLabels.size(); classIdx++)
    {

        // If this is not the first frame, update the features
        if (!_first_frame ) 
        {
            // update box features
            CHECK_DW_ERROR(dwBoxTracker2D_updateFeatures(m_previousFeatureLocations.data(),
                                                            m_featureStatuses.data(),
                                                            featureCount, m_boxTracker[classIdx]));
            
        }

        // Run box tracker
        CHECK_DW_ERROR(dwBoxTracker2D_track(m_currentFeatureLocations.data(), m_featureStatuses.data(),
                                            m_previousFeatureLocations.data(), m_boxTracker[classIdx]));

        // Get tracked boxes
        CHECK_DW_ERROR(dwBoxTracker2D_get(&m_trackedBoxes[classIdx], &m_numTrackedBoxes[classIdx], m_boxTracker[classIdx]));
        
        // Extract boxes from tracked object list
        m_trackedBoxListFloat[classIdx].clear();
        for (uint32_t tIdx = 0U; tIdx < m_numTrackedBoxes[classIdx]; ++tIdx) 
        {
            const dwBox2D &box = m_trackedBoxes[classIdx][tIdx].box;
            dwRectf rectf;
            //rectf.x = static_cast<float32_t>(box.x);
            //rectf.y = static_cast<float32_t>(box.y);
            //rectf.width = static_cast<float32_t>(box.width);
            //rectf.height = static_cast<float32_t>(box.height);

            rectf.x = (box.x);
            rectf.y = (box.y);
            rectf.width = (box.width);
            rectf.height = (box.height);
            m_trackedBoxListFloat[classIdx].push_back(rectf);
        }
        //std::cout<<"Tracker inside m_trackedBoxListFloat["<<classIdx<<"].size() = "<<m_trackedBoxListFloat[classIdx].size()<<"\n";

        //if(m_trackedBoxes[classIdx][m_numTrackedBoxes[classIdx]-1].id > 1000)
        //    CHECK_DW_ERROR(dwBoxTracker2D_reset(m_boxTracker[classIdx]));
    }

    //std::cout<<"Tracker m_trackedBoxListFloat.size() = "<<m_trackedBoxListFloat.size()<<"\n";
    //std::cout<<"Tracker m_trackedBoxListFloat[0].size() = "<<m_trackedBoxListFloat[0].size()<<"\n";
    //std::cout<<"Tracker m_trackedBoxListFloat[1].size() = "<<m_trackedBoxListFloat[1].size()<<"\n";
    //std::cout<<"Tracker m_trackedBoxListFloat[2].size() = "<<m_trackedBoxListFloat[2].size()<<"\n";
    //std::cout<<"Tracker m_trackedBoxListFloat[3].size() = "<<m_trackedBoxListFloat[3].size()<<"\n";
    //std::cout<<"Tracker m_trackedBoxListFloat[4].size() = "<<m_trackedBoxListFloat[4].size()<<"\n";

   // std::cout<<"RENDER _m_trackedBoxes.size() = "<<_m_trackedBoxes.size()<<"\n";
    //std::cout<<"RENDER _m_trackedBoxes.size() = "<<_m_trackedBoxes.size()<<"\n";
    //std::cout<<"RENDER _m_numTrackedBoxes.size() = "<<_m_numTrackedBoxes.size()<<"\n";
   
    _first_frame = false;

}


// ------------------------------------------------
// Feature tracking
// ------------------------------------------------
uint32_t DriveNetBoxTracker::trackFeatures(const dwImageCUDA* image)
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
    //dwImageCUDA planeY{};
    {
        dwImage_getCUDA(&inputImage,inputImageHandle);
        inputImage->array[0] = image->array[0];
        inputImage->dptr[0] = image->dptr[0];
        

    }

    dwImage_getCUDA(&planeY,planeImage);
    {
        CHECK_DW_ERROR(dwImage_copyConvert(planeImage, inputImageHandle, m_sdk));
    }
    
    CHECK_DW_ERROR(dwPyramid_build(planeY, m_pyramidCurrent));

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
uint32_t DriveNetBoxTracker::updateFeatureLocationsStatuses()
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
    for (int classnum = 0; classnum<5; ++classnum)
    {
        for (uint32_t featureIdx = 0; featureIdx < newSize; featureIdx++) 
        {
            m_previousFeatureLocations.push_back(preLocations[featureIdx].x);
            m_previousFeatureLocations.push_back(preLocations[featureIdx].y);
            m_currentFeatureLocations.push_back(curLocations[featureIdx].x);
            m_currentFeatureLocations.push_back(curLocations[featureIdx].y);
            m_featureStatuses.push_back(m_featureData.statuses[featureIdx]);
        }
    }
    return newSize;
}





DriveNetBoxTracker::DriveNetBoxTracker(/* args */)
{
    m_boxTracker.resize(5); // for 5 different classes from dnn

}

DriveNetBoxTracker::~DriveNetBoxTracker()
{
   
    if (m_featureMask) 
    {
        CHECK_CUDA_ERROR(cudaFree(m_featureMask));
    }
   
    if (m_d_validFeatureCount) 
    {
        cudaFree(m_d_validFeatureCount);
    }
    if (m_d_validFeatureIndexes) 
    {
        cudaFree(m_d_validFeatureIndexes);
    }
    if (m_d_invalidFeatureCount) 
    {
        cudaFree(m_d_invalidFeatureCount);
    }
    if (m_d_invalidFeatureIndexes) 
    {
        cudaFree(m_d_invalidFeatureIndexes);
    }

    for(int i =0 ; i<5; ++i)
    {
        // Release box tracker
        CHECK_DW_ERROR(dwBoxTracker2D_release(&m_boxTracker[i]));
    }

    // Release feature tracker and list
    CHECK_DW_ERROR(dwFeatureList_release(&m_featureList));
    CHECK_DW_ERROR(dwFeatureTracker_release(&m_featureTracker));
    // Release pyramids
    CHECK_DW_ERROR(dwPyramid_release(&m_pyramidCurrent));
    CHECK_DW_ERROR(dwPyramid_release(&m_pyramidPrevious));
    CHECK_DW_ERROR(dwRelease(&m_sdk));
}

#endif
