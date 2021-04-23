#ifndef FREE_SPACE_DETECTOR_HPP
#define FREE_SPACE_DETECTOR_HPP

// Sample
#include <framework/DriveWorksSample.hpp>
#include <framework/Checks.hpp>
#include <framework/DataPath.hpp>
#include <framework/SimpleCamera.hpp>
#include <framework/Log.hpp>

// CORE
#include <dw/core/Context.h>
#include <dw/core/Logger.h>
#include <dw/core/VersionCurrent.h>

// IMAGE
#include <dw/image/ImageStreamer.h>

// FreespaceDetector
#include <dw/freespaceperception/FreespaceDetector.h>
#include <dw/dnn/OpenRoadNet.h>
#include <dw/objectperception/camera/ObjectDetector.h>


class FreeSpaceDetector
{
    private:
        bool initRigConfiguration();
        bool initFreeSpaceNN();
        const std::string& getArgument(const char* name) const;
        dwTransformation transformation;

   
    public:

        dwContextHandle_t m_context           = DW_NULL_HANDLE;
        dwRenderEngineHandle_t m_renderEngine = DW_NULL_HANDLE;
        dwFreespaceDetectorHandle_t m_freespaceDetector = DW_NULL_HANDLE;
        dwOpenRoadNetHandle_t m_openRoadNet           = DW_NULL_HANDLE;
        dwFreespaceDetection m_freespaceBoundary{};

        //dwImageGL* m_imgGL;
        dwImageProperties m_cameraImageProperties;
        dwRigConfigurationHandle_t m_rigConfig          = DW_NULL_HANDLE;
        dwCalibratedCameraHandle_t m_calibratedCam      = DW_NULL_HANDLE;
        ProgramArguments m_args;
        bool m_rig = true;
        uint32_t m_cameraWidth  = 0U;
        uint32_t m_cameraHeight = 0U;
        cudaStream_t m_cudaStream  = 0;
        uint32_t m_spatialSmoothFilterWidth = 5;
        float32_t m_temporalSmoothFactor = 0.5f;
        float32_t m_drawScaleX;
        float32_t m_drawScaleY;
        // dwImageCUDA* nextFrame;
        int x;
        std::unique_ptr<WindowBase> m_window;
        dwImageHandle_t m_imageRGBA;


        void setContext(dwContextHandle_t  _m_context)
        {
            m_context =_m_context;
        }


        void setArgsParam(ProgramArguments stPar)
        {
            //inputArgs = stPar;    
            m_args  = stPar;             
        }

        void setTransformation(dwTransformation _transformation)
        {
            transformation = _transformation;
        }

        void setCalibratedCameraHandle(dwCalibratedCameraHandle_t _m_calibratedCam)
        {
            m_calibratedCam = _m_calibratedCam;
        }

        void setImageProperties(dwImageProperties _m_cameraImageProperties)
        {
            m_cameraImageProperties = _m_cameraImageProperties;
        }

        dwFreespaceDetection* getFreeSpaceBoundary()
        {
            return &m_freespaceBoundary;
        }

        /// Initialize input source and rectifier object
        bool Initialize( );             
        void Process(dwImageCUDA*);    
        void Reset();
        FreeSpaceDetector();
        ~FreeSpaceDetector();
};

bool FreeSpaceDetector::Initialize( )
{
    
    m_cameraWidth  = m_cameraImageProperties.width;
    m_cameraHeight = m_cameraImageProperties.height;
    std::cout<<"m_cameraWidth = "<<m_cameraWidth<<"\n";
    std::cout<<"m_cameraHeight = "<<m_cameraHeight<<"\n";
    m_rig = initRigConfiguration();
    

	return initFreeSpaceNN();
}


FreeSpaceDetector::FreeSpaceDetector()
{
}

FreeSpaceDetector::~FreeSpaceDetector()
{
    if (m_imageRGBA)
	        {
	            dwImage_destroy(&m_imageRGBA);
	        }

    if (m_rigConfig)
        dwRigConfiguration_reset(m_rigConfig);

    if (m_freespaceDetector)
        dwFreespaceDetector_release(&m_freespaceDetector);

    if (m_openRoadNet)
        dwOpenRoadNet_release(&m_openRoadNet);
}

void FreeSpaceDetector::Process(dwImageCUDA* imageLeft)
{
    if (m_freespaceDetector)
    {

        dwStatus res = dwFreespaceDetector_processDeviceAsync(imageLeft, m_freespaceDetector);
        res = res == DW_SUCCESS ? dwFreespaceDetector_interpretHost(m_freespaceDetector) : res;

        if (res != DW_SUCCESS)
        {
            logError("Detector failed with: ", dwGetStatusName(res));
        }
        else
        {
            dwFreespaceDetector_getBoundaryDetection(&m_freespaceBoundary, m_freespaceDetector);
        }
    }
}


void FreeSpaceDetector::Reset()
{
	dwFreespaceDetector_reset(m_freespaceDetector);
	dwOpenRoadNet_reset(m_openRoadNet);

}



bool FreeSpaceDetector::initFreeSpaceNN()
{
	dwOpenRoadNetParams openRoadNetParams{};
    {
	    CHECK_DW_ERROR(dwOpenRoadNet_initDefaultParams(&openRoadNetParams));
    }

    openRoadNetParams.networkModel = DW_OPENROADNET_MODEL_FRONT;
	{
        CHECK_DW_ERROR(dwOpenRoadNet_initialize(&m_openRoadNet, m_context, &openRoadNetParams));
    }

	dwStatus res = DW_FAILURE;

    float32_t maxDistance = 50.0f;
    std::string maxDistanceStr = getArgument("maxDistance");
    if(maxDistanceStr!="50.0") {
        try{
            maxDistance = std::stof(maxDistanceStr);
            if (maxDistance < 0.0f) {
                logError("maxDistance cannot be negative.\n");
                return false;
            }
        } catch(...) {
            logError("Given maxDistance can't be parsed\n");
            return false;
        }
    }

    {
        CHECK_DW_ERROR(dwFreespaceDetector_initializeFromOpenRoadNet(&m_freespaceDetector,
                                                                        m_openRoadNet,
                                                                        m_cameraWidth, m_cameraHeight,
                                                                        m_cudaStream,
                                                                        m_context));
    }

    {
        CHECK_DW_ERROR(dwFreespaceDetector_setCameraHandle(m_calibratedCam, m_freespaceDetector));
    }

    {
        CHECK_DW_ERROR(dwFreespaceDetector_setCameraExtrinsics(transformation, m_freespaceDetector));
    }

    {
        CHECK_DW_ERROR(dwFreespaceDetector_setMaxFreespaceDistance(maxDistance, m_freespaceDetector));    
    }

    {
        res = dwFreespaceDetector_setSpatialSmoothFilterWidth(m_spatialSmoothFilterWidth, m_freespaceDetector);
    }

    if (res != DW_SUCCESS)
    {
        logError("Cannot set free space boundary spatial smooth filter: ", dwGetStatusName(res));
        return false;
    }

	return true;
}

bool FreeSpaceDetector::initRigConfiguration()
{
        dwStatus result = DW_SUCCESS;
        //Load vehicle configuration
        result = dwRigConfiguration_initializeFromFile(&m_rigConfig, m_context, getArgument("rigconfig").c_str());
        if (result != DW_SUCCESS) {
            logError("Error dwRigConfiguration_initialize: ", dwGetStatusName(result));
            return false;
        }
        result = dwRigConfiguration_initializeCalibratedCamera(&m_calibratedCam, 0, m_rigConfig);
        if (result != DW_SUCCESS) {
            logError("Error dwCameraRig_initializeFromConfig: ", dwGetStatusName(result));
            return false;
        }
        return true;
}


const std::string& FreeSpaceDetector::getArgument(const char* name) const
{
    return m_args.get(name);
}


#endif