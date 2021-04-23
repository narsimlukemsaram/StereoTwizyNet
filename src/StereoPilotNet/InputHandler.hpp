#ifndef INPUT_HANDLER_HPP
#define INPUT_HANDLER_HPP


// Samples
#include <framework/DriveWorksSample.hpp>
#include <framework/Grid.hpp>
#include <framework/SimpleStreamer.hpp>
#include <framework/SampleFramework.hpp>
#include <framework/ProgramArguments.hpp>
#include <framework/Checks.hpp>

// Core
#include <dw/core/Context.h>
#include <dw/core/Logger.h>
#include <dw/core/VersionCurrent.h>

// HAL
#include <dw/sensors/Sensors.h>

// RCCB
#include <dw/isp/SoftISP.h>

// Stereo
#include "StereoRectifier.hpp"





using namespace dw_samples::common;

class InputHandler 
{


    private:

        dwContextHandle_t m_context           = DW_NULL_HANDLE;
        dwSALHandle_t m_sal                   = DW_NULL_HANDLE;

        ProgramArguments inputArgs;

        std::vector<cudaStream_t> m_cudaStream;

        static constexpr uint32_t MAX_VIDEOS  = 2;
        static constexpr uint32_t MAX_CAMERAS = 2;
        
        std::vector<dwImageCUDA*> imageArrayCameraCuda;        
        std::vector<dwImageGL*> imageArrayCameraGL;        
        std::vector<dwImageCUDA*> rcbImage;        
       
        std::vector<dwImageHandle_t> m_imagesRGBA;
        std::vector<std::unique_ptr<SimpleImageStreamer<>>> m_streamerCUDA2GLArray;

        // ------------------------------------------------
        // Camera
        // ------------------------------------------------

        dwImageProperties  cameraImageProps;
        
        std::vector<std::unique_ptr<SimpleCamera>> m_camera;
        std::vector<dwImageProperties> m_rcbProperties;        

        // image width and height
        std::vector<uint32_t> m_imageWidth;
        std::vector<uint32_t> m_imageHeight;
        bool m_isRaw;


    public:
        StereoRectifier* stereoRectifier;

        InputHandler()
        {
            // resize declared vectors
            imageArrayCameraCuda.resize(2);
            m_camera.resize(2);
            imageArrayCameraGL.resize(2);
            m_imagesRGBA.resize(2);
            m_camera.resize(2);
            m_streamerCUDA2GLArray.resize(2);
            m_rcbProperties.resize(2);
            m_imageWidth.resize(2);
            m_imageHeight.resize(2);
            m_cudaStream.resize(2);
            rcbImage.resize(2);
        }

        std::vector<dwImageCUDA*> getStereoRGBAImages()
        {
            return imageArrayCameraCuda;
        } 

        std::vector<dwImageGL*> getStereoRGBAGLImages()
        {
            return imageArrayCameraGL;
        } 

        void setContext(dwContextHandle_t  _m_context)
        {
            m_context =_m_context;
        }

        void setSalHandle(dwSALHandle_t  _m_sal)
        {
            m_sal =_m_sal;
        }

        void setArgsParam(ProgramArguments stPar)
        {
            inputArgs = stPar;                   
        }

        /// Initialize input source and rectifier object
        bool Initialize( );             
        ///------------------------------------------------------------------------------
        /// Free up used memory here
        ///------------------------------------------------------------------------------
        void Release(); 
        ///------------------------------------------------------------------------------
        /// Main processing of the sample collect sensor frames
        ///------------------------------------------------------------------------------
        void Process();    
        void Reset();
        ~InputHandler()
        {
            delete stereoRectifier;

        }
};

bool InputHandler::Initialize( ) 
{
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

                m_camera.reset(new RawSimpleCamera(params, m_sal, m_context, m_cudaStream, DW_CAMERA_OUTPUT_NATIVE_PROCESSED));
                m_isRaw = true;
            }
            else
            #endif
            {
                
                std::string videos = inputArgs.get("videos");
                int idx            = 0;
                int cam_count      = 0;

                while(true)
                {
                    size_t found = videos.find(",", idx);


                    std::string parameterString = "video=" + videos.substr(idx, found - idx);
                    params.parameters           = parameterString.c_str();
                    params.protocol             = "camera.virtual";

                    std::string videoFormat = videos.substr(idx, found - idx);
                    std::cout<<"videoFormat = "<<videoFormat<<"\n";

                    //std::string videoFormat = getArgument("video");
                    std::size_t founddot       = videoFormat.find_last_of(".");

                    if (videoFormat.substr(founddot + 1).compare("h264") == 0)
                    {
                        std::cout<<"videoFormat = "<<videoFormat<<"\n";
                        m_camera[cam_count].reset(new SimpleCamera(params, m_sal, m_context));
                        dwImageProperties outputProperties = m_camera[cam_count]->getOutputProperties();
                        outputProperties.type              = DW_IMAGE_CUDA;
                        outputProperties.format            = DW_IMAGE_FORMAT_RGB_FLOAT16_PLANAR;
                        m_camera[cam_count]->setOutputProperties(outputProperties);
                        m_isRaw = false;
                    }
                    else
                    {
                        dwImageProperties blockImageprop{};
                        blockImageprop.format = DW_IMAGE_FORMAT_RGBA_UINT8 ;//DW_IMAGE_FORMAT_RGBA_UINT8 DW_IMAGE_FORMAT_YUV420_UINT8_PLANAR
                        blockImageprop.height = 1208 ; // 1208 604
                        blockImageprop.width  = 1920 ; // 1920 960


                        m_camera[cam_count].reset(new RawSimpleCamera(blockImageprop,params, m_sal, m_context, m_cudaStream[cam_count],
                                                        DW_CAMERA_OUTPUT_NATIVE_PROCESSED, DW_SOFTISP_DEMOSAIC_METHOD_INTERPOLATION)); //DW_SOFTISP_DEMOSAIC_METHOD_INTERPOLATION 
                                                        //DW_CAMERA_OUTPUT_NATIVE_PROCESSED


                        //m_camera[cam_count].reset(new RawSimpleCamera(params, m_sal, m_context, m_cudaStream[cam_count],
                        //                                DW_CAMERA_OUTPUT_NATIVE_PROCESSED, DW_SOFTISP_DEMOSAIC_METHOD_INTERPOLATION)); //DW_SOFTISP_DEMOSAIC_METHOD_INTERPOLATION 
                                                        //DW_CAMERA_OUTPUT_NATIVE_PROCESSED

                    }
                    if (found == std::string::npos)
                        break;
                    idx = found + 1;
                    cam_count = cam_count+1;
                }
            }
        }

        if (m_camera[0] == nullptr || m_camera[1] == nullptr)
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
        {
            for (int camit =0;camit<2;++camit)
            {
                std::cout << "Camera image with " << m_camera[camit]->getCameraProperties().resolution.x << "x"
                        << m_camera[camit]->getCameraProperties().resolution.y << " at "
                        << m_camera[camit]->getCameraProperties().framerate << " FPS" << std::endl;

                dwImageProperties displayProperties = m_camera[camit]->getOutputProperties();
                displayProperties.type              = DW_IMAGE_CUDA;
                //displayProperties.memoryLayout      = DW_IMAGE_MEMORY_TYPE_BLOCK; //DW_IMAGE_MEMORY_TYPE_PITCH
                displayProperties.format            = DW_IMAGE_FORMAT_RGBA_UINT8;
                {    
                    CHECK_DW_ERROR(dwImage_create(&m_imagesRGBA[camit], displayProperties, m_context));
                }
                m_streamerCUDA2GLArray[camit].reset(new SimpleImageStreamer<>(displayProperties, DW_IMAGE_GL, 1000, m_context));
                m_rcbProperties[camit] = m_camera[camit]->getOutputProperties();
                m_imageWidth[camit]  = m_camera[camit]->getCameraProperties().resolution.x;
                m_imageHeight[camit] = m_camera[camit]->getCameraProperties().resolution.y;
            }
            {
                cameraImageProps = m_camera[0]->getImageProperties();
                std::cout<<"cameraImageProps.memoryLayout = "<<cameraImageProps.memoryLayout<<"\n";
                std::cout<<"cameraImageProps.format = "<<cameraImageProps.format<<"\n";
                std::cout<<"cameraImageProps.height = "<<cameraImageProps.height<<"\n";
            }
        }


    }

    {
        stereoRectifier = new StereoRectifier();
        stereoRectifier->setContext(m_context);
        stereoRectifier->setSalHandle(m_sal);
        stereoRectifier->setArgsParam(inputArgs);
        stereoRectifier->Initialize();
    }
    return true;
}

void InputHandler::Release() 
{
    for (int camit =0;camit<2;++camit)
    {
        if (m_imagesRGBA[camit])
        {
            dwImage_destroy(&m_imagesRGBA[camit]);
        }
        m_camera[camit].reset();
    }

    CHECK_DW_ERROR(dwSAL_release(&m_sal));
    if (m_context != DW_NULL_HANDLE)
    {
        CHECK_DW_ERROR(dwRelease(&m_context));
    }

    CHECK_DW_ERROR(dwLogger_release());        

}

void InputHandler::Process() 
{
    std::vector<dwImageHandle_t> _cameraFrame;
    _cameraFrame.resize(2);
    for (int camit =0;camit<2;++camit)
    {
        rcbImage[camit]         = nullptr;
        _cameraFrame[camit]     = nullptr;
    }
    // read from camera
    
    for (int camit =0;camit<2;++camit)
    {
        _cameraFrame[camit] = m_camera[camit]->readFrame();            
    }

    if (_cameraFrame[0] == nullptr || _cameraFrame[1] == nullptr)
    {
        m_camera[0]->resetCamera();
        m_camera[1]->resetCamera();
    }
    else
    {
        for (int camit =0;camit<2;++camit)
        {
            {
                dwImage_getCUDA(&imageArrayCameraCuda[camit], _cameraFrame[camit]);     
            }
            {           
                CHECK_DW_ERROR(dwImage_copyConvert(m_imagesRGBA[camit], _cameraFrame[camit], m_context));
            }
            
            dwImageHandle_t frameGL = m_streamerCUDA2GLArray[camit]->post(m_imagesRGBA[camit]);

            {
                dwImage_getGL(&imageArrayCameraGL[camit], frameGL);
            }
            //std::cout<<"Input handler imageArrayCameraCuda props = "<<imageArrayCameraCuda[camit]->prop.memoryLayout<<" "<<imageArrayCameraCuda[camit]->prop.format<<" "<<imageArrayCameraCuda[camit]->pitch[0]<<"\n";
            
        }

        {
            stereoRectifier->Process(imageArrayCameraCuda);        
        }
    }        
            
}

void InputHandler::Reset() 
{
    for (int camit =0;camit<2;++camit)
    {
        m_camera[camit].reset();
    }
} 

#endif