#ifndef STEREO_RECTIFIER_HPP
#define STEREO_RECTIFIER_HPP

#define _CRT_SECURE_NO_WARNINGS

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

// Stereo
#include <dw/stereo/Stereo.h>
#include "stereo_common/stereoCommon.hpp"

#include <vector>
#include <stdio.h>
using namespace dw_samples::common;

class StereoRectifier 
{
    private:
        // stereo rectifier
        dwContextHandle_t m_context           = DW_NULL_HANDLE;
        dwSALHandle_t m_sal                   = DW_NULL_HANDLE;

        dwStereoHandle_t m_stereoAlgorithm;
        std::vector<dwCalibratedCameraHandle_t> m_cameraModel;
        dwStereoRectifierHandle_t m_stereoRectifier;
        dwRigConfigurationHandle_t m_rigConfiguration = DW_NULL_HANDLE;

        std::vector<std::unique_ptr<SimpleImageStreamer<>>> m_streamerCUDA2GLArray;
        std::unique_ptr<SimpleImageStreamer<>> m_streamerimageAnaglyphGL;


        dwImageProperties m_imageProperties{};
        std::vector<dwImageHandle_t> m_imagesRGBA;
        std::vector<dwImageCUDA*> rectifiedCuda;
        std::vector<dwImageGL*> rectifiedGL;
        dwImageGL* imageAnaglyphGL;
        std::vector<dwImageHandle_t> m_outputRectified;
        dwImageHandle_t m_outputAnaglyph;
        dwImageProperties imagePropsRUINT8, rectifiedImageProps;

        ProgramArguments inputArgs;
        dwBox2D m_roi;

        std::vector<dwImageCUDA*> planeImageCUDA;
        std::vector<dwImageHandle_t> planeImage;
        dwTransformation left2Rig, right2Rig;
        dwTransformation  qrMatrix;
        

    public:
        float cX                = 0.0;
        float cY                = 0.0;
        float fX                = 0.0;
        float baseline          = 0.0;// in m
        float rectfiedWidth     = 0.0;
        float rectfiedHeight    = 0.0;
        float cXL               = 0.0;
        float cXR               = 0.0;
        float focalLengthL      = 0.0;

        dwMatrix3f rectificationMatrixLeft, rectificationMatrixRight;
        dwMatrix34f projectionMatrixLeft,projectionMatrixRight;
        
        StereoRectifier()
        {
            m_cameraModel.resize(2);
            rectifiedCuda.resize(2);
            rectifiedGL.resize(2);
            m_imagesRGBA.resize(2);
            m_outputRectified.resize(2);
            m_streamerCUDA2GLArray.resize(2);
            m_imageProperties.type = DW_IMAGE_CUDA;
            m_imageProperties.format = DW_IMAGE_FORMAT_RGBA_UINT8;           
            planeImageCUDA.resize(2);
            planeImage.resize(2);  
        }

        void setContext(dwContextHandle_t  _m_context)
        {
            m_context =_m_context;
        }

        void setSalHandle(dwSALHandle_t  _m_sal)
        {
            m_sal =_m_sal;
        }

        void setImageHandle(std::vector<dwCalibratedCameraHandle_t>  _m_cameraModel)
        {
            m_cameraModel = _m_cameraModel;
        }

        dwCalibratedCameraHandle_t  getCalibratedCameraHandleLeft()
        {
            return m_cameraModel[0];
        }

        std::vector<dwImageCUDA*> getRectifiedCudaImages()
        {
            return rectifiedCuda;
        }

        std::vector<dwImageCUDA*> getRectifiedRCudaImages()
        {
            return planeImageCUDA;
        }

        std::vector<dwImageGL*> getRectifiedGLImages()
        {
            return rectifiedGL;
        }

        dwImageGL* getImageAnaglyphGL()
        {
            return imageAnaglyphGL;
        }

        dwImageProperties getRectifiedImageProperties()
        {
            return rectifiedImageProps;
        }

        std::vector<dwImageHandle_t> getRectifiedImageHandle()
        {
            return m_outputRectified;
        }

        dwTransformation getSensor2RigTransformation()
        {
            return left2Rig;
        }

        void setArgsParam(ProgramArguments stPar)
        {
            inputArgs = stPar;     
            
            if (inputArgs.get("rigconfig").empty()) 
            {
                throw std::runtime_error("Rig configuration file not specified, please provide a rig "
                                    "configuration file with the calibration of the stereo camera");
            }

            CHECK_DW_ERROR(dwRigConfiguration_initializeFromFile(&m_rigConfiguration, m_context, inputArgs.get("rigconfig").c_str()));
            {
                CHECK_DW_ERROR( dwRigConfiguration_initializeCalibratedCamera(&m_cameraModel[0], 0, m_rigConfiguration));
                CHECK_DW_ERROR( dwRigConfiguration_initializeCalibratedCamera(&m_cameraModel[1], 1, m_rigConfiguration));
            }
            uint32_t totalWidth = 0;
            for (int32_t i = 0; i < 2; ++i) 
            {
                dwPinholeCameraConfig pinholeConfig;
                CHECK_DW_ERROR(dwRigConfiguration_getPinholeCameraConfig(&pinholeConfig, i, m_rigConfiguration));
                m_imageProperties.height = pinholeConfig.height;
                m_imageProperties.width = pinholeConfig.width;
                if(i==0)
                {
                    cXL             = pinholeConfig.u0;
                    focalLengthL    = (pinholeConfig.focalX + pinholeConfig.focalX)/2;
                }
                if(i==1)
                    cXR = pinholeConfig.u0;

                totalWidth += pinholeConfig.width;
            }          
        }


        bool Initialize(); 
        void Process(std::vector<dwImageCUDA*> );        
        ~StereoRectifier();
};

bool StereoRectifier::Initialize() 
{
    //dwTransformation left2Rig, right2Rig;
    CHECK_DW_ERROR(dwRigConfiguration_getSensorToRigTransformation(&left2Rig, 0, m_rigConfiguration));
    CHECK_DW_ERROR(dwRigConfiguration_getSensorToRigTransformation(&right2Rig, 1,m_rigConfiguration));

    {
        CHECK_DW_ERROR(dwStereoRectifier_initialize(&m_stereoRectifier, m_cameraModel[0],m_cameraModel[1], left2Rig, right2Rig, m_context)); 
    }
    //container for output of the stereo rectifier, the image size is found in the CropROI
    {
        CHECK_DW_ERROR(dwStereoRectifier_getCropROI(&m_roi, m_stereoRectifier));
    }

    {
        rectifiedImageProps = m_imageProperties;
        rectifiedImageProps.type = DW_IMAGE_CUDA;
        rectifiedImageProps.memoryLayout = DW_IMAGE_MEMORY_TYPE_PITCH ;//DW_IMAGE_MEMORY_TYPE_BLOCK DW_IMAGE_MEMORY_TYPE_PITCH
        rectifiedImageProps.format = DW_IMAGE_FORMAT_RGBA_UINT8;
        rectifiedImageProps.width = m_roi.width;
        rectifiedImageProps.height = m_roi.height;
    }
    {
        imagePropsRUINT8 = m_imageProperties;
        imagePropsRUINT8.type = DW_IMAGE_CUDA;
        imagePropsRUINT8.memoryLayout = DW_IMAGE_MEMORY_TYPE_PITCH ;//DW_IMAGE_MEMORY_TYPE_BLOCK DW_IMAGE_MEMORY_TYPE_PITCH
        imagePropsRUINT8.format = DW_IMAGE_FORMAT_R_UINT8;//DW_IMAGE_FORMAT_RGB_UINT8_PLANAR;
        imagePropsRUINT8.width = m_roi.width;
        imagePropsRUINT8.height = m_roi.height;
    }
    
    for(uint32_t i=0; i<2; ++i) 
    {
        CHECK_DW_ERROR(dwImage_create(&m_outputRectified[i], rectifiedImageProps, m_context));
    }

    std::cout << "Rectified image: " << m_roi.width << "x" << m_roi.height << std::endl;
    

    {
        CHECK_DW_ERROR(dwStereoRectifier_getRectificationMatrix	(&rectificationMatrixLeft,DW_STEREO_SIDE_LEFT,m_stereoRectifier));
    }
    //std::cout<<"LEFT \n"<<rRectMat.array[0]<<" "<<rRectMat.array[1]<<" "<<rRectMat.array[2]<<"\n"
     //       <<" "<<rRectMat.array[3]<<" "<<rRectMat.array[4]<<" "<<rRectMat.array[5]<<"\n"
    //        <<" "<<rRectMat.array[6]<<" "<<rRectMat.array[7]<<" "<<rRectMat.array[8]<<"\n";
            
    {
        CHECK_DW_ERROR(dwStereoRectifier_getRectificationMatrix	(&rectificationMatrixRight,DW_STEREO_SIDE_RIGHT,m_stereoRectifier));
    }
    //std::cout<<"RIGHT \n"<<rRectMat.array[0]<<" "<<rRectMat.array[1]<<" "<<rRectMat.array[2]<<"\n"
    //        <<" "<<rRectMat.array[3]<<" "<<rRectMat.array[4]<<" "<<rRectMat.array[5]<<"\n"
    //        <<" "<<rRectMat.array[6]<<" "<<rRectMat.array[7]<<" "<<rRectMat.array[8]<<"\n";

    //dwMatrix34f rRectMatProj;

    {
        CHECK_DW_ERROR(dwStereoRectifier_getProjectionMatrix(&projectionMatrixLeft,DW_STEREO_SIDE_LEFT,m_stereoRectifier));
    }
    std::cout<<"Projection LEFT \n"<<projectionMatrixLeft.array[0]<<" "<<projectionMatrixLeft.array[1]<<" "<<projectionMatrixLeft.array[2]<<"\n"
            <<" "<<projectionMatrixLeft.array[3]<<" "<<projectionMatrixLeft.array[4]<<" "<<projectionMatrixLeft.array[5]<<"\n"
            <<" "<<projectionMatrixLeft.array[6]<<" "<<projectionMatrixLeft.array[7]<<" "<<projectionMatrixLeft.array[8]<<"\n"
            <<" "<<projectionMatrixLeft.array[9]<<" "<<projectionMatrixLeft.array[10]<<" "<<projectionMatrixLeft.array[11]<<"\n";
            
    {
        CHECK_DW_ERROR(dwStereoRectifier_getProjectionMatrix(&projectionMatrixRight,DW_STEREO_SIDE_RIGHT,m_stereoRectifier));
    }
    //std::cout<<"Projection RIGHT \n"<<projectionMatrixRight.array[0]<<" "<<projectionMatrixRight.array[1]<<" "<<projectionMatrixRight.array[2]<<"\n"
    //        <<" "<<projectionMatrixRight.array[3]<<" "<<projectionMatrixRight.array[4]<<" "<<projectionMatrixRight.array[5]<<"\n"
    //        <<" "<<projectionMatrixRight.array[6]<<" "<<projectionMatrixRight.array[7]<<" "<<projectionMatrixRight.array[8]<<"\n"
    //        <<" "<<projectionMatrixRight.array[9]<<" "<<projectionMatrixRight.array[10]<<" "<<projectionMatrixRight.array[11]<<"\n";

    

    {
        CHECK_DW_ERROR(dwStereoRectifier_getReprojectionMatrix(&qrMatrix, m_stereoRectifier));	
    }
    std::cout<<"Projection LEFT \n"<<qrMatrix.array[0]<<" "<<qrMatrix.array[1]<<" "<<qrMatrix.array[2]<<" "<<qrMatrix.array[3]<<"\n"
            <<" "<<qrMatrix.array[4]<<" "<<qrMatrix.array[5]<<" "<<qrMatrix.array[6]<<" "<<qrMatrix.array[7]<<"\n"
            <<" "<<qrMatrix.array[8]<<" "<<qrMatrix.array[9]<<" "<<qrMatrix.array[10]<<" "<<qrMatrix.array[11]<<"\n"
            <<" "<<qrMatrix.array[12]<<" "<<qrMatrix.array[13]<<" "<<qrMatrix.array[14]<<" "<<qrMatrix.array[15]<<"\n";

    cX              = - qrMatrix.array[12];
    cY              = - qrMatrix.array[13];
    fX              =   qrMatrix.array[14];
    baseline        =  1/qrMatrix.array[11];
    rectfiedWidth   = m_roi.width;
    rectfiedHeight  = m_roi.height;
    //cXL             = projectionMatrixLeft.array[6];
    //cXR             = projectionMatrixRight.array[6];
            

    for (int32_t side = 0; side < 2; ++side) 
    {
        m_streamerCUDA2GLArray[side].reset(new SimpleImageStreamer<>(rectifiedImageProps, DW_IMAGE_GL, 10000, m_context));                
        {
            CHECK_DW_ERROR(dwImage_create(&m_imagesRGBA[side], rectifiedImageProps, m_context));
        }
        {
            CHECK_DW_ERROR(dwImage_create(&planeImage[side], imagePropsRUINT8, m_context)); 
        }
    }

    {
        CHECK_DW_ERROR(dwImage_create(&m_outputAnaglyph, rectifiedImageProps, m_context));
    }  
    
    m_streamerimageAnaglyphGL.reset(new SimpleImageStreamer<>(rectifiedImageProps, DW_IMAGE_GL, 10000, m_context));
    return true;            
}

void StereoRectifier::Process(std::vector<dwImageCUDA*> imageCUDA)
{
    {
        dwImage_getCUDA(&rectifiedCuda[0], m_outputRectified[0]);
        dwImage_getCUDA(&rectifiedCuda[1], m_outputRectified[1]);
    }
    {
        CHECK_DW_ERROR(dwStereoRectifier_rectify(rectifiedCuda[0],rectifiedCuda[1],imageCUDA[0], imageCUDA[1], m_stereoRectifier));
    }

    {   
        std::vector<dwImageHandle_t> frameGL;
        frameGL.resize(2);
        for (int camit =0;camit<2;++camit)
        {
            CHECK_DW_ERROR(dwImage_copyConvert(m_imagesRGBA[camit], m_outputRectified[camit], m_context));
            frameGL[camit] = m_streamerCUDA2GLArray[camit]->post(m_imagesRGBA[camit]);
            {
                dwImage_getGL(&rectifiedGL[camit], frameGL[camit]);
            }
        }
        
    }  
        
    {
        dwImageCUDA* imageAnaglyphCUDA;  
        {          
            dwImage_getCUDA(&imageAnaglyphCUDA, m_outputAnaglyph);            
            createAnaglyph(*imageAnaglyphCUDA, *rectifiedCuda[0], *rectifiedCuda[1]); 
        }  
        dwImageHandle_t imageAnaglyphframeGL = m_streamerimageAnaglyphGL->post(m_outputAnaglyph);
        {
            dwImage_getGL(&imageAnaglyphGL, imageAnaglyphframeGL);            
        }
    }             

    {                     
        for (int32_t i = 0; i < 2; ++i) 
        {
            dwImage_getCUDA(&planeImageCUDA[i],planeImage[i]);
            {
                CHECK_DW_ERROR(dwImage_copyConvert(planeImage[i], m_outputRectified[i], m_context));
            }                
        }                
    }       

}

StereoRectifier::~StereoRectifier()
{

}    

#endif
