#ifndef STEREO_DISPARITY_HPP
#define STEREO_DISPARITY_HPP

#define _CRT_SECURE_NO_WARNINGS

// Samples
#include <framework/DriveWorksSample.hpp>
#include <framework/Grid.hpp>
#include <framework/SimpleStreamer.hpp>
#include <framework/SampleFramework.hpp>
#include <framework/ProgramArguments.hpp>
#include <framework/Checks.hpp>

#include <dw/features/Features.h>

#include <framework/WindowGLFW.hpp>

// Core
#include <dw/core/Context.h>
#include <dw/core/Logger.h>
#include <dw/core/VersionCurrent.h>

// Pyramids
#include <dw/sfm/SFM.h>

// Stereo
#include <dw/stereo/Stereo.h>


#include "stereo_common/stereoCommon.hpp"
#include "StereoRectifier.hpp"
#include <vector>
#include <stdio.h>
using namespace dw_samples::common;

class StereoDisparity 
{
    private:
        // stereo rectifier
        dwContextHandle_t m_context           = DW_NULL_HANDLE;
        dwSALHandle_t m_sal                   = DW_NULL_HANDLE;

        dwRigConfigurationHandle_t m_rigConfiguration = DW_NULL_HANDLE;   
        dwImageProperties m_rectifiedImageProp;        
        
        dwImageCUDA *imageColorDisparityCuda, *imageMonoDisparityCuda;

        dwImageGL *imageColorDisparityGL, *imageMonoDisparityGL;

        dwImageHandle_t m_imageColorDisparityGL, m_imageMonoDisparityCUDA;

        dwStereoSide m_stereoSide = DW_STEREO_SIDE_LEFT;
        bool m_occlusion = true;
        bool m_occlusionInfill = false;
        bool m_infill = false;


        ProgramArguments inputArgs;
        dwBox2D m_roi;

    public:

        const float32_t COLOR_GAIN = 4.0f;
        const float32_t COLOR_GAIN_MAX = 20.0f;
        const float32_t INV_THR_MAX = 6.0f;
        float32_t m_invalidThreshold = -1.0f;
        std::vector<dwImageGL*> m_displayDisparity;

        
        // for color code rendering
        float32_t m_colorGain;        

        dwStereoHandle_t m_stereoAlgorithm;
        
        
        std::vector<dwPyramidHandle_t> m_pyramids;
        std::unique_ptr<SimpleImageStreamer<>> m_cudaDISP2gl[DW_STEREO_SIDE_COUNT];

        dwImageHandle_t m_colorDisparity;
        

        uint32_t m_levelStop;
        dwStereoSide m_side;

        //int disparityImageAs2DArray[1065][1841];// need to make it dynamic for future
        std::vector<std::vector<int>> disparityImageAs2DArray;
        int* disparityImageAs2DArrayPtr;


        StereoDisparity()
        {

            m_pyramids.resize(2);
            m_displayDisparity.resize(2);            
            m_rectifiedImageProp.type = DW_IMAGE_CUDA;
            m_rectifiedImageProp.format = DW_IMAGE_FORMAT_RGBA_UINT8;           

        }

        void setContext(dwContextHandle_t  _m_context)
        {
            m_context =_m_context;
        }

        void setSalHandle(dwSALHandle_t  _m_sal)
        {
            m_sal =_m_sal;
        }

        void setRectifiedImageProp(dwImageProperties _rectifiedImageProp)
        {
            m_rectifiedImageProp = _rectifiedImageProp;
        }


        dwImageGL* getColorDisparityGLImage()
        {
            return imageColorDisparityGL;
        }

        dwImageGL* getMonoDisparityGLImage()
        {
            return imageMonoDisparityGL;
        }

        dwImageCUDA* getColorDisparityCudaLImage()
        {
            return imageColorDisparityCuda;
        }

        dwImageCUDA* getMonoDisparityCudaImage()
        {
            return imageMonoDisparityCuda;
        }

        
        void setArgsParam(ProgramArguments stPar)
        {
            inputArgs = stPar;     
 
            if (inputArgs.get("rigconfig").empty()) 
            {
                throw std::runtime_error("Rig configuration file not specified, please provide a rig "
                                    "configuration file with the calibration of the stereo camera");
            }

            CHECK_DW_ERROR(dwRigConfiguration_initializeFromFile(&m_rigConfiguration, m_context,
                                                                inputArgs.get("rigconfig").c_str()));

            uint32_t totalWidth = 0;
            for (int32_t i = 0; i < 2; ++i) 
            {
                dwPinholeCameraConfig pinholeConfig;
                CHECK_DW_ERROR(dwRigConfiguration_getPinholeCameraConfig(&pinholeConfig, i, m_rigConfiguration));
                totalWidth += pinholeConfig.width;
            }

            m_levelStop = static_cast<uint8_t>(atoi(inputArgs.get("level").c_str()));
            m_side = inputArgs.get("single_side") == std::string("0") ? DW_STEREO_SIDE_BOTH : DW_STEREO_SIDE_LEFT;
                    


        }


        bool Initialize() 
        {

            dwPyramidConfig pyramidConf;
            pyramidConf.dataType = DW_TYPE_UINT8;//DW_TYPE_UINT8  DW_TYPE_UINT16 DW_TYPE_FLOAT16 
            pyramidConf.height = m_rectifiedImageProp.height;//m_imageProperties.height
            pyramidConf.width = m_rectifiedImageProp.width;//m_imageProperties.height
            // the default input has a very high resolution so many levels guarantee a better coverage of the image
            pyramidConf.levelCount = 6;

            for (int32_t i = 0; i < 2; ++i) 
            {
                CHECK_DW_ERROR(dwPyramid_initialize(&m_pyramids[i], &pyramidConf, 0, m_context));
            }

            // stereo parameters setup by the module. By default the level we stop at is 0, which is max resolution
            // in order to improve performance we stop at level specified by default as 1 in the input arguments
            dwStereoParams stereoParams;
            CHECK_DW_ERROR(dwStereo_initParams(&stereoParams));

            // level at which to stop computing disparity map
            stereoParams.levelStop = m_levelStop;
            // specifies which side to compute the disparity map from, if BOTH, LEFT or RIGHT only
            stereoParams.side = m_side;
            // since the pyramid is built for the stereo purpose we set the levels the same as the pyramids. In other
            // use cases it can be possible that the pyramid has too many levels and not all are necessary for the
            // stereo algorithm, so we can decide to use less
            stereoParams.levelCount         = pyramidConf.levelCount;
            stereoParams.maxDisparityRange  = 256;
            stereoParams.holesFilling       = true;
            stereoParams.initType           = DW_STEREO_COST_NCC;// DW_STEREO_COST_NCC DW_STEREO_COST_SAD DW_STEREO_COST_AD
            stereoParams.occlusionFilling   = true;
            stereoParams.refinementLevel    = 3;

            {
                CHECK_DW_ERROR(dwStereo_initialize(&m_stereoAlgorithm, m_rectifiedImageProp.width,
                                            m_rectifiedImageProp.height, &stereoParams,
                                            m_context));           
            }
            
            {
                std::stringstream ss;
                ss << "Disparity image with " << m_rectifiedImageProp.width << "x" << m_rectifiedImageProp.height <<"\n"
                <<"StereoParams.maxDisparityRange =" <<stereoParams.maxDisparityRange<<std::endl;
                log("%s", ss.str().c_str());
            }

            for (int32_t side = 0; side < DW_STEREO_SIDE_BOTH; ++side) 
            {
                m_cudaDISP2gl[side].reset(new SimpleImageStreamer<>(m_rectifiedImageProp, DW_IMAGE_GL, 10000, m_context));
            }
            dwImage_create(&m_colorDisparity, m_rectifiedImageProp, m_context);

            // gain for the color coding, proportional to the level we stop at. Lower gain means flatter colors.
            m_colorGain = COLOR_GAIN*(1<<m_levelStop);

            CHECK_DW_ERROR(dwImage_create(&m_imageMonoDisparityCUDA, m_rectifiedImageProp, m_context)); 

            return true;
            
        }

        void Process(std::vector<dwImageCUDA*> imageCUDA)
        {
           
            {
                for (int32_t i = 0; i < 2; ++i) 
                {
                    CHECK_DW_ERROR(dwPyramid_build(imageCUDA[i], m_pyramids[i]));                   
                }                
            }
            {
                CHECK_DW_ERROR(dwStereo_computeDisparity(m_pyramids[0], m_pyramids[1], m_stereoAlgorithm));
            }

            // get output and prepare for display
            for (int32_t side = 0; side < 2; ++side) 
            {
                const dwImageCUDA *disparity, *confidence; 

                {
                    CHECK_DW_ERROR(dwStereo_getDisparity(&disparity, (dwStereoSide) side, m_stereoAlgorithm));
                }

                /// create 2D vector of int for disparity value analysis in cpu (only left)
                {
                    disparityImageAs2DArray.resize(disparity->prop.height,std::vector<int>(disparity->prop.width));
                }

                if(side == 0)
                {
                    // 8bit daya per pixel even tough its written 16bit in manual
                    void* cpuimage = (void*)malloc(disparity->prop.width * disparity->prop.height); 
                    CHECK_CUDA_ERROR(cudaMemcpy2D(cpuimage,disparity->prop.width,disparity->dptr[0],
                                    disparity->pitch[0], disparity->prop.width , disparity->prop.height,
                                    cudaMemcpyDeviceToHost));  

                    for (int r1 = 0; r1< (int)disparity->prop.height; ++r1)
                    {
                        for(int cl1 = 0; cl1< (int)disparity->prop.width; ++cl1)
                        {
                            disparityImageAs2DArray[r1][cl1] = ((unsigned  char*)cpuimage)[r1*disparity->prop.width+cl1];
                        }
                    }

                    free(cpuimage);
                    //disparityImageAs2DArrayPtr = &disparityImageAs2DArray[0][0]; //points to first element of 2D array

                    {
                        dwImage_getCUDA(&imageMonoDisparityCuda,m_imageMonoDisparityCUDA);
                    }
                    {
                        imageMonoDisparityCuda->array[0] = disparity->array[0];
                        imageMonoDisparityCuda->dptr[0] = disparity->dptr[0];
                        imageMonoDisparityCuda->pitch[0] = disparity->pitch[0];
                        imageMonoDisparityCuda->prop = disparity->prop;
                        imageMonoDisparityCuda->timestamp_us = disparity->timestamp_us;
                    }
                } 

                dwImageCUDA* colorDisparityCUDA;
                dwImage_getCUDA(&colorDisparityCUDA, m_colorDisparity);
                colorCode(colorDisparityCUDA, *disparity, m_colorGain);

                {
                    CHECK_DW_ERROR(dwStereo_getConfidence(&confidence, (dwStereoSide) side, m_stereoAlgorithm));
                }
            {
                dwImageHandle_t imageGLDisparity = m_cudaDISP2gl[side]->post(m_colorDisparity);
                {
                    dwImage_getGL(&m_displayDisparity[side], imageGLDisparity);
                }
            }
   
            }
                                   
        }

        ~StereoDisparity()
        {
             
            dwImage_destroy(&m_colorDisparity);

            CHECK_DW_ERROR(dwSAL_release(&m_sal));
            if (m_context != DW_NULL_HANDLE)
            {
                CHECK_DW_ERROR(dwRelease(&m_context));
            }

        }

};

#endif
