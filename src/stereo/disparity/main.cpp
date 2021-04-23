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
// Copyright (c) 2015-2017 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and proprietary
// rights in and to this software and related documentation and any modifications thereto.
// Any use, reproduction, disclosure or distribution of this software and related
// documentation without an express license agreement from NVIDIA Corporation is
// strictly prohibited.
//
/////////////////////////////////////////////////////////////////////////////////////////

#define _CRT_SECURE_NO_WARNINGS

//Pyramid
#include <dw/features/Features.h>

#include <framework/WindowGLFW.hpp>
#include "stereo_common/stereoCommon.hpp"

class StereoDisparityApp : public StereoApp
{
public:
    const float32_t COLOR_GAIN = 4.0f;
    const float32_t COLOR_GAIN_MAX = 20.0f;
    const float32_t INV_THR_MAX = 6.0f;

    explicit StereoDisparityApp(const ProgramArguments& args);

    bool onInitialize() override final;
    void onProcess() override final;
    void onKeyDown(int key, int scancode, int mods) override final;
    void onRender() override final;
    void onRelease() override final;

    void drawSide(dwStereoSide side);

    dwStereoHandle_t m_stereoAlgorithm;
    dwPyramidHandle_t m_pyramids[DW_STEREO_SIDE_COUNT];
    dwImageCUDA *m_stereoImages[DW_STEREO_SIDE_COUNT];

    std::unique_ptr<SimpleImageStreamer<>> m_cudaRGBA2gl;
    std::unique_ptr<SimpleImageStreamer<>> m_cudaDISP2gl[DW_STEREO_SIDE_COUNT];

    dwImageHandle_t m_colorDisparity;
    dwImageHandle_t m_outputAnaglyph;

    uint32_t m_levelStop;
    dwStereoSide m_side;

private:
    dwStereoSide m_stereoSide = DW_STEREO_SIDE_BOTH;
    bool m_occlusion = true;
    bool m_occlusionInfill = false;
    bool m_infill = false;

    dwBox2D m_roi;

    dwImageGL *m_displayAnaglyph, *m_displayDisparity[DW_STEREO_SIDE_COUNT];
};

//#######################################################################################
StereoDisparityApp::StereoDisparityApp(const ProgramArguments& args) : StereoApp(args)
{
    m_levelStop = static_cast<uint8_t>(atoi(args.get("level").c_str()));
    m_side = args.get("single_side") == std::string("0") ? DW_STEREO_SIDE_BOTH : DW_STEREO_SIDE_LEFT;
}

//#######################################################################################
bool StereoDisparityApp::onInitialize(){
    if (!StereoApp::initSDK()) {
        return false;
    }

    if (!StereoApp::initRenderer()) {
        return false;
    }

    // default StereoApp is set to use RGBA images the video inputs a planar yuv420,
    // for this sample we input it directly to the stereo algorithm
    dwImageProperties inputProp = StereoApp::m_imageProperties;
    inputProp.format = DW_IMAGE_FORMAT_YUV420_UINT8_PLANAR;
    StereoApp::setInputProperties(inputProp);

    if (!StereoApp::onInitialize()) {
        return false;
    }

    // the stereo algorithm inputs two gaussian pyramids built from the rectified input. This is because the
    // algorithm requires a multi resolution representation.
    dwPyramidConfig pyramidConf;
    pyramidConf.dataType = DW_TYPE_UINT8;
    pyramidConf.height = StereoApp::m_imageProperties.height;
    pyramidConf.width = StereoApp::m_imageProperties.width;
    // the default input has a very high resolution so many levels guarantee a better coverage of the image
    pyramidConf.levelCount = 8;

    for (int32_t i = 0; i < DW_STEREO_SIDE_BOTH; ++i) {
        CHECK_DW_ERROR(dwPyramid_initialize(&m_pyramids[i], &pyramidConf, 0, StereoApp::m_context));
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
    stereoParams.levelCount = pyramidConf.levelCount;

    CHECK_DW_ERROR(dwStereo_initialize(&m_stereoAlgorithm, StereoApp::m_imageProperties.width,
                                       StereoApp::m_imageProperties.height, &stereoParams,
                                       StereoApp::m_context));

    // properties for the display of the input image, rendered as anaglyph (both images overlapping)
    dwImageProperties displayProperties {};
    displayProperties.type = DW_IMAGE_CUDA;
    displayProperties.format = DW_IMAGE_FORMAT_RGBA_UINT8;
    displayProperties.width = StereoApp::m_imageProperties.width;
    displayProperties.height = StereoApp::m_imageProperties.height;
    m_cudaRGBA2gl.reset(new SimpleImageStreamer<>(displayProperties, DW_IMAGE_GL, 10000, StereoApp::m_context));

    dwImage_create(&m_outputAnaglyph, displayProperties, StereoApp::m_context);

    // the output of the disaprity map, although it appears scaled, has the resolution of the
    // level we stop at. For this reason we need to setup and image streamer with the proper resolution
    CHECK_DW_ERROR(dwStereo_getSize(&displayProperties.width, &displayProperties.height, m_levelStop,
                                    m_stereoAlgorithm));

    for (int32_t side = 0; side < DW_STEREO_SIDE_BOTH; ++side) {
        m_cudaDISP2gl[side].reset(new SimpleImageStreamer<>(displayProperties, DW_IMAGE_GL, 10000, StereoApp::m_context));
    }

    {
        std::stringstream ss;
        ss << "Disparity image with " << displayProperties.width << "x" << displayProperties.height << std::endl;
        log("%s", ss.str().c_str());
    }

    // the disparity map is single channel, so we color it for better visualization (warm colors closer)
    dwImage_create(&m_colorDisparity, displayProperties, StereoApp::m_context);

    // gain for the color coding, proportional to the level we stop at. Lower gain means flatter colors.
    m_colorGain = COLOR_GAIN*(1<<m_levelStop);

    return true;
}

//#######################################################################################
void StereoDisparityApp::onRelease()
{
    dwImage_destroy(&m_outputAnaglyph);
    dwImage_destroy(&m_colorDisparity);

    for(int i=0; i<DW_STEREO_SIDE_COUNT; ++i)
        dwPyramid_release(&m_pyramids[i]);

    dwStereo_release(&m_stereoAlgorithm);

    StereoApp::onRelease();
}

//#######################################################################################
void StereoDisparityApp::onKeyDown(int key, int /*scancode*/, int /*mods*/)
{
    if (key == GLFW_KEY_O) {
        std::cout<<"Toggle occlusion"<<std::endl;
        m_occlusion = (m_occlusion == true) ? false : true;
        dwStereo_setOcclusionTest(m_occlusion, m_stereoAlgorithm);
    } else if (key == GLFW_KEY_K) {
        if (m_occlusion == true) {
            std::cout<<"Toggle occlusion infill"<<std::endl;
            m_occlusionInfill = (m_occlusionInfill == true) ? false : true;
            dwStereo_setOcclusionInfill(m_occlusionInfill, m_stereoAlgorithm);
        } else {
            std::cout<<"Cannot toggle occlusion infill, occlusion test is off"<<std::endl;
        }
    } else if (key == GLFW_KEY_I) {
        std::cout<<"Toggle invalidity infill"<<std::endl;
        m_infill = (m_infill == true) ? false : true;
        dwStereo_setInfill(m_infill, m_stereoAlgorithm);
    } else if (key == GLFW_KEY_0) {
        std::cout<<"Refinement 0"<<std::endl;
        dwStereo_setRefinementLevel(0, m_stereoAlgorithm);
    } else if (key == GLFW_KEY_1) {
        std::cout<<"Refinement 1"<<std::endl;
        dwStereo_setRefinementLevel(1, m_stereoAlgorithm);
    } else if (key == GLFW_KEY_2) {
        std::cout<<"Refinement 2"<<std::endl;
        dwStereo_setRefinementLevel(2, m_stereoAlgorithm);
    } else if (key == GLFW_KEY_3) {
        std::cout<<"Refinement 3"<<std::endl;
        dwStereo_setRefinementLevel(3, m_stereoAlgorithm);
    } else if (key == GLFW_KEY_4) {
        std::cout<<"Refinement 4"<<std::endl;
        dwStereo_setRefinementLevel(4, m_stereoAlgorithm);
    } else if (key == GLFW_KEY_5) {
        std::cout<<"Refinement 5"<<std::endl;
        dwStereo_setRefinementLevel(5, m_stereoAlgorithm);
    } else if (key == GLFW_KEY_6) {
        std::cout<<"Refinement 6"<<std::endl;
        dwStereo_setRefinementLevel(6, m_stereoAlgorithm);
    } else if (key == GLFW_KEY_W) {
        m_colorGain += 0.5f;
        if (m_colorGain > COLOR_GAIN_MAX) {
            m_colorGain = COLOR_GAIN_MAX;
        }
        std::cout<<"Color gain "<<m_colorGain<<std::endl;
    } else if (key == GLFW_KEY_Q) {
        m_colorGain -= 0.5f;
        if (m_colorGain < 0.0f) {
            m_colorGain = 0.0f;
        }
        std::cout<<"Color gain "<<m_colorGain<<std::endl;
    } else if (key == GLFW_KEY_KP_ADD) {
        StereoApp::m_invalidThreshold += 1.0f;
        if (StereoApp::m_invalidThreshold > INV_THR_MAX) {
            StereoApp::m_invalidThreshold = INV_THR_MAX;
        }
        dwStereo_setInvalidThreshold(StereoApp::m_invalidThreshold, m_stereoAlgorithm);
        std::cout<<"Invalidity thr "<<StereoApp::m_invalidThreshold<<std::endl;
    } else if (key == GLFW_KEY_KP_SUBTRACT) {
        StereoApp::m_invalidThreshold -= 1.0f;
        if (StereoApp::m_invalidThreshold < -1.0f) {
            StereoApp::m_invalidThreshold = -1.0f;
            dwStereo_setInvalidThreshold(0.0f, m_stereoAlgorithm);
        }
        if (StereoApp::m_invalidThreshold >= 0.0f) {
            dwStereo_setInvalidThreshold(StereoApp::m_invalidThreshold, m_stereoAlgorithm);
            std::cout<<"Invalidity thr "<<StereoApp::m_invalidThreshold<<std::endl;
        } else {
            std::cout<<"Invalidity off "<<std::endl;
        }
    }
}

//#######################################################################################
void StereoDisparityApp::onProcess()
{
    dwImageHandle_t stereoImages[2];
    // read images
    {
        dw::common::ProfileCUDASection s(getProfilerCUDA(), "Camera read");
        std::this_thread::yield();
        while (!StereoApp::readStereoImages(stereoImages)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    dwImage_getCUDA(&m_stereoImages[0], stereoImages[0]);
    dwImage_getCUDA(&m_stereoImages[1], stereoImages[1]);

    std::cout<<"stereoImages[0]->prop.height= "<<m_stereoImages[0]->prop.height<<"\n";
    std::cout<<"stereoImages[0]->prop.width= "<<m_stereoImages[0]->prop.width<<"\n";
    std::cout<<"stereoImages[0]->prop.format= "<<m_stereoImages[0]->prop.format<<"\n";
    std::cout<<"stereoImages[0]->prop.memoryLayout= "<<m_stereoImages[0]->prop.memoryLayout<<"\n";

    // create anaglyph of the input (left red, right blue)
    dwImageCUDA* imageAnaglyphCUDA;
    dwImage_getCUDA(&imageAnaglyphCUDA, m_outputAnaglyph);
    createAnaglyph(*imageAnaglyphCUDA, *m_stereoImages[DW_STEREO_SIDE_LEFT],
                   *m_stereoImages[DW_STEREO_SIDE_RIGHT]);

    // get display image for anaglyph
    dwImageHandle_t imageGL = m_cudaRGBA2gl->post(m_outputAnaglyph);
    dwImage_getGL(&m_displayAnaglyph, imageGL);
    std::cout<<"stereoImages[0]->prop.height= "<<m_stereoImages[0]->prop.height<<"\n";
    std::cout<<"stereoImages[0]->prop.width= "<<m_stereoImages[0]->prop.width<<"\n";
    std::cout<<"stereoImages[0]->prop.format= "<<m_stereoImages[0]->prop.format<<"\n";
    std::cout<<"stereoImages[0]->prop.memoryLayout= "<<m_stereoImages[0]->prop.memoryLayout<<"\n";
    std::cout<<"stereoImages[0]->pitch[0]= "<<m_stereoImages[0]->pitch[0]<<"\n";

    // build pyramids
    {
        dw::common::ProfileCUDASection s(getProfilerCUDA(), "Pyramid build");
        dwImageCUDA planeImage{};
        for (int32_t i = 0; i < DW_STEREO_SIDE_BOTH; ++i) {
            CHECK_DW_ERROR(dwImageCUDA_getPlaneAsImage(&planeImage, m_stereoImages[i], 0));
            CHECK_DW_ERROR(dwPyramid_build(&planeImage, m_pyramids[i]));
            std::cout<<"planeImage->prop.height= "<<planeImage.prop.height<<"\n";
            std::cout<<"planeImage->prop.width= "<<planeImage.prop.width<<"\n";
            std::cout<<"planeImage->prop.format= "<<planeImage.prop.format<<"\n";
            std::cout<<"planeImage->prop.memoryLayout= "<<planeImage.prop.memoryLayout<<"\n";
            std::cout<<"planeImage->pitch[0]= "<<planeImage.pitch[0]<<"\n";
        }
    }
    std::cout<<"stereoImages[0]->prop.height= "<<m_stereoImages[0]->prop.height<<"\n";
    std::cout<<"stereoImages[0]->prop.width= "<<m_stereoImages[0]->prop.width<<"\n";
    std::cout<<"stereoImages[0]->prop.format= "<<m_stereoImages[0]->prop.format<<"\n";
    std::cout<<"stereoImages[0]->prop.memoryLayout= "<<m_stereoImages[0]->prop.memoryLayout<<"\n";

    // compute disparity
    {
        dw::common::ProfileCUDASection s(getProfilerCUDA(), "Stereo");

        CHECK_DW_ERROR(dwStereo_computeDisparity(m_pyramids[DW_STEREO_SIDE_LEFT],
                                                 m_pyramids[DW_STEREO_SIDE_RIGHT],
                                                 m_stereoAlgorithm));
    }

    // get output and prepare for display
    for (int32_t side = 0; side < DW_STEREO_SIDE_BOTH; ++side) {
        const dwImageCUDA *disparity, *confidence;

        CHECK_DW_ERROR(dwStereo_getDisparity(&disparity, (dwStereoSide) side, m_stereoAlgorithm));

        dwImageCUDA* colorDisparityCUDA;
        dwImage_getCUDA(&colorDisparityCUDA, m_colorDisparity);
        colorCode(colorDisparityCUDA, *disparity, m_colorGain);

        CHECK_DW_ERROR(dwStereo_getConfidence(&confidence, (dwStereoSide) side, m_stereoAlgorithm));

        // mix disparity and confidence, where confidence is occlusion, show black, where it is invalidity show white, leave
        // as is otherwise. See README for instructions on how to change the threshold of invalidity
        if ((m_occlusionInfill == false) && (m_occlusion == true)) {
            mixDispConf(colorDisparityCUDA, *confidence, StereoApp::m_invalidThreshold >= 0.0f);
        }

        std::cout<<"disparity->prop.height= "<<disparity->prop.height<<"\n";
        std::cout<<"disparity->prop.width= "<<disparity->prop.width<<"\n";
        std::cout<<"disparity->prop.format= "<<disparity->prop.format<<"\n";
        std::cout<<"disparity->prop.memoryLayout= "<<disparity->prop.memoryLayout<<"\n";
        std::cout<<"disparity->pitch[0]= "<<disparity->pitch[0]<<"\n";

        std::cout<<"confidence->prop.height= "<<confidence->prop.height<<"\n";
        std::cout<<"confidence->prop.width= "<<confidence->prop.width<<"\n";
        std::cout<<"confidence->prop.format= "<<confidence->prop.format<<"\n";
        std::cout<<"confidence->pitch[0]= "<<confidence->pitch[0]<<"\n";

        std::cout<<"colorDisparityCUDA->prop.height= "<<colorDisparityCUDA->prop.height<<"\n";
    std::cout<<"colorDisparityCUDA->prop.width= "<<colorDisparityCUDA->prop.width<<"\n";
    std::cout<<"colorDisparityCUDA->prop.format= "<<colorDisparityCUDA->prop.format<<"\n";
    std::cout<<"colorDisparityCUDA->prop.memoryLayout= "<<colorDisparityCUDA->prop.memoryLayout<<"\n";

        dwImageHandle_t imageGLDisparity = m_cudaDISP2gl[side]->post(m_colorDisparity);
        dwImage_getGL(&m_displayDisparity[side], imageGLDisparity);

        std::cout<<"m_displayDisparity[0]->prop.height= "<<m_displayDisparity[0]->prop.height<<"\n";
    std::cout<<"m_displayDisparity[0]->prop.width= "<<m_displayDisparity[0]->prop.width<<"\n";
    std::cout<<"m_displayDisparity[0]->prop.format= "<<m_displayDisparity[0]->prop.format<<"\n";
    std::cout<<"m_displayDisparity[0]->prop.memoryLayout= "<<m_displayDisparity[0]->prop.memoryLayout<<"\n";
    }
}

//#######################################################################################
void StereoDisparityApp::onRender()
{
    // render input as anaglyph
    dwVector2i windowSize{DriveWorksSample::getWindowWidth(), DriveWorksSample::getWindowHeight()};
    StereoApp::m_simpleRenderer->setScreenRect(dwRect{windowSize.x/4, windowSize.y/2, windowSize.x/2, windowSize.y/2});
    StereoApp::m_simpleRenderer->renderQuad(m_displayAnaglyph);
    StereoApp::m_simpleRenderer->renderText(10, 10, DW_RENDERER_COLOR_GREEN, "Input");

    // render disparity and confidence
    drawSide(DW_STEREO_SIDE_LEFT);
    drawSide(DW_STEREO_SIDE_RIGHT);

    renderutils::renderFPS(m_renderEngine, getCurrentFPS());
}

//#######################################################################################
void StereoDisparityApp::drawSide(dwStereoSide side)
{
    dwVector2i windowSize{DriveWorksSample::getWindowWidth(), DriveWorksSample::getWindowHeight()};

    dwRect screenRect{};
    screenRect.x = side == DW_STEREO_SIDE_LEFT ? 0 : windowSize.x/2;
    screenRect.y = 0;
    screenRect.width = windowSize.x/2;
    screenRect.height = windowSize.y/2;
    StereoApp::m_simpleRenderer->setScreenRect(screenRect);
    StereoApp::m_simpleRenderer->renderQuad(m_displayDisparity[side]);
    dwImageProperties prop;
    dwImage_getProperties(&prop, m_colorDisparity);
    StereoApp::m_simpleRenderer->renderText(10, 10, DW_RENDERER_COLOR_WHITE,
                                            std::to_string(prop.width) + "x" +
                                                std::to_string(prop.height));
}

//#######################################################################################
int main(int argc, const char **argv)
{
    ProgramArguments args(argc, argv, {
        ProgramArguments::Option_t{"video0", (DataPath::get() + std::string{"/samples/stereo/leftRect.h264"}).c_str(), "Left input video."},
        ProgramArguments::Option_t{"video1", (DataPath::get() + std::string{"/samples/stereo/rightRect.h264"}).c_str(), "Right input video."},
        ProgramArguments::Option_t{"level","1","Log level"},
        ProgramArguments::Option_t{"single_side", "0", "If set to 1 only left disparity is computed."}
    });

    // -------------------

    StereoDisparityApp app(args);

    app.initializeWindow("StereoDisparityApp", 1280, 800);

    return app.run();
}
