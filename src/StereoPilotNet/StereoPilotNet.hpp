#ifndef STEREO_PILOT_NET_HPP
#define STEREO_PILOT_NET_HPP


#include <framework/DriveWorksSample.hpp>
#include <framework/SimpleStreamer.hpp>
#include <framework/SimpleRenderer.hpp>
#include <framework/SimpleCamera.hpp>
#include <fstream>

#include <thread>
#include <chrono>

#include "InputHandler.hpp"
#include "StereoDisparity.hpp"
#include "StereoRectifier.hpp"
#include "FreeSpaceDetector.hpp"
#include "DriveNet.hpp"
#include "Disparity2Depth.hpp"
#include <vector>
#include <string>

using namespace dw_samples::common;

class StereoPilotNet: public DriveWorksSample
{
        
    private:

    // objects to required classes
    
        //provides rectified images from camera or videos
        InputHandler* inputHandler; 
        //computes disparity from rectified images
        StereoDisparity* stereoDisparity;
        // free space detector
        FreeSpaceDetector* freeSpaceDetector;

        DriveNet* driveNet;    

        Disparity2Depth* disparity2Depth;   

        dwContextHandle_t m_context           = DW_NULL_HANDLE;
        dwSALHandle_t m_sal                   = DW_NULL_HANDLE;
        dwRenderEngineHandle_t m_renderEngine = DW_NULL_HANDLE;
        ProgramArguments args1;
        
        std::vector<dwImageCUDA*> inputImageArray,  rectifiedImageArrayCuda;
        std::vector<dwImageGL*> inputImageArrayGL, rectifiedImageArrayGL;
        dwImageGL* imageAnaglyphGL;

        /// for free space render
        dwFreespaceDetection* m_freespaceBoundaryPtr;
        dwFreespaceDetection m_freespaceBoundary{};

        //for Drivenet render
        std::vector<std::string> m_classLabels;
        // Vectors of boxes and class label ids
        std::vector<std::vector<dwRectf>> m_dnnBoxList;
        std::vector<std::vector<std::string>> m_dnnLabelList;
        std::vector<std::vector<const char*>> m_dnnLabelListPtr;
        std::vector<const dwTrackedBox2D *> m_trackedBoxes ;
        std::vector<size_t> m_numTrackedBoxes;
        std::vector<std::vector<dwRectf>> m_trackedBoxListFloat;
        // Colors for rendering bounding boxes
        static const uint32_t MAX_BOX_COLORS         = DW_DRIVENET_NUM_CLASSES;
        const dwVector4f m_boxColors[MAX_BOX_COLORS] = {{1.0f, 0.0f, 0.0f, 1.0f},
                                                        {0.0f, 1.0f, 0.0f, 1.0f},
                                                        {0.0f, 0.0f, 1.0f, 1.0f},
                                                        {1.0f, 0.0f, 1.0f, 1.0f},
                                                        {1.0f, 0.647f, 0.0f, 1.0f}};
                    
        const dwVector4f m_colorLightBlue{DW_RENDERER_COLOR_LIGHTBLUE[0],
                                      DW_RENDERER_COLOR_LIGHTBLUE[1],
                                      DW_RENDERER_COLOR_LIGHTBLUE[2],
                                      DW_RENDERER_COLOR_LIGHTBLUE[3]};
        const dwVector4f m_colorYellow{DW_RENDERER_COLOR_YELLOW[0],
                                    DW_RENDERER_COLOR_YELLOW[1],
                                    DW_RENDERER_COLOR_YELLOW[2],
                                    DW_RENDERER_COLOR_YELLOW[3]};

        // for render
        static const uint32_t TILE_COUNT      = 9;
        static const uint32_t TILES_PER_ROW   = 3;
        uint32_t m_tiles[TILE_COUNT];
        dwVector2f range{};

        std::vector<std::vector<DepthMeasurement>> objectDepth;
        
        
        std::vector<std::vector<TrackerMeasurement>> objectTracker;
        std::ofstream myfile;
        void write2file(std::vector<std::vector<DepthMeasurement>>, std::vector<std::vector<TrackerMeasurement>>);

        /// -----------------------------
        /// Initialize Logger and DriveWorks context
        /// -----------------------------
        void initializeDriveWorks(dwContextHandle_t& context) const
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

            CHECK_DW_ERROR(dwInitialize(&context, DW_VERSION, &sdkParams));
        }

    public:

        StereoPilotNet(const ProgramArguments& args): DriveWorksSample(args)
        {
            args1 = args;            
        }
        
        bool onInitialize() override;
        void onProcess() override;        
        void onRender()  override;
        void onRelease() override;
        void onReset() override;        
        ///------------------------------------------------------------------------------
        /// Change renderer properties when main rendering window is resized
        ///------------------------------------------------------------------------------
        void onResizeWindow(int width, int height) override;        
        void renderImage(dwImageGL*);
        void renderFreeSpaceBoundary(dwImageGL*, dwFreespaceDetection);
        void renderDrivenet(dwImageGL*, std::vector<std::string>, std::vector<std::vector<dwRectf>>, std::vector<std::vector<const char*>>);
        void renderDrivenetTracker(dwImageGL*, std::vector<std::string>, std::vector<std::vector<dwRectf>>,
                                     std::vector<const dwTrackedBox2D *>,
                                     std::vector<size_t>);
        ~StereoPilotNet();
};

bool StereoPilotNet::onInitialize()
{      
    //Initialize DriveWorks SDK context and Input Handler and Stereo handler
    {
        initializeDriveWorks(m_context);
        CHECK_DW_ERROR(dwSAL_initialize(&m_sal, m_context));
        myfile.open("/home/anweshan/Software/MeasurementData.txt", std::ios_base::out);
    }

    /// pass parameters to inputHandler
    inputHandler = new InputHandler();
    inputHandler->setContext(m_context);
    inputHandler->setSalHandle(m_sal);
    inputHandler->setArgsParam(args1);          
    inputHandler->Initialize();
    std::this_thread::sleep_for(std::chrono::milliseconds(2));

    /// pass parameters to stereo disparity
    stereoDisparity = new StereoDisparity();
    stereoDisparity->setContext(m_context);
    stereoDisparity->setSalHandle(m_sal);
    stereoDisparity->setArgsParam(args1);  
    stereoDisparity->setRectifiedImageProp(inputHandler->stereoRectifier->getRectifiedImageProperties());        
    stereoDisparity->Initialize();  
    std::this_thread::sleep_for(std::chrono::milliseconds(2));

    /// pass parameters for free space detector
    freeSpaceDetector = new FreeSpaceDetector();
    freeSpaceDetector->setContext(m_context);
    freeSpaceDetector->setArgsParam(args1);    
    freeSpaceDetector->setTransformation(inputHandler->stereoRectifier->getSensor2RigTransformation()); 
    freeSpaceDetector->setCalibratedCameraHandle(inputHandler->stereoRectifier->getCalibratedCameraHandleLeft());  
    freeSpaceDetector->setImageProperties(inputHandler->stereoRectifier->getRectifiedImageProperties());   
    freeSpaceDetector->Initialize();
    std::this_thread::sleep_for(std::chrono::milliseconds(2));

    /// pass parameters for DriveNet detector
    driveNet = new DriveNet();
    driveNet->setContext(m_context);
    driveNet->setArgsParam(args1);    
    driveNet->setImageProperties(inputHandler->stereoRectifier->getRectifiedImageProperties());   
    driveNet->Initialize();
    std::this_thread::sleep_for(std::chrono::milliseconds(2));

    disparity2Depth = new Disparity2Depth();
    //disparity2Depth->setContext(m_context);
    //disparity2Depth->Initialize(inputHandler->stereoRectifier->fX, inputHandler->stereoRectifier->baseline);
    disparity2Depth->Initialize(inputHandler->stereoRectifier->fX, inputHandler->stereoRectifier->baseline,inputHandler->stereoRectifier->cX,
                                inputHandler->stereoRectifier->cY,inputHandler->stereoRectifier->cXL,inputHandler->stereoRectifier->cXR,
                                inputHandler->stereoRectifier->rectfiedWidth,inputHandler->stereoRectifier->rectfiedHeight,
                                inputHandler->stereoRectifier->focalLengthL);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(2));

    inputImageArray.resize(2);
    inputImageArrayGL.resize(2);
    rectifiedImageArrayGL.resize(2);
    rectifiedImageArrayCuda.resize(2); 
   
    
    // initialize Renderer        
    {
        // Setup render engine
        dwRenderEngineParams params{};
        CHECK_DW_ERROR(dwRenderEngine_initDefaultParams(&params, getWindowWidth(), getWindowHeight()));
        params.defaultTile.lineWidth = 0.2f;
        params.defaultTile.font      = DW_RENDER_ENGINE_FONT_VERDANA_20;
        CHECK_DW_ERROR(dwRenderEngine_initialize(&m_renderEngine, &params, m_context));

        dwRenderEngineTileState paramList[TILE_COUNT];
        for (uint32_t i = 0; i < TILE_COUNT; ++i)
            dwRenderEngine_initTileState(&paramList[i]);

        dwRenderEngine_addTilesByCount(m_tiles, TILE_COUNT, TILES_PER_ROW, paramList, m_renderEngine);        
        
    }

    return true;
}

void StereoPilotNet::onProcess()
{
    /// Trigger to receive and rectify image from input
    { 
        inputHandler->Process();
    }

    /// Receive input images from input source
    {
        inputImageArray     = inputHandler->getStereoRGBAImages();
        
    }

    /// Receive rectified CUDA image 
    {
        rectifiedImageArrayCuda = inputHandler->stereoRectifier->getRectifiedCudaImages();
    }

    /// Pass rectified images to compute Disparity
    {
        stereoDisparity->Process(inputHandler->stereoRectifier->getRectifiedRCudaImages());
    }

    /// Pass rectified images for free space detection and receive the free space boundaries object
    {
        freeSpaceDetector->Process(inputHandler->stereoRectifier->getRectifiedCudaImages()[0]);
        m_freespaceBoundary    = freeSpaceDetector->m_freespaceBoundary;
    }

    /// Pass rectified images for Drivenet detection and receive boundingbox list
    {
        driveNet->Process(inputHandler->stereoRectifier->getRectifiedCudaImages()[0]);
        //m_freespaceBoundary    = freeSpaceDetector->m_freespaceBoundary;
        // Labels of each class
    }

    {
        m_classLabels           = driveNet->m_classLabels;
        m_dnnBoxList            = driveNet->m_dnnBoxList;
        m_dnnLabelList          = driveNet->m_dnnLabelList;
        m_dnnLabelListPtr       = driveNet->m_dnnLabelListPtr;
        m_trackedBoxes          = driveNet->driveNetBoxTracker->m_trackedBoxes;
        m_numTrackedBoxes       = driveNet->driveNetBoxTracker->m_numTrackedBoxes;
        m_trackedBoxListFloat   = driveNet->driveNetBoxTracker->m_trackedBoxListFloat;
    }

    {
        disparity2Depth->Process( m_dnnBoxList,(long long)inputImageArray[0]->timestamp_us,stereoDisparity->disparityImageAs2DArray);
    }
    {
        disparity2Depth->ProcessTrack( m_trackedBoxListFloat,m_trackedBoxes, m_numTrackedBoxes,(long long)inputImageArray[0]->timestamp_us,stereoDisparity->disparityImageAs2DArray);
    }

    {
        objectDepth     = disparity2Depth->objectDepth;
        objectTracker   = disparity2Depth->objectTracker;
        // write2file(objectDepth, objectTracker); // writing tracker and detector output to txtfile
    }



}

void StereoPilotNet::onRender()
{
    
    inputImageArrayGL   = inputHandler->getStereoRGBAGLImages();
    rectifiedImageArrayGL = inputHandler->stereoRectifier->getRectifiedGLImages();
    imageAnaglyphGL = inputHandler->stereoRectifier->getImageAnaglyphGL();

    dwImageGL* colorDispGL = stereoDisparity->m_displayDisparity[0];

    dwRenderEngine_setTile(m_tiles[0], m_renderEngine);
    renderImage(inputImageArrayGL[0]);

    dwRenderEngine_setTile(m_tiles[1], m_renderEngine);
    renderImage(inputImageArrayGL[1]);

    dwRenderEngine_setTile(m_tiles[3], m_renderEngine);
    renderImage(rectifiedImageArrayGL[0]);

    dwRenderEngine_setTile(m_tiles[4], m_renderEngine);
    renderImage(rectifiedImageArrayGL[1]);

    dwRenderEngine_setTile(m_tiles[2], m_renderEngine);
    renderImage(imageAnaglyphGL);

    dwRenderEngine_setTile(m_tiles[5], m_renderEngine);
    renderImage(colorDispGL);

    dwRenderEngine_setTile(m_tiles[6], m_renderEngine);
    renderFreeSpaceBoundary(rectifiedImageArrayGL[0], m_freespaceBoundary);

    dwRenderEngine_setTile(m_tiles[7], m_renderEngine);
    //StereoPilotNet::renderDrivenet(rectifiedImageArrayGL[0],m_classLabels,m_dnnBoxList,m_dnnLabelListPtr);
    StereoPilotNet::renderDrivenet(colorDispGL,m_classLabels,m_dnnBoxList,m_dnnLabelListPtr);

    dwRenderEngine_setTile(m_tiles[8], m_renderEngine);
    StereoPilotNet::renderDrivenetTracker(rectifiedImageArrayGL[0],m_classLabels,m_trackedBoxListFloat, m_trackedBoxes, m_numTrackedBoxes);



    renderutils::renderFPS(m_renderEngine, getCurrentFPS());

}

void StereoPilotNet::onRelease() 
{
    inputHandler->Release();
    
    CHECK_DW_ERROR(dwRenderEngine_release(&m_renderEngine));
    if (m_renderEngine != DW_NULL_HANDLE)
    {
        CHECK_DW_ERROR(dwRenderEngine_release(&m_renderEngine));
    }

    if (m_context != DW_NULL_HANDLE)
    {
        CHECK_DW_ERROR(dwRelease(&m_context));
    }

    CHECK_DW_ERROR(dwLogger_release());
}

void StereoPilotNet::onReset() 
{
    inputHandler->Reset();
}

void StereoPilotNet::onResizeWindow(int width, int height)
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

void StereoPilotNet::renderImage(dwImageGL* _renderImage)
{
    {
        range.x = _renderImage->prop.width;
        range.y = _renderImage->prop.height;

        dwRenderEngineTileLayout layout{};
        dwRenderEngine_getLayout(&layout, m_renderEngine);
        layout.useAspectRatio = true;
        layout.aspectRatio    = range.x / range.y;
        dwRenderEngine_setLayout(layout, m_renderEngine);
        {
        
            CHECK_DW_ERROR(dwRenderEngine_setCoordinateRange2D(range, m_renderEngine));
            CHECK_DW_ERROR(dwRenderEngine_renderImage2D(_renderImage, {0.0f, 0.0f, range.x, range.y}, m_renderEngine)); 
        }
    }
}

void StereoPilotNet::renderFreeSpaceBoundary(dwImageGL* _renderImage ,dwFreespaceDetection _m_freespaceBoundary)
{
    {
        renderImage(_renderImage);
    }

    float32_t maxWidth = 8.0; //10 meters as a step, [0, 10) will have max line width
    float32_t witdhRatio = 0.8;
    float32_t dist2Width[20];
    dist2Width[0] = maxWidth;
    for(uint32_t i = 1; i < 20; i++)
        dist2Width[i] = dist2Width[i-1]*witdhRatio;

    float32_t prevWidth, curWidth = maxWidth/2;
    prevWidth = curWidth;
        
    uint32_t index = 0;
    uint32_t count = 1;
 
    for (uint32_t i = 1; i < _m_freespaceBoundary.numberOfBoundaryPoints; ++i)
    {
        {
            CHECK_DW_ERROR(dwRenderEngine_setLineWidth(curWidth, m_renderEngine));
        }
        if (_m_freespaceBoundary.boundaryType[i] != _m_freespaceBoundary.boundaryType[i-1] ||
                (curWidth != prevWidth && count > 0))
        {
            dwFreespaceBoundaryType category = _m_freespaceBoundary.boundaryType[i-1];
            if (category==DW_BOUNDARY_TYPE_OTHER) {
                CHECK_DW_ERROR(dwRenderEngine_setColor(renderutils::colorYellow, m_renderEngine));
            } else if (category==DW_BOUNDARY_TYPE_CURB) {
                CHECK_DW_ERROR(dwRenderEngine_setColor(renderutils::colorGreen, m_renderEngine));
            } else if (category==DW_BOUNDARY_TYPE_VEHICLE) {
                CHECK_DW_ERROR(dwRenderEngine_setColor(renderutils::colorRed, m_renderEngine));
            } else if (category==DW_BOUNDARY_TYPE_PERSON) {
                CHECK_DW_ERROR(dwRenderEngine_setColor(renderutils::colorLightBlue, m_renderEngine));
            }

            {
                CHECK_DW_ERROR(dwRenderEngine_render(DW_RENDER_ENGINE_PRIMITIVE_TYPE_LINESTRIP_2D,
                                                        &_m_freespaceBoundary.boundaryImagePoint[index],
                                                        sizeof(dwVector2f),
                                                        0,
                                                        count,
                                                        m_renderEngine));
            }   
            index = i;
            count = 1;
            prevWidth = curWidth;
        }
        ++count;
    }   

    {
        CHECK_DW_ERROR(dwRenderEngine_setLineWidth(2, m_renderEngine));
    }
    {
        CHECK_DW_ERROR(dwRenderEngine_setColor(renderutils::colorYellow, m_renderEngine));
    }
}


void StereoPilotNet::renderDrivenet(dwImageGL* _renderImage, std::vector<std::string> _m_classLabels ,
                                     std::vector<std::vector<dwRectf>> _m_dnnBoxList,
                                     std::vector<std::vector<const char*>> _m_dnnLabelListPtr)
{
    {
        renderImage(_renderImage);
    }
    std::vector<std::vector<const char*>> gg = _m_dnnLabelListPtr;

    for (size_t classIdx = 0; classIdx < _m_classLabels.size(); classIdx++)
    {

        //if (&_m_dnnBoxList[classIdx][0] == nullptr)
        //    continue;

        CHECK_DW_ERROR(dwRenderEngine_setColor(m_boxColors[classIdx % MAX_BOX_COLORS], m_renderEngine));
        CHECK_DW_ERROR(dwRenderEngine_setLineWidth(2.0f, m_renderEngine));
        std::vector<std::string> depthstring;
        std::vector<const char*> depthptr;
        //std::cout<<" objectDepth[classIdx].size()=" <<objectDepth[classIdx].size()<<"\n";
        
        for(int jj = 0; jj < (int)(objectDepth[classIdx].size()) ; ++jj)
        {
            if (&_m_dnnBoxList[classIdx][jj] == nullptr)
                continue;
            //std::cout<<" jj=" <<jj<<"\n";
            std::string tempstringdepth = std::to_string(objectDepth[classIdx][jj].depth);
            std::string tempstringstd   = std::to_string(objectDepth[classIdx][jj].standardDeviationDepth);
            std::string tempstringazi   = std::to_string(objectDepth[classIdx][jj].azimuth);
            std::string tempstringele   = std::to_string(objectDepth[classIdx][jj].elevation);
            //depthstring.push_back(_m_classLabels[classIdx]+"=" + std::to_string(objectDepth[classIdx][jj].depth));
            depthstring.push_back(_m_classLabels[classIdx]+"=" + tempstringdepth.substr(0,tempstringdepth.find(".")+3)
                                 + " "+  tempstringstd.substr(0,tempstringstd.find(".")+3)
                                 + " "+  tempstringazi.substr(0,tempstringazi.find(".")+3)
                                 + " "+  tempstringele.substr(0,tempstringele.find(".")+3));
            depthptr.push_back(depthstring.back().c_str());
            
            {
                CHECK_DW_ERROR(dwRenderEngine_renderWithLabel(DW_RENDER_ENGINE_PRIMITIVE_TYPE_BOXES_2D,
                                                                &_m_dnnBoxList[classIdx][jj],
                                                                sizeof(dwRectf),
                                                                0,
                                                                depthptr[jj],
                                                                1,
                                                                m_renderEngine));
            }
            //std::cout<<"\n"<<&_m_dnnLabelListPtr[classIdx][0]<<"\n";
            //depthptr[jj]
        }
    }

    //&_m_dnnLabelListPtr[classIdx][0]
}


void StereoPilotNet::renderDrivenetTracker(dwImageGL* _renderImage, std::vector<std::string> _m_classLabels ,
                                     std::vector<std::vector<dwRectf>> _m_dnnBoxList,
                                     std::vector<const dwTrackedBox2D *> _m_trackedBoxes,
                                     std::vector<size_t> _m_numTrackedBoxes)
{
    //std::vector<std::vector<const char*>> gg = _m_dnnLabelListPtr;
    {
        renderImage(_renderImage);
    }

    std::vector<std::string> gyu= _m_classLabels;
    //for (size_t classIdx = 0; classIdx < _m_classLabels.size(); classIdx++)
    for (int classIdx = 0; classIdx < 5; classIdx++)
    {

        if (&_m_dnnBoxList[classIdx][0] == nullptr)
            continue;
 
        //CHECK_DW_ERROR(dwRenderEngine_setColor(DW_RENDER_ENGINE_COLOR_RED, m_renderEngine));
        CHECK_DW_ERROR(dwRenderEngine_setColor(m_boxColors[classIdx % MAX_BOX_COLORS], m_renderEngine));
        CHECK_DW_ERROR(dwRenderEngine_setLineWidth(2.0f, m_renderEngine));

        for (int jjj = 0; jjj<(int) _m_dnnBoxList[classIdx].size();++jjj)
        {
            CHECK_DW_ERROR(dwRenderEngine_render(DW_RENDER_ENGINE_PRIMITIVE_TYPE_BOXES_2D,
                            &_m_dnnBoxList[classIdx][jjj], sizeof(dwRectf), 0,
                            1, m_renderEngine));
        }
        //CHECK_DW_ERROR(dwRenderEngine_render(DW_RENDER_ENGINE_PRIMITIVE_TYPE_BOXES_2D,
        //                                       _m_dnnBoxList[classIdx].data(), sizeof(dwRectf), 0,
        //                                        _m_dnnBoxList.size(), m_renderEngine));

        CHECK_DW_ERROR(dwRenderEngine_setPointSize(2.0f, m_renderEngine));
        // Draw tracked features that belong to detected objects
        for (uint32_t boxIdx = 0U; boxIdx < _m_numTrackedBoxes[classIdx]; ++boxIdx) {
            const dwTrackedBox2D &trackedBox = _m_trackedBoxes[classIdx][boxIdx];
            {
                dwRenderEngine_render(DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_2D,
                                    trackedBox.featureLocations, sizeof(dwVector2f),
                                    0, trackedBox.nFeatures, m_renderEngine);
                                           
            }
            // Render box id
            dwVector2f pos{static_cast<float32_t>(trackedBox.box.x),
                            static_cast<float32_t>(trackedBox.box.y)};
            //std::string post(std::string(_m_dnnLabelListPtr[classIdx][0]) + std::string(": ") +std::to_string(trackedBox.id));
            //const char* postptr = post.c_str();
            dwRenderEngine_renderText2D(std::to_string(trackedBox.id).c_str(), pos, m_renderEngine);
            //dwRenderEngine_renderText2D(postptr, pos, m_renderEngine);
        }
    }

}

void StereoPilotNet::write2file(std::vector<std::vector<DepthMeasurement>>_objectDepth, std::vector<std::vector<TrackerMeasurement>>_objectTracker)
{
    for (int i =0;i<(int)_objectDepth.size();++i)
    {
        for(int j = 0; j<(int)_objectDepth[i].size();++j)
        {
            myfile<<"Depth: "<< _objectDepth[i][j].blockClass <<" "<<_objectDepth[i][j].azimuth<<" "<<_objectDepth[i][j].elevation<<" "
                  <<_objectDepth[i][j].depth<<" "<<_objectDepth[i][j].standardDeviationDepth<<" "<<_objectDepth[i][j].time<<"\n";
        }
    }

    for (int i =0;i<(int)_objectTracker.size();++i)
    {
        for(int j = 0; j<(int)_objectTracker[i].size();++j)
        {
            myfile<<"Track: "<< _objectTracker[i][j].blockClass <<" "<<_objectTracker[i][j].id<< " "<<_objectTracker[i][j].rateAzi<<" "
                  <<_objectTracker[i][j].rateEle<<" "<<_objectTracker[i][j].velocity<<" "
                  <<_objectTracker[i][j].azimuth<<" "<<_objectTracker[i][j].elevation<<" "
                  <<_objectTracker[i][j].depth<<" "<<_objectTracker[i][j].standardDeviationDepth<<" "<<_objectTracker[i][j].time<<"\n";
        }
    }

}

StereoPilotNet::~StereoPilotNet()
{
    delete inputHandler;
    delete stereoDisparity;
    myfile.close();
}



#endif