#ifndef DISPARITY_2_DEPTH_HPP
#define DISPARITY_2_DEPTH_HPP

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

class DepthMeasurement
{
    public:
        float depth                     = 0; // m
        float standardDeviationDisp     = 0;   
        float standardDeviationDepth    = 0; // m
        float azimuth                   = 0; // deg
        float elevation                 = 0; // deg
        int blockClass                  = -1; 
        long long time                  = 0;
};

class TrackerMeasurement
{
    public:
        float depth                     = 0; // m
        float standardDeviationDisp     = 0;   
        float standardDeviationDepth    = 0; // m
        float azimuth                   = 0; // deg
        float elevation                 = 0; // deg
        int blockClass                  = -1; 
        float velocity                  = 0; // m/s
        float rateAzi                   = 0; // deg/s
        float rateEle                   = 0; // deg/s
        double id                       = -1;
        long long time                  = 0;
};


class MeasurementPair
{
    public:
        std::vector<std::vector<dwRectf>>   firstTrackedBoxListFloat;
        std::vector<std::vector<dwTrackedBox2D >> firstTrackedBoxes ;
        std::vector<std::vector<DepthMeasurement>> firstDepthMeasurement;
        std::vector<std::vector<std::vector<dwVector2f>>> firstFeatureLocations;
        long long firstTime;

        std::vector<std::vector<dwRectf>>   secondTrackedBoxListFloat;
        std::vector<std::vector<dwTrackedBox2D >> secondTrackedBoxes ;
        std::vector<std::vector<DepthMeasurement>> secondDepthMeasurement;
        std::vector<std::vector<std::vector<dwVector2f>>> secondFeatureLocations;
        long long secondTime;

        std::vector<std::vector<TrackerMeasurement>> trackMeas;

};


class Disparity2Depth
{
    private:
        dwContextHandle_t m_sdk = DW_NULL_HANDLE;
        
        dwImageCUDA* inputDisparityImage;
        dwImageCPU* disparityImageCPU;
        dwImageProperties dispImagepropCuda, dispImagepropCPU;
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
        //std::vector<const dwTrackedBox2D *> m_trackedPointsInBox;
        

        dwImageHandle_t inputImageHandle, dispImageCPUHandle;
        
        MeasurementPair bufferMeasurements;

        

        float focalLength       = 1935.71f;// updated at initialize
        float baseline          = 0.3f;   // updated at initialize
        float cX                = 0.0;
        float cY                = 0.0;
        float cXL               = 0.0;
        float cXR               = 0.0;
        float rectfiedWidth     = 0.0;
        float rectfiedHeight    = 0.0;
        float focalLengthL      = 0.0;



    public:
        std::vector<std::vector<DepthMeasurement>> objectDepth;
        
        std::vector<std::vector<TrackerMeasurement>> objectTracker;
        Disparity2Depth();
        bool Initialize(float , float);
        bool Initialize(float , float, float, float, float, float, float, float, float);
        void Process(std::vector<std::vector<dwRectf>>, long long, std::vector<std::vector<int>>);
        void ProcessTrack(std::vector<std::vector<dwRectf>>, std::vector<const dwTrackedBox2D *>,std::vector<size_t> ,long long, std::vector<std::vector<int>>);

        void setContext(dwContextHandle_t  _m_context)
        {
            m_sdk =_m_context;
        }


        ~Disparity2Depth();
};

bool Disparity2Depth::Initialize(float _focalLength, float _baseLine)
{
    focalLength = _focalLength;
    baseline    = _baseLine;
    return true;

}

bool Disparity2Depth::Initialize(float _focalLength, float _baseLine, float _cX, float _cY, float _cXL, float _cXR, float _rectfiedWidth, float _rectfiedHeight , float _focalLengthL)
{
    focalLength     = _focalLength;
    baseline        = _baseLine;
    cX              = _cX;
    cY              = _cY;
    cXL             = _cXL;
    cXR             = _cXR;
    rectfiedWidth   = _rectfiedWidth;
    rectfiedHeight  = _rectfiedHeight;
    focalLengthL    = _focalLengthL;
    //firstFeatureLocations.resize(5);
    
    return true;

}

void Disparity2Depth::Process(std::vector<std::vector<dwRectf>> _m_dnnBoxList, long long _time, std::vector<std::vector<int>> disparityImageAsVector)
{
    //std::vector<std::vector<dwRectf>> jj = _m_dnnBoxList;
    { 
        objectDepth.clear();
        objectDepth.resize(_m_dnnBoxList.size());
        for (int i = 0; i<(int)_m_dnnBoxList.size();++i)
        {
            objectDepth[i].resize(_m_dnnBoxList[i].size());
            for (int j = 0; j<(int)_m_dnnBoxList[i].size();++j)
            {
                
                int32_t x1 = (int32_t)(_m_dnnBoxList[i][j].x + _m_dnnBoxList[i][j].width/5*2); //  x+width/4
                int32_t y1 = (int32_t)(_m_dnnBoxList[i][j].y + _m_dnnBoxList[i][j].height/5*2); // y+height/4
                int32_t width1  =  (int32_t)(_m_dnnBoxList[i][j].width/5); // 
                int32_t height1 = (int32_t)(_m_dnnBoxList[i][j].height/5) ;

                float avgdisp       = 0;
                double sumdisp    = 0.0;
                double sumdepth   = 0.0;
                std::vector<float> dispall;
                std::vector<float> depthall;

                /// read values of disparity in the detected box
                for (int r1 = y1; r1< y1 + height1; ++r1)
                {
                    for(int cl1 = x1; cl1< x1 + width1; ++cl1)
                    {
                        float correctedDisp = disparityImageAsVector[r1][cl1]-(cXL-cXR);
                        //float correctedDisp = disparityImageAsVector[r1][cl1];
                        if (correctedDisp>0)
                        {
                            
                            sumdisp   = sumdisp  + correctedDisp;
                            float deptheachpix = focalLength * baseline / (correctedDisp);

                            sumdepth  = sumdepth + deptheachpix;

                            dispall.push_back(correctedDisp);                        
                            depthall.push_back(deptheachpix);
                        }
                    }
                }

                avgdisp = sumdisp/dispall.size();
                //float deptht = focalLength * baseline / avgdisp;
                float deptht = focalLength * baseline / (avgdisp);

                double variancedisp     = 0.0;
                double variancedepth    = 0.0;
                double stdDeviDisp      = 0.0;
                double stdDeviDepth     = 0.0;

                for (int stdcal = 0; stdcal< (int)dispall.size(); ++stdcal)
                {
                    variancedisp  = variancedisp  + ((dispall[stdcal] - avgdisp) * (dispall[stdcal] - avgdisp));
                    variancedepth = variancedepth + ((depthall[stdcal] - deptht) * (depthall[stdcal] - deptht));
                }

                variancedisp  = variancedisp/dispall.size();
                variancedepth = variancedepth/depthall.size();

                stdDeviDisp         = sqrt(variancedisp);
                stdDeviDepth        = sqrt(variancedepth);

                objectDepth[i][j].depth                     = deptht;
                objectDepth[i][j].standardDeviationDisp     = stdDeviDisp;   
                objectDepth[i][j].standardDeviationDepth    = stdDeviDepth;  
                objectDepth[i][j].blockClass                = i; 
                objectDepth[i][j].time                      = _time;

                

                int32_t xmid = (int32_t)(_m_dnnBoxList[i][j].x +_m_dnnBoxList[i][j].width/2); 
                int32_t ymid = (int32_t)(_m_dnnBoxList[i][j].y +_m_dnnBoxList[i][j].height/2); 

                if(xmid >= rectfiedWidth/2)
                    objectDepth[i][j].azimuth = -(atan((xmid - rectfiedWidth/2)/focalLengthL))*180/M_PI; 
                if(xmid < rectfiedWidth/2)
                    objectDepth[i][j].azimuth =  (atan((rectfiedWidth/2 - xmid)/focalLengthL))*180/M_PI; 

                if(ymid >= rectfiedHeight/2)
                    objectDepth[i][j].elevation =-(atan((ymid - rectfiedHeight/2)/focalLengthL))*180/M_PI; 
                if(ymid < rectfiedHeight/2)
                    objectDepth[i][j].elevation = (atan((rectfiedHeight/2 - xmid)/focalLengthL))*180/M_PI; 

                //std::cout<<"Depth std class = "<<deptht<<" "<<stdDeviDisp<<" "<<stdDeviDepth<<" "<<objectDepth[i][j].elevation
                //         <<" "<<objectDepth[i][j].azimuth<<" "<<i<<" "<<depthall.size()<<"\n";

                depthall.clear();
                dispall.clear();

            }
        }      
    }
}







void Disparity2Depth::ProcessTrack(std::vector<std::vector<dwRectf>> _m_dnnBoxList, std::vector<const dwTrackedBox2D *> _m_trackedBoxes,
                                    std::vector<size_t> _m_numTrackedBoxes, long long _time, std::vector<std::vector<int>> disparityImageAsVector)
{
    std::vector<std::vector< dwTrackedBox2D>> b2b;
    std::vector<std::vector<std::vector<dwVector2f>>> locationsAll;
    objectTracker.clear(); 
    objectTracker.resize(5);
    
    
    for (int ccc=0; ccc < 5;++ccc)
    {
        std::vector<dwTrackedBox2D> vectTrackBox;
        
        std::vector<std::vector<dwVector2f>> locations;

        for(int tb2d =0 ; tb2d < (int)_m_numTrackedBoxes[ccc];++tb2d)
        {
            std::vector<dwVector2f> templocations;
            const dwTrackedBox2D &trackedBox = _m_trackedBoxes[ccc][tb2d];
            vectTrackBox.push_back(trackedBox); 
            //std::cout<<"bb.x = "<<trackedBox.box.x<<" "<<trackedBox.featureLocations[0]<<" "<<trackedBox.id<<"\n";
            dwVector2f templ;
            for (int ttt=0; ttt< (int)trackedBox.nFeatures;++ttt)
            {
                templ.x = trackedBox.featureLocations[2*ttt];
                templ.y = trackedBox.featureLocations[2*ttt+1];
                templocations.push_back(templ);
                //std::cout<<"\n "<<templ.x<<"\n";
            }
            locations.push_back(templocations);
            
            //buf_m_trackedBoxes[tb2d].push_back(trackedBox);
        }
        b2b.push_back(vectTrackBox);
        locationsAll.push_back(locations);
        
        //std::cout<<" vect array size = "<<locationsAll[ccc].size()<<" "<<_m_dnnBoxList[ccc].size()<<"\n";
    }

    
    bufferMeasurements.firstTrackedBoxes            = b2b;
    bufferMeasurements.firstTrackedBoxListFloat     = _m_dnnBoxList;
    bufferMeasurements.firstTime                    = _time;
    bufferMeasurements.firstFeatureLocations        = locationsAll;
    //std::cout<<"\n ************&&&&&&&&&&&&&&*************\n";
    //std::cout<<"bufferMeasurements.firstTrackedBoxes.size() = "<<bufferMeasurements.firstTrackedBoxes.size()<<" "<<bufferMeasurements.firstTrackedBoxes[0].size()
            //<<" "<<bufferMeasurements.firstTrackedBoxes[0][0].featureLocations[0] <<"\n";
    
    //std::vector<std::vector<dwRectf>> jj = _m_dnnBoxList;
    std::vector<std::vector<DepthMeasurement>> objectDepthtemp;
    { 

        objectDepthtemp.clear();
        objectDepthtemp.resize(_m_dnnBoxList.size());
        for (int i = 0; i<(int)_m_dnnBoxList.size();++i)
        {
            objectDepthtemp[i].resize(_m_dnnBoxList[i].size());
            for (int j = 0; j<(int)_m_dnnBoxList[i].size();++j)
            {
                
                int32_t x1 = (int32_t)(_m_dnnBoxList[i][j].x + _m_dnnBoxList[i][j].width/5*2); //  
                int32_t y1 = (int32_t)(_m_dnnBoxList[i][j].y + _m_dnnBoxList[i][j].height/5*2); // 
                int32_t width1  =  (int32_t)(_m_dnnBoxList[i][j].width/5); // 
                int32_t height1 = (int32_t)(_m_dnnBoxList[i][j].height/5) ;

                float avgdisp       = 0;
                double sumdisp    = 0.0;
                double sumdepth   = 0.0;
                std::vector<float> dispall;
                std::vector<float> depthall;

                /// read values of disparity in the detected box
                for (int r1 = y1; r1< y1 + height1; ++r1)
                {
                    for(int cl1 = x1; cl1< x1 + width1; ++cl1)
                    {
                        float correctedDisp = disparityImageAsVector[r1][cl1]-(cXL-cXR);
                        //float correctedDisp = disparityImageAsVector[r1][cl1];
                        if (correctedDisp>0)
                        {
                            
                            sumdisp   = sumdisp  + correctedDisp;
                            float deptheachpix = focalLength * baseline / (correctedDisp);

                            sumdepth  = sumdepth + deptheachpix;

                            dispall.push_back(correctedDisp);                        
                            depthall.push_back(deptheachpix);
                        }
                    }
                }

                avgdisp = sumdisp/dispall.size();
                //float deptht = focalLength * baseline / avgdisp;
                float deptht = focalLength * baseline / (avgdisp);

                double variancedisp     = 0.0;
                double variancedepth    = 0.0;
                double stdDeviDisp      = 0.0;
                double stdDeviDepth     = 0.0;

                for (int stdcal = 0; stdcal< (int)dispall.size(); ++stdcal)
                {
                    variancedisp  = variancedisp  + ((dispall[stdcal] - avgdisp) * (dispall[stdcal] - avgdisp));
                    variancedepth = variancedepth + ((depthall[stdcal] - deptht) * (depthall[stdcal] - deptht));
                }

                variancedisp  = variancedisp/dispall.size();
                variancedepth = variancedepth/depthall.size();

                stdDeviDisp         = sqrt(variancedisp);
                stdDeviDepth        = sqrt(variancedepth);

                objectDepthtemp[i][j].depth                     = deptht;
                objectDepthtemp[i][j].standardDeviationDisp     = stdDeviDisp;   
                objectDepthtemp[i][j].standardDeviationDepth    = stdDeviDepth;  
                objectDepthtemp[i][j].blockClass                = i; 

                

                int32_t xmid = (int32_t)(_m_dnnBoxList[i][j].x +_m_dnnBoxList[i][j].width/2); 
                int32_t ymid = (int32_t)(_m_dnnBoxList[i][j].y +_m_dnnBoxList[i][j].height/2); 

                if(xmid >= rectfiedWidth/2)
                    objectDepthtemp[i][j].azimuth = -(atan((xmid - rectfiedWidth/2)/focalLengthL))*180/M_PI; 
                if(xmid < rectfiedWidth/2)
                    objectDepthtemp[i][j].azimuth =  (atan((rectfiedWidth/2 - xmid)/focalLengthL))*180/M_PI; 

                if(ymid >= rectfiedHeight/2)
                    objectDepthtemp[i][j].elevation =-(atan((ymid - rectfiedHeight/2)/focalLengthL))*180/M_PI; 
                if(ymid < rectfiedHeight/2)
                    objectDepthtemp[i][j].elevation = (atan((rectfiedHeight/2 - xmid)/focalLengthL))*180/M_PI; 

                //std::cout<<"Depth std class = "<<deptht<<" "<<stdDeviDisp<<" "<<stdDeviDepth<<" "<<objectDepthtemp[i][j].elevation
                //         <<" "<<objectDepthtemp[i][j].azimuth<<" "<<i<<" "<<depthall.size()<<"\n";

                depthall.clear();
                dispall.clear();

            }
        }
        bufferMeasurements.firstDepthMeasurement = objectDepthtemp;          
    }


    if(bufferMeasurements.secondTrackedBoxes.size() != 0)
    {
        bufferMeasurements.trackMeas.resize(bufferMeasurements.secondTrackedBoxes.size());
        //std::cout<<" first and second id = "<<bufferMeasurements.firstTrackedBoxes[0][0].id<<" "<<bufferMeasurements.secondTrackedBoxes[0][0].id<<"\n";
        //std::cout<<" first and second id = "<<bufferMeasurements.firstTrackedBoxes[0][0].featureLocations[0]<<" "<<bufferMeasurements.firstTrackedBoxes[0][0].featureLocations[1]<<"\n";
        //std::cout<<" first and second id = "<<bufferMeasurements.firstTrackedBoxes[0][0].featureLocations[2]<<" "<<bufferMeasurements.firstTrackedBoxes[0][0].featureLocations[3]<<"\n";
        //std::cout<<" first and second id = "<<bufferMeasurements.firstTrackedBoxes[0][0].nFeatures<<" "<<bufferMeasurements.firstTrackedBoxes[0][1].nFeatures<<"\n";
        //std::cout<<" first and second id = "<<bufferMeasurements.secondTrackedBoxes[0][0].nFeatures<<" "<<bufferMeasurements.secondTrackedBoxes[0][1].nFeatures<<"\n";
        //std::cout<<"bufferMeasurements.secondTrackedBoxes.size() = "<<bufferMeasurements.secondTrackedBoxes.size()<<" "<<bufferMeasurements.firstTrackedBoxes.size()
        //    <<" " <<"\n";
        for(int i =0; i<(int)bufferMeasurements.secondTrackedBoxes.size();++i )
        {
            //std::cout<<"bufferMeasurements.secondTrackedBoxes["<<i<<"].size() = "<<bufferMeasurements.secondTrackedBoxes[i].size()<<" "<<bufferMeasurements.firstTrackedBoxes[i].size()
            //<<" " <<"\n";
            for(int j =0;j<(int)bufferMeasurements.secondTrackedBoxes[i].size();++j)
            {
                //float rateOfChangeAzi = 0;
                //float rateOfChangeEle = 0;

                for(int k =0; k<(int)bufferMeasurements.firstTrackedBoxes[i].size();++k)
                {
                    //std::cout<<"bufferMeasurements.secondTrackedBoxes["<<i<<"]["<<j<<"].id = "<<bufferMeasurements.secondTrackedBoxes[i][j].id<<" "<<bufferMeasurements.firstTrackedBoxes[i][k].id
                    //<<" " <<"\n";
                    if(bufferMeasurements.secondTrackedBoxes[i][j].id == bufferMeasurements.firstTrackedBoxes[i][k].id )
                    {
                        // compute change in azimuth, elevation and depth
                        //bufferMeasurements.secondTrackedBoxes[i][j].
                        //std::cout<<" second and first nfeatures = "<<bufferMeasurements.secondTrackedBoxes[i][j].nFeatures<<" "<<bufferMeasurements.firstTrackedBoxes[i][k].nFeatures<<"\n";
                        
                        //std::cout<<" first and second id = "<<bufferMeasurements.secondTrackedBoxes[i][j].nFeatures<<" "<<bufferMeasurements.secondTrackedBoxes[i][k].nFeatures<<"\n";
                        
                        float firstX  =0;
                        float secondX =0;
                        float firstY  =0;
                        float secondY =0;
                        int featureCounter = 0;
                        float ciX,ciY;
                        float totalX =0;
                        float totalY =0;
                        std::vector<float> changeX,changeY;

                        if (bufferMeasurements.secondTrackedBoxes[i][j].nFeatures <= bufferMeasurements.firstTrackedBoxes[i][k].nFeatures)
                            featureCounter = bufferMeasurements.secondTrackedBoxes[i][j].nFeatures;
                        else
                            featureCounter = bufferMeasurements.firstTrackedBoxes[i][k].nFeatures;

                            

                        for(int l = 0; l<(int)featureCounter;++l)
                        {
                            
                            secondX = bufferMeasurements.secondFeatureLocations[i][j][l].x;
                            secondY = bufferMeasurements.secondFeatureLocations[i][j][l].y;

                            firstX  = bufferMeasurements.firstFeatureLocations[i][k][l].x;
                            firstY  = bufferMeasurements.firstFeatureLocations[i][k][l].y;

                            //std::cout<<" second and first x y = "<<" "<<secondX<<" "<<secondY<<" "<<firstX<<" "<<firstY
                            //         <<" "<<bufferMeasurements.secondTrackedBoxes[i][j].id <<"\n";
                            //std::cout<<" box x y width height = "<<bufferMeasurements.secondTrackedBoxes[i][j].box.x<<" "
                            //         <<" "<<bufferMeasurements.secondTrackedBoxes[i][j].box.y
                            //         <<" "<<bufferMeasurements.secondTrackedBoxes[i][j].box.width
                            //         <<" "<<bufferMeasurements.secondTrackedBoxes[i][j].box.height<<"\n";


                            if(secondX>= bufferMeasurements.secondTrackedBoxes[i][j].box.x + bufferMeasurements.secondTrackedBoxes[i][j].box.width/5
                              && secondX<= bufferMeasurements.secondTrackedBoxes[i][j].box.x + bufferMeasurements.secondTrackedBoxes[i][j].box.width*4/5
                              && secondY>= bufferMeasurements.secondTrackedBoxes[i][j].box.y + bufferMeasurements.secondTrackedBoxes[i][j].box.height/5
                              && secondY<= bufferMeasurements.secondTrackedBoxes[i][j].box.y + bufferMeasurements.secondTrackedBoxes[i][j].box.height*4/5)
                            {
                                ciX = secondX - firstX;
                                ciY = secondY - firstY;
                                totalX = ciX + totalX;
                                totalY = ciY + totalY;

                                changeX.push_back(ciX);
                                changeY.push_back(ciY);
                            }
                            

                        }
                        //std::cout<<"Size of pixels considered for tracking inside bb = "<<changeX.size()<<"\n";

                        // simple avg for now

                        float avgAzt =totalX/changeX.size();
                        float avgEle = totalY/changeY.size();

                        float changeAztDeg = avgAzt*((atan(rectfiedWidth/2/focalLengthL)*180/M_PI)/rectfiedWidth/2);
                        float changeEleDeg = avgEle*((atan(rectfiedHeight/2/focalLengthL)*180/M_PI)/rectfiedHeight/2);

                        float changetime = (float)(bufferMeasurements.firstTime - bufferMeasurements.secondTime)/1000000;

                        float rateChangeAztDeg = changeAztDeg/changetime;
                        float rateChangeEleDeg = changeEleDeg/changetime;

                        float changedepth = (bufferMeasurements.secondDepthMeasurement[i][j].depth - bufferMeasurements.firstDepthMeasurement[i][j].depth);
                        float ratechandedepth = changedepth/changetime;

                        TrackerMeasurement tempTrackMeas;
                        tempTrackMeas.rateAzi                   = rateChangeAztDeg;
                        tempTrackMeas.rateEle                   = rateChangeEleDeg;
                        tempTrackMeas.velocity                  = ratechandedepth;
                        tempTrackMeas.id                        = bufferMeasurements.secondTrackedBoxes[i][j].id;
                        tempTrackMeas.depth                     = bufferMeasurements.firstDepthMeasurement[i][k].depth; // m
                        tempTrackMeas.standardDeviationDisp     = bufferMeasurements.firstDepthMeasurement[i][k].standardDeviationDisp;   
                        tempTrackMeas.standardDeviationDepth    = bufferMeasurements.firstDepthMeasurement[i][k].standardDeviationDepth; // m
                        tempTrackMeas.azimuth                   = bufferMeasurements.firstDepthMeasurement[i][k].azimuth; // deg
                        tempTrackMeas.elevation                 = bufferMeasurements.firstDepthMeasurement[i][k].elevation; // deg
                        tempTrackMeas.blockClass                = bufferMeasurements.firstDepthMeasurement[i][k].blockClass; 
                        tempTrackMeas.time                      = _time;

                        bufferMeasurements.trackMeas[i].push_back(tempTrackMeas);
                        objectTracker[i].push_back(tempTrackMeas);
                        //std::cout<<"change in time= "<<changetime<<" "<<bufferMeasurements.firstTime<<" "<<bufferMeasurements.secondTime<<"\n";

                        //std::cout<<"TRACK MEASUREMENTS = "<< tempTrackMeas.rateAzi <<" "<<tempTrackMeas.rateEle<<" "<<tempTrackMeas.velocity<<" "<<tempTrackMeas.id
                        //                                   <<" "<<tempTrackMeas.depth<<" "<<tempTrackMeas.standardDeviationDisp<<" "<<tempTrackMeas.standardDeviationDepth
                        //                                   <<" "<<tempTrackMeas.azimuth<<" "<<tempTrackMeas.elevation<<" "<<tempTrackMeas.blockClass<<"\n";                  
                        
                        break;

                    }
                }
            }
        }

    }

    bufferMeasurements.secondTrackedBoxes           = bufferMeasurements.firstTrackedBoxes;
    bufferMeasurements.secondTrackedBoxListFloat    = bufferMeasurements.firstTrackedBoxListFloat;
    bufferMeasurements.secondDepthMeasurement       = bufferMeasurements.firstDepthMeasurement;
    bufferMeasurements.secondFeatureLocations       = bufferMeasurements.firstFeatureLocations;
    bufferMeasurements.secondTime                   = bufferMeasurements.firstTime;
}

Disparity2Depth::Disparity2Depth(/* args */)
{
}

Disparity2Depth::~Disparity2Depth()
{
}





#endif