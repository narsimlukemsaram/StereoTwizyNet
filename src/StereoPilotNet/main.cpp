#include "StereoPilotNet.hpp"

using namespace dw_samples::common;

int main(int argc, const char** argv)
{
    // Program arguments
    std::string videosString = DataPath::get() + "/samples/raw/rccb.raw";
    videosString += "," + DataPath::get() + "/samples/raw/rccb.raw";
    videosString += "," + DataPath::get() + "/samples/raw/rccb.raw";
    videosString += "," + DataPath::get() + "/samples/raw/rccb.raw";

    ProgramArguments args(argc, argv,
                          {
#ifdef VIBRANTE
                              ProgramArguments::Option_t("camera-type", "ar0231-rccb-bae-sf3324", "camera gmsl type (see sample_sensors_info for all available camera types on this platform)"),
                              ProgramArguments::Option_t("slave", "0"),
                              ProgramArguments::Option_t("input-type", "video"),
                              ProgramArguments::Option_t("fifo-size", "5"),
                              ProgramArguments::Option_t("selector-mask", "0001"),
#endif
                              ProgramArguments::Option_t("videos", videosString.c_str()),
                              ProgramArguments::Option_t{"video0", (DataPath::get() + std::string{"/samples/stereo/left_1.h264"}).c_str(), "Left input video."},
                              ProgramArguments::Option_t{"video1", (DataPath::get() + std::string{"/samples/stereo/right_1.h264"}).c_str(), "Right input video."},
                              ProgramArguments::Option_t{"rigconfig", (DataPath::get() + "/samples/stereo/full.json").c_str(), "Rig configuration file."},
                              ProgramArguments::Option_t{"level","0","Log level"},
                              ProgramArguments::Option_t{"single_side", "0", "If set to 1 only left disparity is computed."},
                              ProgramArguments::Option_t("stopFrame", "0"),
                              ProgramArguments::Option_t("skipFrame", "0"), // 0 skip no frames, 1 skip 1 frame every 2, and so on
                              ProgramArguments::Option_t("maxDistance", "50.0"),
                              ProgramArguments::Option_t("dla", "0", "run inference on dla"),
                              ProgramArguments::Option_t("dlaEngineNo", "0", "dla engine to run DriveNet on if --dla=1"),
                              ProgramArguments::Option_t("enableFoveal", "0", "run drivenet in foveal mode."),
                              ProgramArguments::Option_t("precision", "fp32", "network precision. Possible options are \"int8\", \"fp16\" and \"fp32\"."),
                              ProgramArguments::Option_t("skipInference", "0")},
                              
                          "DriveWorks DrivenetNCameras sample");

    // -------------------
    // initialize and start a window application (with offscreen support if required)
    StereoPilotNet app(args);
    app.initializeWindow("DriveNet Simple", 1280, 800, args.enabled("offscreen"));
    app.setStopFrame(stoi(args.get("stopFrame")));
    //FreespaceModule freespace(args);

    //std::cout << bat.getX();
    return app.run();
   

}

