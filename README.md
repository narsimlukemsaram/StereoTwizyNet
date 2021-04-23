Architecture Design and Development of an On-board Stereo Vision System with Motion Estimation for Cooperative Automated Vehicles: Developed an on-board stereo vision system to estimate the detected object‘s motion parameters (distance, velocity, azimuth, and elevation angles) in C++ on Ubuntu 16.04 LTS, and it is deployed on Nvidia Drive PX2 embedded hardware platform with a custom-built GMSL stereo camera using Nvidia DriveWorks 1.2, CUDA 9.2, TensorRT 4.0, and cuDNN 7.4.1 libraries, and evaluated on the CARLA Simulator and the Renault Twizy Cooperative Automated Vehicle research platform.
@This is the implementation of "Architecture Design and Development of an On-board Stereo Vision System for Cooperative Automated Vehicles", Narsimlu Kemsaram, Anweshan Das, Gijs Dubbelman.

## Prerequisites

### Basic Hardware Requirements

- NVIDIA DRIVE<sup>&trade;</sup> PX 2 platform with the latest PDK flashed in the system.
- Desktop PC running Linux x86/x64

### Linux System Requirements

These are the basic prerequisites for Linux:

- Ubuntu Linux 16.04 or 14.04 (out of the box installation)
- GCC >= 4.8.X && GCC <= 4.9.x (for 16.04, use GCC 4.9.3 or greater)
- cmake version >= 3.3

  By default, Ubuntu 14.04 installs cmake version 2.8. For guidance on
  installing cmake 3.x, see:<br> <a href="http://askubuntu.com/questions/610291/
  how-to-install-cmake-3-2-on-ubuntu-14-04"> http://askubuntu.com/questions/6102
  91/how-to-install-cmake-3-2-on-ubuntu-14-04</a>

- NVIDIA<sup>&reg;</sup> CUDA<sup>&reg;</sup> Toolkit version 9.0 or later
- NVIDIA DRIVE<sup>&trade;</sup> 5.0 Linux SDK/PDK installation on the Linux Host
- You may also need to install (using `apt-get install`) the following packages:

       libx11-dev
       libxrandr-dev
       libxcursor-dev
       libxxf86vm-dev
       libxinerama-dev
       libxi-dev
       libglu1-mesa-dev
       libglew-dev

Desktop development relies on NVCUVID for video decoding, which is included with
the NVIDIA drivers. In general, the `cmake` build scripts can find NVCUVID
installation. However, if this fails, you must set a symbolic link
`/usr/lib/nvidia-current` pointing to your NVIDIA driver lib, for instance to
`/usr/lib/nvidia-384`.

<a name="building_on_linux">
## Building on Linux Desktop or Linux x86/x64 Only

The DriveWorks SDK samples use a standard cmake-based build system. The default
approach is to create a build folder for an "out of the source tree build"
and to point `cmake` to build the samples from there.

### To build the DriveWorks samples
* On the host system, enter:

      $ mkdir build
      $ cd build
      $ cmake -DCMAKE_BUILD_TYPE=Release /path/to/driveworks/samples/folder
      $ make -j

<a name="cross-compilation">
## Cross-Compiling for the NVIDIA DRIVE Platform

To compile the samples or any user code using DriveWorks, you must use
cross-compilation from a Linux host. For this, in addition to the requirements
above, you must have the additional components:

- DriveWorks SDK cross-compilation package
- NVIDIA DRIVE SDK/PDK installation for NVIDIA DRIVE PX 2 in the Linux Host
- CUDA cross-compilation libraries.

### To cross compile the DriveWorks SDK samples

1. On the host system, install CUDA cross-compilation libraries.

       $ sudo dpkg --add-architecture arm64
       $ sudo apt-get update
       $ sudo apt-get install cuda-cross-aarch64-8-0

   @note Some repositories may not have binaries for arm64 and apt-get
   update causes errors. To prevent these errors, edit
   `/etc/apt/sources.list` and files under `/etc/apt/sources.list.d/` and add
   `[arch=amd64,i386]` into each line starting with `deb`. Ensure you specify
   `amd64` and not `arm64`.

2. Configure the location of the NVIDIA DRIVE SDK/PDK.

   Set the environment variable `DriveSDK` to point to the SDK/PDK folders:

       /path/to/drive-t186ref-linux (NVIDIA DRIVE PX 2)

   -or-

   Define this location as a `cmake` parameter:

       -DVIBRANTE_PDK:STRING=/path/to/drive-t186ref-linux

3. Cross-compile.

       $ mkdir build
       $ cd build
       $ cmake -DCMAKE_BUILD_TYPE=Release \
             -DCMAKE_TOOLCHAIN_FILE=/path/to/samples/cmake/Toolchain-V4L.cmake \
             -DVIBRANTE_PDK:STRING=/path/to/drive-t186ref-linux \
              /path/to/driveworks/samples/folder
       $ make -j
       $ make install

   To streamline the deploy process, you can set the `cmake` definitions as
   environment variables.

<a name="load">
## Loading the Sample on the NVIDIA DRIVE Platform

After cross-compiling your custom DriveWorks sample on the Linux host
system, you must load it to the target platform. You can use automatic
upload (recommended) or manually upload it.

The package-based installation will install libraries, binaries and resources in
`/usr/local/driveworks-<version>`. However, if you build a custom DriveWorks
sample, you should upload those files to a separate directory such as:

    /home/nvidia/driveworks

The above path assumes your user name is "nvidia". Of course, you would use
a name more appropriate to your situation. Using this path rather than
`/usr/local/<name>` avoids security constraints that can make the upload process
more difficult.

@note *Important*: The correct `rpath` for the binaries is set only during
installation. As such, NVIDIA discourages copying pre-install binaries to the
board because this could result in undefined behavior.

### Automatic Upload (Recommended)

An `upload` target is provided as a convenience. Invoking this target uses
`rsync` to sync the builds to a specified location on the device. `target` is
set to depend on building and packaging.

The following cmake parameters configure the `upload` target:

* `VIBRANTE_HOST` : IP address of the board, the default value is
  `192.168.10.10`.
* `VIBRANTE_INSTALL_PATH` : destination folder of the compiled products. The
   default value is `/home/nvidia/driveworks`.
* `VIBRANTE_USER` : Specifies the user that will be used to `rsync` the
   binaries. The default value is `nvidia`.
* `VIBRANTE_PASSWORD` : Password for `VIBRANTE_USER`. The default value is
  `nvidia`.
* `VIBRANTE_PORT` : Port on which `ssh` is listening on the target board. The
  default is `22`.

### To invoke the upload target

1. Cross-compile the binaries as described in
   [Cross Compiling for the NVIDIA DRIVE Platform](#cross-compilation).

2. On the Linux system, enter:

       $ make upload

@note If the `VIBRANTE_HOST` parameter is set to the IP address of a running
NVIDIA DRIVE PX 2 board, users can call the upload target to `rsync` the build folder
with a folder on the board.

### Manual Upload

When you manually upload the sample to the target device, you copy the cross-
compiled binaries to a device with IP `DPX_IP` as user `nvidia`.

### To manually upload

1. Cross-compile the binaries as described in
   [Cross Compiling for the NVIDIA DRIVE Platform](#cross-compilation)

2. On the host system, execute:

        $ scp -r install/bin/* nvidia@DPX_IP:/home/nvidia/<destination_dir>/

<a name=device_compilation>
## Compiling on the Device
 
## Related Publications
### What to cite
If you use this code for your research, please kindly cite:

@INPROCEEDINGS{9294435,  author={Kemsaram, Narsimlu and Das, Anweshan and Dubbelman, Gijs},  booktitle={2020 IEEE 23rd International Conference on Intelligent Transportation Systems (ITSC)},   title={Architecture Design and Development of an On-board Stereo Vision System for Cooperative Automated Vehicles},   year={2020},  volume={},  number={},  pages={1-8},  doi={10.1109/ITSC45102.2020.9294435}}
