# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
# FindCUDA
# --------
#
# Tools for building CUDA C files: libraries and build dependencies.
#
# This script locates the NVIDIA CUDA C tools.  It should work on linux,
# windows, and mac and should be reasonably up to date with CUDA C
# releases.
#
# This script makes use of the standard find_package arguments of
# <VERSION>, REQUIRED and QUIET.  CUDA_FOUND will report if an
# acceptable version of CUDA was found.
#
# The script will prompt the user to specify CUDA_TOOLKIT_ROOT_DIR if
# the prefix cannot be determined by the location of nvcc in the system
# path and REQUIRED is specified to find_package().  To use a different
# installed version of the toolkit set the environment variable
# CUDA_BIN_PATH before running cmake (e.g.
# CUDA_BIN_PATH=/usr/local/cuda1.0 instead of the default
# /usr/local/cuda) or set CUDA_TOOLKIT_ROOT_DIR after configuring.  If
# you change the value of CUDA_TOOLKIT_ROOT_DIR, various components that
# depend on the path will be relocated.
#
# It might be necessary to set CUDA_TOOLKIT_ROOT_DIR manually on certain
# platforms, or to use a cuda runtime not installed in the default
# location.  In newer versions of the toolkit the cuda library is
# included with the graphics driver- be sure that the driver version
# matches what is needed by the cuda runtime version.
#
# The following variables affect the behavior of the macros in the
# script (in alphebetical order).  Note that any of these flags can be
# changed multiple times in the same directory before calling
# CUDA_ADD_EXECUTABLE, CUDA_ADD_LIBRARY, CUDA_COMPILE, CUDA_COMPILE_PTX,
# CUDA_COMPILE_FATBIN, CUDA_COMPILE_CUBIN or CUDA_WRAP_SRCS::
#
#   CUDA_64_BIT_DEVICE_CODE (Default matches host bit size)
#   -- Set to ON to compile for 64 bit device code, OFF for 32 bit device code.
#      Note that making this different from the host code when generating object
#      or C files from CUDA code just won't work, because size_t gets defined by
#      nvcc in the generated source.  If you compile to PTX and then load the
#      file yourself, you can mix bit sizes between device and host.
#
#   CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE (Default ON)
#   -- Set to ON if you want the custom build rule to be attached to the source
#      file in Visual Studio.  Turn OFF if you add the same cuda file to multiple
#      targets.
#
#      This allows the user to build the target from the CUDA file; however, bad
#      things can happen if the CUDA source file is added to multiple targets.
#      When performing parallel builds it is possible for the custom build
#      command to be run more than once and in parallel causing cryptic build
#      errors.  VS runs the rules for every source file in the target, and a
#      source can have only one rule no matter how many projects it is added to.
#      When the rule is run from multiple targets race conditions can occur on
#      the generated file.  Eventually everything will get built, but if the user
#      is unaware of this behavior, there may be confusion.  It would be nice if
#      this script could detect the reuse of source files across multiple targets
#      and turn the option off for the user, but no good solution could be found.
#
#   CUDA_BUILD_CUBIN (Default OFF)
#   -- Set to ON to enable and extra compilation pass with the -cubin option in
#      Device mode. The output is parsed and register, shared memory usage is
#      printed during build.
#
#   CUDA_BUILD_EMULATION (Default OFF for device mode)
#   -- Set to ON for Emulation mode. -D_DEVICEEMU is defined for CUDA C files
#      when CUDA_BUILD_EMULATION is TRUE.
#
#   CUDA_GENERATED_OUTPUT_DIR (Default CMAKE_CURRENT_BINARY_DIR)
#   -- Set to the path you wish to have the generated files placed.  If it is
#      blank output files will be placed in CMAKE_CURRENT_BINARY_DIR.
#      Intermediate files will always be placed in
#      CMAKE_CURRENT_BINARY_DIR/CMakeFiles.
#
#   CUDA_HOST_COMPILATION_CPP (Default ON)
#   -- Set to OFF for C compilation of host code.
#
#   CUDA_HOST_COMPILER (Default CMAKE_CXX_COMPILER, $(VCInstallDir)/bin for VS)
#   -- Set the host compiler to be used by nvcc. In case ccache or pynvccache is
#      used as a host C++ compiler the specified C++ compiler from it's argument
#      is used. Ignored if -ccbin or --compiler-bindir is already present in the
#      CUDA_NVCC_FLAGS or CUDA_NVCC_FLAGS_<CONFIG> variables.  For Visual Studio
#      targets $(VCInstallDir)/bin is a special value that expands out to the
#      path when the command is run from within VS.
#
#   CUDA_USE_PYNVCCCACHE (Default OFF)
#   -- Use pynvccache as cuda compilation cache if available and dependencies
#      are fulfilled. Otherwise fall-back to regular non-cached cuda compilation.
#
#   CUDA_DETERMINE_HOST_GPU_CODE_FLAGS (Default ON)
#   -- Determine GPU architecture of CUDAS GPUs in host system. Saved to
#      variable CUDA_HOST_GPU_CODE_FLAGS (e.g., '--gpu-code=sm_30,sm_61') to be
#      used in NVCC command line arguments.
#
#   CUDA_NON_PROPAGATED_HOST_FLAGS
#   -- If CUDA_PROPAGATE_HOST_FLAGS is set to ON, a list of flags which
#      will not be propagated. If CUDA_PROPAGATE_HOST_FLAGS is OFF, this
#      variable is ignored.
#
#   CUDA_NVCC_FLAGS
#   CUDA_NVCC_FLAGS_<CONFIG>
#   -- Additional NVCC command line arguments.  NOTE: multiple arguments must be
#      semi-colon delimited (e.g. --compiler-options;-Wall)
#
#   CUDA_PROPAGATE_HOST_FLAGS (Default ON)
#   -- Set to ON to propagate CMAKE_{C,CXX}_FLAGS and their configuration
#      dependent counterparts (e.g. CMAKE_C_FLAGS_DEBUG) automatically to the
#      host compiler through nvcc's -Xcompiler flag.  This helps make the
#      generated host code match the rest of the system better.  Sometimes
#      certain flags give nvcc problems, and this will help you turn the flag
#      propagation off.  This does not affect the flags supplied directly to nvcc
#      via CUDA_NVCC_FLAGS or through the OPTION flags specified through
#      CUDA_ADD_LIBRARY, CUDA_ADD_EXECUTABLE, or CUDA_WRAP_SRCS.  Flags used for
#      shared library compilation are not affected by this flag.
#
#   CUDA_SEPARABLE_COMPILATION (Default OFF)
#   -- If set this will enable separable compilation for all CUDA runtime object
#      files.  If used outside of CUDA_ADD_EXECUTABLE and CUDA_ADD_LIBRARY
#      (e.g. calling CUDA_WRAP_SRCS directly),
#      CUDA_COMPUTE_SEPARABLE_COMPILATION_OBJECT_FILE_NAME and
#      CUDA_LINK_SEPARABLE_COMPILATION_OBJECTS should be called.
#
#   CUDA_VERBOSE_BUILD (Default OFF)
#   -- Set to ON to see all the commands used when building the CUDA file.  When
#      using a Makefile generator the value defaults to VERBOSE (run make
#      VERBOSE=1 to see output), although setting CUDA_VERBOSE_BUILD to ON will
#      always print the output.
#
# The script creates the following macros (in alphebetical order)::
#
#   CUDA_ADD_CUFFT_TO_TARGET( cuda_target )
#   -- Adds the cufft library to the target (can be any target).  Handles whether
#      you are in emulation mode or not.
#
#   CUDA_ADD_CUBLAS_TO_TARGET( cuda_target )
#   -- Adds the cublas library to the target (can be any target).  Handles
#      whether you are in emulation mode or not.
#
#   CUDA_ADD_EXECUTABLE( cuda_target file0 file1 ...
#                        [WIN32] [MACOSX_BUNDLE] [EXCLUDE_FROM_ALL] [OPTIONS ...] )
#   -- Creates an executable "cuda_target" which is made up of the files
#      specified.  All of the non CUDA C files are compiled using the standard
#      build rules specified by CMAKE and the cuda files are compiled to object
#      files using nvcc and the host compiler.  In addition CUDA_INCLUDE_DIRS is
#      added automatically to include_directories().  Some standard CMake target
#      calls can be used on the target after calling this macro
#      (e.g. set_target_properties and target_link_libraries), but setting
#      properties that adjust compilation flags will not affect code compiled by
#      nvcc.  Such flags should be modified before calling CUDA_ADD_EXECUTABLE,
#      CUDA_ADD_LIBRARY or CUDA_WRAP_SRCS.
#      To keep the dependencies of the object files up to date, you must call
#      ADD_CUDA_DEPENDENCIES_TARGET() and build that target.
#
#   CUDA_ADD_LIBRARY( cuda_target file0 file1 ...
#                     [STATIC | SHARED | MODULE] [EXCLUDE_FROM_ALL] [OPTIONS ...] )
#   -- Same as CUDA_ADD_EXECUTABLE except that a library is created.
#
#   ADD_CUDA_DEPENDENCIES_TARGET(target_name is_all)
#   -- creates a custom target that when built will
#      update all the dependencies of the cuda files added so far.
#      If is_all is true then the target is added to the ALL group.
#      This function can be called multiple times, each time it will create a new
#      target that updates only the deps of the files added since the last call.
#
#         This function uses the following variables constructed by CUDA_WRAP_SRCS:
#           CUDA_DEPENDENCY_SCRIPTS,
#           CUDA_DEPENDENCY_OUT_FILES,
#           CUDA_DEPENDENCY_FLAGS,
#           CUDA_DEPENDENCY_DEPEND_FILES
#           CUDA_DEPENDENCY_DIRS
#         They are used to create the custom commands to update dependencies. These
#         variables have to be empty before the first time CUDA_WRAP_SRCS is called.
#         The function CUDA_RESET_INTERNAL_CACHE() can be used to clear them.
#
#   CUDA_RESET_INTERNAL_CACHE()
#   -- Resets all internal cache variables. Should be called at the end of the main
#      CMake script.
#
#   CUDA_BUILD_CLEAN_TARGET()
#   -- Creates a convience target that deletes all the dependency files
#      generated.  You should make clean after running this target to ensure the
#      dependency files get regenerated.
#
#   CUDA_COMPILE( generated_files file0 file1 ... [STATIC | SHARED | MODULE]
#                 [OPTIONS ...] )
#   -- Returns a list of generated files from the input source files to be used
#      with ADD_LIBRARY or ADD_EXECUTABLE.
#
#   CUDA_COMPILE_PTX( generated_files file0 file1 ... [OPTIONS ...] )
#   -- Returns a list of PTX files generated from the input source files.
#
#   CUDA_COMPILE_FATBIN( generated_files file0 file1 ... [OPTIONS ...] )
#   -- Returns a list of FATBIN files generated from the input source files.
#
#   CUDA_COMPILE_CUBIN( generated_files file0 file1 ... [OPTIONS ...] )
#   -- Returns a list of CUBIN files generated from the input source files.
#
#   CUDA_COMPUTE_SEPARABLE_COMPILATION_OBJECT_FILE_NAME( output_file_var
#                                                        cuda_target
#                                                        object_files )
#   -- Compute the name of the intermediate link file used for separable
#      compilation.  This file name is typically passed into
#      CUDA_LINK_SEPARABLE_COMPILATION_OBJECTS.  output_file_var is produced
#      based on cuda_target the list of objects files that need separable
#      compilation as specified by object_files.  If the object_files list is
#      empty, then output_file_var will be empty.  This function is called
#      automatically for CUDA_ADD_LIBRARY and CUDA_ADD_EXECUTABLE.  Note that
#      this is a function and not a macro.
#
#   CUDA_INCLUDE_DIRECTORIES( path0 path1 ... )
#   -- Sets the directories that should be passed to nvcc
#      (e.g. nvcc -Ipath0 -Ipath1 ... ). These paths usually contain other .cu
#      files.
#
#
#
#   CUDA_LINK_SEPARABLE_COMPILATION_OBJECTS( output_file_var cuda_target
#                                            nvcc_flags object_files)
#
#   -- Generates the link object required by separable compilation from the given
#      object files.  This is called automatically for CUDA_ADD_EXECUTABLE and
#      CUDA_ADD_LIBRARY, but can be called manually when using CUDA_WRAP_SRCS
#      directly.  When called from CUDA_ADD_LIBRARY or CUDA_ADD_EXECUTABLE the
#      nvcc_flags passed in are the same as the flags passed in via the OPTIONS
#      argument.  The only nvcc flag added automatically is the bitness flag as
#      specified by CUDA_64_BIT_DEVICE_CODE.  Note that this is a function
#      instead of a macro.
#
#   CUDA_WRAP_SRCS ( cuda_target format generated_files file0 file1 ...
#                    [STATIC | SHARED | MODULE] [OPTIONS ...] )
#   -- This is where all the magic happens.  CUDA_ADD_EXECUTABLE,
#      CUDA_ADD_LIBRARY, CUDA_COMPILE, and CUDA_COMPILE_PTX all call this
#      function under the hood.
#
#      Given the list of files (file0 file1 ... fileN) this macro generates
#      custom commands that generate either PTX or linkable objects (use "PTX" or
#      "OBJ" for the format argument to switch).  Files that don't end with .cu
#      or have the HEADER_FILE_ONLY property are ignored.
#
#      The arguments passed in after OPTIONS are extra command line options to
#      give to nvcc.  You can also specify per configuration options by
#      specifying the name of the configuration followed by the options.  General
#      options must precede configuration specific options.  Not all
#      configurations need to be specified, only the ones provided will be used.
#
#         OPTIONS -DFLAG=2 "-DFLAG_OTHER=space in flag"
#         DEBUG -g
#         RELEASE --use_fast_math
#         RELWITHDEBINFO --use_fast_math;-g
#         MINSIZEREL --use_fast_math
#
#      For certain configurations (namely VS generating object files with
#      CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE set to ON), no generated file will
#      be produced for the given cuda file.  This is because when you add the
#      cuda file to Visual Studio it knows that this file produces an object file
#      and will link in the resulting object file automatically.
#
#      This script will also generate a separate cmake script that is used at
#      build time to invoke nvcc.  This is for several reasons.
#
#        1. nvcc can return negative numbers as return values which confuses
#        Visual Studio into thinking that the command succeeded.  The script now
#        checks the error codes and produces errors when there was a problem.
#
#        2. nvcc has been known to not delete incomplete results when it
#        encounters problems.  This confuses build systems into thinking the
#        target was generated when in fact an unusable file exists.  The script
#        now deletes the output files if there was an error.
#
#        3. By putting all the options that affect the build into a file and then
#        make the build rule dependent on the file, the output files will be
#        regenerated when the options change.
#
#      This script also looks at optional arguments STATIC, SHARED, or MODULE to
#      determine when to target the object compilation for a shared library.
#      BUILD_SHARED_LIBS is ignored in CUDA_WRAP_SRCS, but it is respected in
#      CUDA_ADD_LIBRARY.  On some systems special flags are added for building
#      objects intended for shared libraries.  A preprocessor macro,
#      <target_name>_EXPORTS is defined when a shared library compilation is
#      detected.
#
#      Flags passed into add_definitions with -D or /D are passed along to nvcc.
#
#
#
# The script defines the following variables::
#
#   CUDA_VERSION_MAJOR    -- The major version of cuda as reported by nvcc.
#   CUDA_VERSION_MINOR    -- The minor version.
#   CUDA_VERSION
#   CUDA_VERSION_STRING   -- CUDA_VERSION_MAJOR.CUDA_VERSION_MINOR
#
#   CUDA_TOOLKIT_ROOT_DIR -- Path to the CUDA Toolkit (defined if not set).
#   CUDA_SDK_ROOT_DIR     -- Path to the CUDA SDK.  Use this to find files in the
#                            SDK.  This script will not directly support finding
#                            specific libraries or headers, as that isn't
#                            supported by NVIDIA.  If you want to change
#                            libraries when the path changes see the
#                            FindCUDA.cmake script for an example of how to clear
#                            these variables.  There are also examples of how to
#                            use the CUDA_SDK_ROOT_DIR to locate headers or
#                            libraries, if you so choose (at your own risk).
#   CUDA_INCLUDE_DIRS     -- Include directory for cuda headers.  Added automatically
#                            for CUDA_ADD_EXECUTABLE and CUDA_ADD_LIBRARY.
#   CUDA_LIBRARIES        -- Cuda RT library.
#   CUDA_CUFFT_LIBRARIES  -- Device or emulation library for the Cuda FFT
#                            implementation (alternative to:
#                            CUDA_ADD_CUFFT_TO_TARGET macro)
#   CUDA_CUBLAS_LIBRARIES -- Device or emulation library for the Cuda BLAS
#                            implementation (alterative to:
#                            CUDA_ADD_CUBLAS_TO_TARGET macro).
#   CUDA_cupti_LIBRARY    -- CUDA Profiling Tools Interface library.
#                            Only available for CUDA version 4.0+.
#   CUDA_curand_LIBRARY   -- CUDA Random Number Generation library.
#                            Only available for CUDA version 3.2+.
#   CUDA_cusolver_LIBRARY -- CUDA Direct Solver library.
#                            Only available for CUDA version 7.0+.
#   CUDA_cusparse_LIBRARY -- CUDA Sparse Matrix library.
#                            Only available for CUDA version 3.2+.
#   CUDA_npp_LIBRARY      -- NVIDIA Performance Primitives lib.
#                            Only available for CUDA version 4.0+.
#   CUDA_nppc_LIBRARY     -- NVIDIA Performance Primitives lib (core).
#                            Only available for CUDA version 5.5+.
#   CUDA_nppi_LIBRARY     -- NVIDIA Performance Primitives lib (image processing).
#                            Only available for CUDA version 5.5+.
#   CUDA_npps_LIBRARY     -- NVIDIA Performance Primitives lib (signal processing).
#                            Only available for CUDA version 5.5+.
#   CUDA_nvcuvenc_LIBRARY -- CUDA Video Encoder library.
#                            Only available for CUDA version 3.2+.
#                            Windows only.
#   CUDA_nvcuvid_LIBRARY  -- CUDA Video Decoder library.
#                            Only available for CUDA version 3.2+.
#                            Windows only.
#   CUDA_culibos_LIBRARY  -- CUDA Thread Abstraction Layer library
#                            Needed for static linking
#
#   James Bigler, NVIDIA Corp (nvidia.com - jbigler)
#   Abe Stephens, SCI Institute -- http://www.sci.utah.edu/~abe/FindCuda.html
#
#   Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
#
#   Copyright (c) 2007-2009
#   Scientific Computing and Imaging Institute, University of Utah
#
#   This code is licensed under the MIT License.  See the FindCUDA.cmake script
#   for the text of the license.

# The MIT License
#
# License for the specific language governing rights and limitations under
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
###############################################################################

# FindCUDA.cmake

# This macro helps us find the location of helper files we will need the full path to
macro(CUDA_FIND_HELPER_FILE _name _extension)
  set(_full_name "${_name}.${_extension}")
  # CMAKE_CURRENT_LIST_FILE contains the full path to the file currently being
  # processed.  Using this variable, we can pull out the current path, and
  # provide a way to get access to the other files we need local to here.
  get_filename_component(CMAKE_CURRENT_LIST_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
  set(CUDA_${_name} "${CMAKE_CURRENT_LIST_DIR}/FindCUDA/${_full_name}")
  if(NOT EXISTS "${CUDA_${_name}}")
    set(error_message "${_full_name} not found in ${CMAKE_CURRENT_LIST_DIR}/FindCUDA")
    if(CUDA_FIND_REQUIRED)
      message(FATAL_ERROR "${error_message}")
    else()
      if(NOT CUDA_FIND_QUIETLY)
        message(STATUS "${error_message}")
      endif()
    endif()
  endif()
  # Set this variable as internal, so the user isn't bugged with it.
  set(CUDA_${_name} ${CUDA_${_name}} CACHE INTERNAL "Location of ${_full_name}" FORCE)
endmacro()

#####################################################################
## CUDA_INCLUDE_NVCC_DEPENDENCIES
##

# So we want to try and include the dependency file if it exists.  If
# it doesn't exist then we need to create an empty one, so we can
# include it.

# If it does exist, then we need to check to see if all the files it
# depends on exist.  If they don't then we should clear the dependency
# file and regenerate it later.  This covers the case where a header
# file has disappeared or moved.

macro(CUDA_INCLUDE_NVCC_DEPENDENCIES dependency_file)
  set(CUDA_NVCC_DEPEND)
  set(CUDA_NVCC_DEPEND_REGENERATE FALSE)


  # Include the dependency file.  Create it first if it doesn't exist .  The
  # INCLUDE puts a dependency that will force CMake to rerun and bring in the
  # new info when it changes.  DO NOT REMOVE THIS (as I did and spent a few
  # hours figuring out why it didn't work.
  if(NOT EXISTS ${dependency_file})
    file(WRITE ${dependency_file} "#FindCUDA.cmake generated file.  Do not edit.\n")
  endif()

  # Always include this file to force CMake to run again next
  # invocation and rebuild the dependencies.
  #message("including dependency_file = ${dependency_file}")
  include(${dependency_file})

  # Now we need to verify the existence of all the included files
  # here.  If they aren't there we need to just blank this variable and
  # make the file regenerate again.
#   if(DEFINED CUDA_NVCC_DEPEND)
#     message("CUDA_NVCC_DEPEND set")
#   else()
#     message("CUDA_NVCC_DEPEND NOT set")
#   endif()
  if(CUDA_NVCC_DEPEND)
    #message("CUDA_NVCC_DEPEND found")
    foreach(f ${CUDA_NVCC_DEPEND})
      # message("searching for ${f}")
      if(NOT EXISTS ${f})
        #message("file ${f} not found")
        set(CUDA_NVCC_DEPEND_REGENERATE TRUE)
      endif()
    endforeach()
  else()
    #message("CUDA_NVCC_DEPEND false")
    # No dependencies, so regenerate the file.
    set(CUDA_NVCC_DEPEND_REGENERATE TRUE)
  endif()

  #message("CUDA_NVCC_DEPEND_REGENERATE = ${CUDA_NVCC_DEPEND_REGENERATE}")
  # No incoming dependencies, so we need to generate them.  Make the
  # output depend on the dependency file itself, which should cause the
  # rule to re-run.
  if(CUDA_NVCC_DEPEND_REGENERATE)
    set(CUDA_NVCC_DEPEND ${dependency_file})
    #message("Generating an empty dependency_file: ${dependency_file}")
    file(WRITE ${dependency_file} "#FindCUDA.cmake generated file.  Do not edit.\n")
  endif()

endmacro()

###############################################################################
###############################################################################
# Setup variables' defaults
###############################################################################
###############################################################################
# Allow the user to specify if the device code is supposed to be 32 or 64 bit.
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(CUDA_64_BIT_DEVICE_CODE_DEFAULT ON)
else()
  set(CUDA_64_BIT_DEVICE_CODE_DEFAULT OFF)
endif()
option(CUDA_64_BIT_DEVICE_CODE "Compile device code in 64 bit mode" ${CUDA_64_BIT_DEVICE_CODE_DEFAULT})

# Attach the build rule to the source file in VS.
option(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE "Attach the build rule to the CUDA source file.  Enable only when the CUDA source file is added to at most one target." ON)

# Prints out extra information about the cuda file during compilation
option(CUDA_BUILD_CUBIN "Generate and parse .cubin files in Device mode." OFF)

# Set whether we are using emulation or device mode
option(CUDA_BUILD_EMULATION "Build in Emulation mode" OFF)

# Where to put the generated output
set(CUDA_GENERATED_OUTPUT_DIR "" CACHE PATH "Directory to put all the output files.  If blank it will default to the CMAKE_CURRENT_BINARY_DIR")

# Parse HOST_COMPILATION mode
option(CUDA_HOST_COMPILATION_CPP "Generated file extension" ON)

# Extra user settable flags
set(CUDA_NVCC_FLAGS "" CACHE STRING "Semi-colon delimit multiple arguments.")

# Set HOST_COMPILER
if(CMAKE_GENERATOR MATCHES "Visual Studio")
  set(CUDA_HOST_COMPILER "$(VCInstallDir)bin" CACHE FILEPATH "Host side compiler used by NVCC")
else()
  # Determine C++ host compiler by inspecting CMAKE_CXX_COMPILER.
  # In case pynvccache is used as a host C++ compiler cache parse the host compiler from it's argument.
  if(NOT DEFINED CUDA_HOST_COMPILER)
    if(DEFINED CMAKE_CXX_COMPILER)
      if(CMAKE_CXX_COMPILER_ARG1 MATCHES "--nvcccache-compiler=(.+)")
        get_filename_component(c_compiler_realpath "${CMAKE_MATCH_1}" ABSOLUTE)
      elseif(CMAKE_CXX_COMPILER MATCHES "ccache")
        # there are 2 modes in which ccache can be invoked:
        # /usr/lib/ccache/<compiler> where compiler is a symlink to ccache who will determine which compiler to use
        # depending on its invokation.
        # /usr/bin/ccache <compiler> where ccache will redirect the call, we will differenciate between the 2
        # by checking the first argument for an empty string
        if ("${CMAKE_CXX_COMPILER_ARG1}" STREQUAL "")
            set(c_compiler_realpath "${CMAKE_CXX_COMPILER}")
        else()
            get_filename_component(c_compiler_realpath "${CMAKE_CXX_COMPILER_ARG1}" ABSOLUTE)
        endif()
      else()
        get_filename_component(c_compiler_realpath "${CMAKE_CXX_COMPILER}" ABSOLUTE)
      endif()
    else()
      set(c_compiler_realpath "")
    endif()
    set(CUDA_HOST_COMPILER "${c_compiler_realpath}" CACHE FILEPATH "Host side compiler used by NVCC")
  endif()
endif()

# Propagate the host flags to the host compiler via -Xcompiler
option(CUDA_PROPAGATE_HOST_FLAGS "Propagate C/CXX_FLAGS and friends to the host compiler via -Xcompile" ON)

# Use pynvccache cache
if(EXISTS ${CMAKE_CURRENT_LIST_DIR}/pynvcccache)
  option(CUDA_USE_PYNVCCCACHE "Use pynvccache compiler cache" OFF)
endif()

if(EXISTS ${CMAKE_CURRENT_LIST_DIR}/FindCUDA/get_cuda_sm.sh)
  option(CUDA_DETERMINE_HOST_GPU_CODE_FLAGS "Determine current real host GPU code nvcc flags" ON)
endif()

# Prevent some flags from being propagated
set(CUDA_NON_PROPAGATED_HOST_FLAGS "" CACHE STRING "Flags which will not be automatically propagated to the host compiler.")

# Enable CUDA_SEPARABLE_COMPILATION
option(CUDA_SEPARABLE_COMPILATION "Compile CUDA objects with separable compilation enabled.  Requires CUDA 5.0+" OFF)

# Specifies whether the commands used when compiling the .cu file will be printed out.
option(CUDA_VERBOSE_BUILD "Print out the commands run while compiling the CUDA source file.  With the Makefile generator this defaults to VERBOSE variable specified on the command line, but can be forced on with this option." OFF)

mark_as_advanced(
  CUDA_64_BIT_DEVICE_CODE
  CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE
  CUDA_GENERATED_OUTPUT_DIR
  CUDA_HOST_COMPILATION_CPP
  CUDA_NVCC_FLAGS
  CUDA_PROPAGATE_HOST_FLAGS
  CUDA_BUILD_CUBIN
  CUDA_BUILD_EMULATION
  CUDA_VERBOSE_BUILD
  CUDA_SEPARABLE_COMPILATION
  )

# Makefile and similar generators don't define CMAKE_CONFIGURATION_TYPES, so we
# need to add another entry for the CMAKE_BUILD_TYPE.  We also need to add the
# standerd set of 4 build types (Debug, MinSizeRel, Release, and RelWithDebInfo)
# for completeness.  We need run this loop in order to accommodate the addition
# of extra configuration types.  Duplicate entries will be removed by
# REMOVE_DUPLICATES.
set(CUDA_configuration_types ${CMAKE_CONFIGURATION_TYPES} ${CMAKE_BUILD_TYPE} Debug MinSizeRel Release RelWithDebInfo)
list(REMOVE_DUPLICATES CUDA_configuration_types)
foreach(config ${CUDA_configuration_types})
    string(TOUPPER ${config} config_upper)
    set(CUDA_NVCC_FLAGS_${config_upper} "" CACHE STRING "Semi-colon delimit multiple arguments.")
    mark_as_advanced(CUDA_NVCC_FLAGS_${config_upper})
endforeach()

###############################################################################
###############################################################################
# Locate CUDA, Set Build Type, etc.
###############################################################################
###############################################################################

macro(cuda_unset_include_and_libraries)
  unset(CUDA_TOOLKIT_INCLUDE CACHE)
  unset(CUDA_CUDART_LIBRARY CACHE)
  unset(CUDA_CUDA_LIBRARY CACHE)
  # Make sure you run this before you unset CUDA_VERSION.
  if(CUDA_VERSION VERSION_EQUAL "3.0")
    # This only existed in the 3.0 version of the CUDA toolkit
    unset(CUDA_CUDARTEMU_LIBRARY CACHE)
  endif()
  unset(CUDA_cupti_LIBRARY CACHE)
  unset(CUDA_cublas_LIBRARY CACHE)
  unset(CUDA_cublasemu_LIBRARY CACHE)
  unset(CUDA_cufft_LIBRARY CACHE)
  unset(CUDA_cufftemu_LIBRARY CACHE)
  unset(CUDA_curand_LIBRARY CACHE)
  unset(CUDA_cusolver_LIBRARY CACHE)
  unset(CUDA_cusparse_LIBRARY CACHE)
  unset(CUDA_npp_LIBRARY CACHE)
  unset(CUDA_nppc_LIBRARY CACHE)
  unset(CUDA_nppi_LIBRARY CACHE)
  unset(CUDA_npps_LIBRARY CACHE)
  unset(CUDA_nvcuvenc_LIBRARY CACHE)
  unset(CUDA_nvcuvid_LIBRARY CACHE)
  unset(CUDA_culibos_LIBRARY CACHE)
endmacro()

# Check to see if the CUDA_TOOLKIT_ROOT_DIR and CUDA_SDK_ROOT_DIR have changed,
# if they have then clear the cache variables, so that will be detected again.
if(NOT "${CUDA_TOOLKIT_ROOT_DIR}" STREQUAL "${CUDA_TOOLKIT_ROOT_DIR_INTERNAL}")
  unset(CUDA_TOOLKIT_TARGET_DIR CACHE)
  unset(CUDA_NVCC_EXECUTABLE CACHE)
  unset(CUDA_VERSION CACHE)
  cuda_unset_include_and_libraries()
endif()

if(NOT "${CUDA_TOOLKIT_TARGET_DIR}" STREQUAL "${CUDA_TOOLKIT_TARGET_DIR_INTERNAL}")
  cuda_unset_include_and_libraries()
endif()

if(NOT "${CUDA_SDK_ROOT_DIR}" STREQUAL "${CUDA_SDK_ROOT_DIR_INTERNAL}")
  # No specific variables to catch.  Use this kind of code before calling
  # find_package(CUDA) to clean up any variables that may depend on this path.

  #   unset(MY_SPECIAL_CUDA_SDK_INCLUDE_DIR CACHE)
  #   unset(MY_SPECIAL_CUDA_SDK_LIBRARY CACHE)
endif()

# Search for the cuda distribution.
if(NOT CUDA_TOOLKIT_ROOT_DIR)

  # Search in the CUDA_BIN_PATH first.
  find_path(CUDA_TOOLKIT_ROOT_DIR
    NAMES nvcc nvcc.exe
    PATHS
      ENV CUDA_BIN_PATH
      ENV CUDA_PATH
    PATH_SUFFIXES bin bin64
    DOC "Toolkit location."
    NO_DEFAULT_PATH
    )
  # Now search default paths
  find_path(CUDA_TOOLKIT_ROOT_DIR
    NAMES nvcc nvcc.exe
    PATHS /usr/local/bin
          /usr/local/cuda/bin
    DOC "Toolkit location."
    )

  if (CUDA_TOOLKIT_ROOT_DIR)
    string(REGEX REPLACE "[/\\\\]?bin[64]*[/\\\\]?$" "" CUDA_TOOLKIT_ROOT_DIR ${CUDA_TOOLKIT_ROOT_DIR})
    # We need to force this back into the cache.
    set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_TOOLKIT_ROOT_DIR} CACHE PATH "Toolkit location." FORCE)
  endif()
  if (NOT EXISTS ${CUDA_TOOLKIT_ROOT_DIR})
    if(CUDA_FIND_REQUIRED)
      message(FATAL_ERROR "Specify CUDA_TOOLKIT_ROOT_DIR")
    elseif(NOT CUDA_FIND_QUIETLY)
      message("CUDA_TOOLKIT_ROOT_DIR not found or specified")
    endif()
  endif ()
endif ()

# CUDA_NVCC_EXECUTABLE
find_program(CUDA_NVCC_EXECUTABLE
  NAMES nvcc
  PATHS "${CUDA_TOOLKIT_ROOT_DIR}"
  ENV CUDA_PATH
  ENV CUDA_BIN_PATH
  PATH_SUFFIXES bin bin64
  NO_DEFAULT_PATH
  )
# Search default search paths, after we search our own set of paths.
find_program(CUDA_NVCC_EXECUTABLE nvcc)
mark_as_advanced(CUDA_NVCC_EXECUTABLE)

if(CUDA_NVCC_EXECUTABLE AND NOT CUDA_VERSION)
  # Compute the version.
  execute_process (COMMAND ${CUDA_NVCC_EXECUTABLE} "--version" OUTPUT_VARIABLE NVCC_OUT)
  string(REGEX REPLACE ".*release ([0-9]+)\\.([0-9]+).*" "\\1" CUDA_VERSION_MAJOR ${NVCC_OUT})
  string(REGEX REPLACE ".*release ([0-9]+)\\.([0-9]+).*" "\\2" CUDA_VERSION_MINOR ${NVCC_OUT})
  set(CUDA_VERSION "${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}" CACHE STRING "Version of CUDA as computed from nvcc.")
  mark_as_advanced(CUDA_VERSION)
else()
  # Need to set these based off of the cached value
  string(REGEX REPLACE "([0-9]+)\\.([0-9]+).*" "\\1" CUDA_VERSION_MAJOR "${CUDA_VERSION}")
  string(REGEX REPLACE "([0-9]+)\\.([0-9]+).*" "\\2" CUDA_VERSION_MINOR "${CUDA_VERSION}")
endif()

if(CUDA_USE_PYNVCCCACHE)
  set(CUDA_PYNVCCCACHE_SCRIPT ${CMAKE_CURRENT_LIST_DIR}/pynvcccache/nvcccache.py)

  find_package(PythonInterp 3)

  if(NOT PYTHONINTERP_FOUND)
    message(WARNING "Unable to find python interpreter."
                    "Deactivating pynvcccache-based caching")
    set(CUDA_USE_PYNVCCCACHE OFF)
  else()
    # Check if nvcccache dependencies are installed
    execute_process(COMMAND
                    ${PYTHON_EXECUTABLE} ${CUDA_PYNVCCCACHE_SCRIPT} --help
                    RESULT_VARIABLE NVCCCACHE_RES
                    OUTPUT_VARIABLE NVCCCACHE_OUT
                    ERROR_VARIABLE  NVCCCACHE_ERR
                    )

    if(${NVCCCACHE_RES})
      message(WARNING "Unable to run pynvcccache with ${PYTHON_EXECUTABLE}:\n${NVCCCACHE_OUT}${NVCCCACHE_ERR}"
              "Hint: check correct python version, try installing missing dependencies with\npip3 install --user <module-name>\n"
              "Deactivating pynvcccache-based caching")
      set(CUDA_USE_PYNVCCCACHE OFF)
    endif()
  endif()
endif()

# Always set this convenience variable
set(CUDA_VERSION_STRING "${CUDA_VERSION}")

# Support for arm cross compilation with CUDA 5.5
if(CUDA_VERSION VERSION_GREATER "5.0" AND CMAKE_CROSSCOMPILING)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm" AND EXISTS "${CUDA_TOOLKIT_ROOT_DIR}/targets/armv7-linux-gnueabihf")
    set(CUDA_TOOLKIT_TARGET_DIR "${CUDA_TOOLKIT_ROOT_DIR}/targets/armv7-linux-gnueabihf" CACHE PATH "Toolkit target location.")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" AND VIBRANTE_V5Q AND EXISTS "${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-qnx")
    set(CUDA_TOOLKIT_TARGET_DIR "${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-qnx" CACHE PATH "Toolkit target location.")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" AND VIBRANTE_V5L AND EXISTS "${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux")
    set(CUDA_TOOLKIT_TARGET_DIR "${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux" CACHE PATH "Toolkit target location.")
  endif()
else()
  set(CUDA_TOOLKIT_TARGET_DIR "${CUDA_TOOLKIT_ROOT_DIR}" CACHE PATH "Toolkit target location.")
endif()
mark_as_advanced(CUDA_TOOLKIT_TARGET_DIR)

# Target CPU architecture
if(CUDA_VERSION VERSION_GREATER "5.0" AND CMAKE_CROSSCOMPILING)
  if (CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
    set(_cuda_target_cpu_arch_initial "ARM")
  elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(_cuda_target_cpu_arch_initial "AARCH64")
  endif()
else()
  set(_cuda_target_cpu_arch_initial "")
endif()
set(CUDA_TARGET_CPU_ARCH ${_cuda_target_cpu_arch_initial} CACHE STRING "Specify the name of the class of CPU architecture for which the input files must be compiled.")
mark_as_advanced(CUDA_TARGET_CPU_ARCH)

# CUDA_TOOLKIT_INCLUDE
find_path(CUDA_TOOLKIT_INCLUDE
  device_functions.h # Header included in toolkit
  PATHS "${CUDA_TOOLKIT_TARGET_DIR}" # target folder is implictly root folder if not cross-compiling
  ENV CUDA_PATH
  ENV CUDA_INC_PATH
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
  )
# Search default search paths, after we search our own set of paths.
find_path(CUDA_TOOLKIT_INCLUDE device_functions.h)
mark_as_advanced(CUDA_TOOLKIT_INCLUDE)

# Set the user list of include dir to nothing to initialize it.
set (CUDA_NVCC_INCLUDE_ARGS_USER "")
set (CUDA_NVCC_SYSTEM_INCLUDE_ARGS_USER "")
set (CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_INCLUDE})

# Find nvidia driver version
# Determine currently installed linux driver major version hint
set(nvidia-driver-version_FOUND FALSE)
set(nvidia-driver-version "current")
if(NOT WIN32)
  if(EXISTS /proc/driver/nvidia/version)
    file(READ /proc/driver/nvidia/version nvidia-driver-version-file)
    if(${nvidia-driver-version-file} MATCHES "NVIDIA UNIX.*Kernel Module  ([0123456789]+)\\.[0123456789]+")
      set(nvidia-driver-version_FOUND TRUE)
      set(nvidia-driver-version ${CMAKE_MATCH_1})
    endif()
  endif()
endif()

macro(cuda_find_library_local_first_with_path_ext _var _names _doc _path_ext )
  if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    # CUDA 3.2+ on Windows moved the library directories, so we need the new
    # and old paths.
    set(_cuda_lib_dir "${_path_ext}lib/x64" "${_path_ext}lib64" "${_path_ext}libx64" )
  endif()
  if(CMAKE_CROSSCOMPILING AND (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "arm"))
    set(_cuda_lib_dir "${_path_ext}lib/stubs")
  endif()
  # CUDA 3.2+ on Windows moved the library directories, so we need to new
  # (lib/Win32) and the old path (lib).
  find_library(${_var}
    NAMES ${_names}
    PATHS "${CUDA_TOOLKIT_TARGET_DIR}" # target folder is implictly root folder if not cross-compiling
    ENV CUDA_PATH
    ENV CUDA_LIB_PATH
    PATH_SUFFIXES ${_cuda_lib_dir} "${_path_ext}lib/Win32" "${_path_ext}lib" "${_path_ext}libWin32"
    DOC ${_doc}
    NO_DEFAULT_PATH
    )
  # Search default search paths, after we search our own set of paths.
  find_library(${_var}
    NAMES ${_names}
    PATHS "/usr/lib/nvidia-current"
    DOC ${_doc}
    HINTS ${VIBRANTE_PDK}/lib-target
          "/usr/lib/nvidia-${nvidia-driver-version}"
          "/usr/lib/nvidia-396"
          "/usr/lib/nvidia-390"
          "/usr/lib/nvidia-367"
          "/usr/lib/nvidia-364"
          "/usr/lib/nvidia-361"
          "/usr/lib/nvidia-358"
          "/usr/lib/nvidia-355"
          "/usr/lib/nvidia-352"
          "/usr/lib/nvidia-346"
          "/usr/lib/nvidia-343"
          "/usr/lib/nvidia-340"
          "/usr/lib/nvidia-331"
          "/usr/lib/nvidia-319"
          "/usr/lib/nvidia-310"
          "/usr/lib/nvidia-304"
          "/usr/lib/nvidia-173"
    )
endmacro()

macro(cuda_find_library_local_first _var _names _doc)
  cuda_find_library_local_first_with_path_ext( "${_var}" "${_names}" "${_doc}" "" )
endmacro()

macro(find_library_local_first _var _names _doc )
  cuda_find_library_local_first( "${_var}" "${_names}" "${_doc}" "" )
endmacro()


# CUDA_LIBRARIES
cuda_find_library_local_first(CUDA_CUDART_LIBRARY "cudart" "\"cudart\" library")
if(CUDA_VERSION VERSION_EQUAL "3.0")
  # The cudartemu library only existed for the 3.0 version of CUDA.
  cuda_find_library_local_first(CUDA_CUDARTEMU_LIBRARY cudartemu "\"cudartemu\" library")
  mark_as_advanced(
    CUDA_CUDARTEMU_LIBRARY
    )
endif()

# CUPTI library showed up in cuda toolkit 4.0
if(NOT CUDA_VERSION VERSION_LESS "4.0")
  cuda_find_library_local_first_with_path_ext(CUDA_cupti_LIBRARY "cupti" "\"cupti\" library" "extras/CUPTI/")
  mark_as_advanced(CUDA_cupti_LIBRARY)
endif()

# If we are using emulation mode and we found the cudartemu library then use
# that one instead of cudart.
if(CUDA_BUILD_EMULATION AND CUDA_CUDARTEMU_LIBRARY)
  set(CUDA_LIBRARIES ${CUDA_CUDARTEMU_LIBRARY})
else()
  set(CUDA_LIBRARIES ${CUDA_CUDART_LIBRARY})
endif()

# 1.1 toolkit on linux doesn't appear to have a separate library on
# some platforms.
cuda_find_library_local_first(CUDA_CUDA_LIBRARY cuda "\"cuda\" library (older versions only).")

mark_as_advanced(
  CUDA_CUDA_LIBRARY
  CUDA_CUDART_LIBRARY
  )

#######################
# Look for some of the toolkit helper libraries
macro(FIND_CUDA_HELPER_LIBS _name)
  cuda_find_library_local_first(CUDA_${_name}_LIBRARY "${_name}" "\"${_name}\" library")
  mark_as_advanced(CUDA_${_name}_LIBRARY)
endmacro()

#######################
# Disable emulation for v3.1 onward
if(CUDA_VERSION VERSION_GREATER "3.0")
  if(CUDA_BUILD_EMULATION)
    message(FATAL_ERROR "CUDA_BUILD_EMULATION is not supported in version 3.1 and onwards.  You must disable it to proceed.  You have version ${CUDA_VERSION}.")
  endif()
endif()

# Search for additional CUDA toolkit libraries.
if(CUDA_VERSION VERSION_LESS "3.1")
  # Emulation libraries aren't available in version 3.1 onward.
  find_cuda_helper_libs(cufftemu)
  find_cuda_helper_libs(cublasemu)
endif()
find_cuda_helper_libs(cufft)
find_cuda_helper_libs(cublas)
if(NOT CUDA_VERSION VERSION_LESS "3.2")
  # cusparse showed up in version 3.2
  find_cuda_helper_libs(cusparse)
  find_cuda_helper_libs(curand)
  find_cuda_helper_libs(nvcuvenc)
  find_cuda_helper_libs(nvcuvid)
endif()
if(CUDA_VERSION VERSION_GREATER "5.0")
  # In CUDA 5.5 NPP was splitted onto 3 separate libraries.
  find_cuda_helper_libs(nppc)
  find_cuda_helper_libs(nppi)
  find_cuda_helper_libs(npps)
  set(CUDA_npp_LIBRARY "${CUDA_nppc_LIBRARY};${CUDA_nppi_LIBRARY};${CUDA_npps_LIBRARY}")
elseif(NOT CUDA_VERSION VERSION_LESS "4.0")
  find_cuda_helper_libs(npp)
endif()
if(NOT CUDA_VERSION VERSION_LESS "6.5")
  # culibos showed up in version 6.5
  find_cuda_helper_libs(culibos)
endif()
if(NOT CUDA_VERSION VERSION_LESS "7.0")
  # cusolver showed up in version 7.0
  find_cuda_helper_libs(cusolver)
endif()

if (CUDA_BUILD_EMULATION)
  set(CUDA_CUFFT_LIBRARIES ${CUDA_cufftemu_LIBRARY})
  set(CUDA_CUBLAS_LIBRARIES ${CUDA_cublasemu_LIBRARY})
else()
  set(CUDA_CUFFT_LIBRARIES ${CUDA_cufft_LIBRARY})
  set(CUDA_CUBLAS_LIBRARIES ${CUDA_cublas_LIBRARY})
endif()

########################
# Look for the SDK stuff.  As of CUDA 3.0 NVSDKCUDA_ROOT has been replaced with
# NVSDKCOMPUTE_ROOT with the old CUDA C contents moved into the C subdirectory
find_path(CUDA_SDK_ROOT_DIR common/inc/cutil.h
 HINTS
  "$ENV{NVSDKCOMPUTE_ROOT}/C"
  ENV NVSDKCUDA_ROOT
  "[HKEY_LOCAL_MACHINE\\SOFTWARE\\NVIDIA Corporation\\Installed Products\\NVIDIA SDK 10\\Compute;InstallDir]"
 PATHS
  "/Developer/GPU\ Computing/C"
  )

# Keep the CUDA_SDK_ROOT_DIR first in order to be able to override the
# environment variables.
set(CUDA_SDK_SEARCH_PATH
  "${CUDA_SDK_ROOT_DIR}"
  "${CUDA_TOOLKIT_ROOT_DIR}/local/NVSDK0.2"
  "${CUDA_TOOLKIT_ROOT_DIR}/NVSDK0.2"
  "${CUDA_TOOLKIT_ROOT_DIR}/NV_CUDA_SDK"
  "$ENV{HOME}/NVIDIA_CUDA_SDK"
  "$ENV{HOME}/NVIDIA_CUDA_SDK_MACOSX"
  "/Developer/CUDA"
  )

# Example of how to find an include file from the CUDA_SDK_ROOT_DIR

# find_path(CUDA_CUT_INCLUDE_DIR
#   cutil.h
#   PATHS ${CUDA_SDK_SEARCH_PATH}
#   PATH_SUFFIXES "common/inc"
#   DOC "Location of cutil.h"
#   NO_DEFAULT_PATH
#   )
# # Now search system paths
# find_path(CUDA_CUT_INCLUDE_DIR cutil.h DOC "Location of cutil.h")

# mark_as_advanced(CUDA_CUT_INCLUDE_DIR)


# Example of how to find a library in the CUDA_SDK_ROOT_DIR

# # cutil library is called cutil64 for 64 bit builds on windows.  We don't want
# # to get these confused, so we are setting the name based on the word size of
# # the build.

# if(CMAKE_SIZEOF_VOID_P EQUAL 8)
#   set(cuda_cutil_name cutil64)
# else()
#   set(cuda_cutil_name cutil32)
# endif()

# find_library(CUDA_CUT_LIBRARY
#   NAMES cutil ${cuda_cutil_name}
#   PATHS ${CUDA_SDK_SEARCH_PATH}
#   # The new version of the sdk shows up in common/lib, but the old one is in lib
#   PATH_SUFFIXES "common/lib" "lib"
#   DOC "Location of cutil library"
#   NO_DEFAULT_PATH
#   )
# # Now search system paths
# find_library(CUDA_CUT_LIBRARY NAMES cutil ${cuda_cutil_name} DOC "Location of cutil library")
# mark_as_advanced(CUDA_CUT_LIBRARY)
# set(CUDA_CUT_LIBRARIES ${CUDA_CUT_LIBRARY})



#############################
# Check for required components
set(CUDA_FOUND TRUE)

set(CUDA_TOOLKIT_ROOT_DIR_INTERNAL "${CUDA_TOOLKIT_ROOT_DIR}" CACHE INTERNAL
  "This is the value of the last time CUDA_TOOLKIT_ROOT_DIR was set successfully." FORCE)
set(CUDA_TOOLKIT_TARGET_DIR_INTERNAL "${CUDA_TOOLKIT_TARGET_DIR}" CACHE INTERNAL
  "This is the value of the last time CUDA_TOOLKIT_TARGET_DIR was set successfully." FORCE)
set(CUDA_SDK_ROOT_DIR_INTERNAL "${CUDA_SDK_ROOT_DIR}" CACHE INTERNAL
  "This is the value of the last time CUDA_SDK_ROOT_DIR was set successfully." FORCE)

include(${CMAKE_CURRENT_LIST_DIR}/FindPackageHandleStandardArgs.cmake)
find_package_handle_standard_args(CUDA
  REQUIRED_VARS
    CUDA_TOOLKIT_ROOT_DIR
    CUDA_NVCC_EXECUTABLE
    CUDA_INCLUDE_DIRS
    CUDA_CUDART_LIBRARY
  VERSION_VAR
    CUDA_VERSION
  )



###############################################################################
###############################################################################
# Macros
###############################################################################
###############################################################################

###############################################################################
# Add include directories to pass to the nvcc command.
macro(CUDA_INCLUDE_DIRECTORIES)
  foreach(dir ${ARGN})
    list(APPEND CUDA_NVCC_INCLUDE_ARGS_USER -I${dir})
  endforeach()
endmacro()

macro(CUDA_SYSTEM_INCLUDE_DIRECTORIES)
  foreach(dir ${ARGN})
    list(APPEND CUDA_NVCC_SYSTEM_INCLUDE_ARGS_USER "${dir}")
  endforeach()
endmacro()


##############################################################################
cuda_find_helper_file(parse_cubin cmake)
cuda_find_helper_file(make2cmake cmake)
cuda_find_helper_file(run_nvcc cmake)
cuda_find_helper_file(run_nvcc_deps cmake)

##############################################################################
# Separate the OPTIONS out from the sources
#
macro(CUDA_GET_SOURCES_AND_OPTIONS _sources _cmake_options _options)
  set( ${_sources} )
  set( ${_cmake_options} )
  set( ${_options} )
  set( _found_options FALSE )
  foreach(arg ${ARGN})
    if("x${arg}" STREQUAL "xOPTIONS")
      set( _found_options TRUE )
    elseif(
        "x${arg}" STREQUAL "xWIN32" OR
        "x${arg}" STREQUAL "xMACOSX_BUNDLE" OR
        "x${arg}" STREQUAL "xEXCLUDE_FROM_ALL" OR
        "x${arg}" STREQUAL "xSTATIC" OR
        "x${arg}" STREQUAL "xSHARED" OR
        "x${arg}" STREQUAL "xMODULE"
        )
      list(APPEND ${_cmake_options} ${arg})
    else()
      if ( _found_options )
        list(APPEND ${_options} ${arg})
      else()
        # Assume this is a file
        list(APPEND ${_sources} ${arg})
      endif()
    endif()
  endforeach()
endmacro()

##############################################################################
# Parse the OPTIONS from ARGN and set the variables prefixed by _option_prefix
#
macro(CUDA_PARSE_NVCC_OPTIONS _option_prefix)
  set( _found_config )
  foreach(arg ${ARGN})
    # Determine if we are dealing with a perconfiguration flag
    foreach(config ${CUDA_configuration_types})
      string(TOUPPER ${config} config_upper)
      if (arg STREQUAL "${config_upper}")
        set( _found_config _${arg})
        # Set arg to nothing to keep it from being processed further
        set( arg )
      endif()
    endforeach()

    if ( arg )
      list(APPEND ${_option_prefix}${_found_config} "${arg}")
    endif()
  endforeach()
endmacro()

##############################################################################
# Helper to add the include directory for CUDA only once
function(CUDA_ADD_CUDA_INCLUDE_ONCE)
  get_directory_property(_include_directories INCLUDE_DIRECTORIES)
  set(_add TRUE)
  if(_include_directories)
    foreach(dir ${_include_directories})
      if("${dir}" STREQUAL "${CUDA_INCLUDE_DIRS}")
        set(_add FALSE)
      endif()
    endforeach()
  endif()
  if(_add)
    include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
  endif()
endfunction()

function(CUDA_BUILD_SHARED_LIBRARY shared_flag)
  set(cmake_args ${ARGN})
  # If SHARED, MODULE, or STATIC aren't already in the list of arguments, then
  # add SHARED or STATIC based on the value of BUILD_SHARED_LIBS.
  list(FIND cmake_args SHARED _cuda_found_SHARED)
  list(FIND cmake_args MODULE _cuda_found_MODULE)
  list(FIND cmake_args STATIC _cuda_found_STATIC)
  if( _cuda_found_SHARED GREATER -1 OR
      _cuda_found_MODULE GREATER -1 OR
      _cuda_found_STATIC GREATER -1)
    set(_cuda_build_shared_libs)
  else()
    if (BUILD_SHARED_LIBS)
      set(_cuda_build_shared_libs SHARED)
    else()
      set(_cuda_build_shared_libs STATIC)
    endif()
  endif()
  set(${shared_flag} ${_cuda_build_shared_libs} PARENT_SCOPE)
endfunction()

##############################################################################
# Helper to avoid clashes of files with the same basename but different paths.
# This doesn't attempt to do exactly what CMake internals do, which is to only
# add this path when there is a conflict, since by the time a second collision
# in names is detected it's already too late to fix the first one.  For
# consistency sake the relative path will be added to all files.
function(CUDA_COMPUTE_BUILD_PATH path build_path)
  #message("CUDA_COMPUTE_BUILD_PATH([${path}] ${build_path})")
  # Only deal with CMake style paths from here on out
  file(TO_CMAKE_PATH "${path}" bpath)
  if (IS_ABSOLUTE "${bpath}")
    # Absolute paths are generally unnessary, especially if something like
    # file(GLOB_RECURSE) is used to pick up the files.

    string(FIND "${bpath}" "${CMAKE_CURRENT_BINARY_DIR}" _binary_dir_pos)
    if (_binary_dir_pos EQUAL 0)
      file(RELATIVE_PATH bpath "${CMAKE_CURRENT_BINARY_DIR}" "${bpath}")
    else()
      file(RELATIVE_PATH bpath "${CMAKE_CURRENT_SOURCE_DIR}" "${bpath}")
    endif()
  endif()

  # This recipe is from cmLocalGenerator::CreateSafeUniqueObjectFileName in the
  # CMake source.

  # Remove leading /
  string(REGEX REPLACE "^[/]+" "" bpath "${bpath}")
  # Avoid absolute paths by removing ':'
  string(REPLACE ":" "_" bpath "${bpath}")
  # Avoid relative paths that go up the tree
  string(REPLACE "../" "__/" bpath "${bpath}")
  # Avoid spaces
  string(REPLACE " " "_" bpath "${bpath}")

  # Strip off the filename.  I wait until here to do it, since removin the
  # basename can make a path that looked like path/../basename turn into
  # path/.. (notice the trailing slash).
  get_filename_component(bpath "${bpath}" PATH)

  set(${build_path} "${bpath}" PARENT_SCOPE)
  #message("${build_path} = ${bpath}")
endfunction()

##############################################################################
# This helper macro populates the following variables and setups up custom
# commands and targets to invoke the nvcc compiler to generate C or PTX source
# dependent upon the format parameter.  The compiler is invoked once with -M
# to generate a dependency file and a second time with -cuda or -ptx to generate
# a .cpp or .ptx file.
# INPUT:
#   cuda_target         - Target name
#   format              - PTX, CUBIN, FATBIN or OBJ
#   FILE1 .. FILEN      - The remaining arguments are the sources to be wrapped.
#   OPTIONS             - Extra options to NVCC
# OUTPUT:
#   generated_files     - List of generated files
##############################################################################
##############################################################################

macro(CUDA_WRAP_SRCS cuda_target format generated_files)

  # If CMake doesn't support separable compilation, complain
  if(CUDA_SEPARABLE_COMPILATION AND CMAKE_VERSION VERSION_LESS "2.8.10.1")
    message(SEND_ERROR "CUDA_SEPARABLE_COMPILATION isn't supported for CMake versions less than 2.8.10.1")
  endif()

  # Set up all the command line flags here, so that they can be overridden on a per target basis.

  set(nvcc_flags "")

  # Emulation if the card isn't present.
  if (CUDA_BUILD_EMULATION)
    # Emulation.
    set(nvcc_flags ${nvcc_flags} --device-emulation -D_DEVICEEMU -g)
  else()
    # Device mode.  No flags necessary.
  endif()

  if(CUDA_HOST_COMPILATION_CPP)
    set(CUDA_C_OR_CXX CXX)
  else()
    if(CUDA_VERSION VERSION_LESS "3.0")
      set(nvcc_flags ${nvcc_flags} --host-compilation C)
    else()
      message(WARNING "--host-compilation flag is deprecated in CUDA version >= 3.0.  Removing --host-compilation C flag" )
    endif()
    set(CUDA_C_OR_CXX C)
  endif()

  set(generated_extension ${CMAKE_${CUDA_C_OR_CXX}_OUTPUT_EXTENSION})

  if(CUDA_64_BIT_DEVICE_CODE)
    set(nvcc_flags ${nvcc_flags} -m64)
  else()
    set(nvcc_flags ${nvcc_flags} -m32)
  endif()

  #if(CUDA_TARGET_CPU_ARCH)
  #  set(nvcc_flags ${nvcc_flags} "--target-cpu-architecture=${CUDA_TARGET_CPU_ARCH}")
  #endif()

  # This needs to be passed in at this stage, because VS needs to fill out the
  # value of VCInstallDir from within VS.  Note that CCBIN is only used if
  # -ccbin or --compiler-bindir isn't used and CUDA_HOST_COMPILER matches
  # $(VCInstallDir)/bin.
  if(CMAKE_GENERATOR MATCHES "Visual Studio")
    set(ccbin_flags -D "\"CCBIN:PATH=$(VCInstallDir)bin\"" )
  else()
    set(ccbin_flags)
  endif()

  # Figure out which configure we will use and pass that in as an argument to
  # the script.  We need to defer the decision until compilation time, because
  # for VS projects we won't know if we are making a debug or release build
  # until build time.
  if(CMAKE_GENERATOR MATCHES "Visual Studio")
    set( CUDA_build_configuration "$(ConfigurationName)" )
  else()
    set( CUDA_build_configuration "${CMAKE_BUILD_TYPE}")
  endif()

  # Initialize our list of includes with the user ones followed by the CUDA system ones.
  set(CUDA_NVCC_INCLUDE_ARGS ${CUDA_NVCC_INCLUDE_ARGS_USER} "-I${CUDA_INCLUDE_DIRS}")
  # Get the include directories for this directory and use them for our nvcc command.
  # Remove duplicate entries which may be present since include_directories
  # in CMake >= 2.8.8 does not remove them.
  get_directory_property(CUDA_NVCC_INCLUDE_DIRECTORIES INCLUDE_DIRECTORIES)
  list(REMOVE_DUPLICATES CUDA_NVCC_INCLUDE_DIRECTORIES)
  if(CUDA_NVCC_INCLUDE_DIRECTORIES)
    foreach(dir ${CUDA_NVCC_INCLUDE_DIRECTORIES})
      list(APPEND CUDA_NVCC_INCLUDE_ARGS -I${dir})
    endforeach()
  endif()

  foreach(dir ${CUDA_NVCC_SYSTEM_INCLUDE_ARGS_USER})
    list(APPEND CUDA_NVCC_INCLUDE_ARGS -isystem ${dir})
  endforeach()

  # Reset these variables
  set(CUDA_WRAP_OPTION_NVCC_FLAGS)
  foreach(config ${CUDA_configuration_types})
    string(TOUPPER ${config} config_upper)
    set(CUDA_WRAP_OPTION_NVCC_FLAGS_${config_upper})
  endforeach()

  CUDA_GET_SOURCES_AND_OPTIONS(_cuda_wrap_sources _cuda_wrap_cmake_options _cuda_wrap_options ${ARGN})
  CUDA_PARSE_NVCC_OPTIONS(CUDA_WRAP_OPTION_NVCC_FLAGS ${_cuda_wrap_options})

  # Figure out if we are building a shared library.  BUILD_SHARED_LIBS is
  # respected in CUDA_ADD_LIBRARY.
  set(_cuda_build_shared_libs FALSE)
  # SHARED, MODULE
  list(FIND _cuda_wrap_cmake_options SHARED _cuda_found_SHARED)
  list(FIND _cuda_wrap_cmake_options MODULE _cuda_found_MODULE)
  if(_cuda_found_SHARED GREATER -1 OR _cuda_found_MODULE GREATER -1)
    set(_cuda_build_shared_libs TRUE)
  endif()
  # STATIC
  list(FIND _cuda_wrap_cmake_options STATIC _cuda_found_STATIC)
  if(_cuda_found_STATIC GREATER -1)
    set(_cuda_build_shared_libs FALSE)
  endif()

  # CUDA_HOST_FLAGS
  if(_cuda_build_shared_libs)
    # If we are setting up code for a shared library, then we need to add extra flags for
    # compiling objects for shared libraries.
    set(CUDA_HOST_SHARED_FLAGS ${CMAKE_SHARED_LIBRARY_${CUDA_C_OR_CXX}_FLAGS})
  else()
    set(CUDA_HOST_SHARED_FLAGS)
  endif()
  # Only add the CMAKE_{C,CXX}_FLAGS if we are propagating host flags.  We
  # always need to set the SHARED_FLAGS, though.
  if(CUDA_PROPAGATE_HOST_FLAGS)
    set(_cuda_host_flags "set(CMAKE_HOST_FLAGS ${CMAKE_${CUDA_C_OR_CXX}_FLAGS} ${CUDA_HOST_SHARED_FLAGS})")
  else()
    set(_cuda_host_flags "set(CMAKE_HOST_FLAGS ${CUDA_HOST_SHARED_FLAGS})")
  endif()

  set(_cuda_nvcc_flags_config "# Build specific configuration flags")
  # Loop over all the configuration types to generate appropriate flags for run_nvcc.cmake
  foreach(config ${CUDA_configuration_types})
    string(TOUPPER ${config} config_upper)
    # CMAKE_FLAGS are strings and not lists.  By not putting quotes around CMAKE_FLAGS
    # we convert the strings to lists (like we want).

    if(CUDA_PROPAGATE_HOST_FLAGS)
      # nvcc chokes on -g3 in versions previous to 3.0, so replace it with -g
      set(_cuda_fix_g3 FALSE)

      if(CMAKE_COMPILER_IS_GNUCC)
        if (CUDA_VERSION VERSION_LESS  "3.0" OR
            CUDA_VERSION VERSION_EQUAL "4.1" OR
            CUDA_VERSION VERSION_EQUAL "4.2"
            )
          set(_cuda_fix_g3 TRUE)
        endif()
      endif()
      if(_cuda_fix_g3)
        string(REPLACE "-g3" "-g" _cuda_C_FLAGS "${CMAKE_${CUDA_C_OR_CXX}_FLAGS_${config_upper}}")
      else()
        set(_cuda_C_FLAGS "${CMAKE_${CUDA_C_OR_CXX}_FLAGS_${config_upper}}")
      endif()

      set(_cuda_host_flags "${_cuda_host_flags}\nset(CMAKE_HOST_FLAGS_${config_upper} ${_cuda_C_FLAGS})")

      # Some flags are automatically propagated by nvcc. For example, one shouldn't specify -std=c++11
      # as a host flag, since nvcc will try to pass it as an argument to the C host compiler during compilation,
      # causing unnecessary warnings.
      #
      # This code will prevent the propagation of any flags in the CUDA_NON_PROPAGATED_HOST_FLAGS.
      # Flags assigned to _cuda_non_propagated_host_flags initially are ones known to cause issues in all cases.
      set(_cuda_non_propagated_host_flags -std=c++11)
      if(CUDA_NON_PROPAGATED_HOST_FLAGS)
        list(APPEND _cuda_non_propagated_host_flags ${CUDA_NON_PROPAGATED_HOST_FLAGS})
      endif()

      if(VIBRANTE_V5Q)
        list(APPEND _cuda_non_propagated_host_flags "-nostdinc++")
        list(APPEND _cuda_non_propagated_host_flags "-nostdinc")
        list(APPEND _cuda_non_propagated_host_flags "-isystem ${QNX_TARGET}/usr/include/c++/v1")
      endif()

      foreach(flag ${_cuda_non_propagated_host_flags})
        string(REPLACE "${flag}" "" _cuda_host_flags "${_cuda_host_flags}")
      endforeach()

    endif()

    # Note that if we ever want CUDA_NVCC_FLAGS_<CONFIG> to be string (instead of a list
    # like it is currently), we can remove the quotes around the
    # ${CUDA_NVCC_FLAGS_${config_upper}} variable like the CMAKE_HOST_FLAGS_<CONFIG> variable.
    set(_cuda_nvcc_flags_config "${_cuda_nvcc_flags_config}\nset(CUDA_NVCC_FLAGS_${config_upper} ${CUDA_NVCC_FLAGS_${config_upper}} ;; ${CUDA_WRAP_OPTION_NVCC_FLAGS_${config_upper}})")
  endforeach()

  # Get the list of definitions from the directory property
  get_directory_property(CUDA_NVCC_DEFINITIONS COMPILE_DEFINITIONS)
  if(CUDA_NVCC_DEFINITIONS)
    foreach(_definition ${CUDA_NVCC_DEFINITIONS})
      list(APPEND nvcc_flags "-D${_definition}")
    endforeach()
  endif()

  if(_cuda_build_shared_libs)
    list(APPEND nvcc_flags "-D${cuda_target}_EXPORTS")
  endif()

  # Reset the output variable
  set(_cuda_wrap_generated_files "")

  # Iterate over the macro arguments and create custom
  # commands for all the .cu files.
  foreach(file ${ARGN})
    # Ignore any file marked as a HEADER_FILE_ONLY
    get_source_file_property(_is_header ${file} HEADER_FILE_ONLY)
    if(${file} MATCHES "\\.cu$" AND NOT _is_header)

      # Allow per source file overrides of the format.
      get_source_file_property(_cuda_source_format ${file} CUDA_SOURCE_PROPERTY_FORMAT)
      if(NOT _cuda_source_format)
        set(_cuda_source_format ${format})
      endif()

      if( ${_cuda_source_format} MATCHES "OBJ")
        set( cuda_compile_to_external_module OFF )
      else()
        set( cuda_compile_to_external_module ON )
        if( ${_cuda_source_format} MATCHES "PTX" )
          set( cuda_compile_to_external_module_type "ptx" )
        elseif( ${_cuda_source_format} MATCHES "CUBIN")
          set( cuda_compile_to_external_module_type "cubin" )
        elseif( ${_cuda_source_format} MATCHES "FATBIN")
          set( cuda_compile_to_external_module_type "fatbin" )
        else()
          message( FATAL_ERROR "Invalid format flag passed to CUDA_WRAP_SRCS for file '${file}': '${_cuda_source_format}'.  Use OBJ, PTX, CUBIN or FATBIN.")
        endif()
      endif()

      if(cuda_compile_to_external_module)
        # Don't use any of the host compilation flags for PTX targets.
        set(CUDA_HOST_FLAGS)
        set(CUDA_NVCC_FLAGS_CONFIG)
      else()
        set(CUDA_HOST_FLAGS ${_cuda_host_flags})
        set(CUDA_NVCC_FLAGS_CONFIG ${_cuda_nvcc_flags_config})
      endif()

      # Determine output directory
      cuda_compute_build_path("${file}" cuda_build_path)
      set(cuda_compile_intermediate_directory "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${cuda_target}.dir/${cuda_build_path}")
      if(CUDA_GENERATED_OUTPUT_DIR)
        set(cuda_compile_output_dir "${CUDA_GENERATED_OUTPUT_DIR}")
      else()
        if ( cuda_compile_to_external_module )
          set(cuda_compile_output_dir "${CMAKE_CURRENT_BINARY_DIR}")
        else()
          set(cuda_compile_output_dir "${cuda_compile_intermediate_directory}")
        endif()
      endif()

      # Add a custom target to generate a c or ptx file. ######################

      get_filename_component( basename ${file} NAME )
      if( cuda_compile_to_external_module )
        set(generated_file_basename "gen_${basename}.${cuda_compile_to_external_module_type}")
        set(format_flag "-${cuda_compile_to_external_module_type}")
        file(MAKE_DIRECTORY "${cuda_compile_output_dir}")
      else()
        set(generated_file_path "${cuda_compile_output_dir}/${CMAKE_CFG_INTDIR}")
        set(generated_file_basename "gen_${basename}${generated_extension}")
        if(CUDA_SEPARABLE_COMPILATION)
          set(format_flag "-dc")
        else()
          set(format_flag "-c")
        endif()
      endif()

      # Set all of our file names.  Make sure that whatever filenames that have
      # generated_file_path in them get passed in through as a command line
      # argument, so that the ${CMAKE_CFG_INTDIR} gets expanded at run time
      # instead of configure time.
      set(generated_file "${generated_file_path}/${generated_file_basename}")
      set(cmake_dependency_file "${cuda_compile_intermediate_directory}/${generated_file_basename}.depend")
      set(NVCC_generated_dependency_file "${cuda_compile_intermediate_directory}/${generated_file_basename}.NVCC-depend")
      set(generated_cubin_file "${generated_file_path}/${generated_file_basename}.cubin.txt")
      set(custom_target_script "${cuda_compile_intermediate_directory}/${generated_file_basename}.cmake")
      set(custom_target_deps_script "${cuda_compile_intermediate_directory}/${generated_file_basename}_deps.cmake")

      # Setup properties for obj files:
      if( NOT cuda_compile_to_external_module )
        set_source_files_properties("${generated_file}"
          PROPERTIES
          EXTERNAL_OBJECT true # This is an object file not to be compiled, but only be linked.
          )
      endif()

      # Don't add CMAKE_CURRENT_SOURCE_DIR if the path is already an absolute path.
      get_filename_component(file_path "${file}" PATH)
      if(IS_ABSOLUTE "${file_path}")
        set(source_file "${file}")
      else()
        set(source_file "${CMAKE_CURRENT_SOURCE_DIR}/${file}")
      endif()

      if( NOT cuda_compile_to_external_module AND CUDA_SEPARABLE_COMPILATION)
        list(APPEND ${cuda_target}_SEPARABLE_COMPILATION_OBJECTS "${generated_file}")
      endif()

      # Bring in the dependencies.  Creates a variable CUDA_NVCC_DEPEND #######
      cuda_include_nvcc_dependencies(${cmake_dependency_file})

      # Convience string for output ###########################################
      if(CUDA_BUILD_EMULATION)
        set(cuda_build_type "Emulation")
      else()
        set(cuda_build_type "Device")
      endif()

      # Build the NVCC made dependency file ###################################
      set(build_cubin OFF)
      if ( NOT CUDA_BUILD_EMULATION AND CUDA_BUILD_CUBIN )
         if ( NOT cuda_compile_to_external_module )
           set ( build_cubin ON )
         endif()
      endif()

      # Configure the build script
      configure_file("${CUDA_run_nvcc}" "${custom_target_script}" @ONLY)

      # Configure the dependencies script
      configure_file("${CUDA_run_nvcc_deps}" "${custom_target_deps_script}" @ONLY)

      # So if a user specifies the same cuda file as input more than once, you
      # can have bad things happen with dependencies.  Here we check an option
      # to see if this is the behavior they want.
      if(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE)
        set(main_dep MAIN_DEPENDENCY ${source_file})
      else()
        set(main_dep DEPENDS ${source_file})
      endif()

      if(CUDA_VERBOSE_BUILD)
        set(verbose_output ON)
      elseif(CMAKE_GENERATOR MATCHES "Makefiles")
        set(verbose_output "$(VERBOSE)")
      else()
        set(verbose_output OFF)
      endif()

      # Create up the comment string
      file(RELATIVE_PATH generated_file_relative_path "${CMAKE_BINARY_DIR}" "${generated_file}")
      if(cuda_compile_to_external_module)
        set(cuda_build_comment_string "Building NVCC ${cuda_compile_to_external_module_type} file ${generated_file_relative_path}")
      else()
        set(cuda_build_comment_string "Building NVCC (${cuda_build_type}) object ${generated_file_relative_path}")
      endif()
    set(cuda_deps_comment_string "Updating NVCC file dependencies ${generated_file_relative_path}")

      # Build the generated file and dependency file ##########################
      add_custom_command(
        OUTPUT ${generated_file}
        # These output files depend on the source_file and the contents of cmake_dependency_file
        ${main_dep}
        DEPENDS ${CUDA_NVCC_DEPEND}
        DEPENDS ${custom_target_script}
        # Make sure the output directory exists before trying to write to it.
        COMMAND ${CMAKE_COMMAND} -E make_directory "${generated_file_path}"
        COMMAND ${CMAKE_COMMAND} ARGS
          -D verbose:BOOL=${verbose_output}
          ${ccbin_flags}
          -D build_configuration:STRING=${CUDA_build_configuration}
          -D "generated_file:STRING=${generated_file}"
          -D "generated_cubin_file:STRING=${generated_cubin_file}"
          -P "${custom_target_script}"
        WORKING_DIRECTORY "${cuda_compile_intermediate_directory}"
        COMMENT "${cuda_build_comment_string}"
        )

      #Store all the info to make the update dependency command
      # Note: cannot create command here because we want to create a custom target
      # that depends on all these commands but dependencies only work between targets
      # and commands created in the same dir.
      set(ccbin_flags_str "${ccbin_flags}")
      string(REPLACE ";" "" ccbin_flags_str "${ccbin_flags_str}")

      set(depends_str "${CUDA_NVCC_DEPEND}")
      string(REPLACE ";" "|" depends_str "${depends_str}")
      set(depends_str "${source_file}|${custom_target_deps_script}|${depends_str}")

      set(DEP_DIR ${cuda_compile_intermediate_directory})

      set(CUDA_DEPENDENCY_SCRIPTS ${CUDA_DEPENDENCY_SCRIPTS} "${custom_target_deps_script}" CACHE INTERNAL "")
      set(CUDA_DEPENDENCY_OUT_FILES ${CUDA_DEPENDENCY_OUT_FILES} "${cmake_dependency_file}.stamp" CACHE INTERNAL "")
      set(CUDA_DEPENDENCY_FLAGS ${CUDA_DEPENDENCY_FLAGS} "${ccbin_flags_str} " CACHE INTERNAL "")
      set(CUDA_DEPENDENCY_DEPEND_FILES ${CUDA_DEPENDENCY_DEPEND_FILES} "${depends_str}" CACHE INTERNAL "")
      set(CUDA_DEPENDENCY_COMMANDS ${CUDA_DEPENDENCY_COMMANDS} ${DEP_COMMAND} CACHE INTERNAL "")
      set(CUDA_DEPENDENCY_DIRS ${CUDA_DEPENDENCY_DIRS} ${DEP_DIR} CACHE INTERNAL "")

      # Make sure the build system knows the file is generated.
      set_source_files_properties(${generated_file} PROPERTIES GENERATED TRUE)

      list(APPEND _cuda_wrap_generated_files ${generated_file})

      # Add the other files that we want cmake to clean on a cleanup ##########
      list(APPEND CUDA_ADDITIONAL_CLEAN_FILES "${cmake_dependency_file}")
      list(REMOVE_DUPLICATES CUDA_ADDITIONAL_CLEAN_FILES)
      set(CUDA_ADDITIONAL_CLEAN_FILES ${CUDA_ADDITIONAL_CLEAN_FILES} CACHE INTERNAL "List of intermediate files that are part of the cuda dependency scanning.")

    endif()
  endforeach()

  # Set the return parameter
  set(${generated_files} ${_cuda_wrap_generated_files})
endmacro()

function(_cuda_get_important_host_flags important_flags flag_string)
  if(CMAKE_GENERATOR MATCHES "Visual Studio")
    string(REGEX MATCHALL "/M[DT][d]?" flags ${flag_string})
    list(APPEND ${important_flags} ${flags})
  else()
    string(REGEX MATCHALL "-fPIC" flags ${flag_string})
    list(APPEND ${important_flags} ${flags})
  endif()
  set(${important_flags} ${${important_flags}} PARENT_SCOPE)
endfunction()

###############################################################################
###############################################################################
# Create a target to update all cuda dependencies
###############################################################################
###############################################################################
function(ADD_CUDA_DEPENDENCIES_TARGET target_name is_all)
    if(CMAKE_GENERATOR MATCHES "Visual Studio")
      set( CUDA_build_configuration "$(ConfigurationName)" )
    else()
      set( CUDA_build_configuration "${CMAKE_BUILD_TYPE}")
    endif()

    if(CUDA_VERBOSE_BUILD)
      set(verbose_output ON)
    elseif(CMAKE_GENERATOR MATCHES "Makefiles")
      set(verbose_output "$(VERBOSE)")
    else()
      set(verbose_output OFF)
    endif()

    set(all_deps)
    if(CUDA_DEPENDENCY_OUT_FILES)
        # Only loop if list not empty

        # Iterate the dependency command list and create a custom command
        # to update the dependencies of each cuda file
        list(LENGTH CUDA_DEPENDENCY_OUT_FILES dep_count)
        math(EXPR dep_count_m "${dep_count}-1")

        foreach(dep_idx RANGE ${dep_count_m})
            list(GET CUDA_DEPENDENCY_SCRIPTS ${dep_idx} dep_script)
            list(GET CUDA_DEPENDENCY_OUT_FILES ${dep_idx} out_file)
            list(GET CUDA_DEPENDENCY_FLAGS ${dep_idx} dep_flags)
            list(GET CUDA_DEPENDENCY_DEPEND_FILES ${dep_idx} dep_files)
            list(GET CUDA_DEPENDENCY_DIRS ${dep_idx} dep_dir)
            string(REPLACE "|" ";" dep_files ${dep_files})

              add_custom_command(
                OUTPUT ${out_file}
                DEPENDS ${dep_files}
                COMMAND ${CMAKE_COMMAND} ARGS
                   -D verbose:BOOL=${verbose_output}
                   ${dep_flags}
                   -D build_configuration:STRING=${CUDA_build_configuration}
                   -P "${dep_script}"
                WORKING_DIRECTORY "${dep_dir}"
                COMMENT "Dependencies for ${out_file}"
              )
            set(all_deps ${all_deps} ${out_file})
        endforeach()
    endif()

    # Create a final custom target that will update all dependencies
    if(${is_all})
        add_custom_target(${target_name} ALL
            DEPENDS ${all_deps}
        )
    else()
        add_custom_target(${target_name}
            DEPENDS ${all_deps}
        )
    endif()

  # Reset all dependencies
  unset(CUDA_DEPENDENCY_SCRIPTS CACHE)
  unset(CUDA_DEPENDENCY_OUT_FILES CACHE)
  unset(CUDA_DEPENDENCY_FLAGS CACHE)
  unset(CUDA_DEPENDENCY_DEPEND_FILES CACHE)
  unset(CUDA_DEPENDENCY_DIRS CACHE)
endfunction()

function(CUDA_RESET_INTERNAL_CACHE)
    #check for non-empty vars
    if(CUDA_DEPENDENCY_OUT_FILES)
        message(WARNING "Not all cuda files have a target to update its dependencies. \
                         Call add_cuda_dependencies_target() to create the appropriate target.")
    endif()
    unset(CUDA_DEPENDENCY_SCRIPTS CACHE)
    unset(CUDA_DEPENDENCY_OUT_FILES CACHE)
    unset(CUDA_DEPENDENCY_FLAGS CACHE)
    unset(CUDA_DEPENDENCY_DEPEND_FILES CACHE)
    unset(CUDA_DEPENDENCY_DIRS CACHE)
endfunction()

###############################################################################
###############################################################################
# Separable Compilation Link
###############################################################################
###############################################################################

# Compute the filename to be used by CUDA_LINK_SEPARABLE_COMPILATION_OBJECTS
function(CUDA_COMPUTE_SEPARABLE_COMPILATION_OBJECT_FILE_NAME output_file_var cuda_target object_files)
  if (object_files)
    set(generated_extension ${CMAKE_${CUDA_C_OR_CXX}_OUTPUT_EXTENSION})
    set(output_file "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${cuda_target}.dir/${CMAKE_CFG_INTDIR}/${cuda_target}_intermediate_link${generated_extension}")
  else()
    set(output_file)
  endif()

  set(${output_file_var} "${output_file}" PARENT_SCOPE)
endfunction()

# Setup the build rule for the separable compilation intermediate link file.
function(CUDA_LINK_SEPARABLE_COMPILATION_OBJECTS output_file cuda_target options object_files)
  if (object_files)

    set_source_files_properties("${output_file}"
      PROPERTIES
      EXTERNAL_OBJECT TRUE # This is an object file not to be compiled, but only
                           # be linked.
      GENERATED TRUE       # This file is generated during the build
      )

    # For now we are ignoring all the configuration specific flags.
    set(nvcc_flags)
    CUDA_PARSE_NVCC_OPTIONS(nvcc_flags ${options})
    if(CUDA_64_BIT_DEVICE_CODE)
      list(APPEND nvcc_flags -m64)
    else()
      list(APPEND nvcc_flags -m32)
    endif()
    # If -ccbin, --compiler-bindir has been specified, don't do anything.  Otherwise add it here.
    list( FIND nvcc_flags "-ccbin" ccbin_found0 )
    list( FIND nvcc_flags "--compiler-bindir" ccbin_found1 )
    if( ccbin_found0 LESS 0 AND ccbin_found1 LESS 0 AND CUDA_HOST_COMPILER )
      list(APPEND nvcc_flags -ccbin "\"${CUDA_HOST_COMPILER}\"")
    endif()

    # Create a list of flags specified by CUDA_NVCC_FLAGS_${CONFIG} and CMAKE_${CUDA_C_OR_CXX}_FLAGS*
    set(config_specific_flags)
    set(flags)
    foreach(config ${CUDA_configuration_types})
      string(TOUPPER ${config} config_upper)
      # Add config specific flags
      foreach(f ${CUDA_NVCC_FLAGS_${config_upper}})
        list(APPEND config_specific_flags $<$<CONFIG:${config}>:${f}>)
      endforeach()
      set(important_host_flags)
      _cuda_get_important_host_flags(important_host_flags ${CMAKE_${CUDA_C_OR_CXX}_FLAGS_${config_upper}})
      foreach(f ${important_host_flags})
        list(APPEND flags $<$<CONFIG:${config}>:-Xcompiler> $<$<CONFIG:${config}>:${f}>)
      endforeach()
    endforeach()
    # Add CMAKE_${CUDA_C_OR_CXX}_FLAGS
    set(important_host_flags)
    _cuda_get_important_host_flags(important_host_flags ${CMAKE_${CUDA_C_OR_CXX}_FLAGS})
    foreach(f ${important_host_flags})
      list(APPEND flags -Xcompiler ${f})
    endforeach()

    # Add our general CUDA_NVCC_FLAGS with the configuration specifig flags
    set(nvcc_flags ${CUDA_NVCC_FLAGS} ${config_specific_flags} ${nvcc_flags})

    file(RELATIVE_PATH output_file_relative_path "${CMAKE_BINARY_DIR}" "${output_file}")

    # Some generators don't handle the multiple levels of custom command
    # dependencies correctly (obj1 depends on file1, obj2 depends on obj1), so
    # we work around that issue by compiling the intermediate link object as a
    # pre-link custom command in that situation.
    set(do_obj_build_rule TRUE)
    if (MSVC_VERSION GREATER 1599)
      # VS 2010 and 2012 have this problem.  If future versions fix this issue,
      # it should still work, it just won't be as nice as the other method.
      set(do_obj_build_rule FALSE)
    endif()

    if (do_obj_build_rule)
      add_custom_command(
        OUTPUT ${output_file}
        DEPENDS ${object_files}
        COMMAND ${CUDA_NVCC_EXECUTABLE} ${nvcc_flags} -dlink ${object_files} -o ${output_file}
        ${flags}
        COMMENT "Building NVCC intermediate link file ${output_file_relative_path}"
        )
    else()
      add_custom_command(
        TARGET ${cuda_target}
        PRE_LINK
        COMMAND ${CMAKE_COMMAND} -E echo "Building NVCC intermediate link file ${output_file_relative_path}"
        COMMAND ${CUDA_NVCC_EXECUTABLE} ${nvcc_flags} ${flags} -dlink ${object_files} -o "${output_file}"
        )
    endif()
 endif()
endfunction()

###############################################################################
###############################################################################
# ADD LIBRARY
###############################################################################
###############################################################################
macro(CUDA_ADD_LIBRARY cuda_target)

  CUDA_ADD_CUDA_INCLUDE_ONCE()

  # Separate the sources from the options
  CUDA_GET_SOURCES_AND_OPTIONS(_sources _cmake_options _options ${ARGN})
  CUDA_BUILD_SHARED_LIBRARY(_cuda_shared_flag ${ARGN})
  # Create custom commands and targets for each file.
  CUDA_WRAP_SRCS( ${cuda_target} OBJ _generated_files ${_sources}
    ${_cmake_options} ${_cuda_shared_flag}
    OPTIONS ${_options} )

  # Compute the file name of the intermedate link file used for separable
  # compilation.
  CUDA_COMPUTE_SEPARABLE_COMPILATION_OBJECT_FILE_NAME(link_file ${cuda_target} "${${cuda_target}_SEPARABLE_COMPILATION_OBJECTS}")

  # Add the library.
  add_library(${cuda_target} ${_cmake_options}
    ${_generated_files}
    ${_sources}
    ${link_file}
    )

  # Add a link phase for the separable compilation if it has been enabled.  If
  # it has been enabled then the ${cuda_target}_SEPARABLE_COMPILATION_OBJECTS
  # variable will have been defined.
  CUDA_LINK_SEPARABLE_COMPILATION_OBJECTS("${link_file}" ${cuda_target} "${_options}" "${${cuda_target}_SEPARABLE_COMPILATION_OBJECTS}")

  target_link_libraries(${cuda_target} PRIVATE
    ${CUDA_LIBRARIES}
    )

  # We need to set the linker language based on what the expected generated file
  # would be. CUDA_C_OR_CXX is computed based on CUDA_HOST_COMPILATION_CPP.
  set_target_properties(${cuda_target}
    PROPERTIES
    LINKER_LANGUAGE ${CUDA_C_OR_CXX}
    )

endmacro()


###############################################################################
###############################################################################
# ADD EXECUTABLE
###############################################################################
###############################################################################
macro(CUDA_ADD_EXECUTABLE cuda_target)

  CUDA_ADD_CUDA_INCLUDE_ONCE()

  # Separate the sources from the options
  CUDA_GET_SOURCES_AND_OPTIONS(_sources _cmake_options _options ${ARGN})
  # Create custom commands and targets for each file.
  CUDA_WRAP_SRCS( ${cuda_target} OBJ _generated_files ${_sources} OPTIONS ${_options} )

  # Compute the file name of the intermedate link file used for separable
  # compilation.
  CUDA_COMPUTE_SEPARABLE_COMPILATION_OBJECT_FILE_NAME(link_file ${cuda_target} "${${cuda_target}_SEPARABLE_COMPILATION_OBJECTS}")

  # Add the library.
  add_executable(${cuda_target} ${_cmake_options}
    ${_generated_files}
    ${_sources}
    ${link_file}
    )

  # Add a link phase for the separable compilation if it has been enabled.  If
  # it has been enabled then the ${cuda_target}_SEPARABLE_COMPILATION_OBJECTS
  # variable will have been defined.
  CUDA_LINK_SEPARABLE_COMPILATION_OBJECTS("${link_file}" ${cuda_target} "${_options}" "${${cuda_target}_SEPARABLE_COMPILATION_OBJECTS}")

  target_link_libraries(${cuda_target} PRIVATE
    ${CUDA_LIBRARIES}
    )

  # We need to set the linker language based on what the expected generated file
  # would be. CUDA_C_OR_CXX is computed based on CUDA_HOST_COMPILATION_CPP.
  set_target_properties(${cuda_target}
    PROPERTIES
    LINKER_LANGUAGE ${CUDA_C_OR_CXX}
    )

endmacro()


###############################################################################
###############################################################################
# (Internal) helper for manually added cuda source files with specific targets
###############################################################################
###############################################################################
macro(cuda_compile_base cuda_target format generated_files)

  # Separate the sources from the options
  CUDA_GET_SOURCES_AND_OPTIONS(_sources _cmake_options _options ${ARGN})
  # Create custom commands and targets for each file.
  CUDA_WRAP_SRCS( ${cuda_target} ${format} _generated_files ${_sources} ${_cmake_options}
    OPTIONS ${_options} )

  set( ${generated_files} ${_generated_files})

endmacro()

###############################################################################
###############################################################################
# CUDA COMPILE
###############################################################################
###############################################################################
macro(CUDA_COMPILE generated_files)
  cuda_compile_base(cuda_compile OBJ ${generated_files} ${ARGN})
endmacro()

###############################################################################
###############################################################################
# CUDA COMPILE PTX
###############################################################################
###############################################################################
macro(CUDA_COMPILE_PTX generated_files)
  cuda_compile_base(cuda_compile_ptx PTX ${generated_files} ${ARGN})
endmacro()

###############################################################################
###############################################################################
# CUDA COMPILE FATBIN
###############################################################################
###############################################################################
macro(CUDA_COMPILE_FATBIN generated_files)
  cuda_compile_base(cuda_compile_fatbin FATBIN ${generated_files} ${ARGN})
endmacro()

###############################################################################
###############################################################################
# CUDA COMPILE CUBIN
###############################################################################
###############################################################################
macro(CUDA_COMPILE_CUBIN generated_files)
  cuda_compile_base(cuda_compile_cubin CUBIN ${generated_files} ${ARGN})
endmacro()


###############################################################################
###############################################################################
# CUDA ADD CUFFT TO TARGET
###############################################################################
###############################################################################
macro(CUDA_ADD_CUFFT_TO_TARGET target)
  if (CUDA_BUILD_EMULATION)
    target_link_libraries(${target} PRIVATE ${CUDA_cufftemu_LIBRARY})
  else()
    target_link_libraries(${target} PRIVATE ${CUDA_cufft_LIBRARY})
  endif()
endmacro()

###############################################################################
###############################################################################
# CUDA ADD CUBLAS TO TARGET
###############################################################################
###############################################################################
macro(CUDA_ADD_CUBLAS_TO_TARGET target)
  if (CUDA_BUILD_EMULATION)
    target_link_libraries(${target} PRIVATE ${CUDA_cublasemu_LIBRARY})
  else()
    target_link_libraries(${target} PRIVATE ${CUDA_cublas_LIBRARY})
  endif()
endmacro()

###############################################################################
###############################################################################
# CUDA BUILD CLEAN TARGET
###############################################################################
###############################################################################
macro(CUDA_BUILD_CLEAN_TARGET)
  # Call this after you add all your CUDA targets, and you will get a convience
  # target.  You should also make clean after running this target to get the
  # build system to generate all the code again.

  set(cuda_clean_target_name clean_cuda_depends)
  if (CMAKE_GENERATOR MATCHES "Visual Studio")
    string(TOUPPER ${cuda_clean_target_name} cuda_clean_target_name)
  endif()
  add_custom_target(${cuda_clean_target_name}
    COMMAND ${CMAKE_COMMAND} -E remove ${CUDA_ADDITIONAL_CLEAN_FILES})

  # Clear out the variable, so the next time we configure it will be empty.
  # This is useful so that the files won't persist in the list after targets
  # have been removed.
  set(CUDA_ADDITIONAL_CLEAN_FILES "" CACHE INTERNAL "List of intermediate files that are part of the cuda dependency scanning.")
endmacro()

###############################################################################
###############################################################################
# CUDA LOCAL GPU ARCH DETERMINATION
###############################################################################
###############################################################################
if(CUDA_DETERMINE_HOST_GPU_CODE_FLAGS)
  if(NOT CUDA_HOST_GPU_CODE_FLAGS)
    execute_process(COMMAND ${CMAKE_CURRENT_LIST_DIR}/FindCUDA/get_cuda_sm.sh
                    RESULT_VARIABLE result
                    OUTPUT_VARIABLE output)
    if(NOT result)
      string(STRIP ${output} output)
      set(CUDA_HOST_GPU_CODE_FLAGS ${output} CACHE STRING "NVCC GPU code flags determined by host introspection")
      message(STATUS "Determined CUDA_HOST_GPU_CODE_FLAGS ${CUDA_HOST_GPU_CODE_FLAGS}")
    endif()
  endif()
else()
  unset(CUDA_HOST_GPU_CODE_FLAGS CACHE) # make sure no previously cached results are used
endif()
