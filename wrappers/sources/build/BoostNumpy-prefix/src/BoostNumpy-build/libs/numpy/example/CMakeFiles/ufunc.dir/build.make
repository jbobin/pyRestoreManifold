# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.9

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/local/bin/cmake

# The command to remove a file.
RM = /opt/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/pyWrappers/MacWrappers/pystarlet_manifold/build/BoostNumpy-prefix/src/BoostNumpy

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/pyWrappers/MacWrappers/pystarlet_manifold/build/BoostNumpy-prefix/src/BoostNumpy-build

# Include any dependencies generated for this target.
include libs/numpy/example/CMakeFiles/ufunc.dir/depend.make

# Include the progress variables for this target.
include libs/numpy/example/CMakeFiles/ufunc.dir/progress.make

# Include the compile flags for this target's objects.
include libs/numpy/example/CMakeFiles/ufunc.dir/flags.make

libs/numpy/example/CMakeFiles/ufunc.dir/ufunc.cpp.o: libs/numpy/example/CMakeFiles/ufunc.dir/flags.make
libs/numpy/example/CMakeFiles/ufunc.dir/ufunc.cpp.o: /Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/pyWrappers/MacWrappers/pystarlet_manifold/build/BoostNumpy-prefix/src/BoostNumpy/libs/numpy/example/ufunc.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/pyWrappers/MacWrappers/pystarlet_manifold/build/BoostNumpy-prefix/src/BoostNumpy-build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object libs/numpy/example/CMakeFiles/ufunc.dir/ufunc.cpp.o"
	cd /Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/pyWrappers/MacWrappers/pystarlet_manifold/build/BoostNumpy-prefix/src/BoostNumpy-build/libs/numpy/example && /opt/local/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ufunc.dir/ufunc.cpp.o -c /Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/pyWrappers/MacWrappers/pystarlet_manifold/build/BoostNumpy-prefix/src/BoostNumpy/libs/numpy/example/ufunc.cpp

libs/numpy/example/CMakeFiles/ufunc.dir/ufunc.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ufunc.dir/ufunc.cpp.i"
	cd /Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/pyWrappers/MacWrappers/pystarlet_manifold/build/BoostNumpy-prefix/src/BoostNumpy-build/libs/numpy/example && /opt/local/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/pyWrappers/MacWrappers/pystarlet_manifold/build/BoostNumpy-prefix/src/BoostNumpy/libs/numpy/example/ufunc.cpp > CMakeFiles/ufunc.dir/ufunc.cpp.i

libs/numpy/example/CMakeFiles/ufunc.dir/ufunc.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ufunc.dir/ufunc.cpp.s"
	cd /Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/pyWrappers/MacWrappers/pystarlet_manifold/build/BoostNumpy-prefix/src/BoostNumpy-build/libs/numpy/example && /opt/local/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/pyWrappers/MacWrappers/pystarlet_manifold/build/BoostNumpy-prefix/src/BoostNumpy/libs/numpy/example/ufunc.cpp -o CMakeFiles/ufunc.dir/ufunc.cpp.s

libs/numpy/example/CMakeFiles/ufunc.dir/ufunc.cpp.o.requires:

.PHONY : libs/numpy/example/CMakeFiles/ufunc.dir/ufunc.cpp.o.requires

libs/numpy/example/CMakeFiles/ufunc.dir/ufunc.cpp.o.provides: libs/numpy/example/CMakeFiles/ufunc.dir/ufunc.cpp.o.requires
	$(MAKE) -f libs/numpy/example/CMakeFiles/ufunc.dir/build.make libs/numpy/example/CMakeFiles/ufunc.dir/ufunc.cpp.o.provides.build
.PHONY : libs/numpy/example/CMakeFiles/ufunc.dir/ufunc.cpp.o.provides

libs/numpy/example/CMakeFiles/ufunc.dir/ufunc.cpp.o.provides.build: libs/numpy/example/CMakeFiles/ufunc.dir/ufunc.cpp.o


# Object files for target ufunc
ufunc_OBJECTS = \
"CMakeFiles/ufunc.dir/ufunc.cpp.o"

# External object files for target ufunc
ufunc_EXTERNAL_OBJECTS =

bin/ufunc: libs/numpy/example/CMakeFiles/ufunc.dir/ufunc.cpp.o
bin/ufunc: libs/numpy/example/CMakeFiles/ufunc.dir/build.make
bin/ufunc: lib/libboost_numpy.a
bin/ufunc: /usr/local/lib/libboost_python.dylib
bin/ufunc: /opt/local/lib/libpython2.7.dylib
bin/ufunc: libs/numpy/example/CMakeFiles/ufunc.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/pyWrappers/MacWrappers/pystarlet_manifold/build/BoostNumpy-prefix/src/BoostNumpy-build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../../bin/ufunc"
	cd /Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/pyWrappers/MacWrappers/pystarlet_manifold/build/BoostNumpy-prefix/src/BoostNumpy-build/libs/numpy/example && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ufunc.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
libs/numpy/example/CMakeFiles/ufunc.dir/build: bin/ufunc

.PHONY : libs/numpy/example/CMakeFiles/ufunc.dir/build

libs/numpy/example/CMakeFiles/ufunc.dir/requires: libs/numpy/example/CMakeFiles/ufunc.dir/ufunc.cpp.o.requires

.PHONY : libs/numpy/example/CMakeFiles/ufunc.dir/requires

libs/numpy/example/CMakeFiles/ufunc.dir/clean:
	cd /Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/pyWrappers/MacWrappers/pystarlet_manifold/build/BoostNumpy-prefix/src/BoostNumpy-build/libs/numpy/example && $(CMAKE_COMMAND) -P CMakeFiles/ufunc.dir/cmake_clean.cmake
.PHONY : libs/numpy/example/CMakeFiles/ufunc.dir/clean

libs/numpy/example/CMakeFiles/ufunc.dir/depend:
	cd /Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/pyWrappers/MacWrappers/pystarlet_manifold/build/BoostNumpy-prefix/src/BoostNumpy-build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/pyWrappers/MacWrappers/pystarlet_manifold/build/BoostNumpy-prefix/src/BoostNumpy /Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/pyWrappers/MacWrappers/pystarlet_manifold/build/BoostNumpy-prefix/src/BoostNumpy/libs/numpy/example /Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/pyWrappers/MacWrappers/pystarlet_manifold/build/BoostNumpy-prefix/src/BoostNumpy-build /Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/pyWrappers/MacWrappers/pystarlet_manifold/build/BoostNumpy-prefix/src/BoostNumpy-build/libs/numpy/example /Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/pyWrappers/MacWrappers/pystarlet_manifold/build/BoostNumpy-prefix/src/BoostNumpy-build/libs/numpy/example/CMakeFiles/ufunc.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : libs/numpy/example/CMakeFiles/ufunc.dir/depend

