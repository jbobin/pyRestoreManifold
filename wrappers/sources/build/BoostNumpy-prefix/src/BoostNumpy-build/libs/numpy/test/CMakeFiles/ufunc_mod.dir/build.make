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
include libs/numpy/test/CMakeFiles/ufunc_mod.dir/depend.make

# Include the progress variables for this target.
include libs/numpy/test/CMakeFiles/ufunc_mod.dir/progress.make

# Include the compile flags for this target's objects.
include libs/numpy/test/CMakeFiles/ufunc_mod.dir/flags.make

libs/numpy/test/CMakeFiles/ufunc_mod.dir/ufunc_mod.cpp.o: libs/numpy/test/CMakeFiles/ufunc_mod.dir/flags.make
libs/numpy/test/CMakeFiles/ufunc_mod.dir/ufunc_mod.cpp.o: /Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/pyWrappers/MacWrappers/pystarlet_manifold/build/BoostNumpy-prefix/src/BoostNumpy/libs/numpy/test/ufunc_mod.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/pyWrappers/MacWrappers/pystarlet_manifold/build/BoostNumpy-prefix/src/BoostNumpy-build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object libs/numpy/test/CMakeFiles/ufunc_mod.dir/ufunc_mod.cpp.o"
	cd /Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/pyWrappers/MacWrappers/pystarlet_manifold/build/BoostNumpy-prefix/src/BoostNumpy-build/libs/numpy/test && /opt/local/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ufunc_mod.dir/ufunc_mod.cpp.o -c /Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/pyWrappers/MacWrappers/pystarlet_manifold/build/BoostNumpy-prefix/src/BoostNumpy/libs/numpy/test/ufunc_mod.cpp

libs/numpy/test/CMakeFiles/ufunc_mod.dir/ufunc_mod.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ufunc_mod.dir/ufunc_mod.cpp.i"
	cd /Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/pyWrappers/MacWrappers/pystarlet_manifold/build/BoostNumpy-prefix/src/BoostNumpy-build/libs/numpy/test && /opt/local/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/pyWrappers/MacWrappers/pystarlet_manifold/build/BoostNumpy-prefix/src/BoostNumpy/libs/numpy/test/ufunc_mod.cpp > CMakeFiles/ufunc_mod.dir/ufunc_mod.cpp.i

libs/numpy/test/CMakeFiles/ufunc_mod.dir/ufunc_mod.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ufunc_mod.dir/ufunc_mod.cpp.s"
	cd /Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/pyWrappers/MacWrappers/pystarlet_manifold/build/BoostNumpy-prefix/src/BoostNumpy-build/libs/numpy/test && /opt/local/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/pyWrappers/MacWrappers/pystarlet_manifold/build/BoostNumpy-prefix/src/BoostNumpy/libs/numpy/test/ufunc_mod.cpp -o CMakeFiles/ufunc_mod.dir/ufunc_mod.cpp.s

libs/numpy/test/CMakeFiles/ufunc_mod.dir/ufunc_mod.cpp.o.requires:

.PHONY : libs/numpy/test/CMakeFiles/ufunc_mod.dir/ufunc_mod.cpp.o.requires

libs/numpy/test/CMakeFiles/ufunc_mod.dir/ufunc_mod.cpp.o.provides: libs/numpy/test/CMakeFiles/ufunc_mod.dir/ufunc_mod.cpp.o.requires
	$(MAKE) -f libs/numpy/test/CMakeFiles/ufunc_mod.dir/build.make libs/numpy/test/CMakeFiles/ufunc_mod.dir/ufunc_mod.cpp.o.provides.build
.PHONY : libs/numpy/test/CMakeFiles/ufunc_mod.dir/ufunc_mod.cpp.o.provides

libs/numpy/test/CMakeFiles/ufunc_mod.dir/ufunc_mod.cpp.o.provides.build: libs/numpy/test/CMakeFiles/ufunc_mod.dir/ufunc_mod.cpp.o


# Object files for target ufunc_mod
ufunc_mod_OBJECTS = \
"CMakeFiles/ufunc_mod.dir/ufunc_mod.cpp.o"

# External object files for target ufunc_mod
ufunc_mod_EXTERNAL_OBJECTS =

lib/ufunc_mod.so: libs/numpy/test/CMakeFiles/ufunc_mod.dir/ufunc_mod.cpp.o
lib/ufunc_mod.so: libs/numpy/test/CMakeFiles/ufunc_mod.dir/build.make
lib/ufunc_mod.so: lib/libboost_numpy.a
lib/ufunc_mod.so: /usr/local/lib/libboost_python.dylib
lib/ufunc_mod.so: /opt/local/lib/libpython2.7.dylib
lib/ufunc_mod.so: libs/numpy/test/CMakeFiles/ufunc_mod.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/pyWrappers/MacWrappers/pystarlet_manifold/build/BoostNumpy-prefix/src/BoostNumpy-build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared module ../../../lib/ufunc_mod.so"
	cd /Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/pyWrappers/MacWrappers/pystarlet_manifold/build/BoostNumpy-prefix/src/BoostNumpy-build/libs/numpy/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ufunc_mod.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
libs/numpy/test/CMakeFiles/ufunc_mod.dir/build: lib/ufunc_mod.so

.PHONY : libs/numpy/test/CMakeFiles/ufunc_mod.dir/build

libs/numpy/test/CMakeFiles/ufunc_mod.dir/requires: libs/numpy/test/CMakeFiles/ufunc_mod.dir/ufunc_mod.cpp.o.requires

.PHONY : libs/numpy/test/CMakeFiles/ufunc_mod.dir/requires

libs/numpy/test/CMakeFiles/ufunc_mod.dir/clean:
	cd /Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/pyWrappers/MacWrappers/pystarlet_manifold/build/BoostNumpy-prefix/src/BoostNumpy-build/libs/numpy/test && $(CMAKE_COMMAND) -P CMakeFiles/ufunc_mod.dir/cmake_clean.cmake
.PHONY : libs/numpy/test/CMakeFiles/ufunc_mod.dir/clean

libs/numpy/test/CMakeFiles/ufunc_mod.dir/depend:
	cd /Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/pyWrappers/MacWrappers/pystarlet_manifold/build/BoostNumpy-prefix/src/BoostNumpy-build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/pyWrappers/MacWrappers/pystarlet_manifold/build/BoostNumpy-prefix/src/BoostNumpy /Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/pyWrappers/MacWrappers/pystarlet_manifold/build/BoostNumpy-prefix/src/BoostNumpy/libs/numpy/test /Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/pyWrappers/MacWrappers/pystarlet_manifold/build/BoostNumpy-prefix/src/BoostNumpy-build /Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/pyWrappers/MacWrappers/pystarlet_manifold/build/BoostNumpy-prefix/src/BoostNumpy-build/libs/numpy/test /Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/pyWrappers/MacWrappers/pystarlet_manifold/build/BoostNumpy-prefix/src/BoostNumpy-build/libs/numpy/test/CMakeFiles/ufunc_mod.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : libs/numpy/test/CMakeFiles/ufunc_mod.dir/depend

