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
CMAKE_SOURCE_DIR = /Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/pyWrappers/MacWrappers/pystarlet_manifold

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/pyWrappers/MacWrappers/pystarlet_manifold/build

# Include any dependencies generated for this target.
include CMakeFiles/sparse2d_Sn.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/sparse2d_Sn.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/sparse2d_Sn.dir/flags.make

CMakeFiles/sparse2d_Sn.dir/sparse2d/cxx/sparse2d.cpp.o: CMakeFiles/sparse2d_Sn.dir/flags.make
CMakeFiles/sparse2d_Sn.dir/sparse2d/cxx/sparse2d.cpp.o: ../sparse2d/cxx/sparse2d.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/pyWrappers/MacWrappers/pystarlet_manifold/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/sparse2d_Sn.dir/sparse2d/cxx/sparse2d.cpp.o"
	/opt/local/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sparse2d_Sn.dir/sparse2d/cxx/sparse2d.cpp.o -c /Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/pyWrappers/MacWrappers/pystarlet_manifold/sparse2d/cxx/sparse2d.cpp

CMakeFiles/sparse2d_Sn.dir/sparse2d/cxx/sparse2d.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sparse2d_Sn.dir/sparse2d/cxx/sparse2d.cpp.i"
	/opt/local/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/pyWrappers/MacWrappers/pystarlet_manifold/sparse2d/cxx/sparse2d.cpp > CMakeFiles/sparse2d_Sn.dir/sparse2d/cxx/sparse2d.cpp.i

CMakeFiles/sparse2d_Sn.dir/sparse2d/cxx/sparse2d.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sparse2d_Sn.dir/sparse2d/cxx/sparse2d.cpp.s"
	/opt/local/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/pyWrappers/MacWrappers/pystarlet_manifold/sparse2d/cxx/sparse2d.cpp -o CMakeFiles/sparse2d_Sn.dir/sparse2d/cxx/sparse2d.cpp.s

CMakeFiles/sparse2d_Sn.dir/sparse2d/cxx/sparse2d.cpp.o.requires:

.PHONY : CMakeFiles/sparse2d_Sn.dir/sparse2d/cxx/sparse2d.cpp.o.requires

CMakeFiles/sparse2d_Sn.dir/sparse2d/cxx/sparse2d.cpp.o.provides: CMakeFiles/sparse2d_Sn.dir/sparse2d/cxx/sparse2d.cpp.o.requires
	$(MAKE) -f CMakeFiles/sparse2d_Sn.dir/build.make CMakeFiles/sparse2d_Sn.dir/sparse2d/cxx/sparse2d.cpp.o.provides.build
.PHONY : CMakeFiles/sparse2d_Sn.dir/sparse2d/cxx/sparse2d.cpp.o.provides

CMakeFiles/sparse2d_Sn.dir/sparse2d/cxx/sparse2d.cpp.o.provides.build: CMakeFiles/sparse2d_Sn.dir/sparse2d/cxx/sparse2d.cpp.o


CMakeFiles/sparse2d_Sn.dir/sparse2d/cxx/starlet2d.cpp.o: CMakeFiles/sparse2d_Sn.dir/flags.make
CMakeFiles/sparse2d_Sn.dir/sparse2d/cxx/starlet2d.cpp.o: ../sparse2d/cxx/starlet2d.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/pyWrappers/MacWrappers/pystarlet_manifold/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/sparse2d_Sn.dir/sparse2d/cxx/starlet2d.cpp.o"
	/opt/local/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sparse2d_Sn.dir/sparse2d/cxx/starlet2d.cpp.o -c /Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/pyWrappers/MacWrappers/pystarlet_manifold/sparse2d/cxx/starlet2d.cpp

CMakeFiles/sparse2d_Sn.dir/sparse2d/cxx/starlet2d.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sparse2d_Sn.dir/sparse2d/cxx/starlet2d.cpp.i"
	/opt/local/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/pyWrappers/MacWrappers/pystarlet_manifold/sparse2d/cxx/starlet2d.cpp > CMakeFiles/sparse2d_Sn.dir/sparse2d/cxx/starlet2d.cpp.i

CMakeFiles/sparse2d_Sn.dir/sparse2d/cxx/starlet2d.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sparse2d_Sn.dir/sparse2d/cxx/starlet2d.cpp.s"
	/opt/local/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/pyWrappers/MacWrappers/pystarlet_manifold/sparse2d/cxx/starlet2d.cpp -o CMakeFiles/sparse2d_Sn.dir/sparse2d/cxx/starlet2d.cpp.s

CMakeFiles/sparse2d_Sn.dir/sparse2d/cxx/starlet2d.cpp.o.requires:

.PHONY : CMakeFiles/sparse2d_Sn.dir/sparse2d/cxx/starlet2d.cpp.o.requires

CMakeFiles/sparse2d_Sn.dir/sparse2d/cxx/starlet2d.cpp.o.provides: CMakeFiles/sparse2d_Sn.dir/sparse2d/cxx/starlet2d.cpp.o.requires
	$(MAKE) -f CMakeFiles/sparse2d_Sn.dir/build.make CMakeFiles/sparse2d_Sn.dir/sparse2d/cxx/starlet2d.cpp.o.provides.build
.PHONY : CMakeFiles/sparse2d_Sn.dir/sparse2d/cxx/starlet2d.cpp.o.provides

CMakeFiles/sparse2d_Sn.dir/sparse2d/cxx/starlet2d.cpp.o.provides.build: CMakeFiles/sparse2d_Sn.dir/sparse2d/cxx/starlet2d.cpp.o


# Object files for target sparse2d_Sn
sparse2d_Sn_OBJECTS = \
"CMakeFiles/sparse2d_Sn.dir/sparse2d/cxx/sparse2d.cpp.o" \
"CMakeFiles/sparse2d_Sn.dir/sparse2d/cxx/starlet2d.cpp.o"

# External object files for target sparse2d_Sn
sparse2d_Sn_EXTERNAL_OBJECTS =

sparse2d_Sn.so: CMakeFiles/sparse2d_Sn.dir/sparse2d/cxx/sparse2d.cpp.o
sparse2d_Sn.so: CMakeFiles/sparse2d_Sn.dir/sparse2d/cxx/starlet2d.cpp.o
sparse2d_Sn.so: CMakeFiles/sparse2d_Sn.dir/build.make
sparse2d_Sn.so: /usr/local/lib/libboost_python.dylib
sparse2d_Sn.so: /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/libpython2.7.dylib
sparse2d_Sn.so: CMakeFiles/sparse2d_Sn.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/pyWrappers/MacWrappers/pystarlet_manifold/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX shared library sparse2d_Sn.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sparse2d_Sn.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/sparse2d_Sn.dir/build: sparse2d_Sn.so

.PHONY : CMakeFiles/sparse2d_Sn.dir/build

CMakeFiles/sparse2d_Sn.dir/requires: CMakeFiles/sparse2d_Sn.dir/sparse2d/cxx/sparse2d.cpp.o.requires
CMakeFiles/sparse2d_Sn.dir/requires: CMakeFiles/sparse2d_Sn.dir/sparse2d/cxx/starlet2d.cpp.o.requires

.PHONY : CMakeFiles/sparse2d_Sn.dir/requires

CMakeFiles/sparse2d_Sn.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/sparse2d_Sn.dir/cmake_clean.cmake
.PHONY : CMakeFiles/sparse2d_Sn.dir/clean

CMakeFiles/sparse2d_Sn.dir/depend:
	cd /Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/pyWrappers/MacWrappers/pystarlet_manifold/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/pyWrappers/MacWrappers/pystarlet_manifold /Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/pyWrappers/MacWrappers/pystarlet_manifold /Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/pyWrappers/MacWrappers/pystarlet_manifold/build /Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/pyWrappers/MacWrappers/pystarlet_manifold/build /Users/jbobin/Documents/Python/LENA_DEVL/Toolbox/pyWrappers/MacWrappers/pystarlet_manifold/build/CMakeFiles/sparse2d_Sn.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/sparse2d_Sn.dir/depend
