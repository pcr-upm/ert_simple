#------------------------------------------------------------------------------
# Test face alignment algorithms
#------------------------------------------------------------------------------
project(ert_simple)
cmake_minimum_required(VERSION 3.2)

include(${CMAKE_CURRENT_LIST_DIR}/../../CMakeLists.txt)

find_package(OpenMP)
if(OPENMP_FOUND)
  message(STATUS FOUND_OPENMP)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -std=c++11")
else()
  message(STATUS NOT_FOUND_OPENMP)
endif()

#-- Find include files
set(face_alignment_ert_simple_include
  ${faces_framework_include}
  ${CMAKE_CURRENT_LIST_DIR}/cereal
  ${CMAKE_CURRENT_LIST_DIR}/include
)

set(face_alignment_ert_simple_src
  ${faces_framework_src}
  ${CMAKE_CURRENT_LIST_DIR}/src/FaceAlignmentEnsemble.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/EnsembleTrainer.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/ChannelFeaturesSimple.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/NearestEncoding.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/LearningAlgorithmSimple.cpp
)

set(face_alignment_ert_simple_test
  ${CMAKE_CURRENT_LIST_DIR}/test/face_alignment_ert_simple_test.cpp
)

set(face_alignment_ert_simple_libs
  ${faces_framework_libs}
)

#-- Setup CMake to run tests
enable_testing()

include_directories(${face_alignment_ert_simple_include})

foreach(test ${face_alignment_ert_simple_test})
  get_filename_component(test_name ${test} NAME_WE)
  add_executable(${test_name}
    ${face_alignment_ert_simple_src}
    ${test}
  )
  target_link_libraries(${test_name} ${face_alignment_ert_simple_libs})
  add_test(NAME ${test_name} WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR} COMMAND ${test_name} --database 300w_public)
endforeach()
