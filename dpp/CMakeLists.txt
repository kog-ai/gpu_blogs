cmake_minimum_required(VERSION 3.21)

project(dpp_examples LANGUAGES HIP)

file(GLOB HIP_EXAMPLE_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.hip"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.hip.cpp"
)

foreach(src IN LISTS HIP_EXAMPLE_SOURCES)
  get_filename_component(tgt "${src}" NAME_WE)
  add_executable(${tgt} "${src}")
  target_include_directories(${tgt} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  set_target_properties(
    ${tgt}
    PROPERTIES
      HIP_STANDARD 20
      HIP_STANDARD_REQUIRED ON
  )
endforeach()

if(NOT HIP_EXAMPLE_SOURCES)
  message(WARNING "No .hip or .hip.cpp files found in ${CMAKE_CURRENT_SOURCE_DIR}")
endif()
