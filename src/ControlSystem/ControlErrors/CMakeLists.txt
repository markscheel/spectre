# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ControlErrors)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  SizeControlInfo.cpp
  SizeControlStateAhSpeed.cpp
  SizeControlStateDeltaR.cpp
  SizeControlStateInitial.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Expansion.hpp
  Rotation.hpp
  SizeControlInfo.hpp
  SizeControlState.hpp
  SizeControlStateAhSpeed.hpp
  SizeControlStateDeltaR.hpp
  SizeControlStateInitial.hpp
  Translation.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Boost::boost
  DataStructures
  ErrorHandling
  )
