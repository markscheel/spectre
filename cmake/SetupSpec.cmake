# Distributed under the MIT License.
# See LICENSE.txt for details.

option(SPEC_ROOT "Path to the git directory of SpEC." OFF)

if (DEFINED ENV{SPEC_ROOT} AND "${SPEC_ROOT}" STREQUAL "")
  set(SPEC_ROOT "$ENV{SPEC_ROOT}")
endif()

if(SPEC_ROOT)
  if (NOT EXISTS "${SPEC_ROOT}")
    message(FATAL_ERROR
      "Could not find ${SPEC_ROOT}")
  endif()
  set(SPEC_EXPORTER_ROOT ${SPEC_ROOT}/Support/ApplyObservers/Exporter)
  if (NOT EXISTS "${SPEC_EXPORTER_ROOT}")
    message(FATAL_ERROR
      "Could not find ${SPEC_EXPORTER_ROOT}")
  endif()
  if (NOT EXISTS "${SPEC_EXPORTER_ROOT}/libExporter.a")
    message(FATAL_ERROR
      "Could not find ${SPEC_EXPORTER_ROOT}/libExporter.a")
  endif()
  add_library(SpEC::Exporter INTERFACE IMPORTED)
  target_include_directories(SpEC::Exporter INTERFACE ${SPEC_EXPORTER_ROOT})
  add_interface_lib_headers(
    TARGET SpEC::Exporter
    HEADERS
    Exporter.hpp
  )
  target_link_libraries(SpEC::Exporter
    INTERFACE ${SPEC_EXPORTER_ROOT}/libExporter.a)
endif(SPEC_ROOT)
