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
  set(_SPEC_LIB_NAME "libPackagedExporter.a")
  if (NOT EXISTS "${SPEC_EXPORTER_ROOT}/${_SPEC_LIB_NAME}")
    message(FATAL_ERROR
      "Could not find ${SPEC_EXPORTER_ROOT}/${_SPEC_LIB_NAME}")
  endif()

  add_library(SpEC::Exporter INTERFACE IMPORTED)
  target_include_directories(SpEC::Exporter INTERFACE ${SPEC_EXPORTER_ROOT})
  add_interface_lib_headers(
    TARGET SpEC::Exporter
    HEADERS
    Exporter.hpp
  )
  # The order of these next two target_link_libraries lines is important.
  target_link_libraries(SpEC::Exporter
    INTERFACE ${SPEC_EXPORTER_ROOT}/ExporterFactoryObjects.o)
  target_link_libraries(SpEC::Exporter
    INTERFACE ${SPEC_EXPORTER_ROOT}/${_SPEC_LIB_NAME})

  # NOTE: You should use the same MPI as SpEC. At least the same distribution. So
  # mixing OpenMPI and MPICH would be bad.
  find_package(MPI COMPONENTS C)
  target_link_libraries(SpEC::Exporter INTERFACE MPI::MPI_C)
endif(SPEC_ROOT)
