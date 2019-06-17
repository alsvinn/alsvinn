SET(ALSVINN_MAJOR_VERSION "${PROJECT_VERSION_MAJOR}")
SET(ALSVINN_MINOR_VERSION "${PROJECT_VERSION_MINOR}")
SET(ALSVINN_PATCH_VERSION "${PROJECT_VERSION_PATCH}")
SET(ALSVINN_VERSION "${ALSVINN_MAJOR_VERSION}.${ALSVINN_MINOR_VERSION}.${ALSVINN_PATCH_VERSION}")



if (EXISTS "${CMAKE_SOURCE_DIR}/cmake/git_find/GetGitRevisionDescription.cmake")
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/git_find")
  include(GetGitRevisionDescription)
  get_git_head_revision(GIT_REFSPEC GIT_SHA1)
  git_local_changes(GIT_VERSION_STATUS)
else()
  message(WARNING "Could not find the git_find submodule, try a \n\tgit submodule update --init")
  SET(GIT_REFSPEC_GIT_SHA1 "Alsvinn version ${ALSVINN_VERSION} (could not get git commit)")
  SET(GIT_VERSION_STATUS, "Clean (could not get git version)")
endif()
