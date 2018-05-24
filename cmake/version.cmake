# Set this to ON only for commits you will tag
set(IS_TAGGED_RELEASE_VERSION OFF)

SET(ALSVINN_MAJOR_VERSION "0")
SET(ALSVINN_MINOR_VERSION "1")
SET(ALSVINN_PATCH_VERSION "1")
SET(ALSVINN_VERSION "${ALSVINN_MAJOR_VERSION}.${ALSVINN_MINOR_VERSION}.${ALSVINN_PATCH_VERSION}")

if (IS_TAGGED_RELEASE_VERSION)
  SET(GIT_REFSPEC_GIT_SHA1 "Alsvinn version ${ALSVINN_VERSION}")
  SET(GIT_VERSION_STATUS, "Clean (this was a pure release version)")
else()
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/git_find")
  include(GetGitRevisionDescription)
  get_git_head_revision(GIT_REFSPEC GIT_SHA1)
  git_local_changes(GIT_VERSION_STATUS)
endif()
