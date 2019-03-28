cmake_minimum_required (VERSION 3.1.0)

FILE(GLOB_RECURSE SRC src/*.cpp)

FILE(GLOB_RECURSE EXAMPLES examples/*.py examples/*.xml)

ADD_EXECUTABLE(alsuqcli ${SRC} ${EXAMPLES})



TARGET_LINK_LIBRARIES(alsuqcli alsuq 
   Boost::program_options 
   Boost::system
   Boost::chrono
   Boost::thread
   Boost::date_time	
   ${OpenMP_CXX_LIB_NAMES})

INSTALL(TARGETS
    alsuqcli
    EXPORT alsvinn-targets
    RUNTIME
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT alsuqcli)
