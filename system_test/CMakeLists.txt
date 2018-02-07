cmake_minimum_required (VERSION 2.8.8)


FILE(GLOB_RECURSE SRC src/*.cpp)

ADD_EXECUTABLE(alssystemtest ${SRC})

TARGET_LINK_LIBRARIES(alssystemtest alsfvm GTest::GTest GTest::Main)
