# Google Flags
find_package(gflags REQUIRED)
# TODO
#find_package(glog 0.4.0 REQUIRED)

add_executable(pt main.cpp scene_factory.cpp)
target_link_libraries(pt PRIVATE tinypt gflags glog)
