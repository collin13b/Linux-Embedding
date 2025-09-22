cmake_minimum_required(VERSION 3.10)
project(prj)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# 输出系统的基本信息，包括系统名称和版本号，便于调试和确认构建环境。
message(STATUS "Ststem: ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_VERSION}")
#寻找opencv库
find_package(OpenCV REQUIRED)
# 输出 OpenCV 库的信息，包括版本、库路径和头文件路径，帮助确认配置是否正确。
message(STATUS "Opencv version: ${OpenCV_VERSION}")# 输出 OpenCV 的版本
message(STATUS "Opencv lib ${OpenCV_LIBS}")#输出 OpenCV 链接的库
message(STATUS "Opencv include path ${OpenCV_INCLUDE_DIRS}")# 输出 OpenCV 包含文件的路径
message("this is a prj")
set(LIB_ARCH "aarch64")
set(DEVICE_NAME "RK3588")

# 设置第三方库的路径

# 这使得源文件可以正确包含 OpenCV 的头文件。
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    
)
add_executable(prj 
               main.cpp
               threadpool.cpp
)
# 将 OpenCV 的库与目标可执行文件 cv 链接，确保在程序运行时能够调用 OpenCV 函数。
target_link_libraries(prj 
                        ${OpenCV_LIBS}
                        pthread
)           