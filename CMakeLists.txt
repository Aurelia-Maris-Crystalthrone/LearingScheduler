cmake_minimum_required(VERSION 3.15)
project(LRSchedulerDemo)

# 查找 gflags 库（支持 vcpkg 和系统安装）
find_package(gflags REQUIRED)

# 添加可执行文件
add_executable(lr_scheduler_demo examples/main.cc)

# 包含目录（自动处理）
target_include_directories(lr_scheduler_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# 链接 gflags
target_link_libraries(lr_scheduler_demo gflags::gflags)

# 设置 C++ 标准
set_target_properties(lr_scheduler_demo PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
)