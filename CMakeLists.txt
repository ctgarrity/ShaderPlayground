cmake_minimum_required(VERSION 4.0)
project(ShaderPlayground)

set(CMAKE_CXX_STANDARD 20)

find_package (Vulkan REQUIRED)
add_subdirectory(vendored/SDL EXCLUDE_FROM_ALL)
add_subdirectory(vendored/fastgltf EXCLUDE_FROM_ALL)

add_executable(ShaderPlayground
        src/main.cpp
        src/Renderer.cpp
        src/external/VkBootstrap.cpp
        vendored/imgui/imgui.cpp
        vendored/imgui/imgui_draw.cpp
        vendored/imgui/imgui_tables.cpp
        vendored/imgui/imgui_widgets.cpp
        vendored/imgui/imgui_demo.cpp
        vendored/imgui/backends/imgui_impl_vulkan.cpp
        vendored/imgui/backends/imgui_impl_sdl3.cpp
        src/Initializers.cpp
        src/Utilities.cpp
        src/VmaUsage.cpp
        src/PipelineBuilder.cpp
        src/Loader.cpp
        src/Descriptors.cpp
        src/Camera.cpp
        src/StbUsage.cpp
        src/Actor.cpp
)

target_compile_definitions(ShaderPlayground PRIVATE GLM_ENABLE_EXPERIMENTAL GLM_FORCE_RADIANS GLM_FORCE_DEPTH_ZERO_TO_ONE)
target_include_directories(ShaderPlayground PRIVATE ${Vulkan_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/vendored/imgui ${CMAKE_SOURCE_DIR}/vendored/fastgltf/include)
target_link_libraries(ShaderPlayground PRIVATE Vulkan::Vulkan SDL3::SDL3 fastgltf::fastgltf)