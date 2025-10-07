#ifndef PORTFOLIO_RENDERER_H
#define PORTFOLIO_RENDERER_H

#include <deque>
#include <functional>
#include <memory>
#include <ranges>
#include <span>

#include "Descriptors.h"
#include "Types.h"

#include "external/VkBootstrap.h"
#include "SDL3/SDL.h"
#include "external/vk_mem_alloc.h"

//Todo: Change calls to push_function to be capture by value and not reference? (according to ChatGPT)
struct DeletionQueue {
    std::deque<std::function<void()>> deletion_queue;
    void push_function(std::function<void()>&& func) {
        deletion_queue.emplace_back(std::move(func));
    }
    void flush() {
        for (auto& func : std::ranges::reverse_view(deletion_queue)) {
            func();
        }
        deletion_queue.clear();
    }
};

struct FrameData {
    DeletionQueue deletion_queue;

    VkCommandPool command_pool;
    VkCommandBuffer main_command_buffer;

    VkSemaphore acquire_semaphore;
    VkFence render_fence;

    DescriptorAllocatorGrowable frame_descriptors;
};

struct ComputePushConstants {
    glm::vec4 data1;
    glm::vec4 data2;
    glm::vec4 data3;
    glm::vec4 data4;
};

struct ComputeEffect {
    const char* name;
    VkPipeline pipeline;
    VkPipelineLayout layout;
    ComputePushConstants data;
};

struct MousePosition {
    float x = 0.0f;
    float y = 0.0f;
};

struct Vertex {
    glm::vec3 position;
    float uv_x;
    glm::vec3 normal;
    float uv_y;
    glm::vec4 color;
};

struct GPUSceneData {
    glm::mat4 view;
    glm::mat4 proj;
    glm::mat4 view_proj;
    glm::vec4 ambient_color;
    glm::vec4 sunlight_direction; // w for sun power
    glm::vec4 sunlight_color;
};

struct GPUMeshBuffers {
    AllocatedBuffer index_buffer;
    AllocatedBuffer vertex_buffer;
    VkDeviceAddress vertex_buffer_address;
};

struct GPUDrawPushConstants {
    glm::mat4 world_matrix;
    VkDeviceAddress vertex_buffer;
};

struct EngineStats {
    float frame_time;
    int triangle_count;
    int draw_call_count;
    float scene_update_time;
    float mesh_draw_time;
};

constexpr unsigned int FRAME_OVERLAP = 2;

class Renderer {
public:
    Renderer();
    ~Renderer();
    void run();
    GPUMeshBuffers upload_mesh(std::span<uint32_t> indices, std::span<Vertex> vertices);
    AllocatedBuffer create_buffer(size_t alloc_size, VkBufferUsageFlags usage, VmaMemoryUsage memory_usage);

    vkb::Device m_vkb_device = {};
    VkDescriptorSetLayout m_gpu_scene_data_descriptor_layout;
    AllocatedImage m_draw_image = {};
    AllocatedImage m_depth_image = {};
    AllocatedImage m_error_checkerboard_image;
    AllocatedImage m_white_image;
    AllocatedImage m_black_image;
    AllocatedImage m_grey_image;
    VkSampler m_default_sampler_linear;
    VkSampler m_default_sampler_nearest;

    AllocatedImage create_image(VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped = false);
    AllocatedImage create_image(void* data, VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped = false);
    void destroy_buffer(const AllocatedBuffer &buffer);
    void destroy_image(const AllocatedImage& image);

private:
    bool m_is_initialized = false;
    int m_frame_index = 0;
    bool stop_rendering = false;
    bool resize_requested = false;
    VkExtent2D m_draw_extent = {};
    float m_render_scale = 1.0f;


    VkExtent2D m_window_extent = {1700, 900};
    DeletionQueue m_deletion_queue = {};
    VmaAllocator m_allocator = {};

    SDL_Window* m_window = nullptr;
    vkb::Instance m_vkb_instance = {};
    VkSurfaceKHR m_surface = VK_NULL_HANDLE;
    vkb::PhysicalDevice m_vkb_physical_device = {};

    vkb::Swapchain m_vkb_swapchain = {};
    VkExtent2D m_swapchain_extent = {};
    VkFormat m_swapchain_image_format = VK_FORMAT_UNDEFINED;
    std::vector<VkImage> m_swapchain_images;
    std::vector<VkImageView> m_swapchain_image_views;

    FrameData m_frames[FRAME_OVERLAP];
    FrameData& get_current_frame() {return m_frames[m_frame_index % FRAME_OVERLAP];}
    std::vector<VkSemaphore> m_submit_semaphores;

    VkQueue m_graphics_queue = VK_NULL_HANDLE;
    uint32_t m_graphics_queue_index = 0;

    VkExtent2D m_draw_image_extent = {};

    DescriptorAllocatorGrowable m_global_descriptor_allocator;
    VkDescriptorSet m_draw_image_descriptors;
    VkDescriptorSetLayout m_draw_image_descriptor_layout;

    VkPipeline m_compute_pipeline = VK_NULL_HANDLE;
    VkPipelineLayout m_compute_pipeline_layout = VK_NULL_HANDLE;

    VkFence m_imm_fence = VK_NULL_HANDLE;
    VkCommandBuffer m_imm_command_buffer = VK_NULL_HANDLE;
    VkCommandPool m_imm_command_pool = VK_NULL_HANDLE;

    std::vector<ComputeEffect> m_background_effects;
    int m_current_background_effect = 0;

    MousePosition m_mouse_position = {};

    VkDescriptorSetLayout m_single_image_descriptor_layout;
    MaterialInstance m_default_data;

    EngineStats m_stats;

    void init_sdl();
    void init_vulkan();
    void create_instance();
    void create_surface();
    void create_physical_device();
    void create_device();
    void create_swapchain(uint32_t width, uint32_t height);
    void resize_swapchain(uint32_t width, uint32_t height);
    void init_swapchain();
    void destroy_swapchain();
    void init_commands();
    void init_sync_objects();
    void init_vma();
    void init_descriptors();
    void draw_frame();
    void draw_background(VkCommandBuffer cmd_buffer);
    void init_pipelines();
    void init_background_pipelines();
    void init_imgui();
    void draw_imgui(VkCommandBuffer cmd, VkImageView target_image_view);
    void immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function);
    void init_default_data();
};

#endif //PORTFOLIO_RENDERER_H