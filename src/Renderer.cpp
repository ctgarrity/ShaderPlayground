#include "Renderer.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <thread>
#include <array>

#include "SDL3/SDL_vulkan.h"
#include "Initializers.h"
#include "Utilities.h"

#include "imgui.h"
#include "backends/imgui_impl_vulkan.h"
#include "backends/imgui_impl_sdl3.h"

Renderer::Renderer() {
    init_sdl();
    init_vulkan();
    m_is_initialized = true;
    std::cout << "Vulkan initialized" << std::endl;
}

Renderer::~Renderer() {
    vkDeviceWaitIdle(m_vkb_device.device);

    // Swapchain
    for (const VkImageView& image_view : m_swapchain_image_views) {
        vkDestroyImageView(m_vkb_device.device, image_view, nullptr);
    }
    vkb::destroy_swapchain(m_vkb_swapchain);

    // Leftover
    for (auto& frame: m_frames) {
        frame.deletion_queue.flush();
    }
    m_deletion_queue.flush();
    std::cout << "Vulkan destroyed" << std::endl;
}

void Renderer::init_sdl() {
    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD)) {
        std::cerr << "Failed to initialize SDL: " << SDL_GetError() << std::endl;
    }

    const float main_scale = SDL_GetDisplayContentScale(SDL_GetPrimaryDisplay());
    constexpr SDL_WindowFlags window_flags = SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY; // | SDL_WINDOW_HIDDEN

    const SDL_DisplayMode* mode = SDL_GetCurrentDisplayMode(SDL_GetPrimaryDisplay());
    m_window_extent.width = (mode->w * 0.8f) / main_scale;
    m_window_extent.height = (mode->h * 0.8f) / main_scale;

    m_window = SDL_CreateWindow("Vulkan Portfolio", m_window_extent.width, m_window_extent.height, window_flags);
    if (!m_window)
    {
        std::cerr << "Failed to create window: " << SDL_GetError() << std::endl;
    }

    m_deletion_queue.push_function([&](){
        std::cout << "m_deletion_queue SDL_Quit" << std::endl;
        SDL_Quit();
    });
    m_deletion_queue.push_function([&](){
        std::cout << "m_deletion_queue SDL_DestroyWindow" << std::endl;
        SDL_DestroyWindow(m_window);
    });
}

void Renderer::init_vulkan() {
    create_instance();
    create_surface();
    create_physical_device();
    create_device();
    init_vma();
    init_swapchain();
    init_commands();
    init_sync_objects();
    init_descriptors();
    init_pipelines();
    init_imgui();
    init_default_data();
    m_is_initialized = true;
}

void Renderer::create_instance() {
    auto system_info_ret = vkb::SystemInfo::get_system_info();
    if (!system_info_ret) {
        std::cerr << "get_system_info() failed" << std::endl;
    }
    auto system_info = system_info_ret.value();

    uint32_t sdl_extension_count = 0;
    const char* const* sdl_instance_extensions = SDL_Vulkan_GetInstanceExtensions(&sdl_extension_count);
    std::vector<const char*> extensions(sdl_extension_count);
    for (int i = 0; i < sdl_extension_count; i++) {
        extensions[i] = sdl_instance_extensions[i];
    }
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    vkb::InstanceBuilder instance_builder;
    auto builder_return = instance_builder.set_app_name("Vulkan Portfolio")
        .set_app_version(VK_MAKE_API_VERSION(0, 1, 0, 0))
        .set_engine_name("Casual Distraction Games Engine")
        .set_engine_version(VK_MAKE_API_VERSION(0, 1, 0, 0))
        .require_api_version(VK_MAKE_API_VERSION(0, 1, 4, 0))
        .enable_extensions(extensions)
        .enable_validation_layers()
        .use_default_debug_messenger()
        .build();

    m_vkb_instance = builder_return.value();
    std::cout << "vkb instance created" << std::endl;
    m_deletion_queue.push_function([&](){
        std::cout << "m_deletion_queue destroy_instance" << std::endl;
        vkb::destroy_instance(m_vkb_instance);
    });
}

void Renderer::create_surface() {
    if (!SDL_Vulkan_CreateSurface(m_window, m_vkb_instance.instance, nullptr, &m_surface)) {
        std::cerr << "Failed to create surface: " << SDL_GetError() << std::endl;
    }
    m_deletion_queue.push_function([&](){
        std::cout << "m_deletion_queue SDL_Vulkan_DestroySurface" << std::endl;
        SDL_Vulkan_DestroySurface(m_vkb_instance.instance, m_surface, nullptr);
    });
    std::cout << "SDL Vulkan surface created" << std::endl;
}

void Renderer::create_physical_device() {
    VkPhysicalDeviceVulkan13Features features13 = {};
    features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    features13.dynamicRendering = true;
    features13.synchronization2 = true;

    VkPhysicalDeviceVulkan12Features features12{};
    features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    features12.bufferDeviceAddress = true;
    features12.descriptorIndexing = true;

    vkb::PhysicalDeviceSelector selector(m_vkb_instance);
    m_vkb_physical_device = selector.set_minimum_version(1, 4)
        .set_required_features_13(features13)
        .set_required_features_12(features12)
        .set_surface(m_surface)
        .select()
        .value();

    std::cout << "vkb physical device created" << std::endl;
}

void Renderer::create_device() {
    const vkb::DeviceBuilder device_builder(m_vkb_physical_device);
    m_vkb_device = device_builder.build().value();
    std::cout << "vkb device created" << std::endl;

    m_deletion_queue.push_function([&](){vkb::destroy_device(m_vkb_device);});

    m_graphics_queue = m_vkb_device.get_queue(vkb::QueueType::graphics).value();
    m_graphics_queue_index = m_vkb_device.get_queue_index(vkb::QueueType::graphics).value();
}

void Renderer::create_swapchain(const uint32_t width, const uint32_t height) {
    vkb::SwapchainBuilder swapchain_builder(m_vkb_physical_device.physical_device, m_vkb_device.device, m_surface);
    m_swapchain_image_format = VK_FORMAT_B8G8R8A8_UNORM;

    m_vkb_swapchain = swapchain_builder.set_desired_format(VkSurfaceFormatKHR{.format = m_swapchain_image_format, .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
        .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR) // VSync
        .set_desired_extent(width, height)
        .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
        .set_desired_min_image_count(3)
        .build()
        .value();

    m_swapchain_extent = m_vkb_swapchain.extent;
    m_swapchain_images = m_vkb_swapchain.get_images().value();
    m_swapchain_image_views = m_vkb_swapchain.get_image_views().value();
    m_submit_semaphores.resize(m_swapchain_images.size());
}

void Renderer::resize_swapchain(const uint32_t width, const uint32_t height) {
    vkDeviceWaitIdle(m_vkb_device.device);

    vkb::SwapchainBuilder swapchain_builder{ m_vkb_device };
    auto swap_ret = swapchain_builder.set_old_swapchain(m_vkb_swapchain)
        .set_desired_format(VkSurfaceFormatKHR{.format = m_swapchain_image_format, .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
        .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR) // VSync
        .set_desired_extent(width, height)
        .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
        .set_desired_min_image_count(3)
        .build();
    if (!swap_ret){
        m_vkb_swapchain.swapchain = VK_NULL_HANDLE;
    }

    for (const VkImageView& image_view : m_swapchain_image_views) {
        vkDestroyImageView(m_vkb_device.device, image_view, nullptr);
    }
    vkb::destroy_swapchain(m_vkb_swapchain);
    m_vkb_swapchain = swap_ret.value();
    m_swapchain_extent = m_vkb_swapchain.extent;
    m_swapchain_images = m_vkb_swapchain.get_images().value();
    m_swapchain_image_views = m_vkb_swapchain.get_image_views().value();
    resize_requested = false;
}

void Renderer::init_swapchain() {
    create_swapchain(m_window_extent.width, m_window_extent.height);
    std::cout << "Initial swapchain created" << std::endl;
    //Todo: Might need to change when these get destroyed when I resize the swapchain/window
    // m_deletion_queue.push_function([&](){vkb::destroy_swapchain(m_vkb_swapchain);});
    // for (const VkImageView& image_view : m_swapchain_image_views) {
    //     m_deletion_queue.push_function([&](){vkDestroyImageView(m_vkb_device.device, image_view, nullptr);});
    // }

    VkExtent3D draw_image_extent = {
        m_window_extent.width,
        m_window_extent.height,
        1
    };

    m_draw_image.image_format = VK_FORMAT_R16G16B16A16_SFLOAT;
    m_draw_image.image_extent = draw_image_extent;

    VkImageUsageFlags draw_image_usages = {};
    draw_image_usages |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    draw_image_usages |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    draw_image_usages |= VK_IMAGE_USAGE_STORAGE_BIT;
    draw_image_usages |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    VkImageCreateInfo render_img_info = init::image_create_info(m_draw_image.image_format, draw_image_usages, draw_image_extent);
    VmaAllocationCreateInfo render_img_alloc_info = {};
    render_img_alloc_info.usage = VMA_MEMORY_USAGE_GPU_ONLY; // Allocate it from gpu local memory
    render_img_alloc_info.requiredFlags = static_cast<VkMemoryPropertyFlags>(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    vmaCreateImage(m_allocator, &render_img_info, &render_img_alloc_info, &m_draw_image.image, &m_draw_image.allocation, nullptr);
    VkImageViewCreateInfo render_view_info = init::image_view_create_info(m_draw_image.image_format, m_draw_image.image, VK_IMAGE_ASPECT_COLOR_BIT);
    VK_CHECK(vkCreateImageView(m_vkb_device.device, &render_view_info, nullptr, &m_draw_image.image_view));

    m_deletion_queue.push_function([this]() {
        std::cout << "m_deletion_queue vmaDestroyImage(m_allocator, m_draw_image.image, m_draw_image.allocation);" << std::endl;
        vkDestroyImageView(m_vkb_device.device, m_draw_image.image_view, nullptr);
        vmaDestroyImage(m_allocator, m_draw_image.image, m_draw_image.allocation);
    });

    m_depth_image.image_format = VK_FORMAT_D32_SFLOAT;
    m_depth_image.image_extent = draw_image_extent;
    VkImageUsageFlags depth_image_usages = {};
    depth_image_usages |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

    VkImageCreateInfo depth_img_info = init::image_create_info(m_depth_image.image_format, depth_image_usages, draw_image_extent);
    vmaCreateImage(m_allocator, &depth_img_info, &render_img_alloc_info, &m_depth_image.image, &m_depth_image.allocation, nullptr);
    VkImageViewCreateInfo depth_view_info = init::image_view_create_info(m_depth_image.image_format, m_depth_image.image, VK_IMAGE_ASPECT_DEPTH_BIT);
    VK_CHECK(vkCreateImageView(m_vkb_device.device, &depth_view_info, nullptr, &m_depth_image.image_view));

    m_deletion_queue.push_function([this]() {
        std::cout << "m_deletion_queue vmaDestroyImage(m_allocator, m_depth_image.image, m_depth_image.allocation);" << std::endl;
        vkDestroyImageView(m_vkb_device.device, m_depth_image.image_view, nullptr);
        vmaDestroyImage(m_allocator, m_depth_image.image, m_depth_image.allocation);
});
}

void Renderer::destroy_swapchain() {
    //Todo: Change to match above! Also, I think this will need to be in the Frame deletion queue later?
    for (int i = 0; i < m_swapchain_image_views.size(); i++) {
        m_deletion_queue.push_function([&](){vkDestroyImageView(m_vkb_device.device, m_swapchain_image_views[i], nullptr);});
    }
    m_deletion_queue.push_function([&](){vkb::destroy_swapchain(m_vkb_swapchain);});
}

void Renderer::init_commands() {
    VkCommandPoolCreateInfo command_pool_info = init::command_pool_create_info(m_graphics_queue_index, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
    for (int i = 0; i < FRAME_OVERLAP; i++) {
        VK_CHECK(vkCreateCommandPool(m_vkb_device.device, &command_pool_info, nullptr, &m_frames[i].command_pool));

        VkCommandBufferAllocateInfo cmd_alloc_info = init::command_buffer_allocate_info(m_frames[i].command_pool, 1);
        VK_CHECK(vkAllocateCommandBuffers(m_vkb_device.device, &cmd_alloc_info, &m_frames[i].main_command_buffer));
    }

    for (const auto& frame : m_frames) {
        m_deletion_queue.push_function([&](){
            std::cout << "m_deletion_queue vkDestroyCommandPool" << std::endl;
            vkDestroyCommandPool(m_vkb_device.device, frame.command_pool, nullptr);
        });
    }
    std::cout << "FIF Command buffers allocated" << std::endl;

    VK_CHECK(vkCreateCommandPool(m_vkb_device.device, &command_pool_info, nullptr, &m_imm_command_pool));
    VkCommandBufferAllocateInfo cmd_alloc_info = init::command_buffer_allocate_info(m_imm_command_pool, 1);
    VK_CHECK(vkAllocateCommandBuffers(m_vkb_device.device, &cmd_alloc_info, &m_imm_command_buffer));
    std::cout << "Immediate command buffers allocated" << std::endl;

    m_deletion_queue.push_function([this]() {
        std::cout << "m_deletion_queue vkDestroyCommandPool" << std::endl;
        vkDestroyCommandPool(m_vkb_device.device, m_imm_command_pool, nullptr);
    });

}

void Renderer::init_sync_objects() {
    VkFenceCreateInfo fence_info = init::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);
    VkSemaphoreCreateInfo semaphore_info = init::semaphore_create_info();

    for (int i = 0; i < FRAME_OVERLAP; i++) {
        VK_CHECK(vkCreateFence(m_vkb_device.device, &fence_info, nullptr, &m_frames[i].render_fence));
        VK_CHECK(vkCreateSemaphore(m_vkb_device.device, &semaphore_info, nullptr, &m_frames[i].acquire_semaphore));
    }

    for (int i = 0; i < m_swapchain_images.size(); i++) {
        VK_CHECK(vkCreateSemaphore(m_vkb_device.device, &semaphore_info, nullptr, &m_submit_semaphores[i]));
    }

    VK_CHECK(vkCreateFence(m_vkb_device.device, &fence_info, nullptr, &m_imm_fence));
    std::cout << "Synchronization objects created" << std::endl;

    // This might need to get deleted on the fly when resizing the window/swapchain
    for (const auto& frame : m_frames) {
        m_deletion_queue.push_function([&](){
            std::cout << "m_deletion_queue vkDestroyFence" << std::endl;
            vkDestroyFence(m_vkb_device.device, frame.render_fence, nullptr);
        });
        m_deletion_queue.push_function([&]() {
            std::cout << "m_deletion_queue vkDestroySemaphore frame" << std::endl;
            vkDestroySemaphore(m_vkb_device.device, frame.acquire_semaphore, nullptr);
        });
    }

    for (const auto& semaphore : m_submit_semaphores) {
        m_deletion_queue.push_function([&]() {
            std::cout << "m_deletion_queue vkDestroySemaphore semaphore" << std::endl;
            vkDestroySemaphore(m_vkb_device.device, semaphore, nullptr);
        });
    }

    m_deletion_queue.push_function([this]() {
        std::cout << "m_deletion_queue vkDestroyFence" << std::endl;
        vkDestroyFence(m_vkb_device.device, m_imm_fence, nullptr);
    });
}

void Renderer::init_vma() {
    VmaAllocatorCreateInfo allocator_info = {};
    allocator_info.physicalDevice = m_vkb_physical_device.physical_device;
    allocator_info.device = m_vkb_device.device;
    allocator_info.instance = m_vkb_instance.instance;
    allocator_info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    vmaCreateAllocator(&allocator_info, &m_allocator);

    m_deletion_queue.push_function([&]() {
        std::cout << "m_deletion_queue vmaDestroyAllocator" << std::endl;
        vmaDestroyAllocator(m_allocator);
    });

    std::cout << "VMA allocator created" << std::endl;
}

void Renderer::init_descriptors() {
    std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> sizes =
    {
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4 },
    };

    m_global_descriptor_allocator.init(m_vkb_device.device, 10, sizes);
    {
        DescriptorLayoutBuilder builder = {};
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        m_draw_image_descriptor_layout = builder.build(m_vkb_device.device, VK_SHADER_STAGE_COMPUTE_BIT);
    }

    m_draw_image_descriptors = m_global_descriptor_allocator.allocate(m_vkb_device.device, m_draw_image_descriptor_layout);

    VkDescriptorImageInfo img_info = {};
    img_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    img_info.imageView = m_draw_image.image_view;

    VkWriteDescriptorSet draw_image_write = {};
    draw_image_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    draw_image_write.pNext = nullptr;
    draw_image_write.dstBinding = 0;
    draw_image_write.dstSet = m_draw_image_descriptors;
    draw_image_write.descriptorCount = 1;
    draw_image_write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    draw_image_write.pImageInfo = &img_info;

    vkUpdateDescriptorSets(m_vkb_device.device, 1, &draw_image_write, 0, nullptr);

    // m_deletion_queue.push_function([&]() {
    //     m_global_descriptor_allocator.destroy_pools(m_vkb_device.device);
    //     vkDestroyDescriptorSetLayout(m_vkb_device.device, m_draw_image_descriptor_layout, nullptr);
    // });

    for (int i = 0; i < FRAME_OVERLAP; i++) {
        // create a descriptor pool
        std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> frame_sizes = {
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3 },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4 },
        };

        m_frames[i].frame_descriptors = DescriptorAllocatorGrowable{};
        m_frames[i].frame_descriptors.init(m_vkb_device.device, 1000, frame_sizes);

        m_deletion_queue.push_function([&, i]() {
            m_frames[i].frame_descriptors.clear_pools(m_vkb_device.device);
            m_frames[i].frame_descriptors.destroy_pools(m_vkb_device.device);
        });
    }

    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        m_gpu_scene_data_descriptor_layout = builder.build(m_vkb_device.device, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
    }

    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        m_single_image_descriptor_layout = builder.build(m_vkb_device.device, VK_SHADER_STAGE_FRAGMENT_BIT);
    }

    m_deletion_queue.push_function([&]() {
        m_global_descriptor_allocator.destroy_pools(m_vkb_device.device);
        vkDestroyDescriptorSetLayout(m_vkb_device.device, m_draw_image_descriptor_layout, nullptr);
        vkDestroyDescriptorSetLayout(m_vkb_device.device, m_gpu_scene_data_descriptor_layout, nullptr);
        vkDestroyDescriptorSetLayout(m_vkb_device.device, m_single_image_descriptor_layout, nullptr);
    });

    std::cout << "Descriptors initialized" << std::endl;
}

void Renderer::draw_frame() {
    VK_CHECK(vkWaitForFences(m_vkb_device.device, 1, &get_current_frame().render_fence, true, 1'000'000'000));
    VK_CHECK(vkResetFences(m_vkb_device.device, 1, &get_current_frame().render_fence));

    get_current_frame().deletion_queue.flush();
    get_current_frame().frame_descriptors.clear_pools(m_vkb_device.device);

    uint32_t swapchain_image_index;
    // VK_CHECK(vkAcquireNextImageKHR(m_vkb_device.device, m_vkb_swapchain.swapchain, 1'000'000'000, get_current_frame().acquire_semaphore, nullptr, &swapchain_image_index));
    VkResult result = vkAcquireNextImageKHR(m_vkb_device.device, m_vkb_swapchain.swapchain, 1'000'000'000, get_current_frame().acquire_semaphore, nullptr, &swapchain_image_index);
    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        resize_requested = true;
        return;
    }

    VkCommandBuffer cmd_buffer = get_current_frame().main_command_buffer;
    VK_CHECK(vkResetCommandBuffer(cmd_buffer, 0));

    m_draw_extent.height = std::min(m_swapchain_extent.height, m_draw_image.image_extent.height) * m_render_scale;
    m_draw_extent.width= std::min(m_swapchain_extent.width, m_draw_image.image_extent.width) * m_render_scale;

    // Todo: Replace where these are used?
    // m_draw_image_extent.width = m_draw_image.image_extent.width;
    // m_draw_image_extent.height = m_draw_image.image_extent.height;

    // For compute
    VkCommandBufferBeginInfo begin_info = init::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    VK_CHECK(vkBeginCommandBuffer(cmd_buffer, &begin_info));
    util::transition_image(cmd_buffer, m_draw_image.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    draw_background(cmd_buffer);

    // For imgui
    util::transition_image(cmd_buffer, m_draw_image.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    util::transition_image(cmd_buffer, m_swapchain_images[swapchain_image_index], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    util::copy_image_to_image(cmd_buffer, m_draw_image.image, m_swapchain_images[swapchain_image_index], m_draw_extent, m_swapchain_extent);
    util::transition_image(cmd_buffer, m_swapchain_images[swapchain_image_index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    draw_imgui(cmd_buffer,  m_swapchain_image_views[swapchain_image_index]);
    util::transition_image(cmd_buffer, m_swapchain_images[swapchain_image_index], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    VK_CHECK(vkEndCommandBuffer(cmd_buffer));

    VkCommandBufferSubmitInfo cmd_buffer_info = init::command_buffer_submit_info(cmd_buffer);
    VkSemaphoreSubmitInfo wait_info = init::semaphore_submit_info(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR, get_current_frame().acquire_semaphore);
    VkSemaphoreSubmitInfo signal_info = init::semaphore_submit_info(VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, m_submit_semaphores[swapchain_image_index]);
    VkSubmitInfo2 submit = init::submit_info(&cmd_buffer_info, &signal_info, &wait_info);
    VK_CHECK(vkQueueSubmit2(m_graphics_queue, 1, &submit, get_current_frame().render_fence));

    VkPresentInfoKHR present_info = {};
    present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present_info.pNext = nullptr;
    present_info.pSwapchains = &m_vkb_swapchain.swapchain;
    present_info.swapchainCount = 1;
    present_info.pWaitSemaphores = &m_submit_semaphores[swapchain_image_index];
    present_info.waitSemaphoreCount = 1;
    present_info.pImageIndices = &swapchain_image_index;

    // VK_CHECK(vkQueuePresentKHR(m_graphics_queue, &present_info));
    result = vkQueuePresentKHR(m_graphics_queue, &present_info);
    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        resize_requested = true;
    }
    m_frame_index++;
}

void Renderer::draw_background(VkCommandBuffer cmd_buffer) {
    ComputeEffect& compute_effect = m_background_effects[m_current_background_effect];
    compute_effect.data.data3.x = std::floor(m_mouse_position.x / 16.0f);
    compute_effect.data.data3.y = std::floor(m_mouse_position.y / 16.0f);

    vkCmdBindPipeline(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute_effect.pipeline);
    vkCmdBindDescriptorSets(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_compute_pipeline_layout, 0, 1, &m_draw_image_descriptors, 0, nullptr);
    vkCmdPushConstants(cmd_buffer, m_compute_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ComputePushConstants), &compute_effect.data);

    vkCmdDispatch(cmd_buffer, std::ceil(m_draw_extent.width / 16.0), std::ceil(m_draw_extent.height / 16.0), 1);
}

void Renderer::init_pipelines() {
    init_background_pipelines(); // Compute
}

void Renderer::init_background_pipelines() {
    VkPipelineLayoutCreateInfo compute_layout_info = {};
    compute_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    compute_layout_info.pNext = nullptr;
    compute_layout_info.pSetLayouts = &m_draw_image_descriptor_layout;
    compute_layout_info.setLayoutCount = 1;

    VkPushConstantRange push_constant = {};
    push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_constant.offset = 0;
    push_constant.size = sizeof(ComputePushConstants);
    compute_layout_info.pPushConstantRanges = &push_constant;
    compute_layout_info.pushConstantRangeCount = 1;

    VK_CHECK(vkCreatePipelineLayout(m_vkb_device.device, &compute_layout_info, nullptr, &m_compute_pipeline_layout));

    VkShaderModule compute_gradient_shader = {};
    if (!util::load_shader_module("../src/shaders/gradient_color.spv", m_vkb_device.device, &compute_gradient_shader)) {
        std::cerr << "Failed to load gradient shader" << std::endl;
    }

    VkShaderModule compute_sky_shader = {};
    if (!util::load_shader_module("../src/shaders/sky.spv", m_vkb_device.device, &compute_sky_shader)) {
        std::cerr << "Failed to load sky shader" << std::endl;
    }

    VkShaderModule compute_grid_shader = {};
    if (!util::load_shader_module("../src/shaders/grid.spv", m_vkb_device.device, &compute_grid_shader)) {
        std::cerr << "Failed to load grid shader" << std::endl;
    }

    VkPipelineShaderStageCreateInfo stage_info = {};
    stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage_info.pNext = nullptr;
    stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_info.module = compute_gradient_shader;
    stage_info.pName = "main";

    VkComputePipelineCreateInfo compute_pipeline_create_info = {};
    compute_pipeline_create_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    compute_pipeline_create_info.pNext = nullptr;
    compute_pipeline_create_info.layout = m_compute_pipeline_layout;
    compute_pipeline_create_info.stage = stage_info;

    ComputeEffect gradient = {};
    gradient.layout = m_compute_pipeline_layout;
    gradient.name = "gradient";
    gradient.data = {};
    gradient.data.data1 = glm::vec4(1, 0, 0, 1);
    gradient.data.data2 = glm::vec4(0, 0, 1, 1);
    VK_CHECK(vkCreateComputePipelines(m_vkb_device.device, VK_NULL_HANDLE, 1, &compute_pipeline_create_info, nullptr, &gradient.pipeline));

    compute_pipeline_create_info.stage.module = compute_sky_shader;
    ComputeEffect sky = {};
    sky.layout = m_compute_pipeline_layout;
    sky.name = "sky";
    sky.data = {};
    sky.data.data1 = glm::vec4(0.1, 0.2, 0.4 ,0.97);
    VK_CHECK(vkCreateComputePipelines(m_vkb_device.device, VK_NULL_HANDLE, 1, &compute_pipeline_create_info, nullptr, &sky.pipeline));

    compute_pipeline_create_info.stage.module = compute_grid_shader;
    ComputeEffect grid = {};
    grid.layout = m_compute_pipeline_layout;
    grid.name = "grid";
    grid.data = {};
    grid.data.data1 = glm::vec4(1.0, 1.0, 1.0 ,1.0);
    grid.data.data2 = glm::vec4(0.0, 0.0, 0.0 ,1.0);
    grid.data.data3 = glm::vec4(0.0, 0.0, 0.0 ,1.0);
    VK_CHECK(vkCreateComputePipelines(m_vkb_device.device, VK_NULL_HANDLE, 1, &compute_pipeline_create_info, nullptr, &grid.pipeline));

    m_background_effects.push_back(gradient);
    m_background_effects.push_back(sky);
    m_background_effects.push_back(grid);

    vkDestroyShaderModule(m_vkb_device.device, compute_gradient_shader, nullptr);
    vkDestroyShaderModule(m_vkb_device.device, compute_sky_shader, nullptr);
    vkDestroyShaderModule(m_vkb_device.device, compute_grid_shader, nullptr);

    m_deletion_queue.push_function([this, gradient, sky, grid]() {
        vkDestroyPipelineLayout(m_vkb_device.device, m_compute_pipeline_layout, nullptr);
        vkDestroyPipeline(m_vkb_device.device, gradient.pipeline, nullptr);
        vkDestroyPipeline(m_vkb_device.device, sky.pipeline, nullptr);
        vkDestroyPipeline(m_vkb_device.device, grid.pipeline, nullptr);
    });

    std::cout << "Background pipelines initialized" << std::endl;
}

void Renderer::init_imgui() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    ImGui_ImplSDL3_InitForVulkan(m_window);
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = m_vkb_instance.instance;
    init_info.PhysicalDevice = m_vkb_physical_device.physical_device;
    init_info.Device = m_vkb_device.device;
    init_info.QueueFamily = m_graphics_queue_index;
    init_info.Queue = m_graphics_queue;
    //init_info.PipelineCache = YOUR_PIPELINE_CACHE; // optional
    //init_info.DescriptorPool = YOUR_DESCRIPTOR_POOL; // see below Todo: Check if the DescriptorPoolSize is correct
    init_info.DescriptorPoolSize = IMGUI_IMPL_VULKAN_MINIMUM_IMAGE_SAMPLER_POOL_SIZE; // (Optional) Set to create internal descriptor pool instead of using DescriptorPool
    init_info.PipelineInfoMain.Subpass = 0;
    init_info.MinImageCount = 2;
    init_info.ImageCount = 2;
    init_info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    init_info.UseDynamicRendering = true;
    //init_info.Allocator = YOUR_ALLOCATOR; // optional
    //init_info.CheckVkResultFn = check_vk_result; // optional

    VkPipelineRenderingCreateInfo pipeline_rendering_create_info = {};
    pipeline_rendering_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    pipeline_rendering_create_info.pNext = nullptr;
    pipeline_rendering_create_info.colorAttachmentCount = 1;
    pipeline_rendering_create_info.pColorAttachmentFormats = &m_swapchain_image_format;

    init_info.PipelineInfoMain.PipelineRenderingCreateInfo = pipeline_rendering_create_info;

    ImGui_ImplVulkan_Init(&init_info);
    // (this gets a bit more complicated, see example app for full reference)
    //ImGui_ImplVulkan_CreateFontsTexture(get_current_frame().main_command_buffer);
    // (your code submit a queue)
    //ImGui_ImplVulkan_DestroyFontUploadObjects();
    m_deletion_queue.push_function([&]() {
        std::cout << "m_deletion_queue ImGui_ImplVulkan_Shutdown" << std::endl;
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplSDL3_Shutdown();
        ImGui::DestroyContext();
    });

    std::cout << "imgui initialized" << std::endl;
}

void Renderer::draw_imgui(VkCommandBuffer cmd, VkImageView target_image_view) {
    VkRenderingAttachmentInfo color_attachment = init::color_attachment_info(target_image_view, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkRenderingInfo render_info = init::rendering_info(m_swapchain_extent, &color_attachment, nullptr);

    vkCmdBeginRendering(cmd, &render_info);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
    vkCmdEndRendering(cmd);
}

void Renderer::immediate_submit(std::function<void(VkCommandBuffer cmd)> &&function) {
    //Todo: Switch the queue to use another queue rather than graphics
    VK_CHECK(vkResetFences(m_vkb_device.device, 1, &m_imm_fence));
    VK_CHECK(vkResetCommandBuffer(m_imm_command_buffer, 0));

    VkCommandBuffer imm_cmd = m_imm_command_buffer;
    VkCommandBufferBeginInfo cmdBeginInfo = init::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    VK_CHECK(vkBeginCommandBuffer(imm_cmd, &cmdBeginInfo));
    function(imm_cmd);
    VK_CHECK(vkEndCommandBuffer(imm_cmd));

    VkCommandBufferSubmitInfo cmd_info = init::command_buffer_submit_info(imm_cmd);
    VkSubmitInfo2 submit_info = init::submit_info(&cmd_info, nullptr, nullptr);
    VK_CHECK(vkQueueSubmit2(m_graphics_queue, 1, &submit_info, m_imm_fence));
    VK_CHECK(vkWaitForFences(m_vkb_device.device, 1, &m_imm_fence, true, 9999999999));
}

AllocatedBuffer Renderer::create_buffer(size_t alloc_size, VkBufferUsageFlags usage, VmaMemoryUsage memory_usage) {
    VkBufferCreateInfo buffer_info = {};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.pNext = nullptr;
    buffer_info.size = alloc_size;
    buffer_info.usage = usage;

    VmaAllocationCreateInfo vma_alloc_info = {};
    vma_alloc_info.usage = memory_usage;
    vma_alloc_info.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

    AllocatedBuffer new_buffer = {};
    VK_CHECK(vmaCreateBuffer(m_allocator, &buffer_info, &vma_alloc_info, &new_buffer.buffer, &new_buffer.allocation, &new_buffer.info));
    return new_buffer;
}

void Renderer::destroy_buffer(const AllocatedBuffer& buffer) {
    vmaDestroyBuffer(m_allocator, buffer.buffer, buffer.allocation);
}

GPUMeshBuffers Renderer::upload_mesh(std::span<uint32_t> indices, std::span<Vertex> vertices) {
    // Todo: Put this on a background thread?
    const size_t vertex_buffer_size = vertices.size() * sizeof(Vertex);
    const size_t index_buffer_size = indices.size() * sizeof(uint32_t);

    GPUMeshBuffers new_surface;
    new_surface.vertex_buffer = create_buffer(vertex_buffer_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VMA_MEMORY_USAGE_AUTO);

    VkBufferDeviceAddressInfo device_adress_info = { .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = new_surface.vertex_buffer.buffer };
    new_surface.vertex_buffer_address = vkGetBufferDeviceAddress(m_vkb_device.device, &device_adress_info);
    new_surface.index_buffer = create_buffer(index_buffer_size, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO);

    AllocatedBuffer staging = create_buffer(vertex_buffer_size + index_buffer_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO);
    void* data = staging.info.pMappedData; // had to change this from vkguides staging.alloction.GetMappedData()
    memcpy(data, vertices.data(), vertex_buffer_size);
    memcpy((char*)data + vertex_buffer_size, indices.data(), index_buffer_size);

    immediate_submit([vertex_buffer_size, index_buffer_size, staging, new_surface](VkCommandBuffer cmd) {
        VkBufferCopy vertex_copy = {};
        vertex_copy.dstOffset = 0;
        vertex_copy.srcOffset = 0;
        vertex_copy.size = vertex_buffer_size;
        vkCmdCopyBuffer(cmd, staging.buffer, new_surface.vertex_buffer.buffer, 1, &vertex_copy);

        VkBufferCopy index_copy = {};
        index_copy.dstOffset = 0;
        index_copy.srcOffset = vertex_buffer_size;
        index_copy.size = index_buffer_size;
        vkCmdCopyBuffer(cmd, staging.buffer, new_surface.index_buffer.buffer, 1, &index_copy);
    });

    destroy_buffer(staging);
    return new_surface;
}

void Renderer::init_default_data() {
    std::array<Vertex, 4> rect_vertices;
    rect_vertices[0].position = {0.5,-0.5, 0};
    rect_vertices[1].position = {0.5,0.5, 0};
    rect_vertices[2].position = {-0.5,-0.5, 0};
    rect_vertices[3].position = {-0.5,0.5, 0};

    rect_vertices[0].color = {0,0, 0,1};
    rect_vertices[1].color = { 0.5,0.5,0.5 ,1};
    rect_vertices[2].color = { 1,0, 0,1 };
    rect_vertices[3].color = { 0,1, 0,1 };

    std::array<uint32_t, 6> rect_indices;
    rect_indices[0] = 0;
    rect_indices[1] = 1;
    rect_indices[2] = 2;

    rect_indices[3] = 2;
    rect_indices[4] = 1;
    rect_indices[5] = 3;

    //m_test_meshes = load_gltf_meshes(this, "../assets/basicmesh.glb").value();

    uint32_t white = glm::packUnorm4x8(glm::vec4(1, 1, 1, 1));
    m_white_image = create_image((void*)&white, VkExtent3D{ 1, 1, 1 }, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);

    uint32_t grey = glm::packUnorm4x8(glm::vec4(0.66f, 0.66f, 0.66f, 1));
    m_grey_image = create_image((void*)&grey, VkExtent3D{ 1, 1, 1 }, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);

    uint32_t black = glm::packUnorm4x8(glm::vec4(0, 0, 0, 0));
    m_black_image = create_image((void*)&black, VkExtent3D{ 1, 1, 1 }, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);

    uint32_t magenta = glm::packUnorm4x8(glm::vec4(1, 0, 1, 1));
    std::array<uint32_t, 16 *16 > pixels;
    for (int x = 0; x < 16; x++) {
        for (int y = 0; y < 16; y++) {
            pixels[y*16 + x] = ((x % 2) ^ (y % 2)) ? magenta : black;
        }
    }
    m_error_checkerboard_image = create_image(pixels.data(), VkExtent3D{16, 16, 1}, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);

    VkSamplerCreateInfo sample = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sample.magFilter = VK_FILTER_NEAREST;
    sample.minFilter = VK_FILTER_NEAREST;
    vkCreateSampler(m_vkb_device.device, &sample, nullptr, &m_default_sampler_nearest);

    sample.magFilter = VK_FILTER_LINEAR;
    sample.minFilter = VK_FILTER_LINEAR;
    vkCreateSampler(m_vkb_device.device, &sample, nullptr, &m_default_sampler_linear);

    m_deletion_queue.push_function([&](){
        std::cout << "m_deletion_queue destroy_images" << std::endl;
        vkDestroySampler(m_vkb_device.device, m_default_sampler_nearest,nullptr);
        vkDestroySampler(m_vkb_device.device, m_default_sampler_linear,nullptr);

        destroy_image(m_white_image);
        destroy_image(m_grey_image);
        destroy_image(m_black_image);
        destroy_image(m_error_checkerboard_image);
    });
}

AllocatedImage Renderer::create_image(VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped) {
    AllocatedImage new_image;
    new_image.image_format = format;
    new_image.image_extent = size;

    VkImageCreateInfo img_info = init::image_create_info(format, usage, size);
    if (mipmapped) {
        img_info.mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(size.width, size.height)))) + 1;
    }

    VmaAllocationCreateInfo alloc_info = {};
    alloc_info.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    alloc_info.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vmaCreateImage(m_allocator, &img_info, &alloc_info, &new_image.image, &new_image.allocation, nullptr));

    VkImageAspectFlags aspectFlag = VK_IMAGE_ASPECT_COLOR_BIT;
    if (format == VK_FORMAT_D32_SFLOAT) {
        aspectFlag = VK_IMAGE_ASPECT_DEPTH_BIT;
    }

    VkImageViewCreateInfo view_info = init::image_view_create_info(format, new_image.image, aspectFlag);
    view_info.subresourceRange.levelCount = img_info.mipLevels;
    VK_CHECK(vkCreateImageView(m_vkb_device.device, &view_info, nullptr, &new_image.image_view));

    return new_image;
}

AllocatedImage Renderer::create_image(void *data, VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped) {
    size_t data_size = size.depth * size.width * size.height * 4;
    AllocatedBuffer upload_buffer = create_buffer(data_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

    memcpy(upload_buffer.info.pMappedData, data, data_size);

    AllocatedImage new_image = create_image(size, format, usage | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, mipmapped);

    immediate_submit([&](VkCommandBuffer cmd) {
        util::transition_image(cmd, new_image.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        VkBufferImageCopy copy_region = {};
        copy_region.bufferOffset = 0;
        copy_region.bufferRowLength = 0;
        copy_region.bufferImageHeight = 0;
        copy_region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy_region.imageSubresource.mipLevel = 0;
        copy_region.imageSubresource.baseArrayLayer = 0;
        copy_region.imageSubresource.layerCount = 1;
        copy_region.imageExtent = size;

        vkCmdCopyBufferToImage(cmd, upload_buffer.buffer, new_image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
            &copy_region);

        util::transition_image(cmd, new_image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        });

    destroy_buffer(upload_buffer);
    return new_image;
}

void Renderer::destroy_image(const AllocatedImage &image) {
    vkDestroyImageView(m_vkb_device.device, image.image_view, nullptr);
    vmaDestroyImage(m_allocator, image.image, image.allocation);
}

void Renderer::run() {
    SDL_Event e;
    bool quit = false;
    bool show_demo_window = true;
    while (!quit) {

        auto start = std::chrono::system_clock::now();
        // Todo: Normalize the mouse coords
        while (SDL_PollEvent(&e) != 0) {
            ImGui_ImplSDL3_ProcessEvent(&e);
            if (e.type == SDL_EVENT_QUIT)
                quit = true;

            if (e.window.type == SDL_EVENT_WINDOW_MINIMIZED) {
                stop_rendering = true;
            }

            if (e.window.type == SDL_EVENT_WINDOW_RESTORED) {
                stop_rendering = false;
            }

            if (e.window.type == SDL_EVENT_WINDOW_RESIZED) {
                resize_requested = true;
            }

            if (e.type == SDL_EVENT_MOUSE_MOTION) {
                SDL_GetMouseState(&m_mouse_position.x, &m_mouse_position.y);
                //std::cout << "X Mouse:" << get_current_frame().mouse_position.x << '\n' << "Y Mouse:" << get_current_frame().mouse_position.y << std::endl;
            }

            // Todo: Game of Life?
            // if (e.type == SDL_EVENT_MOUSE_BUTTON_DOWN) {
            //     std::cout << "Mouse clicked" << std::endl;
            // }
        }

        if (stop_rendering) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        if (resize_requested) {
            int width, height;
            SDL_GetWindowSize(m_window, &width, &height);
            m_window_extent.width = width;
            m_window_extent.height = height;
            resize_swapchain(m_window_extent.width, m_window_extent.height);
        }

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();
        if (ImGui::Begin("background")) {
            ComputeEffect& selected = m_background_effects[m_current_background_effect];
            ImGui::SliderFloat("Render Scale", &m_render_scale, 0.3f, 1.f);
            ImGui::Text("Selected effect: %s", selected.name);
            ImGui::SliderInt("Effect Index", &m_current_background_effect,0, m_background_effects.size() - 1);
            ImGui::InputFloat4("data1",(float*)& selected.data.data1);
            ImGui::InputFloat4("data2",(float*)& selected.data.data2);
            ImGui::InputFloat4("data3",(float*)& selected.data.data3);
            ImGui::InputFloat4("data4",(float*)& selected.data.data4);
        }
        ImGui::End();

        ImGui::Begin("Stats");
        ImGui::Text("frametime %f ms", m_stats.frame_time);
        ImGui::Text("draw time %f ms", m_stats.mesh_draw_time);
        ImGui::Text("update time %f ms", m_stats.scene_update_time);
        ImGui::Text("triangles %i", m_stats.triangle_count);
        ImGui::Text("draws %i", m_stats.draw_call_count);
        ImGui::End();
        //ImGui::ShowDemoWindow(&show_demo_window);
        //Todo: Move the imgui functions?
        ImGui::Render();
        draw_frame();

        auto end = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        m_stats.frame_time = elapsed.count() / 1000.0f;
    }
}


