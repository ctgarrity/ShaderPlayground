#ifndef PORTFOLIO_TYPES_H
#define PORTFOLIO_TYPES_H

#include "vulkan/vk_enum_string_helper.h"
#include <iostream>
#include <cassert>
#include <memory>
#include <vector>

#include "glm/glm.hpp"
#include <glm/gtx/transform.hpp>
#include "glm/gtc/quaternion.hpp"
#include "glm/gtx/quaternion.hpp"
#include "external/vk_mem_alloc.h"

#define VK_CHECK(func) { \
    const VkResult result = func; \
    if (result != VK_SUCCESS) { \
        std::cerr << "Error calling function " << #func \
            << "at " << __FILE__ << ":" \
            << __LINE__ << ". Result is " \
            << string_VkResult(result) \
            << std::endl; \
        assert(false); \
    } \
}

enum class MaterialPass : uint8_t {
    MainColor,
    Transparent,
    Other
};

struct MaterialPipeline {
    VkPipeline pipeline;
    VkPipelineLayout layout;
};

struct MaterialInstance {
    MaterialPipeline* pipeline;
    VkDescriptorSet materialSet;
    MaterialPass passType;
};

struct DrawContext;

// base class for a renderable dynamic object
class IRenderable {

    virtual void Draw(const glm::mat4& topMatrix, DrawContext& ctx) = 0;
};

// implementation of a drawable scene node.
// the scene node can hold children and will also keep a transform to propagate to them
struct Node : public IRenderable {

    // parent pointer must be a weak pointer to avoid circular dependencies
    std::weak_ptr<Node> parent;
    std::vector<std::shared_ptr<Node>> children;

    glm::mat4 local_transform;
    glm::mat4 world_transform;

    void refreshTransform(const glm::mat4& parent_matrix)
    {
        world_transform = parent_matrix * local_transform;
        for (const auto& child : children) {
            child->refreshTransform(world_transform);
        }
    }

    void Draw(const glm::mat4& top_matrix, DrawContext& draw_context) override {
        // draw children
        for (const auto& child : children) {
            child->Draw(top_matrix, draw_context);
        }
    }
};

struct AllocatedImage {
    VkImage image;
    VkImageView image_view;
    VmaAllocation allocation;
    VkExtent3D image_extent;
    VkFormat image_format;
};

struct AllocatedBuffer {
    VkBuffer buffer;
    VmaAllocation allocation;
    VmaAllocationInfo info;
};

#endif //PORTFOLIO_TYPES_H