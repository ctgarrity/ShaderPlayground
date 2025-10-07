#ifndef PORTFOLIO_LOADER_H
#define PORTFOLIO_LOADER_H

#include "Types.h"
#include <unordered_map>
#include <filesystem>
#include <vector>
#include <iostream>
#include <optional>
#include "fastgltf/glm_element_traits.hpp"
#include "fastgltf/tools.hpp"
#include "fastgltf/core.hpp"

#include "Descriptors.h"
#include "Initializers.h"

class Renderer;
struct GPUMeshBuffers;

//#include "glm/gtx/quaternion.hpp"

struct GLTFMaterial {
    MaterialInstance data;
};

struct Bounds {
    glm::vec3 origin;
    float sphere_radius;
    glm::vec3 extents;
};

struct GeoSurface {
    uint32_t start_index;
    uint32_t count;
    Bounds bounds;
    std::shared_ptr<GLTFMaterial> material;
};

struct MeshAsset {
    std::string name;
    std::vector<GeoSurface> surfaces;
    std::shared_ptr<GPUMeshBuffers> mesh_buffers;
};

struct LoadedGLTF : public IRenderable {
    std::unordered_map<std::string, std::shared_ptr<MeshAsset>> meshes;
    std::unordered_map<std::string, std::shared_ptr<Node>> nodes;
    std::unordered_map<std::string, AllocatedImage> images;
    std::unordered_map<std::string, std::shared_ptr<GLTFMaterial>> materials;
    std::vector<std::shared_ptr<Node>> top_nodes;
    std::vector<VkSampler> samplers;
    DescriptorAllocatorGrowable descriptor_pool;
    AllocatedBuffer material_data_buffer;
    Renderer* creator;

    ~LoadedGLTF() { clear_all(); };
    virtual void Draw(const glm::mat4& top_matrix, DrawContext& draw_context);

private:
    void clear_all();
};

std::optional<std::shared_ptr<LoadedGLTF>> load_gltf(Renderer* renderer, const std::string_view file_path);
std::optional<AllocatedImage> load_image(Renderer* renderer, fastgltf::Asset& asset, fastgltf::Image& image);

#endif //PORTFOLIO_LOADER_H