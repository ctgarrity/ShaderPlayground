#ifndef PORTFOLIO_CAMERA_H
#define PORTFOLIO_CAMERA_H

#include "Types.h"
#include "SDL3/SDL_events.h"

class Camera {
public:
    glm::vec3 velocity;
    glm::vec3 position;
    float pitch { 0.0f };
    float yaw { 0.0f };
    bool rmb_down { false };

    glm::mat4 get_view_matrix();
    glm::mat4 get_rotation_matrix();
    void process_sdl_event(SDL_Event& event);
    void update();
};

#endif //PORTFOLIO_CAMERA_H