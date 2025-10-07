#include "Camera.h"

glm::mat4 Camera::get_view_matrix() {
    // to create a correct model view, we need to move the world in opposite
    // direction to the camera
    //  so we will create the camera model matrix and invert
    glm::mat4 camera_translation = glm::translate(glm::mat4(1.f), position);
    glm::mat4 camera_rotation = get_rotation_matrix();
    return glm::inverse(camera_translation * camera_rotation);
}

glm::mat4 Camera::get_rotation_matrix() {
    // fairly typical FPS style camera. we join the pitch and yaw rotations into
    // the final rotation matrix

    glm::quat pitch_rotation = glm::angleAxis(pitch, glm::vec3 { 1.f, 0.f, 0.f });
    glm::quat yaw_rotation = glm::angleAxis(yaw, glm::vec3 { 0.f, -1.f, 0.f });

    return glm::toMat4(yaw_rotation) * glm::toMat4(pitch_rotation);
}

void Camera::process_sdl_event(SDL_Event &event) {
    if (event.type == SDL_EVENT_KEY_DOWN) {
        if (event.key.key == SDLK_W) { velocity.z = -1; }
        if (event.key.key == SDLK_S) { velocity.z = 1; }
        if (event.key.key == SDLK_A) { velocity.x = -1; }
        if (event.key.key == SDLK_D) { velocity.x = 1; }
    }

    if (event.type == SDL_EVENT_KEY_UP) {
        if (event.key.key == SDLK_W) { velocity.z = 0; }
        if (event.key.key == SDLK_S) { velocity.z = 0; }
        if (event.key.key == SDLK_A) { velocity.x = 0; }
        if (event.key.key == SDLK_D) { velocity.x = 0; }
    }

    if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN) {
        if (event.button.button == SDL_BUTTON_RIGHT) {
            rmb_down = true;
        }
    }

    if (event.type == SDL_EVENT_MOUSE_BUTTON_UP) {
        if (event.button.button == SDL_BUTTON_RIGHT) {
            rmb_down = false;
        }
    }

    if (event.type == SDL_EVENT_MOUSE_MOTION && rmb_down) {
        yaw += (float)event.motion.xrel / 200.0f;
        pitch -= (float)event.motion.yrel / 200.0f;
    }
}

void Camera::update() {
    glm::mat4 camera_rotation = get_rotation_matrix();
    position += glm::vec3(camera_rotation * glm::vec4(velocity * 0.1f, 0.0f));
}
