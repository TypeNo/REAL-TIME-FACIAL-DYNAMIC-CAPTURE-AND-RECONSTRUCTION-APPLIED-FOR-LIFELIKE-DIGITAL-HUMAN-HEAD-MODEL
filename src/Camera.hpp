// include/Camera.hpp
#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class Camera {
public:
    Camera();
    mutable glm::vec3 Position; // allow it to be modified in const method
    mutable glm::vec3 Front;
    void ProcessMouseMovement(float dx, float dy);
    void ProcessMouseScroll(float dy);
    glm::mat4 GetViewMatrix() const;
    float GetDistance() const;

    void Reset();

private:
    float yaw, pitch;
    float distance;
    glm::vec3 target;
};