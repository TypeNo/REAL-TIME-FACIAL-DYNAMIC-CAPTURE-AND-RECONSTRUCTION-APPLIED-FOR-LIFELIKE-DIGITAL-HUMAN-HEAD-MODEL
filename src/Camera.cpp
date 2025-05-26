// src/Camera.cpp
#include "Camera.hpp"
#include <algorithm>
#include <cmath>

Camera::Camera()
    : yaw(0.0f), pitch(0.0f), distance(1.0f), target(0.0f, 0.0f, 0.0f), Position(glm::vec3(0.0f, 0.0f, 1.0f)) {}

void Camera::ProcessMouseMovement(float dx, float dy) {
    yaw += dx * 0.25f;
    pitch -= dy * 0.25f;

    pitch = std::clamp(pitch, -89.0f, 89.0f);
}

void Camera::ProcessMouseScroll(float dy) {
    distance -= dy * 0.1f;
    distance = std::clamp(distance, 0.5f, 10.0f);
}

glm::mat4 Camera::GetViewMatrix() const {
    float radYaw = glm::radians(yaw);
    float radPitch = glm::radians(pitch);

    glm::vec3 direction = {
        cos(radPitch) * sin(radYaw),
        sin(radPitch),
        cos(radPitch) * cos(radYaw)
    };

    Front = glm::normalize(direction); // <- Update Front here

    Position = target - direction * distance;
    return glm::lookAt(Position, target, glm::vec3(0, 1, 0));
}

float Camera::GetDistance() const {
    return distance;
}

void Camera::Reset() {
    yaw = 0.0f;
    pitch = 0.0f;
    distance = 3.0f;
}
