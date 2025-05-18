// include/Model.hpp
#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <assimp/scene.h>
#include <string>
#include <vector>

struct Mesh {
    unsigned int VAO, VBO, EBO;
    unsigned int indexCount;
};

class Model {
public:
    Model();
    Model(const std::string& path);
    void Draw();
    void UpdateAnimation(float time); // Stub

private:
    void loadModel(const std::string& path);
    void processNode(aiNode* node, const aiScene* scene);
    Mesh processMesh(aiMesh* mesh, const aiScene* scene);

    std::vector<Mesh> meshes;
};