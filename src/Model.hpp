// include/Model.hpp
#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <assimp/scene.h>
#include <string>
#include <vector>
#include <map> // Add this if not already included

struct Mesh {
    unsigned int VAO, VBO, EBO;
    GLuint textureID = 0; // for diffuse texture
    unsigned int indexCount;
};

class Model {
public:
    Model();
    Model(const std::string& path);
    void Draw();
    void UpdateAnimation(float time); // Stub
    GLuint GetTextureID(int meshIndex = 0) const {
        if (meshIndex >= 0 && meshIndex < meshes.size()) {
            return meshes[meshIndex].textureID;
        }
        return 0; // Invalid index or no texture
    }     



private:
    void loadModel(const std::string& path);
    void processNode(aiNode* node, const aiScene* scene);
    Mesh processMesh(aiMesh* mesh, const aiScene* scene);
    std::map<std::string, GLuint> loadedTextures;

    std::vector<Mesh> meshes;
};