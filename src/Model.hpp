// include/Model.hpp
#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <assimp/scene.h>
#include <string>
#include <vector>
#include <map> // Add this if not already included
#include "Shader.hpp"


struct MorphTarget {
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
};

struct MorphAnimKey {
    double time;
    std::vector<float> weights;
};


struct Mesh {
    unsigned int VAO, VBO, EBO;
    GLuint textureID = 0; // for diffuse texture
    unsigned int indexCount;

    // Morph targets (blend shapes)
    std::vector<MorphTarget> morphTargets;
    std::map<int, std::vector<MorphAnimKey>> morphAnimations; // mesh index → keyframes
    GLuint morphSSBO = 0;
    GLuint weightsSSBO = 0;
    GLuint debugSSBO = 0;
    int numMorphTargets = 0;
    int numVertices = 0;
};

class Model {
public:
    Model();
    Model(const std::string& path);
    Model(const std::string& path, Shader& shader);
    void Draw();
    void UpdateAnimation(float time); // Stub
    GLuint GetTextureID(int meshIndex = 0) const {
        if (meshIndex >= 0 && meshIndex < meshes.size()) {
            return meshes[meshIndex].textureID;
        }
        return 0; // Invalid index or no texture
    }
    void UpdateAnimationWithFrame(int frameIndex, float alphaBetweenFrames);     

private:
    void loadModel(const std::string& path);
    void processNode(aiNode* node, const aiScene* scene);
    Mesh processMesh(aiMesh* mesh, const aiScene* scene);
    std::map<std::string, GLuint> loadedTextures;
    std::vector<Mesh> meshes;
    Shader *shader = nullptr;
    std::vector<float> morphWeights; // store this somewhere in your class
    bool loaded;
    
};