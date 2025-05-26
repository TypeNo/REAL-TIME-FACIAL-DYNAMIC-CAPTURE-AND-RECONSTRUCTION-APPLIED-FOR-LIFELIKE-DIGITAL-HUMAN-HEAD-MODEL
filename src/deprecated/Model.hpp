// include/Model.hpp
#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <assimp/Importer.hpp> // This is required for Assimp::Importer
#include <assimp/scene.h>
#include <string>
#include <vector>
#include <map> // Add this if not already included

struct Mesh {
    unsigned int VAO, VBO, EBO;
    GLuint textureID = 0; // for diffuse texture
    unsigned int indexCount;
    std::vector<int> boneIDs;
    std::vector<float> weights;
    // Inside Mesh struct
    std::vector<GLuint> morphVBOs;  // VBOs for morph target positions
    std::vector<std::string> morphTargetNames;
    std::vector<float> morphWeights; // You can update this dynamically in UpdateAnimation()
    //std::vector<GLuint> morphSSBOs;

    
};

struct BoneInfo {
    glm::mat4 offsetMatrix;
    glm::mat4 finalTransformation;
};

class Model {
public:
    Model();
    Model(const std::string& path);
    ~Model();

    void Draw();
    void UpdateAnimation(float time); // Stub
    void SetMorphWeights(const std::vector<float>& weights);
    void UpdateMorphWeights(const std::vector<float>& weights);
    GLuint GetTextureID(int meshIndex = 0) const {
        if (meshIndex >= 0 && meshIndex < meshes.size()) {
            return meshes[meshIndex].textureID;
        }
        return 0; // Invalid index or no texture
    }
    GLuint GetMorphSSBO() const { return morphSSBO; }     

    private:
    void loadModel(const std::string& path);
    void processNode(aiNode* node, const aiScene* scene);
    Mesh processMesh(aiMesh* mesh, const aiScene* scene);

    void readNodeHierarchy(float animationTime, const aiNode* node, const glm::mat4& parentTransform);
    const aiNodeAnim* FindNodeAnim(const aiAnimation* animation, const std::string& nodeName);
    glm::mat4 aiMatrixToGlm(const aiMatrix4x4& mat);

    void calculateBoneTransform(float timeInSeconds);
    glm::vec3 CalcInterpolatedScaling(float time, const aiNodeAnim* nodeAnim);
    unsigned int FindScalingIndex(float time, const aiNodeAnim* nodeAnim);
    glm::quat CalcInterpolatedRotation(float time, const aiNodeAnim* nodeAnim);
    unsigned int FindRotationIndex(float time, const aiNodeAnim* nodeAnim);
    glm::vec3 CalcInterpolatedPosition(float time, const aiNodeAnim* nodeAnim);
    unsigned int FindPositionIndex(float time, const aiNodeAnim* nodeAnim);
    
private:
    std::vector<Mesh> meshes;
    std::map<std::string, GLuint> loadedTextures;

    std::map<std::string, int> boneMapping; // bone name -> index
    std::vector<BoneInfo> boneInfos;
    unsigned int boneCount = 0;

    const aiScene* scene = nullptr;
    Assimp::Importer importer;
    glm::mat4 globalInverseTransform;

    std::map<std::string, const aiMeshAnim*> meshMorphAnimations;

    float ticksPerSecond = 25.0f;
    float duration = 0.0f;

    GLuint morphSSBO = 0;        // SSBO for morph weights
    int numMorphs = 0;           // Number of morph targets active
};