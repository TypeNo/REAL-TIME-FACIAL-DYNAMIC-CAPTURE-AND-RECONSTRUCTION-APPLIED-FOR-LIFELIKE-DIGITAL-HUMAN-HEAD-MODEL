// src/Model.cpp
#include "Model.hpp"
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <iostream>
#include "TextureLoader.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/glm.hpp>

glm::mat4 ConvertMatrixToGLMFormat(const aiMatrix4x4& from);
void AddBoneDataToVertex(int vertexID, int boneIndex, float weight,
    std::vector<glm::ivec4>& boneIDs,
    std::vector<glm::vec4>& weights);

Model::Model() : morphSSBO(0), numMorphs(0), boneCount(0), scene(nullptr) {
    glGenBuffers(1, &morphSSBO);
}

Model::Model(const std::string& path) : Model() {
    loadModel(path);
}

Model::~Model() {
    if (morphSSBO) {
        glDeleteBuffers(1, &morphSSBO);
        morphSSBO = 0;
    }
}

void Model::SetMorphWeights(const std::vector<float>& weights) {
    numMorphs = static_cast<int>(weights.size());

    if (!morphSSBO) glGenBuffers(1, &morphSSBO);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, morphSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * numMorphs, weights.data(), GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, morphSSBO); // match binding=0 in shader
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void Model::UpdateMorphWeights(const std::vector<float>& weights) {
    if (!morphSSBO) {
        std::cerr << "Morph SSBO not initialized. Call SetMorphWeights first.\n";
        return;
    }

    if (static_cast<int>(weights.size()) != numMorphs) {
        std::cerr << "UpdateMorphWeights size mismatch. Expected " << numMorphs << " but got " << weights.size() << "\n";
        return;
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, morphSSBO);
    float* ptr = (float*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_WRITE_ONLY);
    if (ptr) {
        std::memcpy(ptr, weights.data(), sizeof(float) * numMorphs);
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    } else {
        std::cerr << "Failed to map SSBO for morph weights update." << std::endl;
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void Model::Draw() {
    static bool hasPrinted = false;  // ensures logging only happens once

    int meshIndex = 0;
    for (const auto& mesh : meshes) {
        if (!hasPrinted) {
            std::cout << "[Mesh " << meshIndex << "] ";
            if (mesh.textureID) {
                std::cout << "Texture ID: " << mesh.textureID << " (texture bound)\n";
            } else {
                std::cout << "No texture (textureID = 0)\n";
            }
        }

        if (mesh.textureID)
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, mesh.textureID);

        glBindVertexArray(mesh.VAO);
        /*for (size_t i = 0; i < mesh.morphWeights.size(); ++i) {
            std::string uniformName = "morphWeights[" + std::to_string(i) + "]";
            GLint location = glGetUniformLocation(shaderProgramID, uniformName.c_str());
            glUniform1f(location, mesh.morphWeights[i]);
        }*/
        glDrawElements(GL_TRIANGLES, mesh.indexCount, GL_UNSIGNED_INT, 0);
        meshIndex++;
    }
    glBindVertexArray(0);

    hasPrinted = true;  // prevent further prints
}


void Model::UpdateAnimation(float timeInSeconds) {
    /*
    if (!scene || !scene->mAnimations || scene->mNumAnimations == 0) return;

    aiAnimation* animation = scene->mAnimations[0];
    float ticksPerSecond = animation->mTicksPerSecond != 0 ? animation->mTicksPerSecond : 25.0f;
    float timeInTicks = timeInSeconds * ticksPerSecond;
    float animationTime = fmod(timeInTicks, animation->mDuration);

    readNodeHierarchy(animationTime, scene->mRootNode, glm::mat4(1.0f));
    */
   /*
    for (auto& mesh : meshes) {
        for (size_t i = 0; i < mesh.morphWeights.size(); ++i) {
            mesh.morphWeights[i] = 0.5f * (1.0f + sin(timeInSeconds + i));
        }
    }
        */
}

void Model::readNodeHierarchy(float animationTime, const aiNode* node, const glm::mat4& parentTransform) {
    std::string nodeName(node->mName.data);

    glm::mat4 nodeTransformation = ConvertMatrixToGLMFormat(node->mTransformation);

    const aiAnimation* animation = scene->mAnimations[0];
    const aiNodeAnim* nodeAnim = FindNodeAnim(animation, nodeName);

    if (nodeAnim) {
        // Interpolate transformation
        glm::vec3 scaling = CalcInterpolatedScaling(animationTime, nodeAnim);
        glm::mat4 scalingM = glm::scale(glm::mat4(1.0f), scaling);

        glm::quat rotationQ = CalcInterpolatedRotation(animationTime, nodeAnim);
        glm::mat4 rotationM = glm::mat4_cast(rotationQ);

        glm::vec3 translation = CalcInterpolatedPosition(animationTime, nodeAnim);
        glm::mat4 translationM = glm::translate(glm::mat4(1.0f), translation);

        nodeTransformation = translationM * rotationM * scalingM;
    }

    glm::mat4 globalTransformation = parentTransform * nodeTransformation;

    if (boneMapping.find(nodeName) != boneMapping.end()) {
        int boneIndex = boneMapping[nodeName];
        boneInfos[boneIndex].finalTransformation = globalTransformation * boneInfos[boneIndex].offsetMatrix;
    }

    for (unsigned int i = 0; i < node->mNumChildren; i++) {
        readNodeHierarchy(animationTime, node->mChildren[i], globalTransformation);
    }
}

void Model::loadModel(const std::string& path) {
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(path,
        aiProcess_Triangulate |
        //aiProcess_FlipUVs |
        aiProcess_CalcTangentSpace |
        aiProcess_GenSmoothNormals |
        aiProcess_JoinIdenticalVertices
    );
    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        std::cerr << "Assimp error: " << importer.GetErrorString() << std::endl;
        return;
    }
    
    processNode(scene->mRootNode, scene);
    // Extract morph animation if present
    if (scene->mNumAnimations > 0) {
        const aiAnimation* anim = scene->mAnimations[0];

        for (unsigned int i = 0; i < anim->mNumMeshChannels; ++i) {
            const aiMeshAnim* meshAnim = anim->mMeshChannels[i];
            std::string targetName(meshAnim->mName.C_Str());
            meshMorphAnimations[targetName] = meshAnim;
        }

        ticksPerSecond = anim->mTicksPerSecond != 0 ? anim->mTicksPerSecond : 25.0f;
        duration = anim->mDuration;
    }
}

void Model::processNode(aiNode* node, const aiScene* scene) {
    for (unsigned int i = 0; i < node->mNumMeshes; i++) {
        aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
        meshes.push_back(processMesh(mesh, scene));
    }
    for (unsigned int i = 0; i < node->mNumChildren; i++) {
        processNode(node->mChildren[i], scene);
    }
}

Mesh Model::processMesh(aiMesh* mesh, const aiScene* scene) {
    std::vector<float> vertices;
    std::vector<unsigned int> indices;
    // Before vertex loop
    std::vector<glm::ivec4> boneIDs(mesh->mNumVertices, glm::ivec4(0));
    std::vector<glm::vec4> weights(mesh->mNumVertices, glm::vec4(0.0f));



    /*if (mesh->HasBones()) {
        for (unsigned int i = 0; i < mesh->mNumBones; i++) {
            aiBone* bone = mesh->mBones[i];
            std::string boneName = bone->mName.C_Str();
    
            int boneIndex = 0;
            if (boneMapping.find(boneName) == boneMapping.end()) {
                boneIndex = boneCount++;
                BoneInfo bi;
                aiMatrix4x4 aiOffset = bone->mOffsetMatrix;
                bi.offsetMatrix = ConvertMatrixToGLMFormat(bone->mOffsetMatrix);

                boneInfos.push_back(bi);
                boneMapping[boneName] = boneIndex;
            } else {
                boneIndex = boneMapping[boneName];
            }
    
            for (unsigned int j = 0; j < bone->mNumWeights; j++) {
                int vertexID = bone->mWeights[j].mVertexId;
                float weight = bone->mWeights[j].mWeight;
                AddBoneDataToVertex(vertexID, boneIndex, weight, boneIDs, weights);
            }
        }
    }*/

    for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
        vertices.push_back(mesh->mVertices[i].x);
        vertices.push_back(mesh->mVertices[i].y);
        vertices.push_back(mesh->mVertices[i].z);

        vertices.push_back(mesh->mNormals[i].x);
        vertices.push_back(mesh->mNormals[i].y);
        vertices.push_back(mesh->mNormals[i].z);

        // Texture coordinates
        if (mesh->mTextureCoords[0]) {
            
            vertices.push_back(mesh->mTextureCoords[0][i].x);
            vertices.push_back(mesh->mTextureCoords[0][i].y);
        } else {
            vertices.push_back(0.0f);
            vertices.push_back(0.0f);
        }

        /*
        // Append bone IDs (cast to float)
        vertices.push_back(static_cast<float>(boneIDs[i].x));
        vertices.push_back(static_cast<float>(boneIDs[i].y));
        vertices.push_back(static_cast<float>(boneIDs[i].z));
        vertices.push_back(static_cast<float>(boneIDs[i].w));

        // Append weights
        vertices.push_back(weights[i].x);
        vertices.push_back(weights[i].y);
        vertices.push_back(weights[i].z);
        vertices.push_back(weights[i].w);
        */
    }

    for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
        for (unsigned int j = 0; j < mesh->mFaces[i].mNumIndices; j++) {
            indices.push_back(mesh->mFaces[i].mIndices[j]);
        }
    }

    Mesh result;
    result.indexCount = indices.size();

    glGenVertexArrays(1, &result.VAO);
    glGenBuffers(1, &result.VBO);
    glGenBuffers(1, &result.EBO);

    glBindVertexArray(result.VAO);

    glBindBuffer(GL_ARRAY_BUFFER, result.VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, result.EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

    // Setup vertex attribute pointers
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0); // pos
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float))); // normal
    glEnableVertexAttribArray(1);

    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float))); // tex
    glEnableVertexAttribArray(2);

    /*
    glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 16 * sizeof(float), (void*)(8 * sizeof(float))); // bone IDs
    glEnableVertexAttribArray(3);

    glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, 16 * sizeof(float), (void*)(12 * sizeof(float))); // weights
    glEnableVertexAttribArray(4);
    */

    // Morph targets
    if (mesh->mNumAnimMeshes > 0) {
        for (unsigned int i = 0; i < mesh->mNumAnimMeshes; ++i) {
            aiAnimMesh* animMesh = mesh->mAnimMeshes[i];
            std::vector<float> deltaVertices;

            for (unsigned int v = 0; v < mesh->mNumVertices; ++v) {
                aiVector3D delta = animMesh->mVertices[v] - mesh->mVertices[v];
                deltaVertices.push_back(delta.x);
                deltaVertices.push_back(delta.y);
                deltaVertices.push_back(delta.z);
            }

            GLuint morphVBO;
            glGenBuffers(1, &morphVBO);
            glBindBuffer(GL_ARRAY_BUFFER, morphVBO);
            glBufferData(GL_ARRAY_BUFFER, deltaVertices.size() * sizeof(float), deltaVertices.data(), GL_STATIC_DRAW);

            glBindVertexArray(result.VAO);
            glBindBuffer(GL_ARRAY_BUFFER, morphVBO);
            glVertexAttribPointer(5 + i, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
            glEnableVertexAttribArray(5 + i);

            result.morphVBOs.push_back(morphVBO);
            result.morphTargetNames.push_back(animMesh->mName.C_Str());
            result.morphWeights.push_back(0.0f); // Will be set in UpdateAnimation
        }
    }

    glBindVertexArray(0);


    if (mesh->mMaterialIndex >= 0) {
        aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
    
        unsigned int diffuseCount = material->GetTextureCount(aiTextureType_DIFFUSE);
        std::cout << "Diffuse texture count: " << diffuseCount << std::endl;
    
        if (diffuseCount > 0) {
            aiString str;
            material->GetTexture(aiTextureType_DIFFUSE, 0, &str);
    
            std::string texPath = str.C_Str();
            std::cout << "Diffuse texture path string: " << texPath << std::endl;
    
            if (!texPath.empty() && texPath[0] == '*') {
                std::cout << "Checking embedded texture cache for key: " << texPath << std::endl;
            
                std::cout << "Currently loaded embedded textures:\n";
                for (const auto& pair : loadedTextures) {
                    std::cout << " - " << pair.first << " => Texture ID: " << pair.second << std::endl;
                }
            
                auto found = loadedTextures.find(texPath);
                auto endIt = loadedTextures.end();
            
                if (found != endIt) {
                    std::cout << "✅ Found texture in cache. Reusing texture ID: " << found->second << std::endl;
                    result.textureID = found->second;
                } else {
                    std::cout << "❌ Texture not found in cache. Loading new embedded texture." << std::endl;
            
                    int texIndex = std::stoi(texPath.substr(1));
                    std::cout << "Embedded texture index: " << texIndex << std::endl;
            
                    if (scene->mTextures && texIndex < scene->mNumTextures) {
                        aiTexture* texture = scene->mTextures[texIndex];
                        std::cout << "Embedded texture width: " << texture->mWidth << ", height: " << texture->mHeight << std::endl;
            
                        if (texture->mHeight == 0) {
                            std::cout << "Loading compressed embedded texture from memory (e.g., PNG/JPEG)" << std::endl;
                            GLuint texID = LoadTextureFromMemory(reinterpret_cast<unsigned char*>(texture->pcData), texture->mWidth);
                            result.textureID = texID;
                            loadedTextures[texPath] = texID;  // ✅ Store in cache
                        } else {
                            std::cerr << "Uncompressed embedded texture detected (not supported yet)" << std::endl;
                        }
                    } else {
                        std::cerr << "Invalid embedded texture index or no embedded textures present." << std::endl;
                    }
                }
            } else {
                std::cout << "Loading external texture from file: " << texPath << std::endl;
                std::string fullPath = "textures/" + texPath;
                result.textureID = LoadTextureFromFile(fullPath.c_str());
            }
        } else {
            std::cout << "No diffuse texture found for material." << std::endl;
        }
    }
    
    return result;
}

glm::mat4 ConvertMatrixToGLMFormat(const aiMatrix4x4& from) {
    glm::mat4 to;
    to[0][0] = from.a1; to[1][0] = from.a2; to[2][0] = from.a3; to[3][0] = from.a4;
    to[0][1] = from.b1; to[1][1] = from.b2; to[2][1] = from.b3; to[3][1] = from.b4;
    to[0][2] = from.c1; to[1][2] = from.c2; to[2][2] = from.c3; to[3][2] = from.c4;
    to[0][3] = from.d1; to[1][3] = from.d2; to[2][3] = from.d3; to[3][3] = from.d4;
    return to;
}

const aiNodeAnim* Model::FindNodeAnim(const aiAnimation* animation, const std::string& nodeName) {
    for (unsigned int i = 0; i < animation->mNumChannels; i++) {
        const aiNodeAnim* nodeAnim = animation->mChannels[i];
        if (std::string(nodeAnim->mNodeName.data) == nodeName) {
            return nodeAnim;
        }
    }
    return nullptr;
}

glm::vec3 Model::CalcInterpolatedScaling(float time, const aiNodeAnim* nodeAnim) {
    if (nodeAnim->mNumScalingKeys == 1) {
        return glm::vec3(nodeAnim->mScalingKeys[0].mValue.x,
                         nodeAnim->mScalingKeys[0].mValue.y,
                         nodeAnim->mScalingKeys[0].mValue.z);
    }

    unsigned int index = FindScalingIndex(time, nodeAnim);
    unsigned int nextIndex = index + 1;
    float delta = (float)(nodeAnim->mScalingKeys[nextIndex].mTime - nodeAnim->mScalingKeys[index].mTime);
    float factor = (time - (float)nodeAnim->mScalingKeys[index].mTime) / delta;

    aiVector3D start = nodeAnim->mScalingKeys[index].mValue;
    aiVector3D end = nodeAnim->mScalingKeys[nextIndex].mValue;
    aiVector3D result = start + factor * (end - start);
    return glm::vec3(result.x, result.y, result.z);
}

unsigned int Model::FindScalingIndex(float time, const aiNodeAnim* nodeAnim) {
    for (unsigned int i = 0; i < nodeAnim->mNumScalingKeys - 1; i++) {
        if (time < (float)nodeAnim->mScalingKeys[i + 1].mTime)
            return i;
    }
    return 0;
}

glm::quat Model::CalcInterpolatedRotation(float time, const aiNodeAnim* nodeAnim) {
    if (nodeAnim->mNumRotationKeys == 1) {
        return glm::quat(nodeAnim->mRotationKeys[0].mValue.w,
                         nodeAnim->mRotationKeys[0].mValue.x,
                         nodeAnim->mRotationKeys[0].mValue.y,
                         nodeAnim->mRotationKeys[0].mValue.z);
    }

    unsigned int index = FindRotationIndex(time, nodeAnim);
    unsigned int nextIndex = index + 1;
    float delta = (float)(nodeAnim->mRotationKeys[nextIndex].mTime - nodeAnim->mRotationKeys[index].mTime);
    float factor = (time - (float)nodeAnim->mRotationKeys[index].mTime) / delta;

    aiQuaternion start = nodeAnim->mRotationKeys[index].mValue;
    aiQuaternion end = nodeAnim->mRotationKeys[nextIndex].mValue;
    aiQuaternion interpolated;
    aiQuaternion::Interpolate(interpolated, start, end, factor);
    interpolated.Normalize();
    return glm::quat(interpolated.w, interpolated.x, interpolated.y, interpolated.z);
}

unsigned int Model::FindRotationIndex(float time, const aiNodeAnim* nodeAnim) {
    for (unsigned int i = 0; i < nodeAnim->mNumRotationKeys - 1; i++) {
        if (time < (float)nodeAnim->mRotationKeys[i + 1].mTime)
            return i;
    }
    return 0;
}

glm::vec3 Model::CalcInterpolatedPosition(float time, const aiNodeAnim* nodeAnim) {
    if (nodeAnim->mNumPositionKeys == 1) {
        return glm::vec3(nodeAnim->mPositionKeys[0].mValue.x,
                         nodeAnim->mPositionKeys[0].mValue.y,
                         nodeAnim->mPositionKeys[0].mValue.z);
    }

    unsigned int index = FindPositionIndex(time, nodeAnim);
    unsigned int nextIndex = index + 1;
    float delta = (float)(nodeAnim->mPositionKeys[nextIndex].mTime - nodeAnim->mPositionKeys[index].mTime);
    float factor = (time - (float)nodeAnim->mPositionKeys[index].mTime) / delta;

    aiVector3D start = nodeAnim->mPositionKeys[index].mValue;
    aiVector3D end = nodeAnim->mPositionKeys[nextIndex].mValue;
    aiVector3D result = start + factor * (end - start);
    return glm::vec3(result.x, result.y, result.z);
}

unsigned int Model::FindPositionIndex(float time, const aiNodeAnim* nodeAnim) {
    for (unsigned int i = 0; i < nodeAnim->mNumPositionKeys - 1; i++) {
        if (time < (float)nodeAnim->mPositionKeys[i + 1].mTime)
            return i;
    }
    return 0;
}

void AddBoneDataToVertex(int vertexID, int boneIndex, float weight,
    std::vector<glm::ivec4>& boneIDs,
    std::vector<glm::vec4>& weights)
{
    for (int i = 0; i < 4; i++) {
    if (weights[vertexID][i] == 0.0f) {
    boneIDs[vertexID][i] = boneIndex;
    weights[vertexID][i] = weight;
    return;
    }
    }
    // If we get here, that vertex already has 4 weights, which we ignore or log warning
    std::cerr << "Warning: More than 4 bone influences for vertex " << vertexID << std::endl;
}