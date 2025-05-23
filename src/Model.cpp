// src/Model.cpp
#include "Model.hpp"
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <iostream>
#include "TextureLoader.hpp"

Model::Model() {
}

Model::Model(const std::string& path) {
    loadModel(path);
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
            //glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, mesh.textureID);

        glBindVertexArray(mesh.VAO);
        glDrawElements(GL_TRIANGLES, mesh.indexCount, GL_UNSIGNED_INT, 0);
        meshIndex++;
    }

    hasPrinted = true;  // prevent further prints
}


void Model::UpdateAnimation(float time) {
    // Animation system stub (future implementation)
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

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);           // position
    glEnableVertexAttribArray(0);
    
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float))); // normal
    glEnableVertexAttribArray(1);
    
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float))); // tex coords
    glEnableVertexAttribArray(2);    

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

