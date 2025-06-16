// src/Model.cpp
#include "Model.hpp"
#include <assimp/Importer.hpp>
#include <assimp/Exporter.hpp>
#include <assimp/postprocess.h>
#include <iostream>
#include "TextureLoader.hpp"
#define PYBIND11_DETAILED_ERROR_MESSAGES
#include <pybind11/embed.h> // Everything needed for embedding
#include <Python.h>
namespace py = pybind11;

Model::Model() : loaded(false) {}

Model::Model(const std::string& path) {
    loadModel(path);
}

Model::Model(const std::string& path, Shader& globalshader) {
    loadModel(path);
    shader = &globalshader;
    loaded = true;
}

void Model::Draw() {
    if (!loaded) return;
    static bool hasPrinted = false;

    for (int meshIndex = 0; meshIndex < meshes.size(); ++meshIndex) {
        const auto& mesh = meshes[meshIndex];

        shader->setInt("numMorphTargets", (int)morphWeights.size());
        shader->setInt("numVertices", mesh.numVertices);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, mesh.morphSSBO);
        //glBindBuffer(GL_SHADER_STORAGE_BUFFER, mesh.morphSSBO);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, mesh.weightsSSBO);
        //glBindBuffer(GL_SHADER_STORAGE_BUFFER, mesh.weightsSSBO);


        if (!hasPrinted) {
            std::cout << "[Mesh " << meshIndex << "] ";
            if (mesh.textureID) {
                std::cout << "Texture ID: " << mesh.textureID << " (texture bound)\n";
            } else {
                std::cout << "No texture (textureID = 0)\n";
            }
        }

        if (mesh.textureID) {
            glBindTexture(GL_TEXTURE_2D, mesh.textureID);
        } else {
            glBindTexture(GL_TEXTURE_2D, 0); // Unbind any texture if none for this mesh
        }

        glBindVertexArray(mesh.VAO);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, mesh.debugSSBO);
        
        glDrawElements(GL_TRIANGLES, mesh.indexCount, GL_UNSIGNED_INT, 0);

        // Optional: unbind VAO and texture here if needed
        glBindVertexArray(0);
        glBindTexture(GL_TEXTURE_2D, 0);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, mesh.debugSSBO);

        glm::vec4* mapped = (glm::vec4*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);

        /*if (mapped) {
            for (int i = 0; i < mesh.numVertices; ++i) {
                glm::vec4 pos = mapped[i];
                std::cout << "Vertex[" << i << "] VERTEX INDEX: " << pos.w << "\n";
                std::cout << "Morphed DELTA[" << i << "]: (" 
                        << pos.x << ", " << pos.y << ", " << pos.z << ")\n";
            }
            glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
        }*/

    }

    hasPrinted = true;
}

void Model::UpdateAnimation(float time) {
    if (!loaded) return;
    auto it = meshes[0].morphAnimations.find(0); // Assume mesh index 0
    if (it == meshes[0].morphAnimations.end()) return;

    const auto& keys = it->second;
    if (keys.empty()) return;

    size_t morphTargetsCount = morphWeights.size(); // Number of morph targets
    //size_t totalFrames = keys.size() / morphTargetsCount;
    size_t totalFrames = keys.size();


    //std::cout << "[DEBUG] morphTargetsCount = " << morphTargetsCount << "\n";
    //std::cout << "[DEBUG] totalFrames = " << totalFrames << "\n";

    if (totalFrames < 2) return;

    //float scaledTime = time * 1000.0f;
    float maxTime = keys.back().time;
    float scaledTime = fmod(time * 1000.0f, maxTime);

    //std::cout << "[DEBUG] scaledTime = " << scaledTime << "\n";

    // Find current frame
    int frame1 = 0, frame2 = 1;
    for (size_t f = 1; f < totalFrames; ++f) {
        //float frameTime = keys[f * morphTargetsCount].time;
        float frameTime = keys[f].time;
        if (frameTime > scaledTime) {
            frame2 = f;
            frame1 = f - 1;
            break;
        }
    }

    //float t1 = keys[frame1 * morphTargetsCount].time;
    float t1 = keys[frame1].time;
    //float t2 = keys[frame2 * morphTargetsCount].time;
    float t2 = keys[frame2].time;
    float alpha = (t2 != t1) ? (scaledTime - t1) / (t2 - t1) : 0.0f;

    //std::cout << "[DEBUG] Using frames: " << frame1 << " (time = " << t1 << "), "
    //         << frame2 << " (time = " << t2 << "), alpha = " << alpha << "\n";

    // Print full weights of frame1 and frame2
    /*std::cout << "[DEBUG] Keyframe " << frame1 << " weights:\n";
    for (size_t j = 0; j < keys[frame1].weights.size(); ++j) {
        std::cout << "  weights[" << j << "] = " << keys[frame1].weights[j] << "\n";
    }

    std::cout << "[DEBUG] Keyframe " << frame2 << " weights:\n";
    for (size_t j = 0; j < keys[frame2].weights.size(); ++j) {
        std::cout << "  weights[" << j << "] = " << keys[frame2].weights[j] << "\n";
    }*/

    
    // Interpolate each morph weight
    for (size_t j = 0; j < morphTargetsCount; ++j) {
        //size_t idx1 = frame1 * morphTargetsCount + j;
        //size_t idx2 = frame2 * morphTargetsCount + j;

        //float w1 = keys[idx1].weights[0]; // each is a scalar
        float w1 = keys[frame1].weights[j]; // each is a scalar
        //float w2 = keys[idx2].weights[0];
        float w2 = keys[frame2].weights[j];

        float interp = (1 - alpha) * w1 + alpha * w2;

        morphWeights[j] = interp;

        //std::cout << "[DEBUG] morphWeight[" << j << "]: w1 = " << w1
        //          << ", w2 = " << w2 << ", interpolated = " << interp << "\n";
    }

    //std::fill(morphWeights.begin(), morphWeights.end(), 0.0f);
    //morphWeights[100] = 1.0f;  // only activate first morph


    /*
    // 1. Clear morph weights before interpolation
    std::fill(morphWeights.begin(), morphWeights.end(), 0.0f);

    // 2. Create temporary "one-hot" versions of keys[frame1] and keys[frame2]
    std::vector<float> weights1(morphTargetsCount, 0.0f);
    std::vector<float> weights2(morphTargetsCount, 0.0f);

    // Only the "identity" index is set to 1.0
    if (frame1 < morphTargetsCount)
        weights1[frame1] = 1.0f;

    if (frame2 < morphTargetsCount)
        weights2[frame2] = 1.0f;

    // 3. Interpolate each morph weight
    for (size_t j = 0; j < morphTargetsCount; ++j) {
        float w1 = weights1[j];
        float w2 = weights2[j];
        float interp = (1 - alpha) * w1 + alpha * w2;

        morphWeights[j] = interp;

        //std::cout << "[DEBUG] morphWeight[" << j << "]: w1 = " << w1
        //        << ", w2 = " << w2 << ", interpolated = " << interp << "\n";
    }*/

    // Upload to SSBO
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, meshes[0].weightsSSBO);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, morphWeights.size() * sizeof(float), morphWeights.data());
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, meshes[0].weightsSSBO);
}

void Model::UpdateAnimationWithFrame(int frameIndex, float alphaBetweenFrames) {
    if (!loaded) return;

    auto it = meshes[0].morphAnimations.find(0);
    if (it == meshes[0].morphAnimations.end()) return;

    const auto& keys = it->second;
    if (keys.empty()) return;

    size_t morphTargetsCount = morphWeights.size();
    size_t totalFrames = keys.size();

    if (totalFrames < 2) return;

    // Clamp frameIndex to valid range just in case
    frameIndex = frameIndex % totalFrames;
    int frame2 = (frameIndex + 1) % totalFrames;

    // Optional: you can retrieve times t1 and t2 if needed
    float t1 = keys[frameIndex].time;
    float t2 = keys[frame2].time;

    float alpha = alphaBetweenFrames;
    alpha = std::clamp(alpha, 0.0f, 1.0f);

    // Safety check
    if (keys[frameIndex].weights.size() != morphTargetsCount || keys[frame2].weights.size() != morphTargetsCount) {
        // Handle error, or early return
        return;
    }

    for (size_t j = 0; j < morphTargetsCount; ++j) {
        float w1 = keys[frameIndex].weights[j];
        float w2 = keys[frame2].weights[j];
        morphWeights[j] = (1.0f - alpha) * w1 + alpha * w2;
    }

    // Upload morphWeights to GPU
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, meshes[0].weightsSSBO);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, morphWeights.size() * sizeof(float), morphWeights.data());
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, meshes[0].weightsSSBO);
}

void Model::ExpressionControl(const float* Expressions){
    if (!loaded) return;

    size_t morphTargetsCount = morphWeights.size();

    for (size_t j = 0; j < morphTargetsCount; ++j)
    {
        morphWeights[j] = std::clamp(Expressions[j], -1.0f, 1.0f);
    }

    // Upload updated weights to GPU
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, meshes[0].weightsSSBO);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, morphWeights.size() * sizeof(float), morphWeights.data());
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, meshes[0].weightsSSBO);
}

void Model::loadModel(const std::string& path) {
    
    ModelPath = path;
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
    Mesh result;
    std::vector<float> vertices;
    std::vector<unsigned int> indices;
    std::vector<glm::vec4> flattenedMorphData;

    result.numVertices = mesh->mNumVertices;
    //std::cout << "Mesh has " << result.numVertices << " vertices and "
    //      << result.indexCount << " indices." << std::endl;
    for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
        vertices.push_back(mesh->mVertices[i].x);
        vertices.push_back(mesh->mVertices[i].y);
        vertices.push_back(mesh->mVertices[i].z);

        /*{ // Only show debug for first few vertices for brevity
            std::cout << "[DEBUG] Mesh Vertex " << i
                      << " Pos = (" << mesh->mVertices[i].x << ", " << mesh->mVertices[i].y << ", " << mesh->mVertices[i].z << ")\n";
                      //<< " normalDelta = (" << normalDelta.x << ", " << normalDelta.y << ", " << normalDelta.z << ")\n";
        }*/

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

    // Check for morph targets
    if (mesh->mNumAnimMeshes > 0) {
        //std::cout << "[DEBUG] Mesh has " << mesh->mNumAnimMeshes << " morph targets.\n";

        result.numMorphTargets = mesh->mNumAnimMeshes;

        for (unsigned int i = 0; i < mesh->mNumAnimMeshes; ++i) {
            aiAnimMesh* animMesh = mesh->mAnimMeshes[i];
            MorphTarget target;

            //std::cout << "[DEBUG] Morph Target #" << i << " has vertex count = " << mesh->mNumVertices << "\n";

            for (unsigned int v = 0; v < mesh->mNumVertices; ++v) {
                glm::vec4 posDelta(0.0f);
                glm::vec4 normalDelta(0.0f);

                if (animMesh->mVertices) {
                    const aiVector3D& base = mesh->mVertices[v];
                    const aiVector3D& target = animMesh->mVertices[v];
                    if (target == base) {
                        posDelta = glm::vec4(0.0f);  // no change
                    } else {
                        posDelta = glm::vec4(target.x - base.x, target.y - base.y, target.z - base.z, 0.0f);
                    }
                }

                if (animMesh->mNormals) {
                    const aiVector3D& baseN = mesh->mNormals[v];
                    const aiVector3D& targetN = animMesh->mNormals[v];
                    if (targetN == baseN) {
                        normalDelta = glm::vec4(0.0f);
                    } else {
                        normalDelta = glm::vec4(targetN.x - baseN.x, targetN.y - baseN.y, targetN.z - baseN.z, 0.0f);
                    }
                }

                // Store both position and normal deltas sequentially
                flattenedMorphData.push_back(posDelta);
                flattenedMorphData.push_back(normalDelta);
                
                target.positions.push_back(posDelta);
                target.normals.push_back(normalDelta);

               
                /*if(i == 100)
                { // Only show debug for first few vertices for brevity
                    int baseIndex = i * mesh->mNumVertices * 2; // each vertex has pos + normal = 2 entries
                    glm::vec3 posDelta = flattenedMorphData[baseIndex + v * 2];
    
                    std::cout << "[FlatData DEBUG] Vertex " << v
                              << " posDelta = (" << posDelta.x << ", " << posDelta.y << ", " << posDelta.z << ")\n";
                }*/
            }

            result.morphTargets.push_back(std::move(target));
        }

        /*{ // Only show debug for first few vertices for brevity
            aiAnimMesh* animMesh = mesh->mAnimMeshes[100];
            std::cout << "[DEBUG] Morph " << 100;
            for (unsigned int v = 0; v < animMesh->mNumVertices; ++v) {
                std::cout << "[DEBUG] Morph Vertex " << v 
                        << " Pos = (" << animMesh->mVertices[v].x << ", " << animMesh->mVertices[v].y << ", " << animMesh->mVertices[v].z << ")\n";
                        //<< " normalDelta = (" << normalDelta.x << ", " << normalDelta.y << ", " << normalDelta.z << ")\n";
            }

        // Print faces (indices referencing mVertices/morph target vertices)
            std::cout << "[DEBUG] Faces using vertex indices:\n";
            for (unsigned int f = 0; f < mesh->mNumFaces; ++f) {
                const aiFace& face = mesh->mFaces[f];
                std::cout << "  Face " << f << ": ";
                for (unsigned int j = 0; j < face.mNumIndices; ++j) {
                    std::cout << face.mIndices[j] << " ";
                }
                std::cout << "\n";
            }
        }*/


    }


    for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
        for (unsigned int j = 0; j < mesh->mFaces[i].mNumIndices; j++) {
            indices.push_back(mesh->mFaces[i].mIndices[j]);
        }
    }

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

    glGenBuffers(1, &result.morphSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, result.morphSSBO);
    //glBufferData(GL_SHADER_STORAGE_BUFFER, result.morphTargets.size() * sizeof(MorphTarget),
    //             result.morphTargets.data(), GL_STATIC_DRAW);
    glBufferData(GL_SHADER_STORAGE_BUFFER, flattenedMorphData.size() * sizeof(glm::vec4),
    flattenedMorphData.data(), GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, result.morphSSBO); // Binding = 0
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    // Load default weights if morph targets exist
    if (mesh->mNumAnimMeshes > 0) {
        std::vector<float> defaultWeights(mesh->mNumAnimMeshes);
        for (unsigned int i = 0; i < mesh->mNumAnimMeshes; ++i) {
            defaultWeights[i] = mesh->mAnimMeshes[i]->mWeight;
        }

        // Debug print
        //std::cout << "[DEBUG] Initial morph weights from mAnimMeshes:\n";
        //for (size_t i = 0; i < defaultWeights.size(); ++i) {
        //    std::cout << "  Weight[" << i << "] = " << defaultWeights[i] << "\n";
        //}

        morphWeights = defaultWeights;

        glGenBuffers(1, &result.weightsSSBO);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, result.weightsSSBO);
        glBufferData(GL_SHADER_STORAGE_BUFFER, morphWeights.size() * sizeof(float), morphWeights.data(), GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, result.weightsSSBO); // Binding = 1
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }

    if (mesh->mMaterialIndex >= 0) {
        aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
    
        unsigned int diffuseCount = material->GetTextureCount(aiTextureType_DIFFUSE);
        //std::cout << "Diffuse texture count: " << diffuseCount << std::endl;
    
        if (diffuseCount > 0) {
            aiString str;
            material->GetTexture(aiTextureType_DIFFUSE, 0, &str);
    
            std::string texPath = str.C_Str();
            //std::cout << "Diffuse texture path string: " << texPath << std::endl;
    
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
    
    for (unsigned int i = 0; i < scene->mNumAnimations; ++i) {
        aiAnimation* anim = scene->mAnimations[i];
        //std::cout << "[DEBUG] Animation " << i << ": name = " 
        //          << (anim->mName.C_Str() ? anim->mName.C_Str() : "Unnamed") 
        //          << ", NumMorphMeshChannels = " << anim->mNumMorphMeshChannels << "\n";
    
        for (unsigned int j = 0; j < anim->mNumMorphMeshChannels; ++j) {
            aiMeshMorphAnim* morphAnim = anim->mMorphMeshChannels[j];
            //std::cout << "  [DEBUG] MorphMeshChannel " << j << ": node name = " 
            //          << morphAnim->mName.C_Str() 
            //          << ", NumKeys = " << morphAnim->mNumKeys << "\n";
    
            aiNode* node = scene->mRootNode->FindNode(morphAnim->mName);
            if (!node) {
                //std::cout << "    [WARNING] Node not found for morph channel: " << morphAnim->mName.C_Str() << "\n";
                continue;
            }
    
            if (node->mNumMeshes == 0) {
                //std::cout << "    [WARNING] Node has no meshes: " << morphAnim->mName.C_Str() << "\n";
                continue;
            }
    
            int meshIndex = node->mMeshes[0];
            //std::cout << "    [DEBUG] Using mesh index: " << meshIndex << "\n";
    
            std::vector<MorphAnimKey> keys;
            for (unsigned int k = 0; k < morphAnim->mNumKeys; ++k) {
                const aiMeshMorphKey& key = morphAnim->mKeys[k];
                MorphAnimKey newKey;
                newKey.time = key.mTime;
                newKey.weights.resize(key.mNumValuesAndWeights);
    
                //std::cout << "      [DEBUG] Keyframe " << k << ": time = " << key.mTime 
                //          << ", NumValues = " << key.mNumValuesAndWeights << "\n";
    
                for (unsigned int w = 0; w < key.mNumValuesAndWeights; ++w) {
                    newKey.weights[w] = key.mWeights[w];
                    //std::cout << "        [DEBUG] Weight[" << w << "] = " << key.mWeights[w] << "\n";
                }
    
                keys.push_back(newKey);
            }
    
            result.morphAnimations[meshIndex] = keys;
        }
    }
    
    glBindVertexArray(0);

    
    glGenBuffers(2, &result.debugSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, result.debugSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(glm::vec4) * result.numVertices, nullptr, GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, result.debugSSBO);  // binding = 1 matches GLSL
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    
    return result;
}

void Model::ExportModel2(const std::string& outputPath) {
    if (!loaded) return;

    
    Assimp::Importer importer;
    const aiScene* export_scene = importer.ReadFile(ModelPath,
        aiProcess_ValidateDataStructure |
        aiProcess_Triangulate |
        //aiProcess_FlipUVs |
        aiProcess_CalcTangentSpace |
        aiProcess_GenSmoothNormals |
        //aiProcess_JoinIdenticalVertices |
        aiProcess_SortByPType
    );;

    std::string exportFormat = "glb2"; // e.g., "obj", "ply", "stl", "fbx", "glb2"

    // Step 1: Check if the scene is valid
    if (!export_scene || export_scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !export_scene->mRootNode) {
        std::cerr << "Assimp scene is incomplete or null: " << importer.GetErrorString() << std::endl;
        return;
    }

    /*
    // Step 2: Print some debug info about the scene
    std::cout << "Scene debug info:\n";
    std::cout << "  Mesh count: " << export_scene->mNumMeshes << "\n";
    std::cout << "  Material count: " << export_scene->mNumMaterials << "\n";
    std::cout << "  Texture count: " << export_scene->mNumTextures << "\n";
    std::cout << "  Animation count: " << export_scene->mNumAnimations << "\n";
    std::cout << "  Has Root Node: " << (export_scene->mRootNode ? "Yes" : "No") << "\n";

    */

    std::cout << "Animations: " << export_scene->mNumAnimations << "\n";
    for (unsigned int a = 0; a < export_scene->mNumAnimations; ++a) {
        aiAnimation* anim = export_scene->mAnimations[a];
        std::cout << "Animation " << a << ": " << anim->mName.C_Str()
                << ", duration: " << anim->mDuration
                << ", ticks/sec: " << anim->mTicksPerSecond
                << ", morph mesh channels: " << anim->mNumMorphMeshChannels << "\n";

        for (unsigned int m = 0; m < anim->mNumMorphMeshChannels; ++m) {
            aiMeshMorphAnim* morphAnim = anim->mMorphMeshChannels[m];
            std::cout << "  Morph animation for mesh: " << morphAnim->mName.C_Str()
                    << ", keys: " << morphAnim->mNumKeys << "\n";

            for (unsigned int k = 0; k < morphAnim->mNumKeys; ++k) {
                const aiMeshMorphKey& key = morphAnim->mKeys[k];
                std::cout << "    Time: " << key.mTime
                        << ", Num targets: " << key.mNumValuesAndWeights << "\n";
            }
        }
    }



    // Loop through meshes to display vertex data
    for (unsigned int i = 0; i < export_scene->mNumMeshes; ++i) {
        aiMesh* mesh = export_scene->mMeshes[i];
        std::cout << "Mesh " << i << ": " << mesh->mNumVertices << " vertices\n";

        for (unsigned int i = 0; i < mesh->mNumFaces; ++i) {
            aiFace& face = mesh->mFaces[i];
            for (unsigned int j = 0; j < face.mNumIndices; ++j) {
                if (face.mIndices[j] >= mesh->mNumVertices) {
                    std::cerr << "Invalid index found: " << face.mIndices[j]
                            << " (vertex count: " << mesh->mNumVertices << ")\n";
                }
            }
        }

    for (unsigned int j = 0; j < mesh->mNumAnimMeshes; ++j) {
        aiAnimMesh* morph = mesh->mAnimMeshes[j];
        for (unsigned int i = 0; i < mesh->mNumFaces; ++i) {
            aiFace& face = mesh->mFaces[i];
            for (unsigned int k = 0; k < face.mNumIndices; ++k) {
                if (face.mIndices[k] >= morph->mNumVertices) {
                    std::cerr << "Morph target " << j << " has insufficient vertices.\n";
                    std::cerr << "Index " << face.mIndices[k]
                            << " exceeds morph vertex count: " << morph->mNumVertices << "\n";
                }
            }
        }
    }

    /*
        for (unsigned int v = 0; v < mesh->mNumVertices; ++v) {
            aiVector3D pos = mesh->mVertices[v];
            std::cout << "  Vertex " << v << ": ("
                    << pos.x << ", " << pos.y << ", " << pos.z << ")";
            
            if (mesh->HasNormals()) {
                aiVector3D normal = mesh->mNormals[v];
                std::cout << ", Normal: ("
                        << normal.x << ", " << normal.y << ", " << normal.z << ")";
            }

            if (mesh->HasTextureCoords(0)) {
                aiVector3D uv = mesh->mTextureCoords[0][v];
                std::cout << ", UV: (" << uv.x << ", " << uv.y << ")";
            }

            std::cout << "\n";
        }*/
    }

    // Step 3: Check if the export format is supported
    Assimp::Exporter exporter;
    bool formatSupported = false;
    /*
    unsigned int numFormats = exporter.GetExportFormatCount();
    for (unsigned int i = 0; i < numFormats; ++i) {
        const aiExportFormatDesc* desc = exporter.GetExportFormatDescription(i);
        if (desc && exportFormat == desc->id) {
            formatSupported = true;
            std::cout << "Export format supported: " << desc->description << " (" << desc->fileExtension << ")\n";
            break;
        }
    }

    if (!formatSupported) {
        std::cerr << "Export format not supported: " << exportFormat << "\n";
        std::cerr << "Available formats:\n";
        for (unsigned int i = 0; i < numFormats; ++i) {
            const aiExportFormatDesc* desc = exporter.GetExportFormatDescription(i);
            std::cerr << "  " << desc->id << " - " << desc->description << " (*." << desc->fileExtension << ")\n";
        }
        return;
    }
    */
   
    // Step 4: Attempt to export the model
    if (exporter.Export(export_scene, exportFormat.c_str(), outputPath) != AI_SUCCESS) {
        std::cerr << "Assimp export error: " << exporter.GetErrorString() << std::endl;
    } else {
        std::cout << "Export successful to: " << outputPath << std::endl;
    }
}

void Model::ExportModel(const std::string& FrameDir,
                        const std::string& outputPath,
                        float FPS,
                        int start_frame,
                        int end_frame) 
{
    if (!loaded) return;

    try {
        py::gil_scoped_acquire gil;
        py::module sys = py::module::import("sys");
        py::module face_export = py::module::import("obj2glb");

        std::cout << "Exporting with:\n"
          << "  FrameDir: " << FrameDir << "\n"
          << "  Output: " << outputPath << "\n"
          << "  FPS: " << FPS << "\n"
          << "  start_frame: " << start_frame << ", end_frame: " << end_frame << "\n";

        if (py::hasattr(face_export, "export_glb")) {
            face_export.attr("export_glb")(FrameDir, outputPath, FPS, start_frame, end_frame);
            std::cout << "✅ Export successful!\n";
        } else {
            std::cerr << "❌ Error: 'export_glb' function not found in obj2glb.\n";
        }
    }
    catch (const py::error_already_set& e) {
        std::cerr << "❌ Python Error:\n" << e.what() << "\n";
    }
    catch (const std::exception& e) {
        std::cerr << "❌ Export Failed:\n" << e.what() << "\n";
    }
}

void Model::ExportCustomizedModel(const std::string& FrameDir,
                        const std::string& outputPath,
                        float FPS,
                        int frames,
                        const std::vector<float>& expressions) 
{
    if (!loaded) return;

    std::cout << "expressions size: " << expressions.size() << std::endl;

    try {
        py::gil_scoped_acquire gil;
        py::module sys = py::module::import("sys");
        py::module face_export = py::module::import("obj2glb");

        std::cout << "Exporting with:\n"
          << "  FrameDir: " << FrameDir << "\n"
          << "  Output: " << outputPath << "\n"
          << "  FPS: " << FPS << "\n"
          << "  frames: " << frames << "\n";

        py::int_ pyFrames = py::int_(frames);
        py::list pyExpressions;
        for (float val : expressions) {
            pyExpressions.append(val);
        }


        if (py::hasattr(face_export, "export_animated_glb")) {
                face_export.attr("export_animated_glb")(
                py::cast(FrameDir),
                py::cast(outputPath),
                py::cast(FPS),
                pyFrames,
                pyExpressions
            );
            std::cout << "✅ Export successful!\n";
        } else {
            std::cerr << "❌ Error: 'export_animated_glb' function not found in obj2glb.\n";
        }
    }
    catch (const py::error_already_set& e) {
        std::cerr << "❌ Python Error:\n" << e.what() << "\n";
    }
    catch (const std::exception& e) {
        std::cerr << "❌ Export Failed:\n" << e.what() << "\n";
    }
}

