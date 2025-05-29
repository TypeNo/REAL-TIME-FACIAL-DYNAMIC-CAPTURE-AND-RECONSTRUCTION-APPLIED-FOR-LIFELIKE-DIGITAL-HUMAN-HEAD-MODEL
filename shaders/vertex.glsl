#version 430 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;

// Morph target deltas (interleaved: posDelta, normalDelta per vertex per morph target)
layout(std430, binding = 0) buffer MorphTargetsBuffer {
    vec4 morphData[]; // .w is ignored
};


// Morph weights for each morph target
layout(std430, binding = 1) buffer MorphWeights {
    float weights[];
};

layout(std430, binding = 2) buffer DebugOutput {
    vec4 debugPositions[];  // output morphed positions (can also store normals or deltas)
};


// Uniforms
uniform int numMorphTargets;
uniform int numVertices;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

// Output to fragment shader
out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;

void main() {
    int vertexID = gl_VertexID;

    vec3 pos = aPos;
    vec3 normal = aNormal;

    // Apply morph deltas
    for (int i = 0; i < numMorphTargets; ++i) {
        int baseIndex = (i * numVertices + vertexID) * 2; // 2: pos + normal
        vec3 posDelta = morphData[baseIndex].xyz;
        vec3 normalDelta = morphData[baseIndex + 1].xyz;

        float weight = clamp(weights[i], -1.0, 1.0);
        if(weight != 0.0 ){
            debugPositions[gl_VertexID] = vec4(posDelta, int(baseIndex));
        }

        pos += posDelta * weight;
        normal += normalDelta * weight;
    }

    // Compute final transformed values
    FragPos = vec3(model * vec4(pos, 1.0));
    Normal = mat3(transpose(inverse(model))) * normalize(normal);
    TexCoord = aTexCoord;

    gl_Position = projection * view * vec4(FragPos, 1.0);
}
