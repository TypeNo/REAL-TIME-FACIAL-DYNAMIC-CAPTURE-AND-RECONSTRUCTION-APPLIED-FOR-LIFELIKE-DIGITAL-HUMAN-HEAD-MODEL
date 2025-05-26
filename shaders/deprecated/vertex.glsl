#version 430 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;
layout (location = 3) in vec4 aBoneIDs;
layout (location = 4) in vec4 aWeights;


uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

layout(std430, binding = 0) buffer MorphWeightsBuffer {
    float morphWeights[];
};
uniform sampler2D morphTexture;
uniform int numMorphs;
//uniform float morphWeights[numMorphs]; // or however many targets you support



//const int MAX_BONES = 100;
//uniform mat4 bones[MAX_BONES];

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;

void main() {
    //mat4 skinMatrix = aWeights.x * bones[int(aBoneIDs.x)] +
    //                  aWeights.y * bones[int(aBoneIDs.y)] +
    //                  aWeights.z * bones[int(aBoneIDs.z)] +
    //                  aWeights.w * bones[int(aBoneIDs.w)];

    vec3 morphed = aPos
    for (int i = 0; i < numMorphs; ++i) {
        vec3 delta = texelFetch(morphTexture, ivec2(gl_VertexID, i), 0).xyz;
        morphed += morphWeights[i] * delta;
    }
    //vec4 skinnedPos = skinMatrix * vec4(aPos, 1.0);
    //vec3 skinnedNormal = mat3(skinMatrix) * aNormal;

    //FragPos = vec3(model * skinnedPos);
    FragPos = vec3(model * vec4(morphed, 1.0))
    //Normal = mat3(transpose(inverse(model))) * skinnedNormal;
    Normal = mat3(transpose(inverse(model))) * aNormal;
    TexCoord = aTexCoord;
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
