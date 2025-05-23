#ifndef TEXTURE_LOADER_HPP
#define TEXTURE_LOADER_HPP

#include <glad/glad.h>
#include <filesystem>

GLuint LoadTextureFromFile(const char* path);
GLuint LoadTextureFromMemory(unsigned char* data, int size);  // <-- Add this

#endif