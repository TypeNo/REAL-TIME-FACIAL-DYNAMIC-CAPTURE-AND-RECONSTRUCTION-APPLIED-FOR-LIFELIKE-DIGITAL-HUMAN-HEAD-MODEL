//#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <iostream>
#include "TextureLoader.hpp"
#include <fstream>

//LoadTextureFromFile
GLuint LoadTextureFromFile(const char* filename)
{
    if (!std::filesystem::exists(filename)) {
        std::cout << "[ERROR] File does not exist: " << filename << std::endl;
    }
    
    std::cout << "[DEBUG] Attempting to load texture from: " << filename << std::endl;

    int width, height, channels;
    unsigned char* data = stbi_load(filename, &width, &height, &channels, 4); // Force RGBA
    if (data == nullptr)
    {
        std::cout << "[ERROR] Failed to load image: " << filename << std::endl;
        return 0;
    }

    std::cout << "[INFO] Image loaded: " << filename << " (" << width << "x" << height 
              << ", channels requested: 4, original channels: " << channels << ")" << std::endl;

    GLuint textureID = 0;
    glGenTextures(1, &textureID);
    if (textureID == 0)
    {
        std::cout << "[ERROR] glGenTextures failed!" << std::endl;
        stbi_image_free(data);
        return 0;
    }

    glBindTexture(GL_TEXTURE_2D, textureID);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cout << "[ERROR] OpenGL error after glTexImage2D: " << err << std::endl;
    }

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glBindTexture(GL_TEXTURE_2D, 0); // Unbind for safety
    stbi_image_free(data);

    std::cout << "[DEBUG] Texture loaded successfully with ID: " << textureID << std::endl;

    return textureID;
}

GLuint LoadTextureFromMemory(unsigned char* data, int size) {
    int width, height, nrChannels;
    //stbi_set_flip_vertically_on_load(true); // Flip if needed

    std::cout << "[LoadTextureFromMemory] Loading texture from memory... Size: " << size << " bytes\n";

    std::ofstream outFile("debug_output.png", std::ios::binary);
    outFile.write(reinterpret_cast<const char*>(data), size);
    outFile.close();

    unsigned char* imageData = stbi_load_from_memory(data, size, &width, &height, &nrChannels, 0);
    if (!imageData) {
        std::cerr << "[LoadTextureFromMemory] Failed to load texture from memory\n";
        return 0;
    }

    std::cout << "[LoadTextureFromMemory] Image loaded. Width: " << width 
              << ", Height: " << height 
              << ", Channels: " << nrChannels << "\n";

    GLenum format;
    if (nrChannels == 1) format = GL_RED;
    else if (nrChannels == 3) format = GL_RGB;
    else if (nrChannels == 4) format = GL_RGBA;
    else {
        std::cerr << "[LoadTextureFromMemory] Unsupported number of channels: " << nrChannels << "\n";
        stbi_image_free(imageData);
        return 0;
    }

    //std::cout << "[LoadTextureFromMemory] First 10 pixel RGB values:\n";

    /*int maxPixelsToPrint = width * height;
    int componentsPerPixel = nrChannels;

    for (int i = 0; i < maxPixelsToPrint && i * componentsPerPixel < width * height * componentsPerPixel; ++i) {
        int baseIndex = i * componentsPerPixel;
        unsigned char r = imageData[baseIndex + 0];
        unsigned char g = (componentsPerPixel > 1) ? imageData[baseIndex + 1] : 0;
        unsigned char b = (componentsPerPixel > 2) ? imageData[baseIndex + 2] : 0;
        unsigned char a = (componentsPerPixel > 3) ? imageData[baseIndex + 3] : 255;

        std::cout << "Pixel " << i << ": (R,G,B,A) = ("
                << static_cast<int>(r) << ", "
                << static_cast<int>(g) << ", "
                << static_cast<int>(b) << ", "
                << static_cast<int>(a) << ")\n";
    }*/

    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);

    std::cout << "[LoadTextureFromMemory] Creating OpenGL texture ID: " << textureID << "\n";

    glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, imageData);
    glGenerateMipmap(GL_TEXTURE_2D);

    // Texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);	
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    stbi_image_free(imageData);
    std::cout << "[LoadTextureFromMemory] Texture loaded and OpenGL texture created successfully.\n";
    return textureID;
}

