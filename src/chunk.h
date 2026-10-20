#pragma once

#include <iostream>
#include <string>
#include <cstddef>
#include <fstream>
#include <vector>
#include <cstdint>

struct Chunk
{
    std::size_t id = 0;
    std::string text;
    std::string metadata;
    std::vector<float> embedding;

    void serialize(std::ofstream &out) const;
    static Chunk deserialize(std::ifstream &in);
};
