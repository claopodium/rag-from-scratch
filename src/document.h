#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "chunk.h"

class DocumentProcessor {
public:
    DocumentProcessor(int chunk_size, int overlap_size);

    std::vector<Chunk> processNovel(const std::string& filepath);

    static size_t nextUtf8Boundary(const std::string& s, size_t pos);

private:
    int chunk_size_;
    int overlap_size_;
    size_t next_id_ = 0;
};
