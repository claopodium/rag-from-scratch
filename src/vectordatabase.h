#pragma once
#include <cstddef>
#include <string>
#include <vector>
#include "chunk.h"

class VectorDatabase {
public:
    void insert(const Chunk& c);
    void insert(const std::vector<Chunk>& cs);

    std::vector<Chunk> search(const std::vector<float>& query_emb, int top_k) const;

    bool saveToDisk(const std::string& path) const;
    bool loadFromDisk(const std::string& path);

    std::size_t size() const;

private:
    std::vector<Chunk> chunks_;
};
