#pragma once

#include "llamamodel.h"
#include "chunk.h"

class GenerationEngine : public LlamaModelBase{
public:
    explicit GenerationEngine(const AppConfig& cfg);
    void generateStream(const std::string query, const std::vector<Chunk> chunks);
    std::string buildPrompt(const std::string query, const std::vector<Chunk> chunks);

private:
    AppConfig cfg_;
};