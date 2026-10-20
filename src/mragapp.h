#pragma once

#include "config.h"
#include "embeddingengine.h"
#include "generationengine.h"
#include "document.h"
#include "vectordatabase.h"
#include "llama.h"
#include "chunk.h"

class MragApp{
private:
    struct BackendGuard{
        BackendGuard();
        ~BackendGuard();
    };

    BackendGuard guard_;
    AppConfig cfg_;

    EmbeddingEngine emb_;
    GenerationEngine gen_;
    
public:
    explicit MragApp(const AppConfig& cfg);
    void buildKnowledgeBase(const std::string& txt_path, const std::string& db_path);
    void chatLoop(const std::string& db_path);
};
