#pragma once
#include <string>
#include <vector>
#include "llama.h"
#include "config.h"

enum class ModelMode { Embedding, Generation };

class LlamaModelBase {
public:
    LlamaModelBase(ModelMode mode, const AppConfig& cfg);
    virtual ~LlamaModelBase();

    LlamaModelBase(const LlamaModelBase&)            = delete;   // 禁拷贝构造
    LlamaModelBase& operator=(const LlamaModelBase&) = delete;   // 禁拷贝赋值

protected:
    std::vector<llama_token> tokenize(const std::string& text, bool add_special, bool parse_special) const;

    const llama_vocab* vocab_ = nullptr;
    llama_model*   model_ = nullptr;
    llama_context* ctx_   = nullptr;
};
