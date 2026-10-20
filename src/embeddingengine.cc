#include "embeddingengine.h"
#include <algorithm>
#include <iostream>

EmbeddingEngine::EmbeddingEngine(const AppConfig& cfg)
    :LlamaModelBase(ModelMode::Embedding, cfg){
    ctx_limit_ = std::min(cfg.emb_n_ctx, cfg.emb_n_batch);
}

std::string EmbeddingEngine::sanitizeUtf8(const std::string& text){
    std::string out = "";
    std::string charactor = "";
    int L = 0;
    for (size_t i = 0; i < text.size(); i++){
        if ((text[i] & 0xC0) == 0x80 && (L != 0)){
            charactor += text[i];  
            L--;
        }

        if ((text[i] & 0x80) == 0x00){
            charactor = text[i];
            L = 0;
        }else if((text[i] & 0xE0) == 0xC0){
            charactor = text[i];
            L = 1;
        }else if((text[i] & 0xF0) == 0xE0){
            charactor = text[i];
            L = 2;
        }else if((text[i] & 0xF8) == 0xF0){
            charactor = text[i];
            L = 3;
        }

        if (L == 0){
            out += charactor;
            charactor = "";
        }
    }
    return out;
}

std::vector<float> EmbeddingEngine::generateEmbedding(const std::string& text){
    std::string clean_text = sanitizeUtf8(text);
    std::vector<llama_token> tokens = tokenize(clean_text, true, false);
    if (tokens.size() > ctx_limit_){
        tokens.resize(ctx_limit_);
    }

    llama_memory_clear(llama_get_memory(ctx_), true);

    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); i++){
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0]= 0;
        batch.logits[i] = (i == tokens.size()-1) ? 1 : 0;
    }
    batch.n_tokens = tokens.size();

    if (llama_decode(ctx_, batch) != 0){
        std::cerr << "Fail to decode";
        llama_batch_free(batch);
        return {};
    }

    const float* p = llama_get_embeddings_seq(ctx_, 0);
    if (p == nullptr) p = llama_get_embeddings(ctx_);
    if (p == nullptr) {
        llama_batch_free(batch);
        return {};
    }
    int32_t d = llama_model_n_embd(model_);
    std::vector<float> out(p, p+d);
    llama_batch_free(batch);
    return out;

}