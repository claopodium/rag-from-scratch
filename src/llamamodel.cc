#include "llamamodel.h"
#include <stdexcept>

LlamaModelBase::LlamaModelBase(ModelMode model, const AppConfig& cfg){
    
    const bool is_emb = (model == ModelMode::Embedding);
    const std::string path = is_emb ? cfg.emb_model_path : cfg.gen_model_path;
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = cfg.n_gpu_layers;
    model_ = llama_model_load_from_file(path.c_str(), params);
    vocab_ = llama_model_get_vocab(model_);
    
    if (model_ == nullptr){
        throw std::runtime_error("Fail to load model parameters " + path);
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = is_emb ? cfg.emb_n_ctx : cfg.gen_n_ctx;
    ctx_params.n_batch = is_emb ? cfg.emb_n_batch : cfg.gen_n_batch;
    ctx_params.n_ubatch = is_emb ? cfg.emb_n_ubatch : cfg.gen_n_ubatch;

    ctx_params.embeddings = is_emb;
    if (is_emb){
        ctx_params.pooling_type = LLAMA_POOLING_TYPE_MEAN;
    }

    ctx_ = llama_init_from_model(model_, ctx_params);

    if (ctx_ == nullptr){
        llama_model_free(model_);
        model_ = nullptr;
        throw std::runtime_error("Fail to get context");
    }
}

LlamaModelBase::~LlamaModelBase(){
    if (ctx_){
        llama_free(ctx_);
    }
    
    if (model_){
        llama_model_free(model_);
    }
    
    ctx_ = nullptr;
    model_ = nullptr;
}

std::vector<llama_token> LlamaModelBase::tokenize(const std::string& text, bool add_special, bool parse_special) const{
    int32_t num = llama_tokenize(vocab_, text.c_str(), (int32_t)text.size(), nullptr, 0, add_special, parse_special);
    if (num < 0) num = -num;
    std::vector<llama_token> tokens(num);
    int32_t true_num = llama_tokenize(vocab_, text.c_str(), (int32_t)text.size(), tokens.data(), (int32_t)tokens.size(), add_special, parse_special);
    if (true_num < 0) num = 0;
    tokens.resize(true_num);
    return tokens;
}