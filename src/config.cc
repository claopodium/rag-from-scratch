#include "config.h"
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

AppConfig AppConfig::LoadFromFile(const std::string& path){
    std::ifstream in(path);

    AppConfig cfg;  
    if (!in){
        std::cout << "Fail to load " << path << std::endl;
        return cfg;
    }

    json jsonData;
    try {
        in >> jsonData;
    } catch (const std::exception& e){
        std::cout << "Errors occur when .json is loaded. Default parameters are used." << std::endl;
        return cfg;
    }

    json models = jsonData.value("models", json::object());

    cfg.emb_model_path = models.value("embedding", cfg.emb_model_path);
    cfg.gen_model_path = models.value("generation", cfg.gen_model_path);
    cfg.n_gpu_layers = models.value("n_gpu_layers", cfg.n_gpu_layers);
    

    json document = jsonData.value("document", json::object());
    cfg.chunk_size = document.value("chunk_size", cfg.chunk_size);
    cfg.overlap_size = document.value("overlap_size", cfg.overlap_size);

    json retrieval = jsonData.value("retrieval", json::object());
    cfg.top_k = retrieval.value("top_k", cfg.top_k);
    
    json generation = jsonData.value("generation", json::object());
    cfg.gen_n_ctx = generation.value("n_ctx", cfg.gen_n_ctx);
    cfg.gen_n_batch = generation.value("n_batch", cfg.gen_n_batch);
    cfg.gen_n_ubatch = generation.value("n_ubatch", cfg.gen_n_ubatch);
    cfg.temperature = generation.value("temperature", cfg.temperature);
    cfg.top_k_sampler = generation.value("top_k", cfg.top_k_sampler);
    cfg.top_p = generation.value("top_p", cfg.top_p);
    cfg.max_output_tokens = generation.value("max_output_tokens", cfg.max_output_tokens);
    

    json embedding = jsonData.value("embedding", json::object());
    cfg.emb_n_ctx = embedding.value("n_ctx", cfg.emb_n_ctx);
    cfg.emb_n_batch = embedding.value("n_batch", cfg.emb_n_batch);
    cfg.emb_n_ubatch = embedding.value("n_ubatch", cfg.emb_n_ubatch);

    return cfg;
}