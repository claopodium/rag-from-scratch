#pragma once

#include <string>

struct AppConfig {
    std::string emb_model_path = "./models/bge-small-zh-v1.5-f16.gguf";
    std::string gen_model_path = "./models/qwen2.5-1.5b-instruct-q4_k_m.gguf";

    int n_gpu_layers = 0;

    int chunk_size   = 500;
    int overlap_size = 50;

    int top_k = 3;

    int   gen_n_ctx         = 4096;
    int   gen_n_batch       = 4096;
    int   gen_n_ubatch      = 512;
    int   max_output_tokens = 512;
    float temperature       = 0.7f;
    int   top_k_sampler     = 40;
    float top_p             = 0.9f;

    int emb_n_ctx    = 512;
    int emb_n_batch  = 512;
    int emb_n_ubatch = 512;

    static AppConfig LoadFromFile(const std::string& path);
};
