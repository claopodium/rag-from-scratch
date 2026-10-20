#include "generationengine.h"

GenerationEngine::GenerationEngine(const AppConfig& cfg)
    : LlamaModelBase(ModelMode::Generation, cfg){
        cfg_ = cfg;
    }

std::string GenerationEngine::buildPrompt(const std::string query, const std::vector<Chunk> chunks){
    std::string prompt = "";

    std::string systemPrompt = 
    "<|im_start|>system\n你是一个严谨的小说阅读助手。请你【必须并且只能】根据下面提供的[参考上下文]来回答用户的问题。如果上下文中有答案，请详细回答并引用出处；如果上下文中没有提到相关信息，请直接回答\"根据当前片段无法得知\"";

    std::string userPrompt = 
    "<|im_start|>user\n[参考上下文]";

    for (const Chunk& c : chunks){
        std::string prefix = "【片段 " + std::to_string(c.id) + " | " + c.metadata + "】";
        prefix += prefix + "\n" + c.text + "\n\n";
        userPrompt += prefix;
    }

    userPrompt += "用户的问题" + query + "<|im_end|>";

    prompt = systemPrompt + userPrompt + "<|im_start|>assistant";
    return prompt;
}

void GenerationEngine::generateStream(const std::string query, const std::vector<Chunk> chunks){
    llama_memory_clear(llama_get_memory(ctx_), true);

    std::string prompt = buildPrompt(query, chunks);

    std::vector<llama_token> tokens = tokenize(prompt, true, true);
    if (tokens.size() > llama_n_ctx(ctx_)){
        std::cerr << "The context exceeds the limitation";
        return;
    }

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
        return;
    }

    llama_batch_free(batch);

    llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(cfg_.top_k_sampler));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(cfg_.top_p, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(cfg_.temperature));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(0));


    for (int i = 0; i < cfg_.max_output_tokens; i++){
        llama_token new_token = llama_sampler_sample(smpl, ctx_, -1);

        if (llama_vocab_is_eog(vocab_, new_token)){
            break;
        }

        char buf[256];
        int32_t len = llama_token_to_piece(vocab_, new_token, buf, sizeof(buf), 0, false);
        if (len > 0) std::cout << std::string(buf, len) << std::flush;

        llama_batch b1 = llama_batch_get_one(&new_token, 1);
        if (llama_decode(ctx_, b1) != 0) break;
    }
    llama_sampler_free(smpl);
}