#include "mragapp.h"
#include <fstream>
#include <iostream>

MragApp::BackendGuard::BackendGuard(){
    llama_backend_init();
}

MragApp::BackendGuard::~BackendGuard(){
    llama_backend_free();
}

MragApp::MragApp(const AppConfig& cfg)
    : cfg_(cfg), emb_(cfg), gen_(cfg){

}

void MragApp::buildKnowledgeBase(const std::string& txt_path, const std::string& db_path){
    DocumentProcessor doc_proc = DocumentProcessor(cfg_.chunk_size, cfg_.overlap_size);

    std::vector<Chunk> chunks = doc_proc.processNovel(txt_path);
    VectorDatabase vec_db;

    std::cout << "Ingesting " << chunks.size() << " chunks from " << txt_path << " ...\n";
    for (std::size_t i = 0; i < chunks.size(); ++i){
        chunks[i].embedding = emb_.generateEmbedding(chunks[i].text);
        vec_db.insert(chunks[i]);
        if ((i + 1) % 20 == 0 || i + 1 == chunks.size())
            std::cout << "\r  embedded " << (i + 1) << "/" << chunks.size() << std::flush;
    }
    std::cout << "\n";

    vec_db.saveToDisk(db_path);
    std::cout << "Saved database: " << db_path << "\n";
}

void MragApp::chatLoop(const std::string& db_path){
    VectorDatabase vec_db;
    if (!vec_db.loadFromDisk(db_path)){
        std::cerr << "Failed to load database: " << db_path << "\n";
        return;
    }

    std::cout << "Loaded knowledge base: " << db_path
              << "  (" << vec_db.size() << " chunks)\n"
              << "Type a question, or 'exit' / 'quit' to leave.\n\n";

    std::string question;
    while (true){
        std::cout << "> " << std::flush;
        if (!std::getline(std::cin, question)) break;          // EOF (Ctrl+D)
        if (question == "exit" || question == "quit") break;
        if (question.empty()) continue;

        std::vector<float> question_emb = emb_.generateEmbedding(question);
        std::vector<Chunk> hits = vec_db.search(question_emb, cfg_.top_k);

        std::cout << "A: " << std::flush;
        gen_.generateStream(question, hits);
        std::cout << "\n\n";
    }

    std::cout << "Bye.\n";
}