#include "config.h"
#include "mragapp.h"
#include <iostream>
#include <string>

static void printUsage() {
    std::cout << "Usage:\n"
              << "  mrag --ingest <novel.txt> <database.bin>   # build knowledge base\n"
              << "  mrag --chat   <database.bin>               # ask questions\n";
}

int main(int argc, char* argv[]) {
    llama_log_set([](ggml_log_level, const char*, void*) {}, nullptr);   // 屏蔽 llama 底层日志

    if (argc < 2) { printUsage(); return 1; }

    const std::string mode = argv[1];
    try {
        AppConfig cfg = AppConfig::LoadFromFile("config.json");
        MragApp app(cfg);

        if (mode == "--ingest") {
            if (argc < 4) { printUsage(); return 1; }
            app.buildKnowledgeBase(argv[2], argv[3]);
        } else if (mode == "--chat") {
            if (argc < 3) { printUsage(); return 1; }
            app.chatLoop(argv[2]);
        } else {
            printUsage();
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
