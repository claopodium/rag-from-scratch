#include "llamamodel.h"

class EmbeddingEngine : public LlamaModelBase {
public:
    explicit EmbeddingEngine(const AppConfig& cfg);
    std::vector<float> generateEmbedding(const std::string& text);
    static std::string sanitizeUtf8(const std::string& text);

private:
    int32_t ctx_limit_;
};
