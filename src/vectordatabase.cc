#include "vectordatabase.h"
#include <vector>
#include <queue>
#include <cmath>
#include <fstream>
#include <iostream>

float dot(const std::vector<float>& A, const std::vector<float>& B);

struct QandK{
    float cos;
    Chunk c;
};

struct compare{
    bool operator()(QandK a, QandK b){
        return a.cos > b.cos;
    }
};


void VectorDatabase::insert(const Chunk& c){
    chunks_.push_back(c);
}
void VectorDatabase::insert(const std::vector<Chunk>& cs){
    for (const Chunk& c : cs){
        chunks_.push_back(c);
    }
}

std::vector<Chunk> VectorDatabase::search(const std::vector<float>& query_emb, int top_k) const{
    std::priority_queue<QandK, std::vector<QandK>, compare> heap;
    std::vector<Chunk> result(top_k);
    int cnt = 0;
    float cosine;

    for (const Chunk& c : chunks_){
        if (c.embedding.size() != query_emb.size()){
            continue;
        }
        float dot_prod = dot(c.embedding, query_emb);
        float modu_A = std::sqrt(dot(c.embedding, c.embedding));
        float modu_B = std::sqrt(dot(query_emb, query_emb));

        if (modu_A == 0 || modu_B == 0){
            cosine = 0.0;
        }else{
            cosine = dot_prod / (modu_A * modu_B);
        }
        QandK qandk;
        qandk.c = c;
        qandk.cos = cosine;
        
        if (cnt >= top_k){
            if (heap.top().cos < cosine){
                heap.pop();
                heap.push(qandk);
            }
        }else{
            heap.push(qandk);
            cnt++;
        }
    }

    for (int i = 0; i < top_k && !heap.empty(); i++){
        Chunk tmp = heap.top().c;
        heap.pop();
        result[top_k-i-1] = tmp;
    }

    return result;
}

bool VectorDatabase::saveToDisk(const std::string& path) const{
    std::ofstream outfile(path, std::ios::binary);
    if (!outfile){
        std::cerr << "[vdb] 无法写入数据库文件: " << path << "\n";
        return false;
    }

    size_t magic = 0x4D524147;
    size_t v = 1;
    outfile.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    outfile.write(reinterpret_cast<const char*>(&v), sizeof(v));

    size_t num = chunks_.size();
    outfile.write(reinterpret_cast<const char*>(&num), sizeof(num));

    for (size_t i = 0; i < num; i++){
        chunks_[i].serialize(outfile);
    }
    return true;
}

bool VectorDatabase::loadFromDisk(const std::string& path){
    std::ifstream infile(path, std::ios::binary);
    if (!infile){
        std::cerr << "[vdb] 无法打开数据库文件: " << path << "\n";
        return false;
    }

    size_t magic = 0;
    infile.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != 0x4D524147){
        std::cerr << "[vdb] 魔数不匹配（该文件不是本程序的数据库）\n";
        return false;
    }

    size_t v = 0;
    infile.read(reinterpret_cast<char*>(&v), sizeof(v));
    if (v != 1) {
        std::cerr << "[vdb] 版本号不匹配（期望 1，实际 " << v << "）\n";
        return false;
    }

    size_t len;
    infile.read((char*)(&len), sizeof(len));

    chunks_.clear();
    for (size_t i = 0; i < len; i++){
        Chunk c = Chunk::deserialize(infile);
        chunks_.push_back(c);
    }
    
    return true;
}

std::size_t VectorDatabase::size() const{
    return chunks_.size();
}

float dot(const std::vector<float>& A, const std::vector<float>& B){
    float prod = 0.0;
    if (A.size() != B.size()) return 0.0;
    for (std::size_t i = 0; i < A.size(); i++){
        prod += A[i] * B[i];
    }
    return prod;
}

