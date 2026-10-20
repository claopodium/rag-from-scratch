#include "chunk.h"
#include <fstream>
#include <cstdint>
#include <vector>

void Chunk::serialize(std::ofstream &out) const{
    // write ID
    out.write(reinterpret_cast<const char*>(&id), (uint64_t)sizeof(id));

    // write text
    uint64_t txt_len = text.size();
    out.write(reinterpret_cast<const char*>(&txt_len), (uint64_t)sizeof(text.size()));
    out.write(text.data(), text.size());

    //write metadata
    uint64_t meta_len = metadata.size();
    out.write(reinterpret_cast<const char*>(&meta_len), (uint64_t)sizeof(metadata.size()));
    out.write(metadata.data(), metadata.size());

    //write embedding
    uint64_t emb_len = embedding.size();
    out.write(reinterpret_cast<const char*>(&emb_len), (uint64_t)sizeof(embedding.size()));
    out.write(reinterpret_cast<const char*>(embedding.data()), embedding.size() * sizeof(float));
    
}


Chunk Chunk::deserialize(std::ifstream &in){
    Chunk c;

    //read id
    in.read((char*)&c.id, sizeof(uint64_t));

    //read txt
    uint64_t txt_len;
    in.read((char*)&txt_len, sizeof(uint64_t));
    std::string txt(txt_len, '.');
    in.read(txt.data(), txt_len);
    c.text = txt;

    // read meta
    uint64_t meta_len;
    in.read((char*)&meta_len, sizeof(uint64_t));
    std::string meta(meta_len, '.');
    in.read(meta.data(), meta_len);
    c.metadata = meta;

    uint64_t emb_len;
    in.read((char*)&emb_len, sizeof(uint64_t));
    std::vector<float> emb(emb_len);
    in.read((char*)emb.data(), emb_len * sizeof(float));
    c.embedding = emb;

    return c;
}