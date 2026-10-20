#include "chunk.h"
#include "document.h"
#include <iostream>
#include <fstream>
#include <stdexcept>

DocumentProcessor::DocumentProcessor(int chunk_size, int overlap_size){
    chunk_size_ = chunk_size;
    overlap_size_ = overlap_size;
}

// 行首为「第」且含 章/节/回/卷，或行首为 Chapter；用 find 做字节精确匹配，避开 regex 的字节陷阱
static bool isChapterTitle(const std::string& line) {
    if (line.compare(0, 3, "第") == 0) {
        for (const char* m : {"章", "节", "回", "卷"})
            if (line.find(m, 3) != std::string::npos) return true;
    }
    if (line.compare(0, 7, "Chapter") == 0) return true;
    return false;
}

std::vector<Chunk> DocumentProcessor::processNovel(const std::string& filepath){
    std::vector<Chunk> chunks;

    std::ifstream in(filepath);
    if (!in) throw std::runtime_error("Fail to open the file");

    std::string line;
    std::string curr_title;
    std::string buffer = "";

    while (std::getline(in, line)){

        if (!line.empty() && line.back() == '\r'){
            line.pop_back();
        }
        
        if (isChapterTitle(line)){
            curr_title = line;
        }

        buffer += line;
        buffer += '\n';

        while (buffer.size() >= static_cast<size_t>(chunk_size_)){
            Chunk chunk;

            size_t cut = DocumentProcessor::nextUtf8Boundary(buffer, chunk_size_);
            
            size_t best = std::string::npos;
            for (const char* p : {"。","？","！","\n"}){
                size_t k = buffer.rfind(p, cut);
                if (k != std::string::npos && (best == std::string::npos || k > best)){
                    best = k;
                }
            }

            size_t pos = (best == std::string::npos) ? cut : best + 3;
            pos = (pos <= static_cast<size_t>(overlap_size_)) ? cut : pos;

            std::string txt = buffer.substr(0, pos);
            size_t del_pos = DocumentProcessor::nextUtf8Boundary(buffer, pos - overlap_size_);
            buffer.erase(0, del_pos);

            chunk.text = txt;
            chunk.id = next_id_++;
            chunk.metadata = curr_title;
            chunks.push_back(chunk);

        }
    }
    if (!buffer.empty()){
        Chunk c;
        c.id = next_id_ ++;
        c.text = buffer;
        c.metadata = curr_title;
        chunks.push_back(c);
    }

    return chunks;
}

size_t DocumentProcessor::nextUtf8Boundary(const std::string& s, size_t pos){
    for (size_t i = pos; i < s.size(); i++){
        unsigned char b = static_cast<unsigned char>(s[i]);
        if ((b & 0xc0) != 0x80){
            return i;
        }
    }
    return s.size();
}
