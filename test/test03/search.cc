// 载入向量索引文件，使用 hnswlib进行搜索
// 命令行参数：
// --index=索引文件
// --meta=索引文件元信息文件
// --v=给定一个向量，查询与其最相似的k个向量
// --id=给定一个id，先查询其对应的向量，再查询与其最相似的k个向量
// --k=搜索结果数量，默认10
// --threads=线程数，默认为cpu核心数
// --help 显示帮助信息

// 索引文件元信息文件格式
// index:索引类型（hnsw, brute）
// distance:距离类型（l2, cos, inner）
// dim:向量维度

// 索引文件元信息文件样例：
// index:hnsw
// distance:cos
// dim:10

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <thread>
#include "../../hnswlib/hnswlib/hnswlib.h"

void printUsage() {
    std::cout << "用法：" << std::endl;
    std::cout << "--index=索引文件" << std::endl;
    std::cout << "--meta=索引文件元信息文件" << std::endl;
    std::cout << "--v=给定一个向量，查询与其最相似的k个向量" << std::endl;
    std::cout << "--id=给定一个id，先查询其对应的向量，再查询与其最相似的k个向量" << std::endl;
    std::cout << "--threads: 线程数，默认为cpu核心数" << std::endl;
    std::cout << "示例：" << std::endl;
    std::cout << "./search --index=data.dim10.txt.index --meta=data.dim10.txt.meta --v=0.1,0.2,0.3,0.4 --k=5" << std::endl;
    std::cout << "./search --index=data.dim10.txt.index --meta=data.dim10.txt.meta --id=1 --k=5 --threads=4" << std::endl;
}

struct Config {
    std::string indexFile;
    std::string metaFile;
    std::vector<float> queryVector;
    int queryId = -1;
    int k = 10;
    int threads = std::thread::hardware_concurrency();
};

struct MetaInfo {
    std::string indexType;
    std::string distanceType;
    int dim = 0;
};

// 解析命令行参数
Config parseArgs(int argc, char** argv) {
    Config config;
    
    // 检查是否有帮助参数
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printUsage();
            exit(0);
        }
    }
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg.substr(0, 8) == "--index=") {
            config.indexFile = arg.substr(8);
        } else if (arg.substr(0, 7) == "--meta=") {
            config.metaFile = arg.substr(7);
        } else if (arg.substr(0, 4) == "--v=") {
            std::string vectorStr = arg.substr(4);
            std::stringstream ss(vectorStr);
            std::string item;
            while (std::getline(ss, item, ',')) {
                config.queryVector.push_back(std::stof(item));
            }
        } else if (arg.substr(0, 5) == "--id=") {
            config.queryId = std::stoi(arg.substr(5));
        } else if (arg.substr(0, 4) == "--k=") {
            config.k = std::stoi(arg.substr(4));
        } else if (arg.substr(0, 10) == "--threads=") {
            config.threads = std::stoi(arg.substr(10));
        }
    }
    return config;
}

// 解析元信息文件
MetaInfo parseMetaFile(const std::string& metaFile) {
    MetaInfo meta;
    std::ifstream file(metaFile);
    if (!file.is_open()) {
        std::cerr << "无法打开元信息文件: " << metaFile << std::endl;
        exit(1);
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        size_t pos = line.find(':');
        if (pos != std::string::npos) {
            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 1);
            
            if (key == "index") {
                meta.indexType = value;
            } else if (key == "distance") {
                meta.distanceType = value;
            } else if (key == "dim") {
                meta.dim = std::stoi(value);
            }
        }
    }

    // 打印元信息
    std::cout << "索引类型: " << meta.indexType << std::endl;
    std::cout << "距离类型: " << meta.distanceType << std::endl;
    std::cout << "向量维度: " << meta.dim << std::endl;
    
    return meta;
}

template<typename T>
void printTopK(std::priority_queue<std::pair<float, hnswlib::labeltype>>& result, int k) {
    std::cout << "查询结果 (ID, 距离):" << std::endl;
    std::vector<std::pair<float, hnswlib::labeltype>> sortedResults;
    
    while (!result.empty()) {
        sortedResults.push_back(result.top());
        result.pop();
    }
    
    // 反转结果使其按距离升序排列
    std::reverse(sortedResults.begin(), sortedResults.end());
    
    for (int i = 0; i < std::min(k, (int)sortedResults.size()); i++) {
        std::cout << sortedResults[i].second << "\t" << sortedResults[i].first << std::endl;
    }
}

int main(int argc, char** argv) {
    // 如果没有参数，显示使用说明
    if (argc == 1) {
        printUsage();
        return 0;
    }
    
    // 解析命令行参数
    Config config = parseArgs(argc, argv);
    
    if (config.indexFile.empty() || config.metaFile.empty()) {
        std::cerr << "请提供索引文件和元信息文件" << std::endl;
        printUsage();
        return 1;
    }
    
    if (config.queryVector.empty() && config.queryId < 0) {
        std::cerr << "请提供查询向量(--v)或查询ID(--id)" << std::endl;
        printUsage();
        return 1;
    }
    
    // 解析元信息文件
    MetaInfo meta = parseMetaFile(config.metaFile);
    
    if (meta.dim <= 0) {
        std::cerr << "元信息文件中的维度无效" << std::endl;
        return 1;
    }
    
    // 检查查询向量的维度是否正确
    if (!config.queryVector.empty() && static_cast<int>(config.queryVector.size()) != meta.dim) {
        std::cerr << "查询向量维度 (" << config.queryVector.size() 
                 << ") 与索引维度 (" << meta.dim << ") 不匹配" << std::endl;
        return 1;
    }
    
    try {
        // 创建适当的空间
        hnswlib::SpaceInterface<float>* space = nullptr;
        
        if (meta.distanceType == "l2") {
            space = new hnswlib::L2Space(meta.dim);
        } else if (meta.distanceType == "cos" || meta.distanceType == "inner") {
            space = new hnswlib::InnerProductSpace(meta.dim);
        } else {
            std::cerr << "不支持的距离类型: " << meta.distanceType << std::endl;
            return 1;
        }
        
        // 加载索引
        hnswlib::AlgorithmInterface<float>* index = nullptr;
        
        if (meta.indexType == "hnsw") {
            index = new hnswlib::HierarchicalNSW<float>(space, config.indexFile);
        } else if (meta.indexType == "brute") {
            index = new hnswlib::BruteforceSearch<float>(space, config.indexFile);
        } else {
            std::cerr << "不支持的索引类型: " << meta.indexType << std::endl;
            delete space;
            return 1;
        }
        
        // 设置查询时的线程数
        if (meta.indexType == "hnsw") {
            ((hnswlib::HierarchicalNSW<float>*)index)->setEf(config.k * 2);
            ((hnswlib::HierarchicalNSW<float>*)index)->ef_ = config.k * 2;
        }
        
        // 执行查询
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result;
        
        if (config.queryId >= 0) {
            // 通过ID查询
            if (meta.indexType == "hnsw") {
                // 对于HNSW索引，先检查ID是否存在
                auto* hnsw_index = (hnswlib::HierarchicalNSW<float>*)index;
                if (!hnsw_index->label_lookup_.count(config.queryId)) {
                    std::cerr << "ID " << config.queryId << " 不存在于索引中" << std::endl;
                    delete index;
                    delete space;
                    return 1;
                }
                
                // 获取ID对应的向量
                hnswlib::tableint internalId = hnsw_index->label_lookup_[config.queryId];
                float* vector = (float*)hnsw_index->getDataByInternalId(internalId);

                // 打印向量
                std::cout << "查询向量: ";
                for (int i = 0; i < meta.dim; i++) {
                    std::cout << vector[i] << " ";
                }
                std::cout << std::endl;
                
                // 使用该向量查询
                result = index->searchKnn(vector, config.k);
            } else if (meta.indexType == "brute") {
                // 对于暴力索引，暂不支持通过ID查询
                std::cerr << "暴力索引不支持通过ID查询" << std::endl;
                delete index;
                delete space;
                return 1;
            }
        } else {
            // 通过向量查询
            result = index->searchKnn(config.queryVector.data(), config.k);
        }
        
        // 输出结果
        printTopK<float>(result, config.k);
        
        // 释放资源
        delete index;
        delete space;
        
    } catch (const std::exception& e) {
        std::cerr << "发生错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}




