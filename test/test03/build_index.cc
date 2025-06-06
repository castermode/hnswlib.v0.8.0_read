// 给定一个向量文件，使用 hnswlib为其建立索引
// 命令行参数：
// --input: 输入的向量文件
// --output: 输出的索引文件
// --index: 索引类型（hnsw, brute）
// --distance: 距离类型（l2, cosine, inner）(欧几里得距离，余弦距离，内积距离)
// --dim: 向量维度
// --threads: 线程数，默认为cpu核心数

// 输入的向量文件格式：
// 每行一个向量，向量之间用,分隔
// 数据为float类型，有正有负
// 数据样例：
// 0.0423799,-0.052559,-0.273673,-0.720855,0.405319,-0.107581,0.902083,-0.214004,0.0193734,0.974551
// -0.424708,-0.0696349,0.915268,-0.695467,0.127032,-0.788173,0.881132,0.143714,0.868893,-0.528071
// -0.558684,-0.493159,-0.366166,0.690315,0.600875,-0.357938,0.444509,0.500975,0.0225797,0.351535
// 0.344367,-0.623892,-0.024657,-0.541492,0.522897,0.637507,0.804872,-0.366504,0.562115,0.33685

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <thread>
#include <memory>
#include "hnswlib/hnswlib/hnswlib.h"
#include <getopt.h>

void printUsage() {
    std::cout << "用法：" << std::endl;
    std::cout << "--input: 输入的向量文件" << std::endl;
    std::cout << "--output: 输出的索引文件" << std::endl;
    std::cout << "--index: 索引类型（hnsw, brute）" << std::endl;
    std::cout << "--distance: 距离类型（l2, cosine, inner）" << std::endl;
    std::cout << "--dim: 向量维度" << std::endl;
    std::cout << "--threads: 线程数，默认为cpu核心数" << std::endl;
    std::cout << "示例：" << std::endl;
    std::cout << "./build_index --input=./data.dim10.txt --output=data.dim10.index --index=hnsw --distance=cosine --dim=10 --threads 4" << std::endl;
}

int main(int argc, char** argv) {
    // 解析命令行参数
    std::string inputFile;
    std::string outputFile;
    std::string indexType = "hnsw";
    std::string distanceType = "l2";
    int dim = 0;
    int numThreads = std::thread::hardware_concurrency();

    static struct option long_options[] = {
        {"input", required_argument, 0, 'i'},
        {"output", required_argument, 0, 'o'},
        {"index", required_argument, 0, 'x'},
        {"distance", required_argument, 0, 'd'},
        {"dim", required_argument, 0, 'm'},
        {"threads", required_argument, 0, 't'},
        {0, 0, 0, 0}
    };

    int opt;
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "i:o:x:d:m:t:", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'i':
                inputFile = optarg;
                break;
            case 'o':
                outputFile = optarg;
                break;
            case 'x':
                indexType = optarg;
                break;
            case 'd':
                distanceType = optarg;
                break;
            case 'm':
                dim = std::stoi(optarg);
                break;
            case 't':
                numThreads = std::stoi(optarg);
                break;
            default:
                printUsage();
                return 1;
        }
    }

    // 检查必需参数
    if (inputFile.empty() || outputFile.empty() || dim <= 0) {
        std::cerr << "错误：必须指定输入文件、输出文件和维度！" << std::endl;
        printUsage();
        return 1;
    }

    if (indexType != "hnsw" && indexType != "brute") {
        std::cerr << "错误：索引类型必须是 hnsw 或 brute" << std::endl;
        return 1;
    }

    if (distanceType != "l2" && distanceType != "cosine" && distanceType != "inner") {
        std::cerr << "错误：距离类型必须是 l2、cosine 或 inner" << std::endl;
        return 1;
    }

    std::cout << "参数：" << std::endl;
    std::cout << "  输入文件: " << inputFile << std::endl;
    std::cout << "  输出文件: " << outputFile << std::endl;
    std::cout << "  索引类型: " << indexType << std::endl;
    std::cout << "  距离类型: " << distanceType << std::endl;
    std::cout << "  维度: " << dim << std::endl;
    std::cout << "  线程数: " << numThreads << std::endl;

    // 读取向量文件
    std::ifstream infile(inputFile);
    if (!infile.is_open()) {
        std::cerr << "无法打开输入文件：" << inputFile << std::endl;
        return 1;
    }

    std::vector<std::vector<float>> vectors;
    std::string line;
    while (std::getline(infile, line)) {
        std::vector<float> vec;
        std::stringstream ss(line);
        std::string value;
        while (std::getline(ss, value, ',')) {
            vec.push_back(std::stof(value));
        }

        if (vec.size() != dim) {
            std::cerr << "错误：向量维度不匹配。期望 " << dim << "，实际 " << vec.size() << std::endl;
            return 1;
        }

        vectors.push_back(vec);
    }
    infile.close();

    size_t num_elements = vectors.size();
    std::cout << "读取了 " << num_elements << " 个向量" << std::endl;

    if (num_elements == 0) {
        std::cerr << "错误：输入文件为空" << std::endl;
        return 1;
    }

    // 创建距离空间
    std::unique_ptr<hnswlib::SpaceInterface<float>> space;
    if (distanceType == "l2") {
        space.reset(new hnswlib::L2Space(dim));
    } else if (distanceType == "cosine") {
        space.reset(new hnswlib::InnerProductSpace(dim));
    } else if (distanceType == "inner") {
        space.reset(new hnswlib::InnerProductSpace(dim));
    }

    // 创建索引
    std::unique_ptr<hnswlib::AlgorithmInterface<float>> index;
    const int M = 16;                // 节点连接数，影响内存消耗
    const int ef_construction = 200; // 索引构建速度/搜索速度的权衡参数

    if (indexType == "hnsw") {
        auto hnsw_index = new hnswlib::HierarchicalNSW<float>(space.get(), num_elements, M, ef_construction);
        index.reset(hnsw_index);
    } else if (indexType == "brute") {
        auto bf_index = new hnswlib::BruteforceSearch<float>(space.get(), dim);
        index.reset(bf_index);
    }

    // 添加向量到索引
    for (size_t i = 0; i < num_elements; i++) {
        index->addPoint(vectors[i].data(), i);
        
        // 显示进度
        if (i % 10000 == 0 || i == num_elements - 1) {
            std::cout << "已添加 " << i + 1 << "/" << num_elements << " 个向量" << std::endl;
        }
    }

    // 保存索引
    if (indexType == "hnsw") {
        auto hnsw_index = dynamic_cast<hnswlib::HierarchicalNSW<float>*>(index.get());
        hnsw_index->saveIndex(outputFile);
    } else if (indexType == "brute") {
        auto bf_index = dynamic_cast<hnswlib::BruteforceSearch<float>*>(index.get());
        bf_index->saveIndex(outputFile);
    }

    std::cout << "索引已保存到：" << outputFile << std::endl;

    return 0;
}

