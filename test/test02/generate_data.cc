// 生成测试的向量数据
// 通过命令行参数传入参数
// --dim=维度
// --count=数据量
// --min=最小值, 默认0
// --max=最大值, 默认1
// 直接将生成浮点向量数据打印输出到标准输出


#include <iostream>
#include <vector>
#include <random>
#include <cstdlib>
#include <string>
#include <unordered_map>

// 解析命令行参数，格式为 --key=value
std::unordered_map<std::string, std::string> parseArgs(int argc, char** argv) {
    std::unordered_map<std::string, std::string> args;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg.substr(0, 2) == "--") {
            size_t pos = arg.find('=');
            if (pos != std::string::npos) {
                std::string key = arg.substr(2, pos - 2);
                std::string value = arg.substr(pos + 1);
                args[key] = value;
            }
        }
    }
    return args;
}

int main(int argc, char** argv) {
    // 解析命令行参数
    auto args = parseArgs(argc, argv);

    // 检查必须参数
    if (args.find("dim") == args.end() || args.find("count") == args.end()) {
        std::cerr << "用法: " << argv[0] << " --dim=维度 --count=数据量 [--min=最小值] [--max=最大值]" << std::endl;
        return 1;
    }

    // 获取参数值
    int dim = std::stoi(args["dim"]);
    int count = std::stoi(args["count"]);
    float min = args.find("min") != args.end() ? std::stof(args["min"]) : 0.0f;
    float max = args.find("max") != args.end() ? std::stof(args["max"]) : 1.0f;

    // 检查参数有效性
    if (dim <= 0 || count <= 0) {
        std::cerr << "错误: 维度和数据量必须为正整数" << std::endl;
        return 1;
    }

    if (min >= max) {
        std::cerr << "错误: 最小值必须小于最大值" << std::endl;
        return 1;
    }

    // 设置随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);

    // 生成并输出向量数据
    for (int i = 0; i < count; i++) {
        for (int j = 0; j < dim; j++) {
            if (j > 0) std::cout << ",";
            std::cout << dis(gen);
        }
        std::cout << std::endl;
    }

    return 0;
}
