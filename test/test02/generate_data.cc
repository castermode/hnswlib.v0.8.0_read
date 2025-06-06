// 生成测试的向量数据
// 通过命令行参数传入参数
// 参数1: 维度
// 参数2: 数据量
// 直接将生成浮点向量数据打印输出到标准输出

#include <iostream>
#include <vector>
#include <random>
#include <cstdlib>
#include <string>

int main(int argc, char** argv) {
    // 检查参数数量
    if (argc != 3) {
        std::cerr << "用法: " << argv[0] << " <维度> <数据量>" << std::endl;
        return 1;
    }

    // 解析命令行参数
    int dim = std::stoi(argv[1]);
    int count = std::stoi(argv[2]);

    // 检查参数有效性
    if (dim <= 0 || count <= 0) {
        std::cerr << "错误: 维度和数据量必须为正整数" << std::endl;
        return 1;
    }

    // 设置随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0, 1.0);

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
