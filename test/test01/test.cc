#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include "hnswlib/hnswlib/hnswlib.h"

using namespace std;
using namespace hnswlib;

// 生成随机浮点数的辅助函数
float random_float(float min, float max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);
    return dis(gen);
}

int main() {
    // 示例参数
    const size_t dim = 128;            // 向量维度
    const size_t max_elements = 10000; // 最大元素数量
    const size_t num_elements = 5000;  // 要插入的元素数量
    const size_t k = 10;               // 查找的最近邻数量
    
    // 创建随机数据
    std::vector<std::vector<float>> data_points(num_elements);
    for (size_t i = 0; i < num_elements; i++) {
        data_points[i].resize(dim);
        for (size_t j = 0; j < dim; j++) {
            data_points[i][j] = random_float(-1.0f, 1.0f);
        }
    }
    
    // 创建一个查询向量
    std::vector<float> query_point(dim);
    for (size_t i = 0; i < dim; i++) {
        query_point[i] = random_float(-1.0f, 1.0f);
    }

    std::cout << "使用HNSW算法进行相似性搜索示例" << std::endl;
    std::cout << "维度: " << dim << ", 元素数量: " << num_elements << std::endl;

    // ------------------ 使用L2距离空间 ------------------
    {
        std::cout << "\n1. 使用L2距离空间的示例：" << std::endl;
        
        // 初始化L2空间
        L2Space space(dim);
        
        // 创建HNSW索引
        // M - 每个层级的最大出边数量
        // ef_construction - 构建时的动态列表大小
        HierarchicalNSW<float> alg_l2(&space, max_elements, 16, 200);
        
        // 添加点到索引
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < num_elements; i++) {
            alg_l2.addPoint(data_points[i].data(), i);
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        std::cout << "构建索引时间: " 
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() 
                  << " ms" << std::endl;
        
        // 设置搜索参数
        alg_l2.setEf(50); // ef越高，搜索越精确但也越慢
        
        // 执行搜索
        start = std::chrono::high_resolution_clock::now();
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = 
            alg_l2.searchKnn(query_point.data(), k);
        end = std::chrono::high_resolution_clock::now();
        
        std::cout << "搜索时间: " 
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() 
                  << " μs" << std::endl;
        
        // 输出结果
        std::cout << "查询结果 (ID, 距离):" << std::endl;
        while (!result.empty()) {
            std::cout << "(" << result.top().second << ", " << result.top().first << ")" << std::endl;
            result.pop();
        }
        
        // 保存索引到文件
        alg_l2.saveIndex("l2_index.bin");
        std::cout << "索引已保存到 l2_index.bin" << std::endl;
        
        // 从文件加载索引的示例
        L2Space new_space(dim);
        HierarchicalNSW<float> loaded_index(&new_space, "l2_index.bin");
        std::cout << "索引已从文件加载" << std::endl;
    }
    
    // ------------------ 使用内积空间 ------------------
    {
        std::cout << "\n2. 使用内积空间的示例 (余弦相似度)：" << std::endl;
        
        // 初始化内积空间
        InnerProductSpace space(dim);
        
        // 创建HNSW索引
        HierarchicalNSW<float> alg_ip(&space, max_elements, 16, 200);
        
        // 对于内积相似度（余弦相似度），我们通常需要归一化向量
        std::vector<std::vector<float>> normalized_data(num_elements);
        for (size_t i = 0; i < num_elements; i++) {
            normalized_data[i].resize(dim);
            
            // 计算向量范数
            float norm = 0.0f;
            for (size_t j = 0; j < dim; j++) {
                norm += data_points[i][j] * data_points[i][j];
            }
            norm = sqrt(norm);
            
            // 归一化向量
            for (size_t j = 0; j < dim; j++) {
                normalized_data[i][j] = data_points[i][j] / norm;
            }
            
            // 添加归一化的点到索引
            alg_ip.addPoint(normalized_data[i].data(), i);
        }
        
        // 归一化查询向量
        std::vector<float> normalized_query(dim);
        float query_norm = 0.0f;
        for (size_t i = 0; i < dim; i++) {
            query_norm += query_point[i] * query_point[i];
        }
        query_norm = sqrt(query_norm);
        for (size_t i = 0; i < dim; i++) {
            normalized_query[i] = query_point[i] / query_norm;
        }
        
        // 设置搜索参数
        alg_ip.setEf(50);
        
        // 执行搜索
        auto start = std::chrono::high_resolution_clock::now();
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = 
            alg_ip.searchKnn(normalized_query.data(), k);
        auto end = std::chrono::high_resolution_clock::now();
        
        std::cout << "搜索时间: " 
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() 
                  << " μs" << std::endl;
        
        // 输出结果 (对于内积相似度，值越大表示越相似)
        std::cout << "查询结果 (ID, 相似度):" << std::endl;
        while (!result.empty()) {
            // 将距离转换为相似度（内积搜索中，距离实际上是-内积）
            float similarity = -result.top().first; // 转换为正的相似度值
            std::cout << "(" << result.top().second << ", " << similarity << ")" << std::endl;
            result.pop();
        }
        
        // 演示删除元素功能
        size_t id_to_delete = 42;
        std::cout << "删除ID为 " << id_to_delete << " 的元素" << std::endl;
        alg_ip.markDelete(id_to_delete);
        
        // 保存索引到文件
        alg_ip.saveIndex("ip_index.bin");
        std::cout << "索引已保存到 ip_index.bin" << std::endl;
    }
    
    return 0;
}
