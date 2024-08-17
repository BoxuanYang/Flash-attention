#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <vector>
#include <immintrin.h>
#include <cmath>
#include <chrono>
#include <random>

// 打印矩阵
void print_matrix(const std::vector<float> &matrix, int M, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << matrix[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// 检查两个矩阵是否相等
bool matrices_are_equal(const std::vector<float> &A, const std::vector<float> &B, int M, int N, float tolerance = 1e-6) {
    if (A.size() != B.size()) return false;
    for (size_t i = 0; i < A.size(); ++i) {
        if (std::fabs(A[i] - B[i]) > tolerance) return false;
    }
    return true;
}

inline void blocked_matmul(std::vector<float> &Z, std::vector<float> &X, std::vector<float> &Y, int &M, int &d, int &N){
    int blocksize = 16;
    //std::cout << "blocksize: " << blocksize << std::endl;
    
    // start row index of Z-block
    for(int i = 0; i < M; i += blocksize){
        // size of Z-block in rows
        int Z_block_rows = blocksize < M - i ? blocksize : M - i;

        // start col index of Z-block
        for(int j = 0; j < N; j += blocksize){
            // size of Z-block in cols
            int Z_block_cols = blocksize < N - j ? blocksize : N - j;

            // start index of X's blocks(row) and Y's block(col)
            for(int k = 0; k < d; k += blocksize){
                int d_size = blocksize < d - k ? blocksize : d - k;
                // i-offset within block of Z 
                for(int ii = 0; ii < Z_block_rows; ii++){
                    // j-offset within block of Z
                    for(int jj = 0; jj < Z_block_cols; jj++){
                        // k-offset within block of X(row-wise) and Y(col-wise)
                        for(int kk = 0; kk < d_size; kk++){
                            // Z[i + ii][j + jj] = X[i + ii][kk] * Y[kk][j + jj]
                            Z[(i + ii) * N + (j + jj)] += X[(i + ii) * d + (k + kk)] * Y[(k + kk) * N + (j + jj)];
                        }
                    }                  
                }
            }
        }
    }
    return;
}

inline void naive_matmul(std::vector<float> &Z, std::vector<float> &X, std::vector<float> &Y, int &M, int &d, int &N){

    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            for(int k = 0; k < d; k++){
                // Z[i][j] == X[i][k] * Y[k][j]
                Z[i * N + j] += X[i * d + k] * Y[k * N + j];
            }
        }
    }
    return;
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " M d N" << std::endl;
        return 1;
    }

    // 将命令行参数转换为整数
    int M = std::atoi(argv[1]);
    int d = std::atoi(argv[2]);
    int N = std::atoi(argv[3]);

    // 创建和初始化矩阵 X 和 Y
    std::vector<float> X(M * d), Y(d * N), naive_result(M * N), blocked_result(M * N);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-10.0, 10.0);

    // 用随机初始化X, Y
    for (int i = 0; i < M * d; ++i) X[i] = dis(gen);
    for (int i = 0; i < d * N; ++i) Y[i] = dis(gen);

    // 测量naive_matmul时间
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 5; ++i) {
        naive_matmul(naive_result, X, Y, M, d, N);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken for naive matmul: " << duration.count() << " seconds" << std::endl;

    std::cout << std::endl;

    // 测量blocked_matmul时间
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 5; ++i) {
        blocked_matmul(blocked_result, X, Y, M, d, N);
    }
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Time taken for blocked matmul: " << duration.count() << " seconds" << std::endl;




    

    /*
    std::cout << "X matrix: " << std::endl;
    print_matrix(X, M, d);

    std::cout << "Y matrix: " << std::endl;
    print_matrix(Y, d, N);
    */

    

    // 检查结果
    if (matrices_are_equal(naive_result, blocked_result, M, N)) {
        std::cout << "Test passed: Results are equal." << std::endl;
    } else {
        std::cout << "Test failed: Results are not equal." << std::endl;
        /*
        std::cout << "Naive result:" << std::endl;
        print_matrix(naive_result, M, N);
        std::cout << "Blocked result:" << std::endl;
        print_matrix(blocked_result, M, N);
        */
    }

    return 0;
}

