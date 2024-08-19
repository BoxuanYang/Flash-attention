#include <torch/extension.h>
#include <ATen/ATen.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <vector>
#include <immintrin.h>
#include <cmath>

// Uncomment for ISPC
//#include "module_ispc.h"
//using namespace ispc;

// ------------------------------------ //
// 	WARM-UP: ACCESSING TENSORS      //
// ------------------------------------ //

// Step #1: Understand Read/Write Accessors for a 2D Tensor
inline float twoDimRead(std::vector<float> &tensor, int x, int y, const int sizeX) {
    // Note that sizeX is the size of a Row, not the number of rows
    return tensor[x * (sizeX)+ y];
}

inline void twoDimWrite(std::vector<float> &tensor, int x, int y, const int sizeX, float val) {
    tensor[x * (sizeX) + y] = val;
}

// Step #2: Implement Read/Write Accessors for a 4D Tensor
inline float fourDimRead(std::vector<float> &tensor, int x, int y, int z, int b, 
        const int &sizeX, const int &sizeY, const int &sizeZ) {
    // TODO
    return tensor[x * (sizeX * sizeY + sizeY) + y * (sizeY * sizeZ) + z * (sizeZ) + b];
}

inline void fourDimWrite(std::vector<float> &tensor, int x, int y, int z, int b, 
        const int sizeX, const int sizeY, const int sizeZ, float val) {
    // TODO
    tensor[x * (sizeX * sizeY + sizeY) + y * (sizeY * sizeZ) + z * (sizeZ) + b] = val;
    return; 
}

// Transform a torch::Tensor into std::vector. 
// DO NOT EDIT THIS FUNCTION
std::vector<float> formatTensor(torch::Tensor tensor) {
    tensor = tensor.flatten();
    tensor = tensor.contiguous();
    std::vector<float> vec(tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel());
    return vec;
}

/* Programming Your Attention Modules.
 * 
 * You are given Q, K, and V Tensors as inputs that are formatted as vectors. We have also created O and QK^t Tensors 
 * that are formatted as vectors. After you have implemented your accessors in the Warm-Up you should be able to
 * read/write to these tensors via the read/write functions above.
 *
 * You are also given 4 integers as parameters: B, H, N, d:
 *
 * B (Batch Size) - The number of samples for your attention layer. Think of it this way - if I asked my dnn
 * a question and it output 5 different answers it had a batch size of 5. These samples are independent of each
 * other and thus can be parallelized.
 *
 * H (Number of Heads) - Each head runs on its own set of Q, K, V matrices. This effectively allows each head
 * to operate the same attention algorithm, but each with each head using different hyperparameters. These
 * allow each head to have their own definition of what relevance is when looking at a token. These heads
 * can operate independently of one another and thus can be parallized.
 *
 * N (Sequence Length) - The number of tokens. You may think of this as the number of words in a sample.
 *
 * d (Embedding Dimensionality) - The number of features each token encodes per attention head. Let's
 * say I encoded a word using the follow (length, number of vowels, has a capital letters). The
 * emvedded dimensionaliy would be 3.
 * */

// ---------------------------------------------------------- //
//                  PART 1: NAIVE ATTENTION                   //
// ---------------------------------------------------------- //
torch::Tensor myNaiveAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                int B, int H, int N, int d){

    // Q, K, V are passed in with Shape: (B, H, N, d)
    //QK^t Intermediate Tensor has Shape (N, N)
    
    //Make O Tensor with Shape (B, H, N, d) 
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    //Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    //Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);
    
    /* Here is an example of how to read/write 0's to  Q (B, H, N, d) using the 4D accessors

        //loop over Batch Size
         for (int b = 0; b < B; b++) {

             //loop over Heads
             for (int h = 0; h < H; h++) {

                 //loop over Sequence Length
                 for (int i = 0; i < N; i++) {

                     //loop over Embedding Dimensionality
                     for (int j = 0; j < d; j++) {
                        float val = fourDimRead(Q, b, h, i, j, H, N, d);
                        val = 0.0;
                        fourDimWrite(Q, b, h, i, j, H, N, d, val);
                     }
                 }
             }
         }
    */

    /* Here is an example of how to read/write 0's to  QK_t (N, N) using the 2D accessors

           for (int i = 0; i < N; i++) {
	       for (int j = 0; j < N; j++) {
	           float val = twoDimRead(QK_t, i, j, N);
               val = 0.0;
	           twoDimWrite(QK_t, i, j, N, val);
             }
         }
    */
    
    // -------- YOUR CODE HERE  -------- //

    // Loop through all batches and heads
    for(int b = 0; b < B; b++){
        for(int h = 0; h < H; h++){

            // 1: Multiply Q, K^T and store in QK_t. Q, K are N by d in size.
            for(int i = 0; i < N; i++){
                for(int j = 0; j < N; j++){
                    // compute QK_t[i][j]
                    float QK_t_ij = 0;
                    for(int k = 0; k < d; k++){
                        // read Q[i][k]
                        float Q_ik = fourDimRead(Q, b, h, i, k, H, N, d);

                        // read K[j][k], i.e., K^T[k][j]
                        float K_jk = fourDimRead(K, b, h, j, k, H, N, d);

                        // Accumulate Q[i][k] * K[j][k] in QK_t
                        QK_t_ij += Q_ik * K_jk;
                    }
                    twoDimWrite(QK_t, i, j, N, QK_t_ij);
                }
            }

            // 2: Softmax Q K^T, in QK_t, QK_t is N by N in size
            for(int i = 0; i < N; i++){
                // 2.1: Transform i-th row of QK_t into exp and compute the norm
                float row_i_norm = 0;
                for(int j = 0; j < N; j++){ 
                    float QK_t_ij = twoDimRead(QK_t, i, j, N);
                    QK_t_ij = exp(QK_t_ij);
                    twoDimWrite(QK_t, i, j, N, QK_t_ij);

                    // Accumulate row norm in row_i_norm
                    row_i_norm += QK_t_ij;
                }

                // 2.2: Divide i-th row by QK_t_i_sum
                for(int j = 0; j < N; j++){ // divide by row sum
                    float QK_t_ij = twoDimRead(QK_t, i, j, N);
                    QK_t_ij = QK_t_ij / row_i_norm;
                    twoDimWrite(QK_t, i, j, N, QK_t_ij);
                }
            }

            // 3: Multiply Q K^T, V and store in O, O is in shape: (B, H, N, d)
            for(int i = 0; i < N; i++){
                for(int j = 0; j < d; j++){ 
                    float O_ij= 0;
                    for(int k = 0; k < N; k++){
                        // read QK_t[i][k] from QK_t_ik
                        float QK_t_ik = twoDimRead(QK_t, i, k, N);

                        // read [k][j] from V
                        float V_kj = fourDimRead(V, b, h, k, j, H, N, d);

                        O_ij += QK_t_ik * V_kj;
                    }
                    fourDimWrite(O, b, h, i, j, H, N, d, O_ij);
                }
            }
        }
    }
    
    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//     PART 2: BLOCKED MATRIX MULTIPLY AND UNFUSED SOFTMAX    //
// ---------------------------------------------------------- //

/*
Compute Z = XY with blocked matrix multiply. X is M by d, Y is d by N.

Size of L1 cache: 768 KiB
Size of a cache line: 64 bytes(16 floats). 16 * 16 * 3 = 768 bytes < 768 KiB. Hence, block size B should be 16.
*/
inline void blocked_matmul(std::vector<float> &Z, std::vector<float> &X, std::vector<float> &Y, int M, int d, int N){
    int blocksize = 16;
    // Assume for simplicity that M, N, d are all multiples of blocksize

    // Hint: min(tile_size, N-tileIndex*tileSize)
    // helper function: int min_value = std::min(a, b);
    // std::min(blocksize, () - () * blocksize)


    // start row index of Z-block
    for(int i = 0; i < M; i += blocksize){
        // size of Z-block in rows
        int Z_block_rows = std::min(blocksize, M - i);

        // start col index of Z-block
        for(int j = 0; j < N; j += blocksize){
            // size of Z-block in cols
            int Z_block_cols = std::min(blocksize, N - j);

            // start index of X's blocks(row) and Y's block(col)
            for(int k = 0; k < d; k += blocksize){
                int d_size = std::min(blocksize, d - k);
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

torch::Tensor myUnfusedAttentionBlocked(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                int B, int H, int N, int d){
    
    // Q, K, V are passed in with Shape: (B, H, N, d)
    //QK^t Intermediate Tensor has Shape (N, N)

    //Make O Tensor with Shape (B, H, N, d) 
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    //Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    //Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);

    int blocksize = 16;

    // -------- YOUR CODE HERE  -------- //

    for(int b = 0; b < B; b++){
        for(int h = 0; h < H; h++){

        // 1: Multiply Q, K^T and store in QK_t. Q, K are N by d in size.

        // start row index of Z-block
        for(int i = 0; i < N; i += blocksize){
            // size of Z-block in rows
            int QK_t_block_rows = std::min(blocksize, N - i);
            
            // start col index of QK_t block
            for(int j = 0; j < N; j += blocksize){
                // size of QK_t block in cols
                int QK_t_block_cols = blocksize < N - j ? blocksize : N - j; // std::min(blocksize, N - j);
                
                // start index of X's blocks(row) and Y's block(col)
                for(int k = 0; k < d; k += blocksize){
                    int d_size = blocksize < d - k ? blocksize : d - k;
                    // i-offset within block of Z 
                    for(int ii = 0; ii < QK_t_block_rows; ii++){
                        // j-offset within block of Z
                        for(int jj = 0; jj < QK_t_block_cols; jj++){
                            float QK_t_i_ii_j_jj = twoDimRead(QK_t, i + ii, j + jj, N);
                            
                            // k-offset within block of X(row-wise) and Y(col-wise)
                            for(int kk = 0; kk < d_size; kk++){
                                
                                // read Q[i + ii][k + kk]
                                float Q_ik = fourDimRead(Q, b, h, i + ii, k + kk, H, N, d);

                                // read K^T[k + kk][j + jj], i.e., K[j + jj][k + kk]
                                float K_jk = fourDimRead(K, b, h, j + jj, k + kk, H, N, d);

                                // Compute QK_t[i + ii][j + jj] += Q[i + ii][k + kk] * K^T[k + kk][j + jj]
                                QK_t_i_ii_j_jj += Q_ik * K_jk;
                            }
                            
                            twoDimWrite(QK_t, i + ii, j + jj, N, QK_t_i_ii_j_jj);
                        }                  
                    }
                } 
            }
        }


            // 2: Softmax Q K^T, in QK_t, QK_t is N by N in size
            for(int i = 0; i < N; i++){
                // 2.1: Transform i-th row of QK_t into exp and compute the norm
                float row_i_norm = 0;
                for(int j = 0; j < N; j++){ 
                    float QK_t_ij = twoDimRead(QK_t, i, j, N);
                    QK_t_ij = exp(QK_t_ij);
                    twoDimWrite(QK_t, i, j, N, QK_t_ij);

                    // Accumulate row norm in row_i_norm
                    row_i_norm += QK_t_ij;
                }

                // 2.2: Divide i-th row by QK_t_i_sum
                for(int j = 0; j < N; j++){ // divide by row sum
                    float QK_t_ij = twoDimRead(QK_t, i, j, N);
                    QK_t_ij = QK_t_ij / row_i_norm;
                    twoDimWrite(QK_t, i, j, N, QK_t_ij);
                }
            }


           // 3: Multiply QK_t, V and store in O, O is in shape: (B, H, N, d)

           // start row index of O block
           for(int i = 0; i < N; i += blocksize){
            // size of O-block in rows
            int O_block_rows = blocksize < N - i ? blocksize : N - i;
            
            // start col index of O block
            for(int j = 0; j < d; j += blocksize){
                // size of O block in cols
                int O_block_cols = blocksize < d - j ? blocksize : d - j; // std::min(blocksize, N - j);
                
                // start index of QK_t's blocks(row) and V's block(col)
                for(int k = 0; k < N; k += blocksize){
                    int d_size = blocksize < N - k ? blocksize : N - k;
                    // i-offset within block of O
                    for(int ii = 0; ii < O_block_rows; ii++){
                        // j-offset within block of O
                        for(int jj = 0; jj < O_block_cols; jj++){
                            // read O[i + ii][j + jj]
                            float O_i_ii_j_jj = fourDimRead(O, b, h, i + ii, j + jj, H, N, d);
                            
                            // k-offset within block of QK_t(row-wise) and V(col-wise)
                            for(int kk = 0; kk < d_size; kk++){
                                // read QK_t[i + ii][k + kk] 
                                float QK_t_ik = twoDimRead(QK_t, i + ii, k + kk, N);

                                // read V[k + kk][j + jj]
                                float V_jk = fourDimRead(V, b, h, k + kk, j + jj, H, N, d);

                                // Compute O[i + ii][j + jj] += QK_t[i + ii][k + kk] * V[k + kk][j + jj]
                                O_i_ii_j_jj += QK_t_ik * V_jk;
                            }
                            fourDimWrite(O, b, h, i + ii, j + jj, H, N, d, O_i_ii_j_jj);
                        }                  
                    }
                } 
            }
        }
        }
    }

    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//                 PART 3: FUSED ATTENTION     	              //
// ---------------------------------------------------------- //

torch::Tensor myFusedAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor temp,
                int B, int H, int N, int d){

    // Q, K, V are passed in with Shape: (B, H, N, d)

    //Make O Tensor with Shape (B, H, N, d)
    //and O Row Tensor with Shape (N)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
    at::Tensor ORowTensor = at::zeros({N}, at::kFloat);

    //Format Y, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);
    
    //Format ORow Tensor into a 1D vector
    // You can simply access this as ORow[i]
    std::vector<float> ORow = formatTensor(ORowTensor);


    // -------- YOUR CODE HERE  -------- //
    // We give you a template of the first three loops for your convenience
    //loop over batch
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++){

        //loop over heads
        for (int h = 0; h < H; h++){

            for (int i = 0; i < N ; i++){
                
                // YRow is moved inside so each OpenMP thread gets a local copy.
                at::Tensor ORowTensor = temp.index({torch::indexing::Slice(omp_get_thread_num(), torch::indexing::None)});      
                std::vector<float> ORow = formatTensor(ORowTensor);

                // a vector that stores Q[i][:] * K^T
                std::vector<float> tmp_vec(N, 0.0f);

                // 1. Multiply Q[i][:] with K^T to get a row, Q[i * d + j], ORow is of size d

                // loop through all rows of K^T
                for(int ii = 0; ii < d; ii++){
                    // read Q[i][ii]
                    float Q_i_ii = fourDimRead(Q, b, h, i, ii, H, N, d);
                    
                    // loop through all entries of ii-th row of K^T
                    for(int j = 0; j < N; j++){

                        // read K_t[ii][j], i.e., K[j][ii]
                        float K_t_ii_j = fourDimRead(K, b, h, j, ii, H, N, d);

                        // tmp_vec[j] += Q[i][ii] * K_t[ii][j]
                        tmp_vec[j] += Q_i_ii * K_t_ii_j;
                    }
                }
                // 2. Softmax the row COMPLETED
                float sum = 0;
                for(int j = 0; j < N; j++){
                    tmp_vec[j] = exp(tmp_vec[j]);
                    sum += tmp_vec[j];
                }
                for(int j = 0; j < N; j++){
                    tmp_vec[j] = tmp_vec[j] / sum;
                }

                // 3. Multiply the softmax'd row with V to get ORow and write back to O
                
                // loop through all rows of V
                for(int  ii = 0; ii < N; ii++){
                    // loop through all entries of ii-th row of V
                    for(int j = 0; j < d; j++){
                        // read V[ii][j]
                        float V_ii_j = fourDimRead(V, b, h, ii, j, H, N, d);

                        // read O[i][j]
                        float O_i_j = fourDimRead(O, b, h, i, j, H, N, d);

                        // O[i][j] += ORow[j] * V[ii][j]
                        O_i_j += tmp_vec[ii] * V_ii_j;

                        // write back to O
                        fourDimWrite(O, b, h, i, j, H, N, d, O_i_j); 
                    }
                }
                //YOUR CODE HERE
                
            }
        }
    }
	    
	
    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//                PART 4: FLASH ATTENTION 		      //
// ---------------------------------------------------------- //

torch::Tensor myFlashAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor,
               torch::Tensor QiTensor, torch::Tensor KjTensor, torch::Tensor VjTensor,
               torch::Tensor SijTensor, torch::Tensor PijTensor, torch::Tensor PVTensor,
               torch::Tensor OiTensor, torch::Tensor LTensor,  torch::Tensor LiTensor, 
	       torch::Tensor LijTensor, torch::Tensor LnewTensor, int Bc, int Br,
                int B, int H, int N, int d) {
        
    // Q, K, V are passed in with Shape: (B, H, N, d)
    // Sij, Pij are passed in with Shape: (Br, Bc)
    // Kj, Vj are passed in with Shape: (Bc, d)
    // Qi, Oi, and PV  are passed in with Shape: (Br, d)
    // L in passed in with Shape: (N)
    // Li, Lij, and Lnew are passed in with shape (Br)

    //Make O Tensor with Shape (B, H, N, d)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
   
    //Format All Tensors into Vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);
    std::vector<float> Sij = formatTensor(SijTensor);
    std::vector<float> Pij = formatTensor(PijTensor);
    std::vector<float> Kj = formatTensor(KjTensor);
    std::vector<float> Vj = formatTensor(VjTensor);
    std::vector<float> Qi = formatTensor(QiTensor);
    std::vector<float> Oi = formatTensor(OiTensor);
    std::vector<float> l = formatTensor(LTensor);
    std::vector<float> PV = formatTensor(PVTensor);
    std::vector<float> li = formatTensor(LiTensor);
    std::vector<float> lij = formatTensor(LijTensor);
    std::vector<float> lnew = formatTensor(LnewTensor);

    // -------- YOUR CODE HERE  -------- //
    // The number of blocks among rows(Q)
    int Tr = (N + Br - 1) / Br;
    // The number of blocks among cols(K)
    int Tc = (N + Bc - 1) / Bc;

    for(int b = 0; b < B; b++){
        for(int h = 0; h < H; h++){


            for(int j = 0; j < Tc; j++){
                // 1. Load Kj, Vj
                // Kj, Vj are passed in with Shape: (Bc, d)
                int j_start = j * Bc;
                int j_size = Bc < N - j * Bc ? Bc : N - j * Bc;

                for(int ii = 0; ii < j_size; ii++){
                    for(int jj = 0; jj < d; jj++){
                        float Kj_ii_jj = fourDimRead(K, b, h, ii + j_start, jj, H, N, d);
                        twoDimWrite(Kj, ii, jj, d, Kj_ii_jj);

                        float Vj_ii_jj = fourDimRead(V, b, h, ii + j_start, jj, H, N, d);
                        twoDimWrite(Vj, ii, jj, d, Vj_ii_jj);
                    }
                }


                for(int i = 0; i < Tr; i++){
                    // 2. Load Qi, Oi, li
                    // Qi:  (Br,d)  = (i_size,d)
                    // Oi:  (Br,d)  = (i_size,d)
                    // li:   Br

                    int i_start = i * Br;
                    int i_size = Br < N - i * Br ? Br : N - i * Br;

                    for(int ii = 0; ii < i_size; ii++){
                        // load li
                        li[ii] = l[ii + i_start]
                        for(int jj = 0; jj < d; jj++){
                            // load Qi
                            float Qi_ii_jj = fourDimRead(Q, b, h, ii + i_start, jj, H, N, d);
                            twoDimWrite(Qi, ii, jj, d, Qi_ii_jj);

                            // load Oi
                            float Oi_ii_jj = fourDimRead(O, b, h, ii + i_start, jj, H, N, d);
                            twoDimWrite(Kj, ii, jj, d, Oi_ii_jj);
                        }
                    }
                    
                    
                    // 3. Compute Sij = Qi * Kj^T
                    // Qi   : (Br,d)  = (i_size,d)
                    // Kj^T : (d,Bc)  = (d,j_size)
                    // Sij  : (Br,Bc)
                    // Sij[ii][jj] += Qi[ii][kk] * Kj[jj][kk](Kj^T[kk][jj])
                    for(int ii = 0; ii < i_size; ii++){
                        for(int jj = 0; jj < j_size; jj++){
                            float Sij_ii_jj = 0;
                            for(int kk = 0; kk < d; kk++){
                                float Qi_ii_kk = twoDimRead(Qi, ii, kk, d);
                                float Kj_jj_kk = twoDimRead(Kj, jj, kk, d)
                                Sij_ii_jj += Qi_ii_kk * Kj_jj_kk;
                            }
                            twoDimWrite(Sij, ii, jj, Bc, Sij_ii_jj);

                            // 4. Compute Pij = exp(Sij)
                            float Pij_ii_jj = exp(Sij_ii_jj);
                            twoDimWrite(Pij, ii, jj, Bc, Pij_ii_jj);
                        }
                    }

                    // 5. Compute lij = rowsum(Pij)
                    // Pij: (Br,Bc)
                    // lij: Br
                    for(int ii = 0; ii < i_size; ii++){
                        float lij_ii = 0;
                        for(int jj = 0; jj < j_size; jj++){
                            float Pij_ii_jj = twoDimRead(Pij, ii, jj, Bc);
                            lij_ii += Pij_ii_jj;
                        }
                        lij[ii] = lij_ii;
                    }
                    
                    
                    // 6. Compute lnew = li + lij
                    // li, lij, and lnew are passed in with shape (Br)
                    for(int ii = 0; ii < Br; ii++){
                        lnew[ii] = li[ii] + lij[ii];
                    }
                    

                    // 7. Compute Oi = (li * Oi + Pij * Vj) / lnew. Note: li * Oi is elementwise multiply
                    // Oi:  (Br,d)  = (i_size,d)
                    // Pij: (Br,Bc) = (i_size,j_size)
                    // Vj:  (Bc,d)  = (j_size,d)
                    // li:   Br     =  i_size
                    // lnew: Br     =  i_size

                    
                   

                    // 8. Write Oi and lnew back to O and l
                    
                    

                    
                }
            }
            

        }
    }


    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


/* DO NOT EDIT THESE BINDINGS */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("myNaiveAttention", &myNaiveAttention, "Naive Attention");
  m.def("myUnfusedAttentionBlocked", &myUnfusedAttentionBlocked, " Blocked Unfused Attention");
  m.def("myFusedAttention", &myFusedAttention, "Fused Attention");
  m.def("myFlashAttention", &myFlashAttention, "Flash Attention");
  m.def("twoDimRead", &twoDimRead, "twoDimRead");
  m.def("fourDimRead", &fourDimRead, "fourDimRead");
}
