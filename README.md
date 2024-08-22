# Flash Attention

## Overview 

This is my personal implementation of the famous [flash attention](https://arxiv.org/abs/2205.14135) paper. This implementation will be used as a part of [NanoGPT](https://github.com/karpathy/nanoGPT) model to generate Shaespere-like text.

In this project, I implemented and optimized the key components of a transformer-based deep neural network that synthesizes Shakespeare text. Due to hardware limitations, the DNN I work with is fairly simple model, the basic components of the model are the same as those featured in LLMs. I implemented the attention layer with C++ with state-of-art optimization techniques like blocked
matrix multiuplication as well as flash attention to generate Shakespeare-like text.


## Docker Setup

It is recommended to run this project in docker environment. To run the docker container, first build the docker image:

    docker build -t boxuan-flash-attention .

Then, run the docker container:

    docker run -it boxuan-flash-attention



Run the command below to run inference using a model trained by the [Stanford](https://gfxcourses.stanford.edu/cs149/fall23) staff. You will see some randomly generated Shakespeare text.

     python3 gpt.py part0 --inference -m shakes128

Note that the first time you run the program, it will perform a compilation step that may take a few seconds, you'll see the text `Compiling code into a PyTorch module...`. <br><br>
After this is complete, you'll see some text that begins something like this:

    Running inference using dnn model shakes128
    number of parameters: 0.80M
    Loading meta from data/shakespeare_char/meta.pkl...

    BOTtaps along my lord.

    DUKE OF AUMERLE:
    The this is needs! Camillo, put I will make be strong.

    QUEEN MARGARET:
    My lord, yet t
    -------------------------------------------------------------
    CAMILLO:
    The shadows men sweet thy will burn comes.
    
    FLORIZEL:
    But of appear, good from thy heart
    As I be come of repeal of a w
    -------------------------------------------------------------

Feel free to change to larger sequence lengths by changing the `-m` parameter to larger models like `shakes256`, `shakes1024`, or `shakes2048`. You'll see the performance of NanoGPT token generation slow considerably with the bigger models.

## Project Description

### Development
The project is developed incrementally. First I implement the most basic(i.e., most unoptimized) attention layer in part 1. Then in part 2, an attention layer with blocked matrix multiplication is developed. In part 3, I added thread-level parallelism in OpenMP to build a fused attention. Then in part 3, I implement the flash attention as described in paper [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135). 

### How Attention works?

The attention mechanism takes as input three matrices `Q`, `K`, and `V`, referred to as "query", "key", and "value" vectors.  Each of these matrices are `Nxd` in size. `N` is the number of tokens (words) in the input sequence, so each row in these matrices is a length-`d` vector containing an embedding (a neural code) for one of the input words.  In other words `Q`, `K`, and `V` all contain different `d`-dimensional embeddings of the input tokens.

The first step of an attention module is to compute all pairs of interactions between the words.  This is done by multiplying the query matrix $Q$ against the key matrix $K$ to compute:

$$S = QK^T.$$

The next computation is a [softmax operation](https://machinelearningmastery.com/softmax-activation-function-with-python/) performed per-row of $S$.  The softmax produces normalized probabilities per row. 

For each row of the matrix, the softmax operation performs the following computation. Note that we give you the math for computing a softmax on a 1D vector $X$.  You'll need to perform this math for each row of the matrix $S$ above.

$$\text{softmax}(x) = \frac{\mathbf f(x)}{l(x)}$$

where

$$\mathbf f(x) = \begin{bmatrix}e^{x_1} & e^{x_2} &\cdots & e^{x_N} \end{bmatrix}\qquad \text{and} \qquad l(x) = \sum_{i=1}^N f(x)_i.$$



This yields a matrix of attention weights $P$, where

$$P = \texttt{softmax}(\texttt{each row of }S).$$

Finally, the attention weights are used to aggregate a set of learned **value** vectors, which are provided as a matrix $V$ of shape $N \times d$, to produce a final output $O$:

$$O = PV.$$

In summary, the attention layer consists of an expensive matrix multiply, followed by a softmax layer, followed by one more matrix multiply.




## Part 1: A Simple (But Not So Efficient) Implementation of Attention

This can be found in the myNaiveAttention function in module.cpp file. In this function, I implemented the most basic attention layer.

The function works as follows(as well as the following parts):

    1) For each Batch:
    2) For each Head:
        a) Loop through Q and K and multiply Q with K^t, storing the result in QK^t. 
        QK^t is preallocated and passed as an arg to myNaiveAttention and hence no extra memory allocation is needed.      
   
        b) After getting QK^t -- which should have shape (N, N) -- we need to  loop through each row. For each row, we get the exponential of each row element. Then, divide each of these resulting exponentials by the sum of all exponentials in its row and then store it back into QK^t. 
   
        c) Finally, I need to do matrix multiply QK^t with V and store the result into O. Then simply store the resulting shape (N, d) back into O.

### Testing
Run the following test to check program's correctness:

    python3 gpt.py part1

While running the test, we show results of the pytorch profiler - this information is presented in a table which presents detailed statistics on all function calls called in the test. The table that is dumped will look like the following:

    -----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                         Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls  
    -----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                  aten::empty         0.01%      23.000us         0.01%      23.000us       3.286us       5.00 Mb       5.00 Mb             7  
                  aten::zeros         0.14%     321.000us         0.18%     408.000us     102.000us       4.50 Mb           4 b             4  
    STUDENT - NAIVE ATTENTION        99.56%     229.600ms        99.97%     230.538ms     230.538ms       4.50 Mb      -1.00 Mb             1  
                  aten::clone         0.02%      37.000us         0.10%     231.000us     115.500us       1.00 Mb           0 b             2  
                aten::flatten         0.02%      48.000us         0.07%     153.000us      30.600us     512.00 Kb           0 b             5  
             aten::empty_like         0.00%       3.000us         0.00%       8.000us       8.000us     512.00 Kb           0 b             1  
          aten::empty_strided         0.01%      16.000us         0.01%      16.000us      16.000us     512.00 Kb     512.00 Kb             1  
              model_inference         0.02%      38.000us        99.98%     230.578ms     230.578ms     512.00 Kb      -4.00 Mb             1  
                  aten::zero_         0.02%      42.000us         0.15%     354.000us      88.500us           0 b           0 b             4  
                  aten::fill_         0.14%     312.000us         0.14%     312.000us     156.000us           0 b           0 b             2  
    -----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  

After the table is dumped, we also display two relevant statistics, cpu time (in milliseconds) and mem usage (in bytes). If implemented correctly, we should see those two values output like so:

    REFERENCE - NAIVE ATTENTION statistics
    cpu time:  230.724ms
    mem usage:  4718588 bytes
    
    STUDENT - NAIVE ATTENTION statistics
    cpu time:  232.561ms
    mem usage:  4718588 bytes

If attention is not producing the correct output, below message will be presented:

    ATTENTION PRODUCED INCORRECT RESULTS
    




To test the function works for different N's, run the command below:

    python3 gpt.py part1 -N <val>



## Part 2: Blocked Matrix Multiply and Unfused Softmax
In this part, we optimize the matrix multiplication in part 1(i.e., $Q K^T$ and
$PV$) with blocked matrix multiply. The implementation can be found in the myUnfusedAttentionBlocked function in module.cpp. 

### How does the blocked matrix multiply works?

<p align="center">
  <img src="https://github.com/stanford-cs149/cs149gpt/blob/main/assets/current_matmul.png" width=40% height=40%>
</p>

The above cache access pattern is extremely poor. Assuming each cache line can contain 8 float numbers. The above matrix multiply will generate $(\frac{n}{8} + n) \times n^2 = \frac{9n^3}{8}$ cache misses.

Hence, we need to optimize it with [blocked matrix multiply](https://csapp.cs.cmu.edu/public/waside/waside-blocking.pdf). 

<p align="center">
  <img src="https://github.com/stanford-cs149/cs149gpt/blob/main/assets/blocked_matmul.png" width=40% height=40%>
</p>

As shown above, the cache misses for each sub-block will be $\frac{B^2}{8}$. Hence, overall cache miss will be $(\frac{B^2}{8} + \frac{B^2}{8} ) \times \frac{n}{B} \times (\frac{n}{B})^2 = \frac{n^3}{4B} $, which dramatically decreases cache misses.

### Testing:
Run the following test to test program's correctness:

    python3 gpt.py part2

A correct implementation should yield the following output:

    REFERENCE - BLOCKED MATMUL + UNFUSED SOFTMAX statistics
    cpu time:  156.271ms
    mem usage:  4718588 bytes

    STUDENT - BLOCKED MATMUL + UNFUSED SOFTMAX statistics
    cpu time:  160.891ms
    mem usage:  4718588 bytes

An incorrect implementation will have the output:

    ATTENTION PRODUCED INCORRECT RESULTS


The function works for different values of N, as this is how we will be grading you on correctness. We have provided a command line argument that works as so:

    python3 gpt.py part2 -N <val>

We can see the DNN use the attention layer to generate text, optionally changing the model to shakes256, shakes1024, or shakes2048 if you wish to output more text:

   python3 gpt.py part2 --inference -m shakes128



## Part 3: Fused Attention with thread-level parallelism
The implementation of this function can be found in myFusedAttention function in module.cpp file. 

By now we've seen that multiplying $Q * K^{T}$ results in a massive NxN matrix. Doing the matrix multiplies and softmax in seperate functions requies that we write each row of our NxN matrix, and then do another pass over this NxN matrix in the subsequent softmax, and then do a third pass over the softmax'd matrix when multipling it by V. Not only is this bad for cache performance, but it is very bad for our program's memory footprint. 

Fortunately, we can resolve both issues by "fusing" the calculation, such that we only require one Nx1 temporary vector instead of an NxN temporary matrix.

You can do this by observing the following fact. Once we've calculated a single row of the $Q * K^t$ NxN matrix, we are actually ready to softmax that entire row, and we don't have to calculate the rest of the NxN matrix to do so.

Once that row is softmax'd, we can then immediately multiply the softmax'd row by V to fully compute the first row of our attention output (which is of reasonable size: Nxd). In other words, we can calculate just one row of $Q * K^{t}$, softmax it, then multiply that softmax's row by V. Doing this does not require creating the NxN matrix...it requires creating only one Nx1 size intermediate vector to hold the first row of $Q*K^{t}$ and then its softmax. We can then re-use this same Nx1 array to calculate the 2nd row of attention, and then the third, etc. This means that we never materialize the NxN matrix, which is great because that matrix is never used again later in the network anyways. 

### Parallelizing with OpenMP
The thread-level parallelism is implemented with the below OpenMP command.

    #pragma omp parallel for collapse(3)
    
    -- code here --



### Testing:
Run the following test to check your program's correctness:

    python3 gpt.py part3

A correct implementation should yield the following output:

    REFERENCE - FUSED ATTENTION statistics
    cpu time:  32.361ms
    mem usage:  557052 bytes

    STUDENT - FUSED ATTENTION statistics
    cpu time:  33.209ms
    mem usage:  557052 bytes

An incorrect implementation will have the output:

     ATTENTION PRODUCED INCORRECT RESULTS



The function should work for different values of N, to test it, run the below command:

    python3 gpt.py part3 -N <val>

We can use our attention layer to generate text, optionally changing the model to `shakes256`, `shakes1024`, or `shakes2048` if you wish to output more text:

    python3 gpt.py part3 --inference -m shakes128



## Part 4 : Flash Attention
The implementation can be found in myFlashAttention function in module.cpp file.

### Why Are Matrix Multiply and Softmax Hard to Fuse as Blocks?
The attention formula is very awkward to fuse for a couple reasons. Notice how the formula consists of a matrix multiply, followed by a row-wise calculation from softmax, and concluded with another matrix multiplication. The true thing that makes it difficult from fusing these three operations as blocks is the fact that softmax has to operate on the entire row. So, if we want to bypass this dependency we really have to think outside the box. That is where Flash Attention comes in.

### Breaking Softmax into Blocks
Let's say that we have a BLOCKSIZE vector, we will denote it as $x \in \mathbb{R}^{B}$.The softmax of $x$ can be formulated as:

<p align="center">
  <img src="https://github.com/stanford-cs149/cs149gpt/blob/main/assets/Softmax_decomp1.png" width=55% height=55%>
</p>

It follows that if we have two BLOCKSIZE vectors, denoted as $x \in \mathbb{R}^{B}$ and $y \in \mathbb{R}^{B}$, then we can decompose $softmax([x\ y]$ as:

<p align="center">
  <img src="https://github.com/stanford-cs149/cs149gpt/blob/main/assets/Softmax_decomp2.png" width=55% height=55%>
</p>


### Implement Flash Attention
I broke softmax into blocks so it can be fused with our blocked matrix multiply. Therefore, for each block, we will multiply $Q$ (BLOCKROWSIZE x d) with $K^{t}$ (d x BLOCKCOLUMNSIZE) to get $QK^t$ (BLOCKROWSIZE x BLOCKCOLUMNSIZE). Then, we will calculate $\texttt{softmax}(QK^t)$ (BLOCKROWSIZE x BLOCKCOLUMNSIZE) and multiply this with $V$ (BLOCKCOLUMNSIZE x d) to get $O$ (BLOCKROWSIZE x d). This is an accumulative process just like blocked matrix multiply.

By doing this we can significantly decrease the memory footprint. Rather than having a memory footprint of $O(N^{2})$, we will be able to reduce this to a linear scaling footprint of $O(N)$.

### Flash Attention Pseudocode
The flash attention algorithm shown below(it's slightly different from the original paper) imports blocks of the matrices $Q$, $K$, and $V$ into smaller physical tiles(in GPU it is shared memory). It then computes a local softmax in each tile, and then writes this result tile back to the full output matrix $O$. For $Q$, for example, each tile's size is (Br x d), and the tile size for $K$ is (Bc x d). Calculating $Br$ and $Bc$, as shown in the pseudocode below, requires knowing the size $M$ of your SRAM/cache, which in this case is $M=131072$ floats.

<p align="center">
  <img src="https://github.com/stanford-cs149/cs149gpt/blob/main/assets/FlashAttentionPseudo.png" width=65% height=65%>
</p>

### Testing:
Run the following test to check the program's correctness:

    python3 gpt.py part4

**Testing on program's correctness** 


A correct implementation should yield the following output:

    REFERENCE - FLASH ATTENTION statistics
    cpu time:  435.709ms
    mem usage:  524284 bytes

    STUDENT - FLASH ATTENTION statistics
    cpu time:  435.937ms
    mem usage:  524284 bytes

An incorrect implementation will have the output:

    ATTENTION PRODUCED INCORRECT RESULTS

**Test on different values of N, br and bc**

The program should work for different values of N, br and bc. To test it, run the below command:

    python3 gpt.py part4 -N <val> -br <val> -bc <val>


Now, we can see the DNN use our attention layer to generate text, optionally changing the model to `shakes256`, `shakes1024`, or `shakes2048` if you wish to output more text:

    python3 gpt.py part4 --inference -m shakes128


