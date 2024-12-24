---
title: Parallel Promming HW4 Report

---

# <center>Parallel Programming HW4</center>

<center>113062518 陳家輝</center>

## Implementation
1. Describe how you implemented the FlashAttention forward pass using CUDA. Mention the algorithm's key steps, such as matrix blocking, SRAM usage, and how intermediate results like scaling factors(l and m) were calculated.
    - 首先觀察FlashAttention的sequential code，可以發現在function內，已經將多個計算步驟拆分成多個sub function進行呼叫，讓整體架構看起來清晰明了。而在每個sub function內，大致上都是由`for loop`所組成，並且沒有太多的dependency，代表該演算法具有很好的平行架構。
    - 在轉換成CUDA版本時，主要就是將一個block的計算進行平行，並且盡量將沒有dependency的loop進行平展，充分利用GPU的平行計算資源。因為原先sequential的code所使用的`br`和`bc`都是32，剛好可以設定GPU一個block的size為32x32，這樣整體的架構就不需要太多變更。
    - 在一個block計算前，需要從GPU的`Global Memory`當中將計算一個`output block`所需要的data都搬進SRAM裡面，從而提高memory access的速度，降低memory bound的情況。而因為`dim`不是32就是64，剛好會是`BLOCK_SIZE`的整數倍，所以在將data放進SRAM的時候，僅需根據`dim`的大小，就可以決定一個thread需要搬一個data還是兩個data，這樣就可以將`Q`，`K`，`V`和`O`正確的size放入SRAM。
    ![圖片](https://i.imgur.com/LzEfOgO.png)
    - 此處需要注意的是，因為`K`和`V`在`attention`的計算中是需要做`Transpose`的，在原本的sequential code中是先執行`column loop`再執行`row loop`，所以我們在搬移資料放進SRAM時也需要注意`K`和`V`是column major。那麼在index時，就不是平常所使用的的`thread_y * dim + thread_x`，而是要將`thread_y`和`thread_x`調換，變成`thread_x * dim + thread_y`。並且當`dim=64`時，就只需要讓每個thread在其index上往後`BLOCK_SIZE`，多load一個data放進SRAM即可。
    ![圖片](https://i.imgur.com/gXWRntl.png)
    - 將data都正確的放進SRAM之後，後續的計算就顯得比較簡單了，就是將原本sequential code沒有dependency的`for loop`進行平展，讓每個threads平行計算。但需要注意的是，因為在多個步驟內包含了`Reduction`或是`Maximize`的操作，這一類型的`for loop`就沒有辦法平行，需要按照原本sequential的方式進行計算。另外還有一個細節比較容易遺漏的是，在每個步驟結束後，需要加上`__syncthreads()`，確保每一個thread都已經執行完該步驟，才可以繼續往下，否則下一個步驟就有可能使用到錯誤的值。
    ![圖片](https://i.imgur.com/ThpWewl.png)
    - 在演算法中，最重要的就是如何更新`l`和`m`，並且使用它們去計算正確的`O`。原先我的寫法是整個FlashAttention只會launch一次kernel，也就是說，整個Matrix的計算會被全部拆成blocks，分別放進GPU中進行平行計算。但有一個問題是，同一個row中，`l`和`m`的值是會有dependency存在，若是剛好GPU同時計算了同一個row中的兩個block，那麼這兩個block會使用到同一組`l`和`m`，可是正確的計算方式應該是一個block計算完，更新其對應的`l`和`m`之後，下一個block才會使用這一組新的`l`和`m`進行計算，這樣的平行方式就會導致錯誤。
    - 所以我將kernel launch改成一次只會計算同一個column，不同row的blocks，所以就會有`N / BLOCK_SIZE`次kernel launch，這樣`l`和`m`的計算就不會存在dependency，因為不同row所對應的`li`和`mi`會是不同的，就可以解決dependency的問題，讓整個FlashAttention可以正確的平行計算。
    ![圖片](https://i.imgur.com/pJ2NBp5.png)
    - 而在kernel裡最後一步的計算中，需要去更新`l`和`m`，因為我們已經解掉dependency的問題，此時的平行就會比較容易，只需要每個row使用一個thread去做處理，就可以將前面步驟中所計算出來的中間值`mij`和`lij`納入更新計算，從而得到正確的`li`和`mi`。
    ![圖片](https://i.imgur.com/OvT6aUH.png)
    - 最後，就是使用更新過的`l`和`m`去更新`O`，此處比較需要注意的地方是，原先sequential code是使用三個`for loop`進行計算，其中第二個`for loop`是遍歷`dim`。一開始我以為有關於`dim`的loop都不能平行，所以我只有對最外圈進行平行，但這樣會使得最後幾個testcases出現TLE的結果。而就在我仔細拆解運算之後，發現只有最內層的`Reduction`是具有dependency，反而`dim`這個loop是independent的，所以我才又對第二層`for loop`進行平行，這樣就可以讓全部的測資都能順利通過。
    ![圖片](https://i.imgur.com/OT7AYJP.png)

2. Explain how matrices Q, K and V are divided into blocks and processed in parallel.
    - 在kernel一個block的計算當中，`Q`，`K`和`V`會被切分成`BLOCK_SIZE * dim`的大小，而又因為一個block的大小是`BLOCK_SIZE * BLOCK_SIZE`，其中`BLOCK_SIZE=32`，所以在使用不同的threads將data從HBM放進SRAM時，僅需要考慮`dim`的大小是否為64，從而決定一個thread需要搬移一個或兩個data。
    - 而在資料搬移時，也需要注意`Q`是row major，`K`和`V`則是column major。在做index時需要將`x`和`y`做對換，這樣計算時才會是正確的計算`Transpose`。
    - 那麼在對`Q`，`K`和`V`做計算時，原先sequential code的計算都會使用兩個`for loop`遍歷matrix，此時就可以簡單的使用`thread_x`和`thread_y`將loop平展，從而達到平行計算的優化。主要需要小心的地方就是要記得`K`和`V`是column major，index時要使用`thread_x`才不會使用到錯誤的值。
    ![圖片](https://i.imgur.com/RJZakvX.png)
    
3. Describe how you chose the block size B_r and B_c and why.
    - 在`B_r`和`B_c`的選擇上，正如前面所提到，因為sequential code中`B_r`和`B_c`的設定為32和32，所以我就沿用sequential code的設定，同樣使用32和32。這麼做的好處不僅是整個架構不需要做太大的變更，另一個好處便是將`BLOCK_SIZE`同樣設定為32時，32x32=1024 threads per block剛好可以將一個block內的計算資源全部用滿，達到fully utilization。
    - 還有另外一層原因是因為`dim`的大小不是32就是64，若是將`BLOCK_SIZE`剛好設定為32，在處理`dim=64`的情況就會非常容易，那麼再將`B_r`和`B_c`直接設定為`BLOCK_SIZE`，就可以更加簡化實作的過程。

4. Specify the configurations for CUDA kernel launches, such as the number of threads per block, shared memory allocation, and grid dimensions.
    - `number of threads per block`：正如前面所提到，我將`BLOCK_SIZE`設定為32，那麼一個block就總共會有32x32=1024個threads，並且因為device的`maxThreadsPerBlock`也是1024，就可以fully utilize一個block內的所有計算資源。
    - `shared memory allocation`：在一個block內的計算，我會將一個block的`qi`, `kj`, `vj`, `oi`， `li`和`mi`給load進來，並且為了加速計算，中間值我同樣也先在SRAM上allocate了一塊memory進行儲存，這樣所有allocate的memory加起來總共是`41728 bytes`，而該device的`sharedMemPerBlock`為`49152`，對比起我使用總量，已經非常接近最大值，代表資源已經很大程度的進行了利用。
    ![圖片](https://i.imgur.com/iaGJSNR.png)
    - `grid dim`：原先我的`grid dim`設定為`(N / BLOCK_SIZE, N / BLOCK_SIZE)`，這樣就可以將整個matrix拆分成blocks進行計算。但是後來發現同一個row中`li`和`mi`存在著dependency，不能一次launch所有的blocks，否則答案會有錯誤，所以我將`grid dim`修改成`(1, N / BLOCK_SIZE)`，這樣就會是同一個column的blocks會一起launch，而且不會有dependency的問題。
    ![圖片](https://i.imgur.com/68vDbjS.png)
    
5. Justify your choices and how they relate to the blocking factors and the SRAM size.
    - 參數選擇的合理性說明已經在question 4中進行了詳述，該題專注在說明這些參數的選擇是如何與`blocking factor`和`SRAM size`相關。
    - `blocking factor`：選用`BLOCK_SIZE=32`作為`blocking factor`時，這個參數的設定剛好可以對應原先sequential code的設定，還可以fully utilize一個blocke內所有的threads，並且在`dim`的大小為64時，可以更容易的處理邊界狀況，使得整體的實作可以更加簡潔。
    - `SRAM size`：根據先前的參數設定，一個block內的SRAM用量為`41728 bytes`，約為最大size的`49152 bytes`的85%，這顯現出我所選擇的參數在allocate SRAM memory時，可以很大程度的利用資源，並且不會有超出硬體限制的問題，讓code整體的表現可以更好。

## Profiling Result

| Kernel Name            | Metric Name             | Min        | Max        | Avg        |
| ---------------------- | ----------------------- | ---------- | ---------- | ---------- |
| flash_attention_kernel | achieved_occupancy      | 0.977967   | 0.983150   | 0.980594   |
| flash_attention_kernel | sm_efficiency           | 95.66%     | 98.60%     | 97.98%     |
| flash_attention_kernel | shared_load_throughput  | 2837.1GB/s | 2895.2GB/s | 2876.3GB/s |
| flash_attention_kernel | shared_store_throughput | 198.71GB/s | 202.78GB/s | 201.45GB/s |
| flash_attention_kernel | gld_throughput          | 402.93GB/s | 411.19GB/s | 408.50GB/s |
| flash_attention_kernel | gst_throughput          | 27.598GB/s | 28.164GB/s | 27.980GB/s |

## Experiment & Analysis

### System Spec
- 本次作業的實作和實驗都是在`apollo-gpu`的server上進行。
- 在各項實驗當中，testcase的選用上都是使用`t30`作為測量，該測資在最終優化版本的code上的執行時間約為`0.92s`，應是可以很好的觀察各項實驗數據的差異，並且進行有效的分析。
- 若有其他情況導致選擇的testcase不同，會在各個實驗中進行闡述。

### Optimization
- 在本次作業中，我主要實作的優化方法有`Shared Memory`和`Handle Bank Conflict`，而因為實作完這兩項之後，基本上每一個testcase就已經是2s以下，並且本次作業並不需要衝榜，就沒有再多做其他的優化。以下是兩個優化方法的詳細介紹，以及各個實作版本的時間對比圖。
- `Shared Memory`：因為`FlashAttention`本身就需要將`Global Memory`的data給搬移到`SRAM`當中進行`Memory Access`的加速，所以為了完成這個演算法的實作，本身就會使用的`Shared Memory`進行優化，那麼具體的實作過程已經在上面的`Implementation`詳細介紹過了，此處就不在贅述。
- `Handle Bank Conflict`：`GPU`的`shared memory`是由多個 bank 組成，如果大家同時存取的`address`都落在同一個`bank`，就會造成conflict的發生。而且在本次作業中，`dim`恰好是 32或64 剛好會與`warp size`對其，就會容易產生`Bank Conflict`。
- 為了解決這個問題，首先需要做的就是將`shared memory`的每一個column結尾多加1個`Padding`，這樣就不會剛好對齊`bank`的邊界，而會是相互錯開，就可以降低threads同時在使用同一個`bank`的情形。
![image](https://i.imgur.com/faBfcfq.png)
- 隨後在資料搬移上，因為此時我們已經對`Shared Memory`進行了`Padding`，所以在做index的時候，都要使用`Padding`過後的dimension，而`Global Memory`的index則是不變。
![image](https://i.imgur.com/A2A7vcJ.png)
- 最後則是在計算時，使用到`Shared Memory`的地方都要記得改成使用`Padding`的dimension，這樣就完成了`Handle Bank Conflict`的優化實作。
![image](https://i.imgur.com/HX1cPT4.png)
- 以下是不同優化版本的時間比較圖：
![image](https://i.imgur.com/KsNPxKm.png)
- 從圖中可以看出，最終優化的版本比起原先的版本快了20倍左右，達到了很好的優化效果。

### Memory Access Compare：
- 因為`FlashAttention`的想法是將需要計算的資料從`HBM`搬到`SRAM`，從而加速`Memory Access`的時間，所以我想知道普通版本的`Attention`在`Memory Access Throughput`上，會與`FlashAttention`差距多少。
- 為了執行該項實驗，我簡單對`seq-attention`這個版本的sequential code實作了一個`cuda`版本的code，在這份code中，就沒有使用到`Shared Memory`等優化技巧，單純使用`Global Memory`進行值的存取。如下圖，`attn`和`v`都是直接從`Global Memory`進行存取。
![image](https://i.imgur.com/MedLJ2W.png)
- 隨後我使用該份code取得了`Global Memory load/store Bandwidth`的指標，如下：
    | Kernel Name     | Metric Name    | Min        | Max        | Avg         |
    | --------------- | -------------- | ---------- | ---------- | ---------- |
    | AttentionKernel | gld_throughput | 772.97GB/s | 794.03GB/s | 783.36GB/s |
    | AttentionKernel | gst_throughput | 51.954GB/s | 53.607GB/s | 52.767GB/s |
- 最後，將該數據的`Max`取出，並且與`FlashAttention`的`Shared Memory load/store Bandwidth`當中的`Max`進行繪圖比較，實驗結果圖如下：
![image](https://i.imgur.com/FOmBbWq.png)
- 從圖中可以看出，`FlashAttention`在使用了`SRAM`的情況下，`Memory Access`的`Bandwidth`都可以比`Attention`使用`HBM`快上將近4倍，在`Memory Bandwidth`較高的情況下，計算的效能也會隨之增幅，從而使整體的效能提升。

## Experience & Conclusion
- 本次作業的實作和優化比起`hw3`來說快上了不少，可能是因為在`hw3`中經歷了漫長的磨練，所以在實作`hw4`的時候顯得比較得心應手了一些，而且因為本身對於`FlashAttention`具備了一定的了解，在將sequential code轉成`GPU baseline`的時候更是迅速，但也有一個關鍵的原因是因為本次作業的`input dim`剛好是32或是64，使得實作起來需要考慮的邊界狀況大幅減少，讓整個架構可以更加簡潔。
- 經過本次作業，我更加熟悉了`FlashAttention`的運作，也更加了解到`Memory Access`的加速對於高效能計算有多麼重要，若是能將這個想法應用在其他常見的`AI Algorithm`當中，想必可以更加縮短訓練所需要的時間，從而幫助研究快速進行。