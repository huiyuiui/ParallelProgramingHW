---
title: Parallel Programming Hw3 Report

---

# <center>Parallel Programming HW3</center>

<center>113062518 陳家輝</center>

## Implementation
1. Which algorithm do you choose in hw3-1?
    - 在hw3-1當中，因為hw3-2和hw3-3都要求使用`Blocked Floyd-Warshall Algorithm`，所以我就想說不做普通版本的`Floyd-Warshall`，直接實作`Blocked`版本的，這樣在後續的GPU實作中也會更加了解演算法如何運作。
    - 大致上code的實作都是直接沿用sequential code，而在平行優化上是使用`OpenMP`進行`threading`的平行，將`#pragma omp parallel for`加在對於一個Block內的Distance的計算迴圈上，便可以達成對於`Blocked Floyd-Warshall`的平行。
    ![image](https://i.imgur.com/ul2SDQg.png)
    
2. How do you divide your data in hw3-2, hw3-3?
    - 在hw3-2當中，我沿用hw3-1的code的架構，也就是將phase2和3同樣分成四個function進行分別計算。
    ![image](https://i.imgur.com/KJxiwuz.png)
    - 以phase3為例，如圖，我將data切分成：左上，右上，左下，右下四個區域進行計算，每個區域會根據目前的`block_start_x/y`和`height/width_num_blocks`決定該區域需要計算的範圍。
    ![image](https://i.imgur.com/YVl33eH.png)
    - 再來是hw3-3的data切分，因為此時使用了兩張GPU進行平行計算，在分別計算的過程中還會需要兩張GPU進行溝通，以保證雙方此時計算的data都會是最新的。所以我將整個distance matrix分成上半和下半，分別由兩個GPU負責，在個別計算完之後再進行data的溝通。
    ![image](https://i.imgur.com/DnyT4XX.png)
    - 而因為hw3-3要同時操控兩個GPU，所以我使用課程所教的方法，使用`OpenMP`創建兩個threads，使用這兩個threads分別操控兩個GPU。如圖，每個GPU負責的範圍就會使用`thread_id`進行`thread_start`和`thread_range`的計算，這兩個變數分別代表了兩個GPU各自的起始位置和計算範圍。
    
3. What's your configuration in hw3-2, hw3-3? And Why?(e.g. blockingfactor, #blocks, #threads)
    - 在configuration的選擇上，首先我有執行圖中的兩行code，得到device的property後再做決定。根據打印的資訊，`maxThreadsPerBlock=1024`，若要使計算資源可以最大化的利用，那麼`blocking factor`就要選擇`32x32`，這樣一個block的threads數剛好就會是1024。
    ![image](https://i.imgur.com/U02CmTK.png)
    - 而因為我的code沿用hw3-1的架構，所以一次計算的`#blocks`是由傳進`cal_function`的參數進行決定。正如前面所提到的，我會將phase2和3分開四個區域進行計算，那麼每個區域需要計算的範圍都不同，就需要根據目前的`round`計算出各自的範圍，才能決定該GPU kernel需要一次計算幾個blocks。
    ![image](https://i.imgur.com/ykMW654.png)
    - 如圖，其中`B`代表的是`block_size=32`，是一個常數。而`width/height_num_blocks`則是傳入`cal_function`的參數，決定了這一次kernel所需要計算的範圍是多大，從而讓kernel可以一次執行這麼多個blocks的計算。
    - 而在hw3-3當中，`#threads`的選擇就比較單純，因為要同時操控兩個GPU，所以我便按照課堂所教的那樣，使用兩個threads並且搭配`omp parallel`分別操作兩個GPU。
     
4. How do you implement the communication in hw3-3?
    - 在hw3-3的溝通上，正如前面所說，我將整個distance matrix切成上半和下半，分別由兩個GPU負責。具體的實作是在每一個round開始前，根據目前的round是在上半部還是下半部，決定此次data的溝通是哪個GPU需要進行傳輸。若是在上半，則是`GPU 0`需要將data傳輸給`GPU 1`，反之亦然。
    ![image](https://i.imgur.com/JZFpojR.png)
    - 在傳輸時，因為實作上的方便，所以我直接將該round的`pivot block`所在的一整個row傳輸給對方。而因為此時雙方都有最新的`pivot block`，所以就可以分別計算phase1和phase2的更新，那麼在計算phase3時，所需要的`dependent blocks`都會是最新的。此時，兩個GPU就可以根據各自負責的上半或是下半分別更新各自的distance matrix，直到下一個round的時候再做溝通，從而保證兩個GPU在做各自計算時所需要的data都會是最新的。
    ![image](https://i.imgur.com/RfCEb8m.png)
    - 在此處還有一點需要提到的是，原先phase3的實作是切分成四個區域進行計算，但是此時兩個GPU會各自負責上半和下半，那麼四個區域的切分會變的比較困難，所以我將其改成各自的kernel直接一次計算半個distance matrix大小的blocks數。而因為這樣的計算範圍可能會包含到`pivot`所在的row或column，這些blocks是在phase1和phase2就已經更新過，不能再被計算，所以我在kernel的計算內多加了一行判斷式，從而保證更新的過程是正確的。
    ![image](https://i.imgur.com/Cb6LCmP.png)

    
5. Briefly describe your implementation in diagrams, figures or sentences.
    - 因為hw3-1大部分沿用sequential code，使用`OpenMP`進行`threading`的平行也在前面有介紹，所以在這裡直接從hw3-2開始介紹。
    - 在hw3-2內，我沿用了hw3-1的架構，將一個round內的三個phase的計算分別呼叫各自的function，並且傳入各自需要計算的範圍和起點，如圖。
    ![image](https://i.imgur.com/XP1yjDi.png)
    - 而在function內，就是根據傳入的參數，設定此次計算的kernel需要一次launch多少個blocks，並且同樣將要計算的起點傳入kernel，從而確保在kernel內的計算範圍是合理的。而需要注意的一點是，phase1的計算只有一個`pivot block`，所以在phase1的kernel計算上，只需要launch一個block即可。
    ![image](https://i.imgur.com/ZRXhYIr.png)
    - 而在各個不同phase的kernel裡的計算上，則是應用了`Shared Memory`，`Padding`，`Register`和`Unroll`等優化的技巧，這部分的實作會在後續的report中再做詳細介紹。
    - 最後是hw3-3的實作，因為multi-gpu的計算和single-gpu差異並沒有很大，所以大部分的code和hw3-2都是一樣的，例如phase1和phase2的計算就都沒有改變。而需要改動的地方就是兩個GPU各自在phase3的計算，以及每一個round開始前需要進行溝通，這些實作的內容都有在前面的report中提到，這裡就簡單的再提到一點需要注意的是，在GPU各自的計算開始前，需要保證資料的搬移和溝通都已經完成，否則就有可能拿到舊的值進行計算，從而導致答案錯誤。
    ![image](https://i.imgur.com/WrErytI.png)
    - 如圖，我在資料從host搬移到device以及GPU之間的溝通後，都有加上`#pragma omp barrier`，以確保兩個GPU的資料都已經是最新的狀態，到達這個barrier之後，才會繼續往下計算，從而保證後續的計算都可以基於最新的data。
    
## Profiling Results

- 在執行`profiling`時，因為要求使用`biggset kernel`，所以我選用的是testcases當中的`p30k1`，而這是我通過的最久的一筆測資，使用其來做`profiling`應是可以很好的看出各項效能指標。

- occupancy

| Kernel Name  | Metric Name        | Min      | Max      | Avg      |
| ------------ | ------------------ | -------- | -------- | -------- |
| KernelPhase1 | achieved_occupancy | 0.497922 | 0.498034 | 0.497960 |
| KernelPhase2 | achieved_occupancy | 0.487079 | 0.944682 | 0.894846 |
| KernelPhase3 | achieved_occupancy | 0.487570 | 0.926401 | 0.919822 |

- sm efficiency

| Kernel Name  | Metric Name   | Min   | Max    | Avg    |
| ------------ | ------------- | ----- | ------ | ------ |
| KernelPhase1 | sm_efficiency | 4.43% | 4.54%  | 4.50%  |
| KernelPhase2 | sm_efficiency | 3.29% | 96.52% | 85.84% |
| KernelPhase3 | sm_efficiency | 3.36% | 99.98% | 99.18% |

- shared memory load throughput

| Kernel Name  | Metric Name            | Min        | Max        | Avg        |
| ------------ | ---------------------- | ---------- | ---------- | ---------- |
| KernelPhase1 | shared_load_throughput | 113.47GB/s | 123.03GB/s | 119.06GB/s |
| KernelPhase2 | shared_load_throughput | 111.65GB/s | 3452.1GB/s | 3048.1GB/s |
| KernelPhase3 | shared_load_throughput | 106.46GB/s | 3718.5GB/s | 3467.5GB/s |


- shared memory store throughput

| Kernel Name  | Metric Name             | Min        | Max        | Avg        |
| ------------ | ----------------------- | ---------- | ---------- | ---------- |
| KernelPhase1 | shared_store_throughput | 37.119GB/s | 41.547GB/s | 39.983GB/s |
| KernelPhase2 | shared_store_throughput | 4.5198GB/s | 143.94GB/s | 126.31GB/s |
| KernelPhase3 | shared_store_throughput | 4.3747GB/s | 156.05GB/s | 144.19GB/s |



- global load throughput 


| Kernel Name  | Metric Name    | Min        | Max        | Avg        |
| ------------ | -------------- | ---------- | ---------- | ---------- |
| KernelPhase1 | gld_throughput | 314.00MB/s | 372.16MB/s | 341.21MB/s |
| KernelPhase2 | gld_throughput | 790.08MB/s | 17.417GB/s | 15.203GB/s |
| KernelPhase3 | gld_throughput | 840.88MB/s | 18.744GB/s | 17.108GB/s |



- global store throughput 

| Kernel Name  | Metric Name    | Min        | Max        | Avg        |
| ------------ | -------------- | ---------- | ---------- | ---------- |
| KernelPhase1 | gst_throughput | 573.10MB/s | 646.73MB/s | 610.63MB/s |
| KernelPhase2 | gst_throughput | 2.2076GB/s | 71.075GB/s | 62.056GB/s |
| KernelPhase3 | gst_throughput | 2.1287GB/s | 76.495GB/s | 70.855GB/s |


## Experiment & Analysis

### System Spec
- 本次作業的實作和實驗都是在`apollo-gpu`的server上進行。
- 為了實驗方便，testcase的選用上基本上是使用`p15k1`作為測量，因為更大的testcase會在獲取實驗數據時發生`TLE`的狀況，而該測資在hw3-2的優化版本的code上的執行時間約為5s，應是可以很好的觀察各項實驗數據的差異，並且進行有效的分析。
- 若有其他情況導致選擇的testcase不同，會在各個實驗中進行闡述。

### Blocking Factor

- 因為`metrics`的結果同時會有`min`, `Max`和`avg`三項數據，為了實驗數據統一，以下的實驗結果都是以`Max`的值作為討論。
- 在`computation performance`中，我的測量方式是使用`inst_integer`獲得總共執行的`int instructions`數量，再取得kernel的執行時間，將兩者相除並調整時間單位，就可以得到`Integer GOPS`。而因為使用`inst_integer`測量時，經常會出現`TLE`，所以在該實驗上我一路往下降級，最終選擇的testcase為`c21.1`。
- 在`global/shared memory bandwidth`中，會有`load`和`store`兩項動作，我先取出兩者的`Max`之後，將兩者的數據取平均，作為整體bandwidth的表現。
- 而因為我的實作會將演算法分為三個phase分別呼叫kernel，其中phase1只有計算pivot block，在各項數據上與phase2和3差距甚遠，會導致所繪畫出的圖表難以呈現，所以此處就只討論計算和memory access最多的phase2和3。以下是實驗圖。
- `Computation Performance`
![image](https://i.imgur.com/avPjADe.png)
- 從圖中可以看出，當`Blocking Factor`越大時，`Integer GOPS`就會隨之越大，並且成長幅度很大，這應是因為當一個block可以處理的資料越多時，就可以將資料一次放進`shared memory`當中，那麼在後續計算時，`memory bound`的情形就可以減少，從而使得計算的效能可以提升。
- `Global Memory Performance`
![image](https://i.imgur.com/dxX07X4.png)
- `Shared Memory Performance`
![image](https://i.imgur.com/2YerUYl.png)
- 從`memory performance`的圖中可以看出，當`blocking factor`越大時，不論是`global memory`還是`shared memory`的bandwidth，基本上都可以得到有效的提升，代表一次處理越多資料時，越可以有效的利用資源，從而讓整體的效能提升。
- 但在`Global Memory Performance`的`Blocking Factor`為64時，其bandwidth相比起32其實是下降的，這個原因我推測應該是因為我是手動將`Blocking Factor`調整成64(該優化實作會在後續`Optimization`處詳述)，也就是說，在一個block的計算內，一個thread需要同時處理四個點的`load`和`store`，這樣才可以將原本`BLOCK_SIZE=32`變成`BlockingFactor=64`。雖然讓一個block可以計算的點變多了，但也讓memory的bandwidth下降，說明此處是存在著一個balance的。


### Optimziation (hw3-2):

- 在hw3-2的optimziation上，我總共實作了5個優化方法，分別是：`Shared Memory`, `Padding`, `Blocking Factor Tunning`, `Register`, `Unroll`。以下會分別對各個優化進行詳述，隨後繪製效能圖。
- `Shared Memory`：將`GPU baseline`進行優化的第一個方法就是使用`Shared Memory`對memory access進行加速。具體的做法就是先宣告一塊`shared memory`，再根據該threads的index去將對應的`global memory`的data放進`shared memory`，需要注意的是，因為`n`有可能不是`BLOCK_SIZE`的整數倍，所以會有邊界狀況的出現，在搬移資料前要先處理邊界狀況再做搬移，否則就會出錯。將資料放進`shared memory`後，同一個block內的threads都可以存取這一塊memory，在計算時就可以達到更快的memory access的速度。
![image](https://i.imgur.com/QLHr76V.png)
- `Padding`：因為在將資料搬移到`shared memory`時，需要處理邊界狀況，這會使得有某些warp內的執行情況不一，導致`warp divergence`，所以我將整個distance matrix進行`padding`，把size變成`BLOCK_SIZE`的整數倍，這樣就不用處理邊界狀況，使每一個warp內的執行狀況都可以一致。
![image](https://i.imgur.com/Vub0xib.png)
- `Blocking Factor Tunning`：原先我所使用的`BLOCK_SIZE`是32，因為這樣剛好可以將`maxThreadsPerBlock`用滿。在後續優化中，我有嘗試對`BLOCK_SIZE`進行調整，但將其調小只會使效能變慢，而又因為一個block的threads最多就是1024，沒有辦法再把32調大。後來在觀察device的property時發現，`sharedMemPerBlock`是`49152 bytes`，使用`BLOCK_SIZE=32`時的`shared memory`用量最多才`12288 bytes`。所以我將code修改成`BLOCK_SIZE`一樣維持32，這是為了遵循硬體的限制，但是在一個kernel內的計算，讓一個thread同時計算4個點，這樣本來`32x32`的block就會變成同時處理`64x64`的block，也就是說`Blocking Factor`會變成64。經過修改後，`shared memory`的用量也是剛好用滿`49152 bytes`，達到資源的fully utilization，從而使效能進一步提升。
![image](https://i.imgur.com/2ADTJp5.png)
- `Register`：觀察在kernel內的計算，可以發現需要更新的點(ex: `Dist[i][j]`)在`k`這個loop內會一直被`load`出來與新計算的distance進行minimum的比較，然後再寫回去。但是取minimum這件事其實可以不需要一直從`shared memory`中將值取出再存入，可以使用`register`將值進行儲存並比較，最後在loop結束後，將該`register`的值再存回`global memory`當中，就可以大幅減少memory access的時間。
![image](https://i.imgur.com/b8JDNP9.png)
- `Unroll`：最後一個優化是對計算的`k` loop進行`unroll`，起初我以為不論是哪一個phase，在計算時都會有dependecy，所以每執行一次loop我就會使用`__syncthreads`進行同步，導致無法使用`unroll`進行平展。隨後我仔細將phase3的計算拆解之後發現，在`other blocks`內的計算只會依賴於`pivot row/column`，不會依賴自己block內的data，而`pivot row/column`的data在phase2就已經計算完成，那麼在phase3的計算中，是不會有dependecy的。因此，我就將`#pragma unroll`應用在phase3的計算中，從而進一步提升效能。
![image](https://i.imgur.com/YBLTjc2.png)
- 在優化方法的時間測量上，我是直接使用`clock_gettime`放在演算法前後，從而得到優化後演算法的執行時間。以下是我實作的優化方法的時間圖，每一個優化方法都是基於前面的方法再作改進，所以效能會越來越好，使得執行時間可以越來越少。
![image](https://i.imgur.com/h2ExDKo.png)
- 從圖中可以看出，在`p15k1`這個testcase下，從原本baseline的32秒進步到最終優化版本的1.2秒，`Speedup Facotr`達到了將近30倍，說明了這些優化方法可以很大程度的增進程式在計算上的效能。
- 而此處並沒有放`CPU`的執行時間是因為該testcase下`CPU`的執行時間需要幾百秒，繪製成圖形後會使其他時間難以觀察差異，所以就沒有將其放上來。

### Weak Scalability (hw3-3):
- 進行`weak scalability`實驗的目的是測試當問題規模隨著計算資源（GPU數量）增大時，程式的效能是否能保持穩定。因此，需要根據GPU數量調整問題的大小，並觀察執行時間的變化。而因為在hw3-3當中，只有使用到兩張GPU，因此當一張GPU處理的`n x n`大小的矩陣時，兩張GPU需要處理的矩陣大小就要是`2 x n x n`，所以我們要尋找兩個testcases的`n`剛好是$\sqrt{2}$倍的關係。
- 經過搜尋，我在multi-gpu上選擇的testcase為`hw3-3/testcases/c06.1`，該測資的`n`為`39857`，除以$\sqrt{2}$約等於`28183`，對應這個大小，在single-gpu上的testcase選擇就是`p28k1`，該測資的`n`為`28000`，剛好約等於$\sqrt{2}$倍。
- 時間測量上，我同樣使用`clock_gettime`放在演算法執行的前後，以測量不同GPU數量的情況下，計算時間的變化。以下是實驗結果數據：
    | Configuration | Problem Size (n) | Execution Time (s) |
    | ------------- | ---------------- | ------------------ |
    | Single GPU    | 28000            | 13.26              |
    | Multi GPU     | 39857            | 39.43              |
- 從表格可以看出，multi-gpu的時間遠超single-gpu，在理想狀況下，執行時間應該要維持不變或是少量增加，此時卻是多了將近3倍的執行時間。
- 這個結果可能是因為我在分配計算工作時，實際上只有phase3的各半是有在平行處理的，phase1和2其實都是類似single-gpu，需要將該計算的dependency blocks全部計算完，這導致了沒有充分利用multi-gpu的計算資源，使得執行時間不理想。
- 根據這個結果，我的multi-gpu在`Blocked Floyd-Warshall`的執行下是不具備`Weak Scalability`的，代表還有可以優化的空間存在。

### Time Distribution (hw3-2):
- 在該實驗中，我測量時間的方式同樣是使用`clock_gettime`，分別測量不同種類的時間花費，以下是各個時間花費的包含範圍：
    - `Computing`：演算法整體的計算時間。
    - `Communication`：因為在`single-gpu`上，不存在device間的溝通，所以在此處我將其定義為——`CPU`為了使用`GPU`所需要做出的額外動作，例如：`cudaMalloc`，`cudaFree`和kernel launch等。
    - `Memory Copy`：`cudaMemCpy`從`host`到`device`以及從`device`到`host`的總時間。
    - `I/O`：`input`和`output`兩個funciton所佔的時間。
- 以下是該實驗結果的繪圖，同樣測試在`p15k1`這個testcase上。
![image](https://i.imgur.com/uULM0Uj.png)
- 從圖中可以看出，程式大部份的時間其實還是集中在`I/O`身上，佔了`41.7%`，而`Computing`所佔的時間倒數第二，僅有`23.8%`，比`Memory Copy`的時間還要再少了`9%`，可以說明`I/O`和`Memory`的效能遠比`Computing`還要低，特別是在`GPU`這種計算能力強大的device上。但轉念一想，也有可能是data size並沒有巨大到讓`GPU`的計算時間成為domination，所以我又多使用`p30k1`執行了另一次的實驗。
![image](https://i.imgur.com/HqiXVvW.png)
- 從第二次實驗結果就可以明顯看出，`Computing`的時間占據了整個程式的一半，這也是因為data size增大時，該演算法需要進行更多次的迭代更新distance matrix，才讓`GPU`的計算能力充分發揮，從而讓`Memory`和`I/O`對於效能的影響下降。

### Memory Usage (hw3-3):
- 原本在撰寫hw3-3的時候，最後一筆測資`c07.1`一直會`runtime error`，起初以為是testcase的size太大，導致`GPU`的memory不夠用，但是在獲取相關資訊後，發現`GPU` memory是可以剛好夠用的。
    | GPU Memory   | Value (GB) |
    | ------------ | ---------- |
    | Free memory  | 0.27       |
    | Used memory  | 7.65       |
    | Total memory | 7.92       |
- 隨後我又仔細檢查了程式中的溝通和執行的邊界狀況，確保沒有超出`Memory`邊界的情況發生，但即便已經萬般確認程式正確的情況下，還是沒有辦法找到bug在哪裡。後續在與同學討論時，發現他在`host memory`只使用一個array進行儲存，也就是說，並沒有區分`host_dist_s`和`host_dist_t`。
![圖片](https://i.imgur.com/tHSLcVk.png)
- 抱著嘗試看看的想法將code改成只使用一個`host array`進行儲存，就可以順利通過最後一筆測資了。
- 對於這個狀況，我進行了`Memory Usage`的實驗，在前面已經確定`GPU memory`並不會使用超過，所以此處專門檢查`CPU memory`的用量。
- 在最後一個case當中，`n`經過`padding`完的大小會是`44992`，所以一個distance matrix會有總共約`2e-9`個element。而因為每一個element都是`int`型別（`4bytes`），那麼總共`memory`的用量就會是`2e9 * 4 bytes`，約等於`8e9 bytes`，換算成`GB`，就會是`7.54GB`左右，那麼如果allocate兩個`host array`的情況下，總共需要的`memory`就會是`15.08GB`。
- 隨後我使用`srun free -h`查看可以使用的`Memory`資訊，發現可以使用的`memory`僅有`15GB`，在allocate兩個`host array`的情況下，剛好會超過用量，所以才會一直出現`runtime error`
![圖片](https://i.imgur.com/S6rGbKK.png) 

## Experiment on AMD GPU
- 根據`Lab3`所教學的方式將`cuda`編譯成`amd`之後，首先我觀察了code內的變化，我將兩份code放在了文件比對的網站上，發現其實大部份的code都是一樣的，包含kernel內的計算和`shared memory`的使用，只有在使用`GPU function`時，前綴會從`cuda`變成`hip`，其它的使用方式也是一模一樣，如圖。
![image](https://i.imgur.com/aAWRzJ4.png)
- 隨後我使用`hw3-2-amd-judge`去對`amd`的程式進行測試，發現原本`p31k1`之後的case都可以順利通過，並且原本我的`p31k1`的時間是32s，此時僅需要11s，變快了將近3倍。
![image](https://i.imgur.com/pbXfxFk.pngg)
- 我好奇為什麼在code架構幾乎沒變的情況下，效能可以得到這麼大的提升，隨後我去搜尋了server上兩個GPU的對比，也就是`GTX 1080`比較`AMD MI210`的效能。
![image](https://i.imgur.com/e7nCgjn.png)
- 如圖，左邊的為`GTX 1080`，右邊的是`AMD MI210`，從圖中可以看出`MI210`的`memory bandwidth`比`GTX 1080`好上了許多，在本次作業中需要頻繁從`shared memory`取值並存值的情況下，效能就可以大幅得到提升。
- 隨後在測試`multi-gpu`使用`hw3-3-amd-judge`的時候，卻會有部份測資產生`Wrong Answer`，但從執行時間來觀察，同樣變快了不少，例如`c07.1`從66秒的時間變成了16秒。
![圖片](https://i.imgur.com/vgXq303.png)


## Experience & Conclusion

- 這次的作業是實作上花費最久時間的一次，並且因為是使用`cuda`進行實作，debug上又會比`cpu`來的更加困難，所以前後debug的時間可能就佔了不少的比例。而且本次作業的測資在`p21k1`之後都蠻久的，觀察效能瓶頸再去做優化的時間也花費了很久。
- 但是正因為一步一步的去完成並優化此次作業，使我對`cuda`的實作更加熟悉，並且對於如何優化演算法的觀念也更加了解。在實驗中看著優化後的code比原本實作的code快上了幾十倍時，會有一種成就感油然而生。
- `GPU`的使用在如今已經廣泛應用，也希望透過此次作業，我可以將這個實作的經驗和觀念帶到我的研究當中，協助我發想出更多可以嘗試的方向。