# <center>Parallel Programming Hw2</center>

<center>113062518 陳家輝</center>

## Implementation

### Pthread
- 首先根據所提供的sequential code，將`mandelbrot set`的計算包成一個function，隨後使用`pthread_create()`創建出多個threads去做`mandelbrot set`的計算，而因為我將`image`放在global，所以每個thread都可以直接存取`image`，並且計算範圍切分好是不會有衝突的。最後使用`pthread_join()`等待所有的thread都計算完成，再一次呼叫`write_png()`去做寫入。
- 因為有多個thread可以同時進行計算，所以需要為每個thread分配工作，分配的方式又可以分為**對height做切分**以及**對width做切分**，以下是我嘗試過不同分配方式的實作：
    1. 對width做切分：將width除以threads的數量，分別得到各個thread的`start`和`end`做為計算範圍，便可以讓各個thread平行去做運算。
    2. 對height做切分：具體切分方式與width相似，不過是以height去除以threads數量，得到`start`和`end`。呈現效果如下圖：

        ![圖片](https://i.imgur.com/7oB0aDC.png)

    3. 對height做切分但thread的計算範圍不連續：先前所提到的計算範圍是讓threads負責一個連續的範圍，但經過我觀察testcases的圖片發現，很多時候需要大量計算的pixel，都會在一個特定的區域內，意思就是說，剛好負責到那一塊區域的thread就會嚴重的拖慢整體的時間，所以我將height loop的條件作出更改:
    原本：`for(int j = start; j < end; j++)`
    改成：`for(int j = thread_id; j < height; j += ncpus)`
    這樣就可以使得thread負責的範圍不是連續的，在遇到需要大量計算的區域時，也可以由不同的threads進行分擔，從而使整體的執行效能更好，這也是我最終選擇分配工作的方式。呈現效果如下圖：

        ![圖片](https://i.imgur.com/nvvhjHj.png)

- `mandelbrot set`的計算在每一個loop內的都是完全相同的，所以可以套用在`Lab2`所教學的`vectorization`的技巧，再去對整個程式進行優化。而因為目前的server有支援較新的`avx512`，所以我便以`avx512`進行實作，實作方式如下：
    - 因為一個thread一次計算一個row，所以`vectorization`是對width做切分。
    - 首先計算`vectorization`可以一次計算幾個pixel，因為`avx512`有512bits，一個doulbe是64bits，所以`avx512`可以同時塞下8個double。
    - 那麼在width的loop內，一次計算8個pixels時，loop的index就需要一次加8，即為`i+=8`。
    - 在計算前，需要先將一些variable變成`__m512d`的形式，讓後續計算可以一次計算8個pixels。特別要提到的是`x0`，因為需要計算的pixel位置不一樣，我手動讓`x0`初始化為`i+0`~`i+7`，這樣才可以符合一次計算8個pixels而且不會計算錯誤。
    - 進入到`manderbrot set`的`iteration loop`裡面，這裡需要做的事情只是將`+`、`-`和`*`的計算改成使用`_mm512_add_pd()`、`_mm512_sub_pd()`以及`_mm512_mul_pd()`，函式內所使用的變數便是先前已經進行初始化的`__m512d`的向量化參數。
    - 重要的是該如何檢查`iteration loop`什麼時候會停止。我在這裡使用了`_mm512_cmplt_pd_mask(a, b)`這個函式，它可以比較a和b的大小，若是`a<=b`，則`mask=1`，反之則為`mask=0`。隨後將所計算的`length_squared`和`4`去做比較，便可以得到`vector`中8個index哪些已經不符合`mandelbrot set`，在後續的計算中，`repeats`就不會再被更新。
    - 如果`mask_vector`裡所有的mask都等於0，代表全部的值都已經超過4，則跳出loop；或是`iteration loop`執行的次數超過所給定的`iter`，則loop同樣結束。
    - 最後則是將所計算的8個`repeats`依次放入到`image`對應的位置當中，便完成了對`mandelbrot set`的`vectorization`。
    - 需要注意若是width不是整除8的話，在`vectorization`結束後，仍需要一個`sequential loop`將剩餘的pixels依次進行計算，才可以保證程式正確。


### Hybrid

- Hybrid的實作大致上與Pthread相差不多，因為已經完成了`vectorization`的實作，代表大部分需要coding的地方已經完成，所需要做出的更改就是將threads之間的平行改成使用`MPI`和`OpenMP`進行。大致上程式執行的流程如下：
    - 先做`MPI`的initialization，因為分配工作是對height及width做切分，而不管是哪一種方式，其計算的總量(n)都遠大於processes的數量，所以便不需要如hw1一般再去細分若是`n < p`的特殊情況。
    - 計算每個processes分配到的工作範圍，隨後依據這個範圍平行去做`mandelbrot set`的計算，而因為每個process裡都有thread可以多加利用，所以在每個process分配到的工作範圍內，還可以使用`OpenMP`進一步對工作進行切分。
    - 所有processes完成計算後，使用`MPI`將各自負責的結果匯整到`rank 0`，再由`rank 0`呼叫`write_png()`統一將結果寫入。
- 在分配工作的方式上，因為Hybrid的版本可以同時使用`MPI`和`OpenMP`，所以可以實作的方式有很多種，具體嘗試過的方式如下：
    1. 用`MPI`切分height，`OpenMP`切分width：該實作方式類似`Pthread`的工作範圍連續版本，僅僅是將`Pthread`所使用到的變數改成`MPI`的版本，然後在witdh的loop前加上`OpenMP`的`pragma`便可以正常運行。比較需要注意的是因為`Pthread`版本內的`image`放在global，threads之間可以存取，但processes之間是沒有共享的，所以`image`仍然維持local，並且在最後計算完才進行溝通匯整，其中溝通匯整的方式又分為以下兩種：
        - 將每一個process的`image`設定為和原本相同大小的data array，即為`height * width`，隨後便可以使用如`Pthread`版本的index的方式各自寫入負責的部份，最後溝通時使用`MPI_Reduce(MPI_SUM)`匯整到`rank 0`。因為各自負責的部份不會重疊，代表沒有計算到的地方會是維持0，所以做reduce的時候不用擔心重複疊加導致結果錯誤。
        - 上面所實作的方式在溝通時會多出很多不必要的資料傳輸，所以第二個版本則是將每一個process的`image`設定為`chunk_size * width`，僅保留自己計算的部份，並且在最後溝通時使用`MPI_Gatherv()`，通過設置好每一個process的`offset`和`recv_size`，便可以將各部份結果連續的串接起來變成一個完成的`image`。這樣的做法可以省去不少溝通的成本。
    2. 用`MPI`切分height且不連續分配，`OpenMP`切分width且套用不同的`schedule`：切分height的方法如同`Pthread`中不連續分配版本中的實作方式，而因為先前觀察到height可能會有load unbalancing的問題，所以在此處對width做切分時，我有嘗試過套用`static`、`dynamic`和`guided`等不同的`schedule`的方式，也有嘗試過調整`chunk size`去做更好的load balancing，具體的實驗結果在後面會呈現。分配方式的呈現效果如下圖：

        ![圖片](https://i.imgur.com/6kek2m4.png)

    3. 用`MPI`切分height且不連續分配，同時`OpenMP`也對切分後的height再做切分：考慮到一開始在做`Pthread`時，將分配方式從width換成height後得到了很多的效能提升，所以在想會不會對height切分的方式更平均的話，會使整體的效能再變好，我便將`OpenMP`的`pragma parallel for`從width loop移動到height loop，也就是說，每一個process所分到的不連續的計算範圍內，再使用process內的thread再去做切分，而每一個thread就變成計算一整個row，得到的結果也確實比先前的方式都來的好，因此作為我最終的分配方式。分配方式的呈現效果如下圖：

        ![圖片](https://i.imgur.com/kWJEubX.png)
    
## Experiment & Analysis

### Methodology

* **System Spec:**
    本次作業的程式皆執行在課程所提供的qct server上。
* **Performance Metrics:**
    在實驗中，我使用`Nsight Systems`進行程式執行時間的測量。不同版本的`Computing Time`的測量方式如下：
    * Pthread：在`OS Runtime Summary`處可以看到`System Call`的時間，其中包含了`pthread_create()`和`pthread_join()`。而因為呼叫`pthread_join()`前，所有的threads都必須完成自己的計算工作才會呼叫，所以當`pthread_join()`全部執行完畢時，代表整個程式的`Computing Time`也已經結束。從`osrt_sum`可以看出，`pthread_join()`所花費的時間佔了整個程式的 **98%** 以上，所以我取出`pthread_join()`的`Total Time`作為程式的`Computing Time`。
    * Hybrid：為了準確測量出`menderbrot set`的`Computing Time`，我在計算前後分別加上了`MPI_Barrier()`，隨後使用`MPI Event Trace`取出前後兩個`MPI_Barrier()`，的`End Time`和`Start Time`，再將兩者相減，便可以得到程式的`Computing Time`。

    每次實驗均會在相同實驗數據上執行三次並將所得到的`Computing Time`取平均作為該實驗數據的結果，以降低cluster使用量較大時造成的效能衰減的影響。而因為在`Hybrid`當中有不同的processes，我取processes中`Computing Time`最久作為該次實驗的結果。隨後再將得到的結果使用`matplotlib.pyplot`進行實驗結果的繪圖。


### Plots: Scalability & Load Balancing & Profile

* **Experimental Method:**
    * Test Case Description：所有實驗均使用testcases中的`strict34.txt`作為測量基準。該testcase需要執行**10000**個iteration，在兩個版本的`judge`中都需要花費接近**5s**的時間，作為測量基準應可以很好的看出各項指標
    * Parallel Configuration：
        * Pthread：根據實驗內容決定每一次執行的threads數量，在實驗結果圖中可以得知。
        * Hybrid：不同實驗所使用的nodes和cores基本上以4 cores per node做為基本實驗參數，但根據實驗目的不同會做出些許調整，具體實驗參數會在接下來的報告中再做闡述。

* **Strong Scalability:**
    實驗中使用的testcase固定，所以需要計算的pixels數是固定的。若是計算資源變多時，程式需要的計算時間是有所下降的話，代表該平行程式是具備**Strong Scalability**。以下是兩個版本在`scalability`的實驗結果圖：

    ![圖片](https://i.imgur.com/Cu2QI5N.png)

    從上圖可以看出，`Pthread`版本的`Computing Time`是有隨著threads數變多而有所下降，並且一直保持著良好的下降幅度，在threads數量為9的時候就可以讓計算時間從將近90秒變成10秒以下，代表該版本的程式是具有`Strong scalability`。

    ![圖片](https://i.imgur.com/3lrZEkk.png)

    `Hybrid`版本的實驗中，我不僅測試了threads變多時的優化時間，也測試了當processes變多時，是否也會讓程式的`scalability`上升。上圖中每一條線分別代表了不同processes的數量，x軸的數字為每一個process可以使用的threads數量。從圖中可以看出，不論processes的數量為何，當threads變多時，計算時間是有明顯的下降，並且processes比較多者，計算時間的線圖也確實在較少者的下面，代表processes變多也可以讓程式的計算時間下降，說明了`Hybrid`版本的`Strong Scalability`。

* **Speedup Factor:**
    在以上的實驗中我們得知了程式的`scalability`是不錯的，接下來想要探討當計算資源變多時，程式計算的效能成長是否有達到一個良好的倍數，而最理想的情況是**計算資源與效能成正比**。
    計算`Speedup Factor`的方式如下：將只有一個計算核心(`thread=1`)所得到的`Computing Time`做為**基準**，再分別測量不同計算資源情況下所得到的**優化後計算時間**，並將**基準時間**除以所得到的**優化後計算時間**，得到`Speedup Factor`。
    以下是兩個實作版本在`Speedup`實驗中的結果圖：

    ![圖片](https://i.imgur.com/HzojXs3.png)

    從`Pthread`的實驗結果圖可以看出，程式的`Speedup Factor`幾乎與ideal的情況一致，線圖基本上呈現一條筆直的斜直線，代表計算資源變多時，程式計算的效能是可以隨著正比成長的，顯示程式具有不錯的平行程度。

    ![圖片](https://i.imgur.com/BohNbEQ.png)

    `Hybrid`版本我同樣執行了不同processes數量的實驗，觀察在不同processes數量時的`speedup`是否可以如單個process時的一樣接近理想狀況。上圖中的每一條線同樣代表了不同processes的數量，而從圖中可以看出，不同processes的線圖同樣是很漂亮的斜直線，仔細觀察每一個點的值也確實發現，不論多少個processes，當threads變多時，程式計算的效能是隨之正比成長的，代表即使是在多個processes運行下，程式的平行程度仍然維持著不錯的效果。

* **Load Balancing:**
    在本次作業中，因為計算的code已經幫我們寫好，並且spec規定不能對`menderbrot set`的計算過程有所更改，所以如何增加平行程度讓程式的效能可以更好的重點就是在於**Load Balancing**。因為每一個計算核心所做的事情都一模一樣，並且溝通和I/O的成本都相對較低，那麼如何讓各個計算核心的工作量盡量相同，以求大家同時結束，減少waiting的overhead，便是實驗的目的所在。
    在該實驗中，因為要觀察各個threads的工作量，便需要知道各個threads在自己分配的工作內花費了多少的計算時間。而又因為`Nsight System`僅能透過`time line`觀察各個threads的執行狀況，所以要抓取數據便需要手動將滑鼠放在`time line`上以獲得`CPU Utilization`的訊息，其中包含的`Ends`便可以作為測量的數據。具體取得`load balancing`數據的方式如下：
    * 觀察各個threads執行時間最久以及最短的thread分別為何。
    * 將滑鼠放在該兩個thread的最後一個計算區間上獲得`CPU Utilization`中的`Ends`。
    * 將兩者的`Ends`做相減作為`loading overhead`，以觀察`load balancing`的數據。
    
    根據以上方式，`Pthread`測量出來的實驗結果如下圖：

    ![image](https://i.imgur.com/RCI52AJ.png)

    從圖中可以看出，當threads數量較少時，`loading overhead`的時間會比較長，而隨著threads變多，該overhead會逐漸下降，直到`threads = 8`時，overhead幾乎接近於0。但在`threads = 7`時，overhead比`threads = 6`時上升了一點，考慮到cluster的狀況以及`loading overhead`本身就很小的情況下，該結果仍然是處於可預期的範圍內的。
    `Hybrid`版本在該實驗中，我想探討的是各個processes所分配到的工作量是否均勻，具體實驗過程如下：
    * 測量各個processes的`Computing Time`的方式如`Performance Metrics`中提到的一樣。
    * 將各個processes中最久的`Computing Time`減去最快的，其差值做為該實驗的`loading overhead`，以觀察`load balancing`的數據。

    根據以上方式，`Hybrid`測量出來的實驗結果如下圖：

    ![image](https://i.imgur.com/hwJQohE.png)

    從圖中可以看出，不同processes在threads數量變多時，`loading overhead`都是逐漸下降趨近於0的，只有`processes = 2`時在`threads = 2 & 6`時，overhead稍許上升，但差值都在**0.01**以內，如前面所提到的可能情況，仍在可以預期的範圍內。
    並且當processes數量較多時，呈現出來的線圖收斂也比較漂亮，`processes = 4`的`loading overhead`在`threads = 3`時就已經近乎於零，顯示出較多的processes可以讓loading分配的更均衡。
    從兩個版本的實驗中可以看出，經過調整後的分配工作方式，也就是**對height進行不連續的切分**，所得到的`load balancing`效果很好，基本上在較多計算核心的情況下，每一個計算核心分配到的工作量都很平均。
    
    
### Discussion

* 根據以上的實驗，程式整體的`scalability`是很不錯的，不論是`Pthread`還是`Hybrid`版本都幾乎達到了理想的狀況。我認為這是因為本次作業的演算法本身便具有良好的平行程度，不像是`Hw1`一樣需要各個processes進行溝通，以及`IO_Time`也遠比較小，這才讓程式中絕大部分的時間都是在計算，從而使計算資源可以更加有效的去被利用，所以當計算資源變多時，更能有效的反映在程式執行時間的減少上。
* 在後續進行的`load balancing`的實驗中，經過優化過後的分配方式，從實驗結果中可以很好的看出各個計算資源之間的工作量分配都很平均，使得較多計算資源的`loading overhead`基本上都能維持在**0.01**秒內，避免工作量較大的計算核心拖慢整個程式的執行時間。而較多計算資源的`load balancing`比較好的原因也正如在`Implementation`中提到的一般，若是能將工作切分的越細，就越能減少特定計算核心都分配到工作量較大的情況出現，從而使程式整體的效能可以得到提升。

## Experience & Conclusion

* 本次作業因為主要的演算法已經實作完成，所以在上手時比起`Hw1`來說迅速了不少，也可能是因為已經有了`Hw1`的經驗，因此更知道該從什麼地方去開始實作，當完成兩個版本基礎的AC code的時候，花費的時間可能才不到實作`Hw1`的一半。但是後續在實作`Vectorization`的時候，卻遭遇了一個debug了很久也無法找到的錯誤，那就是因為編譯指令所導致的精度下降。起初我以為是指令運用錯誤，在邏輯debug上就花了好一段時間，後來討論區有同學詢問相關問題，將編譯指令從`-O3`改成了`-O1`才通過了大部份測資，然而最後幾個測資卻是`_mm512_fmadd_pd()`導致的精度下降，我是查閱很多的資料才發現有人提到該指令在某些編譯指令的情況下會有不同的編譯表現，我對此心生疑惑，抱著嘗試的心理將其拆開成三個指令，這才通過了全部測資。雖然`Vectorization`的debug真的花費了我很多時間，而且還不是邏輯或實作上的錯誤讓我很懊惱，但其帶來優化的效果卻是很顯著，也算是可以接受的。
* 另外因為演算法不能更改，所以其實本次作業更多的優化空間是在`load balancing`，將工作切分的越平均，所帶來的優化效果可以越顯著，也讓我了解到平行程式不僅僅是將`sequential`改成`parallel`這麼簡單，還需要考慮到每一個計算核心的工作量是否合理，才不會讓某個計算核心拖延了整個程式的計算效能。
* 經過本次作業，我對於平行程式的觀念以及實作又更加熟悉了幾分，也希望在後續的作業和Final Project中可以順利的完成，並學到更多的知識。