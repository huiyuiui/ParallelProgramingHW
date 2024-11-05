# <center>Parallel Programming HW1</center>

<center>113062518 陳家輝</center>

## Implementation

* 在程式一開始讀取完參數之後，首先需要對`n` (number of items)以及`p` (number of process)作出相應處理，以保證程式可以正常運行，我的實作內分為兩個case：
    1. `n < p`：
    當`n < p`時，因為無法讓每一個process都可以分到item，會導致有部份process會拿到空的data，在計算或是溝通時皆會發生錯誤。所以我使用另一個變數記錄實際上可以運作的process有多少個，命名為`p'`，也就是說`p'=n`，而後將那些rank大於`p'`的process給排除在外。具體的實作方式正如上課投影片所舉例，將真正會運作的process分到同一個group，而沒有運作的則被排除並關閉。這樣便可以使得`n < p`的情況可以正常運行。
    2. `n >= p`:
    當`n >= p`時，便是正常情況，所以直接計算每一個process會分到多少items，並根據分到的items進行運算以及溝通。
* 隨後需要處理當`n`無法整除`p`的情況。我的做法是將剩餘的data全部放在最後一個rank，那麼便會使得它的data size和前面的rank都不相同，所以當倒數第二個rank和最後一個rank做溝通時，就需要知道最後一個rank所持有的data size究竟為多少。那麼實作上，`final_chunk_size`去記錄最後一個rank的data size，以保證在溝通以及計算上不會發生邊界錯誤的問題。
* 對於Local sort，原本我使用的是C++ algorithm裡面的qsort，後來在boost C++ libraries搜尋sort演算法時，發現了專門加速floating point的sorting演算法，即為`boost::sort::spreadsort::float_sort`，便確定使用該sorting演算法作為我的local sort。
* 在Odd-Even Sort階段，主要的大框架可以分為幾個步驟
    1. 首先從Even phase開始做。
    2. Even phase時，偶數的rank向後傳遞data，並且從後方的rank接收data並做`merge`。需要注意若溝通的對象為最後一個rank則在function call時填入的data size要為`final_chunk_size`。
    3. Even phase時，奇數的rank向前傳遞data，並且從前方的rank接收data並做`merge`。
    4. 切換至Odd phase。
    5. Odd phase時，奇數的rank向後傳遞dta，並且從後方的rank接收data並做`merge`。需要注意若溝通的對象為最後一個rank則在function call時填入的data size要為`final_chunk_size`。
    6. Odd phase時，偶數的rank向前傳遞data，並且從前方的rank接收data並做`merge`。需要注意rank 0不參與此次溝通與計算。
    7. Termination Checking。若仍未sort完，則切換至Even phase繼續1~7。
* rank之間的溝通我使用的是`MPI_Sendrecv()`，這樣可以比較方便管理前後兩個rank互相交換data時的溝通。
* 在termination checking的實作上，我使用`rtn_even`和`rtn_odd`記錄兩個phase所執行的`merge`是否有發生data交換，也就是說，若沒有data交換，代表該array是sorted的，那麼若是所有process都沒有發生交換，則代表整個Odd-Even Sort就結束了。所以我使用`rtn = rtn_even + rtn_odd`代表該process是否已經sorted，並且使用`MPI_Allreduce()`去檢查所有process的`rtn`是否為0，若為0則結束。
* `merge`的實作有分為與後面合併和與前面合併。與後面合併時，該rank在merge完所持有的data應該是較小的一半，所以在兩個array比較時，從小到大依序比較，較小的便放入一個temp array，最後將temp array裡面的值替換該rank的data array；而與前面合併時，該rank在merge完所持有的data應該是較大的一半，所以需要從大到小依序比較，從而取得較大的一半。
* 對於`merge`的優化又分為以下幾個版本：
    1. Basic：正如上面所述，比較的值放入temp array，最後使用一個`for loop`將temp array裡面的值替換掉data array。
    2. Pointer Swap：因為最後將temp array的值替換掉data array的`for loop`在data量大時也會造成時間消耗，所以我使用`std::swap`將兩個array的pointer進行交換，從而達到更快的替換速度。
    3. Early Return：當前後兩個array已經是sorted的時候，`merge`也依舊會從頭檢查到尾，增加了不必要的時間消耗。所以在`merge`執行之前，我會先比較前面array（應該持有較小的一半）的最大值和後面array（應該持有較大的一半）的最小值，若最大值小於最小值，代表此次`merge`將不會有data做交換，便可以提前return。（該方法在後續優化中被移除）
    4. Pointer Operation：最後的優化是我觀察到`merge`的執行會是sequential access，所以將`array[idx]`這種陣列操作更改為`*array`，直接對指標進行操作便可以再剩下一點尋找idx的時間。
* 最後一個優化是我在執行nsys profile觀察timeline trace時，發現溝通上的時間消耗也佔了不少，那麼若是可以減少溝通次數，勢必可以加快程式執行。所以我將`merge`內的Early Return的想法搬到外面，在前後兩個process交換data之前，先交換最大值和最小值，若最大值小於最小值，則代表兩者已經sort好了，就不需要再交換data，從而可以省下不少的溝通時間。

## Experiment

### Methodology

* **System Spec:**
    本次作業的程式皆執行在課程所提供的apollo server上。
* **Performance Metrics:**
    在實驗中，我使用`Nsight Systems`進行程式執行時間的測量。具體測量方式如下：
    1. 首先使用lab2中所提供的`wrapper.sh`產生程式各個rank的report。
    2. 以`rank 0`作為測量基準，使用指令`nsys stats -r mpi_event_trace --format csv`獲取各個`MPI Events`的timeline，並且以csv格式輸出。
    3. 將csv格式輸出放入`python`內，使用`pandas`進行資料處理，將時間單位`ns`轉換成`s`。
    4. `Comm_Time`：遍歷`MPI Events`，紀錄`MPI_Send()`、`MPI_Recv()`、`MPI_Sendrecv()`、`MPI_Allreduce()`以及`MPI_Comm_split()`的`Duration (s)`，進行加總。
    5. `IO_Time`：遍歷`MPI Events`，紀錄`MPI_File_open()`、`MPI_File_read_at()`、`MPI_File_close()`以及`MPI_File_write_at()`的`Duration (s)`，進行加總。
    6. `CPU_Time`：將`MPI Events`中的`MPI_Finalize()`的`End (s)`扣掉`MPI_Init()`的`Start (s)`作為`Total_Time`，隨後把`Total_Time`減掉`Comm_Time`和`IO_Time`作為程式的`CPU_Time`。
    
    每次實驗均會在相同實驗數據上執行三次並將所得到的`CPU_Time`、`Comm_Time`和`IO_Time`個別取平均作為該實驗數據的結果，以降低cluster使用量較大時造成的效能衰減的影響。隨後再將得到的結果使用`matplotlib.pyplot`進行實驗結果的繪圖。

### Plots: Speedup Factor & Profile
* **Experimental Method:**
    * Test Case Description：所有實驗均使用testcases中的`40.txt`作為測量基準，該testcase具有`536869888`個element，在`judge`的測試時間約為**7s**，作為測量基準應可以很好的看出各項指標。
    * Parallel Configuration：不同實驗所使用的nodes和cores基本上以**4 cores per node**做為基本實驗參數，但根據實驗目的不同會做出些許調整，具體實驗參數會在接下來的報告中再做闡述。
* **Total Time Profile:**
    首先進行的實驗是觀察在同一個case下增加process數量時，整體執行時間的變化。該實驗參數在使用1、4和16個processes時均使用**4 cores per node**，而因為使用64個processes要allocate 16個nodes，會導致計算資源不足，所以我改成使用**8 cores per node**，也就是在使用64個processes進行實驗時，只使用了8個nodes。
    執行該項實驗可以幫助我了解當執行資源變多時，程式的效能是否符合預期變好，並且執行上的bottleneck會在哪裡，具體的結果如下圖所示：

    ![image](https://i.imgur.com/mUT4wXS.png)

    由圖中可以明顯看出，當processes數量變多時，整體的執行時間是逐漸下降的，代表程式確實有得到提速。具體分析各個執行時間可以看到，當processes變多時，`CPU_Time`下降較為明顯，而相對的`Comm_Time`也會隨之上升，如spec中所繪製的圖類似，只是在64個processes時並沒有像spec圖中`Comm_Time`佔了絕大多數的時間。而且還有一點較為奇怪，就是`IO_Time`在不同processes數量時的執行時間幾乎維持一樣，不符合spec中當processes數量增多時，`IO_Time`也會隨之下降。
     為了得到程式本身效能的分析，也就是僅包含`CPU_Time`和`Comm_Time`，我先將`IO_Time`移除，以觀察當processes數量變多時程式效能的提速是否符合預期，呈現結果如下圖：

     ![image](https://i.imgur.com/VXUnJRr.png)

     將`IO_Time`移除後，對於程式效能的提升便可以比較清晰的看出，當processes變多時，`CPU_Time`會下降，而`Comm_Time`則會上升，是符合預期的。
* **Same # of processes but different # of nodes (Time Profile)：**
    該實驗主要想要探討當processes數量一樣，但是一個是都在同一個node上，而另一個則是分佈在不同的nodes上，兩者的在程式效能上是否會有不一樣的表現。而會想要執行該項實驗是因為我在瀏覽testcases時，發現有一些測資會有同樣的`n`和`p`但`N`(Node數)卻不同。
    實驗參數中的`N`代表Nodes數，`n`代表processes數。因為在前面的實驗中發現`IO_Time`幾乎不隨processes數量改變而改變，所以在該實驗中我將`IO_Time`移除，以更好的觀察效能變化，實驗結果的呈現圖如下：

    ![image](https://i.imgur.com/oKgl1Wz.png)

    從圖中可以看出，基本上processes位於同一個node或不同node的執行時間相近，並沒有太大的差別，但是若是細看仍然可以看出兩者在`Comm_Time`上仍有一絲差別，其中processes分佈在不同nodes上的`Comm_Time`會比在同一個nodes上的稍微久一點。
    套用Lab1中介紹`slurm`時所用到的圖，便可以看到在同一個Node時，core之間的溝通較近，而不同Node之間的溝通則會較遠，就可以很好的理解為什麼processes分佈在不同nodes時，所需要的溝通時間會比較久一點。
    
    ![image](https://i.imgur.com/Y1raHB5.png)
    
* **Speedup Factor:**
    在Speedup Factor的測量上，實驗參數同樣使用**Total Time Profile**中的參數，實驗結果的呈現如下圖：

    ![image](https://i.imgur.com/YHTfqtW.png)

    由圖中可以看出，speedup factor並沒有如預期一般，增加多少的processes數量，就可以讓程式的執行速度下降多少倍，也就是最理想的狀況就是像spec中的圖二。而在實驗結果上，特別在64個processes時，speedup factor才接近2.6而已，離理想的64倍差的很遠。
    根據先前實驗的結果，我猜測是因為`IO_Time`幾乎不隨processes數量的改變而改變，導致即使processes數量變多，但整體程式的執行時間仍無法得到有效的下降。所以我重新執行一次實驗，將`IO_Time`移除再計算speedup factor，便得到以下的結果圖：
    
    ![image](https://i.imgur.com/Mpw92Eu.png)

    從圖中可以看出，將`IO_Time`移除後的speedup factor有著比較明顯的變化，雖然仍無法達到理想的速度，但還是有著不錯的提升。

* **I/O Time Profile:**
    如果按照spec中的**Time Profile**的圖，理應上processes數量變多時，`IO_Time`應該也會隨之下降，但是前面的實驗結果發現`IO_Time`幾乎與processes數量毫無關聯，基於此發現，我對I/O多執行了幾個實驗。
    第一個實驗是將I/O拆開分別做測量，也就是分別測量`Reat_Time`和`Write_Time`，以觀察`IO_Time`幾乎沒有改變的原因究竟是兩者都沒有變化還是只是某一方沒有變化，結果呈現如下：

    ![image](https://i.imgur.com/xs55nyk.png)

    從圖中可以看出，不同processes時確實整體的`IO_Time`都落在11～12秒左右，但是若將Read和Write拆開來看的時候可以發現，`Read_Time`是有隨著processes變多而時間下降，相對的`Write_Time`便沒有什麼太大的變化。
    根據課上講義對`MPI_IO`的介紹，多個process對同一個file是可以concurrent read/write的，只是data consistency需要由user自己負責maintain。然而從實驗出來的結果圖卻發現，只有`Read_Time`下降，而`Write_Time`沒有改變，與講義內容是有所出入的。
    在查閱相關資料後我仍然沒有得到對於這個實驗結果的合理解釋，所以我在課後詢問老師，這才得到答案——雖然`MPI_IO`可以支援Multiple-write，但是也是取決於最底層的**File System**，在本次作業所使用的環境內，可能因為File system內仍有`Locking`的機制，所以使得多個processes要同時write的時候仍然需要取得`Lock`才可以進行寫入，也就是說，write的時間幾乎不隨processes變多而減少正是因為同步化機制使得寫入仍然是Single-write。
    另外，老師也有提到在I/O的時候，所花費的時間也與Network Bandwidth有關，所以我又多執行了幾次實驗，觀察每次執行的時間會不會有所不同。而執行的結果顯示，`CPU_Time`和`Comm_Time`每次執行的誤差基本上不會超過0.3秒，`Read_Time`的誤差則不超過0.5秒，但是`Write_Time`的誤差則最多可以來到1.3秒，實驗結果如下圖：

    ![image](https://i.imgur.com/nVXMuqa.png)

    由圖中可以看出，對於同一個測資以及使用相同實驗參數執行五次所得到的結果，`IO_Time`最長的接近12秒，而最短的則是10秒出頭，代表I/O所需要的時間確實與當下server的狀態有所關係。

- **Strong Scalability:**
    實驗中使用的testcase固定，所以整體上的工作量是固定的。而根據上面的實驗我們可以看出，當processes數量變多的時候，程式執行的速度是有隨之提升的。因此，根據以上的實驗結果我們可以得出，該平行程式是具備**Strong Scalability**。對於該實驗結果的呈現如下圖：

    ![image](https://i.imgur.com/dBXphA5.png)

    從實驗結果圖可以看出，隨著processes變多，整體的執行時間會隨之下降。然而，但processes增加到4以上之後，執行時間下降的幅度幾乎趨緩，即使processes數從4變成8，數量增加了一倍，但是時間也只有差不多下降3～4秒而已。
    考量到`IO_Time`在不同的實驗參數下都會維持在10～12秒的時間，佔了整個程式執行大部分的時間，所以我同樣重新執行實驗，並且將`IO_Time`移除，以觀察在程式執行上的效能的scalability。實驗結果圖如下:

    ![image](https://i.imgur.com/tXLjPTv.png)

    從實驗結果圖可以看出，把`IO_Time`移除後，processes數量增多時，執行時間的下降仍有不錯的幅度。顯示出如果僅考慮程式執行的效能的話，該程式是有不錯的scalability的。

### Discussion

- 根據前面的所有實驗我們不難看出，`IO_Time`一直都是整個程式執行的**bottleneck**，特別是`Write_Time`幾乎不隨著processes數量增多而速度變快，讓不同processes數量下的實驗時間都有一個接近10秒的起始時間。關於`MPI_IO`的優化方式，我有搜尋了相關資料，並且有嘗試將`MPI_File_write_at()`改成例如`MPI_File_Set_View()`、`MPI_File_write`、`MPI_File_write_all()`等等方式，以求可以更快的寫入，然而所得到的結果基本上沒什麼變化，也因此才有去詢問老師關於寫入時間幾乎不變的問題，最終得到的答案就是還是會depends on `File System`。
- 從最後一個實驗可以看出，若是不包含`IO_Time`的話，程式執行效能的scalability是不錯的，隨著processes數量變多會有明顯的速度提升。但如果將`IO_Time`包含進來的話，因為有著10～12秒的起始時間，即使`CPU_Time`和`Comm_Time`加總的時間已經有大幅下降，但觀察整體的程式執行時間還是看不太出明顯的變化。所以也再次驗證了`IO_Time`會是整個程式的bottleneck。
- 若是後續可以針對I/O做出更多的優化，並且讓`Write_Time`可以具備strong scalability的話，相信程式執行的效能可以得到大幅的提升。

## Conclusion

- 這次作業算是我第一次接觸到並且親手實作的平行程式，與以往撰寫sequential code的邏輯上有著很多不同，像是在評估程式效能時，我會時常忘記processes間是平行在運作，而不是一個做完等一個，在撰寫code的時候就經常會忽略一些細節而導致程式錯誤。並且在debug時，平行程式的輸出並不一定會按照順序，使得想要透過打印結果去debug的難度直線上升，讓我在修改程式時花費了不少功夫。但所幸經過與同學討論正確性及優化空間後，我也逐漸將程式一步一步從錯誤修改到正確，從效能欠佳到平行程度不錯，幫助我獲得了很多知識。
- 經過本次作業的實作，我對撰寫平行程式的觀念又更加熟悉了幾分，並且對於不同計算單位間的溝通及合作也多了幾分了解。相信此次作業所獲得的經驗，可以讓我在後續的作業及project中可以更快上手，也希望可以將所學運用到研究當中。

