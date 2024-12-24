---
title: Parallel Programming Hw5 Report

---

# Parallel Programming HW 5 Report

> - Please include both brief and detailed answers.
> - The report should be based on the UCX code.
> - Describe the code using the 'permalink' from [GitHub repository](https://github.com/NTHU-LSALAB/UCX-lsalab).

## 1. Overview

> In conjunction with the UCP architecture mentioned in the lecture, please read [ucp_hello_world.c](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/pp2024/examples/ucp_hello_world.c)

1. Identify how UCP Objects (`ucp_context`, `ucp_worker`, `ucp_ep`) interact through the API, including at least the following functions:
   - `ucp_init`
   - `ucp_worker_create`
   - `ucp_ep_create`
- Ans:
    - UCX在執行時會先對`command line`進行parse，再將讀取到的指令製作成程式所需要的`config`，而這個`config`就會被放進`ucp_init`當中。
    - `ucp_init`會直接呼叫`ucp_context.c`的`ucp_init_version`，在該function內，主要做的事情就是根據`config`配置所需的資源以及**初始化`ucp_context`**。
    - 初始化完的`ucp_context`會作為參數放入`ucp_worker_create`當中，在該funciton內主要做的事情就是使用`ucp_context`內所包含的UCX環境的資源和全局配置去**初始化`ucp_worker`**，並且初始化傳輸所需的資源，例如`ucp_worker_add_resource_ifaces`就是打開對應的傳輸層接口（UCT）。
    - 創建完`ucp_worker`之後，接下來就是對需要通訊的雙方進行`OOB connection`的建立。首先雙方會交換各自的UCX address，隨後根據對方的address分別呼叫`run_ucx_client`和`run_ucx_server`。
    - 以`run_ucx_client`為例，在該function內，使用先前創建好的`ucp_worker`以及server的address呼叫`ucp_ep_create`，**創建與server的`ucp_ep`**。隨後再進行傳輸測試，確認通信建立成功。

2. UCX abstracts communication into three layers as below. Please provide a diagram illustrating the architectural design of UCX.
   - `ucp_context`
   - `ucp_worker`
   - `ucp_ep`

    ![image](https://i.imgur.com/0Uc5kCj.png)
- 借鑑課程的UCX講義中的第42頁。從圖中我們可以清晰的看出UCX的架構設計分為三個layer，分別為`ucp_context`，`ucp_worker`以及`ucp_ep`。
- `ucp_context`：一個process只會創建一個實例，儲存了該process的全局配置資訊，負責資源管理與配置。
- `ucp_worker`：由`ucp_context`所創造，worker主要負責管理不同的傳輸層資源和處理傳輸相關操作，可以有多個worker進行不同的通訊任務。
- `ucp_ep`：由`ucp_worker`所創造，endpoint的主要工作就是負責與遠端通訊對象進行連接，而每個worker可以創建多個endpoint以便與多個通訊對象建立連接。

> Please provide detailed example information in the diagram corresponding to the execution of the command `srun -N 2 ./send_recv.out` or `mpiucx --host HostA:1,HostB:1 ./send_recv.out`

![hw5](https://i.imgur.com/f6pmedV.png)

- 當`Application layer`呼叫`MPI_Init`的時候，`OpenMPI layer`會呼叫`ucp_init`，隨後兩個process會分別根據目前的`global config`創建各自的`ucp_context`。
- 分別創建完`ucp_context`後，各自的`ucp_context`會再創建各自的`ucp_worker`。
- 在創建`ucp_ep`前，兩個process需要透過`OOB Connection`得知對方的`address`，隨後`ucp_worker`就可以根據這個`address`創建各自的`ucp_ep`，並且兩個`ucp_ep`是對應關係，僅用於這兩個processes間進行溝通。
- 而在創建完`ucp_ep`後，兩個processes就可以開始進行溝通，會由`UCT layer`傳輸或接受message。

3. Based on the description in HW5, where do you think the following information is loaded/created?
   - `UCX_TLS`
   - TLS selected by UCX
- `UCX_TLS`：根據上課講義以及對於Hw5的描述，我認為`UCX_TLS`是屬於global configuration，那這種全局資訊會在`ucp_init`的時候被放入`ucp_context`當中進行初始化。而經過trace code之後，實作與我的理解基本符合，在`ucp_init`=>`ucp_init_version`的時候會呼叫`ucp_config_read`，這會將包含`UCX_TLS`的配置資訊製作成一個`ucp_config`，其中`tls`即為所使用的傳輸層的名稱，可以包含多種傳輸層。隨後`ucp_config`就會使用在`ucp_context`的初始化當中。
- TLS selected by UCX：起初對於這個資訊並沒有那麼理解，但有想到老師上課有說過可以根據不同的傳輸應用選擇適合的傳輸形式，基於我對UCX的知識，我認為會是在`ucp_ep`當中獲得這個資訊。經過trace code之後，我發現該資訊確實是在`ucp_ep_create`時被加載，為了根據這個連接選擇適合的傳輸，會使用`ucp_worker_get_ep_config`獲取傳輸所需要的config，這其中就會包含UCX所選擇的TLS，隨後再使用這個`ep_config`進行endpoint的創建和connection的建立。

## 2. Implementation

> Please complete the implementation according to the [spec](https://docs.google.com/document/d/1fmm0TFpLxbDP7neNcbLDn8nhZpqUBi9NGRzWjgxZaPE/edit?usp=sharing)

> Describe how you implemented the two special features of HW5.

1. Which files did you modify, and where did you choose to print Line 1 and Line 2?

- 在此次實作當中，我總共修改了三個files，分別是`parser.c`，`type.h`以及`ucp_worker.c`。首先是觀察在`parser.c`內有此次作業標注的`TODO`，可以發現有一小段code被註解掉：`UCS_CONFIG_PRINT_TLS`。
- 這個變數與code前面的`UCS_CONFIG_PRINT_CONFIG`等很相像，所以我先去看了這是什麼東西，發現是`type.h`裡所定義的一個`enum`，用於根據不同flags打印所需要的資訊。那麼我就現在此處將`UCS_CONFIG_PRINT_TLS`加進去，並設定為`UCS_BIT(5)`。
![image](https://i.imgur.com/jfmy05S.png)
- 隨後回到`parser.c`，`ucs_config_parser_print_opts`這個function是由`ucp_config_print`所呼叫，那麼我們僅需找到對應位置呼叫`ucp_config_print`即可。
- 首先是尋找print Line 1: 因為Line 1的資訊是目前系統被指定的`UCX_TLS`，是一個全局的資訊，而`ucp_context`正是負責將全局資訊儲存起來的物件，所以我原先想要在`ucp_context.c`的`ucp_init_version`裡，在`config`和`ucp_context`都準備就緒後，呼叫`ucp_config_print`將`UCX_TLS`的資訊印出。
- 這麼做同樣可以印出符合預期的資訊，但是輸出格式卻是無法符合sample output，在sample output內，Line 1和Line 2是交錯的，而我的打印結果如下：
![image](https://i.imgur.com/QPD2mec.png)
- 如圖，`UCX_TLS`的資訊會先被print出來，後續才是UCX所選擇的TLS transport method，而這也會導致無法通過測資。
- 那麼根據使用`UCX_LOG_LEVEL=info`的觀察，我推測要讓Line 1和Line 2交錯print出，Line 1的資訊也是需要在創建`ucp_ep`的時候進行打印。
- 順著`ucp_ep_create`一路trace下去，抵達`ucp_worker_get_ep_config`，這個function是initial endpoint config的地方，我在此處呼叫`ucp_config_read`，讀取全局config的資訊，並將該`config`放入`ucp_config_print`裡，使用`UCS_CONFIG_PRINT_TLS`指定要打印`UCX_TLS`的資訊。
- 而在`ucs_config_parser_print_opts`當中，需要將`config`裡關於`UCX_TLS`的資訊取出並打印，就要使用到內建的function。原先我是使用`ucs_config_parser_print_env_vars`，但是除了`UCX_TLS`的資訊，還會順便打印出`UCX_NET_DEVICE`。
- 隨後我發現另一個function：`ucs_config_parser_get_value`，將要查找的字段訊息`TLS`傳入，就可以得到`UCX_TLS`的資訊，從而可以打印出Line 1的資訊。
![image](https://i.imgur.com/tAscstD.png)
- 此處需要注意的是，因為UCX會將環境變數分為前綴和字段，前綴默認是`UCX_`，所以輸入查找的字段僅需為`TLS`而非`UCX_TLS`，否則會造成找不到的情況發生。
- 再來是print Line 2的部份，根據`UCX_LOG_LEVEL=info`的資訊，可以看到選擇的TLS的資訊會在`ucp_worker.c`的1855行被print出，找到這個地方，會發現是名為`ucp_worker_print_used_tls`的funtion進行打印。
- 往回trace function，便可以找到該function在`ucp_worker_get_ep_config`內被呼叫，正是我們打印Line 1時所呼叫的`ucp_config_print`的function內。所以為了讓Line 1和2的順序正確，`ucp_config_print`才需要在`ucp_worker_print_used_tls`前被呼叫。
![image](https://i.imgur.com/Pex1SJK.png)
- 回到`ucp_worker_print_used_tls`，選擇的`UCX_TLS`的資訊是由`ucs_info`所打印，該打印資訊只會在`UCX_LOG_LEVEL=info`時才會顯現，所以我們僅需在該行下方使用`printf`手動打印選擇的`UCX_TLS`即可。
![image](https://i.imgur.com/RwJQqz8.png)


2. How do the functions in these files call each other? Why is it designed this way?
- 結合question 1，在`ucp_worker.c`裡面，`ucp_worker`在創建`ucp_ep`的時候，會需要獲取該`ucp_ep`傳輸時所需要的config，所以在此處就會對`ep_config`做初始化，並且在最後呼叫`ucp_worker_print_used_tls`將目前所選的transport layer的資訊打印出來，也就是Line 2的資訊。
- 而因為Line 1的資訊需要在Line 2之前打印出來，所以在同個`ucp_worker_get_ep_config` function內，首先呼叫`ucp_config_read`，取得全局`config`的資訊，再使用`ucp_config_print`傳入`UCS_CONFIG_PRINT_TLS`，將所需要的`UCX_TLS`資訊給打印出來。而這個步驟需要放在`ucp_worker_print_used_tls`之前。
![hw5.drawio](https://i.imgur.com/pLQktZf.png)
- 那麼至此，就完成了此次作業實作的部份。


3. Observe when Line 1 and 2 are printed during the call of which UCP API?
- 在`ucp_worker`創建完後，要對需要通訊的雙方創建`ucp_ep`，所以此時會呼叫`ucp_ep_create`，在function內，會呼叫`ucp_ep_create_to_sock_addr`，使用對方的`socket address`進行通訊建立。
- 隨後便是呼叫`ucp_ep_init_create_wireup`，初始化需要的訊息，最後呼叫`ucp_wireup_ep_create`正式創建完`ucp_ep`；
- 而在獲取初始化訊息所需要的`ep_config`時，正是使用到了`ucp_worker_get_ep_config`。在該function內，就會有前面實作所提到的那些打印function，其中包含Line 1和Line 2的打印資訊。
- 所以Line 1和Line 2打印時，是在`ucp_ep_crete`時，獲取`ep_config`的`ucp_worker_get_ep_config`中被打印出來。

4. Does it match your expectations for questions **1-3**? Why?
- Line 2的打印時機以及位置是符合我對於UCX的理解的。因為在`ucp_ep`建立時，我們才會根據需求選擇需要傳輸的方式，那麼在建立connection的時候，就會呼叫`ucp_worker_get_ep_config`，獲取`ep_config`，再根據獲取的資訊，將UCX選用的TLS設定好。
- 而Line 1的打印時機以及位置與我對UCX的理解有稍微出入，但整體而言是符合的。首先因為`UCX_TLS`的資訊是全局的configuration，我的理解是這個資訊應該是要被放在`ucp_context`裡面，而在trace code之後也發現確實如此。在`ucp_init`時會呼叫`ucp_init_version`，在此處就會使用`ucp_config_read`讀取全局的configuration，其中當然包括`UCX_TLS`的資訊。隨後就會根據`config`的資訊將`ucp_context`創建完成，所以`UCX_TLS`會被放在`ucp_context`裡面。
- 到前面的trace code都符合我對於UCX的理解，所以我自然而然會想要在`ucp_init_version`時就將Line 1的資訊打印出來，但這樣打印出來的結果就如同我question 1所提到的那樣，雖然正確但格式有誤，所以我才將打印的位置改到`ucp_worker_get_ep_config`當中，這部分是與我的理解有一些出入的。
- 可是轉念一想，Line 2是被UCX選用的TLS，那麼打印資訊的上一行順便打印目前全局資訊的`UCX_TLS`，做為資訊參考，從而可以對比全局資訊和該`ucp_ep`的TLS，所以Line 1在`ucp_worker_get_ep_config`當中進行打印好像也蠻合理的。

5. In implementing the features, we see variables like `lanes`, `tl_rsc`, `tl_name`, `tl_device`, `bitmap`, `iface`, etc., used to store different Layer's protocol information. Please explain what information each of them stores.
- `lanes`: 在`ucp_ep.h`中可以找到較多`lanes`相關的資訊。主要用途為支援multi-path的通信和傳輸協議選擇，儲存了與傳輸層相關的資源索引、路徑索引及操作類型等資訊。
- `tl_rsc`: 在`ucp_context.h`的209行可以找到，是一個可以表示傳輸網路資源的物件，儲存傳輸協議的具體資源資訊，例如：Transport Name，Hardware device name等等
- `tl_name`: 即為Transport Name，定義在`uct_tl_resource_desc_t`當中，儲存傳輸協議的名稱，是一個`char array`。
- `tl_device`: 即為Device Name，同樣定義在`uct_tl_resource_desc_t`當中，儲存設備名稱的字串，用於指定傳輸層使用的physical device。
- `bitmap`: 型別是`ucp_tl_bitmap_t`，使用bitmap的方式儲存著每一個TL resource是否正在使用，可以用於查找資源狀態，啟用或禁用傳輸資源。
- `iface`: `iface`本身在UCX中是傳輸層的一個抽象，代表了某一個傳輸層協議的物件。在`tl.h`當中的`uct_iface_ops`為傳輸操作的interface定義。而使用`uct_iface_t`創建的物件，儲存著的資訊即為操作傳輸層的介面。


## 3. Optimize System

1. Below are the current configurations for OpenMPI and UCX in the system. Based on your learning, what methods can you use to optimize single-node performance by setting UCX environment variables?

```plaintext
-------------------------------------------------------------------
/opt/modulefiles/openmpi/ucx-pp:
module-whatis {OpenMPI 4.1.6}
conflict mpi
module load ucx/1.15.0
prepend-path PATH /opt/openmpi-4.1.6/bin
prepend-path LD_LIBRARY_PATH /opt/openmpi-4.1.6/lib
prepend-path MANPATH /opt/openmpi-4.1.6/share/man
prepend-path CPATH /opt/openmpi-4.1.6/include
setenv UCX_TLS ud_verbs
setenv UCX_NET_DEVICES ibp3s0:1
------------------------------------------------------------------
```
- 根據課程講義，`ud_verbs`應該代表的是`Unreliable Datagram`的傳輸協議，並且是基於`RDMA`的技術進行傳輸的Transport Layer。而根據上課所學到的關於`RDMA`的知識，該傳輸技術可以讓資料從一台機器的記憶體直接傳輸到另一台機器的記憶體上，完全繞過CPU，從而降低延遲並提高效能。
- 然而為了實現這種高效能的傳輸，`RDMA`需要在傳輸前先setup好`Control Path`，這會造成一個額外的初始化的時間開銷。所以`RDMA`比較適用於multi-node的高效傳輸，對於single-node的傳輸，可以選用更輕量的傳輸方式。
- 在使用`ucx_info -d`指令查看所有可以使用的`UCX_TLS`後，我發現可以使用`shm`(share memory)和`cma`(cross-memory-attach)去加速在single-node環境底下的傳輸。
- 因為在single-node的環境下，process直接存取memory的速度會比使用`ud_verbs`這種需要透過network(例如InfiniBand)進行傳輸的速度更快。


2. Please use the following commands to test different data sizes for latency and bandwidth, to verify your ideas:

```bash
module load openmpi/ucx-pp
mpiucx -n 2 $HOME/UCX-lsalab/test/mpi/osu/pt2pt/osu_latency
mpiucx -n 2 $HOME/UCX-lsalab/test/mpi/osu/pt2pt/osu_bw
```

3. Please create a chart to illustrate the impact of different parameter options on various data sizes and the effects of different testsuite.

- `osu_latency` and `osu_bw`:
    | UCX_TLS    | Data Size | Time       | Bandwidth   |
    | ---------- | --------- | ---------- | ----------- |
    | `ud_verbs` | 1024      | 5.57       | 1183.74     |
    | `shm`      | 1024      | **0.49**   | **3898.28** |
    | `shm,cma`  | 1024      | 0.50       | 3731.27     |
    | `ud_verbs` | 131072    | 65.02      | 2434.31     |
    | `shm`      | 131072    | 17.39      | 8603.72     |
    | `shm,cma`  | 131072    | **17.21**  | **8945.65** |
    | `ud_verbs` | 4194304   | 1838.91    | 2352.40     |
    | `shm`      | 4194304   | 943.56     | 6029.26     |
    | `shm,cma`  | 4194304   | **935.92** | **7118.73** |
    
![img](https://i.imgur.com/ZSAzwHQ.png)
- special case:
    | UCX_TLS    | Data Size | Bandwidth    |
    | ---------- | --------- | ------------ |
    | `ud_verbs` | 8192      | 2030.59      |
    | `shm`      | 8192      | 10384.28     |
    | `shm,cma`  | 8192      | **10514.12** |
    
![img](https://i.imgur.com/nb5NugW.png)
    
- 實驗數據因為server狀態關係會有些許差別，所以以上的實驗數據我是分別執行了五次取平均作為最終數據。
- 在結果呈現上，我選用data size`1024`作為小size，`131072`作為中size和`4194304`作為大size，觀察不同size情況下`latency`和`bandwidth`的差別。
- 其中，在`osu_bw`中，因為發現使用`shm`和`shm,cma`時，`bandwidth`最高可以來到10000以上，所以特別將該對應的data size`8192`記錄下來。

4. Based on the chart, explain the impact of different TLS implementations and hypothesize the possible reasons (references required).
- 根據實驗數據，很清楚的可以看出，使用`shm`或是`shm,cma`在single-node的環境底下，不論是在任何size下，`latency`和`bandwidth`的表現都比`ud_verbs`來得好上很多，這證實了我前面的想法：`RDMA`需要依賴網路協議進行傳輸，其效能表現在single-node的環境下不會比直接存取memory的方法（例如`shared memory`和`cross-memory attach`）來的更適合，所以`RDMA`比較適合在multi-node的環境下進行高效能傳輸。
- 再觀察兩個實驗數據結果可以發現，`shm`在小size(1024)的情況下表現略優於`shm,cma`，這可能是因為`cma`的初始化需要一點額外的時間開銷，而且其本身設計初衷是為了優化大數據的傳輸效率，讓其在小size下比單純使用`shm`要來的慢一點。
- 但是在中size(131072)和大size(4194304)時，`shm,cma`的效能就比`shm`來得好，這可能是因為`shm,cma`在處理較大的數據時，結合`shm`的zero-copy的能力，再通過`cma`的跨記憶體訪問技術，可以直接訪問其他process的memory，優化了傳輸效率。
- 一個有趣的結果是`shm`和`shm,cma`的最好的`bandwidth`是出現在了8192這個size下，其`bandwidth`突破了10000，比`ud_verbs`的2030多了接近5倍。在經過查找了關於`osu`benchmark的介紹和一些performance分析的資料後，我猜測是因為8192這個size剛好處於share memory和cross-memory attach處理單位的最大值附近，所以可以充分利用memory access的bandwidth，從而達到最大效能。
- References:
    - https://openucx.readthedocs.io/en/master/faq.html
    - https://dl.acm.org/doi/10.1145/2616498.2616532
    - https://www.quora.com/What-are-the-advantages-and-disadvantages-of-a-shared-memory-architecture-compared-to-a-distributed-memory-architecture-for-parallel-computing-and-in-what-types-of-applications-would-each-architecture-be-most
    - https://github.com/forresti/osu-micro-benchmarks/blob/master/README
### Advanced Challenge: Multi-Node Testing

This challenge involves testing the performance across multiple nodes. You can accomplish this by utilizing the sbatch script provided below. The task includes creating tables and providing explanations based on your findings. Notably, Writing a comprehensive report on this exercise can earn you up to 5 additional points.

- For information on sbatch, refer to the documentation at [Slurm's sbatch page](https://slurm.schedmd.com/sbatch.html).
- To conduct multi-node testing, use the following command:

```bash
cd ~/UCX-lsalab/test/
sbatch run.batch
```

## 4. Experience & Conclusion

- 本次作業所花費的時間應該是所有作業中最久的一次，因為先前的作業只需要知道演算法的原理，並且運用上課所學到的平行方法去將演算法進行平行優化即可，雖然coding上時常遇到bug也會需要解決很久，但此次作業因為涉及到底層的設計，加上對於UCX並沒有那麼了解，所以在閱讀課程講義，理解Hw Spec，再去trace code的時間遠比實作來的更久，並且一開始面對著這麼大型的專案，即使透過課程講義和Hw Spec大致知道所需要的東西會在`ucp_context.c`，`ucp_worker.c`和`ucp_ep.c`這三個檔案內，可是仍然不知道該從何開始看起。
- 所幸Report內有給hint讓我從`ucp_hello_world.c`開始看起，讓我對於UCX的整個運行的過程可以有大致上的了解，再一個一個function進去trace，慢慢的就可以把上課所學到的知識和code實作給兜了起來。再加上使用`UCX_LOG_LEVEL=info`可以取得一些打印資訊，更幫助我找到此次作業的實作大致上會在什麼地方。
- 透過此次作業，我對於UCX的理解又更加深了幾分，也更加理解到，高效能系統不單單只是演算法的優化以及平行技巧的運用這麼簡單，還需要搭配很多底層資源的支持和傳輸協議的選擇，才能打造出一個適合該應用的高效能運算系統。