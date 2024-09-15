## MPI Program

### 檔案開頭
```c
#include "mpi.h"
#include <stdio.h>
#include <string.h>
```
這裡包含了 MPI 庫（`mpi.h`）以及標準的輸入輸出與字串操作函式庫。

### `main` 函式
```c
int main(int argc, char *argv[])
```
`main` 函式是程式的入口，接收命令列參數。

### MPI 初始化
```c
int i, rank, size, namelen;
char name[MPI_MAX_PROCESSOR_NAME];
MPI_Status stat;

MPI_Init(&argc, &argv);
```
- `rank`: 表示每個處理器在 MPI 通訊器中的編號。
- `size`: 表示整個 MPI 通訊器中的處理器總數。
- `namelen`: 儲存處理器名稱的長度。
- `name`: 用來儲存處理器名稱的字串，`MPI_MAX_PROCESSOR_NAME` 是處理器名稱的最大長度。
- `stat`: 儲存 MPI 傳送/接收的狀態。

`MPI_Init` 函式用來初始化 MPI 環境。

### 獲取處理器數量、編號與名稱
```c
MPI_Comm_size(MPI_COMM_WORLD, &size);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Get_processor_name(name, &namelen);
```
- `MPI_Comm_size`：獲取處理器總數。
- `MPI_Comm_rank`：獲取當前處理器的編號（rank）。
- `MPI_Get_processor_name`：取得當前處理器的名稱，並將名稱的長度儲存到 `namelen` 中。
- `MPI_COMM_WORLD`：包含所有 MPI 程式進程的一個通訊群組，並且允許這些進程彼此之間交換訊息。

### 主要邏輯（Master-Slave 模型）
- **主節點（Rank 0）**
  如果處理器的編號是 0，它就是主節點，負責接收其他處理器的訊息，並顯示處理器的資訊：
  ```c
  if (rank == 0) {
      printf("Hello world: rank %d of %d running on %s\n", rank, size, name);

      for (i = 1; i < size; i++) {
          MPI_Recv(&rank, 1, MPI_INT, i, 1, MPI_COMM_WORLD, &stat);
          MPI_Recv(&size, 1, MPI_INT, i, 1, MPI_COMM_WORLD, &stat);
          MPI_Recv(&namelen, 1, MPI_INT, i, 1, MPI_COMM_WORLD, &stat);
          MPI_Recv(name, namelen + 1, MPI_CHAR, i, 1, MPI_COMM_WORLD, &stat);
          printf("Hello world: rank %d of %d running on %s\n", rank, size, name);
      }
  }
  ```
  這段程式碼會：
  - 首先由 rank 0 的主節點輸出自己的資訊。
  - 然後通過 `MPI_Recv` 從每個其他節點（rank 1 到 `size - 1`）接收 rank、size、處理器名稱長度（`namelen`）以及處理器名稱（`name`），並將它們逐一顯示。

- **從節點（Rank 1 及以上）**
  其他處理器（非 rank 0）負責將自己的 rank、size、`namelen` 和 `name` 傳送給主節點：
  ```c
  else {
      MPI_Send(&rank, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
      MPI_Send(&size, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
      MPI_Send(&namelen, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
      MPI_Send(name, namelen + 1, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
  }
  ```
  每個從節點通過 `MPI_Send` 將上述資訊傳送給 rank 0 的主節點。

### 結束 MPI 程式
```c
MPI_Finalize();
return (0);
```
`MPI_Finalize` 函式用來結束 MPI 程式，釋放資源。