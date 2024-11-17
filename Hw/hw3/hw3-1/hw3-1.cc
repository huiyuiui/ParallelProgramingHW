#include <stdio.h>
#include <stdlib.h>
#include <sched.h>
#include <omp.h>
#define CHUNK_SIZE 10

const int INF = ((1 << 30) - 1);
const int V = 50010;
void input(char* inFileName);
void output(char* outFileName);

void block_FW(int B);
int ceil(int a, int b);
void cal(int B, int Round, int block_start_x, int block_start_y, int width_num_blocks, int height_num_blocks);

int n, m, NUM_THREADS;
static int Dist[V][V];

int main(int argc, char* argv[]) {
    input(argv[1]);
    int omp_threads = omp_get_max_threads();
    int B = 512;
    block_FW(B);
    output(argv[2]);
    return 0;
}

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                Dist[i][j] = 0;
            } else {
                Dist[i][j] = INF;
            }
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0]][pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (Dist[i][j] >= INF) Dist[i][j] = INF;
        }
        fwrite(Dist[i], sizeof(int), n, outfile);
    }
    fclose(outfile);
}

int ceil(int a, int b) { return (a + b - 1) / b; }

void block_FW(int B) {
    int round = ceil(n, B);
    for (int r = 0; r < round; ++r) {
        /* Phase 1*/
        cal(B, r, r, r, 1, 1); // pivot block

        /* Phase 2*/
        cal(B, r, r, 0, 1, r); // pivot row: from 0 to now index
        cal(B, r, r, r + 1, 1, round - r - 1); // pivot row: from now index + 1 to end
        cal(B, r, 0, r, r, 1);  // pivot col: from 0 to now index
        cal(B, r, r + 1, r, round - r - 1, 1); // pivot col: from now index + 1 to end

        /* Phase 3*/
        cal(B, r, 0, 0, r, r); // other: left upper
        cal(B, r, 0, r + 1, r, round - r - 1); // other: right upper
        cal(B, r, r + 1, 0, round - r - 1, r); // other: left lower
        cal(B, r, r + 1, r + 1, round - r - 1, round - r - 1); // ohter: right lower
    }
}

void cal(
    int B, int Round, int block_start_y, int block_start_x, int height_num_blocks, int width_num_blocks) {
    int block_end_y = block_start_y + height_num_blocks;
    int block_end_x = block_start_x + width_num_blocks;

    // outer for loop: how many blocks in this calculation
    for (int b_i = block_start_y; b_i < block_end_y; ++b_i) {
        for (int b_j = block_start_x; b_j < block_end_x; ++b_j) {
            // To calculate B*B elements in the block (b_i, b_j)
            // For each block, it need to compute B times
            for (int k = Round * B; k < (Round + 1) * B && k < n; ++k) { // each phase will perform B iterations
                // To calculate original index of elements in the block (b_i, b_j)
                // For instance, original index of (0,0) in block (1,2) is (2,5) for V=6,B=2
                // (b_i, b_j) is block index
                int global_start_y = b_i * B;
                int global_end_y = (b_i + 1) * B;
                int global_start_x = b_j * B;
                int global_end_x = (b_j + 1) * B;

                if (global_end_y > n) global_end_y = n;
                if (global_end_x > n) global_end_x = n;

                # pragma omp parallel for schedule(dynamic) shared(Dist)
                for (int i = global_start_y; i < global_end_y; ++i) {
                    for (int j = global_start_x; j < global_end_x; ++j) {
                        if (Dist[i][k] + Dist[k][j] < Dist[i][j]) {
                            Dist[i][j] = Dist[i][k] + Dist[k][j];
                        }
                    }
                }
            }
        }
    }
}
