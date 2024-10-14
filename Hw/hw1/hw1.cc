#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <iostream>
#include <boost/sort/spreadsort/float_sort.hpp>

using namespace std;

// pick minimum half
void merge_back(float** data_ptr, float* temp, int chunk_size){
    float* data = *data_ptr;
    int now_idx, back_idx;
    now_idx = back_idx = 0;

    float* merged_data = new float[chunk_size];
    for(int i = 0; i < chunk_size; i++){
        if(data[now_idx] <= temp[back_idx]){
            merged_data[i] = data[now_idx];
            now_idx += 1;
        }
        else{
            merged_data[i] = temp[back_idx];
            back_idx += 1;
        }
    }

    std::swap(*data_ptr, merged_data);
    delete[] merged_data;

    return;
}

// pick maximum half
void merge_front(float** data_ptr, float* temp, int chunk_size, int another_chunk_size){
    float* data = *data_ptr;
    int now_idx, front_idx;
    now_idx = chunk_size - 1;
    front_idx = another_chunk_size - 1;

    float* merged_data = new float[chunk_size];
    for(int i = chunk_size - 1; i >= 0; i--){
        if(data[now_idx] < temp[front_idx] && front_idx >= 0){ // front_idx < 0 case only happens in final rank
            merged_data[i] = temp[front_idx];
            front_idx -= 1;
        }
        else{
            merged_data[i] = data[now_idx];
            now_idx -= 1;
        }
    }

    std::swap(*data_ptr, merged_data);
    delete[] merged_data;

    return;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Status stat;

    int n = atoi(argv[1]);
    char *input_filename = argv[2];
    char *output_filename = argv[3];

    // boundary case checking
    int chunk_size, start, end, final_rank, normal_chunk_size, final_chunk_size;
    if(n >= size){ // normal case, n >= p
        chunk_size = n / size;
        final_rank = size - 1;
    }
    else{ // special case n < p
        chunk_size = 1;
        final_rank = n - 1; 
        size = n;
    }
    normal_chunk_size = chunk_size; // since final rank will modify chunk_size, it need to know others chunk_size
    
    MPI_Comm MPI_COMM_ACTIVE;
    if (rank <= final_rank) MPI_Comm_split(MPI_COMM_WORLD, 0, rank, &MPI_COMM_ACTIVE); // assign color 0 to active processors
    else{ // close processors that not using
        MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, rank, &MPI_COMM_ACTIVE); // since this operation is collective, all processors need to engage in.
        MPI_Finalize();
        return 0;
    }
    
    // calculate the computation range
    start = rank * chunk_size;
    if(rank != final_rank) end = (rank + 1) * chunk_size;
    else{
        end = n;
        chunk_size = end - start;
        final_chunk_size = chunk_size; // final_rank - 1 need to know final_rank's size
        if(rank != 0)  // special case, p = 1, then rank 0 is final rank, so don't need to communicate
            MPI_Send(&final_chunk_size, 1, MPI_INT, final_rank - 1, 1, MPI_COMM_ACTIVE); 
    }
    
    // init data array
    MPI_File input_file, output_file;
    float* data = new float[chunk_size];
    float* temp_data; // store data from other processor
    float* final_rank_data; // in order to store the data from final rank (usually bigger than chunk_size)

    if(rank != final_rank) temp_data = new float[chunk_size];
    else temp_data = new float[normal_chunk_size];
    
    if(rank == (final_rank - 1)){ // only final_rank - 1 need to init final_rank_data
        MPI_Recv(&final_chunk_size, 1, MPI_INT, final_rank, 1, MPI_COMM_ACTIVE, &stat);
        final_rank_data = new float[final_chunk_size];
    }

    MPI_File_open(MPI_COMM_ACTIVE, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    MPI_File_read_at(input_file, sizeof(float) * start, data, chunk_size, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&input_file);

    /* main program */
    boost::sort::spreadsort::float_sort(data, data + chunk_size); // local sort

    int not_done = (final_rank == 0 ? 0 : 1); // special case rank 0 is final rank, then not_done=0, will not entry while loop
    int odd_even_flag = 1; // 0: odd, 1: even
    int rtn_even, rtn_odd, rtn;
    rtn = rtn_even = rtn_odd = 0;
    float maximum, minimum; // be used to check whether merge will happen

    while(not_done >= 1){
        if(odd_even_flag == 1){ // even phase
            if(rank % 2 == 0 && rank < final_rank){ // pass backward
                maximum = data[chunk_size - 1]; // minimum will be temp_data[0]
                MPI_Sendrecv(&maximum, 1, MPI_FLOAT, rank + 1, 2, &minimum, 1, MPI_FLOAT, rank + 1, 2, MPI_COMM_ACTIVE, &stat); // notice that tag should different from below data exchange
                rtn_even = maximum <= minimum ? 0 : 1; // if current data's maximum smaller then backward data's minimum, that is, no swap will happen
                if(rtn_even){ // only change data when merge will happen
                    if(rank != (final_rank - 1)){
                        MPI_Sendrecv(data, chunk_size, MPI_FLOAT, rank + 1, 1, temp_data, chunk_size, MPI_FLOAT, rank + 1, 1, MPI_COMM_ACTIVE, &stat);
                        merge_back(&data, temp_data, chunk_size);
                    }
                    else{
                        MPI_Sendrecv(data, chunk_size, MPI_FLOAT, rank + 1, 1, final_rank_data, final_chunk_size, MPI_FLOAT, rank + 1, 1, MPI_COMM_ACTIVE, &stat);
                        merge_back(&data, final_rank_data, chunk_size);
                    }
                }
            }
            else if(rank % 2 == 1){ // pass forward
                minimum = data[0]; // maximum will be temp_data[chunk_size - 1]
                MPI_Sendrecv(&minimum, 1, MPI_FLOAT, rank - 1, 2, &maximum, 1, MPI_FLOAT, rank - 1, 2, MPI_COMM_ACTIVE, &stat);
                rtn_even = maximum <= minimum ? 0 : 1; // if current data's minimum bigger then forward data's maximum, that is, no swap will happen
                if(rtn_even){ // only change data when merge will happen
                    if(rank != final_rank)
                        MPI_Sendrecv(data, chunk_size, MPI_FLOAT, rank - 1, 1, temp_data, chunk_size, MPI_FLOAT, rank - 1, 1, MPI_COMM_ACTIVE, &stat);
                    else
                        MPI_Sendrecv(data, chunk_size, MPI_FLOAT, rank - 1, 1, temp_data, normal_chunk_size, MPI_FLOAT, rank - 1, 1, MPI_COMM_ACTIVE, &stat);
                    merge_front(&data, temp_data, chunk_size, normal_chunk_size);
                } 
            }
            odd_even_flag = 0;
        }
        if(odd_even_flag == 0){ // odd phase
            if(rank % 2 == 1 && rank < final_rank){ // pass backward
                maximum = data[chunk_size - 1]; // minimum will be temp_data[0]
                MPI_Sendrecv(&maximum, 1, MPI_FLOAT, rank + 1, 2, &minimum, 1, MPI_FLOAT, rank + 1, 2, MPI_COMM_ACTIVE, &stat);
                rtn_odd = maximum <= minimum ? 0 : 1; // if current data's maximum smaller then backward data's minimum, that is, no swap will happen
                if(rtn_odd){ // only change data when merge will happen
                    if(rank != final_rank - 1){
                        MPI_Sendrecv(data, chunk_size, MPI_FLOAT, rank + 1, 1, temp_data, chunk_size, MPI_FLOAT, rank + 1, 1, MPI_COMM_ACTIVE, &stat);
                        merge_back(&data, temp_data, chunk_size);
                    }
                    else{
                        MPI_Sendrecv(data, chunk_size, MPI_FLOAT, rank + 1, 1, final_rank_data, final_chunk_size, MPI_FLOAT, rank + 1, 1, MPI_COMM_ACTIVE, &stat);
                        merge_back(&data, final_rank_data, chunk_size);
                    }
                }
            }
            else if(rank % 2 == 0 && rank != 0){ // pass forward
                minimum = data[0]; // maximum will be temp_data[chunk_size - 1]
                MPI_Sendrecv(&minimum, 1, MPI_FLOAT, rank - 1, 2, &maximum, 1, MPI_FLOAT, rank - 1, 2, MPI_COMM_ACTIVE, &stat);
                rtn_odd = maximum <= minimum ? 0 : 1; // if current data's minimum bigger then forward data's maximum, that is, no swap will happen
                if(rtn_odd){ // only change data when merge will happen
                    if(rank != final_rank)
                        MPI_Sendrecv(data, chunk_size, MPI_FLOAT, rank - 1, 1, temp_data, chunk_size, MPI_FLOAT, rank - 1, 1, MPI_COMM_ACTIVE, &stat);
                    else
                        MPI_Sendrecv(data, chunk_size, MPI_FLOAT, rank - 1, 1, temp_data, normal_chunk_size, MPI_FLOAT, rank - 1, 1, MPI_COMM_ACTIVE, &stat);
                    merge_front(&data, temp_data, chunk_size, normal_chunk_size);
                }
            }
            odd_even_flag = 1;
        }

        // termination checking
        rtn = rtn_even + rtn_odd;
        MPI_Allreduce(&rtn, &not_done, 1, MPI_INT, MPI_SUM, MPI_COMM_ACTIVE); // if any change happened in even or odd phase, it will continue
    }

    MPI_File_open(MPI_COMM_ACTIVE, output_filename, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    MPI_File_write_at(output_file, sizeof(float) * start, data, chunk_size, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&output_file);

    MPI_Finalize();
    return 0;
}

