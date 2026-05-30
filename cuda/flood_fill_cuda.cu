/*
flood_fill_cuda.cu

CUDA Batch Flood Fill using shared maze dataset.

Data structure:
    maze size: 16 x 16 = 256
    walls_batch[num_maze * 256]
    distance_batch[num_maze * 256]

xth maze id, yth cell id:
    idx_global = id_maze * 256 + id_cell

CUDA Mapping:
    blockIdx.x = id_maze -> 1 cuda block deal with 1 maze
    threadIdx.x = id_cell -> 1 thread deal with 1 cell(256 cells per maze)

Run empty maze benchmark:
    flood_fill_cuda.exe 10000

Run shared maze benchmark:
    flood_fill_cuda.exe 10000 ..\\data\\mazes_10000_seed0.bin
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#define MAZE_SIZE 16
#define NUM_CELLS 256
#define INF 1000000000

// Wall bit 
#define WALL_NORTH 0x01
#define WALL_EAST  0x02
#define WALL_SOUTH 0x04
#define WALL_WEST  0x08

// Direction
#define NORTH 0
#define EAST  1
#define SOUTH 2
#define WEST  3

// CUDA error checking macro
#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = call;                               \
        if (err != cudaSuccess) {                             \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n",       \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1);                                          \
        }                                                     \
    } while (0)

// __host__: call by CPU
// __device__: call by GPU
__device__ __host__ int grid_to_idx(int row, int col)
{
    return row * MAZE_SIZE + col;
}


__device__ __host__ int idx_to_row(int idx)
{
    return idx / MAZE_SIZE;
}


__device__ __host__ int idx_to_col(int idx)
{
    return idx % MAZE_SIZE;
}


__device__ bool has_wall(uint8_t wall, int dir)
{
    if (dir == NORTH) 
    {
        return wall & WALL_NORTH;
    }
    if (dir == EAST)  
    {
        return wall & WALL_EAST;
    }
    if (dir == SOUTH) 
    {
        return wall & WALL_SOUTH;
    }
    else
    {
        return wall & WALL_WEST;
    }    
}


/*
One block for one maze
One thread for one cell
*/
// CUDA kernel
// __global__: call by CPU, execute on GPU
__global__ void flood_fill_kernel(
    const uint8_t* walls_batch,
    int* distance_batch,
    int num_mazes
)

{
    int id_maze = blockIdx.x;
    int id_cell = threadIdx.x;

    // Location out of maze
    if (id_maze >= num_mazes || id_cell >= NUM_CELLS) 
    {
        return;
    }
    
    // Shared memory
    __shared__ uint8_t walls[NUM_CELLS];
    __shared__ int distance[NUM_CELLS];
    __shared__ int distance_next[NUM_CELLS]; // Store update distance, preventing threads from overwriting each other's update results within the same round
    __shared__ int changed; // changed == 0 -> flood fill converge

    int idx_global = id_maze * NUM_CELLS + id_cell;

    walls[id_cell] = walls_batch[idx_global];

    int row = idx_to_row(id_cell);
    int col = idx_to_col(id_cell);

    // Goal point, A.K.A end point
    bool is_goal =
        (row == 7 && col == 7) ||
        (row == 7 && col == 8) ||
        (row == 8 && col == 7) ||
        (row == 8 && col == 8);

    distance[id_cell] = is_goal ? 0 : INF;
    distance_next[id_cell] = distance[id_cell];

    // Synchronous threads before iterative relaxation
    __syncthreads();

    /*
    Iterative relaxation:
        distance[current] = min(distance[current], distance[neighbor] + 1)

    Each block has 256 threads.
    Each cell checks each neighbor's distance
    */
    for (int iter = 0; iter < NUM_CELLS; iter++) 
    {
        if (id_cell == 0) 
        {
            changed = 0;
        }

        __syncthreads();

        int best = distance[id_cell];

        //North side
        if (!has_wall(walls[id_cell], NORTH) && row > 0) 
        {
            int id_neighbor = grid_to_idx(row - 1, col);
            if (distance[id_neighbor] + 1 < best) 
            {
                best = distance[id_neighbor] + 1;
            }
        }

        // East side
        if (!has_wall(walls[id_cell], EAST) && col < MAZE_SIZE - 1) 
        {
            int id_neighbor = grid_to_idx(row, col + 1);
            if (distance[id_neighbor] + 1 < best) 
            {
                best = distance[id_neighbor] + 1;
            }
        }

        // South side
        if (!has_wall(walls[id_cell], SOUTH) && row < MAZE_SIZE - 1) 
        {
            int id_neighbor = grid_to_idx(row + 1, col);
            if (distance[id_neighbor] + 1 < best) 
            {
                best = distance[id_neighbor] + 1;
            }
        }

        //West side
        if (!has_wall(walls[id_cell], WEST) && col > 0) 
        {
            int id_neighbor = grid_to_idx(row, col - 1);
            if (distance[id_neighbor] + 1 < best) 
            {
                best = distance[id_neighbor] + 1;
            }
        }

        distance_next[id_cell] = best;

        __syncthreads();

        if (distance_next[id_cell] != distance[id_cell]) {
            atomicAdd(&changed, 1);
        }

        __syncthreads();

        distance[id_cell] = distance_next[id_cell];

        __syncthreads();

        // Early stop (flood fill converged)
        if (changed == 0) {
            break;
        }

        __syncthreads();
    }

    distance_batch[idx_global] = distance[id_cell];
}

// Generating empty maze (with outside walls)
void generate_empty_maze_batch(uint8_t* walls_batch, int num_mazes)
{
    for (int maze = 0; maze < num_mazes; maze++) 
    {
        for (int row = 0; row < MAZE_SIZE; row++) 
        {
            for (int col = 0; col < MAZE_SIZE; col++) 
            {
                uint8_t wall = 0;

                if (row == 0) 
                    wall |= WALL_NORTH;
                if (row == MAZE_SIZE - 1) 
                    wall |= WALL_SOUTH;
                if (col == 0) 
                    wall |= WALL_WEST;
                if (col == MAZE_SIZE - 1) 
                    wall |= WALL_EAST;

                int id_cell = grid_to_idx(row, col);
                int idx_global = maze * NUM_CELLS + id_cell;

                walls_batch[idx_global] = wall;
            }
        }
    }
}

bool load_maze_batch_from_file(
    const char* file_path,
    uint8_t* walls_batch,
    int num_mazes
)
{
    FILE* fp = fopen(file_path, "rb");

    if (!fp) {
        printf("Failed to open maze file: %s\n", file_path);
        return false;
    }

    size_t expected_count = (size_t)num_mazes * NUM_CELLS;

    size_t read_count = fread(
        walls_batch,
        sizeof(uint8_t),
        expected_count,
        fp
    );

    fclose(fp);

    if (read_count != expected_count) {
        printf("Maze file size mismatch.\n");
        printf("Expected uint8 count: %zu\n", expected_count);
        printf("Read uint8 count: %zu\n", read_count);
        return false;
    }

    printf("Loaded shared maze dataset: %s\n", file_path);

    return true;
}


bool save_distance_batch_to_file(const char* file_path, const int* dist_batch, int num_mazes)
{
    FILE* fp = fopen(file_path, "wb");

    if (!fp) {
        printf("Failed to open output file: %s\n", file_path);
        return false;
    }

    size_t count = (size_t)num_mazes * NUM_CELLS;

    size_t write_count = fwrite(
        dist_batch,
        sizeof(int),
        count,
        fp
    );

    fclose(fp);

    if (write_count != count) {
        printf("Failed to write full distance batch.\n");
        printf("Expected int count: %zu\n", count);
        printf("Written int count: %zu\n", write_count);
        return false;
    }

    printf("Saved GPU distance output to: %s\n", file_path);

    return true;
}

bool save_summary_to_file(
    const char* summary_file,
    const char* maze_file,
    const char* distance_output_file,
    int num_mazes,
    float kernel_time_ms,
    size_t walls_bytes,
    size_t distance_bytes
)
{
    FILE* fp = fopen(summary_file, "w");

    if (!fp) {
        printf("Failed to open summary file: %s\n", summary_file);
        return false;
    }

    float avg_time_per_maze_ms = kernel_time_ms / num_mazes;

    fprintf(fp, "GPU CUDA Flood Fill Benchmark Result\n");
    fprintf(fp, "------------------------------------\n");
    fprintf(fp, "maze_file=%s\n", maze_file ? maze_file : "None_empty_maze_batch");
    fprintf(fp, "num_mazes=%d\n", num_mazes);
    fprintf(fp, "maze_size=16\n");
    fprintf(fp, "num_cells=256\n");
    fprintf(fp, "kernel_time_ms=%.6f\n", kernel_time_ms);
    fprintf(fp, "kernel_time_sec=%.9f\n", kernel_time_ms / 1000.0f);
    fprintf(fp, "avg_time_per_maze_ms=%.9f\n", avg_time_per_maze_ms);
    fprintf(fp, "distance_output_file=%s\n", distance_output_file);
    fprintf(fp, "distance_output_shape=(%d, 256)\n", num_mazes);
    fprintf(fp, "distance_output_dtype=int32\n");
    fprintf(fp, "walls_bytes=%zu\n", walls_bytes);
    fprintf(fp, "distance_bytes=%zu\n", distance_bytes);
    fprintf(fp, "cuda_mapping=one_block_per_maze_256_threads_per_block\n");

    fclose(fp);

    printf("Summary saved to: %s\n", summary_file);

    return true;
}

void print_first_distance_map(const int* h_dist)
{
    printf("\nFirst maze distance map:\n");

    for (int row = 0; row < MAZE_SIZE; row++) {
        for (int col = 0; col < MAZE_SIZE; col++) {
            int idx = grid_to_idx(row, col);
            printf("%3d ", h_dist[idx]);
        }
        printf("\n");
    }
}

int main(int argc, char** argv)
{
    int num_mazes = 10000;

    if (argc >= 2) 
    {
        num_mazes = atoi(argv[1]);
    }

    const char* maze_file = nullptr;

    if (argc >= 3) 
    {
        maze_file = argv[2];
    }

    const char* output_file = "gpu_distance_output.bin";

    if(argc >= 4) 
    {
        output_file = argv[3];
    }

    const char* summary_file = "gpu_benchmark_summary.txt";

    if (argc >= 5) 
    {
        summary_file = argv[4];
    }

    printf("CUDA Batch Flood Fill - 1D Array Version\n");
    printf("Number of mazes: %d\n", num_mazes);

    if (maze_file) 
    {
        printf("Maze file: %s\n", maze_file);
    } 
    else 
    {
        printf("Maze file: None. Using empty maze batch.\n");
    }

    size_t walls_bytes = (size_t)num_mazes * NUM_CELLS * sizeof(uint8_t);
    size_t distance_bytes = (size_t)num_mazes * NUM_CELLS * sizeof(int);

    // CPU ptr (memory allocation on CPU)
    uint8_t* host_walls = (uint8_t*)malloc(walls_bytes);
    int* host_distance = (int*)malloc(distance_bytes);

    if (!host_walls || !host_distance) 
    {
        printf("Host memory allocation failed.\n");
        return 1;
    }

    if (maze_file) 
    {
        bool ok = load_maze_batch_from_file(maze_file, host_walls, num_mazes);

        if (!ok) 
        {
            free(host_walls);
            free(host_distance);
            return 1;
        }
    } 
    else 
    {
        generate_empty_maze_batch(host_walls, num_mazes);
    }

    // GPU ptr (memory allocation on GPU)
    uint8_t* device_walls = nullptr;
    int* device_distance = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&device_walls, walls_bytes));
    CUDA_CHECK(cudaMalloc((void**)&device_distance, distance_bytes));

    // Copy data from CPU to GPU (host to device)
    CUDA_CHECK(cudaMemcpy(device_walls, host_walls, walls_bytes, cudaMemcpyHostToDevice));

    dim3 block(NUM_CELLS);
    dim3 grid(num_mazes);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Starting timer
    CUDA_CHECK(cudaEventRecord(start));

    // Init CUDA kernel
    flood_fill_kernel<<<grid, block>>>(device_walls, device_distance, num_mazes);

    CUDA_CHECK(cudaGetLastError());

    // Stop timer
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    
    // Copy results back to CPU (device to host)
    CUDA_CHECK(cudaMemcpy(host_distance, device_distance, distance_bytes, cudaMemcpyDeviceToHost));

    printf("\nGPU Benchmark Result\n");
    printf("--------------------\n");
    printf("GPU kernel time: %.6f ms\n", ms);
    printf("Average per maze: %.9f ms\n", ms / num_mazes);

    print_first_distance_map(host_distance);

    bool save_ok = save_distance_batch_to_file(
        output_file,
        host_distance,
        num_mazes
    );

    if (!save_ok) 
    {
        printf("Warning: failed to save GPU distance output.\n");
    }
    else 
    {
        printf("GPU distance output saved to: %s\n", output_file);
    }

    bool summary_ok = save_summary_to_file(
        summary_file,
        maze_file,
        output_file,
        num_mazes,
        ms,
        walls_bytes,
        distance_bytes
    );

    if (!summary_ok) 
    {
        printf("Warning: failed to save benchmark summary.\n");
    }
    else 
    {
        printf("Benchmark summary saved to: %s\n", summary_file);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(device_walls));
    CUDA_CHECK(cudaFree(device_distance));

    free(host_walls);
    free(host_distance);

    return (save_ok && summary_ok) ? 0 : 1;
}