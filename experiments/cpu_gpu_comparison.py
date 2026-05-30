"""
cpu_gpu_comparison.py

Compare CPU Flood Fill distance output and GPU CUDA Flood Fill distance output.

This script compares:

    results/cpu_distances_10000.bin
    results/gpu_distances_10000.bin

Both files should have:

    shape = [10000, 256]
    dtype = int32

Run:

    python -m experiments.cpu_gpu_comparison
"""

import os
import numpy as np


def print_distance_map(distance_1d, title="Distance Map", size=16):
    """
    Print one 16x16 distance map.

    Args:
        distance_1d:
            1D distance array with shape [256].
        title:
            Title printed before the distance map.
        size:
            Maze size. Default is 16.
    """

    print()
    print(title)
    print("-" * len(title))

    for row in range(size):
        row_values = []

        for col in range(size):
            idx = row * size + col
            row_values.append(f"{int(distance_1d[idx]):3d}")

        print(" ".join(row_values))


def compare(
    num_mazes=10000,
    cpu_file="results/cpu_distances_10000.bin",
    gpu_file="results/gpu_distances_10000.bin",
    summary_file="results/cpu_gpu_comparison_summary_shared.txt",
    print_first_map=True,
    print_mismatch_limit=10,
):
    """
    Compare CPU and GPU Flood Fill distance outputs.

    Args:
        num_mazes:
            Number of mazes.
        cpu_file:
            CPU distance binary file.
        gpu_file:
            GPU distance binary file.
        summary_file:
            Output summary text file.
        print_first_map:
            If True, print first CPU/GPU distance map.
        print_mismatch_limit:
            Number of mismatch examples to print.
    """

    os.makedirs("results", exist_ok=True)

    print("CPU/GPU Distance Comparison")
    print("---------------------------")
    print(f"CPU file: {cpu_file}")
    print(f"GPU file: {gpu_file}")
    print(f"Number of mazes: {num_mazes}")
    print(f"Summary file: {summary_file}")
    print()

    # ------------------------------------------------------------
    # Check file existence
    # ------------------------------------------------------------
    if not os.path.exists(cpu_file):
        raise FileNotFoundError(
            f"CPU distance file not found: {cpu_file}\n"
            f"Please run CPU benchmark first:\n"
            f"    python -m experiments.benchmark_cpu_batch"
        )

    if not os.path.exists(gpu_file):
        raise FileNotFoundError(
            f"GPU distance file not found: {gpu_file}\n"
            f"Please run GPU benchmark first:\n"
            f"    cd build\n"
            f"    .\\cuda\\Release\\flood_fill_cuda.exe "
            f"10000 ..\\data\\mazes_10000_seed0.bin "
            f"..\\results\\gpu_distances_10000.bin "
            f"..\\results\\gpu_flood_fill_summary_shared.txt"
        )

    # ------------------------------------------------------------
    # Check file sizes
    # ------------------------------------------------------------
    expected_count = num_mazes * 256
    expected_bytes = expected_count * np.dtype(np.int32).itemsize

    cpu_bytes = os.path.getsize(cpu_file)
    gpu_bytes = os.path.getsize(gpu_file)

    print("File size check")
    print("---------------")
    print(f"Expected bytes: {expected_bytes}")
    print(f"CPU bytes:      {cpu_bytes}")
    print(f"GPU bytes:      {gpu_bytes}")
    print()

    if cpu_bytes != expected_bytes:
        raise ValueError(
            f"CPU file size mismatch.\n"
            f"Expected bytes: {expected_bytes}\n"
            f"Actual bytes: {cpu_bytes}\n"
            f"File: {cpu_file}"
        )

    if gpu_bytes != expected_bytes:
        raise ValueError(
            f"GPU file size mismatch.\n"
            f"Expected bytes: {expected_bytes}\n"
            f"Actual bytes: {gpu_bytes}\n"
            f"File: {gpu_file}"
        )

    # ------------------------------------------------------------
    # Load binary files
    # ------------------------------------------------------------
    cpu = np.fromfile(cpu_file, dtype=np.int32)
    gpu = np.fromfile(gpu_file, dtype=np.int32)

    if cpu.size != expected_count:
        raise ValueError(
            f"CPU array size mismatch.\n"
            f"Expected count: {expected_count}\n"
            f"Actual count: {cpu.size}"
        )

    if gpu.size != expected_count:
        raise ValueError(
            f"GPU array size mismatch.\n"
            f"Expected count: {expected_count}\n"
            f"Actual count: {gpu.size}"
        )

    cpu = cpu.reshape(num_mazes, 256)
    gpu = gpu.reshape(num_mazes, 256)

    # ------------------------------------------------------------
    # Compare CPU and GPU results
    # ------------------------------------------------------------
    same = np.array_equal(cpu, gpu)

    if same:
        max_difference = 0
        mismatch_count = 0
        mismatch_examples = []
    else:
        diff = np.abs(cpu - gpu)
        max_difference = int(diff.max())
        mismatch_count = int(np.count_nonzero(cpu != gpu))

        mismatch_indices = np.argwhere(cpu != gpu)

        mismatch_examples = []

        for item in mismatch_indices[:print_mismatch_limit]:
            id_maze = int(item[0])
            id_cell = int(item[1])

            row = id_cell // 16
            col = id_cell % 16

            cpu_value = int(cpu[id_maze, id_cell])
            gpu_value = int(gpu[id_maze, id_cell])

            mismatch_examples.append(
                {
                    "id_maze": id_maze,
                    "id_cell": id_cell,
                    "row": row,
                    "col": col,
                    "cpu_value": cpu_value,
                    "gpu_value": gpu_value,
                    "difference": abs(cpu_value - gpu_value),
                }
            )

    print("Comparison Result")
    print("-----------------")
    print(f"Shape: {cpu.shape}")
    print(f"Dtype: {cpu.dtype}")
    print(f"All equal: {same}")
    print(f"Mismatch count: {mismatch_count}")
    print(f"Max difference: {max_difference}")

    if not same:
        print()
        print(f"First {len(mismatch_examples)} mismatches:")

        for example in mismatch_examples:
            print(
                f"maze={example['id_maze']}, "
                f"cell={example['id_cell']}, "
                f"row={example['row']}, "
                f"col={example['col']}, "
                f"CPU={example['cpu_value']}, "
                f"GPU={example['gpu_value']}, "
                f"diff={example['difference']}"
            )
    else:
        print("CPU and GPU results are identical.")

    # ------------------------------------------------------------
    # Optional: print first maze map
    # ------------------------------------------------------------
    if print_first_map:
        print_distance_map(
            cpu[0],
            title="First maze CPU distance map",
            size=16,
        )

        print_distance_map(
            gpu[0],
            title="First maze GPU distance map",
            size=16,
        )

    # ------------------------------------------------------------
    # Save summary
    # ------------------------------------------------------------
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("CPU/GPU Distance Comparison Result\n")
        f.write("----------------------------------\n")
        f.write(f"cpu_file={cpu_file}\n")
        f.write(f"gpu_file={gpu_file}\n")
        f.write(f"num_mazes={num_mazes}\n")
        f.write("maze_size=16\n")
        f.write("num_cells=256\n")
        f.write(f"shape=({num_mazes}, 256)\n")
        f.write("dtype=int32\n")
        f.write(f"expected_bytes={expected_bytes}\n")
        f.write(f"cpu_bytes={cpu_bytes}\n")
        f.write(f"gpu_bytes={gpu_bytes}\n")
        f.write(f"all_equal={same}\n")
        f.write(f"mismatch_count={mismatch_count}\n")
        f.write(f"max_difference={max_difference}\n")

        if not same:
            f.write("\nFirst mismatches\n")
            f.write("----------------\n")

            for example in mismatch_examples:
                f.write(
                    f"maze={example['id_maze']}, "
                    f"cell={example['id_cell']}, "
                    f"row={example['row']}, "
                    f"col={example['col']}, "
                    f"cpu={example['cpu_value']}, "
                    f"gpu={example['gpu_value']}, "
                    f"diff={example['difference']}\n"
                )

    print()
    print(f"Summary saved to: {summary_file}")

    return same


def main():
    compare(
        num_mazes=10000,
        cpu_file="results/cpu_distances_10000.bin",
        gpu_file="results/gpu_distances_10000.bin",
        summary_file="results/cpu_gpu_comparison_summary_shared.txt",
        print_first_map=True,
        print_mismatch_limit=10,
    )


if __name__ == "__main__":
    main()