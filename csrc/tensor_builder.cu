#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// CUDA kernel to build padded tensors in parallel
__global__ void build_padded_tensors_kernel(
    const int64_t* props_flat,
    const int64_t* values_flat,
    const int* seq_lengths,
    const int* offsets,
    int64_t* properties_out,
    int64_t* values_out,
    bool* mask_out,
    int num_samples,
    int max_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_samples) {
        int offset = offsets[idx];
        int seq_len = seq_lengths[idx];

        // Copy properties, values, and set mask for this sample
        for (int i = 0; i < seq_len; i++) {
            properties_out[idx * max_len + i] = props_flat[offset + i];
            values_out[idx * max_len + i] = values_flat[offset + i];
            mask_out[idx * max_len + i] = true;
        }

        // Pad remaining with zeros/false
        for (int i = seq_len; i < max_len; i++) {
            properties_out[idx * max_len + i] = 0;
            values_out[idx * max_len + i] = 0;
            mask_out[idx * max_len + i] = false;
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> build_padded_tensors(
    const std::vector<std::vector<int64_t>>& props_lists,
    const std::vector<std::vector<int64_t>>& values_lists
) {
    int num_samples = props_lists.size();
    if (num_samples == 0) {
        return std::make_tuple(
            torch::zeros({0, 0}, torch::dtype(torch::kLong)),
            torch::zeros({0, 0}, torch::dtype(torch::kLong)),
            torch::zeros({0, 0}, torch::dtype(torch::kBool))
        );
    }

    // Compute max length and sequence lengths
    int max_len = 0;
    std::vector<int> seq_lengths(num_samples);
    for (int i = 0; i < num_samples; i++) {
        seq_lengths[i] = props_lists[i].size();
        max_len = std::max(max_len, seq_lengths[i]);
    }

    // Flatten props and values into contiguous arrays
    int total_elements = 0;
    for (const auto& props : props_lists) {
        total_elements += props.size();
    }

    std::vector<int64_t> props_flat(total_elements);
    std::vector<int64_t> values_flat(total_elements);
    std::vector<int> offsets(num_samples);

    int offset = 0;
    for (int i = 0; i < num_samples; i++) {
        offsets[i] = offset;
        std::copy(props_lists[i].begin(), props_lists[i].end(), props_flat.begin() + offset);
        std::copy(values_lists[i].begin(), values_lists[i].end(), values_flat.begin() + offset);
        offset += props_lists[i].size();
    }

    // Move to GPU
    auto options = torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA);
    auto props_flat_gpu = torch::from_blob(props_flat.data(), {total_elements}, torch::kLong).to(torch::kCUDA);
    auto values_flat_gpu = torch::from_blob(values_flat.data(), {total_elements}, torch::kLong).to(torch::kCUDA);
    auto seq_lengths_gpu = torch::from_blob(seq_lengths.data(), {num_samples}, torch::kInt).to(torch::kCUDA);
    auto offsets_gpu = torch::from_blob(offsets.data(), {num_samples}, torch::kInt).to(torch::kCUDA);

    // Allocate output tensors on GPU
    auto properties_out = torch::zeros({num_samples, max_len}, options);
    auto values_out = torch::zeros({num_samples, max_len}, options);
    auto mask_out = torch::zeros({num_samples, max_len}, torch::dtype(torch::kBool).device(torch::kCUDA));

    // Launch CUDA kernel
    int threads = 256;
    int blocks = (num_samples + threads - 1) / threads;

    build_padded_tensors_kernel<<<blocks, threads>>>(
        props_flat_gpu.data_ptr<int64_t>(),
        values_flat_gpu.data_ptr<int64_t>(),
        seq_lengths_gpu.data_ptr<int>(),
        offsets_gpu.data_ptr<int>(),
        properties_out.data_ptr<int64_t>(),
        values_out.data_ptr<int64_t>(),
        mask_out.data_ptr<bool>(),
        num_samples,
        max_len
    );

    // Synchronize to ensure kernel completion
    cudaDeviceSynchronize();

    return std::make_tuple(properties_out, values_out, mask_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("build_padded_tensors", &build_padded_tensors, "Build padded tensors on GPU (CUDA)");
}
