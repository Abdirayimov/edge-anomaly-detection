#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace eanom::utils {

inline void cuda_check(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error at ") + file + ":" +
                                 std::to_string(line) + " - " + cudaGetErrorString(err));
    }
}

class CudaStream {
public:
    CudaStream() { cuda_check(cudaStreamCreate(&stream_), __FILE__, __LINE__); }
    ~CudaStream() {
        if (stream_ != nullptr) cudaStreamDestroy(stream_);
    }
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    CudaStream(CudaStream&& o) noexcept : stream_(o.stream_) { o.stream_ = nullptr; }
    cudaStream_t get() const noexcept { return stream_; }
    void synchronize() const { cuda_check(cudaStreamSynchronize(stream_), __FILE__, __LINE__); }

private:
    cudaStream_t stream_ = nullptr;
};

}  // namespace eanom::utils

#define EANOM_CUDA_CHECK(call) ::eanom::utils::cuda_check((call), __FILE__, __LINE__)
