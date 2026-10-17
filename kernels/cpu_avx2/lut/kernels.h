// Standalone Broadwell expert codecs. See kernels.cpp for MIT provenance.
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include <cstdlib>
#include <limits>
#include <new>
#include <type_traits>
#include <array>
#include <linux/mempolicy.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
namespace lutbench {
// Serving adaptation: fresh mappings prevent libc reuse of resident pages from
// another NUMA worker. Codec arithmetic and Activation layout are unchanged.
inline size_t aligned_mapping_bytes(size_t bytes) {
    static const size_t page = [] {
        const long value = sysconf(_SC_PAGESIZE);
        if (value <= 0) throw std::bad_alloc();
        return size_t(value);
    }();
    if (bytes > std::numeric_limits<size_t>::max() - (page - 1)) throw std::bad_alloc();
    return bytes ? (bytes + page - 1) / page * page : page;
}
inline void preserve_preferred_mapping_policy(void* address, size_t bytes) {
    // Freeze the allocating worker's preferred policy onto the fresh VMA,
    // before vector initialization first-touches it. No migration or global
    // policy change. Unavailable policy syscalls remain visible to the loader's
    // placement gate; sandboxed offline tests may run without NUMA evidence.
    constexpr unsigned long maxnode = 1025;
    constexpr size_t bits = 8 * sizeof(unsigned long);
    std::array<unsigned long, (maxnode + bits - 1) / bits> mask{};
    int mode = -1;
    if (syscall(SYS_get_mempolicy, &mode, mask.data(), maxnode, nullptr, 0) == 0 && mode == MPOL_PREFERRED)
        (void)syscall(SYS_mbind, address, bytes, mode, mask.data(), maxnode, 0);
}
template<class T, size_t Alignment = 64> struct AlignedAllocator {
    using value_type = T;
    using is_always_equal = std::true_type;
    template<class U> struct rebind { using other = AlignedAllocator<U, Alignment>; };
    AlignedAllocator() noexcept = default;
    template<class U> AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}
    T *allocate(size_t n) {
        if (n > std::numeric_limits<size_t>::max() / sizeof(T)) throw std::bad_alloc();
        const size_t bytes = aligned_mapping_bytes(n * sizeof(T));
        void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) throw std::bad_alloc();
        if (reinterpret_cast<uintptr_t>(p) % Alignment || madvise(p, bytes, MADV_NOHUGEPAGE) != 0) {
            munmap(p, bytes); throw std::bad_alloc();
        }
        preserve_preferred_mapping_policy(p, bytes);
        return static_cast<T*>(p);
    }
    void deallocate(T* p, size_t n) noexcept {
        if (p) (void)munmap(p, aligned_mapping_bytes(n * sizeof(T)));
    }
};
template<class T,class U,size_t A> bool operator==(const AlignedAllocator<T,A>&,const AlignedAllocator<U,A>&) { return true; }
template<class T,class U,size_t A> bool operator!=(const AlignedAllocator<T,A>&,const AlignedAllocator<U,A>&) { return false; }
template<class T> using AlignedVector = std::vector<T, AlignedAllocator<T,64>>;
enum class Format { IQ3_K, IQ4_KS, IQ3_S, IQ4_XS, Q6_K, Q3_K, Q4_K, Q8_0 };
const char *format_name(Format format);
enum class Path { Q8, I16 };
enum class Kernel { Baseline, Optimized };
struct Activation {
    Path path = Path::Q8;
    Kernel kernel = Kernel::Optimized;
    int n = 0;
    AlignedVector<float> scales; // one scale per 128 inputs for I16
    AlignedVector<int16_t> i16;  // signed activations, |q| <= 16383
    AlignedVector<int16_t> i16_lo_bsums, i16_hi_bsums;
    AlignedVector<int8_t> i16_lo, i16_hi; // exact signed7-bit: i16=lo+128*hi
    AlignedVector<int16_t> i16_swizzled; // within each32: [0..7,16..23,8..15,24..31]
    AlignedVector<uint8_t> q8;   // exact native block_q8_K layout
};
size_t row_bytes(Format format, int n);
// Copy a contiguous256-weight-aligned range without requantizing.
// IQ4_KS duplicates the original row-scale bytes. Destination capacity must
// be at least row_bytes(format,count); source/destination must not overlap.
// Throws invalid_argument for invalid format, bounds, null pointers or aliasing.
void slice_packed_row(Format format, const void *src, int n,
                      int begin, int count, void *out);
void quantize(Format format, const float *x, void *out, int n);
void dequantize(Format format, const void *x, float *out, int n);
void prepare(Path path, const float *x, int n, Activation &out, Kernel kernel = Kernel::Optimized);
// Serial allocation followed by independent, block-aligned worker ranges.
void prepare_storage(Path path, int n, Activation &out, Kernel kernel = Kernel::Optimized);
void prepare_range(Path path, const float *x, int n, int begin, int end, Activation &out);
void activation_dequantize(const Activation &x, float *out, int n);
float dot(Format format, const void *row, const Activation &x, int n);
float dot(Format format, const void *row, const Activation &x, int n, Kernel kernel);
void dot_multi(Format format, const void *row, const Activation *inputs,
               int n, int m, float *outputs, Kernel kernel = Kernel::Optimized);
// Stable, externally visible hot functions for disassembly and static analysis.
float dot_iq3_q8(const void *row, const Activation &x, int n);
float dot_iq4_q8(const void *row, const Activation &x, int n);
float dot_iq3_i16(const void *row, const Activation &x, int n);
float dot_iq4_i16(const void *row, const Activation &x, int n);
}
