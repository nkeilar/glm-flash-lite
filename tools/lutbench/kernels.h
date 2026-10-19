// Standalone Broadwell expert codecs. See kernels.cpp for MIT provenance.
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include <cstdlib>
#include <limits>
#include <new>
#include <type_traits>
namespace lutbench {
template<class T, size_t Alignment = 64> struct AlignedAllocator {
    using value_type = T;
    using is_always_equal = std::true_type;
    template<class U> struct rebind { using other = AlignedAllocator<U, Alignment>; };
    AlignedAllocator() noexcept = default;
    template<class U> AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}
    T *allocate(size_t n) {
        if(n > (std::numeric_limits<size_t>::max()-Alignment)/sizeof(T)) throw std::bad_alloc();
        const size_t bytes=(n*sizeof(T)+Alignment-1)&~(Alignment-1);
        void *p=std::aligned_alloc(Alignment,bytes ? bytes : Alignment);
        if(!p) throw std::bad_alloc();
        return static_cast<T *>(p);
    }
    void deallocate(T *p,size_t) noexcept { std::free(p); }
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
