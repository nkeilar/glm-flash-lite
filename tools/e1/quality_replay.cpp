// Offline quality replay. Codec implementations are linked unchanged from the
// pinned lutbench revision; see ../lutbench/LICENSE and README.md.
#include "kernels.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lutbench;
namespace {
thread_local std::string last_error;
Format format(int id) {
    switch (id) {
        case 8: return Format::Q8_0;
        case 14: return Format::Q6_K;
        case 21: return Format::IQ3_S;
        case 23: return Format::IQ4_XS;
        case 138: return Format::IQ3_K;
        case 144: return Format::IQ4_KS;
        default: throw std::invalid_argument("unsupported GGML type");
    }
}
float act(float g, float u) {
    const float s = g / (1.f + std::exp(-g));
    return std::fmin(s, 10.f) * std::fmin(std::fmax(u, -10.f), 10.f);
}
void finite(const float *x, size_t n) {
    for (size_t i=0; i<n; ++i)
        if (!std::isfinite(x[i])) throw std::invalid_argument("nonfinite replay value");
}
void forward(const void *gate, const void *up, const void *down,
             Format gf, Format uf, Format df, int h, int in, const float *x,
             int m, Path path, bool tp2, bool reference, float *out) {
    if (!gate || !up || !down || !x || !out || h<=0 || in<=0 || h%256 || in%512 ||
        (m!=1 && m!=2 && m!=4)) throw std::invalid_argument("invalid replay dimensions/M/pointer");
    finite(x, size_t(m)*h);
    if (reference && (gf!=Format::Q8_0 || uf!=gf || df!=gf))
        throw std::invalid_argument("reference must be actual Q8_0 payload");
    std::fill(out, out+size_t(m)*h, 0.f);
    const auto *g=static_cast<const unsigned char *>(gate);
    const auto *u=static_cast<const unsigned char *>(up);
    const auto *d=static_cast<const unsigned char *>(down);
    const size_t gs=row_bytes(gf,h), us=row_bytes(uf,h), ds=row_bytes(df,in);
    if (reference) {
        // Float64 accumulation against dequantized Q8_0 weights, with float32
        // projection/nonlinearity outputs. No activation quantization in reference.
        std::vector<float> gr(h),ur(h),dr(in),hidden(size_t(m)*in);
        for (int r=0;r<in;++r) {
            dequantize(gf,g+r*gs,gr.data(),h); dequantize(uf,u+r*us,ur.data(),h);
            finite(gr.data(),h); finite(ur.data(),h);
            for (int t=0;t<m;++t) {
                double gv=0,uv=0;
                for(int j=0;j<h;++j) { gv+=double(gr[j])*x[t*h+j]; uv+=double(ur[j])*x[t*h+j]; }
                hidden[t*in+r]=act(float(gv),float(uv));
            }
        }
        finite(hidden.data(),hidden.size());
        for(int r=0;r<h;++r) {
            dequantize(df,d+r*ds,dr.data(),in); finite(dr.data(),in);
            for(int t=0;t<m;++t) {
                double v=0; for(int j=0;j<in;++j) v+=double(dr[j])*hidden[t*in+j];
                out[t*h+r]=float(v);
            }
        }
    } else {
        const int parts=tp2?2:1, width=in/parts;
        const size_t slice_bytes=row_bytes(df,width);
        std::vector<unsigned char> packed(slice_bytes);
        std::vector<float> hidden(size_t(m)*width),partial(size_t(m)*h);
        for(int node=0;node<parts;++node) {
            Activation input[4],mid[4];
            for(int t=0;t<m;++t) prepare(path,x+t*h,h,input[t]);
            for(int r=0;r<width;++r) {
                float gv[4],uv[4];
                dot_multi(gf,g+(node*width+r)*gs,input,h,m,gv);
                dot_multi(uf,u+(node*width+r)*us,input,h,m,uv);
                finite(gv,m); finite(uv,m);
                for(int t=0;t<m;++t) hidden[t*width+r]=act(gv[t],uv[t]);
            }
            finite(hidden.data(),hidden.size());
            for(int t=0;t<m;++t) prepare(path,hidden.data()+t*width,width,mid[t]);
            for(int r=0;r<h;++r) {
                // Real packed down-column split: retains IQ4_KS row-global scale.
                slice_packed_row(df,d+r*ds,in,node*width,width,packed.data());
                float values[4]; dot_multi(df,packed.data(),mid,width,m,values);
                finite(values,m);
                for(int t=0;t<m;++t) partial[t*h+r]=values[t];
            }
            for(size_t j=0;j<partial.size();++j) out[j]+=partial[j];
        }
    }
    finite(out,size_t(m)*h);
}
}
extern "C" {
const char *e1_error() { return last_error.c_str(); }
size_t e1_row_bytes(int f,int n) {
    try { last_error.clear(); return row_bytes(format(f),n); }
    catch(const std::exception &e) { last_error=e.what(); return 0; }
}
int e1_quantize(int f,const float *x,int n,void *out) {
    try { finite(x,n); quantize(format(f),x,out,n); return 0; }
    catch(const std::exception &e) { last_error=e.what(); return 1; }
}
int e1_forward(const void *g,const void *u,const void *d,int gf,int uf,int df,
               int h,int in,const float *x,int m,int q8,int tp2,int reference,float *out) {
    try {
        last_error.clear();
        forward(g,u,d,format(gf),format(uf),format(df),h,in,x,m,
                q8?Path::Q8:Path::I16,tp2!=0,reference!=0,out);
        return 0;
    } catch(const std::exception &e) { last_error=e.what(); return 1; }
}
}
