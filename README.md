# VectorCodec
### About
VectorCodec is a lossless compression algorithm for arrays of single-precision floating point values with a focus on speed. It is heavily based on FPC, another fast float compressor.  
The current implementation is an STB-style single header library and it uses AVX2 intrinsics. Support for other architectures will be added in the future.
### API
```cpp
namespace VectorCodec
{
	size_t UpperBound(size_t value_count);
	size_t Encode(const float* values, size_t value_count, uint8_t* out);
	void   Decode(const uint8_t* compressed, size_t value_count, float* out);
	size_t EncodeQuick(const float* values, size_t value_count, uint8_t* out);
	void   DecodeQuick(const uint8_t* compressed, size_t value_count, float* out);
}
```
### Example Code
```cpp
#include <cstdlib>
#include <vector>
#define VECTOR_CODEC_IMPLEMENTATION
#include <VectorCodec.hpp>

// Generate float array:
auto GenerateExampleArray(size_t count)
{
    std::vector<float> r(count);
    for (auto& e : r)
        e = (float)rand();
    return r;
}

int main()
{
    unsigned n;
    scanf("%u", &n);
    const auto values = GenerateExampleArray(n);

    // Compression:
    std::vector<uint8_t> compressed(VectorCodec::UpperBound(n));
    size_t k = VectorCodec::Encode(values.data(), values.size(), compressed.data());

    // Decompression:
    std::vector<float> decompressed(values.size());
    VectorCodec::Decode(compressed.data(), values.size(), decompressed.data());

    return 0;
}
```
### References
##### FPC Paper:
https://www.researchgate.net/publication/224323445_FPC_A_High-Speed_Compressor_for_Double-Precision_Floating-Point_Data