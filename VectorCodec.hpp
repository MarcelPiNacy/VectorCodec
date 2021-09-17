/*
	Copyright (c) 2021 Marcel Pi Nacy
	
	Permission is hereby granted, free of charge, to any person obtaining a copy
	of this software and associated documentation files (the "Software"), to deal
	in the Software without restriction, including without limitation the rights
	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
	copies of the Software, and to permit persons to whom the Software is
	furnished to do so, subject to the following conditions:
	
	The above copyright notice and this permission notice shall be included in all
	copies or substantial portions of the Software.
	
	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
	SOFTWARE.
*/

#ifndef VECTOR_CODEC_INCLUDED
#define VECTOR_CODEC_INCLUDED
#include <cstddef>
#include <cstdint>

#if defined(__clang__) || defined(__GNUC__)
#define VECTOR_CODEC_RESTRICT __restrict__
#else
#define VECTOR_CODEC_RESTRICT __restrict
#endif

namespace VectorCodec
{
	// Returns the size in bytes of a compressed float array in the worst case.
	constexpr size_t UpperBound(size_t value_count) noexcept
	{
		return (value_count + 1) / 2 + value_count * 4;
	}

	// Compresses a float array, storing the result in "out".
	size_t	Encode(const float* VECTOR_CODEC_RESTRICT values, size_t value_count, uint8_t* VECTOR_CODEC_RESTRICT out) noexcept;
	
	// Decompresses a float array, storing the result in "out".
	void	Decode(const uint8_t* VECTOR_CODEC_RESTRICT begin, size_t value_count, float* VECTOR_CODEC_RESTRICT out) noexcept;
}
#endif



#ifdef VECTOR_CODEC_IMPLEMENTATION
#include <cstring>

#if defined(_DEBUG) || !defined(NDEBUG)
#include <cassert>
#define VECTOR_CODEC_INVARIANT assert
#endif

#if defined(__clang__) || defined(__GNUC__)
#include <immintrin.h>
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define VECTOR_CODEC_BSWAP_IF_BE(VALUE) (uint32_t)__builtin_bswap32((VALUE))
#else
#define VECTOR_CODEC_BSWAP_IF_BE(VALUE) (VALUE)
#endif
#define VECTOR_CODEC_INLINE_ALWAYS __attribute__((always_inline))
#define VECTOR_CODEC_UNLIKELY_IF(CONDITION) if (__builtin_expect((CONDITION), 0))
#ifndef VECTOR_CODEC_INVARIANT
#define VECTOR_CODEC_INVARIANT __builtin_assume
#define VECTOR_CODEC_CLZ __builtin_clz
#endif
#else
#include <intrin.h>
#include <Windows.h>
#if REG_DWORD == REG_DWORD_BIG_ENDIAN
#define VECTOR_CODEC_BSWAP_IF_BE(VALUE) (uint32_t)_byteswap_ulong((VALUE))
#else
#define VECTOR_CODEC_BSWAP_IF_BE(VALUE) (VALUE)
#endif
#define VECTOR_CODEC_INLINE_ALWAYS __forceinline
#define VECTOR_CODEC_UNLIKELY_IF(CONDITION) if ((CONDITION))
#ifndef VECTOR_CODEC_INVARIANT
#define VECTOR_CODEC_INVARIANT __assume
#define VECTOR_CODEC_CLZ __lzcnt
#endif
#endif

namespace VectorCodec
{
	constexpr uint32_t LOOKUP_SIZE = 256;
	constexpr uint8_t HASH_SHIFT = 24;

	size_t Encode(const float* VECTOR_CODEC_RESTRICT values, size_t value_count, uint8_t* VECTOR_CODEC_RESTRICT out) noexcept
	{
		const __m256i zero_vec = _mm256_setzero_si256();
		const __m256i one_vec = _mm256_set1_epi32(1);
		const __m256i two_vec = _mm256_set1_epi32(2);
		const __m256i three_vec = _mm256_set1_epi32(3);
		const __m256i four_vec = _mm256_set1_epi32(4);
		const __m256i eight_vec = _mm256_set1_epi32(8);
		const __m256i sixteen_vec = _mm256_set1_epi32(16);
		const __m256i modmask_vec = _mm256_set1_epi32(LOOKUP_SIZE - 1);
		const __m256i lzc_shift_vec = _mm256_set_epi32(14, 12, 10, 8, 6, 4, 2, 0);
		const __m256i tzc_shift_vec = _mm256_set_epi32(30, 28, 26, 24, 22, 20, 18, 16);
		const float* const end = values + value_count;
		const uint8_t* const out_begin = out;
		alignas(64) int32_t lookup[LOOKUP_SIZE] = {};
		__m256i a, b, l, t, i, prior, xprior;
		uint32_t* out_headers = (uint32_t*)out;
		out += (value_count + 1) / 2;
		prior = xprior = i = zero_vec;
		do
		{
			VECTOR_CODEC_UNLIKELY_IF(end - values < 8)
				(void)memcpy(&a, values, (end - values) << 2);
			else
				a = _mm256_loadu_si256((const __m256i*)values);
			b = a;
			a = _mm256_sub_epi32(a, prior);
			prior = b;
			lookup[_mm256_extract_epi32(i, 0)] = _mm256_extract_epi32(a, 0);
			lookup[_mm256_extract_epi32(i, 1)] = _mm256_extract_epi32(a, 1);
			lookup[_mm256_extract_epi32(i, 2)] = _mm256_extract_epi32(a, 2);
			lookup[_mm256_extract_epi32(i, 3)] = _mm256_extract_epi32(a, 3);
			lookup[_mm256_extract_epi32(i, 4)] = _mm256_extract_epi32(a, 4);
			lookup[_mm256_extract_epi32(i, 5)] = _mm256_extract_epi32(a, 5);
			lookup[_mm256_extract_epi32(i, 6)] = _mm256_extract_epi32(a, 6);
			lookup[_mm256_extract_epi32(i, 7)] = _mm256_extract_epi32(a, 7);
			i = _mm256_and_si256(_mm256_xor_si256(a, _mm256_srli_epi32(a, HASH_SHIFT)), modmask_vec);
			a = _mm256_xor_si256(a, xprior);
			xprior = _mm256_i32gather_epi32(lookup, i, 4);
			b = _mm256_andnot_si256(_mm256_sub_epi32(a, one_vec), a);
			t = _mm256_set1_epi32(32);
			t = _mm256_sub_epi32(t, _mm256_andnot_si256(_mm256_cmpeq_epi32(b, zero_vec), one_vec));
			t = _mm256_sub_epi32(t, _mm256_andnot_si256(_mm256_cmpeq_epi32(_mm256_and_si256(b, _mm256_set1_epi32(0x0000ffff)), zero_vec), sixteen_vec));
			t = _mm256_sub_epi32(t, _mm256_andnot_si256(_mm256_cmpeq_epi32(_mm256_and_si256(b, _mm256_set1_epi32(0x00ff00ff)), zero_vec), eight_vec));
			t = _mm256_sub_epi32(t, _mm256_andnot_si256(_mm256_cmpeq_epi32(_mm256_and_si256(b, _mm256_set1_epi32(0x0f0f0f0f)), zero_vec), four_vec));
			t = _mm256_sub_epi32(t, _mm256_andnot_si256(_mm256_cmpeq_epi32(_mm256_and_si256(b, _mm256_set1_epi32(0x33333333)), zero_vec), two_vec));
			t = _mm256_sub_epi32(t, _mm256_andnot_si256(_mm256_cmpeq_epi32(_mm256_and_si256(b, _mm256_set1_epi32(0x55555555)), zero_vec), one_vec));
			t = _mm256_srli_epi32(t, 3);
			t = _mm256_sub_epi32(t, _mm256_srli_epi32(t, 2));
			a = _mm256_srlv_epi32(a, _mm256_slli_epi32(t, 3));
			l = _mm256_srli_epi32(_mm256_set_epi32(
				VECTOR_CODEC_CLZ(_mm256_extract_epi32(a, 7)), VECTOR_CODEC_CLZ(_mm256_extract_epi32(a, 6)),
				VECTOR_CODEC_CLZ(_mm256_extract_epi32(a, 5)), VECTOR_CODEC_CLZ(_mm256_extract_epi32(a, 4)),
				VECTOR_CODEC_CLZ(_mm256_extract_epi32(a, 3)), VECTOR_CODEC_CLZ(_mm256_extract_epi32(a, 2)),
				VECTOR_CODEC_CLZ(_mm256_extract_epi32(a, 1)), VECTOR_CODEC_CLZ(_mm256_extract_epi32(a, 0))), 3);
			b = _mm256_sub_epi32(four_vec, _mm256_sub_epi32(l, _mm256_and_si256(one_vec, _mm256_cmpeq_epi32(l, three_vec))));
			l = _mm256_sub_epi32(l, _mm256_and_si256(one_vec, _mm256_cmpgt_epi32(l, two_vec)));
			*(uint32_t*)out = VECTOR_CODEC_BSWAP_IF_BE(_mm256_extract_epi32(a, 0)); out += _mm256_extract_epi32(b, 0);
			*(uint32_t*)out = VECTOR_CODEC_BSWAP_IF_BE(_mm256_extract_epi32(a, 1)); out += _mm256_extract_epi32(b, 1);
			*(uint32_t*)out = VECTOR_CODEC_BSWAP_IF_BE(_mm256_extract_epi32(a, 2)); out += _mm256_extract_epi32(b, 2);
			*(uint32_t*)out = VECTOR_CODEC_BSWAP_IF_BE(_mm256_extract_epi32(a, 3)); out += _mm256_extract_epi32(b, 3);
			*(uint32_t*)out = VECTOR_CODEC_BSWAP_IF_BE(_mm256_extract_epi32(a, 4)); out += _mm256_extract_epi32(b, 4);
			*(uint32_t*)out = VECTOR_CODEC_BSWAP_IF_BE(_mm256_extract_epi32(a, 5)); out += _mm256_extract_epi32(b, 5);
			*(uint32_t*)out = VECTOR_CODEC_BSWAP_IF_BE(_mm256_extract_epi32(a, 6)); out += _mm256_extract_epi32(b, 6);
			*(uint32_t*)out = VECTOR_CODEC_BSWAP_IF_BE(_mm256_extract_epi32(a, 7)); out += _mm256_extract_epi32(b, 7);
			l = _mm256_sllv_epi32(l, lzc_shift_vec);
			t = _mm256_sllv_epi32(t, tzc_shift_vec);
			l = _mm256_or_si256(l, t);
			l = _mm256_or_si256(l, _mm256_srli_si256(l, 8));
			l = _mm256_or_si256(l, _mm256_srli_epi64(l, 32));
			*out_headers = VECTOR_CODEC_BSWAP_IF_BE((uint32_t)_mm256_extract_epi32(l, 0) | (uint32_t)_mm256_extract_epi32(l, 4));
			++out_headers;
			values += 8;
		} while (values < end);
		_mm256_zeroall();
		VECTOR_CODEC_INVARIANT(out >= out_begin);
		return out - out_begin;
	}

	void Decode(const uint8_t* VECTOR_CODEC_RESTRICT data, size_t value_count, float* VECTOR_CODEC_RESTRICT out) noexcept
	{
		const __m128i one_vec16 = _mm_set1_epi16(1);
		const __m128i three_vec16 = _mm_set1_epi16(3);
		const __m128i four_vec16 = _mm_set1_epi16(4);

		const __m256i one_vec32 = _mm256_set1_epi32(1);
		const __m256i three_vec32 = _mm256_set1_epi32(3);
		const __m256i modmask_vec32 = _mm256_set1_epi32(LOOKUP_SIZE - 1);
		const __m256i tzc_shift_vec32 = _mm256_set_epi32(30, 28, 26, 24, 22, 20, 18, 16);
		const uint32_t* in_headers = (const uint32_t*)data;
		data += (value_count + 1) / 2;
		alignas(32) int32_t lookup[LOOKUP_SIZE] = {};
		__m256i a, b, i, xprior, prior;
		__m128i l, l2;
		xprior = prior = i = _mm256_setzero_si256();
		while (true)
		{
			uint32_t header = VECTOR_CODEC_BSWAP_IF_BE(*in_headers);
			++in_headers;
			l = _mm_and_si128(_mm_set_epi16(header >> 14, header >> 12, header >> 10, header >> 8, header >> 6, header >> 4, header >> 2, header), three_vec16);
			l2 = l = _mm_sub_epi16(four_vec16, _mm_add_epi16(l, _mm_srli_epi16(_mm_add_epi16(l, one_vec16), 2)));
			l2 = _mm_add_epi16(_mm_slli_si128(l2, 2), l2);
			l2 = _mm_add_epi16(_mm_slli_si128(l2, 4), l2);
			l2 = _mm_add_epi16(_mm_slli_si128(l2, 8), l2);
			a = _mm256_i32gather_epi32((const int*)data, _mm256_cvtepi16_epi32(_mm_slli_si128(l2, 2)), 1);
			data += _mm_extract_epi16(l2, 7);
			a = _mm256_and_si256(a, _mm256_sub_epi32(_mm256_sllv_epi32(one_vec32, _mm256_slli_epi32(_mm256_cvtepi16_epi32(l), 3)), one_vec32));
			b = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(header), tzc_shift_vec32), three_vec32);
			a = _mm256_xor_si256(_mm256_sllv_epi32(a, _mm256_slli_epi32(b, 3)), xprior);
			lookup[_mm256_extract_epi32(i, 0)] = _mm256_extract_epi32(a, 0);
			lookup[_mm256_extract_epi32(i, 1)] = _mm256_extract_epi32(a, 1);
			lookup[_mm256_extract_epi32(i, 2)] = _mm256_extract_epi32(a, 2);
			lookup[_mm256_extract_epi32(i, 3)] = _mm256_extract_epi32(a, 3);
			lookup[_mm256_extract_epi32(i, 4)] = _mm256_extract_epi32(a, 4);
			lookup[_mm256_extract_epi32(i, 5)] = _mm256_extract_epi32(a, 5);
			lookup[_mm256_extract_epi32(i, 6)] = _mm256_extract_epi32(a, 6);
			lookup[_mm256_extract_epi32(i, 7)] = _mm256_extract_epi32(a, 7);
			i = _mm256_and_si256(_mm256_xor_si256(a, _mm256_srli_epi32(a, HASH_SHIFT)), modmask_vec32);
			xprior = _mm256_i32gather_epi32(lookup, i, 4);
			a = _mm256_add_epi32(a, prior);
			VECTOR_CODEC_UNLIKELY_IF(value_count < 8)
				break;
			_mm256_storeu_si256((__m256i*)out, a);
			prior = a;
			value_count -= 8;
			out += 8;
		}
		VECTOR_CODEC_UNLIKELY_IF(value_count != 0)
			(void)memcpy(out, &a, value_count << 2);
		_mm256_zeroall();
	}
}
#undef VECTOR_CODEC_BSWAP_IF_BE
#undef VECTOR_CODEC_INLINE_ALWAYS
#undef VECTOR_CODEC_CLZ_U32
#undef VECTOR_CODEC_UNLIKELY_IF
#undef VECTOR_CODEC_INVARIANT
#endif
#undef VECTOR_CODEC_RESTRICT