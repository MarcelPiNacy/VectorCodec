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

#ifndef VECTOR_CODEC_RESTRICT
#if defined(__clang__) || defined(__GNUC__)
#define VECTOR_CODEC_RESTRICT __restrict__
#else
#define VECTOR_CODEC_RESTRICT __restrict
#endif
#endif

#ifndef VECTOR_CODEC_CALL
#define VECTOR_CODEC_CALL
#endif

namespace VectorCodec
{
	/** @brief Returns the size of a compressed array in the worst case.
	* @param value_count The number of floats to compress.
	* @return The maximum size of the compressed data, in bytes.
	*/
	constexpr size_t VECTOR_CODEC_CALL UpperBound(size_t value_count) noexcept
	{
		size_t h = (value_count + 1) / 2;
		size_t d = (value_count + 31) & ~31;
		return h + d;
	}

	/** @brief Compresses an array of floats.
	* @param values A pointer to the array.
	* @param value_count The number of floats to compress.
	* @param out A pointer to a buffer where the compressed array will be stored. The size of this buffer must be set to UpperBound(value_count).
	* @return The number of bytes stored in out.
	* @note This function does NOT perform bounds checking on out.
	*/
	size_t	VECTOR_CODEC_CALL Encode(const float* VECTOR_CODEC_RESTRICT values, size_t value_count, uint8_t* VECTOR_CODEC_RESTRICT out) noexcept;

	/** @brief Decompresses an array of floats.
	* @param compressed A pointer to the compressed data.
	* @param value_count The number of floats to decompress.
	* @param out A pointer to an array where the decompressed values will be stored.
	* @note This function does NOT perform bounds checking on out, be careful to properly size it in relation to value_count.
	*/
	void	VECTOR_CODEC_CALL Decode(const uint8_t* VECTOR_CODEC_RESTRICT compressed, size_t value_count, float* VECTOR_CODEC_RESTRICT out) noexcept;

	/** @brief Compresses an array of floats without using a lookup table.
	* @param values A pointer to the array.
	* @param value_count The number of floats to compress.
	* @param out A pointer to a buffer where the compressed array will be stored. The size of this buffer must be set to UpperBound(value_count).
	* @return The number of bytes stored in out.
	* @note This function does NOT perform bounds checking on out.
	* @note The regular and Quick versions of VectorCodec are not compatible with each other: If you compressed the data using Encode, you must use Decode to get it back.
	*/
	size_t	VECTOR_CODEC_CALL EncodeQuick(const float* VECTOR_CODEC_RESTRICT values, size_t value_count, uint8_t* VECTOR_CODEC_RESTRICT out) noexcept;

	/** @brief Decompresses an array of floats without using a lookup table.
	* @param compressed A pointer to the compressed data.
	* @param value_count The number of floats to decompress.
	* @param out A pointer to an array where the decompressed values will be stored.
	* @note This function does NOT perform bounds checking on out, be careful to properly size it in relation to value_count.
	* @note The regular and Quick versions of VectorCodec are not compatible with each other: If you compressed the data using EncodeQuick, you must use DecodeQuick to get it back.
	*/
	void	VECTOR_CODEC_CALL DecodeQuick(const uint8_t* VECTOR_CODEC_RESTRICT compressed, size_t value_count, float* VECTOR_CODEC_RESTRICT out) noexcept;
}
#endif



#if defined(VECTOR_CODEC_IMPLEMENTATION) || defined(VECTOR_CODEC_INLINE)
#include <cstring>

#if defined(_DEBUG) || !defined(NDEBUG)
#include <cassert>
#define VECTOR_CODEC_INVARIANT assert
#define VECTOR_CODEC_INLINE_ALWAYS
#endif

#if defined(__clang__) || defined(__GNUC__)
#include <immintrin.h>
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define VECTOR_CODEC_BSWAP_IF_BE(VALUE) (uint32_t)__builtin_bswap32((VALUE))
#else
#define VECTOR_CODEC_BSWAP_IF_BE(VALUE) (VALUE)
#endif
#ifndef VECTOR_CODEC_INLINE_ALWAYS
#define VECTOR_CODEC_INLINE_ALWAYS __attribute__((flatten))
#endif
#define VECTOR_CODEC_UNLIKELY_IF(CONDITION) if (__builtin_expect((CONDITION), 0))
#ifndef VECTOR_CODEC_INVARIANT
#define VECTOR_CODEC_INVARIANT __builtin_assume
#endif
#define VECTOR_CODEC_CLZ __builtin_clz
#else
#include <intrin.h>
#include <Windows.h>
#if REG_DWORD == REG_DWORD_BIG_ENDIAN
#define VECTOR_CODEC_BSWAP_IF_BE(VALUE) (uint32_t)_byteswap_ulong((VALUE))
#else
#define VECTOR_CODEC_BSWAP_IF_BE(VALUE) (VALUE)
#endif
#ifndef VECTOR_CODEC_INLINE_ALWAYS
#define VECTOR_CODEC_INLINE_ALWAYS __forceinline
#endif
#define VECTOR_CODEC_UNLIKELY_IF(CONDITION) if ((CONDITION))
#ifndef VECTOR_CODEC_INVARIANT
#define VECTOR_CODEC_INVARIANT __assume
#endif
#define VECTOR_CODEC_CLZ __lzcnt
#endif

namespace VectorCodec
{
	namespace Impl
	{
		namespace Params
		{
			constexpr uint32_t LookupSize = 256;
			constexpr uint32_t ModMask = LookupSize - 1;
			constexpr uint8_t LSBDiscardedCount = 8;
			constexpr uint8_t HashShift = 16;
		}

		VECTOR_CODEC_INLINE_ALWAYS
		static size_t Encode_AVX2(const float* VECTOR_CODEC_RESTRICT values, size_t value_count, uint8_t* VECTOR_CODEC_RESTRICT out)
		{
			alignas(64) int32_t lookup[Params::LookupSize] = {};
			const float* const end = values + value_count;
			const uint8_t* const out_begin = out;
			__m256i indices, prior, xprior;
			uint32_t* out_headers = (uint32_t*)out;
			prior = xprior = indices = _mm256_setzero_si256();
			out += (value_count + 1) / 2;
			do
			{
				__m256i vec = _mm256_setzero_si256();
				VECTOR_CODEC_UNLIKELY_IF(end - values < 8)
					(void)memcpy(&vec, values, (end - values) << 2);
				else
					vec = _mm256_loadu_si256((const __m256i*)values);
				__m256i tmp = vec;
				vec = _mm256_sub_epi32(vec, prior);
				prior = tmp;
				lookup[_mm256_extract_epi32(indices, 0)] = _mm256_extract_epi32(vec, 0);
				lookup[_mm256_extract_epi32(indices, 1)] = _mm256_extract_epi32(vec, 1);
				lookup[_mm256_extract_epi32(indices, 2)] = _mm256_extract_epi32(vec, 2);
				lookup[_mm256_extract_epi32(indices, 3)] = _mm256_extract_epi32(vec, 3);
				lookup[_mm256_extract_epi32(indices, 4)] = _mm256_extract_epi32(vec, 4);
				lookup[_mm256_extract_epi32(indices, 5)] = _mm256_extract_epi32(vec, 5);
				lookup[_mm256_extract_epi32(indices, 6)] = _mm256_extract_epi32(vec, 6);
				lookup[_mm256_extract_epi32(indices, 7)] = _mm256_extract_epi32(vec, 7);
				tmp = _mm256_srli_epi32(vec, Params::LSBDiscardedCount);
				indices = _mm256_and_si256(_mm256_xor_si256(tmp, _mm256_srli_epi32(tmp, Params::HashShift)), _mm256_set1_epi32(Params::ModMask)); // We're going to hope values get hashed to good positions...
				vec = _mm256_xor_si256(vec, xprior);
				xprior = _mm256_i32gather_epi32(lookup, indices, 4);
				tmp = _mm256_andnot_si256(_mm256_sub_epi32(vec, _mm256_set1_epi32(1)), vec);
				__m256i tzcounts = _mm256_set1_epi32(32);
				tzcounts = _mm256_sub_epi32(tzcounts, _mm256_andnot_si256(_mm256_cmpeq_epi32(tmp, _mm256_setzero_si256()), _mm256_set1_epi32(1)));
				tzcounts = _mm256_sub_epi32(tzcounts, _mm256_andnot_si256(_mm256_cmpeq_epi32(_mm256_and_si256(tmp, _mm256_set1_epi32(0x0000ffff)), _mm256_setzero_si256()), _mm256_set1_epi32(16)));
				tzcounts = _mm256_sub_epi32(tzcounts, _mm256_andnot_si256(_mm256_cmpeq_epi32(_mm256_and_si256(tmp, _mm256_set1_epi32(0x00ff00ff)), _mm256_setzero_si256()), _mm256_set1_epi32(8)));
				tzcounts = _mm256_sub_epi32(tzcounts, _mm256_andnot_si256(_mm256_cmpeq_epi32(_mm256_and_si256(tmp, _mm256_set1_epi32(0x0f0f0f0f)), _mm256_setzero_si256()), _mm256_set1_epi32(4)));
				tzcounts = _mm256_sub_epi32(tzcounts, _mm256_andnot_si256(_mm256_cmpeq_epi32(_mm256_and_si256(tmp, _mm256_set1_epi32(0x33333333)), _mm256_setzero_si256()), _mm256_set1_epi32(2)));
				tzcounts = _mm256_sub_epi32(tzcounts, _mm256_andnot_si256(_mm256_cmpeq_epi32(_mm256_and_si256(tmp, _mm256_set1_epi32(0x55555555)), _mm256_setzero_si256()), _mm256_set1_epi32(1)));
				tzcounts = _mm256_srli_epi32(tzcounts, 3);
				tzcounts = _mm256_sub_epi32(tzcounts, _mm256_srli_epi32(tzcounts, 2));
				vec = _mm256_srlv_epi32(vec, _mm256_slli_epi32(tzcounts, 3));
				__m256i lzcounts = _mm256_srli_epi32(_mm256_set_epi32(
					VECTOR_CODEC_CLZ(_mm256_extract_epi32(vec, 7)), VECTOR_CODEC_CLZ(_mm256_extract_epi32(vec, 6)),
					VECTOR_CODEC_CLZ(_mm256_extract_epi32(vec, 5)), VECTOR_CODEC_CLZ(_mm256_extract_epi32(vec, 4)),
					VECTOR_CODEC_CLZ(_mm256_extract_epi32(vec, 3)), VECTOR_CODEC_CLZ(_mm256_extract_epi32(vec, 2)),
					VECTOR_CODEC_CLZ(_mm256_extract_epi32(vec, 1)), VECTOR_CODEC_CLZ(_mm256_extract_epi32(vec, 0))), 3);
				tmp = _mm256_sub_epi32(_mm256_set1_epi32(4), _mm256_sub_epi32(lzcounts, _mm256_and_si256(_mm256_set1_epi32(1), _mm256_cmpeq_epi32(lzcounts, _mm256_set1_epi32(3)))));
				lzcounts = _mm256_sub_epi32(lzcounts, _mm256_and_si256(_mm256_set1_epi32(1), _mm256_cmpgt_epi32(lzcounts, _mm256_set1_epi32(2))));
				*(uint32_t*)out = VECTOR_CODEC_BSWAP_IF_BE(_mm256_extract_epi32(vec, 0)); out += _mm256_extract_epi32(tmp, 0);
				*(uint32_t*)out = VECTOR_CODEC_BSWAP_IF_BE(_mm256_extract_epi32(vec, 1)); out += _mm256_extract_epi32(tmp, 1);
				*(uint32_t*)out = VECTOR_CODEC_BSWAP_IF_BE(_mm256_extract_epi32(vec, 2)); out += _mm256_extract_epi32(tmp, 2);
				*(uint32_t*)out = VECTOR_CODEC_BSWAP_IF_BE(_mm256_extract_epi32(vec, 3)); out += _mm256_extract_epi32(tmp, 3);
				*(uint32_t*)out = VECTOR_CODEC_BSWAP_IF_BE(_mm256_extract_epi32(vec, 4)); out += _mm256_extract_epi32(tmp, 4);
				*(uint32_t*)out = VECTOR_CODEC_BSWAP_IF_BE(_mm256_extract_epi32(vec, 5)); out += _mm256_extract_epi32(tmp, 5);
				*(uint32_t*)out = VECTOR_CODEC_BSWAP_IF_BE(_mm256_extract_epi32(vec, 6)); out += _mm256_extract_epi32(tmp, 6);
				*(uint32_t*)out = VECTOR_CODEC_BSWAP_IF_BE(_mm256_extract_epi32(vec, 7)); out += _mm256_extract_epi32(tmp, 7);
				lzcounts = _mm256_sllv_epi32(lzcounts, _mm256_set_epi32(14, 12, 10, 8, 6, 4, 2, 0));
				tzcounts = _mm256_sllv_epi32(tzcounts, _mm256_set_epi32(30, 28, 26, 24, 22, 20, 18, 16));
				lzcounts = _mm256_or_si256(lzcounts, tzcounts);
				lzcounts = _mm256_or_si256(lzcounts, _mm256_srli_si256(lzcounts, 8));
				lzcounts = _mm256_or_si256(lzcounts, _mm256_srli_epi64(lzcounts, 32));
				*out_headers = VECTOR_CODEC_BSWAP_IF_BE((uint32_t)_mm256_extract_epi32(lzcounts, 0) | (uint32_t)_mm256_extract_epi32(lzcounts, 4));
				++out_headers;
				values += 8;
			} while (values < end);
			_mm256_zeroall();
			VECTOR_CODEC_INVARIANT(out >= out_begin);
			return out - out_begin;
		}

		VECTOR_CODEC_INLINE_ALWAYS
		static void Decode_AVX2(const uint8_t* VECTOR_CODEC_RESTRICT data, size_t value_count, float* VECTOR_CODEC_RESTRICT out) noexcept
		{
			alignas(32) int32_t lookup[Params::LookupSize] = {};
			const uint32_t* in_headers = (const uint32_t*)data;
			__m256i indices, xprior, prior;
			xprior = prior = indices = _mm256_setzero_si256();
			data += (value_count + 1) / 2;
			while (true)
			{
				uint32_t header = VECTOR_CODEC_BSWAP_IF_BE(*in_headers);
				++in_headers;
				__m256i tmp = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(header), _mm256_set_epi32(14, 10, 6, 2, 12, 8, 4, 0)), _mm256_set1_epi32(3));
				__m128i lzcounts = _mm_or_si128(_mm256_extracti128_si256(tmp, 0), _mm256_extracti128_si256(_mm256_slli_epi32(tmp, 16), 1));
				__m128i prefix_sum = lzcounts = _mm_sub_epi16(_mm_set1_epi16(4), _mm_add_epi16(lzcounts, _mm_srli_epi16(_mm_add_epi16(lzcounts, _mm_set1_epi16(1)), 2)));
				prefix_sum = _mm_add_epi16(_mm_slli_si128(prefix_sum, 2), prefix_sum);
				prefix_sum = _mm_add_epi16(_mm_slli_si128(prefix_sum, 4), prefix_sum);
				prefix_sum = _mm_add_epi16(_mm_slli_si128(prefix_sum, 8), prefix_sum);
				__m256i vec = _mm256_i32gather_epi32((const int*)data, _mm256_cvtepi16_epi32(_mm_slli_si128(prefix_sum, 2)), 1);
				data += _mm_extract_epi16(prefix_sum, 7);
				vec = _mm256_and_si256(vec, _mm256_sub_epi32(_mm256_sllv_epi32(_mm256_set1_epi32(1), _mm256_slli_epi32(_mm256_cvtepi16_epi32(lzcounts), 3)), _mm256_set1_epi32(1)));
				tmp = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(header), _mm256_set_epi32(30, 28, 26, 24, 22, 20, 18, 16)), _mm256_set1_epi32(3));
				vec = _mm256_xor_si256(_mm256_sllv_epi32(vec, _mm256_slli_epi32(tmp, 3)), xprior);
				lookup[_mm256_extract_epi32(indices, 0)] = _mm256_extract_epi32(vec, 0);
				lookup[_mm256_extract_epi32(indices, 1)] = _mm256_extract_epi32(vec, 1);
				lookup[_mm256_extract_epi32(indices, 2)] = _mm256_extract_epi32(vec, 2);
				lookup[_mm256_extract_epi32(indices, 3)] = _mm256_extract_epi32(vec, 3);
				lookup[_mm256_extract_epi32(indices, 4)] = _mm256_extract_epi32(vec, 4);
				lookup[_mm256_extract_epi32(indices, 5)] = _mm256_extract_epi32(vec, 5);
				lookup[_mm256_extract_epi32(indices, 6)] = _mm256_extract_epi32(vec, 6);
				lookup[_mm256_extract_epi32(indices, 7)] = _mm256_extract_epi32(vec, 7);
				tmp = _mm256_srli_epi32(vec, Params::LSBDiscardedCount);
				indices = _mm256_and_si256(_mm256_xor_si256(tmp, _mm256_srli_epi32(tmp, Params::HashShift)), _mm256_set1_epi32(Params::ModMask));
				xprior = _mm256_i32gather_epi32(lookup, indices, 4);
				vec = _mm256_add_epi32(vec, prior);
				VECTOR_CODEC_UNLIKELY_IF(value_count < 8)
				{
					VECTOR_CODEC_UNLIKELY_IF(value_count != 0)
						(void)memcpy(out, &vec, value_count << 2);
					_mm256_zeroall();
					return;
				}
				_mm256_storeu_si256((__m256i*)out, vec);
				prior = vec;
				value_count -= 8;
				out += 8;
			}
		}

		VECTOR_CODEC_INLINE_ALWAYS
		static size_t EncodeQuick_AVX2(const float* VECTOR_CODEC_RESTRICT values, size_t value_count, uint8_t* VECTOR_CODEC_RESTRICT out)
		{
			const float* const end = values + value_count;
			const uint8_t* const out_begin = out;
			__m256i prior;
			uint32_t* out_headers = (uint32_t*)out;
			prior = _mm256_setzero_si256();
			out += (value_count + 1) / 2;
			do
			{
				__m256i vec = _mm256_setzero_si256();
				VECTOR_CODEC_UNLIKELY_IF(end - values < 8)
					(void)memcpy(&vec, values, (end - values) << 2);
				else
					vec = _mm256_loadu_si256((const __m256i*)values);
				__m256i tmp = vec;
				vec = _mm256_sub_epi32(vec, prior);
				prior = tmp;
				tmp = _mm256_andnot_si256(_mm256_sub_epi32(vec, _mm256_set1_epi32(1)), vec);
				__m256i tzcounts = _mm256_set1_epi32(32);
				tzcounts = _mm256_sub_epi32(tzcounts, _mm256_andnot_si256(_mm256_cmpeq_epi32(tmp, _mm256_setzero_si256()), _mm256_set1_epi32(1)));
				tzcounts = _mm256_sub_epi32(tzcounts, _mm256_andnot_si256(_mm256_cmpeq_epi32(_mm256_and_si256(tmp, _mm256_set1_epi32(0x0000ffff)), _mm256_setzero_si256()), _mm256_set1_epi32(16)));
				tzcounts = _mm256_sub_epi32(tzcounts, _mm256_andnot_si256(_mm256_cmpeq_epi32(_mm256_and_si256(tmp, _mm256_set1_epi32(0x00ff00ff)), _mm256_setzero_si256()), _mm256_set1_epi32(8)));
				tzcounts = _mm256_sub_epi32(tzcounts, _mm256_andnot_si256(_mm256_cmpeq_epi32(_mm256_and_si256(tmp, _mm256_set1_epi32(0x0f0f0f0f)), _mm256_setzero_si256()), _mm256_set1_epi32(4)));
				tzcounts = _mm256_sub_epi32(tzcounts, _mm256_andnot_si256(_mm256_cmpeq_epi32(_mm256_and_si256(tmp, _mm256_set1_epi32(0x33333333)), _mm256_setzero_si256()), _mm256_set1_epi32(2)));
				tzcounts = _mm256_sub_epi32(tzcounts, _mm256_andnot_si256(_mm256_cmpeq_epi32(_mm256_and_si256(tmp, _mm256_set1_epi32(0x55555555)), _mm256_setzero_si256()), _mm256_set1_epi32(1)));
				tzcounts = _mm256_srli_epi32(tzcounts, 3);
				tzcounts = _mm256_sub_epi32(tzcounts, _mm256_srli_epi32(tzcounts, 2));
				vec = _mm256_srlv_epi32(vec, _mm256_slli_epi32(tzcounts, 3));
				__m256i lzcounts = _mm256_srli_epi32(_mm256_set_epi32(
					VECTOR_CODEC_CLZ(_mm256_extract_epi32(vec, 7)), VECTOR_CODEC_CLZ(_mm256_extract_epi32(vec, 6)),
					VECTOR_CODEC_CLZ(_mm256_extract_epi32(vec, 5)), VECTOR_CODEC_CLZ(_mm256_extract_epi32(vec, 4)),
					VECTOR_CODEC_CLZ(_mm256_extract_epi32(vec, 3)), VECTOR_CODEC_CLZ(_mm256_extract_epi32(vec, 2)),
					VECTOR_CODEC_CLZ(_mm256_extract_epi32(vec, 1)), VECTOR_CODEC_CLZ(_mm256_extract_epi32(vec, 0))), 3);
				tmp = _mm256_sub_epi32(_mm256_set1_epi32(4), _mm256_sub_epi32(lzcounts, _mm256_and_si256(_mm256_set1_epi32(1), _mm256_cmpeq_epi32(lzcounts, _mm256_set1_epi32(3)))));
				lzcounts = _mm256_sub_epi32(lzcounts, _mm256_and_si256(_mm256_set1_epi32(1), _mm256_cmpgt_epi32(lzcounts, _mm256_set1_epi32(2))));
				*(uint32_t*)out = VECTOR_CODEC_BSWAP_IF_BE(_mm256_extract_epi32(vec, 0)); out += _mm256_extract_epi32(tmp, 0);
				*(uint32_t*)out = VECTOR_CODEC_BSWAP_IF_BE(_mm256_extract_epi32(vec, 1)); out += _mm256_extract_epi32(tmp, 1);
				*(uint32_t*)out = VECTOR_CODEC_BSWAP_IF_BE(_mm256_extract_epi32(vec, 2)); out += _mm256_extract_epi32(tmp, 2);
				*(uint32_t*)out = VECTOR_CODEC_BSWAP_IF_BE(_mm256_extract_epi32(vec, 3)); out += _mm256_extract_epi32(tmp, 3);
				*(uint32_t*)out = VECTOR_CODEC_BSWAP_IF_BE(_mm256_extract_epi32(vec, 4)); out += _mm256_extract_epi32(tmp, 4);
				*(uint32_t*)out = VECTOR_CODEC_BSWAP_IF_BE(_mm256_extract_epi32(vec, 5)); out += _mm256_extract_epi32(tmp, 5);
				*(uint32_t*)out = VECTOR_CODEC_BSWAP_IF_BE(_mm256_extract_epi32(vec, 6)); out += _mm256_extract_epi32(tmp, 6);
				*(uint32_t*)out = VECTOR_CODEC_BSWAP_IF_BE(_mm256_extract_epi32(vec, 7)); out += _mm256_extract_epi32(tmp, 7);
				lzcounts = _mm256_sllv_epi32(lzcounts, _mm256_set_epi32(14, 12, 10, 8, 6, 4, 2, 0));
				tzcounts = _mm256_sllv_epi32(tzcounts, _mm256_set_epi32(30, 28, 26, 24, 22, 20, 18, 16));
				lzcounts = _mm256_or_si256(lzcounts, tzcounts);
				lzcounts = _mm256_or_si256(lzcounts, _mm256_srli_si256(lzcounts, 8));
				lzcounts = _mm256_or_si256(lzcounts, _mm256_srli_epi64(lzcounts, 32));
				*out_headers = VECTOR_CODEC_BSWAP_IF_BE((uint32_t)_mm256_extract_epi32(lzcounts, 0) | (uint32_t)_mm256_extract_epi32(lzcounts, 4));
				++out_headers;
				values += 8;
			} while (values < end);
			_mm256_zeroall();
			VECTOR_CODEC_INVARIANT(out >= out_begin);
			return out - out_begin;
		}

		VECTOR_CODEC_INLINE_ALWAYS
		static void DecodeQuick_AVX2(const uint8_t* VECTOR_CODEC_RESTRICT data, size_t value_count, float* VECTOR_CODEC_RESTRICT out) noexcept
		{
			const uint32_t* in_headers = (const uint32_t*)data;
			__m256i prior = _mm256_setzero_si256();
			data += (value_count + 1) / 2;
			while (true)
			{
				uint32_t header = VECTOR_CODEC_BSWAP_IF_BE(*in_headers);
				++in_headers;
				__m256i tmp = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(header), _mm256_set_epi32(14, 10, 6, 2, 12, 8, 4, 0)), _mm256_set1_epi32(3));
				__m128i lzcounts = _mm_or_si128(_mm256_extracti128_si256(tmp, 0), _mm256_extracti128_si256(_mm256_slli_epi32(tmp, 16), 1));
				__m128i prefix_sum = lzcounts = _mm_sub_epi16(_mm_set1_epi16(4), _mm_add_epi16(lzcounts, _mm_srli_epi16(_mm_add_epi16(lzcounts, _mm_set1_epi16(1)), 2)));
				prefix_sum = _mm_add_epi16(_mm_slli_si128(prefix_sum, 2), prefix_sum);
				prefix_sum = _mm_add_epi16(_mm_slli_si128(prefix_sum, 4), prefix_sum);
				prefix_sum = _mm_add_epi16(_mm_slli_si128(prefix_sum, 8), prefix_sum);
				__m256i vec = _mm256_i32gather_epi32((const int*)data, _mm256_cvtepi16_epi32(_mm_slli_si128(prefix_sum, 2)), 1);
				data += _mm_extract_epi16(prefix_sum, 7);
				vec = _mm256_and_si256(vec, _mm256_sub_epi32(_mm256_sllv_epi32(_mm256_set1_epi32(1), _mm256_slli_epi32(_mm256_cvtepi16_epi32(lzcounts), 3)), _mm256_set1_epi32(1)));
				tmp = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(header), _mm256_set_epi32(30, 28, 26, 24, 22, 20, 18, 16)), _mm256_set1_epi32(3));
				vec = _mm256_sllv_epi32(vec, _mm256_slli_epi32(tmp, 3));
				vec = _mm256_add_epi32(vec, prior);
				VECTOR_CODEC_UNLIKELY_IF(value_count < 8)
				{
					VECTOR_CODEC_UNLIKELY_IF(value_count != 0)
						(void)memcpy(out, &vec, value_count << 2);
					_mm256_zeroall();
					return;
				}
				_mm256_storeu_si256((__m256i*)out, vec);
				prior = vec;
				value_count -= 8;
				out += 8;
			}
		}
	}

#ifdef VECTOR_CODEC_INLINE
	VECTOR_CODEC_INLINE_ALWAYS static
#endif
	size_t VECTOR_CODEC_CALL Encode(const float* VECTOR_CODEC_RESTRICT values, size_t value_count, uint8_t* VECTOR_CODEC_RESTRICT out) noexcept
	{
		return Impl::Encode_AVX2(values, value_count, out);
	}

#ifdef VECTOR_CODEC_INLINE
	VECTOR_CODEC_INLINE_ALWAYS static
#endif
	void VECTOR_CODEC_CALL Decode(const uint8_t* VECTOR_CODEC_RESTRICT compressed, size_t value_count, float* VECTOR_CODEC_RESTRICT out) noexcept
	{
		Impl::Decode_AVX2(compressed, value_count, out);
	}

#ifdef VECTOR_CODEC_INLINE
	VECTOR_CODEC_INLINE_ALWAYS static
#endif
	size_t VECTOR_CODEC_CALL EncodeQuick(const float* VECTOR_CODEC_RESTRICT values, size_t value_count, uint8_t* VECTOR_CODEC_RESTRICT out) noexcept
	{
		return Impl::EncodeQuick_AVX2(values, value_count, out);
	}

#ifdef VECTOR_CODEC_INLINE
	VECTOR_CODEC_INLINE_ALWAYS static
#endif
	void VECTOR_CODEC_CALL DecodeQuick(const uint8_t* VECTOR_CODEC_RESTRICT compressed, size_t value_count, float* VECTOR_CODEC_RESTRICT out) noexcept
	{
		Impl::DecodeQuick_AVX2(compressed, value_count, out);
	}
}
#undef VECTOR_CODEC_BSWAP_IF_BE
#undef VECTOR_CODEC_INLINE_ALWAYS
#undef VECTOR_CODEC_UNLIKELY_IF
#undef VECTOR_CODEC_INVARIANT
#undef VECTOR_CODEC_CLZ
#endif
#undef VECTOR_CODEC_RESTRICT