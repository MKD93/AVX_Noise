/**
 File: "Noise.cpp"

 Purpose: To generate Perlin Simplex Noise with eight iterations
 of Fractional Brownian Motion using AVX vectorized code

 Copyright (c) 2016 Michael K. Duncan (fruitless75@gmail.com)

 Distributed under the MIT License (MIT) (See accompanying file LICENSE
 or copy at http://opensource.org/licenses/MIT)
**/

#include "Noise.hpp"
#include "Random.hpp"

#include <algorithm>
#include <immintrin.h>

static __m256 frequencySet = _mm256_set_ps(1.0f, 2.0f, 4.0f, 8.0f, 16.0f, 32.0f, 64.0f, 128.0f);
static __m256 amplitudeSet = _mm256_rcp_ps(frequencySet);
static float amplitudeScale = 1.0f / (255.0f / 128.0f);

static float Height = 1.0f;
static float Frequency = 1.0f;
static float Amplitude = 1.0f;
static float Lacunarity = 2.0f;
static float Persistence = 0.5f;

static float Permutation[256] = { 0.0f };
static float Set[8] = { 0.0f };

void Noise::Seed()
{
    float fSet[8] = { 0.0f };
    float aSet[8] = { 0.0f };
    float frequency = Frequency;
    float amplitude = Amplitude;
    uint32_t index = 0;

    amplitudeScale = 0.0f;

    for(; index < 8; ++index)
    {
        fSet[index] = frequency;
        aSet[index] = amplitude;
        amplitudeScale += amplitude;

        frequency *= Lacunarity;
        amplitude *= Persistence;
    }

    frequencySet = _mm256_loadu_ps(&(fSet[0]));
    amplitudeSet = _mm256_loadu_ps(&(aSet[0]));
    amplitudeScale = 1.0f / amplitudeScale;

    for(index = 0; index < 256; ++index)
        Permutation[index] = static_cast<float>(index);

    for(index = 0; index < 1024; ++index)
        std::swap(Permutation[Random::getIndex()], Permutation[Random::getIndex()]);
}

const float Noise::getHeight() { return Height; }
const float Noise::getFrequency() { return Frequency; }
const float Noise::getAmplitude() { return Amplitude; }
const float Noise::getLacunarity() { return Lacunarity; }
const float Noise::getPersistence() { return Persistence; }

void Noise::setHeight(const float height) { Height = height; }
void Noise::setFrequency(const float frequency) { Frequency = frequency; }
void Noise::setAmplitude(const float amplitude) { Amplitude = Amplitude; }
void Noise::setLacunarity(const float lacunarity) { Lacunarity = lacunarity; }
void Noise::setPersistence(const float persistence) { Persistence = persistence; }

static const __m256 _mm256_fabs_ps(const __m256 &a) { return _mm256_andnot_ps(_mm256_set1_ps(-0.0f), a); }
static const __m256 _mm256_fmod_ps(const __m256 &a, const __m256 &b) { return _mm256_sub_ps(a, _mm256_mul_ps(b, _mm256_floor_ps(_mm256_div_ps(a, b)))); }
static void loadSet(const __m256 &a) { _mm256_storeu_ps(&(Set[0]), a); }
static const float getPerm(const float index) { return Permutation[static_cast<uint8_t>(index)]; }

static const __m256 getHash(const __m256 &index)
{
    loadSet(_mm256_fmod_ps(_mm256_fabs_ps(index), _mm256_set1_ps(256.0f)));

    return _mm256_set_ps
    (
        getPerm(Set[7]), getPerm(Set[6]), getPerm(Set[5]), getPerm(Set[4]),
        getPerm(Set[3]), getPerm(Set[2]), getPerm(Set[1]), getPerm(Set[0])
    );
}

static const __m256 getGradient(const __m256 &hash, const __m256 &x)
{
    const __m256 code = _mm256_fmod_ps(hash, _mm256_set1_ps(16.0f));
    const __m256 delta = _mm256_add_ps(_mm256_fmod_ps(code, _mm256_set1_ps(8.0f)), _mm256_set1_ps(1.0f));
    const __m256 mod = _mm256_fmod_ps(_mm256_mul_ps(code, _mm256_set1_ps(0.125f)), _mm256_set1_ps(2.0f));
    const __m256 mask = _mm256_cmp_ps(mod, _mm256_set1_ps(1.0f), _CMP_GE_OQ);
    const __m256 maskL = _mm256_andnot_ps(mask, _mm256_set1_ps(1.0f));
    const __m256 maskR = _mm256_and_ps(mask, _mm256_set1_ps(1.0f));

    return _mm256_mul_ps(x, _mm256_sub_ps(_mm256_mul_ps(maskL, delta), _mm256_mul_ps(maskR, delta)));
}

static const __m256 getGradient(const __m256 &hash, const __m256 &x, const __m256 &y)
{
    const __m256 code = _mm256_fmod_ps(hash, _mm256_set1_ps(64.0f));
    const __m256 maskUV = _mm256_cmp_ps(code, _mm256_set1_ps(4.0f), _CMP_LT_OQ);
    const __m256 mask0 = _mm256_and_ps(maskUV, _mm256_set1_ps(1.0f));
    const __m256 mask1 = _mm256_andnot_ps(maskUV, _mm256_set1_ps(1.0f));
    const __m256 u = _mm256_add_ps(_mm256_mul_ps(mask0, x), _mm256_mul_ps(mask1, y));
    const __m256 v = _mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(mask0, y), _mm256_mul_ps(mask1, x)), _mm256_set1_ps(2.0f));
    const __m256 maskL = _mm256_cmp_ps(_mm256_fmod_ps(code, _mm256_set1_ps(2.0f)), _mm256_setzero_ps(), _CMP_NEQ_OQ);
    const __m256 maskR = _mm256_cmp_ps(_mm256_fmod_ps(code, _mm256_set1_ps(4.0f)), _mm256_set1_ps(2.0f), _CMP_GE_OQ);
    const __m256 mask2 = _mm256_andnot_ps(maskL, _mm256_set1_ps(1.0f));
    const __m256 mask3 = _mm256_and_ps(maskL, _mm256_set1_ps(1.0f));
    const __m256 mask4 = _mm256_andnot_ps(maskR, _mm256_set1_ps(1.0f));
    const __m256 mask5 = _mm256_and_ps(maskR, _mm256_set1_ps(1.0f));
    const __m256 resultL = _mm256_sub_ps(_mm256_mul_ps(mask2, u), _mm256_mul_ps(mask3, u));
    const __m256 resultR = _mm256_sub_ps(_mm256_mul_ps(mask4, v), _mm256_mul_ps(mask5, v));

    return _mm256_add_ps(resultL, resultR);
}

const float Noise::getValue(const float xValue)
{
    const __m256 x = _mm256_mul_ps(frequencySet, _mm256_set1_ps(xValue));
    const __m256 i0 = _mm256_floor_ps(x);
    const __m256 i1 = _mm256_add_ps(i0, _mm256_set1_ps(1.0f));
    const __m256 x0 = _mm256_sub_ps(x, i0);
    const __m256 x1 = _mm256_sub_ps(x0, _mm256_set1_ps(1.0f));

    __m256 t0 = _mm256_sub_ps(_mm256_set1_ps(1.0f), _mm256_mul_ps(x0, x0));
    __m256 t1 = _mm256_sub_ps(_mm256_set1_ps(1.0f), _mm256_mul_ps(x1, x1));

    t0 = _mm256_mul_ps(t0, t0);
    t0 = _mm256_mul_ps(t0, t0);
    t1 = _mm256_mul_ps(t1, t1);
    t1 = _mm256_mul_ps(t1, t1);

    const __m256 n0 = _mm256_mul_ps(t0, getGradient(getHash(i0), x0));
    const __m256 n1 = _mm256_mul_ps(t1, getGradient(getHash(i1), x1));
    const __m256 result = _mm256_mul_ps(amplitudeSet, _mm256_mul_ps(_mm256_add_ps(n0, n1), _mm256_set1_ps(0.395f)));

    loadSet(result);
    return (Set[0] + Set[1] + Set[2] + Set[3] + Set[4] + Set[5] + Set[6] + Set[7]) * amplitudeScale * Height;
}

const float Noise::getValue(const float xValue, const float yValue)
{
    const __m256 x = _mm256_mul_ps(frequencySet, _mm256_set1_ps(xValue));
    const __m256 y = _mm256_mul_ps(frequencySet, _mm256_set1_ps(yValue));
    const __m256 s = _mm256_mul_ps(_mm256_add_ps(x, y), _mm256_set1_ps(0.366025403f));
    const __m256 xs = _mm256_add_ps(x, s);
    const __m256 ys = _mm256_add_ps(y, s);
    const __m256 i = _mm256_floor_ps(xs);
    const __m256 j = _mm256_floor_ps(ys);
    const __m256 t = _mm256_mul_ps(_mm256_add_ps(i, j), _mm256_set1_ps(0.211324865f));
    const __m256 X0 = _mm256_sub_ps(i, t);
    const __m256 Y0 = _mm256_sub_ps(j, t);
    const __m256 x0 = _mm256_sub_ps(x, X0);
    const __m256 y0 = _mm256_sub_ps(y, Y0);
    const __m256 mask = _mm256_cmp_ps(x0, y0, _CMP_GT_OQ);
    const __m256 i1 = _mm256_and_ps(mask, _mm256_set1_ps(1.0f));
    const __m256 j1 = _mm256_andnot_ps(mask, _mm256_set1_ps(1.0f));
    const __m256 x1 = _mm256_add_ps(_mm256_sub_ps(x0, i1), _mm256_set1_ps(0.211324865f));
    const __m256 y1 = _mm256_add_ps(_mm256_sub_ps(y0, j1), _mm256_set1_ps(0.211324865f));
    const __m256 x2 = _mm256_sub_ps(x0, _mm256_set1_ps(0.577350269f));
    const __m256 y2 = _mm256_sub_ps(y0, _mm256_set1_ps(0.577350269f));

    __m256 t0 = _mm256_sub_ps(_mm256_sub_ps(_mm256_set1_ps(0.5f), _mm256_mul_ps(x0, x0)), _mm256_mul_ps(y0, y0));
    __m256 t1 = _mm256_sub_ps(_mm256_sub_ps(_mm256_set1_ps(0.5f), _mm256_mul_ps(x1, x1)), _mm256_mul_ps(y1, y1));
    __m256 t2 = _mm256_sub_ps(_mm256_sub_ps(_mm256_set1_ps(0.5f), _mm256_mul_ps(x2, x2)), _mm256_mul_ps(y2, y2));

    const __m256 mask0 = _mm256_and_ps(_mm256_cmp_ps(t0, _mm256_setzero_ps(), _CMP_GE_OQ), _mm256_set1_ps(1.0f));
    const __m256 mask1 = _mm256_and_ps(_mm256_cmp_ps(t1, _mm256_setzero_ps(), _CMP_GE_OQ), _mm256_set1_ps(1.0f));
    const __m256 mask2 = _mm256_and_ps(_mm256_cmp_ps(t2, _mm256_setzero_ps(), _CMP_GE_OQ), _mm256_set1_ps(1.0f));

    t0 = _mm256_mul_ps(t0, t0);
    t0 = _mm256_mul_ps(t0, t0);
    t1 = _mm256_mul_ps(t1, t1);
    t1 = _mm256_mul_ps(t1, t1);
    t2 = _mm256_mul_ps(t2, t2);
    t2 = _mm256_mul_ps(t2, t2);

    const __m256 hash0 = getHash(_mm256_add_ps(i, getHash(j)));
    const __m256 hash1 = getHash(_mm256_add_ps(_mm256_add_ps(i, i1), getHash(_mm256_add_ps(j, j1))));
    const __m256 hash2 = getHash(_mm256_add_ps(_mm256_add_ps(i, _mm256_set1_ps(1.0f)), getHash(_mm256_add_ps(j, _mm256_set1_ps(1.0f)))));
    const __m256 sum0 = _mm256_mul_ps(mask0, _mm256_mul_ps(t0, getGradient(hash0, x0, y0)));
    const __m256 sum1 = _mm256_mul_ps(mask1, _mm256_mul_ps(t1, getGradient(hash1, x1, y1)));
    const __m256 sum2 = _mm256_mul_ps(mask2, _mm256_mul_ps(t2, getGradient(hash2, x2, y2)));
    const __m256 result = _mm256_mul_ps(amplitudeSet, _mm256_mul_ps(_mm256_add_ps(sum0, _mm256_add_ps(sum1, sum2)), _mm256_set1_ps(45.23065f)));

    loadSet(result);
    return (Set[0] + Set[1] + Set[2] + Set[3] + Set[4] + Set[5] + Set[6] + Set[7]) * amplitudeScale * Height;
}

const float Noise::getValue(const float xValue, const float yValue, const float zValue)
{
    return 0.0f;
}

const float Noise::getValue(const float xValue, const float yValue, const float zValue, const float wValue)
{
    return 0.0f;
}
