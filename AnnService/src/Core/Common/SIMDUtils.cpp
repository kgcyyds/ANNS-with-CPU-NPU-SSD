// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/Common/SIMDUtils.h"

using namespace SPTAG;
using namespace SPTAG::COMMON;

void SIMDUtils::ComputeSum_NEON(std::int8_t* pX, const std::int8_t* pY, DimensionType length)
{
    const std::int8_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::int8_t* pEnd1 = pX + length;

    while (pX < pEnd16) {
        int8x16_t x_part = vld1q_s8(pX);
        int8x16_t y_part = vld1q_s8(pY);
        vst1q_s8(pX, vaddq_s8(x_part, y_part));
        pX += 16;
        pY += 16;
    }
    while (pX < pEnd1) *pX++ += *pY++;
}

void SIMDUtils::ComputeSum_NEON(std::uint8_t* pX, const std::uint8_t* pY, DimensionType length)
{
    const std::uint8_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::uint8_t* pEnd1 = pX + length;

    while (pX < pEnd16) {
        uint8x16_t x_part = vld1q_u8(pX);
        uint8x16_t y_part = vld1q_u8(pY);
        uint8x16_t sum = vaddq_u8(x_part, y_part);
        vst1q_u8(pX, sum);
        pX += 16;
        pY += 16;
    }

    while (pX < pEnd1) {
        *pX++ += *pY++;
    }
}

void SIMDUtils::ComputeSum_NEON(std::int16_t* pX, const std::int16_t* pY, DimensionType length) {
    const std::int16_t* pEnd8 = pX + ((length >> 3) << 3);
    const std::int16_t* pEnd1 = pX + length;
    while (pX < pEnd8) {
        int16x8_t x_part = vld1q_s16(pX);
        int16x8_t y_part = vld1q_s16(pY);
        vst1q_s16(pX, vaddq_s16(x_part, y_part));
        pX += 8; pY += 8;
    }
    while (pX < pEnd1) *pX++ += *pY++;
}

void SIMDUtils::ComputeSum_NEON(float* pX, const float* pY, DimensionType length) {
    const float* pEnd4 = pX + ((length >> 2) << 2);
    const float* pEnd1 = pX + length;
    while (pX < pEnd4) {
        float32x4_t x_part = vld1q_f32(pX);
        float32x4_t y_part = vld1q_f32(pY);
        vst1q_f32(pX, vaddq_f32(x_part, y_part));
        pX += 4; pY += 4;
    }
    while (pX < pEnd1) *pX++ += *pY++;
}