// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_SIMDUTILS_H_
#define _SPTAG_COMMON_SIMDUTILS_H_

#include <functional>
#include <iostream>

#include "CommonUtils.h"
#include "InstructionUtils.h"

namespace SPTAG
{
    namespace COMMON
    {
        template <typename T>
        using SumCalcReturn = void(*)(T*, const T*, DimensionType);
        template<typename T>
        inline SumCalcReturn<T> SumCalcSelector();

        class SIMDUtils
        {
        public:
            template <typename T>
            static void ComputeSum_Naive(T* pX, const T* pY, DimensionType length)
            {
                const T* pEnd1 = pX + length;
                while (pX < pEnd1) {
                    *pX++ += *pY++;
                }
            }

            template<typename T>
            static inline void ComputeSum(T* p1, const T* p2, DimensionType length)
            {
                auto func = SumCalcSelector<T>();
                return func(p1, p2, length);
            }
        };

        template<typename T>
        inline SumCalcReturn<T> SumCalcSelector()
        {
            return &(SIMDUtils::ComputeSum_Naive);
        }
    }
}

#endif // _SPTAG_COMMON_SIMDUTILS_H_