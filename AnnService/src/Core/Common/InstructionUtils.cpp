// #include "inc/Core/Common/InstructionUtils.h"
// #include "inc/Core/Common.h"

// #ifndef _MSC_VER
// void cpuid(int info[4], int InfoType) {
//     __cpuid_count(InfoType, 0, info[0], info[1], info[2], info[3]);
// }
// #endif

// namespace SPTAG {
//     namespace COMMON {
//         const InstructionSet::InstructionSet_Internal InstructionSet::CPU_Rep;

//         bool InstructionSet::SSE(void) { return CPU_Rep.HW_SSE; }
//         bool InstructionSet::SSE2(void) { return CPU_Rep.HW_SSE2; }
//         bool InstructionSet::AVX(void) { return CPU_Rep.HW_AVX; }
//         bool InstructionSet::AVX2(void) { return CPU_Rep.HW_AVX2; }
//         bool InstructionSet::AVX512(void) { return CPU_Rep.HW_AVX512; }
        
//         void InstructionSet::PrintInstructionSet(void) 
//         {
//             if (CPU_Rep.HW_AVX512)
//                 SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Using AVX512 InstructionSet!\n");
//             else if (CPU_Rep.HW_AVX2)
//                 SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Using AVX2 InstructionSet!\n");
//             else if (CPU_Rep.HW_AVX)
//                 SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Using AVX InstructionSet!\n");
//             else if (CPU_Rep.HW_SSE2)
//                 SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Using SSE2 InstructionSet!\n");
//             else if (CPU_Rep.HW_SSE)
//                 SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Using SSE InstructionSet!\n");
//             else
//                 SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Using NONE InstructionSet!\n");
//         }

//         // from https://stackoverflow.com/a/7495023/5053214
//         InstructionSet::InstructionSet_Internal::InstructionSet_Internal() :
//             HW_SSE{ false },
//             HW_SSE2{ false },
//             HW_AVX{ false },
//             HW_AVX512{ false },
//             HW_AVX2{ false }
//         {
//             int info[4];
//             cpuid(info, 0);
//             int nIds = info[0];

//             //  Detect Features
//             if (nIds >= 0x00000001) {
//                 cpuid(info, 0x00000001);
//                 HW_SSE = (info[3] & ((int)1 << 25)) != 0;
//                 HW_SSE2 = (info[3] & ((int)1 << 26)) != 0;
//                 HW_AVX = (info[2] & ((int)1 << 28)) != 0;
//             }
//             if (nIds >= 0x00000007) {
//                 cpuid(info, 0x00000007);
//                 HW_AVX2 = (info[1] & ((int)1 << 5)) != 0;
//                 HW_AVX512 = (info[1] & (((int)1 << 16) | ((int) 1 << 30)));

// // If we are not compiling support for AVX-512 due to old compiler version, we should not call it
// #ifdef _MSC_VER
// #if _MSC_VER < 1920
//                 HW_AVX512 = false;
// #endif
// #endif
//             }
//             if (HW_AVX512)
//                 SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Using AVX512 InstructionSet!\n");
//             else if (HW_AVX2)
//                 SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Using AVX2 InstructionSet!\n");
//             else if (HW_AVX)
//                 SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Using AVX InstructionSet!\n");
//             else if (HW_SSE2)
//                 SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Using SSE2 InstructionSet!\n");
//             else if (HW_SSE)
//                 SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Using SSE InstructionSet!\n");
//             else
//                 SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Using NONE InstructionSet!\n");
//         }
//     }
// }
#include "inc/Core/Common/InstructionUtils.h"
#include "inc/Core/Common.h"

// x86 平台的 cpuid 实现
#if defined(__x86_64__) || defined(__i386__) || defined(_M_IX86) || defined(_M_X64)
#ifndef _MSC_VER
void cpuid(int info[4], int InfoType) {
    __cpuid_count(InfoType, 0, info[0], info[1], info[2], info[3]);
}
#endif
#endif

namespace SPTAG {
    namespace COMMON {
        const InstructionSet::InstructionSet_Internal InstructionSet::CPU_Rep;

        bool InstructionSet::SSE(void) { 
#if defined(__x86_64__) || defined(__i386__) || defined(_M_IX86) || defined(_M_X64)
            return CPU_Rep.HW_SSE; 
#else
            return false;
#endif
        }
        
        bool InstructionSet::SSE2(void) { 
#if defined(__x86_64__) || defined(__i386__) || defined(_M_IX86) || defined(_M_X64)
            return CPU_Rep.HW_SSE2; 
#else
            return false;
#endif
        }
        
        bool InstructionSet::AVX(void) { 
#if defined(__x86_64__) || defined(__i386__) || defined(_M_IX86) || defined(_M_X64)
            return CPU_Rep.HW_AVX; 
#else
            return false;
#endif
        }
        
        bool InstructionSet::AVX2(void) { 
#if defined(__x86_64__) || defined(__i386__) || defined(_M_IX86) || defined(_M_X64)
            return CPU_Rep.HW_AVX2; 
#else
            return false;
#endif
        }
        
        bool InstructionSet::AVX512(void) { 
#if defined(__x86_64__) || defined(__i386__) || defined(_M_IX86) || defined(_M_X64)
            return CPU_Rep.HW_AVX512; 
#else
            return false;
#endif
        }

        // ARM 指令集检测方法
        bool InstructionSet::NEON(void) { 
#if defined(__aarch64__) || defined(__arm__)
            return CPU_Rep.HW_NEON; 
#else
            return false;
#endif
        }
        
        bool InstructionSet::SVE(void) { 
#if defined(__aarch64__)
            return CPU_Rep.HW_SVE; 
#else
            return false;
#endif
        }
        
        void InstructionSet::PrintInstructionSet(void) 
        {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_IX86) || defined(_M_X64)
            if (CPU_Rep.HW_AVX512)
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Using AVX512 InstructionSet!\n");
            else if (CPU_Rep.HW_AVX2)
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Using AVX2 InstructionSet!\n");
            else if (CPU_Rep.HW_AVX)
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Using AVX InstructionSet!\n");
            else if (CPU_Rep.HW_SSE2)
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Using SSE2 InstructionSet!\n");
            else if (CPU_Rep.HW_SSE)
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Using SSE InstructionSet!\n");
            else
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Using NONE InstructionSet!\n");
                
#elif defined(__aarch64__) || defined(__arm__)
            std::string arch = 
#ifdef __aarch64__
                "ARM64";
#else
                "ARM32";
#endif
                
            if (CPU_Rep.HW_SVE)
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Using %s with SVE InstructionSet!\n", arch.c_str());
            else if (CPU_Rep.HW_NEON)
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Using %s with NEON InstructionSet!\n", arch.c_str());
            else
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Using %s with basic InstructionSet!\n", arch.c_str());
#else
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Using generic InstructionSet!\n");
#endif
        }

        InstructionSet::InstructionSet_Internal::InstructionSet_Internal() :
            HW_SSE{ false },
            HW_SSE2{ false },
            HW_AVX{ false },
            HW_AVX512{ false },
            HW_AVX2{ false },
            HW_NEON{ false },
            HW_SVE{ false }
        {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_IX86) || defined(_M_X64)
            // x86/x64 架构的 CPUID 检测
            int info[4];
            cpuid(info, 0);
            int nIds = info[0];

            // Detect Features
            if (nIds >= 0x00000001) {
                cpuid(info, 0x00000001);
                HW_SSE = (info[3] & ((int)1 << 25)) != 0;
                HW_SSE2 = (info[3] & ((int)1 << 26)) != 0;
                HW_AVX = (info[2] & ((int)1 << 28)) != 0;
            }
            if (nIds >= 0x00000007) {
                cpuid(info, 0x00000007);
                HW_AVX2 = (info[1] & ((int)1 << 5)) != 0;
                HW_AVX512 = (info[1] & (((int)1 << 16) | ((int) 1 << 30)));

// If we are not compiling support for AVX-512 due to old compiler version, we should not call it
#ifdef _MSC_VER
#if _MSC_VER < 1920
                HW_AVX512 = false;
#endif
#endif
            }
            
#elif defined(__aarch64__) || defined(__arm__)
            // ARM/ARM64 架构的 HWCAP 检测
#ifdef __linux__
            #include <sys/auxv.h>
            #include <asm/hwcap.h>
            
            unsigned long hwcap = getauxval(AT_HWCAP);
            
#ifdef __aarch64__
            // ARM64 特定检测
            HW_NEON = true; // ARM64 强制要求 NEON
            HW_SVE = (hwcap & HWCAP_SVE) != 0;
            
#elif defined(__arm__)
            // ARM32 检测
            HW_NEON = (hwcap & HWCAP_NEON) != 0;
            HW_SVE = false; // SVE 只在 ARM64 上可用
#endif

#elif defined(__APPLE__)
            // macOS (Apple Silicon) 检测
            // Apple Silicon 的 ARM64 芯片都支持 NEON
            HW_NEON = true;
            
            // 检测 SVE (Apple M系列可能支持)
            #ifdef __aarch64__
                // 这里可以添加更精确的 SVE 检测
                // 目前假设不支持，因为 Apple 主要使用 AMX
                HW_SVE = false;
            #endif
            
#endif // __linux__
            
#endif // 架构检测
        }
    }
}