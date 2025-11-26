// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_QUERYRESULTSET_H_
#define _SPTAG_COMMON_QUERYRESULTSET_H_

#include "inc/Core/SearchQuery.h"
#include "DistanceUtils.h"
#include <algorithm>
#include "IQuantizer.h"

namespace SPTAG
{
namespace COMMON
{

inline bool operator < (const BasicResult& lhs, const BasicResult& rhs)
{
    return ((lhs.Dist < rhs.Dist) || ((lhs.Dist == rhs.Dist) && (lhs.VID < rhs.VID)));
}


inline bool Compare(const BasicResult& lhs, const BasicResult& rhs)
{
    return ((lhs.Dist < rhs.Dist) || ((lhs.Dist == rhs.Dist) && (lhs.VID < rhs.VID)));
}


// Space to save temporary answer, similar with TopKCache
template<typename T>
class QueryResultSet : public QueryResult
{
public:
    QueryResultSet(const T* _target, int _K) : QueryResult(_target, _K, false)
    {
    }

    QueryResultSet(const QueryResultSet& other) : QueryResult(other)
    {
    }

    ~QueryResultSet()
    {
    }

    inline void SetTarget(const T* p_target, const std::shared_ptr<IQuantizer>& quantizer)
    {
        if (quantizer == nullptr) QueryResult::SetTarget((const void*)p_target);
        else
        {
            if (m_target == m_quantizedTarget || (m_quantizedSize != quantizer->QuantizeSize()))
            {
                if (m_target != m_quantizedTarget) ALIGN_FREE(m_quantizedTarget);
                m_quantizedTarget = ALIGN_ALLOC(quantizer->QuantizeSize());
                m_quantizedSize = quantizer->QuantizeSize();
            }
            m_target = p_target;
            quantizer->QuantizeVector((void*)p_target, (uint8_t*)m_quantizedTarget);
        }
    }

    inline void Setmy_newPQTarget(const std::shared_ptr<IQuantizer>& quantizer)
    {
        m_newPQTarget = ALIGN_ALLOC(quantizer->QuantizeSize());
        quantizer->QuantizeVector((void*)m_target, (uint8_t*)m_newPQTarget);
    }

    inline void Setmy_headPQTarget(const std::shared_ptr<IQuantizer>& quantizer)
    {
        m_headPQTarget = ALIGN_ALLOC(quantizer->QuantizeSize());
        quantizer->QuantizeVector((void*)m_target, (uint8_t*)m_headPQTarget);
    }

    inline const T* GetTarget() const
    {
        return reinterpret_cast<const T*>(m_target);
    }

    T* GetQuantizedTarget()
    {
        return reinterpret_cast<T*>(m_quantizedTarget);
    }

    T* GetnewPQTarget()
    {
        return reinterpret_cast<T*>(m_newPQTarget);
    }

    T* GetheadPQTarget()
    {
        return reinterpret_cast<T*>(m_headPQTarget);
    }

    inline float worstDist() const
    {
        return m_results[0].Dist;
    }

    bool AddPoint(const SizeType index, float dist)
    {
        if (dist < m_results[0].Dist || (dist == m_results[0].Dist && index < m_results[0].VID))
        {
            m_results[0].VID = index;
            m_results[0].Dist = dist;
            Heapify(m_resultNum);
            return true;
        }
        return false;
    }

    bool AddPoint(const SizeType index, float dist, int ResultNum)
    {
        if (dist < m_results[0].Dist || (dist == m_results[0].Dist && index < m_results[0].VID))
        {
            m_results[0].VID = index;
            m_results[0].Dist = dist;
            Heapify(ResultNum);
            return true;
        }
        return false;
    }

    inline void SortResult()
    {
        for (int i = m_resultNum - 1; i >= 0; i--)
        {
            std::swap(m_results[0], m_results[i]);
            Heapify(i);
        }
    }

    void Reverse()
    {
        std::reverse(m_results.Data(), m_results.Data() + m_resultNum);
    }

    void Reverse(int ResultNum)
    {
        std::reverse(m_results.Data(), m_results.Data() + ResultNum);
    }

    inline void initialTopK(int rerankNum)
    {
        for (int i = 0; i < rerankNum; i++)
        {
            initialTopKIDs.insert(m_results[i].VID);
        }
    }

    inline void GetHeapResult()
    {
        std::copy(m_results.Data(), m_results.Data() + m_resultNum, m_heapResults.Data());
        std::sort(m_heapResults.Data(), m_heapResults.Data() + m_resultNum, COMMON::Compare);
    }

    int IsHeapUnchanged(int rerankNum)
    {
        GetHeapResult();
        for (int i = 0; i < rerankNum; ++i) {
            if (initialTopKIDs.find(m_heapResults[i].VID) == initialTopKIDs.end()) {
                initialTopKIDs.clear();
                for (int j = 0; j < rerankNum; ++j) {
                    initialTopKIDs.insert(m_heapResults[j].VID);
                }
                return i + 1;
            }
        }
        return 0;
    }

    int CountHeapChanges(int rerankNum)
    {
        GetHeapResult(); // 更新堆结果
        int changes = 0; // 记录变化的元素总数

        // 遍历前 rerankNum 个堆结果
        for (int i = 0; i < rerankNum; ++i) {
            // 如果当前元素的 VID 不在初始集合中
            if (initialTopKIDs.find(m_heapResults[i].VID) == initialTopKIDs.end()) {
                ++changes; // 增加变化计数
            }
        }

        // 更新 initialTopKIDs 为当前堆的前 rerankNum 个结果
        initialTopKIDs.clear();
        for (int j = 0; j < rerankNum; ++j) {
            initialTopKIDs.insert(m_heapResults[j].VID);
        }

        return changes; // 返回变化的总数
    }


    bool IsHeapUnchanged1(int rerankNum)
    {
        GetHeapResult();
        for (int i = 0; i < rerankNum; ++i) {
            if (initialTopKIDs.find(m_heapResults[i].VID) == initialTopKIDs.end()) {
                initialTopKIDs.clear();
                for (int j = 0; j < rerankNum; ++j) {
                    initialTopKIDs.insert(m_heapResults[j].VID);
                }
                return false;
            }
        }
        return true;
    }

    // int IsHeapUnchanged(int rerankNum)
    // {
    //     GetHeapResult();
    //     int heapUnchanged = 0;
    //     for (int i = 0; i < rerankNum + 10; ++i) {
    //         if (initialTopKIDs.find(m_heapResults[i].VID) == initialTopKIDs.end()) {
    //             heapUnchanged ++;
    //         }
    //     }
    //     if(heapUnchanged > 0){
    //         initialTopKIDs.clear();
    //         for (int j = 0; j < rerankNum + 10; ++j) {
    //             initialTopKIDs.insert(m_heapResults[j].VID);
    //         }
    //     }
    //     return heapUnchanged;
    // }

private:
    void Heapify(int count)
    {
        int parent = 0, next = 1, maxidx = count - 1;
        while (next < maxidx)
        {
            if (m_results[next] < m_results[next + 1]) next++;
            if (m_results[parent] < m_results[next])
            {
                std::swap(m_results[next], m_results[parent]);
                parent = next;
                next = (parent << 1) + 1;
            }
            else break;
        }
        if (next == maxidx && m_results[parent] < m_results[next]) std::swap(m_results[parent], m_results[next]);
    }
};
}
}

#endif // _SPTAG_COMMON_QUERYRESULTSET_H_
