// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/SPANN/Index.h"
#include "inc/Helper/VectorSetReaders/MemoryReader.h"
#include "inc/Core/SPANN/ExtraFullGraphSearcher.h"
#include <chrono>
#include <numeric>

// #include <boost/asio.hpp>
#include <future>
#include <thread>

#include <fcntl.h>
#include <sys/syscall.h>
#include <linux/aio_abi.h>
#include "acl/acl.h"
#include "aclnnop/aclnn_reduce_sum.h"
#include "aclnnop/aclnn_gather.h"
#include "aclnnop/aclnn_cast.h"
// extern void add_custom_do(uint32_t blockDim, void *stream, uint8_t *set, uint8_t *id, uint8_t *table, uint8_t *dist, int dim, int count);
#pragma warning(disable : 4242) // '=' : conversion from 'int' to 'short', possible loss of data
#pragma warning(disable : 4244) // '=' : conversion from 'int' to 'short', possible loss of data
#pragma warning(disable : 4127) // conditional expression is constant

namespace SPTAG
{
    template <typename T>
    thread_local std::unique_ptr<T> COMMON::ThreadLocalWorkSpaceFactory<T>::m_workspace;
    namespace SPANN
    {
        std::atomic_int ExtraWorkSpace::g_spaceCount(0);
        EdgeCompare Selection::g_edgeComparer;

        std::function<std::shared_ptr<Helper::DiskIO>(void)> f_createAsyncIO = []() -> std::shared_ptr<Helper::DiskIO>
        { return std::shared_ptr<Helper::DiskIO>(new Helper::AsyncFileIO()); };

        template <typename T>
        bool Index<T>::CheckHeadIndexType()
        {
            SPTAG::VectorValueType v1 = m_index->GetVectorValueType(), v2 = GetEnumValueType<T>();
            if (v1 != v2)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Head index and vectors don't have the same value types, which are %s %s\n",
                             SPTAG::Helper::Convert::ConvertToString(v1).c_str(),
                             SPTAG::Helper::Convert::ConvertToString(v2).c_str());
                if (!m_pQuantizer)
                    return false;
            }
            return true;
        }

        template <typename T>
        void Index<T>::SetQuantizer(std::shared_ptr<SPTAG::COMMON::IQuantizer> quantizer)
        {
            m_pQuantizer = quantizer;
            if (m_pQuantizer)
            {
                m_fComputeDistance = m_pQuantizer->DistanceCalcSelector<T>(m_options.m_distCalcMethod);
                m_iBaseSquare = (m_options.m_distCalcMethod == DistCalcMethod::Cosine) ? m_pQuantizer->GetBase() * m_pQuantizer->GetBase() : 1;
            }
            else
            {
                m_fComputeDistance = COMMON::DistanceCalcSelector<T>(m_options.m_distCalcMethod);
                m_iBaseSquare = (m_options.m_distCalcMethod == DistCalcMethod::Cosine) ? COMMON::Utils::GetBase<std::uint8_t>() * COMMON::Utils::GetBase<std::uint8_t>() : 1;
            }
            if (m_index)
            {
                m_index->SetQuantizer(quantizer);
            }
        }

        template <typename T>
        ErrorCode Index<T>::LoadConfig(Helper::IniReader &p_reader)
        {
            IndexAlgoType algoType = p_reader.GetParameter("Base", "IndexAlgoType", IndexAlgoType::Undefined);
            VectorValueType valueType = p_reader.GetParameter("Base", "ValueType", VectorValueType::Undefined);
            if ((m_index = CreateInstance(algoType, valueType)) == nullptr)
                return ErrorCode::FailedParseValue;

            std::string sections[] = {"Base", "SelectHead", "BuildHead", "BuildSSDIndex"};
            for (int i = 0; i < 4; i++)
            {
                auto parameters = p_reader.GetParameters(sections[i].c_str());
                for (auto iter = parameters.begin(); iter != parameters.end(); iter++)
                {
                    SetParameter(iter->first.c_str(), iter->second.c_str(), sections[i].c_str());
                }
            }

            if (m_pQuantizer)
            {
                m_pQuantizer->SetEnableADC(m_options.m_enableADC);
            }

            return ErrorCode::Success;
        }

        template <typename T>
        ErrorCode Index<T>::LoadIndexDataFromMemory(const std::vector<ByteArray> &p_indexBlobs)
        {
            m_index->SetQuantizer(m_pQuantizer);
            if (m_index->LoadIndexDataFromMemory(p_indexBlobs) != ErrorCode::Success)
                return ErrorCode::Fail;

            m_index->SetParameter("NumberOfThreads", std::to_string(m_options.m_iSSDNumberOfThreads));
            // m_index->SetParameter("MaxCheck", std::to_string(m_options.m_maxCheck));
            // m_index->SetParameter("HashTableExponent", std::to_string(m_options.m_hashExp));
            m_index->UpdateIndex();
            m_index->SetReady(true);

            if (m_pQuantizer)
            {
                m_extraSearcher.reset(new ExtraFullGraphSearcher<std::uint8_t>());
            }
            else
            {
                m_extraSearcher.reset(new ExtraFullGraphSearcher<T>());
            }

            if (!m_extraSearcher->LoadIndex(m_options))
                return ErrorCode::Fail;

            m_vectorTranslateMap.reset((std::uint64_t *)(p_indexBlobs.back().Data()), [=](std::uint64_t *ptr) {});

            omp_set_num_threads(m_options.m_iSSDNumberOfThreads);
            return ErrorCode::Success;
        }

        template <typename T>
        ErrorCode Index<T>::LoadIndexData(const std::vector<std::shared_ptr<Helper::DiskIO>> &p_indexStreams)
        {
            m_index->SetQuantizer(m_pQuantizer);
            if (m_index->LoadIndexData(p_indexStreams) != ErrorCode::Success)
                return ErrorCode::Fail;

            m_index->SetParameter("NumberOfThreads", std::to_string(m_options.m_iSSDNumberOfThreads));
            // m_index->SetParameter("MaxCheck", std::to_string(m_options.m_maxCheck));
            // m_index->SetParameter("HashTableExponent", std::to_string(m_options.m_hashExp));
            m_index->UpdateIndex();
            m_index->SetReady(true);

            if (m_pQuantizer)
            {
                m_extraSearcher.reset(new ExtraFullGraphSearcher<std::uint8_t>());
            }
            else
            {
                m_extraSearcher.reset(new ExtraFullGraphSearcher<T>());
            }

            if (!m_extraSearcher->LoadIndex(m_options))
                return ErrorCode::Fail;

            m_vectorTranslateMap.reset(new std::uint64_t[m_index->GetNumSamples()], std::default_delete<std::uint64_t[]>());
            IOBINARY(p_indexStreams[m_index->GetIndexFiles()->size()], ReadBinary, sizeof(std::uint64_t) * m_index->GetNumSamples(), reinterpret_cast<char *>(m_vectorTranslateMap.get()));

            omp_set_num_threads(m_options.m_iSSDNumberOfThreads);
            return ErrorCode::Success;
        }

        template <typename T>
        ErrorCode Index<T>::SaveConfig(std::shared_ptr<Helper::DiskIO> p_configOut)
        {
            IOSTRING(p_configOut, WriteString, "[Base]\n");
#define DefineBasicParameter(VarName, VarType, DefaultValue, RepresentStr) \
    IOSTRING(p_configOut, WriteString, (RepresentStr + std::string("=") + SPTAG::Helper::Convert::ConvertToString(m_options.VarName) + std::string("\n")).c_str());

#include "inc/Core/SPANN/ParameterDefinitionList.h"
#undef DefineBasicParameter

            IOSTRING(p_configOut, WriteString, "[SelectHead]\n");
#define DefineSelectHeadParameter(VarName, VarType, DefaultValue, RepresentStr) \
    IOSTRING(p_configOut, WriteString, (RepresentStr + std::string("=") + SPTAG::Helper::Convert::ConvertToString(m_options.VarName) + std::string("\n")).c_str());

#include "inc/Core/SPANN/ParameterDefinitionList.h"
#undef DefineSelectHeadParameter

            IOSTRING(p_configOut, WriteString, "[BuildHead]\n");
#define DefineBuildHeadParameter(VarName, VarType, DefaultValue, RepresentStr) \
    IOSTRING(p_configOut, WriteString, (RepresentStr + std::string("=") + SPTAG::Helper::Convert::ConvertToString(m_options.VarName) + std::string("\n")).c_str());

#include "inc/Core/SPANN/ParameterDefinitionList.h"
#undef DefineBuildHeadParameter

            m_index->SaveConfig(p_configOut);

            Helper::Convert::ConvertStringTo<int>(m_index->GetParameter("HashTableExponent").c_str(), m_options.m_hashExp);
            IOSTRING(p_configOut, WriteString, "[BuildSSDIndex]\n");
#define DefineSSDParameter(VarName, VarType, DefaultValue, RepresentStr) \
    IOSTRING(p_configOut, WriteString, (RepresentStr + std::string("=") + SPTAG::Helper::Convert::ConvertToString(m_options.VarName) + std::string("\n")).c_str());

#include "inc/Core/SPANN/ParameterDefinitionList.h"
#undef DefineSSDParameter

            IOSTRING(p_configOut, WriteString, "\n");
            return ErrorCode::Success;
        }

        template <typename T>
        ErrorCode Index<T>::SaveIndexData(const std::vector<std::shared_ptr<Helper::DiskIO>> &p_indexStreams)
        {
            if (m_index == nullptr || m_vectorTranslateMap == nullptr)
                return ErrorCode::EmptyIndex;

            ErrorCode ret;
            if ((ret = m_index->SaveIndexData(p_indexStreams)) != ErrorCode::Success)
                return ret;

            IOBINARY(p_indexStreams[m_index->GetIndexFiles()->size()], WriteBinary, sizeof(std::uint64_t) * m_index->GetNumSamples(), (char *)(m_vectorTranslateMap.get()));
            return ErrorCode::Success;
        }

#pragma region K-NN search

        template <typename T>
        ErrorCode Index<T>::SearchIndex(QueryResult &p_query, bool p_searchDeleted) const
        {
            if (!m_bReady)
                return ErrorCode::EmptyIndex;

            COMMON::QueryResultSet<T> *p_queryResults;
            if (p_query.GetResultNum() >= m_options.m_searchInternalResultNum)
                p_queryResults = (COMMON::QueryResultSet<T> *)&p_query;
            else
                p_queryResults = new COMMON::QueryResultSet<T>((const T *)p_query.GetTarget(), m_options.m_searchInternalResultNum);

            m_index->SearchIndex(*p_queryResults);

            if (m_extraSearcher != nullptr)
            {
                auto workSpace = m_workSpaceFactory->GetWorkSpace();
                if (!workSpace)
                {
                    workSpace.reset(new ExtraWorkSpace());
                    workSpace->Initialize(m_options.m_maxCheck, m_options.m_hashExp, m_options.m_searchInternalResultNum, max(m_options.m_postingPageLimit, m_options.m_searchPostingPageLimit + 1) << PageSizeEx, m_options.m_enableDataCompression);
                }
                else
                {
                    workSpace->Clear(m_options.m_searchInternalResultNum, max(m_options.m_postingPageLimit, m_options.m_searchPostingPageLimit + 1) << PageSizeEx, m_options.m_enableDataCompression);
                }
                workSpace->m_deduper.clear();
                workSpace->m_postingIDs.clear();

                float limitDist = p_queryResults->GetResult(0)->Dist * m_options.m_maxDistRatio;
                for (int i = 0; i < p_queryResults->GetResultNum(); ++i)
                {
                    auto res = p_queryResults->GetResult(i);
                    if (res->VID == -1)
                        break;

                    auto postingID = res->VID;
                    res->VID = static_cast<SizeType>((m_vectorTranslateMap.get())[res->VID]);
                    if (res->VID == MaxSize)
                    {
                        res->VID = -1;
                        res->Dist = MaxDist;
                    }

                    // Don't do disk reads for irrelevant pages
                    if (workSpace->m_postingIDs.size() >= m_options.m_searchInternalResultNum ||
                        (limitDist > 0.1 && res->Dist > limitDist) ||
                        !m_extraSearcher->CheckValidPosting(postingID))
                        continue;
                    workSpace->m_postingIDs.emplace_back(postingID);
                }

                p_queryResults->Reverse();
                m_extraSearcher->SearchIndex(workSpace.get(), *p_queryResults, m_index, nullptr);
                m_workSpaceFactory->ReturnWorkSpace(std::move(workSpace));
                p_queryResults->SortResult();
            }

            if (p_query.GetResultNum() < m_options.m_searchInternalResultNum)
            {
                std::copy(p_queryResults->GetResults(), p_queryResults->GetResults() + p_query.GetResultNum(), p_query.GetResults());
                delete p_queryResults;
            }

            if (p_query.WithMeta() && nullptr != m_pMetadata)
            {
                for (int i = 0; i < p_query.GetResultNum(); ++i)
                {
                    SizeType result = p_query.GetResult(i)->VID;
                    p_query.SetMetadata(i, (result < 0) ? ByteArray::c_empty : m_pMetadata->GetMetadataCopy(result));
                }
            }
            return ErrorCode::Success;
        }

        template <typename T>
        ErrorCode Index<T>::SearchIndexWithFilter(QueryResult &p_query, std::function<bool(const ByteArray &)> filterFunc, int maxCheck, bool p_searchDeleted) const
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Not Support Filter on SPANN Index!\n");
            return ErrorCode::Fail;
        }

        template <typename T>
        ErrorCode Index<T>::SearchDiskIndex(QueryResult &p_query, SearchStats *p_stats) const
        {
            if (nullptr == m_extraSearcher)
                return ErrorCode::EmptyIndex;

            COMMON::QueryResultSet<T> *p_queryResults = (COMMON::QueryResultSet<T> *)&p_query;

            auto workSpace = m_workSpaceFactory->GetWorkSpace();
            if (!workSpace)
            {
                workSpace.reset(new ExtraWorkSpace());
                workSpace->Initialize(m_options.m_maxCheck, m_options.m_hashExp, m_options.m_searchInternalResultNum, max(m_options.m_postingPageLimit, m_options.m_searchPostingPageLimit + 1) << PageSizeEx, m_options.m_enableDataCompression);
            }
            else
            {
                workSpace->Clear(m_options.m_searchInternalResultNum, max(m_options.m_postingPageLimit, m_options.m_searchPostingPageLimit + 1) << PageSizeEx, m_options.m_enableDataCompression);
            }
            workSpace->m_deduper.clear();
            workSpace->m_postingIDs.clear();

            float limitDist = p_queryResults->GetResult(0)->Dist * m_options.m_maxDistRatio;
            int i = 0;
            for (; i < m_options.m_searchInternalResultNum; ++i)
            {
                auto res = p_queryResults->GetResult(i);
                if (res->VID == -1 || (limitDist > 0.1 && res->Dist > limitDist))
                    break;
                if (m_extraSearcher->CheckValidPosting(res->VID))
                {
                    workSpace->m_postingIDs.emplace_back(res->VID);
                }
                res->VID = static_cast<SizeType>((m_vectorTranslateMap.get())[res->VID]);
                if (res->VID == MaxSize)
                {
                    res->VID = -1;
                    res->Dist = MaxDist;
                }
            }

            for (; i < p_queryResults->GetResultNum(); ++i)
            {
                auto res = p_queryResults->GetResult(i);
                if (res->VID == -1)
                    break;
                res->VID = static_cast<SizeType>((m_vectorTranslateMap.get())[res->VID]);
                if (res->VID == MaxSize)
                {
                    res->VID = -1;
                    res->Dist = MaxDist;
                }
            }

            p_queryResults->Reverse();
            m_extraSearcher->SearchIndex(workSpace.get(), *p_queryResults, m_index, p_stats);
            m_workSpaceFactory->ReturnWorkSpace(std::move(workSpace));
            p_queryResults->SortResult();
            return ErrorCode::Success;
        }
        template <typename T>
        float Index<T>::L2Distance(const std::uint8_t *pX, const std::uint8_t *pY, int dimVectors) const
        // pX must be query distance table for ADC
        {
            float out = 0;
            float *ptr = (float *)pX;
            for (int i = 0; i < dimVectors; i++)
            {
                // out += ptr[pY[i]];
                out += ptr[*pY++];
                ptr += 256; // m_KsPerSubvector
            }
            return out;
        }

        template <typename T>
        float Index<T>::L2Distance_Dmul4(const std::uint8_t *pX, const std::uint8_t *pY, int dimVectors) const
        // pX must be query distance table for ADC
        {
            float dis = 0;
            float *dis_table = (float *)pX;
            int count = dimVectors / 4;
            int reCount = dimVectors % 4;
            for (size_t i = 0; i < count; i ++)
            {
                float subdis = 0;
                subdis = dis_table[*pY++];
                dis_table += 256;
                subdis += dis_table[*pY++];
                dis_table += 256;
                subdis += dis_table[*pY++];
                dis_table += 256;
                subdis += dis_table[*pY++];
                dis_table += 256;
                dis += subdis;
            }

            for (size_t i = 0; i < reCount; i++)
            {
                dis += dis_table[*pY++];
                dis_table += 256;
            }

            return dis;
        }
        template <typename T>
        float Index<T>::InitializeL2Distance(const std::uint8_t *pX, const std::uint8_t *pY, DimensionType m_DimPerSubvector) const
        {
            const std::uint8_t *pEnd4 = pX + ((m_DimPerSubvector >> 2) << 2);
            const std::uint8_t *pEnd1 = pX + m_DimPerSubvector;

            float diff = 0;

            while (pX < pEnd4)
            {
                float c1 = ((float)(*pX++) - (float)(*pY++));
                diff += c1 * c1;
                c1 = ((float)(*pX++) - (float)(*pY++));
                diff += c1 * c1;
                c1 = ((float)(*pX++) - (float)(*pY++));
                diff += c1 * c1;
                c1 = ((float)(*pX++) - (float)(*pY++));
                diff += c1 * c1;
            }
            while (pX < pEnd1)
            {
                float c1 = ((float)(*pX++) - (float)(*pY++));
                diff += c1 * c1;
            }
            return diff;
        }
        /*
            第二个CPU版本，PQ向量在内存中
        */
        template <typename T>
        ErrorCode Index<T>::SearchPQIndex_CPU(QueryResult &p_query, std::shared_ptr<VectorSet> vectorSet, int PQVectorsCount, int PQVectorDim, std::vector<int>& numVecPerPostinglist, std::vector<std::unique_ptr<int[]>>& postinglist, std::unordered_set<int>& postingIDSet, SearchStats *p_stats) const
        {
            auto cutStartTime = std::chrono::high_resolution_clock::now();
            COMMON::QueryResultSet<T> *queryResults = (COMMON::QueryResultSet<T> *)&p_query;

            // 剪枝策略，根据searchInternalResultNum遍历queryResults, queryResults存放postinglistid和postinglist到query的距离
            std::vector<int> postingIDs;
            // std::vector<int> isExistVectorID;
            COMMON::OptHashPosVector deduper;
            deduper.Init(m_options.m_maxCheck, m_options.m_hashExp);
            float limitDist = queryResults->GetResult(0)->Dist * m_options.m_maxDistRatio; // 设置距离限制（最大距离为最近质心到查询向量距离的8倍）
            int ii = 0;
            for (; ii < m_options.m_searchInternalResultNum; ++ii) // 动态剪枝策略
            {
                auto res = queryResults->GetResult(ii);
                if (res->VID == -1 || (limitDist > 0.1 && res->Dist > limitDist))
                    break;
                if (numVecPerPostinglist[res->VID] != 0) // 将符合条件的质心加入到结果中
                {
                    postingIDs.emplace_back(res->VID);
                }
                res->VID = static_cast<SizeType>((m_vectorTranslateMap.get())[res->VID]);
                if (!m_options.m_enableReorderIndex)
                {
                    postingIDSet.insert(res->VID);
                }
                if (res->VID == MaxSize)
                {
                    res->VID = -1;
                    res->Dist = MaxDist;
                }
            }
            for (; ii < queryResults->GetResultNum(); ++ii)
            {
                auto res = queryResults->GetResult(ii);
                if (res->VID == -1)
                    break;
                res->VID = static_cast<SizeType>((m_vectorTranslateMap.get())[res->VID]);
                if (!m_options.m_enableReorderIndex)
                {
                    postingIDSet.insert(res->VID);
                }
                if (res->VID == MaxSize)
                {
                    res->VID = -1;
                    res->Dist = MaxDist;
                }
            }
            queryResults->Reverse();
            auto cutEndTime = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> cutelapsed = cutEndTime - cutStartTime;
            double cutelapsedMilliseconds = cutelapsed.count();
            p_stats->cutTreeLatency = cutelapsedMilliseconds;
            // 搜索逻辑
            const uint32_t postingListCount = static_cast<uint32_t>(postingIDs.size());
            std::vector<int> vectorIDs;
            for (uint32_t pi = 0; pi < postingListCount; ++pi)
            {

                int postingID = postingIDs[pi];
                int numvec = numVecPerPostinglist[postingID];
                for (int vi = 0; vi < numvec; vi++)
                {

                    int vectorID = postinglist[postingID][vi];
                    if (deduper.CheckAndSet(vectorID))
                        continue;
                    vectorIDs.push_back(vectorID);
                    // const uint8_t *vector = reinterpret_cast<const uint8_t *>(vectorSet->GetVector(vectorID));
                    // float dist = L2Distance_Dmul4(reinterpret_cast<const uint8_t *>(queryResults->GetnewPQTarget()), vector, PQVectorDim);
                    // queryResults->AddPoint(vectorID, dist);
                    
                }
            }
            const float *pq_dists = reinterpret_cast<const float *>(queryResults->GetnewPQTarget());
            int totalNumVectors = vectorIDs.size();
            std::vector<uint8_t> pq_ids(totalNumVectors * PQVectorDim);
            std::vector<float> dists_out(totalNumVectors);
            

            std::vector<uint8_t> pq_codes(totalNumVectors * PQVectorDim); // 直接初始化 
            const size_t prefetch_ahead = 4;  // 始终保持预取4个向量 
            const size_t prefetch_batch = 1;  // 每次新增预取1个向量
            
            // 初始预取prefetch_ahead个向量
            for (size_t p = 0; p < prefetch_ahead && p < totalNumVectors; p++) {
                const uint8_t *future_vec = reinterpret_cast<const uint8_t*>(
                    vectorSet->GetVector(vectorIDs[p]));
                // _mm_prefetch((char*)future_vec, _MM_HINT_T0);
            }
            
            for (size_t i = 0; i < totalNumVectors; i++) {
                // 处理当前向量 
                const uint8_t *src = reinterpret_cast<const uint8_t*>(vectorSet->GetVector(vectorIDs[i]));
                std::memcpy(pq_codes.data()  + i * PQVectorDim, src, PQVectorDim);
            
                // 预取下一个向量（如果还有剩余）
                size_t next_prefetch_idx = i + prefetch_ahead;
                if (next_prefetch_idx < totalNumVectors) {
                    const uint8_t *future_vec = reinterpret_cast<const uint8_t*>(
                        vectorSet->GetVector(vectorIDs[next_prefetch_idx]));
                    // _mm_prefetch((char*)future_vec, _MM_HINT_T0);
                }
            }
            for (size_t i = 0; i < totalNumVectors; i++)
            {
                dists_out[i] = L2Distance_Dmul4(reinterpret_cast<const uint8_t *>(queryResults->GetnewPQTarget()), pq_codes.data() + PQVectorDim * i, PQVectorDim);
            }
            // int idx = 0;
            // for (size_t i = 0; i < totalNumVectors; i++)
            // {
            //     const uint8_t *vector = reinterpret_cast<const uint8_t *>(vectorSet->GetVector(vectorIDs[i]));
            //     for (int c = 0; c < PQVectorDim; ++c) // PQ_CODE_SIZE 是一个 PQ 向量的维度
            //     {
            //         pq_ids[idx++] = vector[c];  // 存储 PQ 向量的编码
            //     }
            // }
            // for (size_t chunk = 0; chunk < PQVectorDim; chunk++)
            // {
            //     const float *chunk_dists = pq_dists + 256 * chunk;  // 每个 PQ 分块的距离表
            //     if (chunk < PQVectorDim - 1)
            //     {
            //         __builtin_prefetch((char *)(chunk_dists + 256), 0, 3); // 预取下一块距离表
            //     }
            //     // 遍历所有的点，计算对应的 PQ 距离
            //     for (size_t i = 0; i < totalNumVectors; i++)
            //     {
            //         uint8_t pq_centerid = pq_ids[PQVectorDim * i + chunk];  // 获取每个点对应的 PQ 编码
            //         dists_out[i] += chunk_dists[pq_centerid];  // 累加每个点的 PQ 距离
            //     }
            // }
            for (size_t i = 0; i < totalNumVectors; i++)
                queryResults->AddPoint(vectorIDs[i], dists_out[i]);
            

            auto sortStartTime = std::chrono::high_resolution_clock::now();
            // 堆排序，利用堆排序，将m_result排序为从小打到的顺序
            queryResults->SortResult();
            auto sortEndTime = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> sortelapsed = sortEndTime - sortStartTime;
            double elapsedMilliseconds = sortelapsed.count();
            p_stats->sortLatency = elapsedMilliseconds;
            return ErrorCode::Success;
        }
        /*
            最终GPU版本
        */
        int64_t GetShapeSize(const std::vector<int64_t>& shape) {
            int64_t shape_size = 1;
            for (auto i : shape) {
                shape_size *= i;
            }
            return shape_size;
        }
        int CreateAclTensor(const std::vector<int64_t>& shape, void** deviceAddr,
                            aclDataType dataType, aclTensor** tensor) {
            // 计算连续tensor的strides
            std::vector<int64_t> strides(shape.size(), 1);
            for (int64_t i = shape.size() - 2; i >= 0; i--) {
                strides[i] = shape[i + 1] * strides[i + 1];
            }
            // 调用aclCreateTensor接口创建aclTensor
            *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                        shape.data(), shape.size(), *deviceAddr);
            return 0;
        }
        int CreateTableAclTensor(const std::vector<int64_t>& shape, void** deviceAddr,
                            aclDataType dataType, aclTensor** tensor) {
            std::vector<int64_t> strides = {1, 256};
            // 调用aclCreateTensor接口创建aclTensor
            *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                        shape.data(), shape.size(), *deviceAddr);
            return 0;
        }
        template <typename T>
        ErrorCode Index<T>::SearchPQIndex_GPU(QueryResult& p_query, std::shared_ptr<VectorSet> vectorSet, int PQVectorsCount, int PQVectorDim, 
        std::vector<int>& numVecPerPostinglist, std::vector<std::unique_ptr<int[]>>& postinglist, void* d_PQVectorSet, 
        uint8_t *d_table, int* d_vectorIDs, float *d_dist, int* h_vectorIDs, float *h_dist, int totalNumVec, int threadOrder, 
        std::unordered_set<int>& postingIDSet, SearchStats* p_stats) const
        {
            aclInit(nullptr);
            aclrtSetDevice(0);
            aclrtStream stream = nullptr;
            aclrtCreateStream(&stream);

            auto cutStartTime = std::chrono::high_resolution_clock::now();
            COMMON::QueryResultSet<T> *queryResults = (COMMON::QueryResultSet<T> *)&p_query;

            // deduper.clear();
            COMMON::OptHashPosVector deduper;
            deduper.Init(m_options.m_maxCheck, m_options.m_hashExp);

            // 剪枝策略，根据searchInternalResultNum遍历queryResults, queryResults存放postinglistid和postinglist到query的距离
            std::vector<SizeType> postingIDs;
            float limitDist = queryResults->GetResult(0)->Dist * m_options.m_maxDistRatio; // 设置距离限制（最大距离为最近质心到查询向量距离的8倍）
            int ii = 0;
            for (; ii < m_options.m_searchInternalResultNum; ++ii) // 动态剪枝策略
            {
                auto res = queryResults->GetResult(ii);
                if (res->VID == -1 || (limitDist > 0.1 && res->Dist > limitDist))
                    break;
                if (numVecPerPostinglist[res->VID] != 0) // 将符合条件的质心加入到结果中
                {
                    postingIDs.emplace_back(res->VID);
                }
                res->VID = static_cast<SizeType>((m_vectorTranslateMap.get())[res->VID]);
                if (!m_options.m_enableReorderIndex)
                {
                    postingIDSet.insert(res->VID);
                }
                if (res->VID == MaxSize)
                {
                    res->VID = -1;
                    res->Dist = MaxDist;
                }
            }
            for (; ii < queryResults->GetResultNum(); ++ii)
            {
                auto res = queryResults->GetResult(ii);
                if (res->VID == -1)
                    break;
                res->VID = static_cast<SizeType>((m_vectorTranslateMap.get())[res->VID]);
                if (!m_options.m_enableReorderIndex)
                {
                    postingIDSet.insert(res->VID);
                }
                if (res->VID == MaxSize)
                {
                    res->VID = -1;
                    res->Dist = MaxDist;
                }
            }
            queryResults->Reverse();
            // queryResults->Reverse(max(m_options.m_resultNum, m_options.m_rerank));
            auto cutEndTime = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> cutelapsed = cutEndTime - cutStartTime;
            double cutelapsedMilliseconds = cutelapsed.count();
            p_stats->cutTreeLatency = cutelapsedMilliseconds;

            const SizeType postingListCount = static_cast<SizeType>(postingIDs.size());
            std::vector<SizeType> vectorIDs;

            for (int pi = 1; pi <= postingListCount; ++pi)
            {

                SizeType postingID = postingIDs[pi - 1];
                __builtin_prefetch(reinterpret_cast<const char*>(postinglist[postingID].get()), 0, 3);
                int numvec = numVecPerPostinglist[postingID];
                for (int vi = 0; vi < numvec; vi++)
                {
                    SizeType vectorID = postinglist[postingID][vi];
                    if (deduper.CheckAndSet(vectorID))
                        continue;
                    vectorIDs.push_back(vectorID);
                }
            }

            int totalNumVector = vectorIDs.size();
            int numVectorCPU = vectorIDs.size() * 0.5;
            int numVectorGPU = totalNumVector - numVectorCPU;

            uint8_t *h_table = reinterpret_cast<uint8_t *>(queryResults->GetnewPQTarget());
            int tableBytes = 256 * PQVectorDim * sizeof(float);
            uint8_t *table_temp = d_table + sizeof(uint8_t) * tableBytes * threadOrder;
            float *dist_temp = d_dist + totalNumVec * threadOrder;
            float *h_dist_temp = h_dist + totalNumVec * threadOrder;
            uint8_t *d_set;
            uint8_t *d_gather;
            uint8_t *d_cast;
            uint8_t *tmp = (uint8_t *)malloc(numVectorGPU * PQVectorDim);
            auto comStartTime1 = std::chrono::high_resolution_clock::now();
            aclrtMalloc((void **)&d_cast, numVectorGPU * PQVectorDim * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
            aclrtMalloc((void **)&d_gather, numVectorGPU * PQVectorDim * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
            aclrtMalloc((void **)&d_set, numVectorGPU * PQVectorDim * sizeof(uint8_t), ACL_MEM_MALLOC_HUGE_FIRST);
            aclrtMemcpy(table_temp, tableBytes, h_table, tableBytes, ACL_MEMCPY_HOST_TO_DEVICE);
            for (int i = 0; i < numVectorGPU; i ++)
                memcpy(tmp + i * PQVectorDim, vectorSet->GetVector(vectorIDs[i]), PQVectorDim);
            aclrtMemcpy(d_set, numVectorGPU * PQVectorDim, tmp, numVectorGPU * PQVectorDim, ACL_MEMCPY_HOST_TO_DEVICE);
            auto comEndTime1 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> comelapsed1 = comEndTime1 - comStartTime1;
            double comelapsedMilliseconds1 = comelapsed1.count();
            p_stats->cpu2npu = comelapsedMilliseconds1;
            auto comStartTime = std::chrono::high_resolution_clock::now();
            std::vector<int64_t> set_shape = {numVectorGPU, PQVectorDim};
            std::vector<int64_t> dist_shape = {numVectorGPU};
            std::vector<int64_t> table_shape = {256, PQVectorDim};
            std::vector<int64_t> dimsData = {1};
            bool keepDims = false;
            aclTensor *set_tensor = nullptr;
            aclTensor *dist_tensor = nullptr;
            aclTensor *gather_tensor = nullptr;
            aclTensor *cast_tensor = nullptr;
            aclTensor *table_tensor = nullptr;
            aclIntArray* dims = nullptr;
            dims = aclCreateIntArray(dimsData.data(), dimsData.size());
            CreateAclTensor(set_shape, (void **)&d_set, aclDataType::ACL_UINT8, &set_tensor);
            CreateAclTensor(dist_shape, (void **)&dist_temp, aclDataType::ACL_FLOAT, &dist_tensor);
            CreateAclTensor(set_shape, (void **)&d_gather, aclDataType::ACL_FLOAT, &gather_tensor);
            CreateAclTensor(set_shape, (void **)&d_cast, aclDataType::ACL_INT32, &cast_tensor);
            CreateTableAclTensor(table_shape, (void **)&table_temp, aclDataType::ACL_FLOAT, &table_tensor);
            void* workspaceAddr1 = nullptr;
            void* workspaceAddr2 = nullptr;
            void* workspaceAddr3 = nullptr;
            uint64_t workspaceSize1 = 0;
            aclOpExecutor* executor1;
            aclnnCastGetWorkspaceSize(set_tensor, aclDataType::ACL_INT32, cast_tensor, &workspaceSize1, &executor1);
            if (workspaceSize1 > 0)
                aclrtMalloc(&workspaceAddr1, workspaceSize1, ACL_MEM_MALLOC_HUGE_FIRST);
            aclnnCast(workspaceAddr1, workspaceSize1, executor1, stream);
            uint64_t workspaceSize2 = 0;
            aclOpExecutor* executor2;
            int64_t dim = 0;
            aclnnGatherGetWorkspaceSize(table_tensor, dim, cast_tensor, gather_tensor, &workspaceSize2, &executor2);
            if (workspaceSize2 > 0)
                aclrtMalloc(&workspaceAddr2, workspaceSize2, ACL_MEM_MALLOC_HUGE_FIRST);
            aclnnGather(workspaceAddr2, workspaceSize2, executor2, stream);
            uint64_t workspaceSize3 = 0;
            aclOpExecutor* executor3;
            aclnnReduceSumGetWorkspaceSize(gather_tensor, dims, keepDims, aclDataType::ACL_FLOAT, dist_tensor, &workspaceSize3, &executor3);
            if (workspaceSize3 > 0)
                aclrtMalloc(&workspaceAddr3, workspaceSize3, ACL_MEM_MALLOC_HUGE_FIRST);
            aclnnReduceSum(workspaceAddr3, workspaceSize3, executor3, stream);
            if (numVectorCPU > 0)
            {
                std::vector<uint8_t> pq_ids(numVectorCPU * PQVectorDim);
                std::vector<float> dists_out(numVectorCPU);
                const float *pq_dists = reinterpret_cast<const float *>(h_table);
                int idx = 0;
                for (size_t i = numVectorGPU; i < totalNumVector; i++)
                {
                    const uint8_t *vector = reinterpret_cast<const uint8_t *>(vectorSet->GetVector(vectorIDs[i]));
                    memcpy(pq_ids.data() + idx * PQVectorDim, vector, PQVectorDim);
                    idx ++;
                }
                for (size_t i = 0; i < numVectorCPU; i++)
                {
                    dists_out[i] = L2Distance_Dmul4(h_table, pq_ids.data() + PQVectorDim * i, PQVectorDim);
                }
                for (size_t i = numVectorGPU; i < totalNumVector; i++)
                    queryResults->AddPoint(vectorIDs[i], dists_out[i - numVectorGPU]);
            }
            aclrtSynchronizeStream(stream);
            auto comEndTime = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> comelapsed = comEndTime - comStartTime;
            double comelapsedMilliseconds = comelapsed.count();
            p_stats->vectorLatency0 = comelapsedMilliseconds;
            auto comStartTime2 = std::chrono::high_resolution_clock::now();
            aclrtMemcpy(h_dist_temp, sizeof(float) * numVectorGPU, dist_temp, sizeof(float) * numVectorGPU, ACL_MEMCPY_DEVICE_TO_HOST);
            auto comEndTime2 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> comelapsed2 = comEndTime2 - comStartTime2;
            double comelapsedMilliseconds2 = comelapsed2.count();
            p_stats->npu2cpu = comelapsedMilliseconds2;
            aclDestroyTensor(set_tensor);
            aclDestroyTensor(dist_tensor);
            aclDestroyTensor(gather_tensor);
            aclDestroyTensor(cast_tensor);
            aclDestroyTensor(table_tensor);
            aclDestroyIntArray(dims);
            // free(tmp);
            aclrtFree(d_cast);
            aclrtFree(d_gather);
            aclrtFree(d_set);
            if (workspaceSize1 > 0)
                aclrtFree(workspaceAddr1);
            if (workspaceSize2 > 0)
                aclrtFree(workspaceAddr2);
            if (workspaceSize3 > 0)
                aclrtFree(workspaceAddr3);
            aclrtDestroyStream(stream);
            aclrtResetDevice(0);
            aclFinalize();
            for (int i = 0; i < numVectorGPU; i++)
                queryResults->AddPoint(vectorIDs[i], h_dist_temp[i]);
            queryResults->SortResult();
            return ErrorCode::Success;
        }

        /*
            tyh: 批量异步读方案
        */
        template <typename T>
        ErrorCode Index<T>::RerankFullVector(QueryResult &p_query, std::shared_ptr<VectorSet> vectorSet, int threadOrder, std::unordered_set<int>& postingIDSet, SearchStats* p_stats) const
        {
            COMMON::QueryResultSet<T> *queryResults = (COMMON::QueryResultSet<T> *)&p_query;
            const T* targetVector = reinterpret_cast<const T *>(queryResults->GetTarget());
            std::vector<Helper::AsyncReadRequest> diskRequests;
            // std::vector<char*> buffers;

            Helper::AsyncFileIO* handler = (Helper::AsyncFileIO* )(m_indexFiles[0].get());

            int pageCount = 0;
            int* pageCountref = &pageCount;
            for (int j = 0; j < m_options.m_resultNum; j++) {
                auto result = queryResults->GetResult(j);
                if (result->VID < 0 || postingIDSet.find(result->VID) != postingIDSet.end()) continue;

                if (result->VID < 1000000000 * m_options.m_readRatio) {
                    result->Dist = COMMON::DistanceUtils::ComputeDistance(targetVector, reinterpret_cast<const T *>(vectorSet->GetVector(result->VID)), m_options.m_dim, m_options.m_distCalcMethod);
                } else {
                    size_t alignedOffset = (static_cast<size_t>(result->VID) * m_options.m_dim * sizeof(T) + 2 * sizeof(int)) & ~(m_pageSize - 1);
                    size_t endOffset = static_cast<size_t>(result->VID) * m_options.m_dim * sizeof(T) + 2 * sizeof(int) + m_options.m_dim * sizeof(T);
                    size_t alignedEndOffset = (endOffset + m_pageSize - 1) & ~(m_pageSize - 1);
                    size_t readSize = alignedEndOffset - alignedOffset;

                    char* buffer_ptr = nullptr;
                    posix_memalign((void** )&buffer_ptr, m_pageSize, readSize);
                    memset(buffer_ptr, 0, readSize);
                    // buffers.push_back(buffer_ptr);

                    Helper::AsyncReadRequest request;
                    request.m_offset = alignedOffset;
                    request.m_readSize = readSize;
                    request.m_buffer = buffer_ptr;
                    request.m_status = threadOrder;
                    request.m_success = false;

                    request.m_callback = [queryResults, j, buffer_ptr, alignedOffset, targetVector, pageCountref, readSize, this](bool success)
                    {
                        *pageCountref += readSize;
                        SPTAG::BasicResult* result = queryResults->GetResult(j);
                        const size_t dataOffset = static_cast<size_t>(result->VID) * m_options.m_dim * sizeof(T) + 2 * sizeof(int) - alignedOffset;
                        const T* queryVector = reinterpret_cast<const T*>(buffer_ptr + dataOffset);
                        result->Dist = COMMON::DistanceUtils::ComputeDistance(targetVector, queryVector, m_options.m_dim, m_options.m_distCalcMethod);
                    };
                    diskRequests.push_back(std::move(request));  // Store the request
                }
            }

            size_t num = diskRequests.size();
            struct timespec AIOTimeout = {0, 30000};
            std::vector<struct iocb> myiocbs(num);
            std::vector<struct iocb*> iocbs;
            int submitted = 0;
            int done = 0;
            int totalToSubmit = 0, channel = 0;

            memset(myiocbs.data(), 0, num * sizeof(struct iocb));
            for (int i = 0; i < num; i++) {
                Helper::AsyncReadRequest* readRequest = &(diskRequests[i]);

                channel = readRequest->m_status & 0xffff;

                struct iocb* myiocb = &(myiocbs[totalToSubmit++]);
                myiocb->aio_data = reinterpret_cast<uintptr_t>(readRequest);
                myiocb->aio_lio_opcode = IOCB_CMD_PREAD;
                myiocb->aio_fildes = handler->GetFileHandler();
                myiocb->aio_buf = (std::uint64_t)(readRequest->m_buffer);
                myiocb->aio_nbytes = readRequest->m_readSize;
                myiocb->aio_offset = static_cast<std::int64_t>(readRequest->m_offset);

                iocbs.emplace_back(myiocb);
            }

            if(m_options.m_rerank_batch){
                int batchSize = m_options.m_rerank_batchSize;
                int heapExpandSize = m_options.m_rerank + 8;
                int changeNum = 0;
                std::vector<struct io_event> events(totalToSubmit);
                queryResults->initialTopK(heapExpandSize);
                while (done < totalToSubmit)
                {
                    int totalDone = 0, totalSubmitted = 0, totalQueued = 0;
                    int batchStartIdx = done;
                    int batchEndIdx = std::min(done + batchSize, totalToSubmit); // 计算当前批次的结束位置
                    int totalBatchSize = batchEndIdx - batchStartIdx;
                    while (totalDone < totalBatchSize) {
                        if (totalSubmitted < totalBatchSize) {
                            if (submitted < iocbs.size()) {
                                int s = syscall(__NR_io_submit, handler->GetIOCP(channel), totalBatchSize, iocbs.data() + batchStartIdx);
                                if (s > 0) {
                                    submitted += s;
                                    totalSubmitted += s;
                                }
                                else {
                                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "to submit:%d, submitted:%s\n", totalBatchSize, strerror(-s));
                                }
                            }
                        }

                        for (int i = totalQueued; i < totalDone; i++) {
                            Helper::AsyncReadRequest* req = reinterpret_cast<Helper::AsyncReadRequest*>((events[i].data));
                            if (nullptr != req)
                            {
                                req->m_callback(true);
                            }
                        }
                        totalQueued = totalDone;

                        if (done < submitted) {
                            int wait = submitted - done;
                            auto d = syscall(__NR_io_getevents, handler->GetIOCP(channel), wait, wait, events.data() + totalDone, &AIOTimeout);
                            done += d;
                            totalDone += d;
                        }
                    }

                    for (int i = totalQueued; i < totalDone; i++) {
                        Helper::AsyncReadRequest* req = reinterpret_cast<Helper::AsyncReadRequest*>((events[i].data));
                        if (nullptr != req)
                        {
                            req->m_callback(true);
                        }
                    }

                    float changeRate = 1.0 * queryResults->IsHeapUnchanged(heapExpandSize) / (2 * heapExpandSize);
                    if(changeRate == 0){
                        changeNum ++;
                        if (changeNum > 1)  break;                        
                    }else{
                        changeNum = 0;
                    }
                    float adjustmentFactor = (changeRate > 0.25) ? changeRate : -changeRate;
                    batchSize *= (1 + adjustmentFactor);

                    // auto isHeapUnchanged = queryResults->IsHeapUnchanged1(heapExpandSize);
                    // if(isHeapUnchanged){
                    //     changeNum ++;
                    //     if (changeNum > 1)  break;                        
                    // }else{
                    //     changeNum = 0;
                    // }
                }
            }else{
                 std::vector<struct io_event> events(totalToSubmit);
                int totalDone = 0, totalSubmitted = 0, totalQueued = 0;
                while (totalDone < totalToSubmit) {
                    if (totalSubmitted < totalToSubmit) {
                        if (submitted < iocbs.size()) {
                            int s = syscall(__NR_io_submit, handler->GetIOCP(channel), iocbs.size() - submitted, iocbs.data() + submitted);
                            if (s > 0) {
                                submitted += s;
                                totalSubmitted += s;
                            }
                            else {
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "to submit:%d, submitted:%s\n", iocbs.size() - submitted, strerror(-s));
                            }
                        }
                    }

                    for (int i = totalQueued; i < totalDone; i++) {
                        Helper::AsyncReadRequest* req = reinterpret_cast<Helper::AsyncReadRequest*>((events[i].data));
                        if (nullptr != req)
                        {
                            req->m_callback(true);
                        }
                    }
                    totalQueued = totalDone;

                    if (done < submitted) {
                        int wait = submitted - done;
                        auto d = syscall(__NR_io_getevents, handler->GetIOCP(channel), wait, wait, events.data() + totalDone, &AIOTimeout);
                        done += d;
                        totalDone += d;
                    }
                }

                for (int i = totalQueued; i < totalDone; i++) {
                    Helper::AsyncReadRequest* req = reinterpret_cast<Helper::AsyncReadRequest*>((events[i].data));
                    if (nullptr != req)
                    {
                        req->m_callback(true);
                    }
                }
            }

            p_stats->m_diskAccessCount = pageCount / m_pageSize;
            p_stats->m_diskIOCount = done;
            SPTAG::BasicResult* re = queryResults->GetResults();
            std::sort(re, re + m_options.m_searchInternalResultNum, COMMON::Compare);
            return ErrorCode::Success;
        }
        
        /*使用优化存储布局合并I/O请求*/
        template <typename T>
        ErrorCode Index<T>::RerankFullVectorFusion(QueryResult &p_query, std::shared_ptr<VectorSet> vectorSet, int threadOrder, SearchStats* p_stats) const
        {
            COMMON::QueryResultSet<T> *queryResults = (COMMON::QueryResultSet<T> *)&p_query;
            const T* targetVector = reinterpret_cast<const T *>(queryResults->GetTarget());
            std::unordered_map<int64_t, Helper::AsyncReadRequest> uniqueRequests;

            Helper::AsyncFileIO* handler = (Helper::AsyncFileIO* )(m_indexFiles[0].get());

            int pageCount = 0;
            int* pageCountref = &pageCount;
            for (int j = 0; j < m_options.m_resultNum; j++) {
                auto result = queryResults->GetResult(j);
                if (result->VID < 0 || m_vectorMapPosting[result->VID] < 0)
                {
                    continue;
                }

                if (result->VID < m_totalDocumentCount * m_options.m_readRatio) {
                    result->Dist = COMMON::DistanceUtils::ComputeDistance(targetVector, reinterpret_cast<const T *>(vectorSet->GetVector(result->VID)), m_options.m_dim, m_options.m_distCalcMethod);
                }else{
                    const ListInfo* listInfo = &(m_listInfos[m_vectorMapPosting[result->VID]]);
                    int64_t listOffset = listInfo->listOffset;
                    int readSize = listInfo->listPageCount * m_pageSize;

                    if (uniqueRequests.find(m_vectorMapPosting[result->VID]) == uniqueRequests.end()) {
                        // Create a new read request
                        char* buffer_ptr = nullptr;
                        posix_memalign((void** )&buffer_ptr, m_pageSize, readSize);
                        memset(buffer_ptr, 0, readSize);

                        Helper::AsyncReadRequest request;
                        request.m_offset = listOffset;
                        request.m_readSize = readSize;
                        request.m_buffer = buffer_ptr;
                        request.m_status = threadOrder;
                        request.m_success = false;

                        request.m_callback = [request, queryResults, targetVector, listInfo, pageCountref, this](bool success)
                        {
                            char* postingListFullData = request.m_buffer + listInfo->pageOffset;
                            *pageCountref += listInfo->listPageCount;
                            for (size_t i = 0; i < listInfo->listEleCount; i++)
                            {
                                int vectorID = *(reinterpret_cast<int*>(postingListFullData + m_vectorInfoSize * i));
                                for (int j = 0; j < queryResults->GetResultNum(); j++) {
                                    auto result = queryResults->GetResult(j);
                                    if (vectorID == result->VID) {
                                        const T* queryVector = reinterpret_cast<const T*>(postingListFullData + m_vectorInfoSize * i + sizeof(int));
                                        result->Dist = COMMON::DistanceUtils::ComputeDistance(targetVector, queryVector, m_options.m_dim, m_options.m_distCalcMethod);
                                    }
                                }
                            }
                        };

                        uniqueRequests[m_vectorMapPosting[result->VID]] = std::move(request);
                    }
                }
            }

            std::vector<Helper::AsyncReadRequest> diskRequests;
            for (auto& pair : uniqueRequests) {
                diskRequests.push_back(std::move(pair.second));
            }

            size_t num = diskRequests.size();
            struct timespec AIOTimeout = {0, 30000};
            std::vector<struct iocb> myiocbs(num);
            std::vector<struct iocb*> iocbs;
            int submitted = 0;
            int done = 0;
            int totalToSubmit = 0, channel = 0;

            memset(myiocbs.data(), 0, num * sizeof(struct iocb));
            for (int i = 0; i < num; i++) {
                Helper::AsyncReadRequest* readRequest = &(diskRequests[i]);

                channel = readRequest->m_status & 0xffff;

                struct iocb* myiocb = &(myiocbs[totalToSubmit++]);
                myiocb->aio_data = reinterpret_cast<uintptr_t>(readRequest);
                myiocb->aio_lio_opcode = IOCB_CMD_PREAD;
                myiocb->aio_fildes = handler->GetFileHandler();
                myiocb->aio_buf = (std::uint64_t)(readRequest->m_buffer);
                myiocb->aio_nbytes = readRequest->m_readSize;
                myiocb->aio_offset = static_cast<std::int64_t>(readRequest->m_offset);

                iocbs.emplace_back(myiocb);
            }

            if(m_options.m_rerank_batch){
                int batchSize = totalToSubmit / m_options.m_rerank_batchSize;
                // int heapExpandSize = m_options.m_rerank + 8;
                int heapExpandSize = m_options.m_rerank + 8;
                int changeNum = 0;
                std::vector<struct io_event> events(totalToSubmit);
                queryResults->initialTopK(heapExpandSize);
                while (done < totalToSubmit)
                {
                    int totalDone = 0, totalSubmitted = 0, totalQueued = 0;
                    int batchStartIdx = done;
                    int batchEndIdx = std::min(done + batchSize, totalToSubmit); // 计算当前批次的结束位置
                    int totalBatchSize = batchEndIdx - batchStartIdx;
                    while (totalDone < totalBatchSize) {
                        if (totalSubmitted < totalBatchSize) {
                            if (submitted < iocbs.size()) {
                                int s = syscall(__NR_io_submit, handler->GetIOCP(channel), totalBatchSize, iocbs.data() + batchStartIdx);
                                if (s > 0) {
                                    submitted += s;
                                    totalSubmitted += s;
                                }
                                else {
                                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "to submit:%d, submitted:%s\n", totalBatchSize, strerror(-s));
                                }
                            }
                        }

                        for (int i = totalQueued; i < totalDone; i++) {
                            Helper::AsyncReadRequest* req = reinterpret_cast<Helper::AsyncReadRequest*>((events[i].data));
                            if (nullptr != req)
                            {
                                req->m_callback(true);
                            }
                        }
                        totalQueued = totalDone;

                        if (done < submitted) {
                            int wait = submitted - done;
                            auto d = syscall(__NR_io_getevents, handler->GetIOCP(channel), wait, wait, events.data() + totalDone, &AIOTimeout);
                            done += d;
                            totalDone += d;
                        }
                    }

                    for (int i = totalQueued; i < totalDone; i++) {
                        Helper::AsyncReadRequest* req = reinterpret_cast<Helper::AsyncReadRequest*>((events[i].data));
                        if (nullptr != req)
                        {
                            req->m_callback(true);
                        }
                    }

                    float changeRate = 1.0 * queryResults->CountHeapChanges(heapExpandSize) / heapExpandSize;
                    if(changeRate <= m_options.m_changeThreshold){
                        changeNum ++;
                        if (changeNum > 1)  break;                        
                    }else{
                        changeNum = 0;
                    }
                    // float adjustmentFactor = (changeRate > 0.25) ? changeRate : -changeRate;
                    // batchSize *= (1 + adjustmentFactor);

                    // auto isHeapUnchanged = queryResults->IsHeapUnchanged1(heapExpandSize);
                    // if(isHeapUnchanged){
                    //     changeNum ++;
                    //     if (changeNum > 1)  break;                        
                    // }else{
                    //     changeNum = 0;
                    // }
                }
            }else{
                 std::vector<struct io_event> events(totalToSubmit);
                int totalDone = 0, totalSubmitted = 0, totalQueued = 0;
                while (totalDone < totalToSubmit) {
                    if (totalSubmitted < totalToSubmit) {
                        if (submitted < iocbs.size()) {
                            int s = syscall(__NR_io_submit, handler->GetIOCP(channel), iocbs.size() - submitted, iocbs.data() + submitted);
                            if (s > 0) {
                                submitted += s;
                                totalSubmitted += s;
                            }
                            else {
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "to submit:%d, submitted:%s\n", iocbs.size() - submitted, strerror(-s));
                            }
                        }
                    }

                    for (int i = totalQueued; i < totalDone; i++) {
                        Helper::AsyncReadRequest* req = reinterpret_cast<Helper::AsyncReadRequest*>((events[i].data));
                        if (nullptr != req)
                        {
                            req->m_callback(true);
                        }
                    }
                    totalQueued = totalDone;

                    if (done < submitted) {
                        int wait = submitted - done;
                        auto d = syscall(__NR_io_getevents, handler->GetIOCP(channel), wait, wait, events.data() + totalDone, &AIOTimeout);
                        done += d;
                        totalDone += d;
                    }
                }

                for (int i = totalQueued; i < totalDone; i++) {
                    Helper::AsyncReadRequest* req = reinterpret_cast<Helper::AsyncReadRequest*>((events[i].data));
                    if (nullptr != req)
                    {
                        req->m_callback(true);
                    }
                }
            }

            p_stats->m_diskAccessCount = pageCount;
            p_stats->m_diskIOCount = done;
            SPTAG::BasicResult* re = queryResults->GetResults();
            std::sort(re, re + m_options.m_searchInternalResultNum, COMMON::Compare);
            return ErrorCode::Success;
        }

        /*
        template <typename T>
        ErrorCode Index<T>::RerankFullVectorFusion(QueryResult &p_query, std::shared_ptr<VectorSet> vectorSet, int threadOrder, SearchStats* p_stats) const
        {
            COMMON::QueryResultSet<T> *queryResults = (COMMON::QueryResultSet<T> *)&p_query;
            const T* targetVector = reinterpret_cast<const T *>(queryResults->GetTarget());
            std::vector<Helper::AsyncReadRequest> diskRequests;

            Helper::AsyncFileIO* handler = (Helper::AsyncFileIO* )(m_indexFiles[0].get());

            for (int j = 0; j < m_options.m_resultNum; j++) {
                auto result = queryResults->GetResult(j);
                if (result->VID < 0 || m_vectorMapPosting[result->VID] < 0)
                {
                    continue;
                }

                const ListInfo* listInfo = &(m_listInfos[m_vectorMapPosting[result->VID]]);

                char* buffer_ptr = nullptr;
                posix_memalign((void** )&buffer_ptr, m_pageSize, m_pageSize);
                memset(buffer_ptr, 0, m_pageSize);

                Helper::AsyncReadRequest request;
                request.m_offset = listInfo->listOffset;
                request.m_readSize = m_pageSize;
                request.m_buffer = buffer_ptr;
                request.m_status = threadOrder;
                request.m_success = false;

                request.m_callback = [queryResults, j, buffer_ptr, targetVector, listInfo, this](bool success)
                {
                    SPTAG::BasicResult* result = queryResults->GetResult(j);
                    char* postingListFullData = buffer_ptr + listInfo->pageOffset;
                    for (size_t i = 0; i < listInfo->listEleCount; i++)
                    {
                        int vectorID = *(reinterpret_cast<int*>(postingListFullData + m_vectorInfoSize * i));
                        if(vectorID == result->VID)
                        {
                            const T* queryVector = reinterpret_cast<const T*>(postingListFullData + m_vectorInfoSize * i + sizeof(int));
                            result->Dist = COMMON::DistanceUtils::ComputeDistance(targetVector, queryVector, m_options.m_dim, m_options.m_distCalcMethod);
                        }
                    }
                };
                diskRequests.push_back(std::move(request));
            }

            size_t num = diskRequests.size();
            struct timespec AIOTimeout = {0, 30000};
            std::vector<struct iocb> myiocbs(num);
            std::vector<struct iocb*> iocbs;
            int submitted = 0;
            int done = 0;
            int totalToSubmit = 0, channel = 0;

            memset(myiocbs.data(), 0, num * sizeof(struct iocb));
            for (int i = 0; i < num; i++) {
                Helper::AsyncReadRequest* readRequest = &(diskRequests[i]);

                channel = readRequest->m_status & 0xffff;

                struct iocb* myiocb = &(myiocbs[totalToSubmit++]);
                myiocb->aio_data = reinterpret_cast<uintptr_t>(readRequest);
                myiocb->aio_lio_opcode = IOCB_CMD_PREAD;
                myiocb->aio_fildes = handler->GetFileHandler();
                myiocb->aio_buf = (std::uint64_t)(readRequest->m_buffer);
                myiocb->aio_nbytes = readRequest->m_readSize;
                myiocb->aio_offset = static_cast<std::int64_t>(readRequest->m_offset);

                iocbs.emplace_back(myiocb);
            }

            std::vector<struct io_event> events(totalToSubmit);
            int totalDone = 0, totalSubmitted = 0, totalQueued = 0;
            while (totalDone < totalToSubmit) {
                if (totalSubmitted < totalToSubmit) {
                    if (submitted < iocbs.size()) {
                        int s = syscall(__NR_io_submit, handler->GetIOCP(channel), iocbs.size() - submitted, iocbs.data() + submitted);
                        if (s > 0) {
                            submitted += s;
                            totalSubmitted += s;
                        }
                        else {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "to submit:%d, submitted:%s\n", iocbs.size() - submitted, strerror(-s));
                        }
                    }
                }

                for (int i = totalQueued; i < totalDone; i++) {
                    Helper::AsyncReadRequest* req = reinterpret_cast<Helper::AsyncReadRequest*>((events[i].data));
                    if (nullptr != req)
                    {
                        req->m_callback(true);
                    }
                }
                totalQueued = totalDone;

                if (done < submitted) {
                    int wait = submitted - done;
                    auto d = syscall(__NR_io_getevents, handler->GetIOCP(channel), wait, wait, events.data() + totalDone, &AIOTimeout);
                    done += d;
                    totalDone += d;
                }
            }

            for (int i = totalQueued; i < totalDone; i++) {
                Helper::AsyncReadRequest* req = reinterpret_cast<Helper::AsyncReadRequest*>((events[i].data));
                if (nullptr != req)
                {
                    req->m_callback(true);
                }
            }
            p_stats->m_diskIOCount = done;
            SPTAG::BasicResult* re = queryResults->GetResults();
            std::sort(re, re + m_options.m_searchInternalResultNum, COMMON::Compare);
            return ErrorCode::Success;
        }
        */
        
        template <typename T>
        ErrorCode Index<T>::DebugSearchDiskIndex(QueryResult &p_query, int p_subInternalResultNum, int p_internalResultNum,
                                                 SearchStats *p_stats, std::set<int> *truth, std::map<int, std::set<int>> *found) const
        {
            if (nullptr == m_extraSearcher)
                return ErrorCode::EmptyIndex;

            COMMON::QueryResultSet<T> newResults(*((COMMON::QueryResultSet<T> *)&p_query));
            for (int i = 0; i < newResults.GetResultNum(); ++i)
            {
                auto res = newResults.GetResult(i);
                if (res->VID == -1)
                    break;

                auto global_VID = static_cast<SizeType>((m_vectorTranslateMap.get())[res->VID]);
                if (truth && truth->count(global_VID))
                    (*found)[res->VID].insert(global_VID);
                res->VID = global_VID;
                if (res->VID == MaxSize)
                {
                    res->VID = -1;
                    res->Dist = MaxDist;
                }
            }
            newResults.Reverse();

            auto workSpace = m_workSpaceFactory->GetWorkSpace();
            if (!workSpace)
            {
                workSpace.reset(new ExtraWorkSpace());
                workSpace->Initialize(m_options.m_maxCheck, m_options.m_hashExp, m_options.m_searchInternalResultNum, max(m_options.m_postingPageLimit, m_options.m_searchPostingPageLimit + 1) << PageSizeEx, m_options.m_enableDataCompression);
            }
            else
            {
                workSpace->Clear(m_options.m_searchInternalResultNum, max(m_options.m_postingPageLimit, m_options.m_searchPostingPageLimit + 1) << PageSizeEx, m_options.m_enableDataCompression);
            }
            workSpace->m_deduper.clear();

            int partitions = (p_internalResultNum + p_subInternalResultNum - 1) / p_subInternalResultNum;
            float limitDist = p_query.GetResult(0)->Dist * m_options.m_maxDistRatio;
            for (SizeType p = 0; p < partitions; p++)
            {
                int subInternalResultNum = min(p_subInternalResultNum, p_internalResultNum - p_subInternalResultNum * p);

                workSpace->m_postingIDs.clear();

                for (int i = p * p_subInternalResultNum; i < p * p_subInternalResultNum + subInternalResultNum; i++)
                {
                    auto res = p_query.GetResult(i);
                    if (res->VID == -1 || (limitDist > 0.1 && res->Dist > limitDist))
                        break;
                    if (!m_extraSearcher->CheckValidPosting(res->VID))
                        continue;
                    workSpace->m_postingIDs.emplace_back(res->VID);
                }

                m_extraSearcher->SearchIndex(workSpace.get(), newResults, m_index, p_stats, truth, found);
            }
            m_workSpaceFactory->ReturnWorkSpace(std::move(workSpace));
            newResults.SortResult();
            std::copy(newResults.GetResults(), newResults.GetResults() + newResults.GetResultNum(), p_query.GetResults());
            return ErrorCode::Success;
        }
#pragma endregion

        template <typename T>
        ErrorCode Index<T>::GetPostingDebug(SizeType vid, std::vector<SizeType> &VIDs, std::shared_ptr<VectorSet> &vecs)
        {
            VIDs.clear();
            if (!m_extraSearcher)
                return ErrorCode::EmptyIndex;
            if (!m_extraSearcher->CheckValidPosting(vid))
                return ErrorCode::Fail;

            auto workSpace = m_workSpaceFactory->GetWorkSpace();
            if (!workSpace)
            {
                workSpace.reset(new ExtraWorkSpace());
                workSpace->Initialize(m_options.m_maxCheck, m_options.m_hashExp, m_options.m_searchInternalResultNum, max(m_options.m_postingPageLimit, m_options.m_searchPostingPageLimit + 1) << PageSizeEx, m_options.m_enableDataCompression);
            }
            else
            {
                workSpace->Clear(m_options.m_searchInternalResultNum, max(m_options.m_postingPageLimit, m_options.m_searchPostingPageLimit + 1) << PageSizeEx, m_options.m_enableDataCompression);
            }
            workSpace->m_deduper.clear();

            auto out = m_extraSearcher->GetPostingDebug(workSpace.get(), m_index, vid, VIDs, vecs);
            m_workSpaceFactory->ReturnWorkSpace(std::move(workSpace));
            return out;
        }

        template <typename T>
        void Index<T>::SelectHeadAdjustOptions(int p_vectorCount)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Begin Adjust Parameters...\n");

            if (m_options.m_headVectorCount != 0)
                m_options.m_ratio = m_options.m_headVectorCount * 1.0 / p_vectorCount;
            int headCnt = static_cast<int>(std::round(m_options.m_ratio * p_vectorCount));
            if (headCnt == 0)
            {
                for (double minCnt = 1; headCnt == 0; minCnt += 0.2)
                {
                    m_options.m_ratio = minCnt / p_vectorCount;
                    headCnt = static_cast<int>(std::round(m_options.m_ratio * p_vectorCount));
                }

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Setting requires to select none vectors as head, adjusted it to %d vectors\n", headCnt);
            }

            if (m_options.m_iBKTKmeansK > headCnt)
            {
                m_options.m_iBKTKmeansK = headCnt;
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Setting of cluster number is less than head count, adjust it to %d\n", headCnt);
            }

            if (m_options.m_selectThreshold == 0)
            {
                m_options.m_selectThreshold = min(p_vectorCount - 1, static_cast<int>(1 / m_options.m_ratio));
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Set SelectThreshold to %d\n", m_options.m_selectThreshold);
            }

            if (m_options.m_splitThreshold == 0)
            {
                m_options.m_splitThreshold = min(p_vectorCount - 1, static_cast<int>(m_options.m_selectThreshold * 2));
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Set SplitThreshold to %d\n", m_options.m_splitThreshold);
            }

            if (m_options.m_splitFactor == 0)
            {
                m_options.m_splitFactor = min(p_vectorCount - 1, static_cast<int>(std::round(1 / m_options.m_ratio) + 0.5));
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Set SplitFactor to %d\n", m_options.m_splitFactor);
            }
        }

        template <typename T>
        int Index<T>::SelectHeadDynamicallyInternal(const std::shared_ptr<COMMON::BKTree> p_tree, int p_nodeID,
                                                    const Options &p_opts, std::vector<int> &p_selected)
        {
            typedef std::pair<int, int> CSPair;
            std::vector<CSPair> children;
            int childrenSize = 1;
            const auto &node = (*p_tree)[p_nodeID];
            if (node.childStart >= 0)
            {
                children.reserve(node.childEnd - node.childStart);
                for (int i = node.childStart; i < node.childEnd; ++i)
                {
                    int cs = SelectHeadDynamicallyInternal(p_tree, i, p_opts, p_selected);
                    if (cs > 0)
                    {
                        children.emplace_back(i, cs);
                        childrenSize += cs;
                    }
                }
            }

            if (childrenSize >= p_opts.m_selectThreshold)
            {
                if (node.centerid < (*p_tree)[0].centerid)
                {
                    p_selected.push_back(node.centerid);
                }

                if (childrenSize > p_opts.m_splitThreshold)
                {
                    std::sort(children.begin(), children.end(), [](const CSPair &a, const CSPair &b)
                              { return a.second > b.second; });

                    size_t selectCnt = static_cast<size_t>(std::ceil(childrenSize * 1.0 / p_opts.m_splitFactor) + 0.5);
                    // if (selectCnt > 1) selectCnt -= 1;
                    for (size_t i = 0; i < selectCnt && i < children.size(); ++i)
                    {
                        p_selected.push_back((*p_tree)[children[i].first].centerid);
                    }
                }

                return 0;
            }

            return childrenSize;
        }

        template <typename T>
        void Index<T>::SelectHeadDynamically(const std::shared_ptr<COMMON::BKTree> p_tree, int p_vectorCount, std::vector<int> &p_selected)
        {
            p_selected.clear();
            p_selected.reserve(p_vectorCount);

            if (static_cast<int>(std::round(m_options.m_ratio * p_vectorCount)) >= p_vectorCount)
            {
                for (int i = 0; i < p_vectorCount; ++i)
                {
                    p_selected.push_back(i);
                }

                return;
            }
            Options opts = m_options;

            int selectThreshold = m_options.m_selectThreshold;
            int splitThreshold = m_options.m_splitThreshold;

            double minDiff = 100;
            for (int select = 2; select <= m_options.m_selectThreshold; ++select)
            {
                opts.m_selectThreshold = select;
                opts.m_splitThreshold = m_options.m_splitThreshold;

                int l = m_options.m_splitFactor;
                int r = m_options.m_splitThreshold;

                while (l < r - 1)
                {
                    opts.m_splitThreshold = (l + r) / 2;
                    p_selected.clear();

                    SelectHeadDynamicallyInternal(p_tree, 0, opts, p_selected);
                    std::sort(p_selected.begin(), p_selected.end());
                    p_selected.erase(std::unique(p_selected.begin(), p_selected.end()), p_selected.end());

                    double diff = static_cast<double>(p_selected.size()) / p_vectorCount - m_options.m_ratio;

                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                                 "Select Threshold: %d, Split Threshold: %d, diff: %.2lf%%.\n",
                                 opts.m_selectThreshold,
                                 opts.m_splitThreshold,
                                 diff * 100.0);

                    if (minDiff > fabs(diff))
                    {
                        minDiff = fabs(diff);

                        selectThreshold = opts.m_selectThreshold;
                        splitThreshold = opts.m_splitThreshold;
                    }

                    if (diff > 0)
                    {
                        l = (l + r) / 2;
                    }
                    else
                    {
                        r = (l + r) / 2;
                    }
                }
            }

            opts.m_selectThreshold = selectThreshold;
            opts.m_splitThreshold = splitThreshold;

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                         "Final Select Threshold: %d, Split Threshold: %d.\n",
                         opts.m_selectThreshold,
                         opts.m_splitThreshold);

            p_selected.clear();
            SelectHeadDynamicallyInternal(p_tree, 0, opts, p_selected);
            std::sort(p_selected.begin(), p_selected.end());
            p_selected.erase(std::unique(p_selected.begin(), p_selected.end()), p_selected.end());
        }

        template <typename T>
        template <typename InternalDataType>
        bool Index<T>::SelectHeadInternal(std::shared_ptr<Helper::VectorSetReader> &p_reader)
        {
            std::shared_ptr<VectorSet> vectorset = p_reader->GetVectorSet();
            if (m_options.m_distCalcMethod == DistCalcMethod::Cosine && !p_reader->IsNormalized())
                vectorset->Normalize(m_options.m_iSelectHeadNumberOfThreads);
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Begin initial data (%d,%d)...\n", vectorset->Count(), vectorset->Dimension());

            COMMON::Dataset<InternalDataType> data(vectorset->Count(), vectorset->Dimension(), vectorset->Count(), vectorset->Count() + 1, (InternalDataType *)vectorset->GetData());

            auto t1 = std::chrono::high_resolution_clock::now();
            SelectHeadAdjustOptions(data.R());
            std::vector<int> selected;
            if (data.R() == 1)
            {
                selected.push_back(0);
            }
            else if (Helper::StrUtils::StrEqualIgnoreCase(m_options.m_selectType.c_str(), "Random"))
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Start generating Random head.\n");
                selected.resize(data.R());
                for (int i = 0; i < data.R(); i++)
                    selected[i] = i;
                std::shuffle(selected.begin(), selected.end(), rg);
                int headCnt = static_cast<int>(std::round(m_options.m_ratio * data.R()));
                selected.resize(headCnt);
            }
            else if (Helper::StrUtils::StrEqualIgnoreCase(m_options.m_selectType.c_str(), "BKT"))
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Start generating BKT.\n");
                std::shared_ptr<COMMON::BKTree> bkt = std::make_shared<COMMON::BKTree>();
                bkt->m_iBKTKmeansK = m_options.m_iBKTKmeansK;
                bkt->m_iBKTLeafSize = m_options.m_iBKTLeafSize;
                bkt->m_iSamples = m_options.m_iSamples;
                bkt->m_iTreeNumber = m_options.m_iTreeNumber;
                bkt->m_fBalanceFactor = m_options.m_fBalanceFactor;
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Start invoking BuildTrees.\n");
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "BKTKmeansK: %d, BKTLeafSize: %d, Samples: %d, BKTLambdaFactor:%f TreeNumber: %d, ThreadNum: %d.\n",
                             bkt->m_iBKTKmeansK, bkt->m_iBKTLeafSize, bkt->m_iSamples, bkt->m_fBalanceFactor, bkt->m_iTreeNumber, m_options.m_iSelectHeadNumberOfThreads);

                bkt->BuildTrees<InternalDataType>(data, m_options.m_distCalcMethod, m_options.m_iSelectHeadNumberOfThreads, nullptr, nullptr, true);
                auto t2 = std::chrono::high_resolution_clock::now();
                double elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "End invoking BuildTrees.\n");
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Invoking BuildTrees used time: %.2lf minutes (about %.2lf hours).\n", elapsedSeconds / 60.0, elapsedSeconds / 3600.0);

                if (m_options.m_saveBKT)
                {
                    std::stringstream bktFileNameBuilder;
                    bktFileNameBuilder << m_options.m_vectorPath << ".bkt." << m_options.m_iBKTKmeansK << "_"
                                       << m_options.m_iBKTLeafSize << "_" << m_options.m_iTreeNumber << "_" << m_options.m_iSamples << "_"
                                       << static_cast<int>(m_options.m_distCalcMethod) << ".bin";
                    bkt->SaveTrees(bktFileNameBuilder.str());
                }
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Finish generating BKT.\n");

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Start selecting nodes...Select Head Dynamically...\n");
                SelectHeadDynamically(bkt, data.R(), selected);

                if (selected.empty())
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Can't select any vector as head with current settings\n");
                    return false;
                }
            }

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                         "Seleted Nodes: %u, about %.2lf%% of total.\n",
                         static_cast<unsigned int>(selected.size()),
                         selected.size() * 100.0 / data.R());

            if (!m_options.m_noOutput)
            {
                std::sort(selected.begin(), selected.end());

                std::shared_ptr<Helper::DiskIO> output = SPTAG::f_createIO(), outputIDs = SPTAG::f_createIO();
                if (output == nullptr || outputIDs == nullptr ||
                    !output->Initialize((m_options.m_indexDirectory + FolderSep + m_options.m_headVectorFile).c_str(), std::ios::binary | std::ios::out) ||
                    !outputIDs->Initialize((m_options.m_indexDirectory + FolderSep + m_options.m_headIDFile).c_str(), std::ios::binary | std::ios::out))
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to create output file:%s %s\n",
                                 (m_options.m_indexDirectory + FolderSep + m_options.m_headVectorFile).c_str(),
                                 (m_options.m_indexDirectory + FolderSep + m_options.m_headIDFile).c_str());
                    return false;
                }

                SizeType val = static_cast<SizeType>(selected.size());
                if (output->WriteBinary(sizeof(val), reinterpret_cast<char *>(&val)) != sizeof(val))
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write output file!\n");
                    return false;
                }
                DimensionType dt = data.C();
                if (output->WriteBinary(sizeof(dt), reinterpret_cast<char *>(&dt)) != sizeof(dt))
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write output file!\n");
                    return false;
                }

                for (int i = 0; i < selected.size(); i++)
                {
                    uint64_t vid = static_cast<uint64_t>(selected[i]);
                    if (outputIDs->WriteBinary(sizeof(vid), reinterpret_cast<char *>(&vid)) != sizeof(vid))
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write output file!\n");
                        return false;
                    }

                    if (output->WriteBinary(sizeof(InternalDataType) * data.C(), (char *)(data[vid])) != sizeof(InternalDataType) * data.C())
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write output file!\n");
                        return false;
                    }
                }
            }
            auto t3 = std::chrono::high_resolution_clock::now();
            double elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(t3 - t1).count();
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Total used time: %.2lf minutes (about %.2lf hours).\n", elapsedSeconds / 60.0, elapsedSeconds / 3600.0);
            return true;
        }

        template <typename T>
        void Index<T>::LoadFusionHeadInfo(const std::string& p_file, std::vector<ListInfo>& p_listInfos)
        {
            auto ptr = SPTAG::f_createIO();
            if (ptr == nullptr || !ptr->Initialize(p_file.c_str(), std::ios::binary | std::ios::in)) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to open file: %s\n", p_file.c_str());
                throw std::runtime_error("Failed open file in LoadingHeadInfo");
            }

            int m_listCount;
            int m_listPageOffset;

            if (ptr->ReadBinary(sizeof(m_listCount), reinterpret_cast<char*>(&m_listCount)) != sizeof(m_listCount)) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                throw std::runtime_error("Failed read file in LoadingHeadInfo");
            }
            if (ptr->ReadBinary(sizeof(m_totalDocumentCount), reinterpret_cast<char*>(&m_totalDocumentCount)) != sizeof(m_totalDocumentCount)) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                throw std::runtime_error("Failed read file in LoadingHeadInfo");
            }
            if (ptr->ReadBinary(sizeof(m_iDataDimension), reinterpret_cast<char*>(&m_iDataDimension)) != sizeof(m_iDataDimension)) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                throw std::runtime_error("Failed read file in LoadingHeadInfo");
            }
            if (ptr->ReadBinary(sizeof(m_listPageOffset), reinterpret_cast<char*>(&m_listPageOffset)) != sizeof(m_listPageOffset)) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                throw std::runtime_error("Failed read file in LoadingHeadInfo");
            }

            if (m_vectorInfoSize == 0) m_vectorInfoSize = m_iDataDimension * sizeof(T) + sizeof(int);
            else if (m_vectorInfoSize != m_iDataDimension * sizeof(T) + sizeof(int)) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "m_vectorInfoSize: %d\n", m_vectorInfoSize);

                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read head info file! DataDimension and ValueType are not match!\n");
                throw std::runtime_error("DataDimension and ValueType don't match in LoadingHeadInfo");
            }

            p_listInfos.resize(m_listCount);

            size_t totalListElementCount = 0;

            int pageNum;
            for (int i = 0; i < m_listCount; ++i)
            {
                ListInfo* listInfo = &(p_listInfos[i]);

                if (ptr->ReadBinary(sizeof(pageNum), reinterpret_cast<char*>(&(pageNum))) != sizeof(pageNum)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                    throw std::runtime_error("Failed read file in LoadingHeadInfo");
                }
                if (ptr->ReadBinary(sizeof(listInfo->pageOffset), reinterpret_cast<char*>(&(listInfo->pageOffset))) != sizeof(listInfo->pageOffset)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                    throw std::runtime_error("Failed read file in LoadingHeadInfo");
                }
                if (ptr->ReadBinary(sizeof(listInfo->listEleCount), reinterpret_cast<char*>(&(listInfo->listEleCount))) != sizeof(listInfo->listEleCount)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                    throw std::runtime_error("Failed read file in LoadingHeadInfo");
                }
                if (ptr->ReadBinary(sizeof(listInfo->listPageCount), reinterpret_cast<char*>(&(listInfo->listPageCount))) != sizeof(listInfo->listPageCount)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                    throw std::runtime_error("Failed read file in LoadingHeadInfo");
                }
                listInfo->listOffset = (static_cast<uint64_t>(m_listPageOffset + pageNum) << PageSizeEx);

                totalListElementCount += listInfo->listEleCount;
                int pageCount = listInfo->listPageCount;
            }

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                "Finish reading header info, list count %d, total doc count %d, dimension %d, list page offset %d.\n",
                m_listCount,
                m_totalDocumentCount,
                m_iDataDimension,
                m_listPageOffset);

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Total Element Count: %llu\n", totalListElementCount);

            m_vectorMapPosting.resize(m_totalDocumentCount);
            for (int i = 0; i < m_totalDocumentCount; ++i)
            {
                if (ptr->ReadBinary(sizeof(m_vectorMapPosting[i]), reinterpret_cast<char*>(&(m_vectorMapPosting[i]))) != sizeof(m_vectorMapPosting[i])) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                    throw std::runtime_error("Failed read file in LoadingHeadInfo");
                }
            }
        }
        
        template <typename T>
        ErrorCode Index<T>::BuildIndexInternal(std::shared_ptr<Helper::VectorSetReader> &p_reader)
        {
            if (!m_options.m_indexDirectory.empty())
            {
                if (!direxists(m_options.m_indexDirectory.c_str()))
                {
                    mkdir(m_options.m_indexDirectory.c_str());
                }
            }

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Begin Select Head...\n");
            auto t1 = std::chrono::high_resolution_clock::now();
            if (m_options.m_selectHead)
            {
                omp_set_num_threads(m_options.m_iSelectHeadNumberOfThreads);
                bool success = false;
                if (m_pQuantizer)
                {
                    success = SelectHeadInternal<std::uint8_t>(p_reader);
                }
                else
                {
                    success = SelectHeadInternal<T>(p_reader);
                }
                if (!success)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "SelectHead Failed!\n");
                    return ErrorCode::Fail;
                }
            }
            auto t2 = std::chrono::high_resolution_clock::now();
            double selectHeadTime = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "select head time: %.2lfs\n", selectHeadTime);

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Begin Build Head...\n");
            if (m_options.m_buildHead)
            {
                auto valueType = m_pQuantizer ? SPTAG::VectorValueType::UInt8 : m_options.m_valueType;
                auto dims = m_pQuantizer ? m_pQuantizer->GetNumSubvectors() : m_options.m_dim;

                m_index = SPTAG::VectorIndex::CreateInstance(m_options.m_indexAlgoType, valueType);
                m_index->SetParameter("DistCalcMethod", SPTAG::Helper::Convert::ConvertToString(m_options.m_distCalcMethod));
                m_index->SetQuantizer(m_pQuantizer);
                for (const auto &iter : m_headParameters)
                {
                    m_index->SetParameter(iter.first.c_str(), iter.second.c_str());
                }

                std::shared_ptr<Helper::ReaderOptions> vectorOptions(new Helper::ReaderOptions(valueType, dims, VectorFileType::DEFAULT));
                auto vectorReader = Helper::VectorSetReader::CreateInstance(vectorOptions);
                if (ErrorCode::Success != vectorReader->LoadFile(m_options.m_indexDirectory + FolderSep + m_options.m_headVectorFile))
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read head vector file.\n");
                    return ErrorCode::Fail;
                }
                {
                    auto headvectorset = vectorReader->GetVectorSet();
                    if (m_index->BuildIndex(headvectorset, nullptr, false, true, true) != ErrorCode::Success)
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to build head index.\n");
                        return ErrorCode::Fail;
                    }
                    m_index->SetQuantizerFileName(m_options.m_quantizerFilePath.substr(m_options.m_quantizerFilePath.find_last_of("/\\") + 1));
                    if (m_index->SaveIndex(m_options.m_indexDirectory + FolderSep + m_options.m_headIndexFolder) != ErrorCode::Success)
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to save head index.\n");
                        return ErrorCode::Fail;
                    }
                }
                m_index.reset();
                if (LoadIndex(m_options.m_indexDirectory + FolderSep + m_options.m_headIndexFolder, m_index) != ErrorCode::Success)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot load head index from %s!\n", (m_options.m_indexDirectory + FolderSep + m_options.m_headIndexFolder).c_str());
                }
            }
            auto t3 = std::chrono::high_resolution_clock::now();
            double buildHeadTime = std::chrono::duration_cast<std::chrono::seconds>(t3 - t2).count();
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "select head time: %.2lfs build head time: %.2lfs\n", selectHeadTime, buildHeadTime);

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Begin Build SSDIndex...\n");
            if (m_options.m_enableSSD)
            {
                omp_set_num_threads(m_options.m_iSSDNumberOfThreads);

                if (m_index == nullptr && LoadIndex(m_options.m_indexDirectory + FolderSep + m_options.m_headIndexFolder, m_index) != ErrorCode::Success)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot load head index from %s!\n", (m_options.m_indexDirectory + FolderSep + m_options.m_headIndexFolder).c_str());
                    return ErrorCode::Fail;
                }
                m_index->SetQuantizer(m_pQuantizer);
                if (!CheckHeadIndexType())
                    return ErrorCode::Fail;

                m_index->SetParameter("NumberOfThreads", std::to_string(m_options.m_iSSDNumberOfThreads));
                m_index->SetParameter("MaxCheck", std::to_string(m_options.m_maxCheck));
                m_index->SetParameter("HashTableExponent", std::to_string(m_options.m_hashExp));
                m_index->UpdateIndex();

                if (m_pQuantizer)
                {
                    m_extraSearcher.reset(new ExtraFullGraphSearcher<std::uint8_t>());
                }
                else
                {
                    m_extraSearcher.reset(new ExtraFullGraphSearcher<T>());
                }

                if (m_options.m_buildSsdIndex)
                {
                    if (!m_options.m_excludehead)
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Include all vectors into SSD index...\n");
                        std::shared_ptr<Helper::DiskIO> ptr = SPTAG::f_createIO();
                        if (ptr == nullptr || !ptr->Initialize((m_options.m_indexDirectory + FolderSep + m_options.m_headIDFile).c_str(), std::ios::binary | std::ios::out))
                        {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to open headIDFile file:%s for overwrite\n", (m_options.m_indexDirectory + FolderSep + m_options.m_headIDFile).c_str());
                            return ErrorCode::Fail;
                        }
                        std::uint64_t vid = (std::uint64_t)MaxSize;
                        for (int i = 0; i < m_index->GetNumSamples(); i++)
                        {
                            IOBINARY(ptr, WriteBinary, sizeof(std::uint64_t), (char *)(&vid));
                        }
                    }

                    if (!m_extraSearcher->BuildIndex(p_reader, m_index, m_options))
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "BuildSSDIndex Failed!\n");
                        return ErrorCode::Fail;
                    }
                    if (m_options.m_enableReorderIndex)
                    {
                        if (!m_extraSearcher->BuildIndexFusion(p_reader, m_index, m_options))
                        {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "BuildRerankIndex Failed!\n");
                            return ErrorCode::Fail;
                        }
                    }
                }
                // if (!m_extraSearcher->LoadIndex(m_options))
                // {
                //     SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot Load SSDIndex!\n");
                //     if (m_options.m_buildSsdIndex)
                //     {
                //         return ErrorCode::Fail;
                //     }
                //     else
                //     {
                //         m_extraSearcher.reset();
                //     }
                // }

                if(!m_options.m_enableReorderIndex)
                {
                    if(m_options.m_resultNum > 0){
                        std::string curFile = m_options.m_vectorPath;
                        auto curIndexFile = SPANN::f_createAsyncIO();
                        if (curIndexFile == nullptr || !curIndexFile->Initialize(curFile.c_str(), std::ios::binary | std::ios::in, 
        #ifndef _MSC_VER
        #ifdef BATCH_READ
                            m_options.m_resultNum, 2, 2, m_options.m_iSSDNumberOfThreads
        #else
                            m_options.m_searchInternalResultNum * m_options.m_iSSDNumberOfThreads / m_options.m_ioThreads + 1, 2, 2, m_options.m_ioThreads
        #endif
        /*
        #ifdef BATCH_READ
                            max(m_options.m_searchInternalResultNum*m_vectorInfoSize, 1 << 12), 2, 2, m_options.m_iSSDNumberOfThreads
        #else
                            m_options.m_searchInternalResultNum* m_options.m_iSSDNumberOfThreads / m_options.m_ioThreads + 1, 2, 2, m_options.m_ioThreads
        #endif
        */
        #else
                            (m_options.m_searchPostingPageLimit + 1) * PageSize, 2, 2, (std::uint16_t)m_options.m_ioThreads
        #endif
                        )) {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot open file:%s!\n", curFile.c_str());
                            return ErrorCode::Fail;
                        }
                        m_indexFiles.emplace_back(curIndexFile);
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Initialize Async Read!\n");
                    }
                }else{
                    if(m_options.m_resultNum > 0){
                        std::string curFile = m_options.m_indexDirectory + FolderSep + m_options.m_fusionIndex;
                        auto curIndexFile = SPANN::f_createAsyncIO();
                        if (curIndexFile == nullptr || !curIndexFile->Initialize(curFile.c_str(), std::ios::binary | std::ios::in, 
        #ifndef _MSC_VER
        #ifdef BATCH_READ
                            m_options.m_resultNum, 2, 2, m_options.m_iSSDNumberOfThreads
        #else
                            m_options.m_searchInternalResultNum * m_options.m_iSSDNumberOfThreads / m_options.m_ioThreads + 1, 2, 2, m_options.m_ioThreads
        #endif
        /*
        #ifdef BATCH_READ
                            max(m_options.m_searchInternalResultNum*m_vectorInfoSize, 1 << 12), 2, 2, m_options.m_iSSDNumberOfThreads
        #else
                            m_options.m_searchInternalResultNum* m_options.m_iSSDNumberOfThreads / m_options.m_ioThreads + 1, 2, 2, m_options.m_ioThreads
        #endif
        */
        #else
                            (m_options.m_searchPostingPageLimit + 1) * PageSize, 2, 2, (std::uint16_t)m_options.m_ioThreads
        #endif
                        )) {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot open file:%s!\n", curFile.c_str());
                            return ErrorCode::Fail;
                        }
                        m_indexFiles.emplace_back(curIndexFile);
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Initialize Async Read!\n");
                        LoadFusionHeadInfo(curFile, m_listInfos);
                    }
                }       
                

                if (m_extraSearcher != nullptr)
                {
                    m_vectorTranslateMap.reset(new std::uint64_t[m_index->GetNumSamples()], std::default_delete<std::uint64_t[]>());
                    std::shared_ptr<Helper::DiskIO> ptr = SPTAG::f_createIO();
                    if (ptr == nullptr || !ptr->Initialize((m_options.m_indexDirectory + FolderSep + m_options.m_headIDFile).c_str(), std::ios::binary | std::ios::in))
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to open headIDFile file:%s\n", (m_options.m_indexDirectory + FolderSep + m_options.m_headIDFile).c_str());
                        return ErrorCode::Fail;
                    }
                    IOBINARY(ptr, ReadBinary, sizeof(std::uint64_t) * m_index->GetNumSamples(), (char *)(m_vectorTranslateMap.get()));
                }
            }
            auto t4 = std::chrono::high_resolution_clock::now();
            double buildSSDTime = std::chrono::duration_cast<std::chrono::seconds>(t4 - t3).count();
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "select head time: %.2lfs build head time: %.2lfs build ssd time: %.2lfs\n", selectHeadTime, buildHeadTime, buildSSDTime);

            if (m_options.m_deleteHeadVectors)
            {
                if (fileexists((m_options.m_indexDirectory + FolderSep + m_options.m_headVectorFile).c_str()) &&
                    remove((m_options.m_indexDirectory + FolderSep + m_options.m_headVectorFile).c_str()) != 0)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "Head vector file can't be removed.\n");
                }
            }

            m_bReady = true;
            return ErrorCode::Success;
        }
        template <typename T>
        ErrorCode Index<T>::BuildIndex(bool p_normalized)
        {
            SPTAG::VectorValueType valueType = m_pQuantizer ? SPTAG::VectorValueType::UInt8 : m_options.m_valueType;
            SizeType dim = m_pQuantizer ? m_pQuantizer->GetNumSubvectors() : m_options.m_dim;
            std::shared_ptr<Helper::ReaderOptions> vectorOptions(new Helper::ReaderOptions(valueType, dim, m_options.m_vectorType, m_options.m_vectorDelimiter, m_options.m_iSSDNumberOfThreads, p_normalized));
            auto vectorReader = Helper::VectorSetReader::CreateInstance(vectorOptions);
            if (m_options.m_selectHead || m_options.m_buildHead || m_options.m_buildSsdIndex)
            {
                if (m_options.m_vectorPath.empty())
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Vector file is empty. Skipping loading.\n");
                }
                else
                {
                    if (ErrorCode::Success != vectorReader->LoadFile(m_options.m_vectorPath))
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read vector file.\n");
                        return ErrorCode::Fail;
                    }
                    m_options.m_vectorSize = vectorReader->GetVectorSet()->Count();
                }
            }

            return BuildIndexInternal(vectorReader);
        }

        template <typename T>
        ErrorCode Index<T>::BuildIndex(const void *p_data, SizeType p_vectorNum, DimensionType p_dimension, bool p_normalized, bool p_shareOwnership)
        {
            if (p_data == nullptr || p_vectorNum == 0 || p_dimension == 0)
                return ErrorCode::EmptyData;

            std::shared_ptr<VectorSet> vectorSet;
            if (p_shareOwnership)
            {
                vectorSet.reset(new BasicVectorSet(ByteArray((std::uint8_t *)p_data, sizeof(T) * p_vectorNum * p_dimension, false),
                                                   GetEnumValueType<T>(), p_dimension, p_vectorNum));
            }
            else
            {
                ByteArray arr = ByteArray::Alloc(sizeof(T) * p_vectorNum * p_dimension);
                memcpy(arr.Data(), p_data, sizeof(T) * p_vectorNum * p_dimension);
                vectorSet.reset(new BasicVectorSet(arr, GetEnumValueType<T>(), p_dimension, p_vectorNum));
            }

            if (m_options.m_distCalcMethod == DistCalcMethod::Cosine && !p_normalized)
            {
                vectorSet->Normalize(m_options.m_iSSDNumberOfThreads);
            }
            SPTAG::VectorValueType valueType = m_pQuantizer ? SPTAG::VectorValueType::UInt8 : m_options.m_valueType;
            std::shared_ptr<Helper::VectorSetReader> vectorReader(new Helper::MemoryVectorReader(std::make_shared<Helper::ReaderOptions>(valueType, p_dimension, VectorFileType::DEFAULT, m_options.m_vectorDelimiter, m_options.m_iSSDNumberOfThreads, true),
                                                                                                 vectorSet));

            m_options.m_valueType = GetEnumValueType<T>();
            m_options.m_dim = p_dimension;
            m_options.m_vectorSize = p_vectorNum;
            return BuildIndexInternal(vectorReader);
        }

        template <typename T>
        ErrorCode
        Index<T>::UpdateIndex()
        {
            omp_set_num_threads(m_options.m_iSSDNumberOfThreads);
            m_index->SetParameter("NumberOfThreads", std::to_string(m_options.m_iSSDNumberOfThreads));
            // m_index->SetParameter("MaxCheck", std::to_string(m_options.m_maxCheck));
            // m_index->SetParameter("HashTableExponent", std::to_string(m_options.m_hashExp));
            m_index->UpdateIndex();
            return ErrorCode::Success;
        }

        template <typename T>
        ErrorCode
        Index<T>::SetParameter(const char *p_param, const char *p_value, const char *p_section)
        {
            if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_section, "BuildHead") && !SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_param, "isExecute"))
            {
                if (m_index != nullptr)
                    return m_index->SetParameter(p_param, p_value);
                else
                    m_headParameters[p_param] = p_value;
            }
            else
            {
                m_options.SetParameter(p_section, p_param, p_value);
            }
            if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_param, "DistCalcMethod"))
            {
                if (m_pQuantizer)
                {
                    m_fComputeDistance = m_pQuantizer->DistanceCalcSelector<T>(m_options.m_distCalcMethod);
                    m_iBaseSquare = (m_options.m_distCalcMethod == DistCalcMethod::Cosine) ? m_pQuantizer->GetBase() * m_pQuantizer->GetBase() : 1;
                }
                else
                {
                    m_fComputeDistance = COMMON::DistanceCalcSelector<T>(m_options.m_distCalcMethod);
                    m_iBaseSquare = (m_options.m_distCalcMethod == DistCalcMethod::Cosine) ? COMMON::Utils::GetBase<T>() * COMMON::Utils::GetBase<T>() : 1;
                }
            }
            return ErrorCode::Success;
        }

        template <typename T>
        std::string
        Index<T>::GetParameter(const char *p_param, const char *p_section) const
        {
            if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_section, "BuildHead") && !SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_param, "isExecute"))
            {
                if (m_index != nullptr)
                    return m_index->GetParameter(p_param);
                else
                {
                    auto iter = m_headParameters.find(p_param);
                    if (iter != m_headParameters.end())
                        return iter->second;
                    return "Undefined!";
                }
            }
            else
            {
                return m_options.GetParameter(p_section, p_param);
            }
        }
    }
}

#define DefineVectorValueType(Name, Type) \
    template class SPTAG::SPANN::Index<Type>;

#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType
