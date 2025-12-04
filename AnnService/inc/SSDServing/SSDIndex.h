// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <limits>
#include <future>

#include "inc/Core/Common.h"
#include "inc/Core/Common/DistanceUtils.h"
#include "inc/Core/Common/QueryResultSet.h"
#include "inc/Core/SPANN/Index.h"
#include "inc/Core/SPANN/ExtraFullGraphSearcher.h"
#include "inc/Helper/VectorSetReader.h"
#include "inc/Helper/StringConvert.h"
#include "inc/SSDServing/Utils.h"
#include "acl/acl.h"

namespace SPTAG
{
    namespace SSDServing
    {
        namespace SSDIndex
        {

            template <typename ValueType>
            ErrorCode OutputResult(const std::string &p_output, std::vector<QueryResult> &p_results, int p_resultNum)
            {
                if (!p_output.empty())
                {
                    auto ptr = f_createIO();
                    if (ptr == nullptr || !ptr->Initialize(p_output.c_str(), std::ios::binary | std::ios::out))
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed create file: %s\n", p_output.c_str());
                        return ErrorCode::FailedCreateFile;
                    }
                    int32_t i32Val = static_cast<int32_t>(p_results.size());
                    if (ptr->WriteBinary(sizeof(i32Val), reinterpret_cast<char *>(&i32Val)) != sizeof(i32Val))
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to write result file!\n");
                        return ErrorCode::DiskIOFail;
                    }
                    i32Val = p_resultNum;
                    if (ptr->WriteBinary(sizeof(i32Val), reinterpret_cast<char *>(&i32Val)) != sizeof(i32Val))
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to write result file!\n");
                        return ErrorCode::DiskIOFail;
                    }

                    float fVal = 0;
                    for (size_t i = 0; i < p_results.size(); ++i)
                    {
                        for (int j = 0; j < p_resultNum; ++j)
                        {
                            i32Val = p_results[i].GetResult(j)->VID;
                            if (ptr->WriteBinary(sizeof(i32Val), reinterpret_cast<char *>(&i32Val)) != sizeof(i32Val))
                            {
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to write result file!\n");
                                return ErrorCode::DiskIOFail;
                            }

                            fVal = p_results[i].GetResult(j)->Dist;
                            if (ptr->WriteBinary(sizeof(fVal), reinterpret_cast<char *>(&fVal)) != sizeof(fVal))
                            {
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to write result file!\n");
                                return ErrorCode::DiskIOFail;
                            }
                        }
                    }
                }
                return ErrorCode::Success;
            }

            template <typename T, typename V>
            void PrintPercentiles(const std::vector<V> &p_values, std::function<T(const V &)> p_get, const char *p_format)
            {
                double sum = 0;
                std::vector<T> collects;
                collects.reserve(p_values.size());
                for (const auto &v : p_values)
                {
                    T tmp = p_get(v);
                    sum += tmp;
                    collects.push_back(tmp);
                }

                std::sort(collects.begin(), collects.end());

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Avg\t50tiles\t90tiles\t95tiles\t99tiles\t99.9tiles\tMax\n");

                std::string formatStr("%.3lf");
                for (int i = 1; i < 7; ++i)
                {
                    formatStr += '\t';
                    formatStr += p_format;
                }

                formatStr += '\n';

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                             formatStr.c_str(),
                             sum / collects.size(),
                             collects[static_cast<size_t>(collects.size() * 0.50)],
                             collects[static_cast<size_t>(collects.size() * 0.90)],
                             collects[static_cast<size_t>(collects.size() * 0.95)],
                             collects[static_cast<size_t>(collects.size() * 0.99)],
                             collects[static_cast<size_t>(collects.size() * 0.999)],
                             collects[static_cast<size_t>(collects.size() - 1)]);
            }

            template <typename ValueType>
            void SearchSequential(SPANN::Index<ValueType>* p_index,
                int p_numThreads,
                std::vector<QueryResult>& p_results,
                std::vector<SPANN::SearchStats>& p_stats,
                int p_maxQueryCount, int p_internalResultNum,
                std::shared_ptr<SPTAG::VectorSet> querySet,
                std::shared_ptr<SPTAG::VectorSet> PQVectorSet,
                std::shared_ptr<SPTAG::VectorSet> rerankVectorSet,
                int PQVectorCount, int PQVectorDim, std::vector<int>& numVecPerPostinglist, std::vector<std::unique_ptr<int[]>>& postinglist, void* d_PQVectorSet,
                uint8_t *d_table, int* d_vectorIDs, float *d_dist, int *h_vectorIDs, float *h_dist, int totalNumVec, std::shared_ptr<SPTAG::COMMON::IQuantizer> quantizer, bool isWarmup)
            {
                int numQueries = min(static_cast<int>(p_results.size()), p_maxQueryCount);

                std::atomic_size_t queriesSent(0);
                
                std::vector<std::thread> threads;
                threads.reserve(p_numThreads);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Searching: numThread: %d, numQueries: %d.\n", p_numThreads, numQueries);

                SPANN::Options &p_opts = *(p_index->GetOptions());

                Utils::StopW sw;
                for (int i = 0; i < p_numThreads; i++)
                {
                    threads.emplace_back([&, i]()
                                         {
                    NumaStrategy ns = (p_index->GetDiskIndex() != nullptr) ? NumaStrategy::SCATTER : NumaStrategy::LOCAL; // Only for SPANN, we need to avoid IO threads overlap with search threads.
                    Helper::SetThreadAffinity(i, threads[i], ns, OrderStrategy::ASC); 

                    Utils::StopW threadws;
                    size_t index = 0;
                    while (true)
                    {
                        index = queriesSent.fetch_add(1);
                        if (index < numQueries)
                        {
                            if ((index & ((1 << 14) - 1)) == 0)
                            {
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Sent %.2lf%%...\n", index * 100.0 / numQueries);
                            }

                            double startTime = threadws.getElapsedMs();
                            p_index->GetMemoryIndex()->SearchIndex(p_results[index]);
                            double endTime = threadws.getElapsedMs();

                            (*((COMMON::QueryResultSet<ValueType> *)&p_results[index])).Setmy_newPQTarget(quantizer);
                            std::unordered_set<int> postingIDSet;
                            if(p_opts.m_enableGPU)
                            {
                                p_index->SearchPQIndex_GPU(p_results[index], PQVectorSet, PQVectorCount, PQVectorDim, numVecPerPostinglist, postinglist, d_PQVectorSet, d_table, d_vectorIDs, d_dist, h_vectorIDs, h_dist, totalNumVec, i, postingIDSet, &(p_stats[index]));
                            }else{
                                p_index->SearchPQIndex_CPU(p_results[index], PQVectorSet, PQVectorCount, PQVectorDim, numVecPerPostinglist, postinglist, postingIDSet, &(p_stats[index]));
                            }
                            
                            double searchEndTime = threadws.getElapsedMs();

                            if (p_opts.m_enableReorderIndex)
                            {
                                if (p_opts.m_rerank > 0 && p_opts.m_resultNum > 0) 
                                {
                                    p_index->RerankFullVectorFusion(p_results[index], rerankVectorSet, i, &(p_stats[index]));
                                }
                            }else{
                                if (p_opts.m_rerank > 0 && p_opts.m_resultNum > 0) 
                                {
                                    p_index->RerankFullVector(p_results[index], rerankVectorSet, i, postingIDSet, &(p_stats[index]));
                                }
                            }
                            
                            // 

                            
                            
                            /*
                            //模拟ssd延迟
                            if (p_opts.m_rerank > 0) 
                            {
                                int K = p_opts.m_resultNum;
                                for (int j = 0; j < K; j++)
                                {
                                    if (p_results[index].GetResult(j)->VID < 0) continue;
                                    p_results[index].GetResult(j)->Dist = COMMON::DistanceUtils::ComputeDistance((const ValueType*)p_results[index].GetTarget(),
                                        (const ValueType*)rerankVectorSet->GetVector(p_results[index].GetResult(j)->VID), p_opts.m_dim, p_opts.m_distCalcMethod);
                                }

                                // 假设每次异步读取操作的时间为0.0000002秒，4KB / (20 GB/s) = 0.0000002 秒
                                double sleepTime = 0.0000002 * K;
                                // 计算睡眠时间（微秒）
                                int sleepMicroseconds = sleepTime * 1000000; // 将秒转换为微秒
                                // 通过睡眠来模拟延迟（微秒级别精度）
                                std::this_thread::sleep_for(std::chrono::microseconds(sleepMicroseconds));

                                BasicResult* re = p_results[index].GetResults();
                                std::sort(re, re + p_opts.m_searchInternalResultNum, COMMON::Compare);
                            }
                            */
                            
                            double exEndTime = threadws.getElapsedMs();

                            p_stats[index].m_exLatency = searchEndTime - endTime;
                            p_stats[index].m_totalSearchLatency = searchEndTime - startTime;
                            p_stats[index].rerankLatency = exEndTime - searchEndTime;
                            p_stats[index].m_totalLatency = exEndTime - startTime;
                        }
                        else
                        {
                            return;
                        }
                    } });
                }
                for (auto &thread : threads)
                {
                    thread.join();
                }

                double sendingCost = sw.getElapsedSec();

                // if(!isWarmup)
                // {
                //     SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                //              "Finish sending in %.3lf seconds, actuallQPS is %.2lf, query count %u.\n",
                //              sendingCost,
                //              numQueries / sendingCost,
                //              static_cast<uint32_t>(numQueries));
                // }

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                        "Finish sending in %.3lf seconds, actuallQPS is %.2lf, query count %u.\n",
                        sendingCost,
                        numQueries / sendingCost,
                        static_cast<uint32_t>(numQueries));

                for (int i = 0; i < numQueries; i++)
                {
                    p_results[i].CleanQuantizedTarget();
                }
            }

            template <typename ValueType>
            void Search(SPANN::Index<ValueType> *p_index)
            {
                SPANN::Options &p_opts = *(p_index->GetOptions());
                std::string outputFile = p_opts.m_searchResult;
                std::string truthFile = p_opts.m_truthPath;
                std::string warmupFile = p_opts.m_warmupPath;

                if (p_index->m_pQuantizer)
                {
                    p_index->m_pQuantizer->SetEnableADC(p_opts.m_enableADC);
                }

                if (!p_opts.m_logFile.empty())
                {
                    SetLogger(std::make_shared<Helper::FileLogger>(Helper::LogLevel::LL_Info, p_opts.m_logFile.c_str()));
                }
                int numThreads = p_opts.m_iSSDNumberOfThreads;
                int internalResultNum = p_opts.m_searchInternalResultNum;
                int K = p_opts.m_resultNum;
                int truthK = (p_opts.m_rerank <= 0) ? K : p_opts.m_rerank;

                std::string QuantizervectorFilePath = p_opts.m_quantizerVectorFilePath;
                int count, dim;
                auto ptr_vector = SPTAG::f_createIO();
                if (!ptr_vector->Initialize(QuantizervectorFilePath.c_str(), std::ios::binary | std::ios::in))
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read quantizervector file.\n");
                    return;
                }
                ptr_vector->ReadBinary(sizeof(count), reinterpret_cast<char *>(&(count)));
                ptr_vector->ReadBinary(sizeof(dim), reinterpret_cast<char *>(&(dim)));
                std::shared_ptr<VectorSet> PQVectorSet;
                if (!QuantizervectorFilePath.empty() && fileexists(QuantizervectorFilePath.c_str()))
                {
                    std::shared_ptr<Helper::ReaderOptions> vectorOptions(new Helper::ReaderOptions(VectorValueType::UInt8, dim, VectorFileType::DEFAULT, p_opts.m_vectorDelimiter));
                    auto vectorReader = Helper::VectorSetReader::CreateInstance(vectorOptions);
                    if (ErrorCode::Success == vectorReader->LoadFile(QuantizervectorFilePath))
                    {
                        PQVectorSet = vectorReader->GetVectorSet();
                    }
                }

                aclInit(nullptr);
                aclrtSetDevice(0);

                std::string PostingListFilePath = p_opts.m_indexDirectory + FolderSep + p_opts.m_postingListIndex;
                std::vector<int> numvec;
                std::vector<std::unique_ptr<int[]>> postinglist;
                int num_postinglist, fullVectorCount, MaxNumVec = 0;
                auto fp_read = SPTAG::f_createIO();
                if (fp_read == nullptr || !fp_read->Initialize(PostingListFilePath.c_str(), std::ios::binary | std::ios::in))
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read quantizervector file.\n");
                    return;
                }
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Beign load PQpostinglist\n");
                if (fp_read->ReadBinary(sizeof(int), reinterpret_cast<char *>(&(num_postinglist))) != sizeof(int))
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read PQPostingList file!\n");
                    throw std::runtime_error("Failed read file in PQPostingList");
                }
                if (fp_read->ReadBinary(sizeof(int), reinterpret_cast<char *>(&(fullVectorCount))) != sizeof(int))
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read PQPostingList file!\n");
                    throw std::runtime_error("Failed read file in PQPostingList");
                }

                for (int i = 0; i < num_postinglist; i++)
                {
                    int postingListMeta;
                    if (fp_read->ReadBinary(sizeof(int), reinterpret_cast<char *>(&(postingListMeta))) != sizeof(int))
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read PQPostingList file!\n");
                        throw std::runtime_error("Failed read file in PQPostingList");
                    }
                    numvec.push_back(postingListMeta);
                    if (postingListMeta > MaxNumVec)  MaxNumVec = postingListMeta;
                }
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "The MaxNumVec of postinglist is %d\n", MaxNumVec);

                for (int i = 0; i < num_postinglist; i++)
                {
                    postinglist.push_back(std::unique_ptr<int[]>(new int[numvec[i]]));
                    if (fp_read->ReadBinary(sizeof(int) * numvec[i], reinterpret_cast<char *>(postinglist[i].get())) != sizeof(int) * numvec[i])
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read PQPostingList file!\n");
                        throw std::runtime_error("Failed read file in PQPostingList");
                    }
                }
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Load PQpostinglist finish\n");

                std::shared_ptr<VectorSet> vectorSetRatio;
                if (!p_opts.m_vectorPath.empty() && fileexists(p_opts.m_vectorPath.c_str()))
                {
                    std::shared_ptr<Helper::ReaderOptions> vectorOptions(new Helper::ReaderOptions(p_opts.m_valueType, p_opts.m_dim, p_opts.m_vectorType, p_opts.m_vectorDelimiter));
                    auto vectorReader = Helper::VectorSetReader::CreateInstance(vectorOptions);
                    if (ErrorCode::Success == vectorReader->LoadFile(p_opts.m_vectorPath))
                    {
                        vectorSetRatio = vectorReader->GetVectorSetRatio(p_opts.m_readRatio);
                        if (p_opts.m_distCalcMethod == DistCalcMethod::Cosine)
                            vectorSetRatio->Normalize(numThreads);
                    }
                }

                std::shared_ptr<SPTAG::COMMON::IQuantizer> quantizer;
                std::string QuantizerFilePath = p_opts.m_quantizerPQFilePath;
                auto ptr = SPTAG::f_createIO();
                if (!ptr->Initialize(QuantizerFilePath.c_str(), std::ios::binary | std::ios::in))
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read quantizer file.\n");
                    return;
                }
                quantizer = SPTAG::COMMON::IQuantizer::LoadIQuantizer(ptr);
                quantizer->SetEnableADC(true);

                int totalNumVec = MaxNumVec * p_opts.m_searchInternalResultNum;
                uint8_t *d_table;
                aclrtMalloc((void **)&d_table, 256 * dim * sizeof(float) * numThreads, ACL_MEM_MALLOC_HUGE_FIRST);

                int *d_vectorIDs;
                aclrtMalloc((void **)&d_vectorIDs, sizeof(int) * totalNumVec * numThreads, ACL_MEM_MALLOC_HUGE_FIRST);

                float *d_dist;
                aclrtMalloc((void **)&d_dist, sizeof(float) * totalNumVec * numThreads, ACL_MEM_MALLOC_HUGE_FIRST);

                int *h_vectorIDs;
                aclrtMallocHost((void **)&h_vectorIDs, sizeof(int) * totalNumVec * numThreads);

                float *h_dist;
                aclrtMallocHost((void **)&h_dist, sizeof(float) * totalNumVec * numThreads);

                bool isWarmup = false;

                if (!warmupFile.empty())
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Start loading warmup query set...\n");
                    std::shared_ptr<Helper::ReaderOptions> queryOptions(new Helper::ReaderOptions(p_opts.m_valueType, p_opts.m_dim, p_opts.m_warmupType, p_opts.m_warmupDelimiter));
                    auto queryReader = Helper::VectorSetReader::CreateInstance(queryOptions);
                    if (ErrorCode::Success != queryReader->LoadFile(p_opts.m_warmupPath))
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read query file.\n");
                        exit(1);
                    }
                    auto warmupQuerySet = queryReader->GetVectorSet();
                    int warmupNumQueries = warmupQuerySet->Count();

                    std::vector<QueryResult> warmupResults(warmupNumQueries, QueryResult(NULL, max(K, internalResultNum), false));
                    std::vector<SPANN::SearchStats> warmpUpStats(warmupNumQueries);
                    for (int i = 0; i < warmupNumQueries; ++i)
                    {
                        (*((COMMON::QueryResultSet<ValueType> *)&warmupResults[i])).SetTarget(reinterpret_cast<ValueType *>(warmupQuerySet->GetVector(i)), p_index->m_pQuantizer);
                        //(*((COMMON::QueryResultSet<ValueType> *)&warmupResults[i])).Setmy_newPQTarget(quantizer);
                        warmupResults[i].Reset();
                    }

                    isWarmup = true;
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Start warmup...\n");
                    SearchSequential(p_index, numThreads, warmupResults, warmpUpStats, p_opts.m_queryCountLimit, internalResultNum, warmupQuerySet, PQVectorSet,
                    vectorSetRatio, count, dim, numvec, postinglist, PQVectorSet->GetData(), d_table, d_vectorIDs, d_dist, h_vectorIDs, h_dist, totalNumVec, quantizer, isWarmup);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "\nFinish warmup...\n");
                    isWarmup = false;
                }

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Start loading QuerySet...\n");
                std::shared_ptr<Helper::ReaderOptions> queryOptions(new Helper::ReaderOptions(p_opts.m_valueType, p_opts.m_dim, p_opts.m_queryType, p_opts.m_queryDelimiter));
                auto queryReader = Helper::VectorSetReader::CreateInstance(queryOptions);
                if (ErrorCode::Success != queryReader->LoadFile(p_opts.m_queryPath))
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read query file.\n");
                    exit(1);
                }

                auto querySet = queryReader->GetVectorSet();

                int numQueries = min(querySet->Count(), p_opts.m_queryCountLimit);
                std::vector<QueryResult> results(numQueries, QueryResult(NULL, max(K, internalResultNum), false));
                std::vector<SPANN::SearchStats> stats(numQueries);
                for (int i = 0; i < numQueries; ++i)
                {
                    (*((COMMON::QueryResultSet<ValueType> *)&results[i])).SetTarget(reinterpret_cast<ValueType *>(querySet->GetVector(i)), p_index->m_pQuantizer);
                    //(*((COMMON::QueryResultSet<ValueType> *)&results[i])).Setmy_newPQTarget(quantizer);
                    results[i].Reset();
                }

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Start ANN Search...\n");

                SearchSequential(p_index, numThreads, results, stats, p_opts.m_queryCountLimit, internalResultNum, querySet, PQVectorSet,
                vectorSetRatio, count, dim, numvec, postinglist, PQVectorSet->GetData(), d_table, d_vectorIDs, d_dist, h_vectorIDs, h_dist, totalNumVec, quantizer, isWarmup);

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "\nFinish ANN Search...\n");
                
                aclrtFree(d_table);
                aclrtFree(d_vectorIDs);
                aclrtFree(d_dist);
                aclrtFreeHost(h_dist);
                aclrtFreeHost(h_vectorIDs);
                aclrtResetDevice(0);
                aclFinalize();
                vectorSetRatio.reset();

                K = p_opts.m_rerank;

                std::shared_ptr<VectorSet> vectorSet;
                if (!p_opts.m_vectorPath.empty() && fileexists(p_opts.m_vectorPath.c_str()) && p_opts.m_enableCalRecall)
                {
                    std::shared_ptr<Helper::ReaderOptions> vectorOptions(new Helper::ReaderOptions(p_opts.m_valueType, p_opts.m_dim, p_opts.m_vectorType, p_opts.m_vectorDelimiter));
                    auto vectorReader = Helper::VectorSetReader::CreateInstance(vectorOptions);
                    if (ErrorCode::Success == vectorReader->LoadFile(p_opts.m_vectorPath))
                    {
                        vectorSet.reset();
                        auto newVectorSet = vectorReader->GetVectorSet();
                        if (p_opts.m_distCalcMethod == DistCalcMethod::Cosine)
                            newVectorSet->Normalize(numThreads);
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Load VectorSet(%d,%d).\n", newVectorSet->Count(), newVectorSet->Dimension());
                        vectorSet = newVectorSet;
                    }
                }

                float recall = 0, MRR = 0;
                std::vector<std::set<SizeType>> truth;
                if (!truthFile.empty())
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Start loading TruthFile...\n");

                    auto ptr = f_createIO();
                    if (ptr == nullptr || !ptr->Initialize(truthFile.c_str(), std::ios::in | std::ios::binary))
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed open truth file: %s\n", truthFile.c_str());
                        exit(1);
                    }
                    int originalK = truthK;
                    COMMON::TruthSet::LoadTruth(ptr, truth, numQueries, originalK, truthK, p_opts.m_truthType);
                    char tmp[4];
                    if (ptr->ReadBinary(4, tmp) == 4)
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Truth number is larger than query number(%d)!\n", numQueries);
                    }

                    recall = COMMON::TruthSet::CalculateRecall<ValueType>((p_index->GetMemoryIndex()).get(), results, truth, K, truthK, querySet, vectorSet, numQueries, nullptr, false, &MRR);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Recall%d@%d: %f MRR@%d: %f\n", truthK, K, recall, K, MRR);
                }

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "\nEx Elements Count:\n");
                PrintPercentiles<double, SPANN::SearchStats>(
                    stats,
                    [](const SPANN::SearchStats &ss) -> double
                    {
                        return ss.m_totalListElementsCount;
                    },
                    "%.3lf");

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "\nHead Latency Distribution:\n");
                PrintPercentiles<double, SPANN::SearchStats>(
                    stats,
                    [](const SPANN::SearchStats &ss) -> double
                    {
                        return ss.m_totalSearchLatency - ss.m_exLatency;
                    },
                    "%.3lf");

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "\nEx Latency Distribution:\n");
                PrintPercentiles<double, SPANN::SearchStats>(
                    stats,
                    [](const SPANN::SearchStats &ss) -> double
                    {
                        return ss.m_exLatency;
                    },
                    "%.3lf");

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "\nRerank Latency Distribution:\n");
                PrintPercentiles<double, SPANN::SearchStats>(
                    stats,
                    [](const SPANN::SearchStats &ss) -> double
                    {
                        return ss.rerankLatency;
                    },
                    "%.3lf");

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "\nTotal Latency Distribution:\n");
                PrintPercentiles<double, SPANN::SearchStats>(
                    stats,
                    [](const SPANN::SearchStats &ss) -> double
                    {
                        return ss.m_totalLatency;
                    },
                    "%.3lf");

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "\nCut Tree Latency Distribution:\n");
                PrintPercentiles<double, SPANN::SearchStats>(
                    stats,
                    [](const SPANN::SearchStats &ss) -> double
                    {
                        return ss.cutTreeLatency;
                    },
                    "%.3lf");
                
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "\nNPU to CPU Distribution:\n");
                PrintPercentiles<double, SPANN::SearchStats>(
                    stats,
                    [](const SPANN::SearchStats &ss) -> double
                    {
                        return ss.npu2cpu;
                    },
                    "%.3lf");
                
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "\nCPU to NPU Distribution:\n");
                PrintPercentiles<double, SPANN::SearchStats>(
                    stats,
                    [](const SPANN::SearchStats &ss) -> double
                    {
                        return ss.cpu2npu;
                    },
                    "%.3lf");
                
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "\nVector Latency0 Distribution:\n");
                PrintPercentiles<double, SPANN::SearchStats>(
                    stats,
                    [](const SPANN::SearchStats &ss) -> double
                    {
                        return ss.vectorLatency0;
                    },
                    "%.3lf");

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "\nVector Latency1 Distribution:\n");
                PrintPercentiles<double, SPANN::SearchStats>(
                    stats,
                    [](const SPANN::SearchStats &ss) -> double
                    {
                        return ss.vectorLatency1;
                    },
                    "%.3lf");

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "\nFind Table Latency Distribution:\n");
                PrintPercentiles<double, SPANN::SearchStats>(
                    stats,
                    [](const SPANN::SearchStats &ss) -> double
                    {
                        return ss.findTableLatency;
                    },
                    "%.3lf");

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "\nAdd Point Latency Distribution:\n");
                PrintPercentiles<double, SPANN::SearchStats>(
                    stats,
                    [](const SPANN::SearchStats &ss) -> double
                    {
                        return ss.addPointLatency;
                    },
                    "%.3lf");

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "\nSort Latency Distribution:\n");
                PrintPercentiles<double, SPANN::SearchStats>(
                    stats,
                    [](const SPANN::SearchStats &ss) -> double
                    {
                        return ss.sortLatency;
                    },
                    "%.3lf");

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "\nTotal Disk Page Access Distribution:\n");
                PrintPercentiles<int, SPANN::SearchStats>(
                    stats,
                    [](const SPANN::SearchStats &ss) -> int
                    {
                        return ss.m_diskAccessCount;
                    },
                    "%4d");

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "\nTotal Disk IO Distribution:\n");
                PrintPercentiles<int, SPANN::SearchStats>(
                    stats,
                    [](const SPANN::SearchStats &ss) -> int
                    {
                        return ss.m_diskIOCount;
                    },
                    "%4d");

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "\n");

                if (!outputFile.empty())
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Start output to %s\n", outputFile.c_str());
                    OutputResult<ValueType>(outputFile, results, K);
                }

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                             "Recall@%d: %f MRR@%d: %f\n", K, recall, K, MRR);

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "\n");

                if (p_opts.m_recall_analysis)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Start recall analysis...\n");

                    std::shared_ptr<VectorIndex> headIndex = p_index->GetMemoryIndex();
                    SizeType sampleSize = numQueries < 100 ? numQueries : 100;
                    SizeType sampleK = headIndex->GetNumSamples() < 1000 ? headIndex->GetNumSamples() : 1000;
                    float sampleE = 1e-6f;

                    std::vector<SizeType> samples(sampleSize, 0);
                    std::vector<float> queryHeadRecalls(sampleSize, 0);
                    std::vector<float> truthRecalls(sampleSize, 0);
                    std::vector<int> shouldSelect(sampleSize, 0);
                    std::vector<int> shouldSelectLong(sampleSize, 0);
                    std::vector<int> nearQueryHeads(sampleSize, 0);
                    std::vector<int> annNotFound(sampleSize, 0);
                    std::vector<int> rngRule(sampleSize, 0);
                    std::vector<int> postingCut(sampleSize, 0);
                    for (int i = 0; i < sampleSize; i++)
                        samples[i] = COMMON::Utils::rand(numQueries);

#pragma omp parallel for schedule(dynamic)
                    for (int i = 0; i < sampleSize; i++)
                    {
                        COMMON::QueryResultSet<ValueType> queryANNHeads((const ValueType *)(querySet->GetVector(samples[i])), max(K, internalResultNum));
                        headIndex->SearchIndex(queryANNHeads);
                        float queryANNHeadsLongestDist = queryANNHeads.GetResult(internalResultNum - 1)->Dist;

                        COMMON::QueryResultSet<ValueType> queryBFHeads((const ValueType *)(querySet->GetVector(samples[i])), max(sampleK, internalResultNum));
                        for (SizeType y = 0; y < headIndex->GetNumSamples(); y++)
                        {
                            float dist = headIndex->ComputeDistance(queryBFHeads.GetQuantizedTarget(), headIndex->GetSample(y));
                            queryBFHeads.AddPoint(y, dist);
                        }
                        queryBFHeads.SortResult();

                        {
                            std::vector<bool> visited(internalResultNum, false);
                            for (SizeType y = 0; y < internalResultNum; y++)
                            {
                                for (SizeType z = 0; z < internalResultNum; z++)
                                {
                                    if (visited[z])
                                        continue;

                                    if (fabs(queryANNHeads.GetResult(z)->Dist - queryBFHeads.GetResult(y)->Dist) < sampleE)
                                    {
                                        queryHeadRecalls[i] += 1;
                                        visited[z] = true;
                                        break;
                                    }
                                }
                            }
                        }

                        std::map<int, std::set<int>> tmpFound; // headID->truths
                        p_index->DebugSearchDiskIndex(queryBFHeads, internalResultNum, sampleK, nullptr, &truth[samples[i]], &tmpFound);

                        for (SizeType z = 0; z < K; z++)
                        {
                            truthRecalls[i] += truth[samples[i]].count(queryBFHeads.GetResult(z)->VID);
                        }

                        for (SizeType z = 0; z < K; z++)
                        {
                            truth[samples[i]].erase(results[samples[i]].GetResult(z)->VID);
                        }

                        for (std::map<int, std::set<int>>::iterator it = tmpFound.begin(); it != tmpFound.end(); it++)
                        {
                            float q2truthposting = headIndex->ComputeDistance(querySet->GetVector(samples[i]), headIndex->GetSample(it->first));
                            for (auto vid : it->second)
                            {
                                if (!truth[samples[i]].count(vid))
                                    continue;

                                if (q2truthposting < queryANNHeadsLongestDist)
                                    shouldSelect[i] += 1;
                                else
                                {
                                    shouldSelectLong[i] += 1;

                                    std::set<int> nearQuerySelectedHeads;
                                    float v2vhead = headIndex->ComputeDistance(vectorSet->GetVector(vid), headIndex->GetSample(it->first));
                                    for (SizeType z = 0; z < internalResultNum; z++)
                                    {
                                        if (queryANNHeads.GetResult(z)->VID < 0)
                                            break;
                                        float v2qhead = headIndex->ComputeDistance(vectorSet->GetVector(vid), headIndex->GetSample(queryANNHeads.GetResult(z)->VID));
                                        if (v2qhead < v2vhead)
                                        {
                                            nearQuerySelectedHeads.insert(queryANNHeads.GetResult(z)->VID);
                                        }
                                    }
                                    if (nearQuerySelectedHeads.size() == 0)
                                        continue;

                                    nearQueryHeads[i] += 1;

                                    COMMON::QueryResultSet<ValueType> annTruthHead((const ValueType *)(vectorSet->GetVector(vid)), p_opts.m_debugBuildInternalResultNum);
                                    headIndex->SearchIndex(annTruthHead);

                                    bool found = false;
                                    for (SizeType z = 0; z < annTruthHead.GetResultNum(); z++)
                                    {
                                        if (nearQuerySelectedHeads.count(annTruthHead.GetResult(z)->VID))
                                        {
                                            found = true;
                                            break;
                                        }
                                    }

                                    if (!found)
                                    {
                                        annNotFound[i] += 1;
                                        continue;
                                    }

                                    // RNG rule and posting cut
                                    std::set<int> replicas;
                                    for (SizeType z = 0; z < annTruthHead.GetResultNum() && replicas.size() < p_opts.m_replicaCount; z++)
                                    {
                                        BasicResult *item = annTruthHead.GetResult(z);
                                        if (item->VID < 0)
                                            break;

                                        bool good = true;
                                        for (auto r : replicas)
                                        {
                                            if (p_opts.m_rngFactor * headIndex->ComputeDistance(headIndex->GetSample(r), headIndex->GetSample(item->VID)) < item->Dist)
                                            {
                                                good = false;
                                                break;
                                            }
                                        }
                                        if (good)
                                            replicas.insert(item->VID);
                                    }

                                    found = false;
                                    for (auto r : nearQuerySelectedHeads)
                                    {
                                        if (replicas.count(r))
                                        {
                                            found = true;
                                            break;
                                        }
                                    }

                                    if (found)
                                        postingCut[i] += 1;
                                    else
                                        rngRule[i] += 1;
                                }
                            }
                        }
                    }
                    float headacc = 0, truthacc = 0, shorter = 0, longer = 0, lost = 0, buildNearQueryHeads = 0, buildAnnNotFound = 0, buildRNGRule = 0, buildPostingCut = 0;
                    for (int i = 0; i < sampleSize; i++)
                    {
                        headacc += queryHeadRecalls[i];
                        truthacc += truthRecalls[i];

                        lost += shouldSelect[i] + shouldSelectLong[i];
                        shorter += shouldSelect[i];
                        longer += shouldSelectLong[i];

                        buildNearQueryHeads += nearQueryHeads[i];
                        buildAnnNotFound += annNotFound[i];
                        buildRNGRule += rngRule[i];
                        buildPostingCut += postingCut[i];
                    }

                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Query head recall @%d:%f.\n", internalResultNum, headacc / sampleSize / internalResultNum);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "BF top %d postings truth recall @%d:%f.\n", sampleK, truthK, truthacc / sampleSize / truthK);

                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                                 "Percent of truths in postings have shorter distance than query selected heads: %f percent\n",
                                 shorter / lost * 100);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                                 "Percent of truths in postings have longer distance than query selected heads: %f percent\n",
                                 longer / lost * 100);

                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                                 "\tPercent of truths no shorter distance in query selected heads: %f percent\n",
                                 (longer - buildNearQueryHeads) / lost * 100);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                                 "\tPercent of truths exists shorter distance in query selected heads: %f percent\n",
                                 buildNearQueryHeads / lost * 100);

                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                                 "\t\tRNG rule ANN search loss: %f percent\n", buildAnnNotFound / lost * 100);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                                 "\t\tPosting cut loss: %f percent\n", buildPostingCut / lost * 100);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                                 "\t\tRNG rule loss: %f percent\n", buildRNGRule / lost * 100);
                }
            }
        }
    }
}
