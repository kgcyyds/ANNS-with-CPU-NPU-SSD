/**
 * @file add_custom.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_operator.h"

using namespace AscendC;

class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR set, GM_ADDR id, GM_ADDR table, GM_ADDR dist, int dim, int bigCoreNum, int bigCoreProcessNum, int smallCoreProcessNum)
    {
        int coreNum = GetBlockIdx();
        int globalBufferIndex = coreNum * bigCoreProcessNum;
        if (coreNum < bigCoreNum)
            this->processNum = bigCoreProcessNum;
        else
        {
            this->processNum = smallCoreProcessNum;
            globalBufferIndex -= (bigCoreProcessNum - smallCoreProcessNum) * (coreNum - bigCoreNum);
        }
        setGm.SetGlobalBuffer((__gm__ uint8_t *)set);
        idGm.SetGlobalBuffer((__gm__ int32_t *)id + globalBufferIndex, processNum);
        tableGm.SetGlobalBuffer((__gm__ float *)table, 256 * dim);
        distGm.SetGlobalBuffer((__gm__ float *)dist + globalBufferIndex, processNum);
        pipe.InitBuffer(inQueueSet, 1, processNum * dim * sizeof(uint8_t));
        pipe.InitBuffer(inQueueTable, 1, 256 * dim * sizeof(float));
        pipe.InitBuffer(outQueueDist, 1, processNum * sizeof(float));
        pipe.InitBuffer(B1, processNum * dim * sizeof(half));
        pipe.InitBuffer(B2, processNum * dim * sizeof(int32_t));
        pipe.InitBuffer(B3, processNum * sizeof(int32_t));
        pipe.InitBuffer(B4, processNum * sizeof(int32_t));
        pipe.InitBuffer(B5, processNum * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        LocalTensor<uint8_t> set = inQueueSet.AllocTensor<uint8_t>();
        LocalTensor<float> table = inQueueTable.AllocTensor<float>();
        LocalTensor<float> dist = outQueueDist.AllocTensor<float>();
        for (int i = 0; i < processNum; i ++)
        {
            int id = idGm.GetValue(i);
            DataCopy(set[i * dim], setGm[id * dim], dim);
        }
        DataCopy(table, tableGm, 256 * dim);
        inQueueSet.EnQue(set);
        inQueueTable.EnQue(table);
        set = inQueueSet.DeQue<uint8_t>();
        table = inQueueTable.DeQue<float>();
        LocalTensor<half> buf1 = B1.Get<half>();
        LocalTensor<int32_t> buf2 = B2.Get<int32_t>();
        LocalTensor<int32_t> buf3 = B3.Get<int32_t>();
        LocalTensor<int32_t> buf4 = B4.Get<int32_t>();
        LocalTensor<float> buf5 = B5.Get<float>();
        Cast(buf1, set, RoundMode::CAST_NONE, processNum * dim);
        Cast(buf2, buf1, RoundMode::CAST_ROUND, processNum * dim);
        CreateVecIndex(buf3, 0, processNum);
        Muls(buf3, buf3, (int32_t)(dim * sizeof(int32_t)), processNum);
        Duplicate(dist, (float)1, processNum);
        for (int i = 0; i < dim; i ++)
        {
            Gather(buf4, buf2, buf3.ReinterpretCast<uint32_t>(), (uint32_t)(i * sizeof(int32_t)), processNum);
            Muls(buf4, buf4, (int32_t)(sizeof(float)), processNum);
            Gather(buf5, table[i * 256], buf4.ReinterpretCast<uint32_t>(), 0, processNum);
            Add(dist, dist, buf5, processNum);
        }
        outQueueDist.EnQue(dist);
        dist = outQueueDist.DeQue<float>();
        DataCopyExtParams copyParams{(uint16_t)1, (uint32_t)(processNum * sizeof(float)), 0, 0, 0};
        DataCopyPad(distGm, dist, copyParams);
        inQueueSet.FreeTensor(set);
        inQueueTable.FreeTensor(table);
        outQueueDist.FreeTensor(dist);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueSet, inQueueTable;
    TQue<TPosition::VECOUT, 1> outQueueDist;
    TBuf<TPosition::VECCALC> B1, B2, B3, B4, B5;
    GlobalTensor<uint8_t> setGm;
    GlobalTensor<int32_t> idGm;
    GlobalTensor<float> tableGm, distGm;
    int dim, processNum;
};

extern "C" __global__ __aicore__ void add_custom(GM_ADDR set, GM_ADDR id, GM_ADDR table, GM_ADDR dist, int dim, int bigCoreNum, int bigCoreProcessNum, int smallCoreProcessNum)
{
    KernelAdd op;
    op.Init(set, id, table, dist, dim, bigCoreNum, bigCoreProcessNum, smallCoreProcessNum);
    op.Process();
}

void add_custom_do(uint32_t blockDim, void *stream, uint8_t *set, uint8_t *id, uint8_t *table, uint8_t *dist, int dim, int count)
{
    int bigCoreNum = count % blockDim;
    int smallCoreProcessNum = count / blockDim;
    int bigCoreProcessNum = (bigCoreNum == 0) ? smallCoreProcessNum : smallCoreProcessNum + 1;
    add_custom<<<blockDim, nullptr, stream>>>(set, id, table, dist, dim, bigCoreNum, bigCoreProcessNum, smallCoreProcessNum);
}