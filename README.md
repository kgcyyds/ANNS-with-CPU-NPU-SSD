## **Requirements**

* swig >= 4.0.2
* cmake >= 3.12.0
* boost >= 1.67.0
* cuda >= 11.6.0

## **Getting Started**

### **Fusion**

> Install dependency
```bash
sudo apt install cmake
sudo apt install swig
sudo apt install libboost-all-dev
```

```bash
Please from [NVIDIA CUDA Toolkit] (https://developer.nvidia.com/cuda-toolkit-archive) to download and install version > = 11.6.0 CUDA Toolkit. Follow the installation instructions and make sure the environment variables are set correctly.
```

> Build Fusion
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
```

## **Usage of build**

Dataset download address: https://big-ann-benchmarks.com/neurips21.html#bench-datasets

### **1.Quantizer Training and Quantizing Vectors**
> Use Quantizer.exe to train PQQuantizer and output quantizer & quantized vectors:

  ```bash
  Usage:
  ./Quantizer [options]
  Options:
  -d, --dimension <value>                 Dimension of vector.
  -v, --vectortype <value>                Input vector data type. Default is float.
  -f, --filetype <value>                  Input file type (DEFAULT, TXT, XVEC). Default is DEFAULT.
  -i, --input <value>                     Input raw data.
  -o, --output <value>                    Output quantized vectors.
  -om, --outputmeta <value>               Output metadata.
  -omi, --outputmetaindex <value>         Output metadata index.

  -t, --thread <value>                    Thread Number.
  -dl, --delimiter <value>                Vector delimiter.
  -norm, --normalized <value>             Vector is normalized.
  -oq, --outputquantizer <value>          Output quantizer.
  -qt, --quantizer <value>                Quantizer type.
  -qd, --quantizeddim <value>             Quantized Dimension.
  -ts, --train_samples <value>            Number of samples for training.
  -debug, --debug <value>                 Print debug information.
  -kml, --lambda <value>                  Kmeans lambda parameter.

  Example for SIFT1B: ./quantizer -d 128 -v uint8 -f DEFAULT -i sift1b/base.1B.u8bin -o sift1b/base_pq.1B.u8bin -oq sift1b/quantizer -qt PQQuantizer -qd 32 -ts 10000000
  ```

### **2.Fusion Index Build**

For sift1b dataset, use the default configuration below (Release/buildconfig.ini) and run ./ssdserving buildconfig.ini:
```
[Base]
ValueType=UInt8
DistCalcMethod=L2
IndexAlgoType=BKT
Dim=128
VectorPath=sift1b/base.1B.u8bin
VectorType=DEFAULT
QueryPath==sift1b/query.public.10K.u8bin
QueryType=DEFAULT
WarmupPath==sift1b/query.public.10K.u8bin
WarmupType=DEFAULT
TruthPath==sift1b/public_query_gt100.bin
TruthType=DEFAULT
IndexDirectory=Fusion/SIFT1B
QuantizerPQFilePath=sift1b/quantizer
QuantizerVectorFilePath=sift1b/base_pq.1B.u8bin

[SelectHead]
isExecute=true
TreeNumber=1
BKTKmeansK=32
BKTLeafSize=8
SamplesNumber=1000
SaveBKT=false
SelectThreshold=0
SplitFactor=0
SplitThreshold=0
Ratio=0.06
NumberOfThreads=45
BKTLambdaFactor=1.0

[BuildHead]
isExecute=true
NeighborhoodSize=32
TPTNumber=32
TPTLeafSize=2000
MaxCheck=16324
MaxCheckForRefineGraph=16324
RefineIterations=3
NumberOfThreads=45
BKTLambdaFactor=-1.0

[BuildSSDIndex]
isExecute=true
BuildSsdIndex=true
InternalResultNum=64
ReplicaCount=8
PostingVectorLimit=200
NumberOfThreads=45
MaxCheck=16324
TmpDir=Fusion/SIFT1B

[SearchSSDIndex]
isExecute=false
BuildSsdIndex=false
InternalResultNum=140
NumberOfThreads=60
HashTableExponent=4
ResultNum=50
MaxCheck=1024
MaxDistRatio=8.0
QueryCountLimit=10000
ReadRatio=0
Rerank=10
EnableGPU=true

```

## **Usage of search**

For sift1b dataset, use the default configuration below (Release\searchconfig.ini) and run ./ssdserving searchconfig.ini:
```
[Base]
ValueType=UInt8
DistCalcMethod=L2
IndexAlgoType=BKT
Dim=128
VectorPath=sift1b/base.1B.u8bin
VectorType=DEFAULT
QueryPath==sift1b/query.public.10K.u8bin
QueryType=DEFAULT
WarmupPath==sift1b/query.public.10K.u8bin
WarmupType=DEFAULT
TruthPath==sift1b/public_query_gt100.bin
TruthType=DEFAULT
IndexDirectory=Fusion/SIFT1B
QuantizerPQFilePath=sift1b/quantizer
QuantizerVectorFilePath=sift1b/base_pq.1B.u8bin

[SelectHead]
isExecute=false
TreeNumber=1
BKTKmeansK=32
BKTLeafSize=8
SamplesNumber=1000
SaveBKT=false
SelectThreshold=0
SplitFactor=0
SplitThreshold=0
Ratio=0.06
NumberOfThreads=45
BKTLambdaFactor=1.0

[BuildHead]
isExecute=false
NeighborhoodSize=32
TPTNumber=32
TPTLeafSize=2000
MaxCheck=16324
MaxCheckForRefineGraph=16324
RefineIterations=3
NumberOfThreads=45
BKTLambdaFactor=-1.0

[BuildSSDIndex]
isExecute=false
BuildSsdIndex=true
InternalResultNum=64
ReplicaCount=8
PostingVectorLimit=200
NumberOfThreads=45
MaxCheck=16324
TmpDir=Fusion/SIFT1B

[SearchSSDIndex]
isExecute=true
BuildSsdIndex=false
InternalResultNum=140
NumberOfThreads=20
HashTableExponent=4
ResultNum=50
MaxCheck=1024
MaxDistRatio=8.0
QueryCountLimit=10000
ReadRatio=0
Rerank=10
EnableGPU=true

```

For sift1b dataset, we verify that different accuracies can be obtained by setting the following parameters:

| SearchInternalResultNum  |  ResultNum | Rerank | Rerall |
|---|---|---|---|
| 140 | 50 | 10 | 0.90 |
| 180 | 57 | 10 | 0.92 |
| 270 | 62 | 10 | 0.94 |
| 270 | 75 | 10 | 0.95 |
| 320 | 87 | 10 | 0.96 |
| 500 | 150 | 10 | 0.98 |

## **Parameters**

> Common Parameters

|  ParametersName | type  |  default | definition|
|---|---|---|---|
| Samples | int | 1000 | how many points will be sampled to do tree node split |
|TPTNumber | int | 32 | number of TPT trees to help with graph construction |
|TPTLeafSize | int | 2000 | TPT tree leaf size |
NeighborhoodSize | int | 32 | number of neighbors each node has in the neighborhood graph |
|GraphNeighborhoodScale | int | 2 | number of neighborhood size scale in the build stage |
|CEF | int | 1000 | number of results used to construct RNG | 
|MaxCheckForRefineGraph| int | 10000 | how many nodes each node will visit during graph refine in the build stage | 
|NumberOfThreads | int | 1 | number of threads to uses for speed up the build |
|DistCalcMethod | string | Cosine | choose from Cosine and L2 |
|MaxCheck | int | 8192 | how many nodes will be visited for a query in the search stage

> Search Parameters

|  ParametersName | type  |  default | definition|
|---|---|---|---|
|SearchInternalResultNum | int | 64 | number of results for tree and graph searches |
|ResultNum | int | 20 | number of full precision computations that need to be performed after the lossy computation |
|ReadRatio | float | 1 | ratio of the index on disk cached to memory |
|rerank | int | 5 | number of final top-K |
|EnableGPU | bool | false | whether to use GPU to accelerate distance calculations |

> BKT

|  ParametersName | type  |  default | definition|
|---|---|---|---|
| BKTNumber | int | 1 | number of BKT trees |
| BKTKMeansK | int | 32 | how many childs each tree node has |

> KDT

|  ParametersName | type  |  default | definition|
|---|---|---|---|
| KDTNumber | int | 1 | number of KDT trees |

> Parameters that will affect the index size
* NeighborhoodSize
* BKTNumber
* KDTNumber

> Parameters that will affect the index build time
* NumberOfThreads
* TPTNumber
* TPTLeafSize
* GraphNeighborhoodScale
* CEF
* MaxCheckForRefineGraph

> Parameters that will affect the index quality
* TPTNumber
* TPTLeafSize
* GraphNeighborhoodScale
* CEF
* MaxCheckForRefineGraph
* NeighborhoodSize
* KDTNumber

> Parameters that will affect search latency and recall
* MaxCheck
* SearchInternalResultNum
* ResultNum