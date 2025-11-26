## **Quick start**

### **Quantizer Training and Quantizing Vectors**
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

  Example: ./quantizer -d 128 -v uint8 -f DEFAULT -i sift1b/base.1B.u8bin -o sift1b/base_pq.1B.u8bin -oq sift1b/quantizer -qt PQQuantizer -qd 32 -ts 10000000
  ```

For sift1b dataset, use the default configuration below (buildconfig.ini) and run .\SSDServing.exe buildconfig.ini:
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
IndexDirectory=sift1b
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
TmpDir=/tmp/

[SearchSSDIndex]
isExecute=true
BuildSsdIndex=false
InternalResultNum=140
NumberOfThreads=1
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


### **Input File Format**

#### DEFAULT (Binary)
> Input raw data for index build and input query file for index search (suppose vector dimension is 3):

```
<4 bytes int representing num_vectors><4 bytes int representing num_dimension>
<num_vectors * num_dimension * sizeof(data type) bytes raw data>
```

> Truth file to calculate recall (suppose K is 2):
```
< 4 bytes int representing num_queries><4 bytes int representing K>
<num_queries * K * sizeof(int) representing truth neighbor ids>
```

#### TXT
> Input raw data for index build and input query file for index search (suppose vector dimension is 3):

```
<metadata1>\t<v11>|<v12>|<v13>|
<metadata2>\t<v21>|<v22>|<v23>|
... 
```
where each line represents a vector with its metadata and its value separated by a tab space. Each dimension of a vector is separated by | or use --delimiter to define the separator.

> Truth file to calculate recall (suppose K is 2):
```
<t11> <t12>
<t21> <t22>
...
```
where each line represents the K nearest neighbors of a query separated by a blank space. Each neighbor is given by its vector id.

### **Meta Files Format**
> Data for index build to provide the metadata of the vectors. There are two files:

#### meta.bin
```
<vector 1 meta><vector 2 meta>...
```

#### metaindex.bin
```
<4 bytes int representing num_vectors><sizeof(uint64_t)*(num_vectors + 1) bytes representing position_array where the meta start and end positions in meta.bin for vector i is position_array[i] and position_array[i+1] respectively> 
```

### **Quantizer File Format**
> Data for using PQ quantizer in index build and index search
```
<1 byte uint8 representing QuantizerType -- 0: NONE, 1: PQ, 2: OPQ><1 byte uint8 representing ReconstructDataType -- 0: int8, 1: uint8, 2: int16, 3: float><4 bytes int representing num_codebooks><4 bytes int representing entries_per_codebook><4 bytes int representing codebook_dim>
<sizeof(ReconstructType)*num_codebooks*entries_per_codebook*codebook_dim representing codebook entries>[<sizeof(float) * reconstruct_dim * reconstruct_dim representing OPQ rotation matrix, row major order>]
```

Note that `num_codebooks*codebook_dim=full_dim`. The current PQ implementation only supports `entries_per_codebook <= 256` (i.e. quantizing to `byte`).

### **Server**
```bash
Usage:
./Server [options]
Options: 
  -m, --mode <value>              Service mode, interactive or socket.
  -c, --config <value>            Configure file of the index

Write a server configuration file service.ini as follows:

[Service]
ListenAddr=0.0.0.0
ListenPort=8000
ThreadNumber=8
SocketThreadNumber=8

[QueryConfig]
DefaultMaxResultNumber=6
DefaultSeparator=|

[Index]
List=BKT

[Index_BKT]
IndexFolder=BKT_gist
```

### **Client**
```bash
Usage:
./Client [options]
Options:
-s, --server                       Server address
-p, --port                         Server port
-t,                                Search timeout
-cth,                              Client Thread Number
-sth                               Socket Thread Number
```

### **Aggregator**
```bash
Usage:
./Aggregator

Write Aggregator.ini as follows:

[Service]
ListenAddr=0.0.0.0
ListenPort=8100
ThreadNumber=8
SocketThreadNumber=8

[Servers]
Number=2

[Server_0]
Address=127.0.0.1
Port=8000

[Server_1]
Address=127.0.0.1
Port=8010
```

### **Python Support**
> Singlebox PythonWrapper
 ```python
 
import SPTAG
import numpy as np

n = 100
k = 3
r = 3

def testBuild(algo, distmethod, x, out):
    i = SPTAG.AnnIndex(algo, 'Float', x.shape[1])
    i.SetBuildParam("NumberOfThreads", '4', "Index")
    i.SetBuildParam("DistCalcMethod", distmethod, "Index")
    if i.Build(x, x.shape[0], False):
        i.Save(out)

def testBuildWithMetaData(algo, distmethod, x, s, out):
    i = SPTAG.AnnIndex(algo, 'Float', x.shape[1])
    i.SetBuildParam("NumberOfThreads", '4', "Index")
    i.SetBuildParam("DistCalcMethod", distmethod, "Index")
    if i.BuildWithMetaData(x, s, x.shape[0], False, False):
        i.Save(out)

def testSearch(index, q, k):
    j = SPTAG.AnnIndex.Load(index)
    for t in range(q.shape[0]):
        result = j.Search(q[t], k)
        print (result[0]) # ids
        print (result[1]) # distances

def testSearchWithMetaData(index, q, k):
    j = SPTAG.AnnIndex.Load(index)
    j.SetSearchParam("MaxCheck", '1024', "Index")
    for t in range(q.shape[0]):
        result = j.SearchWithMetaData(q[t], k)
        print (result[0]) # ids
        print (result[1]) # distances
        print (result[2]) # metadata

def testAdd(index, x, out, algo, distmethod):
    if index != None:
        i = SPTAG.AnnIndex.Load(index)
    else:
        i = SPTAG.AnnIndex(algo, 'Float', x.shape[1])
    i.SetBuildParam("NumberOfThreads", '4', "Index")
    i.SetBuildParam("DistCalcMethod", distmethod, "Index")
    if i.Add(x, x.shape[0], False):
        i.Save(out)

def testAddWithMetaData(index, x, s, out, algo, distmethod):
    if index != None:
        i = SPTAG.AnnIndex.Load(index)
    else:
        i = SPTAG.AnnIndex(algo, 'Float', x.shape[1])
    i.SetBuildParam("NumberOfThreads", '4', "Index")
    i.SetBuildParam("DistCalcMethod", distmethod, "Index")
    if i.AddWithMetaData(x, s, x.shape[0], False, False):
        i.Save(out)

def testDelete(index, x, out):
    i = SPTAG.AnnIndex.Load(index)
    ret = i.Delete(x, x.shape[0])
    print (ret)
    i.Save(out)
    
def Test(algo, distmethod):
    x = np.ones((n, 10), dtype=np.float32) * np.reshape(np.arange(n, dtype=np.float32), (n, 1))
    q = np.ones((r, 10), dtype=np.float32) * np.reshape(np.arange(r, dtype=np.float32), (r, 1)) * 2
    m = ''
    for i in range(n):
        m += str(i) + '\n'

    m = m.encode()

    print ("Build.............................")
    testBuild(algo, distmethod, x, 'testindices')
    testSearch('testindices', q, k)
    print ("Add.............................")
    testAdd('testindices', x, 'testindices', algo, distmethod)
    testSearch('testindices', q, k)
    print ("Delete.............................")
    testDelete('testindices', q, 'testindices')
    testSearch('testindices', q, k)

    print ("AddWithMetaData.............................")
    testAddWithMetaData(None, x, m, 'testindices', algo, distmethod)
    testSearchWithMetaData('testindices', q, k)
    print ("Delete.............................")
    testDelete('testindices', q, 'testindices')
    testSearchWithMetaData('testindices', q, k)

if __name__ == '__main__':
    Test('BKT', 'L2')
    Test('KDT', 'L2')

 ```

 > Python Client Wrapper, Suppose there is a sever run at 127.0.0.1:8000 serving ten-dimensional vector datasets:
 ```python
import SPTAGClient
import numpy as np
import time

def testSPTAGClient():
    index = SPTAGClient.AnnClient('127.0.0.1', '8000')
    while not index.IsConnected():
        time.sleep(1)
    index.SetTimeoutMilliseconds(18000)

    q = np.ones((10, 10), dtype=np.float32)
    for t in range(q.shape[0]):
        result = index.Search(q[t], 6, 'Float', False)
        print (result[0])
        print (result[1])

if __name__ == '__main__':
    testSPTAGClient()

 ```
 
 ### **C# Support**
> Singlebox CsharpWrapper
 ```C#
using System;
using System.Text;

public class test
{
    static int dimension = 10;
    static int n = 10;
    static int k = 3;

    static byte[] createFloatArray(int n)
    {
        byte[] data = new byte[n * dimension * sizeof(float)];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < dimension; j++)
                Array.Copy(BitConverter.GetBytes((float)i), 0, data, (i * dimension + j) * sizeof(float), 4);
        return data;
    }

    static byte[] createMetadata(int n)
    {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < n; i++)
            sb.Append(i.ToString() + '\n');
        return Encoding.ASCII.GetBytes(sb.ToString());
    }

    static void Main()
    {
        {
            AnnIndex idx = new AnnIndex("BKT", "Float", dimension);
            idx.SetBuildParam("DistCalcMethod", "L2", "Index");
            byte[] data = createFloatArray(n);
            byte[] meta = createMetadata(n);
            idx.BuildWithMetaData(data, meta, n, false, false);
            idx.Save("testcsharp");
        }

        AnnIndex index = AnnIndex.Load("testcsharp");
        BasicResult[] res = index.SearchWithMetaData(createFloatArray(1), k);
        for (int i = 0; i < res.Length; i++)
            Console.WriteLine("result " + i.ToString() + ":" + res[i].Dist.ToString() + "@(" + res[i].VID.ToString() + "," + Encoding.ASCII.GetString(res[i].Meta) + ")"); 
        Console.WriteLine("test finish!");
    }
}

 ```

  
  
