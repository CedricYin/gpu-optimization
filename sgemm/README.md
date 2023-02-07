# kernel
## sgemm_v0
baseline

## sgemm_v1
using shared memory

## sgemm_v2
using 2D thread tiles

## sgemm_v3
using register frag

## sgemm_v4
using float4

## sgemm_v5
using data prefetch


# 思路
## 优化框架
大迭代
    global -> shared
    小迭代
        shared -> register
回写进global

## float4 向量访存优化
为了小迭代时能够使用 float4 访问矩阵A，读取A时需要对A进行转置

## data prefetch (double buffer) 优化
双缓存使读写同步进行，实现数据预取，隐藏内存延迟：
双缓存通过申请双倍存储空间，将读和写分开，计算数据读取一块存储空间同时，可以同时向另一块内存写入下一轮依赖的数据，因此，只需要保证计算前待读取共享内存完成写入，即一次同步即可。

### 参数说明
write_index: 表示数据预取到哪个空间
load_index: 表示数据从哪个空间读出，与 write_index 二进制相反
以上连个参数都是针对 shared memory 的预取，每进行完一次大迭代，write_index 取反一次
而对于 register 的预取，小迭代次数对2取模即可实现，具体为：预取到1，对0读取；预取到0，对1读取 ...

### prefetch 代码框架
- 进行一次预取（预取到buffer0）：global -> shared；shared -> register
- 大迭代
- - 如果还有下一个迭代，则将下一个迭代的数据块，搬运到寄存器上暂存（这里对A不用转置）
- - BK - 1 次小迭代
- - - 将下一个小迭代的数据块，搬运到寄存器上
- - - 计算tile
- - 将存储在临时寄存器的数据搬运到shared memory中，完成shared memory的预取
- - 完成寄存器的预取
- - 将最后一个小迭代完成
- - write_index ^= 1
- 写回结果