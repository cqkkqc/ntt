#include <cuda_runtime.h>
#include <iostream>
#include "cudastart.h"

//CPU对照组，用于对比加速比
void sumMatrix2DonCPU(float * MatA,float * MatB,float * MatC,int nx,int ny)
{
    float* a = MatA;
    float* b = MatB;
    float* c = MatC;
    for(int j=0; j<ny; j++)
    {
        for(int i=0; i<nx; i++)
        {
          c[i] = a[i]+b[i];
        }
        c += nx;
        b += nx;
        a += nx;
    }
}

//核函数，每一个线程计算矩阵中的一个元素。
__global__ void sumMatrix(float * MatA,float * MatB,float * MatC,int nx,int ny)
{
    int ix = threadIdx.x+blockDim.x*blockIdx.x;
    int iy = threadIdx.y+blockDim.y*blockIdx.y;
    int idx = ix+iy*nx;
    if (ix<nx && iy<ny)
    {
        MatC[idx] = MatA[idx]+MatB[idx];
    }
}

//主函数
int main(int argc,char** argv)
{
    int dev = 0;
    cudaDeviceProp devProp;
    CHECK(cudaGetDeviceProperties(&devProp, dev));
    std::cout << "使用GPU device " << dev << ": " << devProp.name << std::endl;
    std::cout << "SM的数量：" << devProp.multiProcessorCount << std::endl;
    std::cout << "每个线程块的共享内存大小：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    std::cout << "每个线程块的最大线程数：" << devProp.maxThreadsPerBlock << std::endl;
    std::cout << "每个SM的最大线程数：" << devProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "每个SM的最大线程束数：" << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
}