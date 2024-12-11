#include <cuda_runtime.h>
#include <iostream>
#include "cudastart.h"
using namespace std;
typedef unsigned long long ULL;

const ULL mod=0xFFFFFFFF00000001;
const ULL G=17492915097719143606;
// const ULL mod=998244353;
// const ULL G=3;
const int bit=12;
ULL tot=1<<bit;
ULL qmi(ULL a,ULL b,ULL p)
{
    ULL res=1;
    while(b)
    {
        if(b&1)
        res=res*a%p;

        a=a*a%p;
        b>>=1;
    }
    return res;
}
__device__ ULL qmi_gpu(ULL a,ULL b,ULL p)
{
    ULL res=1;
    while(b)
    {
        if(b&1)
        res=res*a%p;

        a=a*a%p;
        b>>=1;
    }
    return res;
}

__global__ void NTT_kernal(ULL *a,ULL *out,int tot,int mid)
{   int idx=blockDim.x*blockIdx.x+threadIdx.x;
    if(idx>=tot)return;
    ULL w1=qmi_gpu(G,(mod-1)/(2*mid),mod);

    int k=idx%(2*mid);
    //printf("%d %d \n",idx,k);
    if(k<mid)
    {   
        ULL wk=qmi_gpu(w1,k,mod);
        ULL x=a[idx];
        ULL y=wk*a[idx+mid]%mod;
        out[idx]=(x+y)%mod;
    }
    else
    {
        ULL wk=qmi_gpu(w1,k-mid,mod);
        ULL x=a[idx-mid];
        ULL y=wk*a[idx]%mod;
        out[idx]=(x-y+mod)%mod;
    }
}
void NTT_GPU(ULL *a,ULL *out,int tot)
{   

    dim3 block(32,1);
    dim3 grid((tot-1)/block.x+1,1);
    ULL *array=(ULL*)malloc(tot*sizeof(ULL));
    for(int mid=1;mid<tot;mid*=2)
    {
        NTT_kernal<<<grid,block>>>(a,out,tot,mid);
        CHECK(cudaMemcpy(a,out,tot*sizeof(ULL),cudaMemcpyDeviceToDevice));
    }
}
void NTT(ULL a[],ULL rev[])
{
    for(int mid=1;mid<tot;mid*=2)
    {
        ULL w1=qmi(G,(mod-1)/(2*mid),mod);
        for(int i=0;i<tot;i+=mid*2)
        {
            ULL wk=1;
            for(int j=0;j<mid;j++,wk=wk*w1%mod)
            {
                ULL x=a[i+j];
                ULL y=wk*a[i+j+mid]%mod;
                a[i+j]=(x+y)%mod;
                a[i+j+mid]=(x-y+mod)%mod;
            }
        }
    }
}

int main()
{
    printf("starting..\n");
    initDevice(2);
    ULL nBytes=tot*sizeof(ULL);

    ULL *array=(ULL*)malloc(nBytes);
    ULL *rev=(ULL*)malloc(nBytes);
    ULL *array_out=(ULL*)malloc(nBytes);
    initialData(array,tot);
    for(int i=0;i<tot;i++) rev[i]=(rev[i>>1]>>1)|((i&1)<<(bit-1));
    for(int i=0;i<tot;i++)
        if(i<rev[i]) swap(array[i],array[rev[i]]); 
    ULL *array_device=NULL;

    ULL *output_device=NULL;
    CHECK(cudaMalloc((void**)&array_device,nBytes));
    CHECK(cudaMalloc((void**)&output_device,nBytes));
    CHECK(cudaMemcpy(array_device,array,nBytes,cudaMemcpyHostToDevice));






    double gpustart=cpuSecond();
    NTT_GPU(array_device,output_device,tot);
    CHECK(cudaDeviceSynchronize());
    double gpuTime=cpuSecond()-gpustart;
    printf("GPU execution Time:%f\n",gpuTime);
    
    CHECK(cudaMemcpy(array_out,array_device,nBytes,cudaMemcpyDeviceToHost));

    double cpustart=cpuSecond();
    NTT(array,rev);
    double cpuTime=cpuSecond()-cpustart;
    printf("CPU execution Time:%f\n",cpuTime);


    checkResult(array,array_out,tot);
    
    cudaFree(array_device);
    cudaFree(output_device);

    free(array);
    free(rev);
    free(array_out);
    cudaDeviceReset();
    
}