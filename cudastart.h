#ifndef CUDASTART_H
#define CUDASTART_H
#define CHECK(call)\
{\
  const cudaError_t error=call;\
  if(error!=cudaSuccess)\
  {\
      printf("ERROR: %s:%d,",__FILE__,__LINE__);\
      printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
      exit(1);\
  }\
}


#include <time.h>
#ifdef _WIN32
#	include <windows.h>
#else
#	include <sys/time.h>
#endif

double cpuSecond()
{
  struct timeval tp;
  gettimeofday(&tp,NULL);
  return((double)tp.tv_sec+(double)tp.tv_usec*1e-6);

}

void initialData(unsigned long long* ip,int size)
{
  time_t t;
  srand((unsigned )time(&t));
  for(int i=0;i<size;i++)
  {
    ip[i]=(unsigned long long)(rand()&0xffff)/1000.0f;
  }
}

void initDevice(int devNum)
{
  int dev = devNum;
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp,dev));
  printf("Using device %d: %s\n",dev,deviceProp.name);
  CHECK(cudaSetDevice(dev));

}
void checkResult(unsigned long long * hostRef,unsigned long long * gpuRef,const int N)
{

  for(int i=0;i<N;i++)
  {
    if(hostRef[i]!=gpuRef[i])
    {
      printf("Results don\'t match!\n");
      printf("%llu(hostRef[%d] )!= %llu(gpuRef[%d])\n",hostRef[i],i,gpuRef[i],i);
      return;
    }
  }
  printf("Check result success!\n");
}

#endif
