#include <humming/datatype/dequant.cuh>
__global__ void k(const uint32_t* q, uint32_t* o) {
  uint32_t r[8];
  dequant<UInt4, Float16, false, false, 1>(q, r, 0, nullptr);
  for (int i=0;i<4;i++) o[threadIdx.x*4+i]=r[i];
}
