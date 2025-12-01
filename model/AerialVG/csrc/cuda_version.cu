#include <cuda_runtime_api.h>

namespace aerialvg {
int get_cudart_version() {
  return CUDART_VERSION;
}
} // namespace aerialvg
