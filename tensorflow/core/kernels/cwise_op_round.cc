/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {
REGISTER5(UnaryOp, CPU, "Round", functor::round, Eigen::half, float, double,
          int32, int64);

#if GOOGLE_CUDA
REGISTER5(UnaryOp, GPU, "Round", functor::round, Eigen::half, float, double,
          int32, int64);
#endif

#ifdef TENSORFLOW_USE_SYCL
REGISTER2(UnaryOp, SYCL, "Round", functor::identity, int32, int64);
#define REGISTER_SYCL(type) \
  REGISTER(UnaryOp, SYCL, "Round", functor::round, type)
TF_CALL_SYCL_NUMBER_TYPES(REGISTER_SYCL);
#undef REGISTER_SYCL
#endif  // TENSORFLOW_USE_SYCL
}  // namespace tensorflow
