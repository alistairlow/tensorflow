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

#ifndef TENSORFLOW_KERNELS_CONV_OPS_H_
#define TENSORFLOW_KERNELS_CONV_OPS_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/util/tensor_format.h"

#if GOOGLE_CUDA
#include "tensorflow/core/kernels/conv_ops_gpu.h"
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

// Forward declaration.
class OpKernelContext;

template <typename Device, typename T>
struct LaunchConv2DOp {
  void operator()(OpKernelContext* ctx, bool use_cudnn, bool cudnn_use_autotune,
                  const Tensor& input, const Tensor& filter, int row_stride,
                  int col_stride, const Padding& padding, Tensor* output,
                  TensorFormat data_format);
};

#ifdef GOOGLE_CUDA
template <typename T>
struct LaunchConv2DOp<Eigen::GpuDevice, T> {
  void operator()(OpKernelContext* ctx, bool use_cudnn, bool cudnn_use_autotune,
                  const Tensor& input, const Tensor& filter, int row_stride,
                  int col_stride, const Padding& padding, Tensor* output,
                  TensorFormat data_format);
};
#endif  // GOOGLE_CUDA

// Used to keep track of persistent memory buffers used within the op.
// It uses malloc and free to avoid the time cost of initializing the memory.
template <class T, size_t size>
struct Im2ColBufferResource : public ResourceBase {
  Im2ColBufferResource<T, size>() {
    data = static_cast<T*>(port::Malloc(size * sizeof(T)));
  }
  ~Im2ColBufferResource<T, size>() { port::Free(data); }
  // This mutex ensures that only a single operation at a time is able to use
  // the buffer memory held by this resource.
  mutex mu;
  T* data;
  string DebugString() { return "Im2ColBufferResource"; }
};

template <class Device, class T, size_t size>
struct Im2ColBufferResourceSYCL : public ResourceBase {
  Im2ColBufferResourceSYCL<Device, T, size>(Device& d) {
    data = static_cast<T*>(d.allocate_temp(size * sizeof(T)));
  }
  mutex mu;
  T* data;
  string DebugString() { return "Im2ColBufferResourceSYCL"; }
};


}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_CONV_OPS_H
