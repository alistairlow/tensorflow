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

// See docs in ../ops/array_ops.cc

#ifndef TENSORFLOW_KERNELS_ONE_HOT_OP_H_
#define TENSORFLOW_KERNELS_ONE_HOT_OP_H_
// Generator definition for OneHotOp, must be compilable by nvcc.

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
#endif  // TENSORFLOW_USE_SYCL

namespace generator {

template <typename T, typename TI>
class OneGenerator {
 public:
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  OneGenerator(const typename TTypes<TI>::ConstMatrix& indices,
               const typename TTypes<T>::ConstScalar& on_value,
               const typename TTypes<T>::ConstScalar& off_value)
      : indices_(indices), on_value_(on_value), off_value_(off_value) {}

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T
  operator()(const Eigen::array<Eigen::DenseIndex, 3>& pre_depth_suff) const {
    return (indices_(pre_depth_suff[0], pre_depth_suff[2]) == pre_depth_suff[1])
               ? on_value_()
               : off_value_();
  }

 private:
  const typename TTypes<TI>::ConstMatrix indices_;
  const typename TTypes<T>::ConstScalar on_value_;
  const typename TTypes<T>::ConstScalar off_value_;
};

#ifdef TENSORFLOW_USE_SYCL
template <typename TI>
struct OneGeneratorSYCL {
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE TI
  operator()(const Eigen::array<Eigen::DenseIndex, 3>& pre_depth_suff) const {
    return pre_depth_suff[1];
  }
};
#endif  // TENSORFLOW_USE_SYCL

}  // namespace generator

namespace functor {

template <typename Device, typename T, typename TI>
struct OneHot {
  EIGEN_ALWAYS_INLINE static void Compute(
      const Device& d, const typename TTypes<TI>::ConstMatrix& indices,
      const typename TTypes<T>::ConstScalar& on_value,
      const typename TTypes<T>::ConstScalar& off_value,
      typename TTypes<T, 3>::Tensor* output) {
    generator::OneGenerator<T, TI> generator(indices, on_value, off_value);
    output->device(d) = output->generate(generator);
  }
};

#ifdef TENSORFLOW_USE_SYCL
template <typename T, typename TI>
struct OneHot<SYCLDevice, T, TI> {
  EIGEN_ALWAYS_INLINE static void Compute(
      const SYCLDevice& d, const typename TTypes<TI>::ConstMatrix& indices,
      const typename TTypes<T>::ConstScalar& on_value,
      const typename TTypes<T>::ConstScalar& off_value,
      typename TTypes<T, 3>::Tensor* output) {
    auto output_dims = output->dimensions();
#if !defined(EIGEN_HAS_INDEX_LIST)
    Eigen::Tensor<Eigen::DenseIndex, 3>::Dimensions reshape_3d{{1, 1, 1}};
    Eigen::Tensor<Eigen::DenseIndex, 3>::Dimensions
      reshape_indices{{output_dims[0], 1, output_dims[2]}};
    Eigen::array<Eigen::DenseIndex, 3>
      broadcast_indices{{1, output_dims[1], 1}};
#else
    Eigen::IndexList<Eigen::type2index<1>,
                     Eigen::type2index<1>,
                     Eigen::type2index<1> > reshape_3d;
    Eigen::IndexList<Eigen::DenseIndex,
                     Eigen::type2index<1>,
                     Eigen::DenseIndex> reshape_indices;
    reshape_indices.set(0, output_dims[0]);
    reshape_indices.set(2, output_dims[2]);
    Eigen::IndexList<Eigen::type2index<1>,
                     Eigen::DenseIndex,
                     Eigen::type2index<1> > broadcast_indices;
    broadcast_indices.set(1, output_dims[1]);
#endif
    auto indices_3d = indices.reshape(reshape_indices)
                             .broadcast(broadcast_indices);
    auto on_value_3d = on_value.reshape(reshape_3d).broadcast(output_dims);
    auto off_value_3d = off_value.reshape(reshape_3d).broadcast(output_dims);

    generator::OneGeneratorSYCL<TI> generator;
    output->device(d) = (indices_3d == indices_3d.generate(generator))
                          .select(on_value_3d, off_value_3d);
  }
};
#endif  // TENSORFLOW_USE_SYCL

}  // namespace functor

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_ONE_HOT_OP_H_
