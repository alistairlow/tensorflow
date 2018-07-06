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

#ifndef TENSORFLOW_KERNELS_GATHER_ND_OP_CPU_IMPL_H_
#define TENSORFLOW_KERNELS_GATHER_ND_OP_CPU_IMPL_H_

// Specialization of GatherNdSlice to CPU

#define EIGEN_USE_THREADS

#include <atomic>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/gather_nd_op.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/util.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
#endif  // TENSORFLOW_USE_SYCL

namespace generator {

template <typename T, typename Index, int IXDIM>
class GatherNdSliceGenerator {
 public:
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE GatherNdSliceGenerator(
      const Index slice_size, typename TTypes<Index>::ConstMatrix Tindices,
      typename TTypes<T, IXDIM + 1>::ConstTensor Tparams,
      typename TTypes<T>::Matrix Tout, std::atomic<Index>* error_loc)
      : slice_size_(slice_size),
        Tindices_(Tindices),
        Tparams_(Tparams),
        Tout_(Tout),
        error_loc_(error_loc) {}

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool GenerateIndices(
      const Index loc, Eigen::array<Eigen::DenseIndex, IXDIM + 1>* ix) const {
    (*ix)[IXDIM] = 0;
    bool out_of_bounds = false;
    for (int i = 0; i < IXDIM; ++i) {
      const Index ix_i = internal::SubtleMustCopy(Tindices_(loc, i));
      (*ix)[i] = ix_i;
      out_of_bounds |= !FastBoundsCheck(ix_i, Tparams_.dimension(i));
    }
    return out_of_bounds;
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE int32
  operator()(const Eigen::array<Eigen::DenseIndex, 1>& loc_array) const {
    const Index loc = loc_array[0];
    Eigen::array<Eigen::DenseIndex, IXDIM + 1> ix;
    Eigen::array<Eigen::DenseIndex, 2> ix_out;
    ix_out[0] = loc;
    ix_out[1] = 0;
    const bool out_of_bounds = GenerateIndices(loc, &ix);
    if (TF_PREDICT_FALSE(out_of_bounds)) {
      error_loc_->store(loc);
      std::fill_n(&Tout_(ix_out), slice_size_, T());
    } else {
      std::copy_n(&Tparams_(ix), slice_size_, &Tout_(ix_out));
    }

    return static_cast<int32>(0);  // Return something...
  }

 private:
  const Index slice_size_;
  const typename TTypes<Index>::ConstMatrix Tindices_;
  const typename TTypes<T, IXDIM + 1>::ConstTensor Tparams_;
  mutable typename TTypes<T>::Matrix Tout_;
  std::atomic<Index>* error_loc_;
};

}  // namespace generator

namespace functor {

template <typename T, typename Index, int IXDIM>
struct GatherNdSlice<CPUDevice, T, Index, IXDIM> {
  Index operator()(const CPUDevice& d, const Index slice_size,
                   typename TTypes<int32>::Scalar Tscratch,
                   typename TTypes<T, IXDIM + 1>::ConstTensor Tparams,
                   typename TTypes<Index>::ConstMatrix Tindices,
                   typename TTypes<T>::Matrix Tout) {
    std::atomic<Index> error_loc(-1);

    const Eigen::DenseIndex batch_size = Tindices.dimension(0);
#if !defined(EIGEN_HAS_INDEX_LIST)
    Eigen::Tensor<Eigen::DenseIndex, 1>::Dimensions reshape_dims{{ 1 }};
    Eigen::array<Eigen::DenseIndex, 1> broadcast_dims{{ batch_size }};
#else
    Eigen::IndexList<Eigen::type2index<1> > reshape_dims;
    Eigen::IndexList<Eigen::DenseIndex> broadcast_dims;
    broadcast_dims.set(0, batch_size);
#endif
    generator::GatherNdSliceGenerator<T, Index, IXDIM> gather_nd_generator(
        slice_size, Tindices, Tparams, Tout, &error_loc);
    Tscratch.device(d) = Tscratch.reshape(reshape_dims)
                             .broadcast(broadcast_dims)
                             .generate(gather_nd_generator)
                             .sum();

    // error_loc() returns -1 if there's no out-of-bounds index,
    // otherwise it returns the location of an OOB index in Tindices.
    return error_loc.load();
  }
};

#define REGISTER_GATHER_ND_FULL(dev, T, Index)                                \
  template Index GatherNdSlice<dev##Device, T, Index, CPU_PROVIDED_IXDIM>::   \
  operator()(const dev##Device& d, const Index slice_size,                    \
             typename TTypes<int32>::Scalar Tscratch,                         \
             typename TTypes<T, CPU_PROVIDED_IXDIM + 1>::ConstTensor Tparams, \
             typename TTypes<Index>::ConstMatrix Tindices,                    \
             typename TTypes<T>::Matrix Tout);

#define REGISTER_GATHER_ND_ALL_INDICES(dev, type) \
  REGISTER_GATHER_ND_FULL(dev, type, int32);      \
  REGISTER_GATHER_ND_FULL(dev, type, int64)

#define REGISTER_GATHER_ND_CPU(type) REGISTER_GATHER_ND_ALL_INDICES(CPU, type)

TF_CALL_ALL_TYPES(REGISTER_GATHER_ND_CPU);

#undef REGISTER_GATHER_ND_CPU

#ifdef TENSORFLOW_USE_SYCL

template <typename T, typename Index, int IXDIM>
struct GatherNdSlice<SYCLDevice, T, Index, IXDIM> {
 private:
  bool FlattenIdx(const typename TTypes<Index>::ConstMatrix& Tindices,
                  const Eigen::array<Eigen::DenseIndex, IXDIM + 1>& params_dims,
                  Eigen::DenseIndex loc, Eigen::DenseIndex& flat_idx,
                  Index& error_loc) {
    flat_idx = 0;
    bool out_of_bounds = false;
    for (int i = 0; i < IXDIM; ++i) {
      const Index ix_i = internal::SubtleMustCopy(Tindices(loc, i));
      flat_idx = (flat_idx + ix_i) * params_dims[i + 1];
      out_of_bounds |= !FastBoundsCheck(ix_i, params_dims[i]);
    }
    if (TF_PREDICT_FALSE(out_of_bounds))
      error_loc = loc;
    return out_of_bounds;
  }

  void MemCpyOrSet(const SYCLDevice& d, Index size, T* dst_ptr,
                   const T* src_ptr, Eigen::DenseIndex out_dim1,
                   bool out_of_bounds, Eigen::DenseIndex loc,
                   Eigen::DenseIndex act_flat_idx,
                   Eigen::DenseIndex nb_continuous) {
    dst_ptr += loc * out_dim1;
    size *= nb_continuous * sizeof(T);
    if (out_of_bounds)
      d.memset(dst_ptr, T(0), size);
    else
      d.memcpy(dst_ptr, src_ptr + act_flat_idx, size);
  }

 public:
  Index operator()(const SYCLDevice& d, const Index slice_size,
                   typename TTypes<int32>::Scalar,
                   typename TTypes<T, IXDIM + 1>::ConstTensor Tparams,
                   typename TTypes<Index>::ConstMatrix Tindices,
                   typename TTypes<T>::Matrix Tout) {
    // Run as litte memcpy as possible by concatenating the calls with
    // adjacent indices
    const Eigen::DenseIndex batch_size = Tindices.dimension(0);
    Eigen::DenseIndex out_dim1 = Tout.dimension(1);
    const auto& params_dims = Tparams.dimensions();

    auto src_ptr = static_cast<const T*>(Tparams.data());
    auto dst_ptr = static_cast<T*>(Tout.data());

    Index error_loc(-1);
    Eigen::DenseIndex act_loc(0);
    Eigen::DenseIndex next_loc(0);
    Eigen::DenseIndex nb_continuous(1);
    Eigen::DenseIndex act_flat_idx;
    Eigen::DenseIndex next_flat_idx;
    bool act_out_of_bounds = FlattenIdx(Tindices, params_dims, next_loc++,
                                        act_flat_idx, error_loc);
    bool next_out_of_bounds;
    while (next_loc < batch_size) {
      // Find next in bounds flatten idx continuous to previous flat_idx
      do {
        next_out_of_bounds = FlattenIdx(Tindices, params_dims, next_loc++,
                                        next_flat_idx, error_loc);
        if (act_out_of_bounds != next_out_of_bounds ||
            next_flat_idx != act_flat_idx + slice_size * nb_continuous)
          break;
        ++nb_continuous;
      } while (next_loc < batch_size);
      MemCpyOrSet(d, slice_size, dst_ptr, src_ptr, out_dim1, act_out_of_bounds,
                  act_loc, act_flat_idx, nb_continuous);
      act_loc += nb_continuous;
      act_flat_idx = next_flat_idx;
      act_out_of_bounds = next_out_of_bounds;
      nb_continuous = 1;
    }
    if (act_loc < batch_size) {
      MemCpyOrSet(d, slice_size, dst_ptr, src_ptr, out_dim1, act_out_of_bounds,
                  act_loc, act_flat_idx, nb_continuous);
    }
    return error_loc;
  }
};

#define REGISTER_GATHER_ND_SYCL(type) \
  REGISTER_GATHER_ND_ALL_INDICES(SYCL, type)

TF_CALL_INTEGRAL_TYPES(REGISTER_GATHER_ND_SYCL);
TF_CALL_SYCL_NUMBER_TYPES(REGISTER_GATHER_ND_SYCL);

#undef REGISTER_GATHER_ND_SYCL

#endif  // TENSORFLOW_USE_SYCL

#undef REGISTER_GATHER_ND_FULL
#undef REGISTER_GATHER_ND_ALL_INDICES

}  // namespace functor

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_GATHER_ND_OP_CPU_IMPL_H_
