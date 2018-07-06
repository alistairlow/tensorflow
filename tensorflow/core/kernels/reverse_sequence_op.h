/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_KERNELS_REVERSE_SEQUENCE_OP_H_
#define TENSORFLOW_KERNELS_REVERSE_SEQUENCE_OP_H_
// Generator definition for ReverseSequenceOp, must be compilable by nvcc.

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
#endif  // TENSORFLOW_USE_SYCL

namespace generator {

template <typename T, typename Tlen, size_t Dims>
class ReverseGenerator {
 public:
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  ReverseGenerator(typename TTypes<T, Dims>::ConstTensor input, int32 batch_dim,
                   int32 seq_dim, typename TTypes<Tlen>::ConstVec seq_lengths)
      : input_(input),
        batch_dim_(batch_dim),
        seq_dim_(seq_dim),
        seq_lengths_(seq_lengths) {}

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T
  operator()(const Eigen::array<Eigen::DenseIndex, Dims>& coords) const {
    Eigen::array<Eigen::DenseIndex, Dims> new_coords = coords;
    if (coords[seq_dim_] < seq_lengths_(coords[batch_dim_])) {
      new_coords[seq_dim_] =
          seq_lengths_(coords[batch_dim_]) - coords[seq_dim_] - 1;
    }

    return input_(new_coords);
  }

 private:
  typename TTypes<T, Dims>::ConstTensor input_;
  int32 batch_dim_;
  int32 seq_dim_;
  typename TTypes<Tlen>::ConstVec seq_lengths_;
};

}  // namespace generator

namespace functor {

template <typename Device, typename T, typename Tlen, size_t Dims>
struct ReverseSequence {
  EIGEN_ALWAYS_INLINE static void Compute(
      const Device& d, typename TTypes<T, Dims>::ConstTensor input,
      int32 batch_dim, int32 seq_dim,
      typename TTypes<Tlen>::ConstVec seq_lengths,
      typename TTypes<T, Dims>::Tensor output) {
    generator::ReverseGenerator<T, Tlen, Dims> generator(input, batch_dim,
                                                         seq_dim, seq_lengths);
    output.device(d) = input.generate(generator);
  }
};

#ifdef TENSORFLOW_USE_SYCL

template <typename T, typename Tlen, size_t Dims>
struct ReverseSequenceKernelSYCL {
  using r_acc =
    cl::sycl::accessor<Eigen::buffer_scalar_t, 1,
                       cl::sycl::access::mode::read,
                       cl::sycl::access::target::global_buffer>;
  using dw_acc =
    cl::sycl::accessor<Eigen::buffer_scalar_t, 1,
                       cl::sycl::access::mode::discard_write,
                       cl::sycl::access::target::global_buffer>;

  ReverseSequenceKernelSYCL(int32 batch_dim, int32 seq_dim,
      const Eigen::DSizes<Eigen::DenseIndex, Dims>& coord_dims,
      r_acc seq_lengths_acc, r_acc input_acc, dw_acc output_acc)
    : batch_dim_(batch_dim), seq_dim_(seq_dim),
      coord_dims_(coord_dims), seq_lengths_acc_(seq_lengths_acc),
      input_acc_(input_acc), output_acc_(output_acc) {}

  // Unflatten the indices and return the dimension dim
  inline int32 get_coord_dim(const int32 coord, const int32 dim) const {
    int32 mod = coord_dims_[Dims - 1];
    int32 div = 1;
    for (int32 i = dim; i < int32(Dims) - 1; ++i) {
      mod *= coord_dims_[i];
      div *= coord_dims_[i + 1];
    }
    return (coord % mod) / div;
  }

  inline void operator()(cl::sycl::item<1> item) {
    Tlen* seq_lengths = ConvertToActualTypeSycl(Tlen, seq_lengths_acc_);
    T* input = ConvertToActualTypeSycl(T, input_acc_);
    T* output = ConvertToActualTypeSycl(T, output_acc_);
    const auto coord = item.get_id(0);
    auto new_coord = coord;
    auto coord_seq_dim = get_coord_dim(coord, seq_dim_);
    auto coord_batch_dim = get_coord_dim(coord, batch_dim_);
    auto seq = seq_lengths[coord_batch_dim];
    if (coord_seq_dim < seq) {
      new_coord = 1;
      for (int32 i = seq_dim_ + 1; i < int32(Dims); ++i)
        new_coord *= coord_dims_[i];
      new_coord = coord + (seq - 2 * coord_seq_dim - 1) * new_coord;
    }
    output[coord] = input[new_coord];
  }

 private:
  int32 batch_dim_;
  int32 seq_dim_;
  Eigen::DSizes<Eigen::DenseIndex, Dims> coord_dims_;
  r_acc seq_lengths_acc_;
  r_acc input_acc_;
  dw_acc output_acc_;
};

template <typename T, typename Tlen, size_t Dims>
struct ReverseSequence<SYCLDevice, T, Tlen, Dims> {
  EIGEN_ALWAYS_INLINE static void Compute(
      const SYCLDevice& d, typename TTypes<T, Dims>::ConstTensor input,
      int32 batch_dim, int32 seq_dim,
      typename TTypes<Tlen>::ConstVec seq_lengths,
      typename TTypes<T, Dims>::Tensor output) {
    auto seq_lengths_buffer = d.get_sycl_buffer(seq_lengths.data());
    auto input_buffer = d.get_sycl_buffer(input.data());
    auto output_buffer = d.get_sycl_buffer(output.data());
    auto coord_dims = input.dimensions();

    auto sycl_queue = d.sycl_queue();
    using mode = typename cl::sycl::access::mode;
    sycl_queue.submit([&](cl::sycl::handler& cgh) {
      auto seq_lengths_acc =
        seq_lengths_buffer.template get_access<mode::read>(cgh);
      auto input_acc = input_buffer.template get_access<mode::read>(cgh);
      auto output_acc =
        output_buffer.template get_access<mode::discard_write>(cgh);
      ReverseSequenceKernelSYCL<T, Tlen, Dims> kernel(batch_dim, seq_dim,
          coord_dims, seq_lengths_acc, input_acc, output_acc);
      cgh.parallel_for(cl::sycl::range<1>(input.size()), kernel);
    });
  }
};

#endif  // TENSORFLOW_USE_SYCL

}  // namespace functor

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_REVERSE_SEQUENCE_OP_H_
