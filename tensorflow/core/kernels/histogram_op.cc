/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/math_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/histogram_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
#endif  // TENSORFLOW_USE_SYCL

namespace functor {

template <typename T, typename Tout>
struct HistogramFixedWidthFunctor<CPUDevice, T, Tout> {
  static Status Compute(OpKernelContext* context,
                        const typename TTypes<T, 1>::ConstTensor& values,
                        const typename TTypes<T, 1>::ConstTensor& value_range,
                        int32 nbins, typename TTypes<Tout, 1>::Tensor& out) {
    const CPUDevice& d = context->eigen_device<CPUDevice>();

    Tensor index_to_bin_tensor;

    TF_RETURN_IF_ERROR(context->forward_input_or_allocate_temp(
        {0}, DataTypeToEnum<int32>::value, TensorShape({values.size()}),
        &index_to_bin_tensor));
    auto index_to_bin = index_to_bin_tensor.flat<int32>();

    const double step = static_cast<double>(value_range(1) - value_range(0)) /
                        static_cast<double>(nbins);

    // The calculation is done by finding the slot of each value in `values`.
    // With [a, b]:
    //   step = (b - a) / nbins
    //   (x - a) / step
    // , then the entries are mapped to output.
    index_to_bin.device(d) =
        ((values.cwiseMax(value_range(0)) - values.constant(value_range(0)))
             .template cast<double>() /
         step)
            .template cast<int32>()
            .cwiseMin(nbins - 1);

    out.setZero();
    for (int32 i = 0; i < index_to_bin.size(); i++) {
      out(index_to_bin(i)) += Tout(1);
    }
    return Status::OK();
  }
};

#ifdef TENSORFLOW_USE_SYCL

namespace {

// Generate a matrix wich values are the index of the column.
template <typename T>
struct ColIndicesGenerator {
  inline T operator()(const Eigen::array<Eigen::DenseIndex, 2>& idx) const {
    return idx[1];
  }
};

} //namespace

template <typename T, typename Tout>
struct HistogramFixedWidthFunctor<SYCLDevice, T, Tout> {
  static Status Compute(OpKernelContext* context,
                        const typename TTypes<T, 1>::ConstTensor& values,
                        const typename TTypes<T, 1>::ConstTensor& value_range,
                        int32 nbins, typename TTypes<Tout, 1>::Tensor& out) {
    const SYCLDevice& d = context->eigen_device<SYCLDevice>();

#ifdef TENSORFLOW_SYCL_NO_DOUBLE
    using InternalT = T;
#else
    using InternalT = double;
#endif  // TENSORFLOW_SYCL_NO_DOUBLE

    auto values_size = values.size();
    if (values_size == 0) {
      out.device(d) = out.constant(0);
      return Status::OK();
    }

#if !defined(EIGEN_HAS_INDEX_LIST)
    Eigen::DSizes<Eigen::Index, 1> sum_dim(0);
    Eigen::DSizes<Eigen::Index, 2> values_size_by_one({values_size, 1});
    Eigen::DSizes<Eigen::Index, 2> one_by_nbins({1, nbins});
#else
    Eigen::IndexList<Eigen::type2index<0>> sum_dim;
    Eigen::IndexList<Eigen::Index, Eigen::type2index<1>> values_size_by_one;
    values_size_by_one.set(0, values_size);
    Eigen::IndexList<Eigen::type2index<1>, Eigen::Index> one_by_nbins;
    one_by_nbins.set(1, nbins);
#endif

    const InternalT step =
      static_cast<InternalT>(value_range(1) - value_range(0)) /
      static_cast<InternalT>(nbins);

    // The calculation is done by finding the slot of each value in `values`.
    // With [a, b]:
    //   step = (b - a) / nbins
    //   (x - a) / step
    // , then the entries are mapped to output.
    auto index_to_bin =
        ((values.cwiseMax(value_range(0)) - values.constant(value_range(0)))
            .template cast<InternalT>() / step)
          .template cast<int32>()
          .cwiseMin(nbins - 1);
    auto index_to_bin_2d = index_to_bin.reshape(values_size_by_one)
                                       .broadcast(one_by_nbins);
    auto col_indices = TTypes<int32, 2>::Tensor(nullptr, values_size, nbins)
                         .generate(ColIndicesGenerator<int32>());

    out.device(d) = (index_to_bin_2d == col_indices)
                      .template cast<Tout>().sum(sum_dim);
    return Status::OK();
  }
};

#endif  // TENSORFLOW_USE_SYCL

}  // namespace functor

template <typename Device, typename T, typename Tout>
class HistogramFixedWidthOp : public OpKernel {
 public:
  explicit HistogramFixedWidthOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& values_tensor = ctx->input(0);
    const Tensor& value_range_tensor = ctx->input(1);
    const Tensor& nbins_tensor = ctx->input(2);

    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(value_range_tensor.shape()),
                errors::InvalidArgument("value_range should be a vector."));
    OP_REQUIRES(ctx, (value_range_tensor.shape().num_elements() == 2),
                errors::InvalidArgument(
                    "value_range should be a vector of 2 elements."));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(nbins_tensor.shape()),
                errors::InvalidArgument("nbins should be a scalar."));

    const auto values = values_tensor.flat<T>();
    const auto value_range = value_range_tensor.flat<T>();
    const auto nbins = nbins_tensor.scalar<int32>()();

    OP_REQUIRES(
        ctx, (value_range(0) < value_range(1)),
        errors::InvalidArgument("value_range should satisfy value_range[0] < "
                                "value_range[1], but got '[",
                                value_range(0), ", ", value_range(1), "]'"));
    OP_REQUIRES(
        ctx, (nbins > 0),
        errors::InvalidArgument("nbins should be a positive number, but got '",
                                nbins, "'"));

    Tensor* out_tensor;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, TensorShape({nbins}), &out_tensor));
    auto out = out_tensor->flat<Tout>();

    OP_REQUIRES_OK(
        ctx, functor::HistogramFixedWidthFunctor<Device, T, Tout>::Compute(
                 ctx, values, value_range, nbins, out));
  }
};

#define REGISTER_KERNELS(type)                                           \
  REGISTER_KERNEL_BUILDER(Name("HistogramFixedWidth")                    \
                              .Device(DEVICE_CPU)                        \
                              .TypeConstraint<type>("T")                 \
                              .TypeConstraint<int32>("dtype"),           \
                          HistogramFixedWidthOp<CPUDevice, type, int32>) \
  REGISTER_KERNEL_BUILDER(Name("HistogramFixedWidth")                    \
                              .Device(DEVICE_CPU)                        \
                              .TypeConstraint<type>("T")                 \
                              .TypeConstraint<int64>("dtype"),           \
                          HistogramFixedWidthOp<CPUDevice, type, int64>)

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#if GOOGLE_CUDA
#define REGISTER_KERNELS(type)                                 \
  REGISTER_KERNEL_BUILDER(Name("HistogramFixedWidth")          \
                              .Device(DEVICE_GPU)              \
                              .HostMemory("value_range")       \
                              .HostMemory("nbins")             \
                              .TypeConstraint<type>("T")       \
                              .TypeConstraint<int32>("dtype"), \
                          HistogramFixedWidthOp<GPUDevice, type, int32>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#endif  // GOOGLE_CUDA

#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_KERNELS(type)                                 \
  REGISTER_KERNEL_BUILDER(Name("HistogramFixedWidth")          \
                              .Device(DEVICE_SYCL)             \
                              .HostMemory("value_range")       \
                              .HostMemory("nbins")             \
                              .TypeConstraint<type>("T")       \
                              .TypeConstraint<int32>("dtype"), \
                          HistogramFixedWidthOp<SYCLDevice, type, int32>)

TF_CALL_SYCL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS
#endif  // TENSORFLOW_USE_SYCL

}  // end namespace tensorflow
