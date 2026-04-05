#ifndef OPS_H
#define OPS_H

#include "../core/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GradFn GradFn;

typedef void (*BackwardFunction)(GradFn* self, Tensor* grad_output);

struct GradFn {
    BackwardFunction backward;
    Tensor** inputs;
    int num_inputs;
    void* saved_data;
};

Tensor* op_add(Tensor* x1, Tensor* x2);
Tensor* op_mul(Tensor* x1, Tensor* x2);
Tensor* op_sub(Tensor* x1, Tensor* x2);
Tensor* op_div(Tensor* x1, Tensor* x2);
Tensor* op_power(Tensor* x1, Tensor* x2);
Tensor* op_logb(Tensor* x1, Tensor* x2);

Tensor* op_exp(Tensor* x);
Tensor* op_log(Tensor* x);
Tensor* op_sqrt(Tensor* x);
Tensor* op_tanh(Tensor* x);
Tensor* op_relu(Tensor* x);
Tensor* op_sigmoid(Tensor* x);
Tensor* op_abs(Tensor* x);
Tensor* op_square(Tensor* x);
Tensor* op_sin(Tensor* x);
Tensor* op_cos(Tensor* x);
Tensor* op_tan(Tensor* x);
Tensor* op_asin(Tensor* x);
Tensor* op_acos(Tensor* x);
Tensor* op_atan(Tensor* x);
Tensor* op_sinh(Tensor* x);
Tensor* op_cosh(Tensor* x);
Tensor* op_gelu(Tensor* x);
Tensor* op_silu(Tensor* x);
Tensor* op_leaky_relu(Tensor* x, float alpha);
Tensor* op_rsqrt(Tensor* x);

Tensor* op_cat(Tensor** tensors, int num_tensors, int dim);
Tensor* op_slice(Tensor* src, int* start, int* stop, int* step);
Tensor* op_view(Tensor* src, int ndim, int* new_shape);
Tensor* op_reshape(Tensor* src, int ndim, int* new_shape);
Tensor* op_transpose(Tensor* src, int dim0, int dim1);
Tensor* op_squeeze(Tensor* src, int dim);
Tensor* op_unsqueeze(Tensor* src, int dim);
Tensor* op_flatten(Tensor* src, int start_dim, int end_dim);
Tensor* op_permute(Tensor* src, int* dims);
Tensor** op_chunk(Tensor* src, int chunks, int dim);
Tensor** op_split(Tensor* src, int* sizes, int num_splits, int dim);
Tensor** op_split_equal(Tensor* src, int num_splits, int dim);
Tensor* op_expand(Tensor* src, int ndim, int* shape);
Tensor* op_repeat(Tensor* src, int* repeats);

Tensor* op_sum_all(Tensor* x);
Tensor* op_mean_all(Tensor* x);

Tensor* op_sum_dim(Tensor* x, int dim, bool keepdim);
Tensor* op_mean_dim(Tensor* x, int dim, bool keepdim);

Tensor* op_softmax(Tensor* x, int dim);
Tensor* op_log_softmax(Tensor* x, int dim);

Tensor* op_add_scalar(Tensor* x, const void* scalar);
Tensor* op_sub_scalar(Tensor* x, const void* scalar);
Tensor* op_mul_scalar(Tensor* x, const void* scalar);
Tensor* op_div_scalar(Tensor* x, const void* scalar);
Tensor* op_pow_scalar(Tensor* x, const void* scalar);
Tensor* op_rsub_scalar(Tensor* x, const void* scalar);
Tensor* op_rdiv_scalar(Tensor* x, const void* scalar);
Tensor* op_rpow_scalar(Tensor* x, const void* scalar);
Tensor* op_logb_scalar(Tensor* x, const void* scalar);
Tensor* op_rlogb_scalar(Tensor* x, const void* scalar);

Tensor* op_matmul(Tensor* a, Tensor* b);

Tensor* op_gather(Tensor* input, int dim, const int* indices, int index_ndim, const int* index_shape, size_t index_size);

Tensor* op_conv2d(Tensor* input, Tensor* weight, Tensor* bias,
                  int stride_h, int stride_w, int pad_h, int pad_w,
                  int dilation_h, int dilation_w);

#define REDUCTION_NONE 0
#define REDUCTION_MEAN 1
#define REDUCTION_SUM 2

Tensor* op_mse_loss(Tensor* pred, Tensor* target, int reduction);
Tensor* op_bce_loss(Tensor* pred, Tensor* target, int reduction, int from_logits);
Tensor* op_nll_loss(Tensor* input, const int* targets, int batch_size, int num_classes, int reduction);
Tensor* op_cross_entropy_loss(Tensor* input, const int* targets, int batch_size, int num_classes, int reduction, int from_logits);

int backwards_add_x1(const void* grad_c, void* grad_x1, size_t size, DType dtype, int device_id);
int backwards_add_x2(const void* grad_c, void* grad_x2, size_t size, DType dtype, int device_id);

int backwards_mul_x1(const void* grad_c, const void* x2, void* grad_x1, size_t size, DType dtype, int device_id);
int backwards_mul_x2(const void* grad_c, const void* x1, void* grad_x2, size_t size, DType dtype, int device_id);

int backwards_sub_x1(const void* grad_c, void* grad_x1, size_t size, DType dtype, int device_id);
int backwards_sub_x2(const void* grad_c, void* grad_x2, size_t size, DType dtype, int device_id);

int backwards_div_x1(const void* grad_c, const void* x2, void* grad_x1, size_t size, DType dtype, int device_id);
int backwards_div_x2(const void* grad_c, const void* x1, const void* x2, void* grad_x2, size_t size, DType dtype, int device_id);

int backwards_power_x1(const void* grad_c, const void* x1, const void* x2, const void* out, void* grad_x1, size_t size, DType dtype, int device_id);
int backwards_power_x2(const void* grad_c, const void* x1, const void* out, void* grad_x2, size_t size, DType dtype, int device_id);

int backwards_logb_x1(const void* grad_c, const void* x1, const void* x2, void* grad_x1, size_t size, DType dtype, int device_id);
int backwards_logb_x2(const void* grad_c, const void* x1, const void* x2, void* grad_x2, size_t size, DType dtype, int device_id);

int backwards_exp(const void* grad_c, const void* fn_output, void* grad_x, size_t size, DType dtype, int device_id);
int backwards_log(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype, int device_id);
int backwards_sqrt(const void* grad_c, const void* fn_output, void* grad_x, size_t size, DType dtype, int device_id);
int backwards_tanh(const void* grad_c, const void* fn_output, void* grad_x, size_t size, DType dtype, int device_id);
int backwards_relu(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype, int device_id);
int backwards_sigmoid(const void* grad_c, const void* fn_output, void* grad_x, size_t size, DType dtype, int device_id);
int backwards_abs(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype, int device_id);
int backwards_square(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype, int device_id);
int backwards_sin(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype, int device_id);
int backwards_cos(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype, int device_id);
int backwards_tan(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype, int device_id);
int backwards_asin(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype, int device_id);
int backwards_acos(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype, int device_id);
int backwards_atan(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype, int device_id);
int backwards_sinh(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype, int device_id);
int backwards_cosh(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype, int device_id);
int backwards_gelu(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype, int device_id);
int backwards_silu(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype, int device_id);
int backwards_leaky_relu(const void* grad_c, const void* x, const void* alpha, void* grad_x, size_t size, DType dtype, int device_id);
int backwards_rsqrt(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype, int device_id);

int backwards_add_x1_host(const void* grad_c, void* grad_x1, size_t size, DType dtype);
int backwards_add_x2_host(const void* grad_c, void* grad_x2, size_t size, DType dtype);

int backwards_mul_x1_host(const void* grad_c, const void* x2, void* grad_x1, size_t size, DType dtype);
int backwards_mul_x2_host(const void* grad_c, const void* x1, void* grad_x2, size_t size, DType dtype);

int backwards_sub_x1_host(const void* grad_c, void* grad_x1, size_t size, DType dtype);
int backwards_sub_x2_host(const void* grad_c, void* grad_x2, size_t size, DType dtype);

int backwards_div_x1_host(const void* grad_c, const void* x2, void* grad_x1, size_t size, DType dtype);
int backwards_div_x2_host(const void* grad_c, const void* x1, const void* x2, void* grad_x2, size_t size, DType dtype);

int backwards_power_x1_host(const void* grad_c, const void* x1, const void* x2, const void* out, void* grad_x1, size_t size, DType dtype);
int backwards_power_x2_host(const void* grad_c, const void* x1, const void* out, void* grad_x2, size_t size, DType dtype);

int backwards_logb_x1_host(const void* grad_c, const void* x1, const void* x2, void* grad_x1, size_t size, DType dtype);
int backwards_logb_x2_host(const void* grad_c, const void* x1, const void* x2, void* grad_x2, size_t size, DType dtype);

int backwards_exp_host(const void* grad_c, const void* fn_output, void* grad_x, size_t size, DType dtype);
int backwards_log_host(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype);
int backwards_sqrt_host(const void* grad_c, const void* fn_output, void* grad_x, size_t size, DType dtype);
int backwards_tanh_host(const void* grad_c, const void* fn_output, void* grad_x, size_t size, DType dtype);
int backwards_relu_host(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype);
int backwards_sigmoid_host(const void* grad_c, const void* fn_output, void* grad_x, size_t size, DType dtype);
int backwards_abs_host(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype);
int backwards_square_host(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype);
int backwards_sin_host(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype);
int backwards_cos_host(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype);
int backwards_tan_host(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype);
int backwards_asin_host(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype);
int backwards_acos_host(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype);
int backwards_atan_host(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype);
int backwards_sinh_host(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype);
int backwards_cosh_host(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype);
int backwards_gelu_host(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype);
int backwards_silu_host(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype);
int backwards_leaky_relu_host(const void* grad_c, const void* x, const void* alpha, void* grad_x, size_t size, DType dtype);
int backwards_rsqrt_host(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype);

int backwards_sum_all(const void* grad_c, void* grad_x, size_t size, DType dtype, int device_id);
int backwards_mean_all(const void* grad_c, void* grad_x, size_t size, DType dtype, int device_id);

int backwards_sum_dim(const void* grad_c, void* grad_x, size_t outer_size, size_t dim_size, size_t inner_size, DType dtype, int device_id);
int backwards_mean_dim(const void* grad_c, void* grad_x, size_t outer_size, size_t dim_size, size_t inner_size, DType dtype, int device_id);

int backwards_sum_all_host(const void* grad_c, void* grad_x, size_t size, DType dtype);
int backwards_mean_all_host(const void* grad_c, void* grad_x, size_t size, DType dtype);

int backwards_sum_dim_host(const void* grad_c, void* grad_x, size_t outer_size, size_t dim_size, size_t inner_size, DType dtype);
int backwards_mean_dim_host(const void* grad_c, void* grad_x, size_t outer_size, size_t dim_size, size_t inner_size, DType dtype);

int backwards_add_x1_device(const void* grad_c, void* grad_x1, size_t size, DType dtype);
int backwards_add_x2_device(const void* grad_c, void* grad_x2, size_t size, DType dtype);

int backwards_mul_x1_device(const void* grad_c, const void* x2, void* grad_x1, size_t size, DType dtype);
int backwards_mul_x2_device(const void* grad_c, const void* x1, void* grad_x2, size_t size, DType dtype);

int backwards_sub_x1_device(const void* grad_c, void* grad_x1, size_t size, DType dtype);
int backwards_sub_x2_device(const void* grad_c, void* grad_x2, size_t size, DType dtype);

int backwards_div_x1_device(const void* grad_c, const void* x2, void* grad_x1, size_t size, DType dtype);
int backwards_div_x2_device(const void* grad_c, const void* x1, const void* x2, void* grad_x2, size_t size, DType dtype);

int backwards_power_x1_device(const void* grad_c, const void* x1, const void* x2, const void* out, void* grad_x1, size_t size, DType dtype);
int backwards_power_x2_device(const void* grad_c, const void* x1, const void* out, void* grad_x2, size_t size, DType dtype);

int backwards_logb_x1_device(const void* grad_c, const void* x1, const void* x2, void* grad_x1, size_t size, DType dtype);
int backwards_logb_x2_device(const void* grad_c, const void* x1, const void* x2, void* grad_x2, size_t size, DType dtype);

int backwards_exp_device(const void* grad_c, const void* fn_output, void* grad_x, size_t size, DType dtype);
int backwards_log_device(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype);
int backwards_sqrt_device(const void* grad_c, const void* fn_output, void* grad_x, size_t size, DType dtype);
int backwards_tanh_device(const void* grad_c, const void* fn_output, void* grad_x, size_t size, DType dtype);
int backwards_relu_device(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype);
int backwards_sigmoid_device(const void* grad_c, const void* fn_output, void* grad_x, size_t size, DType dtype);
int backwards_abs_device(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype);
int backwards_square_device(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype);
int backwards_sin_device(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype);
int backwards_cos_device(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype);
int backwards_tan_device(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype);
int backwards_asin_device(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype);
int backwards_acos_device(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype);
int backwards_atan_device(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype);
int backwards_sinh_device(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype);
int backwards_cosh_device(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype);
int backwards_gelu_device(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype);
int backwards_silu_device(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype);
int backwards_leaky_relu_device(const void* grad_c, const void* x, const void* alpha, void* grad_x, size_t size, DType dtype);
int backwards_rsqrt_device(const void* grad_c, const void* x, void* grad_x, size_t size, DType dtype);

int backwards_sum_all_device(const void* grad_c, void* grad_x, size_t size, DType dtype);
int backwards_mean_all_device(const void* grad_c, void* grad_x, size_t size, DType dtype);

int backwards_sum_dim_device(const void* grad_c, void* grad_x, size_t outer_size, size_t dim_size, size_t inner_size, DType dtype);
int backwards_mean_dim_device(const void* grad_c, void* grad_x, size_t outer_size, size_t dim_size, size_t inner_size, DType dtype);

int backwards_softmax(const void* grad_c, const void* softmax_out, void* grad_x, size_t outer_size, size_t dim_size, size_t inner_size, DType dtype, int device_id);
int backwards_log_softmax(const void* grad_c, const void* log_softmax_out, void* grad_x, size_t outer_size, size_t dim_size, size_t inner_size, DType dtype, int device_id);

int backwards_softmax_host(const void* grad_c, const void* softmax_out, void* grad_x, size_t outer_size, size_t dim_size, size_t inner_size, DType dtype);
int backwards_log_softmax_host(const void* grad_c, const void* log_softmax_out, void* grad_x, size_t outer_size, size_t dim_size, size_t inner_size, DType dtype);

int backwards_softmax_device(const void* grad_c, const void* softmax_out, void* grad_x, size_t outer_size, size_t dim_size, size_t inner_size, DType dtype);
int backwards_log_softmax_device(const void* grad_c, const void* log_softmax_out, void* grad_x, size_t outer_size, size_t dim_size, size_t inner_size, DType dtype);

void free_grad_fn(GradFn* grad_fn);

typedef struct {
    int x1_ndim;
    int* x1_shape;
    int x2_ndim;
    int* x2_shape;
} BroadcastSavedData;

int broadcast_shapes(int ndim1, const int* shape1, int ndim2, const int* shape2,
                     int* out_ndim, int** out_shape);
Tensor* reduce_grad_to_shape(Tensor* grad, int target_ndim, const int* target_shape);

#ifdef __cplusplus
}
#endif

#endif
