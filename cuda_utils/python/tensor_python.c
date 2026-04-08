#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "../core/tensor.h"
#include "../functional/functional.h"
#include "../ops/ops.h"
#include "../optim/optim.h"
#include "../rand/rand.h"

typedef struct {
    PyObject_HEAD
    Tensor* tensor;
} PyTensorObject;

static PyTypeObject PyTensorType;
static PyObject* wrap_tensor_result(Tensor* result);

static DType parse_dtype(const char* dtype_str) {
    if (dtype_str == NULL) return DTYPE_PRESERVE;
    if (strcmp(dtype_str, "float32") == 0) return DTYPE_FLOAT32;
    if (strcmp(dtype_str, "float64") == 0) return DTYPE_FLOAT64;
    if (strcmp(dtype_str, "float16") == 0) return DTYPE_FLOAT16;
    if (strcmp(dtype_str, "bfloat16") == 0) return DTYPE_BFLOAT16;
    return DTYPE_PRESERVE;
}

static const char* dtype_to_string(DType dtype) {
    switch (dtype) {
        case DTYPE_FLOAT32: return "float32";
        case DTYPE_FLOAT64: return "float64";
        case DTYPE_FLOAT16: return "float16";
        case DTYPE_BFLOAT16: return "bfloat16";
        default: return "unknown";
    }
}

static int parse_device(const char* device_str, int* out_device_id) {
    if (device_str == NULL) return 0;

    if (strcmp(device_str, "cpu") == 0 || strcmp(device_str, "host") == 0) {
        *out_device_id = -1;
        return 1;
    }

    if (strcmp(device_str, "cuda") == 0 || strcmp(device_str, "device") == 0) {
        *out_device_id = 0;
        return 1;
    }

    if (strncmp(device_str, "cuda:", 5) == 0) {
        *out_device_id = atoi(device_str + 5);
        return 1;
    }

    PyErr_Format(PyExc_ValueError, "Invalid device string: '%s'. Use 'cpu', 'cuda', or 'cuda:N'", device_str);
    return 0;
}

static PyObject* device_to_pystring(int device_id) {
    if (device_id == -1) {
        return PyUnicode_FromString("cpu");
    } else {
        char device_str[32];
        snprintf(device_str, sizeof(device_str), "cuda:%d", device_id);
        return PyUnicode_FromString(device_str);
    }
}

static void PyTensor_dealloc(PyTensorObject* self) {
    if (self->tensor != NULL) {
        free_tensor(self->tensor);
        self->tensor = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* PyTensor_get_shape(PyTensorObject* self, void* closure) {
    (void)closure;

    if (self->tensor == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized");
        return NULL;
    }

    PyObject* shape_tuple = PyTuple_New(self->tensor->ndim);
    if (shape_tuple == NULL) return NULL;

    for (int i = 0; i < self->tensor->ndim; i++) {
        PyObject* dim = PyLong_FromLong(self->tensor->shape[i]);
        if (dim == NULL) {
            Py_DECREF(shape_tuple);
            return NULL;
        }
        PyTuple_SET_ITEM(shape_tuple, i, dim);
    }

    return shape_tuple;
}

static PyObject* PyTensor_get_dtype(PyTensorObject* self, void* closure) {
    (void)closure;

    if (self->tensor == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized");
        return NULL;
    }
    return PyUnicode_FromString(dtype_to_string(self->tensor->dtype));
}

static PyObject* PyTensor_get_device(PyTensorObject* self, void* closure) {
    (void)closure;

    if (self->tensor == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized");
        return NULL;
    }
    return device_to_pystring(self->tensor->device_id);
}

static PyObject* PyTensor_get_device_id(PyTensorObject* self, void* closure) {
    (void)closure;

    if (self->tensor == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized");
        return NULL;
    }
    return PyLong_FromLong(self->tensor->device_id);
}

static PyObject* PyTensor_get_size(PyTensorObject* self, void* closure) {
    (void)closure;

    if (self->tensor == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized");
        return NULL;
    }
    return PyLong_FromSize_t(self->tensor->size);
}

static PyObject* PyTensor_get_ndim(PyTensorObject* self, void* closure) {
    (void)closure;

    if (self->tensor == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized");
        return NULL;
    }
    return PyLong_FromLong(self->tensor->ndim);
}

static PyObject* PyTensor_get_grad(PyTensorObject* self, void* closure) {
    (void)closure;

    if (self->tensor == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized");
        return NULL;
    }

    if (!self->tensor->metadata || !self->tensor->metadata->grad) {
        Py_RETURN_NONE;
    }

    Tensor* grad_copy = tensor_copy(self->tensor->metadata->grad);
    if (!grad_copy) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to copy gradient tensor");
        return NULL;
    }

    return wrap_tensor_result(grad_copy);
}

static PyObject* PyTensor_get_has_grad(PyTensorObject* self, void* closure) {
    (void)closure;
    if (self->tensor == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized");
        return NULL;
    }
    if (self->tensor->metadata && self->tensor->metadata->grad) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

static PyObject* PyTensor_get_requires_grad(PyTensorObject* self, void* closure) {
    (void)closure;

    if (self->tensor == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized");
        return NULL;
    }

    if (!self->tensor->metadata) {
        Py_RETURN_FALSE;
    }

    return PyBool_FromLong(self->tensor->metadata->requires_grad);
}

static PyObject* PyTensor_to(PyTensorObject* self, PyObject* args, PyObject* kwargs) {
    const char* device_str = NULL;
    const char* dtype_str = NULL;
    int inplace = 0;

    static char* kwlist[] = {"device", "dtype", "inplace", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|ssp", kwlist,
                                     &device_str, &dtype_str, &inplace)) {
        return NULL;
    }

    if (self->tensor == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized");
        return NULL;
    }

    int target_device_id = self->tensor->device_id;
    DType target_dtype = self->tensor->dtype;

    if (device_str != NULL) {
        if (!parse_device(device_str, &target_device_id)) {
            return NULL;
        }
    }

    if (dtype_str != NULL) {
        target_dtype = parse_dtype(dtype_str);
        if (target_dtype == DTYPE_PRESERVE) {
            PyErr_SetString(PyExc_ValueError, "Invalid dtype. Use 'float32', 'float64', 'float16', or 'bfloat16'");
            return NULL;
        }
    }

    Tensor* result = tensor_to(self->tensor, target_device_id,
                               target_dtype, (bool)inplace);

    if (result == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "tensor_to failed");
        return NULL;
    }

    if (inplace) {
        self->tensor = result;
        Py_INCREF(self);
        return (PyObject*)self;
    }

    PyTensorObject* py_result = (PyTensorObject*)PyTensorType.tp_alloc(&PyTensorType, 0);
    if (py_result == NULL) {
        free_tensor(result);
        return NULL;
    }

    py_result->tensor = result;
    return (PyObject*)py_result;
}

static PyObject* PyTensor_copy(PyTensorObject* self, PyObject* Py_UNUSED(ignored)) {
    if (self->tensor == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized");
        return NULL;
    }

    Tensor* result = tensor_copy(self->tensor);
    if (result == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "tensor_copy failed");
        return NULL;
    }

    PyTensorObject* py_result = (PyTensorObject*)PyTensorType.tp_alloc(&PyTensorType, 0);
    if (py_result == NULL) {
        free_tensor(result);
        return NULL;
    }

    py_result->tensor = result;
    return (PyObject*)py_result;
}

static PyObject* PyTensor_data(PyTensorObject* self, PyObject* Py_UNUSED(ignored)) {
    if (self->tensor == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized");
        return NULL;
    }

    Tensor* t = self->tensor;

    Tensor* cpu_tensor = t;
    bool allocated_cpu = false;
    if (t->device_id != -1) {
        cpu_tensor = tensor_to(t, -1, t->dtype, false);
        if (cpu_tensor == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy tensor to CPU");
            return NULL;
        }
        allocated_cpu = true;
    }

    PyObject* list = PyList_New(cpu_tensor->size);
    if (list == NULL) {
        if (allocated_cpu) free_tensor(cpu_tensor);
        return NULL;
    }

    if (cpu_tensor->dtype == DTYPE_FLOAT32) {
        float* data = (float*)cpu_tensor->data;
        for (size_t i = 0; i < cpu_tensor->size; i++) {
            PyObject* val = PyFloat_FromDouble((double)data[i]);
            if (val == NULL) {
                Py_DECREF(list);
                if (allocated_cpu) free_tensor(cpu_tensor);
                return NULL;
            }
            PyList_SET_ITEM(list, i, val);
        }
    } else if (cpu_tensor->dtype == DTYPE_FLOAT64) {
        double* data = (double*)cpu_tensor->data;
        for (size_t i = 0; i < cpu_tensor->size; i++) {
            PyObject* val = PyFloat_FromDouble(data[i]);
            if (val == NULL) {
                Py_DECREF(list);
                if (allocated_cpu) free_tensor(cpu_tensor);
                return NULL;
            }
            PyList_SET_ITEM(list, i, val);
        }
    }

    if (allocated_cpu) free_tensor(cpu_tensor);
    return list;
}

static PyObject* wrap_tensor_result(Tensor* result) {
    if (result == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Operation failed");
        return NULL;
    }

    PyTensorObject* py_result = (PyTensorObject*)PyTensorType.tp_alloc(&PyTensorType, 0);
    if (py_result == NULL) {
        free_tensor(result);
        return NULL;
    }

    py_result->tensor = result;
    return (PyObject*)py_result;
}

#define BINARY_OP_METHOD(name, func) \
static PyObject* PyTensor_##name(PyTensorObject* self, PyObject* args) { \
    PyTensorObject* other; \
    if (!PyArg_ParseTuple(args, "O!", &PyTensorType, &other)) { \
        return NULL; \
    } \
    if (self->tensor == NULL || other->tensor == NULL) { \
        PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized"); \
        return NULL; \
    } \
    Tensor* a = self->tensor; \
    Tensor* b = other->tensor; \
    if (a->ndim != b->ndim || a->size != b->size || a->device_id != b->device_id || a->dtype != b->dtype) { \
        PyErr_SetString(PyExc_ValueError, "Tensors must have same shape, device, and dtype"); \
        return NULL; \
    } \
    Tensor* result = zeros_tensor(a->dtype, a->device_id, a->ndim, a->shape, NULL); \
    if (result == NULL) { \
        PyErr_SetString(PyExc_RuntimeError, "Failed to allocate result tensor"); \
        return NULL; \
    } \
    int status = func(result->data, a->data, b->data, a->size, a->dtype, a->device_id); \
    if (status != 0) { \
        free_tensor(result); \
        PyErr_SetString(PyExc_RuntimeError, #func " operation failed"); \
        return NULL; \
    } \
    return wrap_tensor_result(result); \
}

BINARY_OP_METHOD(add, rp_add)
BINARY_OP_METHOD(sub, rp_sub)
BINARY_OP_METHOD(mul, rp_mul)
BINARY_OP_METHOD(divide, rp_divide)
BINARY_OP_METHOD(power, rp_power)
BINARY_OP_METHOD(logb, rp_logb)

#define SCALAR_OP_METHOD(name, func) \
static PyObject* PyTensor_##name(PyTensorObject* self, PyObject* args) { \
    double scalar_val; \
    if (!PyArg_ParseTuple(args, "d", &scalar_val)) { \
        return NULL; \
    } \
    if (self->tensor == NULL) { \
        PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized"); \
        return NULL; \
    } \
    Tensor* a = self->tensor; \
    Tensor* result = zeros_tensor(a->dtype, a->device_id, a->ndim, a->shape, NULL); \
    if (result == NULL) { \
        PyErr_SetString(PyExc_RuntimeError, "Failed to allocate result tensor"); \
        return NULL; \
    } \
    void* scalar_ptr; \
    float scalar_f32; \
    double scalar_f64; \
    if (a->dtype == DTYPE_FLOAT64) { \
        scalar_f64 = scalar_val; \
        scalar_ptr = &scalar_f64; \
    } else { \
        scalar_f32 = (float)scalar_val; \
        scalar_ptr = &scalar_f32; \
    } \
    int status = func(result->data, a->data, scalar_ptr, a->size, a->dtype, a->device_id); \
    if (status != 0) { \
        free_tensor(result); \
        PyErr_SetString(PyExc_RuntimeError, #func " operation failed"); \
        return NULL; \
    } \
    return wrap_tensor_result(result); \
}

SCALAR_OP_METHOD(add_scalar, rp_add_scalar)
SCALAR_OP_METHOD(sub_scalar, rp_sub_scalar)
SCALAR_OP_METHOD(mul_scalar, rp_mul_scalar)
SCALAR_OP_METHOD(div_scalar, rp_div_scalar)
SCALAR_OP_METHOD(pow_scalar, rp_pow_scalar)
SCALAR_OP_METHOD(logb_scalar, rp_logb_scalar)

#define RSCALAR_OP_METHOD(name, func) \
static PyObject* PyTensor_##name(PyTensorObject* self, PyObject* args) { \
    double scalar_val; \
    if (!PyArg_ParseTuple(args, "d", &scalar_val)) { \
        return NULL; \
    } \
    if (self->tensor == NULL) { \
        PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized"); \
        return NULL; \
    } \
    Tensor* a = self->tensor; \
    Tensor* result = zeros_tensor(a->dtype, a->device_id, a->ndim, a->shape, NULL); \
    if (result == NULL) { \
        PyErr_SetString(PyExc_RuntimeError, "Failed to allocate result tensor"); \
        return NULL; \
    } \
    void* scalar_ptr; \
    float scalar_f32; \
    double scalar_f64; \
    if (a->dtype == DTYPE_FLOAT64) { \
        scalar_f64 = scalar_val; \
        scalar_ptr = &scalar_f64; \
    } else { \
        scalar_f32 = (float)scalar_val; \
        scalar_ptr = &scalar_f32; \
    } \
    int status = func(result->data, scalar_ptr, a->data, a->size, a->dtype, a->device_id); \
    if (status != 0) { \
        free_tensor(result); \
        PyErr_SetString(PyExc_RuntimeError, #func " operation failed"); \
        return NULL; \
    } \
    return wrap_tensor_result(result); \
}

RSCALAR_OP_METHOD(rsub_scalar, rp_rsub_scalar)
RSCALAR_OP_METHOD(rdiv_scalar, rp_rdiv_scalar)
RSCALAR_OP_METHOD(rpow_scalar, rp_rpow_scalar)
RSCALAR_OP_METHOD(rlogb_scalar, rp_rlogb_scalar)

#define UNARY_OP_METHOD(name, func) \
static PyObject* PyTensor_##name(PyTensorObject* self, PyObject* Py_UNUSED(ignored)) { \
    if (self->tensor == NULL) { \
        PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized"); \
        return NULL; \
    } \
    Tensor* a = self->tensor; \
    Tensor* result = zeros_tensor(a->dtype, a->device_id, a->ndim, a->shape, NULL); \
    if (result == NULL) { \
        PyErr_SetString(PyExc_RuntimeError, "Failed to allocate result tensor"); \
        return NULL; \
    } \
    int status = func(result->data, a->data, a->size, a->dtype, a->device_id); \
    if (status != 0) { \
        free_tensor(result); \
        PyErr_SetString(PyExc_RuntimeError, #func " operation failed"); \
        return NULL; \
    } \
    return wrap_tensor_result(result); \
}

UNARY_OP_METHOD(exp, rp_exp)
UNARY_OP_METHOD(log, rp_log)
UNARY_OP_METHOD(sqrt, rp_sqrt)
UNARY_OP_METHOD(abs, rp_abs)
UNARY_OP_METHOD(sin, rp_sin)
UNARY_OP_METHOD(cos, rp_cos)
UNARY_OP_METHOD(tan, rp_tan)
UNARY_OP_METHOD(asin, rp_asin)
UNARY_OP_METHOD(acos, rp_acos)
UNARY_OP_METHOD(atan, rp_atan)
UNARY_OP_METHOD(sinh, rp_sinh)
UNARY_OP_METHOD(cosh, rp_cosh)
UNARY_OP_METHOD(tanh, rp_tanh)
UNARY_OP_METHOD(relu, rp_relu)
UNARY_OP_METHOD(sigmoid, rp_sigmoid)
UNARY_OP_METHOD(gelu, rp_gelu)
UNARY_OP_METHOD(silu, rp_silu)
UNARY_OP_METHOD(square, rp_square)
UNARY_OP_METHOD(ceil, rp_ceil)
UNARY_OP_METHOD(floor, rp_floor)
UNARY_OP_METHOD(round, rp_round)
UNARY_OP_METHOD(trunc, rp_trunc)
UNARY_OP_METHOD(rsqrt, rp_rsqrt)

static PyObject* PyTensor_leaky_relu(PyTensorObject* self, PyObject* args) {
    if (self->tensor == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized");
        return NULL;
    }

    double alpha_val = 0.01;
    if (!PyArg_ParseTuple(args, "d", &alpha_val)) {
        return NULL;
    }

    Tensor* a = self->tensor;
    Tensor* result = zeros_tensor(a->dtype, a->device_id, a->ndim, a->shape, NULL);
    if (result == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to allocate result tensor");
        return NULL;
    }

    float alpha_f32 = (float)alpha_val;
    double alpha_f64 = alpha_val;
    void* alpha_ptr = (a->dtype == DTYPE_FLOAT32) ? (void*)&alpha_f32 : (void*)&alpha_f64;

    int status = rp_leaky_relu(result->data, a->data, alpha_ptr, a->size, a->dtype, a->device_id);
    if (status != 0) {
        free_tensor(result);
        PyErr_SetString(PyExc_RuntimeError, "leaky_relu operation failed");
        return NULL;
    }

    return wrap_tensor_result(result);
}

static PyObject* PyTensor_sum_all(PyTensorObject* self, PyObject* Py_UNUSED(ignored)) {
    if (self->tensor == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized");
        return NULL;
    }

    Tensor* a = self->tensor;

    if (a->dtype == DTYPE_FLOAT32) {
        float result;
        int status = rp_sum_all(&result, a->data, a->size, a->dtype, a->device_id);
        if (status != 0) {
            PyErr_SetString(PyExc_RuntimeError, "sum_all operation failed");
            return NULL;
        }
        return PyFloat_FromDouble((double)result);
    } else if (a->dtype == DTYPE_FLOAT64) {
        double result;
        int status = rp_sum_all(&result, a->data, a->size, a->dtype, a->device_id);
        if (status != 0) {
            PyErr_SetString(PyExc_RuntimeError, "sum_all operation failed");
            return NULL;
        }
        return PyFloat_FromDouble(result);
    }

    PyErr_SetString(PyExc_RuntimeError, "Unsupported dtype");
    return NULL;
}

static PyObject* PyTensor_mean_all(PyTensorObject* self, PyObject* Py_UNUSED(ignored)) {
    if (self->tensor == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized");
        return NULL;
    }

    Tensor* a = self->tensor;

    if (a->dtype == DTYPE_FLOAT32) {
        float result;
        int status = rp_mean_all(&result, a->data, a->size, a->dtype, a->device_id);
        if (status != 0) {
            PyErr_SetString(PyExc_RuntimeError, "mean_all operation failed");
            return NULL;
        }
        return PyFloat_FromDouble((double)result);
    } else if (a->dtype == DTYPE_FLOAT64) {
        double result;
        int status = rp_mean_all(&result, a->data, a->size, a->dtype, a->device_id);
        if (status != 0) {
            PyErr_SetString(PyExc_RuntimeError, "mean_all operation failed");
            return NULL;
        }
        return PyFloat_FromDouble(result);
    }

    PyErr_SetString(PyExc_RuntimeError, "Unsupported dtype");
    return NULL;
}

static PyObject* PyTensor_sum_dim(PyTensorObject* self, PyObject* args, PyObject* kwargs) {
    if (self->tensor == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized");
        return NULL;
    }

    int dim;
    int keepdim = 0;
    static char* kwlist[] = {"dim", "keepdim", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i|p", kwlist, &dim, &keepdim))
        return NULL;

    Tensor* a = self->tensor;

    int normalized_dim = dim;
    if (normalized_dim < 0) normalized_dim += a->ndim;
    if (normalized_dim < 0 || normalized_dim >= a->ndim) {
        PyErr_SetString(PyExc_ValueError, "dim out of range");
        return NULL;
    }

    Tensor* input = a;
    bool made_contiguous = false;
    if (!rp_is_contiguous(a)) {
        input = rp_contiguous(a);
        if (!input) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to make tensor contiguous");
            return NULL;
        }
        made_contiguous = true;
    }

    size_t outer_size = 1;
    for (int i = 0; i < normalized_dim; i++) outer_size *= input->shape[i];
    size_t dim_size = input->shape[normalized_dim];
    size_t inner_size = 1;
    for (int i = normalized_dim + 1; i < input->ndim; i++) inner_size *= input->shape[i];

    int out_ndim;
    int* out_shape;
    if (keepdim) {
        out_ndim = input->ndim;
        out_shape = (int*)malloc(out_ndim * sizeof(int));
        if (!out_shape) { if (made_contiguous) free_tensor(input); return PyErr_NoMemory(); }
        memcpy(out_shape, input->shape, out_ndim * sizeof(int));
        out_shape[normalized_dim] = 1;
    } else {
        out_ndim = input->ndim - 1;
        if (out_ndim == 0) out_ndim = 1;
        out_shape = (int*)malloc(out_ndim * sizeof(int));
        if (!out_shape) { if (made_contiguous) free_tensor(input); return PyErr_NoMemory(); }
        if (input->ndim == 1) {
            out_shape[0] = 1;
        } else {
            int j = 0;
            for (int i = 0; i < input->ndim; i++) {
                if (i != normalized_dim) out_shape[j++] = input->shape[i];
            }
        }
    }

    Tensor* out = zeros_tensor(input->dtype, input->device_id, out_ndim, out_shape, NULL);
    free(out_shape);
    if (!out) {
        if (made_contiguous) free_tensor(input);
        PyErr_SetString(PyExc_RuntimeError, "Failed to allocate output tensor");
        return NULL;
    }

    int status = rp_sum_dim(out->data, input->data, outer_size, dim_size, inner_size,
                            input->dtype, input->device_id);
    if (made_contiguous) free_tensor(input);

    if (status != 0) {
        free_tensor(out);
        PyErr_SetString(PyExc_RuntimeError, "sum_dim operation failed");
        return NULL;
    }

    return wrap_tensor_result(out);
}

static PyObject* PyTensor_mean_dim(PyTensorObject* self, PyObject* args, PyObject* kwargs) {
    if (self->tensor == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized");
        return NULL;
    }

    int dim;
    int keepdim = 0;
    static char* kwlist[] = {"dim", "keepdim", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i|p", kwlist, &dim, &keepdim))
        return NULL;

    Tensor* a = self->tensor;

    int normalized_dim = dim;
    if (normalized_dim < 0) normalized_dim += a->ndim;
    if (normalized_dim < 0 || normalized_dim >= a->ndim) {
        PyErr_SetString(PyExc_ValueError, "dim out of range");
        return NULL;
    }

    Tensor* input = a;
    bool made_contiguous = false;
    if (!rp_is_contiguous(a)) {
        input = rp_contiguous(a);
        if (!input) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to make tensor contiguous");
            return NULL;
        }
        made_contiguous = true;
    }

    size_t outer_size = 1;
    for (int i = 0; i < normalized_dim; i++) outer_size *= input->shape[i];
    size_t dim_size = input->shape[normalized_dim];
    size_t inner_size = 1;
    for (int i = normalized_dim + 1; i < input->ndim; i++) inner_size *= input->shape[i];

    int out_ndim;
    int* out_shape;
    if (keepdim) {
        out_ndim = input->ndim;
        out_shape = (int*)malloc(out_ndim * sizeof(int));
        if (!out_shape) { if (made_contiguous) free_tensor(input); return PyErr_NoMemory(); }
        memcpy(out_shape, input->shape, out_ndim * sizeof(int));
        out_shape[normalized_dim] = 1;
    } else {
        out_ndim = input->ndim - 1;
        if (out_ndim == 0) out_ndim = 1;
        out_shape = (int*)malloc(out_ndim * sizeof(int));
        if (!out_shape) { if (made_contiguous) free_tensor(input); return PyErr_NoMemory(); }
        if (input->ndim == 1) {
            out_shape[0] = 1;
        } else {
            int j = 0;
            for (int i = 0; i < input->ndim; i++) {
                if (i != normalized_dim) out_shape[j++] = input->shape[i];
            }
        }
    }

    Tensor* out = zeros_tensor(input->dtype, input->device_id, out_ndim, out_shape, NULL);
    free(out_shape);
    if (!out) {
        if (made_contiguous) free_tensor(input);
        PyErr_SetString(PyExc_RuntimeError, "Failed to allocate output tensor");
        return NULL;
    }

    int status = rp_mean_dim(out->data, input->data, outer_size, dim_size, inner_size,
                             input->dtype, input->device_id);
    if (made_contiguous) free_tensor(input);

    if (status != 0) {
        free_tensor(out);
        PyErr_SetString(PyExc_RuntimeError, "mean_dim operation failed");
        return NULL;
    }

    return wrap_tensor_result(out);
}

static PyObject* PyTensor_op_sum_dim(PyTensorObject* self, PyObject* args, PyObject* kwargs) {
    if (self->tensor == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized");
        return NULL;
    }

    int dim;
    int keepdim = 0;
    static char* kwlist[] = {"dim", "keepdim", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i|p", kwlist, &dim, &keepdim))
        return NULL;

    Tensor* result = op_sum_dim(self->tensor, dim, (bool)keepdim);
    if (result == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "op_sum_dim operation failed");
        return NULL;
    }
    return wrap_tensor_result(result);
}

static PyObject* PyTensor_op_mean_dim(PyTensorObject* self, PyObject* args, PyObject* kwargs) {
    if (self->tensor == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized");
        return NULL;
    }

    int dim;
    int keepdim = 0;
    static char* kwlist[] = {"dim", "keepdim", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i|p", kwlist, &dim, &keepdim))
        return NULL;

    Tensor* result = op_mean_dim(self->tensor, dim, (bool)keepdim);
    if (result == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "op_mean_dim operation failed");
        return NULL;
    }
    return wrap_tensor_result(result);
}

static int* tensor_to_int_array(Tensor* t) {
    Tensor* host_t = t;
    bool need_free = false;
    if (t->device_id >= 0) {
        host_t = tensor_to(t, -1, t->dtype, false);
        if (!host_t) return NULL;
        need_free = true;
    }
    Tensor* f32_t = host_t;
    bool need_free_f32 = false;
    if (host_t->dtype != DTYPE_FLOAT32) {
        f32_t = tensor_to(host_t, -1, DTYPE_FLOAT32, false);
        if (!f32_t) { if (need_free) free_tensor(host_t); return NULL; }
        need_free_f32 = true;
    }
    int* result = (int*)malloc(f32_t->size * sizeof(int));
    if (result) {
        const float* data = (const float*)f32_t->data;
        for (size_t i = 0; i < f32_t->size; i++) {
            result[i] = (int)data[i];
        }
    }
    if (need_free_f32) free_tensor(f32_t);
    if (need_free) free_tensor(host_t);
    return result;
}

static PyObject* PyTensor_op_embedding(PyTensorObject* self, PyObject* args) {
    if (self->tensor == NULL) { PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized"); return NULL; }

    PyObject* indices_obj;
    if (!PyArg_ParseTuple(args, "O", &indices_obj)) return NULL;

    int* indices = NULL;
    int num_indices = 0;

    if (PyObject_TypeCheck(indices_obj, &PyTensorType)) {
        indices = tensor_to_int_array(((PyTensorObject*)indices_obj)->tensor);
        num_indices = ((PyTensorObject*)indices_obj)->tensor->size;
    } else if (PySequence_Check(indices_obj)) {
        num_indices = (int)PySequence_Size(indices_obj);
        indices = (int*)malloc(num_indices * sizeof(int));
        for (int i = 0; i < num_indices; i++) {
            PyObject* item = PySequence_GetItem(indices_obj, i);
            indices[i] = (int)PyLong_AsLong(item);
            Py_DECREF(item);
        }
    }
    if (!indices) { PyErr_SetString(PyExc_RuntimeError, "Failed to parse indices"); return NULL; }

    Tensor* result = op_embedding(self->tensor, indices, num_indices);
    free(indices);
    if (!result) { PyErr_SetString(PyExc_RuntimeError, "op_embedding failed"); return NULL; }
    return wrap_tensor_result(result);
}

static PyObject* PyTensor_op_layer_norm(PyTensorObject* self, PyObject* args, PyObject* kwargs) {
    if (self->tensor == NULL) { PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized"); return NULL; }

    PyObject* norm_shape_obj;
    PyObject* weight_obj = Py_None;
    PyObject* bias_obj = Py_None;
    float eps = 1e-5f;

    static char* kwlist[] = {"normalized_shape", "weight", "bias", "eps", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OOf", kwlist, &norm_shape_obj, &weight_obj, &bias_obj, &eps))
        return NULL;

    if (!PySequence_Check(norm_shape_obj)) { PyErr_SetString(PyExc_TypeError, "normalized_shape must be a sequence"); return NULL; }
    Py_ssize_t norm_ndim = PySequence_Size(norm_shape_obj);
    int* norm_shape = (int*)malloc(norm_ndim * sizeof(int));
    for (Py_ssize_t i = 0; i < norm_ndim; i++) {
        PyObject* item = PySequence_GetItem(norm_shape_obj, i);
        norm_shape[i] = (int)PyLong_AsLong(item);
        Py_DECREF(item);
    }

    Tensor* x = self->tensor;
    const void* w_ptr = NULL;
    const void* b_ptr = NULL;
    Tensor* w_t = NULL;
    Tensor* b_t = NULL;

    if (weight_obj != Py_None && PyObject_TypeCheck(weight_obj, &PyTensorType)) {
        w_t = ((PyTensorObject*)weight_obj)->tensor;
        w_ptr = w_t->data;
    }
    if (bias_obj != Py_None && PyObject_TypeCheck(bias_obj, &PyTensorType)) {
        b_t = ((PyTensorObject*)bias_obj)->tensor;
        b_ptr = b_t->data;
    }

    Tensor* result = op_layer_norm(x, w_ptr, b_ptr, (int)norm_ndim, norm_shape, eps, x->dtype, x->device_id);
    free(norm_shape);

    if (!result) { PyErr_SetString(PyExc_RuntimeError, "op_layer_norm failed"); return NULL; }

    if (result->metadata && result->metadata->grad_fn) {
        GradFn* gf = (GradFn*)result->metadata->grad_fn;
        if (w_t) { gf->inputs[4] = w_t; gf->num_inputs = 5; }
        if (b_t) { gf->inputs[5] = b_t; gf->num_inputs = 6; }
    }

    return wrap_tensor_result(result);
}

static PyObject* PyTensor_op_batch_norm(PyTensorObject* self, PyObject* args, PyObject* kwargs) {
    if (self->tensor == NULL) { PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized"); return NULL; }

    PyObject* weight_obj = Py_None;
    PyObject* bias_obj = Py_None;
    PyObject* rm_obj = Py_None;
    PyObject* rv_obj = Py_None;
    float eps = 1e-5f;
    float momentum = 0.1f;
    int training = 1;

    static char* kwlist[] = {"weight", "bias", "running_mean", "running_var", "eps", "momentum", "training", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OOOOffp", kwlist,
            &weight_obj, &bias_obj, &rm_obj, &rv_obj, &eps, &momentum, &training))
        return NULL;

    Tensor* x = self->tensor;
    if (x->ndim != 4 && x->ndim != 2) { PyErr_SetString(PyExc_ValueError, "input must be 2D or 4D"); return NULL; }

    int N = x->shape[0];
    int C = x->shape[1];
    int spatial = 1;
    for (int i = 2; i < x->ndim; i++) spatial *= x->shape[i];

    const void* w_ptr = NULL; const void* b_ptr = NULL;
    void* rm_ptr = NULL; void* rv_ptr = NULL;
    if (weight_obj != Py_None && PyObject_TypeCheck(weight_obj, &PyTensorType)) w_ptr = ((PyTensorObject*)weight_obj)->tensor->data;
    if (bias_obj != Py_None && PyObject_TypeCheck(bias_obj, &PyTensorType)) b_ptr = ((PyTensorObject*)bias_obj)->tensor->data;
    if (rm_obj != Py_None && PyObject_TypeCheck(rm_obj, &PyTensorType)) rm_ptr = ((PyTensorObject*)rm_obj)->tensor->data;
    if (rv_obj != Py_None && PyObject_TypeCheck(rv_obj, &PyTensorType)) rv_ptr = ((PyTensorObject*)rv_obj)->tensor->data;

    Tensor* out = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL);
    if (!out) { PyErr_SetString(PyExc_RuntimeError, "allocation failed"); return NULL; }

    int sm_shape[] = {C};
    Tensor* save_mean = zeros_tensor(x->dtype, x->device_id, 1, sm_shape, NULL);
    Tensor* save_rstd = zeros_tensor(x->dtype, x->device_id, 1, sm_shape, NULL);

    int ret = rp_batch_norm(out->data, save_mean ? save_mean->data : NULL, save_rstd ? save_rstd->data : NULL,
                            x->data, w_ptr, b_ptr, rm_ptr, rv_ptr, N, C, spatial, eps, momentum, training,
                            x->dtype, x->device_id);

    if (save_mean) free_tensor(save_mean);
    if (save_rstd) free_tensor(save_rstd);

    if (ret != 0) { free_tensor(out); PyErr_SetString(PyExc_RuntimeError, "batch_norm failed"); return NULL; }
    return wrap_tensor_result(out);
}

static PyObject* PyTensor_op_dropout(PyTensorObject* self, PyObject* args) {
    if (self->tensor == NULL) { PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized"); return NULL; }
    float p = 0.5f;
    if (!PyArg_ParseTuple(args, "|f", &p)) return NULL;
    Tensor* result = op_dropout(self->tensor, p);
    if (!result) { PyErr_SetString(PyExc_RuntimeError, "op_dropout failed"); return NULL; }
    return wrap_tensor_result(result);
}

static PyObject* PyTensor_op_maxpool2d(PyTensorObject* self, PyObject* args, PyObject* kwargs) {
    if (self->tensor == NULL) { PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized"); return NULL; }
    int kH, kW, stride_h = -1, stride_w = -1, pad_h = 0, pad_w = 0;
    static char* kwlist[] = {"kernel_h", "kernel_w", "stride_h", "stride_w", "pad_h", "pad_w", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii|iiii", kwlist, &kH, &kW, &stride_h, &stride_w, &pad_h, &pad_w)) return NULL;
    if (stride_h < 0) stride_h = kH;
    if (stride_w < 0) stride_w = kW;
    Tensor* result = op_maxpool2d(self->tensor, kH, kW, stride_h, stride_w, pad_h, pad_w);
    if (!result) { PyErr_SetString(PyExc_RuntimeError, "op_maxpool2d failed"); return NULL; }
    return wrap_tensor_result(result);
}

static PyObject* PyTensor_op_avgpool2d(PyTensorObject* self, PyObject* args, PyObject* kwargs) {
    if (self->tensor == NULL) { PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized"); return NULL; }
    int kH, kW, stride_h = -1, stride_w = -1, pad_h = 0, pad_w = 0;
    static char* kwlist[] = {"kernel_h", "kernel_w", "stride_h", "stride_w", "pad_h", "pad_w", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii|iiii", kwlist, &kH, &kW, &stride_h, &stride_w, &pad_h, &pad_w)) return NULL;
    if (stride_h < 0) stride_h = kH;
    if (stride_w < 0) stride_w = kW;
    Tensor* result = op_avgpool2d(self->tensor, kH, kW, stride_h, stride_w, pad_h, pad_w);
    if (!result) { PyErr_SetString(PyExc_RuntimeError, "op_avgpool2d failed"); return NULL; }
    return wrap_tensor_result(result);
}

static PyObject* PyTensor_op_conv2d(PyTensorObject* self, PyObject* args, PyObject* kwargs) {
    if (self->tensor == NULL) { PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized"); return NULL; }

    PyObject* weight_obj;
    PyObject* bias_obj = Py_None;
    int stride_h = 1, stride_w = 1;
    int pad_h = 0, pad_w = 0;
    int dilation_h = 1, dilation_w = 1;

    static char* kwlist[] = {"weight", "bias", "stride_h", "stride_w", "pad_h", "pad_w", "dilation_h", "dilation_w", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|Oiiiiii", kwlist,
            &weight_obj, &bias_obj, &stride_h, &stride_w, &pad_h, &pad_w, &dilation_h, &dilation_w))
        return NULL;

    if (!PyObject_TypeCheck(weight_obj, &PyTensorType)) {
        PyErr_SetString(PyExc_TypeError, "weight must be a Tensor");
        return NULL;
    }

    Tensor* weight = ((PyTensorObject*)weight_obj)->tensor;
    Tensor* bias = NULL;
    if (bias_obj != Py_None) {
        if (!PyObject_TypeCheck(bias_obj, &PyTensorType)) {
            PyErr_SetString(PyExc_TypeError, "bias must be a Tensor or None");
            return NULL;
        }
        bias = ((PyTensorObject*)bias_obj)->tensor;
    }

    Tensor* result = op_conv2d(self->tensor, weight, bias, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w);
    if (!result) { PyErr_SetString(PyExc_RuntimeError, "op_conv2d failed"); return NULL; }
    return wrap_tensor_result(result);
}

static int parse_reduction(const char* s) {
    if (!s || strcmp(s, "mean") == 0) return REDUCTION_MEAN;
    if (strcmp(s, "sum") == 0) return REDUCTION_SUM;
    if (strcmp(s, "none") == 0) return REDUCTION_NONE;
    return -1;
}

static PyObject* PyTensor_op_mse_loss(PyTensorObject* self, PyObject* args, PyObject* kwargs) {
    if (self->tensor == NULL) { PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized"); return NULL; }
    PyObject* target_obj;
    const char* reduction_str = "mean";
    static char* kwlist[] = {"target", "reduction", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|s", kwlist, &target_obj, &reduction_str)) return NULL;
    if (!PyObject_TypeCheck(target_obj, &PyTensorType)) { PyErr_SetString(PyExc_TypeError, "target must be a Tensor"); return NULL; }
    int reduction = parse_reduction(reduction_str);
    if (reduction < 0) { PyErr_SetString(PyExc_ValueError, "reduction must be 'mean', 'sum', or 'none'"); return NULL; }
    Tensor* result = op_mse_loss(self->tensor, ((PyTensorObject*)target_obj)->tensor, reduction);
    if (!result) { PyErr_SetString(PyExc_RuntimeError, "op_mse_loss failed"); return NULL; }
    return wrap_tensor_result(result);
}

static PyObject* PyTensor_op_bce_loss(PyTensorObject* self, PyObject* args, PyObject* kwargs) {
    if (self->tensor == NULL) { PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized"); return NULL; }
    PyObject* target_obj;
    const char* reduction_str = "mean";
    int from_logits = -1;
    static char* kwlist[] = {"target", "reduction", "from_logits", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|si", kwlist, &target_obj, &reduction_str, &from_logits)) return NULL;
    if (!PyObject_TypeCheck(target_obj, &PyTensorType)) { PyErr_SetString(PyExc_TypeError, "target must be a Tensor"); return NULL; }
    int reduction = parse_reduction(reduction_str);
    if (reduction < 0) { PyErr_SetString(PyExc_ValueError, "reduction must be 'mean', 'sum', or 'none'"); return NULL; }
    Tensor* result = op_bce_loss(self->tensor, ((PyTensorObject*)target_obj)->tensor, reduction, from_logits);
    if (!result) { PyErr_SetString(PyExc_RuntimeError, "op_bce_loss failed"); return NULL; }
    return wrap_tensor_result(result);
}

static PyObject* PyTensor_op_nll_loss(PyTensorObject* self, PyObject* args, PyObject* kwargs) {
    if (self->tensor == NULL) { PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized"); return NULL; }
    PyObject* targets_obj;
    const char* reduction_str = "mean";
    static char* kwlist[] = {"targets", "reduction", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|s", kwlist, &targets_obj, &reduction_str)) return NULL;
    int reduction = parse_reduction(reduction_str);
    if (reduction < 0) { PyErr_SetString(PyExc_ValueError, "reduction must be 'mean', 'sum', or 'none'"); return NULL; }
    Tensor* input = self->tensor;
    if (input->ndim != 2) { PyErr_SetString(PyExc_ValueError, "input must be 2D [batch, classes]"); return NULL; }
    int batch_size = input->shape[0];
    int num_classes = input->shape[1];
    int* targets = NULL;
    if (PyObject_TypeCheck(targets_obj, &PyTensorType)) {
        targets = tensor_to_int_array(((PyTensorObject*)targets_obj)->tensor);
    } else if (PySequence_Check(targets_obj)) {
        Py_ssize_t n = PySequence_Size(targets_obj);
        targets = (int*)malloc(n * sizeof(int));
        for (Py_ssize_t i = 0; i < n; i++) {
            PyObject* item = PySequence_GetItem(targets_obj, i);
            targets[i] = (int)PyLong_AsLong(item);
            Py_DECREF(item);
        }
    }
    if (!targets) { PyErr_SetString(PyExc_RuntimeError, "Failed to parse targets"); return NULL; }
    Tensor* result = op_nll_loss(input, targets, batch_size, num_classes, reduction);
    free(targets);
    if (!result) { PyErr_SetString(PyExc_RuntimeError, "op_nll_loss failed"); return NULL; }
    return wrap_tensor_result(result);
}

static PyObject* PyTensor_op_cross_entropy_loss(PyTensorObject* self, PyObject* args, PyObject* kwargs) {
    if (self->tensor == NULL) { PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized"); return NULL; }
    PyObject* targets_obj;
    const char* reduction_str = "mean";
    int from_logits = -1;
    static char* kwlist[] = {"targets", "reduction", "from_logits", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|si", kwlist, &targets_obj, &reduction_str, &from_logits)) return NULL;
    int reduction = parse_reduction(reduction_str);
    if (reduction < 0) { PyErr_SetString(PyExc_ValueError, "reduction must be 'mean', 'sum', or 'none'"); return NULL; }
    Tensor* input = self->tensor;
    if (input->ndim != 2) { PyErr_SetString(PyExc_ValueError, "input must be 2D [batch, classes]"); return NULL; }
    int batch_size = input->shape[0];
    int num_classes = input->shape[1];
    int* targets = NULL;
    if (PyObject_TypeCheck(targets_obj, &PyTensorType)) {
        targets = tensor_to_int_array(((PyTensorObject*)targets_obj)->tensor);
    } else if (PySequence_Check(targets_obj)) {
        Py_ssize_t n = PySequence_Size(targets_obj);
        targets = (int*)malloc(n * sizeof(int));
        for (Py_ssize_t i = 0; i < n; i++) {
            PyObject* item = PySequence_GetItem(targets_obj, i);
            targets[i] = (int)PyLong_AsLong(item);
            Py_DECREF(item);
        }
    }
    if (!targets) { PyErr_SetString(PyExc_RuntimeError, "Failed to parse targets"); return NULL; }
    Tensor* result = op_cross_entropy_loss(input, targets, batch_size, num_classes, reduction, from_logits);
    free(targets);
    if (!result) { PyErr_SetString(PyExc_RuntimeError, "op_cross_entropy_loss failed"); return NULL; }
    return wrap_tensor_result(result);
}

static PyObject* PyTensor_op_gather(PyTensorObject* self, PyObject* args) {
    if (self->tensor == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized");
        return NULL;
    }

    int dim;
    PyObject* index_obj;
    if (!PyArg_ParseTuple(args, "iO", &dim, &index_obj))
        return NULL;

    if (!PyObject_TypeCheck(index_obj, &PyTensorType)) {
        PyErr_SetString(PyExc_TypeError, "index must be a Tensor");
        return NULL;
    }

    PyTensorObject* index_py = (PyTensorObject*)index_obj;
    if (!index_py->tensor) {
        PyErr_SetString(PyExc_RuntimeError, "Index tensor is not initialized");
        return NULL;
    }

    Tensor* index_t = index_py->tensor;
    int* indices = tensor_to_int_array(index_t);
    if (!indices) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to convert index tensor to int array");
        return NULL;
    }

    Tensor* result = op_gather(self->tensor, dim, indices, index_t->ndim, index_t->shape, index_t->size);
    free(indices);

    if (result == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "op_gather operation failed");
        return NULL;
    }
    return wrap_tensor_result(result);
}

static PyObject* PyTensor_op_softmax(PyTensorObject* self, PyObject* args, PyObject* kwargs) {
    if (self->tensor == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized");
        return NULL;
    }

    int dim = -1;
    static char* kwlist[] = {"dim", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|i", kwlist, &dim))
        return NULL;

    Tensor* result = op_softmax(self->tensor, dim);
    if (result == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "op_softmax operation failed");
        return NULL;
    }
    return wrap_tensor_result(result);
}

static PyObject* PyTensor_op_log_softmax(PyTensorObject* self, PyObject* args, PyObject* kwargs) {
    if (self->tensor == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized");
        return NULL;
    }

    int dim = -1;
    static char* kwlist[] = {"dim", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|i", kwlist, &dim))
        return NULL;

    Tensor* result = op_log_softmax(self->tensor, dim);
    if (result == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "op_log_softmax operation failed");
        return NULL;
    }
    return wrap_tensor_result(result);
}

static PyObject* PyTensor_backward(PyTensorObject* self, PyObject* Py_UNUSED(ignored)) {
    if (self->tensor == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized");
        return NULL;
    }

    int status = tensor_backward(self->tensor);
    if (status != 0) {
        PyErr_SetString(PyExc_RuntimeError, "backward operation failed");
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject* PyTensor_zero_grad(PyTensorObject* self, PyObject* Py_UNUSED(ignored)) {
    if (self->tensor == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized");
        return NULL;
    }
    tensor_zero_grad(self->tensor);
    Py_RETURN_NONE;
}

static PyObject* PyTensor_sgd_step(PyTensorObject* self, PyObject* args, PyObject* kwargs) {
    if (self->tensor == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized");
        return NULL;
    }

    double lr = 0.01;
    double momentum = 0.0;
    double dampening = 0.0;
    int nesterov = 0;
    double weight_decay = 0.0;
    PyObject* buf_obj = Py_None;
    int has_buf = 0;

    static char* kwlist[] = {"lr", "momentum", "dampening", "nesterov",
                             "weight_decay", "momentum_buf", "has_momentum_buf", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "dddpd|Op", kwlist,
                                      &lr, &momentum, &dampening, &nesterov,
                                      &weight_decay, &buf_obj, &has_buf)) {
        return NULL;
    }

    Tensor* buf = NULL;
    if (buf_obj != Py_None && Py_TYPE(buf_obj) == &PyTensorType) {
        buf = ((PyTensorObject*)buf_obj)->tensor;
    }

    int status = sgd_step(self->tensor, lr, momentum, dampening,
                          (bool)nesterov, weight_decay, buf, (bool)has_buf);
    if (status != 0) {
        PyErr_SetString(PyExc_RuntimeError, "sgd_step failed");
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* PyTensor_adam_step(PyTensorObject* self, PyObject* args, PyObject* kwargs) {
    if (self->tensor == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized");
        return NULL;
    }

    double lr = 0.001;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double eps = 1e-8;
    double weight_decay = 0.0;
    int amsgrad = 0;
    PyObject* m_obj = NULL;
    PyObject* v_obj = NULL;
    PyObject* v_max_obj = Py_None;
    int step = 1;

    static char* kwlist[] = {"lr", "beta1", "beta2", "eps", "weight_decay",
                             "amsgrad", "m", "v", "v_max", "step", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "dddddpOO|Oi", kwlist,
                                      &lr, &beta1, &beta2, &eps,
                                      &weight_decay, &amsgrad,
                                      &m_obj, &v_obj, &v_max_obj, &step)) {
        return NULL;
    }

    if (Py_TYPE(m_obj) != &PyTensorType || Py_TYPE(v_obj) != &PyTensorType) {
        PyErr_SetString(PyExc_TypeError, "m and v must be Tensor objects");
        return NULL;
    }

    Tensor* m = ((PyTensorObject*)m_obj)->tensor;
    Tensor* v = ((PyTensorObject*)v_obj)->tensor;
    Tensor* v_max = NULL;
    if (v_max_obj != Py_None && Py_TYPE(v_max_obj) == &PyTensorType) {
        v_max = ((PyTensorObject*)v_max_obj)->tensor;
    }

    int status = adam_step(self->tensor, lr, beta1, beta2, eps, weight_decay,
                           (bool)amsgrad, m, v, v_max, step);
    if (status != 0) {
        PyErr_SetString(PyExc_RuntimeError, "adam_step failed");
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* PyTensor_view(PyTensorObject* self, PyObject* args) {
    if (self->tensor == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized");
        return NULL;
    }

    PyObject* shape_obj;
    if (!PyArg_ParseTuple(args, "O", &shape_obj)) {
        return NULL;
    }

    if (!PyList_Check(shape_obj) && !PyTuple_Check(shape_obj)) {
        PyErr_SetString(PyExc_TypeError, "Shape must be a list or tuple");
        return NULL;
    }

    Py_ssize_t ndim = PySequence_Size(shape_obj);
    if (ndim <= 0) {
        PyErr_SetString(PyExc_ValueError, "Shape must have at least one dimension");
        return NULL;
    }

    int* new_shape = (int*)malloc(ndim * sizeof(int));
    if (!new_shape) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for shape");
        return NULL;
    }

    for (Py_ssize_t i = 0; i < ndim; i++) {
        PyObject* item = PySequence_GetItem(shape_obj, i);
        if (!item) {
            free(new_shape);
            return NULL;
        }

        if (!PyLong_Check(item)) {
            Py_DECREF(item);
            free(new_shape);
            PyErr_SetString(PyExc_TypeError, "Shape dimensions must be integers");
            return NULL;
        }

        new_shape[i] = (int)PyLong_AsLong(item);
        Py_DECREF(item);
    }

    Tensor* result = op_view(self->tensor, (int)ndim, new_shape);
    free(new_shape);

    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "view operation failed");
        return NULL;
    }

    PyTensorObject* py_result = (PyTensorObject*)PyTensorType.tp_alloc(&PyTensorType, 0);
    if (!py_result) {
        free_tensor(result);
        return NULL;
    }

    py_result->tensor = result;
    return (PyObject*)py_result;
}

static PyObject* PyTensor_reshape(PyTensorObject* self, PyObject* args) {
    if (self->tensor == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized");
        return NULL;
    }

    PyObject* shape_obj;
    if (!PyArg_ParseTuple(args, "O", &shape_obj)) {
        return NULL;
    }

    if (!PyList_Check(shape_obj) && !PyTuple_Check(shape_obj)) {
        PyErr_SetString(PyExc_TypeError, "Shape must be a list or tuple");
        return NULL;
    }

    Py_ssize_t ndim = PySequence_Size(shape_obj);
    if (ndim <= 0) {
        PyErr_SetString(PyExc_ValueError, "Shape must have at least one dimension");
        return NULL;
    }

    int* new_shape = (int*)malloc(ndim * sizeof(int));
    if (!new_shape) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for shape");
        return NULL;
    }

    for (Py_ssize_t i = 0; i < ndim; i++) {
        PyObject* item = PySequence_GetItem(shape_obj, i);
        if (!item) {
            free(new_shape);
            return NULL;
        }

        if (!PyLong_Check(item)) {
            Py_DECREF(item);
            free(new_shape);
            PyErr_SetString(PyExc_TypeError, "Shape dimensions must be integers");
            return NULL;
        }

        new_shape[i] = (int)PyLong_AsLong(item);
        Py_DECREF(item);
    }

    Tensor* result = op_reshape(self->tensor, (int)ndim, new_shape);
    free(new_shape);

    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "reshape operation failed");
        return NULL;
    }

    PyTensorObject* py_result = (PyTensorObject*)PyTensorType.tp_alloc(&PyTensorType, 0);
    if (!py_result) {
        free_tensor(result);
        return NULL;
    }

    py_result->tensor = result;
    return (PyObject*)py_result;
}

static PyObject* PyTensor_transpose(PyTensorObject* self, PyObject* args) {
    if (self->tensor == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized");
        return NULL;
    }

    int dim0, dim1;
    if (!PyArg_ParseTuple(args, "ii", &dim0, &dim1)) {
        return NULL;
    }

    Tensor* result = op_transpose(self->tensor, dim0, dim1);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "transpose operation failed");
        return NULL;
    }

    PyTensorObject* py_result = (PyTensorObject*)PyTensorType.tp_alloc(&PyTensorType, 0);
    if (!py_result) {
        free_tensor(result);
        return NULL;
    }

    py_result->tensor = result;
    return (PyObject*)py_result;
}

static PyObject* PyTensor_squeeze(PyTensorObject* self, PyObject* args) {
    if (self->tensor == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized");
        return NULL;
    }

    int dim = -1;
    if (!PyArg_ParseTuple(args, "|i", &dim)) {
        return NULL;
    }

    Tensor* result = op_squeeze(self->tensor, dim);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "squeeze operation failed");
        return NULL;
    }

    PyTensorObject* py_result = (PyTensorObject*)PyTensorType.tp_alloc(&PyTensorType, 0);
    if (!py_result) {
        free_tensor(result);
        return NULL;
    }

    py_result->tensor = result;
    return (PyObject*)py_result;
}

static PyObject* PyTensor_unsqueeze(PyTensorObject* self, PyObject* args) {
    if (self->tensor == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized");
        return NULL;
    }

    int dim;
    if (!PyArg_ParseTuple(args, "i", &dim)) {
        return NULL;
    }

    Tensor* result = op_unsqueeze(self->tensor, dim);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "unsqueeze operation failed");
        return NULL;
    }

    PyTensorObject* py_result = (PyTensorObject*)PyTensorType.tp_alloc(&PyTensorType, 0);
    if (!py_result) {
        free_tensor(result);
        return NULL;
    }

    py_result->tensor = result;
    return (PyObject*)py_result;
}

static PyObject* PyTensor_flatten(PyTensorObject* self, PyObject* args) {
    if (self->tensor == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized");
        return NULL;
    }

    int start_dim = 0;
    int end_dim = -1;
    if (!PyArg_ParseTuple(args, "|ii", &start_dim, &end_dim)) {
        return NULL;
    }

    Tensor* result = op_flatten(self->tensor, start_dim, end_dim);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "flatten operation failed");
        return NULL;
    }

    PyTensorObject* py_result = (PyTensorObject*)PyTensorType.tp_alloc(&PyTensorType, 0);
    if (!py_result) {
        free_tensor(result);
        return NULL;
    }

    py_result->tensor = result;
    return (PyObject*)py_result;
}

static PyObject* PyTensor_permute(PyTensorObject* self, PyObject* args) {
    if (self->tensor == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized");
        return NULL;
    }

    PyObject* dims_obj;
    if (!PyArg_ParseTuple(args, "O", &dims_obj)) {
        return NULL;
    }

    if (!PyList_Check(dims_obj) && !PyTuple_Check(dims_obj)) {
        PyErr_SetString(PyExc_TypeError, "dims must be a list or tuple");
        return NULL;
    }

    Py_ssize_t ndim = PySequence_Size(dims_obj);
    if (ndim != self->tensor->ndim) {
        PyErr_Format(PyExc_ValueError,
                     "dims length (%zd) must equal tensor ndim (%d)",
                     ndim, self->tensor->ndim);
        return NULL;
    }

    int* dims = (int*)malloc(ndim * sizeof(int));
    if (!dims) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for dims");
        return NULL;
    }

    for (Py_ssize_t i = 0; i < ndim; i++) {
        PyObject* item = PySequence_GetItem(dims_obj, i);
        if (!item) {
            free(dims);
            return NULL;
        }

        if (!PyLong_Check(item)) {
            Py_DECREF(item);
            free(dims);
            PyErr_SetString(PyExc_TypeError, "dims must contain integers");
            return NULL;
        }

        dims[i] = (int)PyLong_AsLong(item);
        Py_DECREF(item);
    }

    Tensor* result = op_permute(self->tensor, dims);
    free(dims);

    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "permute operation failed");
        return NULL;
    }

    PyTensorObject* py_result = (PyTensorObject*)PyTensorType.tp_alloc(&PyTensorType, 0);
    if (!py_result) {
        free_tensor(result);
        return NULL;
    }

    py_result->tensor = result;
    return (PyObject*)py_result;
}

static PyObject* PyTensor_chunk(PyTensorObject* self, PyObject* args) {
    if (self->tensor == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized");
        return NULL;
    }

    int chunks;
    int dim = 0;
    if (!PyArg_ParseTuple(args, "i|i", &chunks, &dim)) {
        return NULL;
    }

    Tensor** result = op_chunk(self->tensor, chunks, dim);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "chunk operation failed");
        return NULL;
    }

    int num_chunks = 0;
    while (result[num_chunks] != NULL) {
        num_chunks++;
    }

    PyObject* py_list = PyList_New(num_chunks);
    if (!py_list) {
        for (int i = 0; i < num_chunks; i++) {
            free_tensor(result[i]);
        }
        free(result);
        return NULL;
    }

    for (int i = 0; i < num_chunks; i++) {
        PyTensorObject* py_tensor = (PyTensorObject*)PyTensorType.tp_alloc(&PyTensorType, 0);
        if (!py_tensor) {
            Py_DECREF(py_list);
            for (int j = i; j < num_chunks; j++) {
                free_tensor(result[j]);
            }
            free(result);
            return NULL;
        }
        py_tensor->tensor = result[i];
        PyList_SET_ITEM(py_list, i, (PyObject*)py_tensor);
    }

    free(result);
    return py_list;
}

static PyObject* PyTensor_split(PyTensorObject* self, PyObject* args) {
    if (self->tensor == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized");
        return NULL;
    }

    PyObject* sizes_obj;
    int dim = 0;
    if (!PyArg_ParseTuple(args, "O|i", &sizes_obj, &dim)) {
        return NULL;
    }

    if (!PyList_Check(sizes_obj) && !PyTuple_Check(sizes_obj)) {
        PyErr_SetString(PyExc_TypeError, "sizes must be a list or tuple");
        return NULL;
    }

    Py_ssize_t num_splits = PySequence_Size(sizes_obj);
    int* sizes = (int*)malloc(num_splits * sizeof(int));
    if (!sizes) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for sizes");
        return NULL;
    }

    for (Py_ssize_t i = 0; i < num_splits; i++) {
        PyObject* item = PySequence_GetItem(sizes_obj, i);
        if (!item) {
            free(sizes);
            return NULL;
        }
        if (!PyLong_Check(item)) {
            Py_DECREF(item);
            free(sizes);
            PyErr_SetString(PyExc_TypeError, "sizes must contain integers");
            return NULL;
        }
        sizes[i] = (int)PyLong_AsLong(item);
        Py_DECREF(item);
    }

    Tensor** result = op_split(self->tensor, sizes, (int)num_splits, dim);
    free(sizes);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "split operation failed");
        return NULL;
    }

    PyObject* py_list = PyList_New((Py_ssize_t)num_splits);
    if (!py_list) {
        for (Py_ssize_t i = 0; i < num_splits; i++) {
            free_tensor(result[i]);
        }
        free(result);
        return NULL;
    }

    for (Py_ssize_t i = 0; i < num_splits; i++) {
        PyTensorObject* py_tensor = (PyTensorObject*)PyTensorType.tp_alloc(&PyTensorType, 0);
        if (!py_tensor) {
            Py_DECREF(py_list);
            for (Py_ssize_t j = i; j < num_splits; j++) {
                free_tensor(result[j]);
            }
            free(result);
            return NULL;
        }
        py_tensor->tensor = result[i];
        PyList_SET_ITEM(py_list, i, (PyObject*)py_tensor);
    }

    free(result);
    return py_list;
}

static PyObject* PyTensor_split_equal(PyTensorObject* self, PyObject* args) {
    if (self->tensor == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized");
        return NULL;
    }

    int num_splits;
    int dim = 0;
    if (!PyArg_ParseTuple(args, "i|i", &num_splits, &dim)) {
        return NULL;
    }

    Tensor** result = op_split_equal(self->tensor, num_splits, dim);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "split_equal operation failed");
        return NULL;
    }

    PyObject* py_list = PyList_New(num_splits);
    if (!py_list) {
        for (int i = 0; i < num_splits; i++) {
            free_tensor(result[i]);
        }
        free(result);
        return NULL;
    }

    for (int i = 0; i < num_splits; i++) {
        PyTensorObject* py_tensor = (PyTensorObject*)PyTensorType.tp_alloc(&PyTensorType, 0);
        if (!py_tensor) {
            Py_DECREF(py_list);
            for (int j = i; j < num_splits; j++) {
                free_tensor(result[j]);
            }
            free(result);
            return NULL;
        }
        py_tensor->tensor = result[i];
        PyList_SET_ITEM(py_list, i, (PyObject*)py_tensor);
    }

    free(result);
    return py_list;
}

static PyObject* PyTensor_expand(PyTensorObject* self, PyObject* args) {
    if (self->tensor == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized");
        return NULL;
    }

    PyObject* shape_obj;
    if (!PyArg_ParseTuple(args, "O", &shape_obj)) {
        return NULL;
    }

    if (!PyList_Check(shape_obj) && !PyTuple_Check(shape_obj)) {
        PyErr_SetString(PyExc_TypeError, "shape must be a list or tuple");
        return NULL;
    }

    Py_ssize_t ndim = PySequence_Size(shape_obj);
    if (ndim < self->tensor->ndim) {
        PyErr_Format(PyExc_ValueError,
                     "target ndim (%zd) must be >= source ndim (%d)",
                     ndim, self->tensor->ndim);
        return NULL;
    }

    int* shape = (int*)malloc(ndim * sizeof(int));
    if (!shape) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for shape");
        return NULL;
    }

    for (Py_ssize_t i = 0; i < ndim; i++) {
        PyObject* item = PySequence_GetItem(shape_obj, i);
        if (!item) {
            free(shape);
            return NULL;
        }

        if (!PyLong_Check(item)) {
            Py_DECREF(item);
            free(shape);
            PyErr_SetString(PyExc_TypeError, "shape must contain integers");
            return NULL;
        }

        shape[i] = (int)PyLong_AsLong(item);
        Py_DECREF(item);
    }

    Tensor* result = op_expand(self->tensor, (int)ndim, shape);
    free(shape);

    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "expand operation failed");
        return NULL;
    }

    PyTensorObject* py_result = (PyTensorObject*)PyTensorType.tp_alloc(&PyTensorType, 0);
    if (!py_result) {
        free_tensor(result);
        return NULL;
    }

    py_result->tensor = result;
    return (PyObject*)py_result;
}

static PyObject* PyTensor_repeat(PyTensorObject* self, PyObject* args) {
    if (self->tensor == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized");
        return NULL;
    }

    PyObject* repeats_obj;
    if (!PyArg_ParseTuple(args, "O", &repeats_obj)) {
        return NULL;
    }

    if (!PyList_Check(repeats_obj) && !PyTuple_Check(repeats_obj)) {
        PyErr_SetString(PyExc_TypeError, "repeats must be a list or tuple");
        return NULL;
    }

    Py_ssize_t ndim = PySequence_Size(repeats_obj);
    if (ndim != self->tensor->ndim) {
        PyErr_Format(PyExc_ValueError,
                     "repeats length (%zd) must equal tensor ndim (%d)",
                     ndim, self->tensor->ndim);
        return NULL;
    }

    int* repeats = (int*)malloc(ndim * sizeof(int));
    if (!repeats) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for repeats");
        return NULL;
    }

    for (Py_ssize_t i = 0; i < ndim; i++) {
        PyObject* item = PySequence_GetItem(repeats_obj, i);
        if (!item) {
            free(repeats);
            return NULL;
        }

        if (!PyLong_Check(item)) {
            Py_DECREF(item);
            free(repeats);
            PyErr_SetString(PyExc_TypeError, "repeats must contain integers");
            return NULL;
        }

        repeats[i] = (int)PyLong_AsLong(item);
        Py_DECREF(item);

        if (repeats[i] <= 0) {
            int bad_val = repeats[i];
            free(repeats);
            PyErr_Format(PyExc_ValueError,
                         "repeat count must be positive at dimension %zd (got %d)",
                         i, bad_val);
            return NULL;
        }
    }

    Tensor* result = op_repeat(self->tensor, repeats);
    free(repeats);

    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "repeat operation failed");
        return NULL;
    }

    PyTensorObject* py_result = (PyTensorObject*)PyTensorType.tp_alloc(&PyTensorType, 0);
    if (!py_result) {
        free_tensor(result);
        return NULL;
    }

    py_result->tensor = result;
    return (PyObject*)py_result;
}

static PyObject* PyTensor_slice(PyTensorObject* self, PyObject* args, PyObject* kwargs) {
    if (self->tensor == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized");
        return NULL;
    }

    PyObject* start_obj = NULL;
    PyObject* stop_obj = NULL;
    PyObject* step_obj = NULL;

    static char* kwlist[] = {"start", "stop", "step", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OOO", kwlist,
                                     &start_obj, &stop_obj, &step_obj)) {
        return NULL;
    }

    int ndim = self->tensor->ndim;

    int* start = NULL;
    int* stop = NULL;
    int* step = NULL;

    if (start_obj && start_obj != Py_None) {
        if (!PyList_Check(start_obj) && !PyTuple_Check(start_obj)) {
            PyErr_SetString(PyExc_TypeError, "start must be a list or tuple");
            return NULL;
        }
        Py_ssize_t len = PySequence_Size(start_obj);
        if (len != ndim) {
            PyErr_Format(PyExc_ValueError, "start must have %d elements", ndim);
            return NULL;
        }
        start = (int*)malloc(ndim * sizeof(int));
        if (!start) {
            PyErr_NoMemory();
            return NULL;
        }
        for (Py_ssize_t i = 0; i < len; i++) {
            PyObject* item = PySequence_GetItem(start_obj, i);
            if (!item) {
                free(start);
                return NULL;
            }
            start[i] = (int)PyLong_AsLong(item);
            Py_DECREF(item);
            if (PyErr_Occurred()) {
                free(start);
                return NULL;
            }
        }
    }

    if (stop_obj && stop_obj != Py_None) {
        if (!PyList_Check(stop_obj) && !PyTuple_Check(stop_obj)) {
            if (start) free(start);
            PyErr_SetString(PyExc_TypeError, "stop must be a list or tuple");
            return NULL;
        }
        Py_ssize_t len = PySequence_Size(stop_obj);
        if (len != ndim) {
            if (start) free(start);
            PyErr_Format(PyExc_ValueError, "stop must have %d elements", ndim);
            return NULL;
        }
        stop = (int*)malloc(ndim * sizeof(int));
        if (!stop) {
            if (start) free(start);
            PyErr_NoMemory();
            return NULL;
        }
        for (Py_ssize_t i = 0; i < len; i++) {
            PyObject* item = PySequence_GetItem(stop_obj, i);
            if (!item) {
                free(start);
                free(stop);
                return NULL;
            }
            stop[i] = (int)PyLong_AsLong(item);
            Py_DECREF(item);
            if (PyErr_Occurred()) {
                free(start);
                free(stop);
                return NULL;
            }
        }
    }

    if (step_obj && step_obj != Py_None) {
        if (!PyList_Check(step_obj) && !PyTuple_Check(step_obj)) {
            if (start) free(start);
            if (stop) free(stop);
            PyErr_SetString(PyExc_TypeError, "step must be a list or tuple");
            return NULL;
        }
        Py_ssize_t len = PySequence_Size(step_obj);
        if (len != ndim) {
            if (start) free(start);
            if (stop) free(stop);
            PyErr_Format(PyExc_ValueError, "step must have %d elements", ndim);
            return NULL;
        }
        step = (int*)malloc(ndim * sizeof(int));
        if (!step) {
            if (start) free(start);
            if (stop) free(stop);
            PyErr_NoMemory();
            return NULL;
        }
        for (Py_ssize_t i = 0; i < len; i++) {
            PyObject* item = PySequence_GetItem(step_obj, i);
            if (!item) {
                free(start);
                free(stop);
                free(step);
                return NULL;
            }
            step[i] = (int)PyLong_AsLong(item);
            Py_DECREF(item);
            if (PyErr_Occurred()) {
                free(start);
                free(stop);
                free(step);
                return NULL;
            }
        }
    }

    Tensor* result = op_slice(self->tensor, start, stop, step);

    if (start) free(start);
    if (stop) free(stop);
    if (step) free(step);

    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "slice operation failed");
        return NULL;
    }

    PyTensorObject* py_result = (PyTensorObject*)PyTensorType.tp_alloc(&PyTensorType, 0);
    if (!py_result) {
        free_tensor(result);
        return NULL;
    }

    py_result->tensor = result;
    return (PyObject*)py_result;
}

static PyObject* PyTensor_get_T(PyTensorObject* self, void* closure) {
    (void)closure;

    if (self->tensor == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized");
        return NULL;
    }

    if (self->tensor->ndim != 2) {
        PyErr_SetString(PyExc_RuntimeError, ".T only valid for 2D tensors");
        return NULL;
    }

    Tensor* result = op_transpose(self->tensor, 0, 1);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "transpose operation failed");
        return NULL;
    }

    PyTensorObject* py_result = (PyTensorObject*)PyTensorType.tp_alloc(&PyTensorType, 0);
    if (!py_result) {
        free_tensor(result);
        return NULL;
    }

    py_result->tensor = result;
    return (PyObject*)py_result;
}

static PyObject* PyTensor_requires_grad_(PyTensorObject* self, PyObject* args) {
    if (self->tensor == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized");
        return NULL;
    }

    int requires_grad = 1;
    if (!PyArg_ParseTuple(args, "|p", &requires_grad)) {
        return NULL;
    }

    if (!self->tensor->metadata) {
        self->tensor->metadata = (Meta*)calloc(1, sizeof(Meta));
        if (!self->tensor->metadata) {
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate metadata");
            return NULL;
        }
    }

    self->tensor->metadata->requires_grad = (bool)requires_grad;
    self->tensor->metadata->is_leaf = true;

    Py_INCREF(self);
    return (PyObject*)self;
}

#define BINARY_AUTOGRAD_OP(name, op_func) \
static PyObject* PyTensor_##name(PyTensorObject* self, PyObject* args) { \
    if (self->tensor == NULL) { \
        PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized"); \
        return NULL; \
    } \
    PyObject* other_obj; \
    if (!PyArg_ParseTuple(args, "O", &other_obj)) { \
        return NULL; \
    } \
    if (!PyObject_TypeCheck(other_obj, &PyTensorType)) { \
        PyErr_SetString(PyExc_TypeError, "Argument must be a Tensor"); \
        return NULL; \
    } \
    PyTensorObject* other = (PyTensorObject*)other_obj; \
    if (other->tensor == NULL) { \
        PyErr_SetString(PyExc_RuntimeError, "Other tensor is not initialized"); \
        return NULL; \
    } \
    Tensor* result = op_func(self->tensor, other->tensor); \
    if (result == NULL) { \
        PyErr_SetString(PyExc_RuntimeError, #name " operation failed"); \
        return NULL; \
    } \
    return wrap_tensor_result(result); \
}

#define UNARY_AUTOGRAD_OP(name, op_func) \
static PyObject* PyTensor_##name(PyTensorObject* self, PyObject* Py_UNUSED(ignored)) { \
    if (self->tensor == NULL) { \
        PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized"); \
        return NULL; \
    } \
    Tensor* result = op_func(self->tensor); \
    if (result == NULL) { \
        PyErr_SetString(PyExc_RuntimeError, #name " operation failed"); \
        return NULL; \
    } \
    return wrap_tensor_result(result); \
}

BINARY_AUTOGRAD_OP(op_add, op_add)
BINARY_AUTOGRAD_OP(op_mul, op_mul)
BINARY_AUTOGRAD_OP(op_sub, op_sub)
BINARY_AUTOGRAD_OP(op_div, op_div)
BINARY_AUTOGRAD_OP(op_power, op_power)
BINARY_AUTOGRAD_OP(op_logb, op_logb)
BINARY_AUTOGRAD_OP(op_matmul, op_matmul)

UNARY_AUTOGRAD_OP(op_exp, op_exp)
UNARY_AUTOGRAD_OP(op_log, op_log)
UNARY_AUTOGRAD_OP(op_sqrt, op_sqrt)
UNARY_AUTOGRAD_OP(op_tanh, op_tanh)
UNARY_AUTOGRAD_OP(op_relu, op_relu)
UNARY_AUTOGRAD_OP(op_sigmoid, op_sigmoid)
UNARY_AUTOGRAD_OP(op_square, op_square)
UNARY_AUTOGRAD_OP(op_abs, op_abs)
UNARY_AUTOGRAD_OP(op_sin, op_sin)
UNARY_AUTOGRAD_OP(op_cos, op_cos)
UNARY_AUTOGRAD_OP(op_tan, op_tan)
UNARY_AUTOGRAD_OP(op_asin, op_asin)
UNARY_AUTOGRAD_OP(op_acos, op_acos)
UNARY_AUTOGRAD_OP(op_atan, op_atan)
UNARY_AUTOGRAD_OP(op_sinh, op_sinh)
UNARY_AUTOGRAD_OP(op_cosh, op_cosh)
UNARY_AUTOGRAD_OP(op_gelu, op_gelu)
UNARY_AUTOGRAD_OP(op_silu, op_silu)
UNARY_AUTOGRAD_OP(op_rsqrt, op_rsqrt)
UNARY_AUTOGRAD_OP(op_sum_all, op_sum_all)
UNARY_AUTOGRAD_OP(op_mean_all, op_mean_all)

#define SCALAR_AUTOGRAD_OP(name, op_func) \
static PyObject* PyTensor_##name(PyTensorObject* self, PyObject* args) { \
    if (self->tensor == NULL) { \
        PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized"); \
        return NULL; \
    } \
    double scalar_val; \
    if (!PyArg_ParseTuple(args, "d", &scalar_val)) { \
        return NULL; \
    } \
    Tensor* a = self->tensor; \
    void* scalar_ptr; \
    float scalar_f32; \
    double scalar_f64; \
    if (a->dtype == DTYPE_FLOAT64) { \
        scalar_f64 = scalar_val; \
        scalar_ptr = &scalar_f64; \
    } else { \
        scalar_f32 = (float)scalar_val; \
        scalar_ptr = &scalar_f32; \
    } \
    Tensor* result = op_func(a, scalar_ptr); \
    if (result == NULL) { \
        PyErr_SetString(PyExc_RuntimeError, #name " operation failed"); \
        return NULL; \
    } \
    return wrap_tensor_result(result); \
}

SCALAR_AUTOGRAD_OP(op_add_scalar, op_add_scalar)
SCALAR_AUTOGRAD_OP(op_sub_scalar, op_sub_scalar)
SCALAR_AUTOGRAD_OP(op_mul_scalar, op_mul_scalar)
SCALAR_AUTOGRAD_OP(op_div_scalar, op_div_scalar)
SCALAR_AUTOGRAD_OP(op_pow_scalar, op_pow_scalar)
SCALAR_AUTOGRAD_OP(op_rsub_scalar, op_rsub_scalar)
SCALAR_AUTOGRAD_OP(op_rdiv_scalar, op_rdiv_scalar)
SCALAR_AUTOGRAD_OP(op_rpow_scalar, op_rpow_scalar)
SCALAR_AUTOGRAD_OP(op_logb_scalar, op_logb_scalar)
SCALAR_AUTOGRAD_OP(op_rlogb_scalar, op_rlogb_scalar)

static PyObject* PyTensor_op_leaky_relu(PyTensorObject* self, PyObject* args) {
    if (self->tensor == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized");
        return NULL;
    }

    float alpha_val = 0.01f;
    if (!PyArg_ParseTuple(args, "|f", &alpha_val)) {
        return NULL;
    }

    Tensor* result = op_leaky_relu(self->tensor, alpha_val);
    if (result == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "op_leaky_relu operation failed");
        return NULL;
    }

    return wrap_tensor_result(result);
}

static PyObject* PyTensor_repr(PyTensorObject* self) {
    if (self->tensor == NULL) {
        return PyUnicode_FromString("Tensor(uninitialized)");
    }

    char shape_str[256] = "[";
    for (int i = 0; i < self->tensor->ndim; i++) {
        char dim_str[32];
        snprintf(dim_str, sizeof(dim_str), "%d", self->tensor->shape[i]);
        strcat(shape_str, dim_str);
        if (i < self->tensor->ndim - 1) strcat(shape_str, ", ");
    }
    strcat(shape_str, "]");

    char device_str[32];
    if (self->tensor->device_id == -1) {
        snprintf(device_str, sizeof(device_str), "cpu");
    } else {
        snprintf(device_str, sizeof(device_str), "cuda:%d", self->tensor->device_id);
    }

    char repr[512];
    snprintf(repr, sizeof(repr), "Tensor(shape=%s, dtype=%s, device=%s)",
             shape_str, dtype_to_string(self->tensor->dtype), device_str);

    return PyUnicode_FromString(repr);
}

static PyGetSetDef PyTensor_getsetters[] = {
    {"shape", (getter)PyTensor_get_shape, NULL, "Tensor shape", NULL},
    {"dtype", (getter)PyTensor_get_dtype, NULL, "Data type", NULL},
    {"device", (getter)PyTensor_get_device, NULL, "Device (cpu or cuda:N)", NULL},
    {"device_id", (getter)PyTensor_get_device_id, NULL, "Device ID", NULL},
    {"size", (getter)PyTensor_get_size, NULL, "Total number of elements", NULL},
    {"ndim", (getter)PyTensor_get_ndim, NULL, "Number of dimensions", NULL},
    {"grad", (getter)PyTensor_get_grad, NULL, "Gradient tensor", NULL},
    {"has_grad", (getter)PyTensor_get_has_grad, NULL, "Whether gradient exists", NULL},
    {"requires_grad", (getter)PyTensor_get_requires_grad, NULL, "Whether tensor requires gradient", NULL},
    {"T", (getter)PyTensor_get_T, NULL, "Transpose (2D only)", NULL},
    {NULL}
};

static PyMethodDef PyTensor_methods[] = {
    {"to", (PyCFunction)PyTensor_to, METH_VARARGS | METH_KEYWORDS,
     "Convert tensor to different device/dtype"},
    {"copy", (PyCFunction)PyTensor_copy, METH_NOARGS,
     "Create a copy of the tensor"},
    {"data", (PyCFunction)PyTensor_data, METH_NOARGS,
     "Get tensor data as Python list"},

    {"add", (PyCFunction)PyTensor_add, METH_VARARGS,
     "Element-wise addition"},
    {"sub", (PyCFunction)PyTensor_sub, METH_VARARGS,
     "Element-wise subtraction"},
    {"mul", (PyCFunction)PyTensor_mul, METH_VARARGS,
     "Element-wise multiplication"},
    {"divide", (PyCFunction)PyTensor_divide, METH_VARARGS,
     "Element-wise division"},
    {"power", (PyCFunction)PyTensor_power, METH_VARARGS,
     "Element-wise power"},
    {"logb", (PyCFunction)PyTensor_logb, METH_VARARGS,
     "Element-wise logarithm with base"},

    {"add_scalar", (PyCFunction)PyTensor_add_scalar, METH_VARARGS,
     "Add scalar to all elements"},
    {"sub_scalar", (PyCFunction)PyTensor_sub_scalar, METH_VARARGS,
     "Subtract scalar from all elements"},
    {"mul_scalar", (PyCFunction)PyTensor_mul_scalar, METH_VARARGS,
     "Multiply all elements by scalar"},
    {"div_scalar", (PyCFunction)PyTensor_div_scalar, METH_VARARGS,
     "Divide all elements by scalar"},
    {"pow_scalar", (PyCFunction)PyTensor_pow_scalar, METH_VARARGS,
     "Raise all elements to scalar power"},
    {"logb_scalar", (PyCFunction)PyTensor_logb_scalar, METH_VARARGS,
     "Logarithm of all elements with scalar base"},
    {"rsub_scalar", (PyCFunction)PyTensor_rsub_scalar, METH_VARARGS,
     "Subtract all elements from scalar"},
    {"rdiv_scalar", (PyCFunction)PyTensor_rdiv_scalar, METH_VARARGS,
     "Divide scalar by all elements"},
    {"rpow_scalar", (PyCFunction)PyTensor_rpow_scalar, METH_VARARGS,
     "Raise scalar to power of all elements"},
    {"rlogb_scalar", (PyCFunction)PyTensor_rlogb_scalar, METH_VARARGS,
     "Logarithm of scalar with all elements as base"},

    {"exp", (PyCFunction)PyTensor_exp, METH_NOARGS,
     "Element-wise exponential"},
    {"log", (PyCFunction)PyTensor_log, METH_NOARGS,
     "Element-wise natural logarithm"},
    {"sqrt", (PyCFunction)PyTensor_sqrt, METH_NOARGS,
     "Element-wise square root"},
    {"abs", (PyCFunction)PyTensor_abs, METH_NOARGS,
     "Element-wise absolute value"},
    {"sin", (PyCFunction)PyTensor_sin, METH_NOARGS,
     "Element-wise sine"},
    {"cos", (PyCFunction)PyTensor_cos, METH_NOARGS,
     "Element-wise cosine"},
    {"tan", (PyCFunction)PyTensor_tan, METH_NOARGS,
     "Element-wise tangent"},
    {"asin", (PyCFunction)PyTensor_asin, METH_NOARGS,
     "Element-wise arc sine"},
    {"acos", (PyCFunction)PyTensor_acos, METH_NOARGS,
     "Element-wise arc cosine"},
    {"atan", (PyCFunction)PyTensor_atan, METH_NOARGS,
     "Element-wise arc tangent"},
    {"sinh", (PyCFunction)PyTensor_sinh, METH_NOARGS,
     "Element-wise hyperbolic sine"},
    {"cosh", (PyCFunction)PyTensor_cosh, METH_NOARGS,
     "Element-wise hyperbolic cosine"},
    {"tanh", (PyCFunction)PyTensor_tanh, METH_NOARGS,
     "Element-wise hyperbolic tangent"},

    {"relu", (PyCFunction)PyTensor_relu, METH_NOARGS,
     "ReLU activation: max(0, x)"},
    {"sigmoid", (PyCFunction)PyTensor_sigmoid, METH_NOARGS,
     "Sigmoid activation: 1/(1+exp(-x))"},
    {"gelu", (PyCFunction)PyTensor_gelu, METH_NOARGS,
     "GELU activation: Gaussian Error Linear Unit"},
    {"silu", (PyCFunction)PyTensor_silu, METH_NOARGS,
     "SiLU activation: x * sigmoid(x)"},
    {"leaky_relu", (PyCFunction)PyTensor_leaky_relu, METH_VARARGS,
     "Leaky ReLU: x if x>0 else alpha*x"},

    {"square", (PyCFunction)PyTensor_square, METH_NOARGS,
     "Element-wise square: x²"},
    {"ceil", (PyCFunction)PyTensor_ceil, METH_NOARGS,
     "Element-wise ceiling (round up)"},
    {"floor", (PyCFunction)PyTensor_floor, METH_NOARGS,
     "Element-wise floor (round down)"},
    {"round", (PyCFunction)PyTensor_round, METH_NOARGS,
     "Element-wise round to nearest"},
    {"trunc", (PyCFunction)PyTensor_trunc, METH_NOARGS,
     "Element-wise truncate (round toward zero)"},
    {"rsqrt", (PyCFunction)PyTensor_rsqrt, METH_NOARGS,
     "Element-wise reciprocal square root: 1/√x"},

    {"sum_all", (PyCFunction)PyTensor_sum_all, METH_NOARGS,
     "Sum all elements"},
    {"mean_all", (PyCFunction)PyTensor_mean_all, METH_NOARGS,
     "Mean of all elements"},
    {"sum_dim", (PyCFunction)PyTensor_sum_dim, METH_VARARGS | METH_KEYWORDS,
     "Sum along a dimension"},
    {"mean_dim", (PyCFunction)PyTensor_mean_dim, METH_VARARGS | METH_KEYWORDS,
     "Mean along a dimension"},

    {"requires_grad_", (PyCFunction)PyTensor_requires_grad_, METH_VARARGS,
     "Enable gradient tracking (in-place, returns self)"},
    {"backward", (PyCFunction)PyTensor_backward, METH_NOARGS,
     "Compute gradients via backpropagation"},
    {"zero_grad", (PyCFunction)PyTensor_zero_grad, METH_NOARGS,
     "Clear gradient (set to None)"},
    {"_sgd_step", (PyCFunction)PyTensor_sgd_step, METH_VARARGS | METH_KEYWORDS,
     "In-place SGD parameter update"},
    {"_adam_step", (PyCFunction)PyTensor_adam_step, METH_VARARGS | METH_KEYWORDS,
     "In-place Adam parameter update"},

    {"op_add", (PyCFunction)PyTensor_op_add, METH_VARARGS,
     "Addition with autograd support"},
    {"op_mul", (PyCFunction)PyTensor_op_mul, METH_VARARGS,
     "Multiplication with autograd support"},
    {"op_sub", (PyCFunction)PyTensor_op_sub, METH_VARARGS,
     "Subtraction with autograd support"},
    {"op_div", (PyCFunction)PyTensor_op_div, METH_VARARGS,
     "Division with autograd support"},
    {"op_power", (PyCFunction)PyTensor_op_power, METH_VARARGS,
     "Power with autograd support"},
    {"op_logb", (PyCFunction)PyTensor_op_logb, METH_VARARGS,
     "Logarithm with custom base with autograd support"},
    {"op_matmul", (PyCFunction)PyTensor_op_matmul, METH_VARARGS,
     "Matrix multiplication with autograd support"},

    {"op_exp", (PyCFunction)PyTensor_op_exp, METH_NOARGS,
     "Exponential with autograd support"},
    {"op_log", (PyCFunction)PyTensor_op_log, METH_NOARGS,
     "Natural logarithm with autograd support"},
    {"op_sqrt", (PyCFunction)PyTensor_op_sqrt, METH_NOARGS,
     "Square root with autograd support"},
    {"op_tanh", (PyCFunction)PyTensor_op_tanh, METH_NOARGS,
     "Hyperbolic tangent with autograd support"},
    {"op_relu", (PyCFunction)PyTensor_op_relu, METH_NOARGS,
     "ReLU activation with autograd support"},
    {"op_sigmoid", (PyCFunction)PyTensor_op_sigmoid, METH_NOARGS,
     "Sigmoid activation with autograd support"},
    {"op_square", (PyCFunction)PyTensor_op_square, METH_NOARGS,
     "Square operation with autograd support"},
    {"op_abs", (PyCFunction)PyTensor_op_abs, METH_NOARGS,
     "Absolute value with autograd support"},
    {"op_sin", (PyCFunction)PyTensor_op_sin, METH_NOARGS,
     "Sine with autograd support"},
    {"op_cos", (PyCFunction)PyTensor_op_cos, METH_NOARGS,
     "Cosine with autograd support"},
    {"op_tan", (PyCFunction)PyTensor_op_tan, METH_NOARGS,
     "Tangent with autograd support"},
    {"op_asin", (PyCFunction)PyTensor_op_asin, METH_NOARGS,
     "Arc sine with autograd support"},
    {"op_acos", (PyCFunction)PyTensor_op_acos, METH_NOARGS,
     "Arc cosine with autograd support"},
    {"op_atan", (PyCFunction)PyTensor_op_atan, METH_NOARGS,
     "Arc tangent with autograd support"},
    {"op_sinh", (PyCFunction)PyTensor_op_sinh, METH_NOARGS,
     "Hyperbolic sine with autograd support"},
    {"op_cosh", (PyCFunction)PyTensor_op_cosh, METH_NOARGS,
     "Hyperbolic cosine with autograd support"},
    {"op_gelu", (PyCFunction)PyTensor_op_gelu, METH_NOARGS,
     "GELU activation with autograd support"},
    {"op_silu", (PyCFunction)PyTensor_op_silu, METH_NOARGS,
     "SiLU (Swish) activation with autograd support"},
    {"op_rsqrt", (PyCFunction)PyTensor_op_rsqrt, METH_NOARGS,
     "Reciprocal square root with autograd support"},
    {"op_leaky_relu", (PyCFunction)PyTensor_op_leaky_relu, METH_VARARGS,
     "Leaky ReLU activation with autograd support (alpha=0.01 default)"},
    {"op_sum_all", (PyCFunction)PyTensor_op_sum_all, METH_NOARGS,
     "Sum all elements with autograd support (returns scalar tensor)"},
    {"op_mean_all", (PyCFunction)PyTensor_op_mean_all, METH_NOARGS,
     "Mean of all elements with autograd support (returns scalar tensor)"},
    {"op_sum_dim", (PyCFunction)PyTensor_op_sum_dim, METH_VARARGS | METH_KEYWORDS,
     "Sum along a dimension with autograd support"},
    {"op_mean_dim", (PyCFunction)PyTensor_op_mean_dim, METH_VARARGS | METH_KEYWORDS,
     "Mean along a dimension with autograd support"},
    {"op_conv2d", (PyCFunction)PyTensor_op_conv2d, METH_VARARGS | METH_KEYWORDS,
     "2D convolution with autograd (im2col + matmul)"},
    {"op_embedding", (PyCFunction)PyTensor_op_embedding, METH_VARARGS,
     "Embedding lookup with fused C/CUDA kernel"},
    {"op_layer_norm", (PyCFunction)PyTensor_op_layer_norm, METH_VARARGS | METH_KEYWORDS,
     "Layer normalization with fused C/CUDA kernel"},
    {"op_batch_norm", (PyCFunction)PyTensor_op_batch_norm, METH_VARARGS | METH_KEYWORDS,
     "Batch normalization with fused C/CUDA kernel"},
    {"op_dropout", (PyCFunction)PyTensor_op_dropout, METH_VARARGS,
     "Dropout with autograd (single CUDA kernel)"},
    {"op_maxpool2d", (PyCFunction)PyTensor_op_maxpool2d, METH_VARARGS | METH_KEYWORDS,
     "2D max pooling with autograd"},
    {"op_avgpool2d", (PyCFunction)PyTensor_op_avgpool2d, METH_VARARGS | METH_KEYWORDS,
     "2D average pooling with autograd"},
    {"op_mse_loss", (PyCFunction)PyTensor_op_mse_loss, METH_VARARGS | METH_KEYWORDS,
     "Mean squared error loss with autograd"},
    {"op_bce_loss", (PyCFunction)PyTensor_op_bce_loss, METH_VARARGS | METH_KEYWORDS,
     "Binary cross-entropy loss with autograd (auto-detects logits vs probs)"},
    {"op_nll_loss", (PyCFunction)PyTensor_op_nll_loss, METH_VARARGS | METH_KEYWORDS,
     "Negative log-likelihood loss with autograd"},
    {"op_cross_entropy_loss", (PyCFunction)PyTensor_op_cross_entropy_loss, METH_VARARGS | METH_KEYWORDS,
     "Cross-entropy loss with autograd (auto-detects logits vs probs)"},
    {"op_gather", (PyCFunction)PyTensor_op_gather, METH_VARARGS,
     "Gather elements along dim using index tensor"},
    {"op_softmax", (PyCFunction)PyTensor_op_softmax, METH_VARARGS | METH_KEYWORDS,
     "Softmax along a dimension with autograd support"},
    {"op_log_softmax", (PyCFunction)PyTensor_op_log_softmax, METH_VARARGS | METH_KEYWORDS,
     "Log-softmax along a dimension with autograd support"},
    {"op_add_scalar", (PyCFunction)PyTensor_op_add_scalar, METH_VARARGS,
     "Add scalar with autograd support"},
    {"op_sub_scalar", (PyCFunction)PyTensor_op_sub_scalar, METH_VARARGS,
     "Subtract scalar with autograd support"},
    {"op_mul_scalar", (PyCFunction)PyTensor_op_mul_scalar, METH_VARARGS,
     "Multiply by scalar with autograd support"},
    {"op_div_scalar", (PyCFunction)PyTensor_op_div_scalar, METH_VARARGS,
     "Divide by scalar with autograd support"},
    {"op_pow_scalar", (PyCFunction)PyTensor_op_pow_scalar, METH_VARARGS,
     "Power with scalar exponent with autograd support"},
    {"op_rsub_scalar", (PyCFunction)PyTensor_op_rsub_scalar, METH_VARARGS,
     "Reverse subtract (scalar - tensor) with autograd support"},
    {"op_rdiv_scalar", (PyCFunction)PyTensor_op_rdiv_scalar, METH_VARARGS,
     "Reverse divide (scalar / tensor) with autograd support"},
    {"op_rpow_scalar", (PyCFunction)PyTensor_op_rpow_scalar, METH_VARARGS,
     "Reverse power (scalar ^ tensor) with autograd support"},
    {"op_logb_scalar", (PyCFunction)PyTensor_op_logb_scalar, METH_VARARGS,
     "Log base scalar with autograd support"},
    {"op_rlogb_scalar", (PyCFunction)PyTensor_op_rlogb_scalar, METH_VARARGS,
     "Reverse log base (log_x(scalar)) with autograd support"},

    {"view", (PyCFunction)PyTensor_view, METH_VARARGS,
     "View tensor with new shape (requires contiguous tensor, zero-copy)"},
    {"reshape", (PyCFunction)PyTensor_reshape, METH_VARARGS,
     "Reshape tensor to new shape (copies if non-contiguous)"},
    {"transpose", (PyCFunction)PyTensor_transpose, METH_VARARGS,
     "Transpose tensor by swapping two dimensions"},
    {"squeeze", (PyCFunction)PyTensor_squeeze, METH_VARARGS,
     "Remove dimensions of size 1 (dim=-1 removes all, dim=N removes specific dimension)"},
    {"unsqueeze", (PyCFunction)PyTensor_unsqueeze, METH_VARARGS,
     "Add a dimension of size 1 at the specified position"},
    {"flatten", (PyCFunction)PyTensor_flatten, METH_VARARGS,
     "Flatten dimensions from start_dim to end_dim (defaults: start=0, end=-1)"},
    {"permute", (PyCFunction)PyTensor_permute, METH_VARARGS,
     "Reorder dimensions according to dims array"},
    {"chunk", (PyCFunction)PyTensor_chunk, METH_VARARGS,
     "Split tensor into chunks (last chunk may be smaller, dim defaults to 0)"},
    {"split", (PyCFunction)PyTensor_split, METH_VARARGS,
     "Split tensor with explicit sizes along dim (dim defaults to 0)"},
    {"split_equal", (PyCFunction)PyTensor_split_equal, METH_VARARGS,
     "Split tensor into equal parts along dim (dim defaults to 0)"},
    {"expand", (PyCFunction)PyTensor_expand, METH_VARARGS,
     "Broadcast tensor to larger shape (expand size-1 dimensions)"},
    {"repeat", (PyCFunction)PyTensor_repeat, METH_VARARGS,
     "Repeat tensor along dimensions (repeats must match ndim)"},
    {"slice", (PyCFunction)PyTensor_slice, METH_VARARGS | METH_KEYWORDS,
     "Slice tensor with start, stop, step"},

    {NULL}
};

static PyTypeObject PyTensorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "recpulse_cuda.Tensor",
    .tp_doc = "CUDA Tensor object",
    .tp_basicsize = sizeof(PyTensorObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_dealloc = (destructor)PyTensor_dealloc,
    .tp_repr = (reprfunc)PyTensor_repr,
    .tp_methods = PyTensor_methods,
    .tp_getset = PyTensor_getsetters,
};

static PyObject* module_zeros(PyObject* self, PyObject* args, PyObject* kwargs) {
    (void)self;

    PyObject* shape_obj;
    const char* dtype_str = "float32";
    const char* device_str = "cpu";

    static char* kwlist[] = {"shape", "dtype", "device", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|ss", kwlist,
                                     &shape_obj, &dtype_str, &device_str)) {
        return NULL;
    }

    if (!PySequence_Check(shape_obj)) {
        PyErr_SetString(PyExc_TypeError, "shape must be a sequence");
        return NULL;
    }

    Py_ssize_t ndim = PySequence_Size(shape_obj);
    if (ndim <= 0) {
        PyErr_SetString(PyExc_ValueError, "shape must have at least 1 dimension");
        return NULL;
    }

    int* shape = (int*)malloc(ndim * sizeof(int));
    if (shape == NULL) {
        PyErr_NoMemory();
        return NULL;
    }

    for (Py_ssize_t i = 0; i < ndim; i++) {
        PyObject* item = PySequence_GetItem(shape_obj, i);
        if (item == NULL) {
            free(shape);
            return NULL;
        }
        shape[i] = (int)PyLong_AsLong(item);
        Py_DECREF(item);
        if (PyErr_Occurred()) {
            free(shape);
            return NULL;
        }
        if (shape[i] <= 0) {
            free(shape);
            PyErr_SetString(PyExc_ValueError, "shape dimensions must be positive");
            return NULL;
        }
    }

    DType dtype = parse_dtype(dtype_str);
    if (dtype == DTYPE_PRESERVE) {
        free(shape);
        PyErr_SetString(PyExc_ValueError, "Invalid dtype");
        return NULL;
    }

    int device_id;
    if (!parse_device(device_str, &device_id)) {
        free(shape);
        return NULL;
    }

    Tensor* tensor = zeros_tensor(dtype, device_id, (int)ndim, shape, NULL);
    free(shape);

    if (tensor == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create tensor");
        return NULL;
    }

    PyTensorObject* py_tensor = (PyTensorObject*)PyTensorType.tp_alloc(&PyTensorType, 0);
    if (py_tensor == NULL) {
        free_tensor(tensor);
        return NULL;
    }

    py_tensor->tensor = tensor;
    return (PyObject*)py_tensor;
}

static PyObject* module_ones(PyObject* self, PyObject* args, PyObject* kwargs) {
    (void)self;

    PyObject* shape_obj;
    const char* dtype_str = "float32";
    const char* device_str = "cpu";

    static char* kwlist[] = {"shape", "dtype", "device", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|ss", kwlist,
                                     &shape_obj, &dtype_str, &device_str)) {
        return NULL;
    }

    if (!PySequence_Check(shape_obj)) {
        PyErr_SetString(PyExc_TypeError, "shape must be a sequence");
        return NULL;
    }

    Py_ssize_t ndim = PySequence_Size(shape_obj);
    if (ndim <= 0) {
        PyErr_SetString(PyExc_ValueError, "shape must have at least 1 dimension");
        return NULL;
    }

    int* shape = (int*)malloc(ndim * sizeof(int));
    if (shape == NULL) {
        PyErr_NoMemory();
        return NULL;
    }

    for (Py_ssize_t i = 0; i < ndim; i++) {
        PyObject* item = PySequence_GetItem(shape_obj, i);
        if (item == NULL) {
            free(shape);
            return NULL;
        }
        shape[i] = (int)PyLong_AsLong(item);
        Py_DECREF(item);
        if (PyErr_Occurred()) {
            free(shape);
            return NULL;
        }
        if (shape[i] <= 0) {
            free(shape);
            PyErr_SetString(PyExc_ValueError, "shape dimensions must be positive");
            return NULL;
        }
    }

    DType dtype = parse_dtype(dtype_str);
    if (dtype == DTYPE_PRESERVE) {
        free(shape);
        PyErr_SetString(PyExc_ValueError, "Invalid dtype");
        return NULL;
    }

    int device_id;
    if (!parse_device(device_str, &device_id)) {
        free(shape);
        return NULL;
    }

    Tensor* tensor = ones_tensor(dtype, device_id, (int)ndim, shape, NULL);
    free(shape);

    if (tensor == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create tensor");
        return NULL;
    }

    PyTensorObject* py_tensor = (PyTensorObject*)PyTensorType.tp_alloc(&PyTensorType, 0);
    if (py_tensor == NULL) {
        free_tensor(tensor);
        return NULL;
    }

    py_tensor->tensor = tensor;
    return (PyObject*)py_tensor;
}

static PyObject* module_values(PyObject* self, PyObject* args, PyObject* kwargs) {
    (void)self;

    PyObject* data_obj;
    const char* dtype_str = "float32";
    const char* device_str = "cpu";

    static char* kwlist[] = {"data", "dtype", "device", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|ss", kwlist,
                                     &data_obj, &dtype_str, &device_str)) {
        return NULL;
    }

    if (!PySequence_Check(data_obj)) {
        PyErr_SetString(PyExc_TypeError, "data must be a sequence");
        return NULL;
    }

    Py_ssize_t size = PySequence_Size(data_obj);
    if (size <= 0) {
        PyErr_SetString(PyExc_ValueError, "data must have at least 1 element");
        return NULL;
    }

    DType dtype = parse_dtype(dtype_str);
    if (dtype == DTYPE_PRESERVE) {
        PyErr_SetString(PyExc_ValueError, "Invalid dtype");
        return NULL;
    }

    int device_id;
    if (!parse_device(device_str, &device_id)) {
        return NULL;
    }

    void* data_buffer = NULL;
    if (dtype == DTYPE_FLOAT32) {
        float* data_f32 = (float*)malloc(size * sizeof(float));
        if (data_f32 == NULL) {
            PyErr_NoMemory();
            return NULL;
        }
        for (Py_ssize_t i = 0; i < size; i++) {
            PyObject* item = PySequence_GetItem(data_obj, i);
            if (item == NULL) {
                free(data_f32);
                return NULL;
            }
            data_f32[i] = (float)PyFloat_AsDouble(item);
            Py_DECREF(item);
            if (PyErr_Occurred()) {
                free(data_f32);
                return NULL;
            }
        }
        data_buffer = data_f32;
    } else if (dtype == DTYPE_FLOAT64) {
        double* data_f64 = (double*)malloc(size * sizeof(double));
        if (data_f64 == NULL) {
            PyErr_NoMemory();
            return NULL;
        }
        for (Py_ssize_t i = 0; i < size; i++) {
            PyObject* item = PySequence_GetItem(data_obj, i);
            if (item == NULL) {
                free(data_f64);
                return NULL;
            }
            data_f64[i] = PyFloat_AsDouble(item);
            Py_DECREF(item);
            if (PyErr_Occurred()) {
                free(data_f64);
                return NULL;
            }
        }
        data_buffer = data_f64;
    } else if (dtype == DTYPE_FLOAT16 || dtype == DTYPE_BFLOAT16) {
        float* data_f32 = (float*)malloc(size * sizeof(float));
        if (data_f32 == NULL) {
            PyErr_NoMemory();
            return NULL;
        }
        for (Py_ssize_t i = 0; i < size; i++) {
            PyObject* item = PySequence_GetItem(data_obj, i);
            if (item == NULL) {
                free(data_f32);
                return NULL;
            }
            data_f32[i] = (float)PyFloat_AsDouble(item);
            Py_DECREF(item);
            if (PyErr_Occurred()) {
                free(data_f32);
                return NULL;
            }
        }
        data_buffer = data_f32;
    }

    DType src_dtype = (dtype == DTYPE_FLOAT16 || dtype == DTYPE_BFLOAT16) ? DTYPE_FLOAT32 : dtype;
    int shape[1] = {(int)size};
    Tensor* tensor = values_tensor(data_buffer, src_dtype, dtype, -1, device_id, 1, shape, NULL);
    free(data_buffer);

    if (tensor == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create tensor");
        return NULL;
    }

    PyTensorObject* py_tensor = (PyTensorObject*)PyTensorType.tp_alloc(&PyTensorType, 0);
    if (py_tensor == NULL) {
        free_tensor(tensor);
        return NULL;
    }

    py_tensor->tensor = tensor;
    return (PyObject*)py_tensor;
}

static PyObject* module_cat(PyObject* self, PyObject* args, PyObject* kwargs) {
    (void)self;

    PyObject* tensors_obj;
    int dim = 0;

    static char* kwlist[] = {"tensors", "dim", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|i", kwlist,
                                     &tensors_obj, &dim)) {
        return NULL;
    }

    if (!PySequence_Check(tensors_obj)) {
        PyErr_SetString(PyExc_TypeError, "tensors must be a sequence");
        return NULL;
    }

    Py_ssize_t num_tensors = PySequence_Size(tensors_obj);
    if (num_tensors <= 0) {
        PyErr_SetString(PyExc_ValueError, "tensors must have at least 1 element");
        return NULL;
    }

    Tensor** tensors = (Tensor**)malloc(num_tensors * sizeof(Tensor*));
    if (!tensors) {
        PyErr_NoMemory();
        return NULL;
    }

    for (Py_ssize_t i = 0; i < num_tensors; i++) {
        PyObject* item = PySequence_GetItem(tensors_obj, i);
        if (!item) {
            free(tensors);
            return NULL;
        }

        if (!PyObject_TypeCheck(item, &PyTensorType)) {
            Py_DECREF(item);
            free(tensors);
            PyErr_SetString(PyExc_TypeError, "All elements must be Tensors");
            return NULL;
        }

        PyTensorObject* py_tensor = (PyTensorObject*)item;
        if (!py_tensor->tensor) {
            Py_DECREF(item);
            free(tensors);
            PyErr_SetString(PyExc_RuntimeError, "Tensor is not initialized");
            return NULL;
        }

        tensors[i] = py_tensor->tensor;
        Py_DECREF(item);
    }

    Tensor* result = op_cat(tensors, (int)num_tensors, dim);
    free(tensors);

    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "cat operation failed");
        return NULL;
    }

    return wrap_tensor_result(result);
}

static PyObject* module_rand(PyObject* self, PyObject* args, PyObject* kwargs) {
    (void)self;
    PyObject* shape_obj;
    const char* dtype_str = "float32";
    const char* device_str = "cpu";

    static char* kwlist[] = {"shape", "dtype", "device", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|ss", kwlist,
                                     &shape_obj, &dtype_str, &device_str))
        return NULL;

    if (!PySequence_Check(shape_obj)) {
        PyErr_SetString(PyExc_TypeError, "shape must be a sequence");
        return NULL;
    }

    Py_ssize_t ndim = PySequence_Size(shape_obj);
    if (ndim <= 0) {
        PyErr_SetString(PyExc_ValueError, "shape must have at least 1 dimension");
        return NULL;
    }

    int* shape = (int*)malloc(ndim * sizeof(int));
    if (!shape) return PyErr_NoMemory();

    for (Py_ssize_t i = 0; i < ndim; i++) {
        PyObject* item = PySequence_GetItem(shape_obj, i);
        if (!item) { free(shape); return NULL; }
        shape[i] = (int)PyLong_AsLong(item);
        Py_DECREF(item);
        if (PyErr_Occurred()) { free(shape); return NULL; }
        if (shape[i] <= 0) { free(shape); PyErr_SetString(PyExc_ValueError, "shape dimensions must be positive"); return NULL; }
    }

    DType dtype = parse_dtype(dtype_str);
    if (dtype == DTYPE_PRESERVE) { free(shape); PyErr_SetString(PyExc_ValueError, "Invalid dtype"); return NULL; }

    int device_id;
    if (!parse_device(device_str, &device_id)) { free(shape); return NULL; }

    Tensor* tensor = rp_rand_tensor(dtype, device_id, (int)ndim, shape);
    free(shape);
    if (!tensor) { PyErr_SetString(PyExc_RuntimeError, "rand failed"); return NULL; }
    return wrap_tensor_result(tensor);
}

static PyObject* module_randn(PyObject* self, PyObject* args, PyObject* kwargs) {
    (void)self;
    PyObject* shape_obj;
    const char* dtype_str = "float32";
    const char* device_str = "cpu";
    const char* method_str = NULL;

    static char* kwlist[] = {"shape", "dtype", "device", "method", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|sss", kwlist,
                                     &shape_obj, &dtype_str, &device_str, &method_str))
        return NULL;

    NormalMethod method = NORMAL_BOX_MULLER;
    if (method_str) {
        if (strcmp(method_str, "box_muller") == 0) method = NORMAL_BOX_MULLER;
        else if (strcmp(method_str, "ziggurat") == 0) method = NORMAL_ZIGGURAT;
        else if (strcmp(method_str, "inverse_cdf") == 0) method = NORMAL_INVERSE_CDF;
        else {
            PyErr_Format(PyExc_ValueError,
                "Unknown method '%s'. Use 'box_muller', 'ziggurat', or 'inverse_cdf'", method_str);
            return NULL;
        }
    }

    if (!PySequence_Check(shape_obj)) {
        PyErr_SetString(PyExc_TypeError, "shape must be a sequence");
        return NULL;
    }

    Py_ssize_t ndim = PySequence_Size(shape_obj);
    if (ndim <= 0) {
        PyErr_SetString(PyExc_ValueError, "shape must have at least 1 dimension");
        return NULL;
    }

    int* shape = (int*)malloc(ndim * sizeof(int));
    if (!shape) return PyErr_NoMemory();

    for (Py_ssize_t i = 0; i < ndim; i++) {
        PyObject* item = PySequence_GetItem(shape_obj, i);
        if (!item) { free(shape); return NULL; }
        shape[i] = (int)PyLong_AsLong(item);
        Py_DECREF(item);
        if (PyErr_Occurred()) { free(shape); return NULL; }
        if (shape[i] <= 0) { free(shape); PyErr_SetString(PyExc_ValueError, "shape dimensions must be positive"); return NULL; }
    }

    DType dtype = parse_dtype(dtype_str);
    if (dtype == DTYPE_PRESERVE) { free(shape); PyErr_SetString(PyExc_ValueError, "Invalid dtype"); return NULL; }

    int device_id;
    if (!parse_device(device_str, &device_id)) { free(shape); return NULL; }

    Tensor* tensor = rp_randn_tensor(dtype, device_id, (int)ndim, shape, method);
    free(shape);
    if (!tensor) { PyErr_SetString(PyExc_RuntimeError, "randn failed"); return NULL; }
    return wrap_tensor_result(tensor);
}

static PyObject* module_randint(PyObject* self, PyObject* args, PyObject* kwargs) {
    (void)self;
    double low, high;
    PyObject* shape_obj;
    const char* dtype_str = "float32";
    const char* device_str = "cpu";

    static char* kwlist[] = {"low", "high", "shape", "dtype", "device", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ddO|ss", kwlist,
                                     &low, &high, &shape_obj, &dtype_str, &device_str))
        return NULL;

    if (!PySequence_Check(shape_obj)) {
        PyErr_SetString(PyExc_TypeError, "shape must be a sequence");
        return NULL;
    }

    Py_ssize_t ndim = PySequence_Size(shape_obj);
    if (ndim <= 0) {
        PyErr_SetString(PyExc_ValueError, "shape must have at least 1 dimension");
        return NULL;
    }

    int* shape = (int*)malloc(ndim * sizeof(int));
    if (!shape) return PyErr_NoMemory();

    for (Py_ssize_t i = 0; i < ndim; i++) {
        PyObject* item = PySequence_GetItem(shape_obj, i);
        if (!item) { free(shape); return NULL; }
        shape[i] = (int)PyLong_AsLong(item);
        Py_DECREF(item);
        if (PyErr_Occurred()) { free(shape); return NULL; }
        if (shape[i] <= 0) { free(shape); PyErr_SetString(PyExc_ValueError, "shape dimensions must be positive"); return NULL; }
    }

    DType dtype = parse_dtype(dtype_str);
    if (dtype == DTYPE_PRESERVE) { free(shape); PyErr_SetString(PyExc_ValueError, "Invalid dtype"); return NULL; }

    int device_id;
    if (!parse_device(device_str, &device_id)) { free(shape); return NULL; }

    Tensor* tensor = rp_randint_tensor(low, high, dtype, device_id, (int)ndim, shape);
    free(shape);
    if (!tensor) { PyErr_SetString(PyExc_RuntimeError, "randint failed"); return NULL; }
    return wrap_tensor_result(tensor);
}

static PyObject* module_manual_seed(PyObject* self, PyObject* args) {
    (void)self;
    unsigned long long seed;
    if (!PyArg_ParseTuple(args, "K", &seed))
        return NULL;
    rp_manual_seed((uint64_t)seed);
    Py_RETURN_NONE;
}

static PyMethodDef module_methods[] = {
    {"zeros", (PyCFunction)module_zeros, METH_VARARGS | METH_KEYWORDS,
     "Create a tensor filled with zeros"},
    {"ones", (PyCFunction)module_ones, METH_VARARGS | METH_KEYWORDS,
     "Create a tensor filled with ones"},
    {"values", (PyCFunction)module_values, METH_VARARGS | METH_KEYWORDS,
     "Create a tensor from a list of values"},
    {"cat", (PyCFunction)module_cat, METH_VARARGS | METH_KEYWORDS,
     "Concatenate tensors along a dimension"},
    {"rand", (PyCFunction)module_rand, METH_VARARGS | METH_KEYWORDS,
     "Create tensor with uniform random values in [0, 1)"},
    {"randn", (PyCFunction)module_randn, METH_VARARGS | METH_KEYWORDS,
     "Create tensor with standard normal random values"},
    {"randint", (PyCFunction)module_randint, METH_VARARGS | METH_KEYWORDS,
     "Create tensor with random integers in [low, high)"},
    {"manual_seed", (PyCFunction)module_manual_seed, METH_VARARGS,
     "Set the random seed for reproducibility"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef recpulse_cuda_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "recpulse_cuda",
    .m_doc = "RecPulse CUDA tensor module",
    .m_size = -1,
    .m_methods = module_methods,
};

PyMODINIT_FUNC PyInit_recpulse_cuda(void) {
    if (PyType_Ready(&PyTensorType) < 0)
        return NULL;

    PyObject* m = PyModule_Create(&recpulse_cuda_module);
    if (m == NULL)
        return NULL;

    Py_INCREF(&PyTensorType);
    if (PyModule_AddObject(m, "Tensor", (PyObject*)&PyTensorType) < 0) {
        Py_DECREF(&PyTensorType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
