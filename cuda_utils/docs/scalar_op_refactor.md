# Scalar Op Refactor Notes

## Problem

The scalar autograd ops (`op_add_scalar`, `op_sub_scalar`, `op_mul_scalar`, `op_div_scalar`, `op_pow_scalar`) share nearly identical forward op boilerplate in `ops.c`. The backward functions also have significant overlap.

## Current Duplication

### Forward ops (~50 lines each, ~250 total)

Every scalar forward op follows this pattern:
1. Validate `x` and `scalar`
2. `zeros_tensor(...)` for output
3. Call `rp_<op>_scalar(out->data, x->data, scalar, ...)`
4. If `requires_grad`: allocate Meta, GradFn, inputs array
5. Optionally save scalar into `ScalarSavedData`
6. Set `grad_fn->backward`

The ONLY differences between them:
- Which `rp_*` function is called (line 3)
- Which backward function is set (line 6)
- Whether `ScalarSavedData` is needed (add/sub: no, mul/div/pow: yes)

### Backward functions

| Backward | Logic | Shared? |
|----------|-------|---------|
| `backward_add_scalar_fn` | `grad_x = grad_output` (copy) | Identical to sub |
| `backward_sub_scalar_fn` | `grad_x = grad_output` (copy) | Identical to add |
| `backward_mul_scalar_fn` | `grad_x = grad_output * c` | Similar to div pattern |
| `backward_div_scalar_fn` | `grad_x = grad_output / c` | Similar to mul pattern |
| `backward_pow_scalar_fn` | `grad_x = grad_output * c * x^(c-1)` | Unique |

## Proposed Refactor

### 1. Forward op macros (two variants)

```c
// For ops where backward doesn't need the scalar (add, sub)
#define SCALAR_OP_NO_SAVE(name, rp_func, bwd_func) \
Tensor* op_##name(Tensor* x, const void* scalar) { \
    if (!x || !scalar) return NULL; \
    Tensor* out = zeros_tensor(x->dtype, x->device_id, x->ndim, x->shape, NULL); \
    if (!out) return NULL; \
    if (rp_func(out->data, x->data, scalar, x->size, x->dtype, x->device_id) != 0) { \
        free_tensor(out); return NULL; \
    } \
    if (x->metadata && x->metadata->requires_grad) { \
        /* ... standard GradFn setup, saved_data = NULL ... */ \
    } \
    return out; \
}

// For ops where backward needs the scalar (mul, div, pow)
#define SCALAR_OP_WITH_SAVE(name, rp_func, bwd_func) \
Tensor* op_##name(Tensor* x, const void* scalar) { \
    /* ... same as above but also allocates ScalarSavedData ... */ \
}
```

Usage:
```c
SCALAR_OP_NO_SAVE(add_scalar, rp_add_scalar, backward_add_scalar_fn)
SCALAR_OP_NO_SAVE(sub_scalar, rp_sub_scalar, backward_add_scalar_fn)  // share backward!
SCALAR_OP_WITH_SAVE(mul_scalar, rp_mul_scalar, backward_mul_scalar_fn)
SCALAR_OP_WITH_SAVE(div_scalar, rp_div_scalar, backward_div_scalar_fn)
SCALAR_OP_WITH_SAVE(pow_scalar, rp_pow_scalar, backward_pow_scalar_fn)
```

### 2. Merge identical backward functions

- Delete `backward_sub_scalar_fn`, point sub_scalar at `backward_add_scalar_fn`
- Both add and sub have `grad_x = grad_output` (identity)

### 3. Consider a generic scalar backward for mul/div

Both mul and div backward follow the same structure — only the `rp_*` call differs:
```c
// mul: rp_mul_scalar(grad_x, grad_output, &saved->scalar, ...)
// div: rp_div_scalar(grad_x, grad_output, &saved->scalar, ...)
```

Could store a function pointer in saved_data, but that's probably over-engineering.

### 4. Reverse scalar ops (rsub, rdiv, rpow, etc.)

When these get implemented, they'll follow the same macro pattern. The `SCALAR_OP_WITH_SAVE` macro will handle them too — only the `rp_*` forward function and backward function differ.

## Estimated Impact

- Before: ~250 lines of forward op boilerplate + ~100 lines of backward
- After: ~40 lines of macro definitions + ~5 instantiation lines + ~60 lines of backward functions
- Net reduction: ~250 lines

## When to Do

After all scalar ops (including reverse variants) are implemented. The refactor will be cleaner when all variants exist, since we'll see the full pattern.
