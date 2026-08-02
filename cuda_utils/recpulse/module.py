import recpulse_cuda as rp


class Module:
    def __init__(self):
        self.tracked = {}
        self._modules = {}
        self._training = True

    def track(self, name, tensor):
        self.tracked[name] = tensor
        return tensor

    def __setattr__(self, name, value):
        if name.startswith('_') or name == 'tracked':
            super().__setattr__(name, value)
            return

        if isinstance(value, Module):
            self._modules[name] = value
            for k, v in value.tracked.items():
                self.tracked[f"{name}.{k}"] = v
            for sub_name, sub_mod in value._modules.items():
                self._modules[f"{name}.{sub_name}"] = sub_mod

        super().__setattr__(name, value)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def keep(self, tensor):
        return tensor

    def learnable(self):
        return {k: v for k, v in self.tracked.items() if v.requires_grad}

    def frozen(self):
        return {k: v for k, v in self.tracked.items() if not v.requires_grad}

    def parameters(self):
        return list(self.learnable().values())

    def to(self, device=None, dtype=None):
        kwargs = {}
        if device is not None:
            kwargs['device'] = device
        if dtype is not None:
            kwargs['dtype'] = dtype
        if not kwargs:
            return self

        for tensor in self.tracked.values():
            rg = tensor.requires_grad
            tensor.to(inplace=True, **kwargs)
            if rg:
                tensor.requires_grad_(True)
        return self

    def load_state(self, state_dict):
        for name, tensor in state_dict.items():
            if name not in self.tracked:
                continue

            old = self.tracked[name]
            rg = old.requires_grad
            old.copy_(tensor)
            if rg:
                old.requires_grad_(True)

    def train(self):
        self._training = True
        for m in self._modules.values():
            m.train()
        return self

    def eval(self):
        self._training = False
        for m in self._modules.values():
            m.eval()
        return self

    def zero_grad(self):
        for t in self.tracked.values():
            if t.requires_grad:
                t.zero_grad()


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        scale = (1.0 / in_features) ** 0.5
        w_raw = rp.randn([in_features, out_features])
        self.weight = w_raw.mul_scalar(scale)
        self.weight.requires_grad_(True)
        self.track("weight", self.weight)

        if bias:
            self.bias = rp.zeros([out_features])
            self.bias.requires_grad_(True)
            self.track("bias", self.bias)
        else:
            self.bias = None

    def forward(self, x):
        out = self.keep(x.op_matmul(self.weight))
        if self.bias is not None:
            out = self.keep(out.op_add(self.bias))
        return out


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int): stride = (stride, stride)
        if isinstance(padding, int): padding = (padding, padding)
        if isinstance(dilation, int): dilation = (dilation, dilation)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        kH, kW = kernel_size
        fan_in = in_channels * kH * kW
        scale = (1.0 / fan_in) ** 0.5
        w_raw = rp.randn([out_channels * in_channels * kH * kW])
        w_scaled = w_raw.mul_scalar(scale)
        self._w_base = w_scaled
        self.weight = w_scaled.reshape([out_channels, in_channels, kH, kW])
        self.weight.requires_grad_(True)
        self.track("weight", self.weight)

        if bias:
            self.bias = rp.zeros([out_channels])
            self.bias.requires_grad_(True)
            self.track("bias", self.bias)
        else:
            self.bias = None

    def forward(self, x):
        return x.op_conv2d(
            self.weight, bias=self.bias,
            stride_h=self.stride[0], stride_w=self.stride[1],
            pad_h=self.padding[0], pad_w=self.padding[1],
            dilation_h=self.dilation[0], dilation_w=self.dilation[1]
        )


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size)
        if stride is None: stride = kernel_size
        if isinstance(stride, int): stride = (stride, stride)
        if isinstance(padding, int): padding = (padding, padding)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return x.op_maxpool2d(
            self.kernel_size[0], self.kernel_size[1],
            stride_h=self.stride[0], stride_w=self.stride[1],
            pad_h=self.padding[0], pad_w=self.padding[1]
        )


class AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size)
        if stride is None: stride = kernel_size
        if isinstance(stride, int): stride = (stride, stride)
        if isinstance(padding, int): padding = (padding, padding)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return x.op_avgpool2d(
            self.kernel_size[0], self.kernel_size[1],
            stride_h=self.stride[0], stride_w=self.stride[1],
            pad_h=self.padding[0], pad_w=self.padding[1]
        )


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = rp.randn([num_embeddings, embedding_dim]).mul_scalar(0.01)
        self.weight.requires_grad_(True)
        self.track("weight", self.weight)

    def forward(self, indices):
        if isinstance(indices, (list, tuple)):
            return self.weight.op_embedding(indices)
        else:
            return self.weight.op_embedding(indices)


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self._training or self.p == 0.0:
            return x
        return x.op_dropout(self.p)


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = [normalized_shape]
        self.normalized_shape = list(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            total = 1
            for s in normalized_shape:
                total *= s
            self.weight = rp.ones([total])
            self.weight.requires_grad_(True)
            self.track("weight", self.weight)
            self.bias = rp.zeros([total])
            self.bias.requires_grad_(True)
            self.track("bias", self.bias)
        else:
            self.weight = None
            self.bias = None

    def forward(self, x):
        return x.op_layer_norm(
            self.normalized_shape,
            weight=self.weight,
            bias=self.bias,
            eps=self.eps
        )


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        scale = (1.0 / hidden_size) ** 0.5
        self.weight_ih = rp.randn([input_size, 4 * hidden_size]).mul_scalar(scale)
        self.weight_ih.requires_grad_(True)
        self.track("weight_ih", self.weight_ih)
        self.weight_hh = rp.randn([hidden_size, 4 * hidden_size]).mul_scalar(scale)
        self.weight_hh.requires_grad_(True)
        self.track("weight_hh", self.weight_hh)

        if bias:
            self.bias_ih = rp.zeros([4 * hidden_size])
            self.bias_ih.requires_grad_(True)
            self.track("bias_ih", self.bias_ih)
            self.bias_hh = rp.zeros([4 * hidden_size])
            self.bias_hh.requires_grad_(True)
            self.track("bias_hh", self.bias_hh)
        else:
            self.bias_ih = None
            self.bias_hh = None

    def forward(self, x, state=None):
        if state is None:
            batch = x.shape[0]
            h = rp.zeros([batch, self.hidden_size], device=str(x.device))
            c = rp.zeros([batch, self.hidden_size], device=str(x.device))
        else:
            h, c = state

        gates = x.op_matmul(self.weight_ih).op_add(h.op_matmul(self.weight_hh))
        if self.bias_ih is not None:
            gates = gates.op_add(self.bias_ih).op_add(self.bias_hh)

        i_g, f_g, g_g, o_g = gates.chunk(4, 1)
        i_g = i_g.op_sigmoid()
        f_g = f_g.op_sigmoid()
        g_g = g_g.op_tanh()
        o_g = o_g.op_sigmoid()

        c_next = f_g.op_mul(c).op_add(i_g.op_mul(g_g))
        h_next = o_g.op_mul(c_next.op_tanh())
        return h_next, c_next


class GRUCell(Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        scale = (1.0 / hidden_size) ** 0.5
        self.weight_ih = rp.randn([input_size, 3 * hidden_size]).mul_scalar(scale)
        self.weight_ih.requires_grad_(True)
        self.track("weight_ih", self.weight_ih)
        self.weight_hh = rp.randn([hidden_size, 3 * hidden_size]).mul_scalar(scale)
        self.weight_hh.requires_grad_(True)
        self.track("weight_hh", self.weight_hh)

        if bias:
            self.bias_ih = rp.zeros([3 * hidden_size])
            self.bias_ih.requires_grad_(True)
            self.track("bias_ih", self.bias_ih)
            self.bias_hh = rp.zeros([3 * hidden_size])
            self.bias_hh.requires_grad_(True)
            self.track("bias_hh", self.bias_hh)
        else:
            self.bias_ih = None
            self.bias_hh = None

    def forward(self, x, h=None):
        if h is None:
            batch = x.shape[0]
            h = rp.zeros([batch, self.hidden_size], device=str(x.device))

        gi = x.op_matmul(self.weight_ih)
        gh = h.op_matmul(self.weight_hh)
        if self.bias_ih is not None:
            gi = gi.op_add(self.bias_ih)
            gh = gh.op_add(self.bias_hh)

        i_r, i_z, i_n = gi.chunk(3, 1)
        h_r, h_z, h_n = gh.chunk(3, 1)

        r = i_r.op_add(h_r).op_sigmoid()
        z = i_z.op_add(h_z).op_sigmoid()
        n = i_n.op_add(r.op_mul(h_n)).op_tanh()

        h_next = z.op_rsub_scalar(1.0).op_mul(n).op_add(z.op_mul(h))
        return h_next


class BatchNorm2d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        if affine:
            self.weight = rp.ones([num_features])
            self.weight.requires_grad_(True)
            self.track("weight", self.weight)
            self.bias = rp.zeros([num_features])
            self.bias.requires_grad_(True)
            self.track("bias", self.bias)
        else:
            self.weight = None
            self.bias = None

        self.running_mean = rp.zeros([num_features])
        self.track("running_mean", self.running_mean)
        self.running_var = rp.ones([num_features])
        self.track("running_var", self.running_var)

    def forward(self, x):
        if not self._training:
            return x.op_batch_norm(
                weight=self.weight,
                bias=self.bias,
                running_mean=self.running_mean,
                running_var=self.running_var,
                eps=self.eps,
                momentum=self.momentum,
                training=False
            )

        C = x.shape[1]
        spatial = 1
        for i in range(2, x.ndim):
            spatial *= x.shape[i]
        N = x.shape[0]
        count = N * spatial

        x_transposed = self.keep(x.permute([1, 0] + list(range(2, x.ndim))))
        x_flat = self.keep(x_transposed.reshape([C, count]))

        mean = self.keep(x_flat.op_mean_dim(1, keepdim=True))
        diff = self.keep(x_flat.op_sub(mean))
        sq = self.keep(diff.op_square())
        var = self.keep(sq.op_mean_dim(1, keepdim=True))

        mean_1d = mean.reshape([C])
        var_1d = var.reshape([C])
        self.keep(mean_1d)
        self.keep(var_1d)

        alpha = self.momentum
        new_rm = self.running_mean.mul_scalar(1.0 - alpha).add(mean_1d.copy().mul_scalar(alpha))
        new_rv = self.running_var.mul_scalar(1.0 - alpha).add(var_1d.copy().mul_scalar(alpha))
        self.running_mean.copy_(new_rm)
        self.running_var.copy_(new_rv)

        var_eps = self.keep(var.op_add_scalar(self.eps))
        std = self.keep(var_eps.op_sqrt())
        normed_flat = self.keep(diff.op_div(std))

        out_shape = [C, N] + [x.shape[i] for i in range(2, x.ndim)]
        normed_t = self.keep(normed_flat.reshape(out_shape))
        normed = self.keep(normed_t.permute([1, 0] + list(range(2, x.ndim))))

        if self.affine:
            reshape_dims = [1, C] + [1] * (x.ndim - 2)
            w = self.keep(self.weight.reshape(reshape_dims))
            b = self.keep(self.bias.reshape(reshape_dims))
            normed = self.keep(normed.op_mul(w))
            normed = self.keep(normed.op_add(b))

        return normed
