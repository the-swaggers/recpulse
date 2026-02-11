import recpulse_cuda as rp


class Optimizer:
    def __init__(self, params, defaults):
        self.params = list(params)
        self.defaults = defaults
        self.state = {}

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, dampening=0.0,
                 weight_decay=0.0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        super().__init__(params, defaults)

    def step(self):
        lr = self.defaults['lr']
        momentum = self.defaults['momentum']
        dampening = self.defaults['dampening']
        weight_decay = self.defaults['weight_decay']
        nesterov = self.defaults['nesterov']

        for p in self.params:
            if not p.has_grad:
                continue

            pid = id(p)
            if pid not in self.state:
                self.state[pid] = {'has_momentum_buf': False}

            st = self.state[pid]
            buf = st.get('momentum_buf', None)

            if momentum != 0.0 and buf is None:
                buf = rp.zeros(list(p.shape), dtype=p.dtype, device=p.device)
                st['momentum_buf'] = buf

            p._sgd_step(lr=lr, momentum=momentum, dampening=dampening,
                        nesterov=nesterov, weight_decay=weight_decay,
                        momentum_buf=buf, has_momentum_buf=st['has_momentum_buf'])

            if momentum != 0.0:
                st['has_momentum_buf'] = True


class Adam(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, amsgrad=False):
        defaults = dict(lr=lr, beta1=betas[0], beta2=betas[1], eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)

    def step(self):
        lr = self.defaults['lr']
        beta1 = self.defaults['beta1']
        beta2 = self.defaults['beta2']
        eps = self.defaults['eps']
        weight_decay = self.defaults['weight_decay']
        amsgrad = self.defaults['amsgrad']

        for p in self.params:
            if not p.has_grad:
                continue

            pid = id(p)
            if pid not in self.state:
                shape = list(p.shape)
                self.state[pid] = {
                    'step': 0,
                    'm': rp.zeros(shape, dtype=p.dtype, device=p.device),
                    'v': rp.zeros(shape, dtype=p.dtype, device=p.device),
                }
                if amsgrad:
                    self.state[pid]['v_max'] = rp.zeros(shape, dtype=p.dtype, device=p.device)

            st = self.state[pid]
            st['step'] += 1

            v_max = st.get('v_max', None)
            p._adam_step(lr=lr, beta1=beta1, beta2=beta2, eps=eps,
                         weight_decay=weight_decay, amsgrad=amsgrad,
                         m=st['m'], v=st['v'], v_max=v_max, step=st['step'])
