class LRScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.base_lr = optimizer.defaults['lr']
        self.current_lr = self.base_lr
        self._step_count = 0

    def get_lr(self):
        return self.current_lr

    def step(self, *args, **kwargs):
        raise NotImplementedError

    def _set_lr(self, lr):
        self.current_lr = lr
        self.optimizer.defaults['lr'] = lr


class StaticLR(LRScheduler):
    def __init__(self, optimizer):
        super().__init__(optimizer)

    def step(self):
        self._step_count += 1


class StepLR(LRScheduler):
    def __init__(self, optimizer, step_size, gamma=0.1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def step(self):
        self._step_count += 1
        if self._step_count % self.step_size == 0:
            new_lr = self.current_lr * self.gamma
            self._set_lr(new_lr)


class ReduceLROnPlateau(LRScheduler):
    def __init__(self, optimizer, patience=10, factor=0.1, min_lr=1e-8, mode='min'):
        super().__init__(optimizer)
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.mode = mode

        self.best = None
        self.num_bad = 0

    def step(self, metric):
        self._step_count += 1

        if self.best is None:
            self.best = metric
            return

        improved = False
        if self.mode == 'min':
            improved = metric < self.best
        else:
            improved = metric > self.best

        if improved:
            self.best = metric
            self.num_bad = 0
        else:
            self.num_bad += 1

        if self.num_bad >= self.patience:
            new_lr = max(self.current_lr * self.factor, self.min_lr)
            self._set_lr(new_lr)
            self.num_bad = 0
