import torch


class OptimizerDetails:
    def __init__(self):
        self.num_steps = None
        self.operation_func = None
        self.backward_operation_func = None
        self.optimizer = None # handle it on string level
        self.lr = None
        self.forward_loss_func = None
        self.backward_loss_func = None
        self.max_iters = 0
        self.loss_cutoff = None
        self.lr_scheduler = None
        self.warm_start = None
        self.old_img = None
        self.fact = 0.5
        self.print = False
        self.print_every = None
        self.folder = None
        self.guidance_3 = False
        self.guidance_2 = False
        self.optim_guidance_3_wt = 0
        self.optim_guidance_3_iters = 1
        self.original_guidance = False
        self.sampling_type = None
        self.loss_save = None
        self.ddpm = False
        # ME
        self.epsilon_w = 1
        self.guidance_increase_factor = 1
        self.eps = 1.0
        self.debug = False
        self.print_diff_stats = False
        self.poison_loss_w = 0
        self.matching_loss_w = 1.0
        self.normalize = False
        self.dm = None
        self.ds = None
        self.augment = None
        # Transform for ImageNet inputs
        self.imagenet_transform = None


class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)