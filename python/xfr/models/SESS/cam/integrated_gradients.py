import torch
from xfr.models.SESS.utils import convert_to_gray
from torch.autograd import Variable

class IntegratedGradients(object):
    def __init__(self, model, n_steps=20):
        super(IntegratedGradients, self).__init__()
        self.n_steps = n_steps
        self.model = model.eval()

    def forward(self, x, x_baseline=None, class_idx=None, retain_graph=False):
        if x_baseline is None:
            x_baseline = torch.zeros_like(x)
            x_baseline = torch.rand_like(x)

        else:
            x_baseline = x_baseline.cuda()
        assert x_baseline.size() == x.size()

        saliency_map = torch.zeros_like(x) # [1, 3, H, W]

        x_diff = x - x_baseline
        for alpha in torch.linspace(0., 1., self.n_steps):
            x_step = x_baseline + alpha * x_diff
            x_step = Variable(x_step, requires_grad=True)
            logit = self.model(x_step)

            if class_idx is None:
                score = logit[:, logit.max(1)[-1]].squeeze()
            else:
                score = logit[:, class_idx].squeeze()

            self.model.zero_grad()
            score.backward(retain_graph=retain_graph)
            saliency_map += x_step.grad

        saliency_map = saliency_map / self.n_steps
        saliency_map = convert_to_gray(saliency_map.cpu().numpy()[0]) # [1, 1, H, W]

        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min) / (saliency_map_max - saliency_map_min).data

        return saliency_map

    def __call__(self, x, x_baseline=None, class_idx=None, retain_graph=False):
        return self.forward(x, x_baseline, class_idx, retain_graph)
