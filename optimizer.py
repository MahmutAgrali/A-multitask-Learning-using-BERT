from typing import Callable, Iterable, Tuple
import math

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            #print(self.state)
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]

                # Complete the implementation of AdamW here, reading and saving
                # your state in the `state` dictionary above.
                # The hyperparameters can be read from the `group` dictionary
                # (they are lr, betas, eps, weight_decay, as saved in the constructor).
                #
                # 1- Update first and second moments of the gradients
                # 2- Apply bias correction
                #    (using the "efficient version" given in https://arxiv.org/abs/1412.6980;
                #     also given in the pseudo-code in the project description).
                # 3- Update parameters (p.data).
                # 4- After that main gradient-based update, update again using weight decay
                #    (incorporating the learning rate again).

                ### TODO_DONE
                
                if not state:                               # check if there is state
                    state["t_step"] = 0                     # set to zero t_step
                    state["m"] = torch.zeros_like(p.data)   # set to zero like tensor m
                    state["v"] = torch.zeros_like(p.data)   # set to zero like tensor v
                    state["theta"] = p.data                 # set to zero theta
                B1, B2 = group["betas"]                     # get betas from the dict
                eps = group["eps"]                          # get eps from the dict
                weight_decay = group["weight_decay"]        # get weight_decay from the dict
                t_step = state["t_step"]                    # get t_step from the dict
                mt_1 = state["m"]                           # get m from the dict
                vt_1 = state["v"]                           # get v from the dict
                thetat_1 = state["theta"]                   # get theta from the dict
              
                t_step += 1                                 # increase step


                mt = B1*mt_1 + (1-B1)*grad                                          # calculate m value for t time
                vt = B2*vt_1 + (1-B2)*(torch.pow(grad,2))                           # calculate v value for t time
                alphat = alpha * math.sqrt(1-B2**t_step)/(1-B1**t_step)             # calculate alpha value for t time
                thetat_head = - alphat*mt/(torch.sqrt(vt) + eps)                    # calculate theta value for t time

                thetat = thetat_1 + thetat_head                                     # sum previous theta and new theta for t time

                state["t_step"] = t_step                                            # save t_step into dict
                state["m"]      = mt                                                # save m into dict
                state["v"]      = vt                                                # save v into dict
                state["theta"]  = thetat                                            # save theta into dict
                
                p.data.add_(thetat_head)                                            # add the calculated theta to the gradient parameter
                if weight_decay != 0:                                               # check weight decay is not equal to zero
                    p.data += -weight_decay*alpha*p.data                            # apply weight decay to the gradient parameter
                #raise NotImplementedError

        return loss
