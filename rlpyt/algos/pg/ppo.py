import numpy as np
import torch
import random

from rlpyt.algos.pg.base import PolicyGradientAlgo, OptInfo
from rlpyt.agents.base import AgentInputs, AgentInputsRnn
from rlpyt.utils.tensor import valid_mean
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.buffer import buffer_to, buffer_method
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.misc import iterate_mb_idxs

LossInputs = namedarraytuple("LossInputs",
    ["agent_inputs", "action", "return_", "advantage", "old_value", "valid", "old_dist_info"])


class PPO(PolicyGradientAlgo):
    """
    Proximal Policy Optimization algorithm.  Trains the agent by taking
    multiple epochs of gradient steps on minibatches of the training data at
    each iteration, with advantages computed by generalized advantage
    estimation.  Uses clipped likelihood ratios in the policy loss.
    """

    def __init__(
            self,
            discount=0.99,
            learning_rate=0.001,
            value_loss_coeff=0.5,
            entropy_loss_coeff=0.01,
            OptimCls=torch.optim.Adam,
            optim_kwargs={'eps':1e-5},
            clip_grad_norm=1.e6,
            initial_optim_state_dict=None,
            gae_lambda=1,
            minibatches=4,
            epochs=4,
            ratio_clip=0.1,
            linear_lr_schedule=False,
            normalize_advantage=False,
            alpha=1.1,
            clip_beta=0.99,
            ac_clip=False
            ):
        """Saves input settings."""
        if optim_kwargs is None:
            optim_kwargs = dict()
        save__init__args(locals())
        self.grad_clip = None

    def initialize(self, *args, **kwargs):
        """
        Extends base ``initialize()`` to initialize learning rate schedule, if
        applicable.
        """
        super().initialize(*args, **kwargs)
        self._batch_size = self.batch_spec.size // self.minibatches  # For logging.
        if self.linear_lr_schedule:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=self.optimizer,
                lr_lambda=lambda itr: (self.n_itr - itr) / self.n_itr)  # Step once per itr.
            self._ratio_clip = self.ratio_clip  # Save base value.


    def compute_minibatch_gradients(self, samples):
        """
        Train the agent, for multiple epochs over minibatches taken from the
        input samples.  Organizes agent inputs from the training data, and
        moves them to device (e.g. GPU) up front, so that minibatches are
        formed within device, without further data transfer.
        """
        recurrent = self.agent.recurrent
        agent_inputs = AgentInputs(  # Move inputs to device once, index there.
            observation=samples.env.observation,
            prev_action=samples.agent.prev_action,
            prev_reward=samples.env.prev_reward,
        )
        agent_inputs = buffer_to(agent_inputs, device=self.agent.device)
        return_, advantage, valid, value = self.process_returns(samples, traj_infos)
        loss_inputs = LossInputs(  # So can slice all.
            agent_inputs=agent_inputs,
            action=samples.agent.action,
            return_=return_,
            old_value=value,
            advantage=advantage,
            valid=valid,
            old_dist_info=samples.agent.agent_info.dist_info,
        )
        if recurrent:
            # Leave in [B,N,H] for slicing to minibatches.
            init_rnn_state = samples.agent.agent_info.prev_rnn_state[0]  # T=0.
        T, B = samples.env.reward.shape[:2]

        # If recurrent, use whole trajectories, only shuffle B; else shuffle all.
        batch_size = T * B if self.agent.recurrent else T * B

        gradients = []

        for idxs in iterate_mb_idxs(batch_size, batch_size, shuffle=True):
            T_idxs = slice(None) if recurrent else idxs % T
            B_idxs = idxs if recurrent else idxs // T
            self.optimizer.zero_grad()
            rnn_state = init_rnn_state[B_idxs] if recurrent else None
            # NOTE: if not recurrent, will lose leading T dim, should be OK.
            loss, pi_loss, value_loss, entropy, perplexity = self.loss(
                *loss_inputs[T_idxs, B_idxs], rnn_state)
            loss.backward()
            # for i, p in enumerate(self.agent.parameters()):
            #     print(i, p.grad)
            # print([p for p in self.agent.parameters()])
            # print([p for p in len(self.agent.parameters())])
            gradient = np.concatenate([p.grad.data.cpu().numpy().flatten() for p in self.agent.parameters()]).ravel()
            gradients.append(gradient)

            self.update_counter += 1

        return gradients

    def optimize_agent(self, itr, samples):
        """
        Train the agent, for multiple epochs over minibatches taken from the
        input samples.  Organizes agent inputs from the training data, and
        moves them to device (e.g. GPU) up front, so that minibatches are
        formed within device, without further data transfer.
        """
        recurrent = self.agent.recurrent
        agent_inputs = AgentInputs(  # Move inputs to device once, index there.
            observation=samples.env.observation,
            prev_action=samples.agent.prev_action,
            prev_reward=samples.env.prev_reward,
        )
        agent_inputs = buffer_to(agent_inputs, device=self.agent.device)
        if hasattr(self.agent, "update_obs_rms"):
            self.agent.update_obs_rms(agent_inputs.observation)
        return_, advantage, valid, value = self.process_returns(samples)
        loss_inputs = LossInputs(  # So can slice all.
            agent_inputs=agent_inputs,
            action=samples.agent.action,
            return_=return_,
            advantage=advantage,
            old_value=value,
            valid=valid,
            old_dist_info=samples.agent.agent_info.dist_info,
        )

        if recurrent:
            # Leave in [B,N,H] for slicing to minibatches.
            init_rnn_state = samples.agent.agent_info.prev_rnn_state[0]  # T=0.
        T, B = samples.env.reward.shape[:2]
        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
        # If recurrent, use whole trajectories, only shuffle B; else shuffle all.
        batch_size = B if self.agent.recurrent else T * B
        mb_size = batch_size // self.minibatches
        for _ in range(self.epochs):
            for idxs in iterate_mb_idxs(batch_size, mb_size, shuffle=True):
                T_idxs = slice(None) if recurrent else idxs % T
                B_idxs = idxs if recurrent else idxs // T
                self.optimizer.zero_grad()
                rnn_state = init_rnn_state[B_idxs] if recurrent else None
                # NOTE: if not recurrent, will lose leading T dim, should be OK.
                loss, pi_loss, value_loss, entropy, perplexity = self.loss(
                    *loss_inputs[T_idxs, B_idxs], rnn_state)
                loss.backward()

                # do better clipping here.
                if self.ac_clip:
                    grad_norm = self.coordinatewise_clip_grad_norm_(self.agent.parameters())
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.agent.parameters(), self.clip_grad_norm)
                self.optimizer.step()

                opt_info.piLoss.append(pi_loss.item())
                opt_info.valueLoss.append(value_loss.item())
                opt_info.loss.append(loss.item())
                opt_info.gradNorm.append(grad_norm)
                opt_info.entropy.append(entropy.item())
                opt_info.perplexity.append(perplexity.item())
                self.update_counter += 1
        if self.linear_lr_schedule:
            self.lr_scheduler.step()
            self.ratio_clip = self._ratio_clip * (self.n_itr - itr) / self.n_itr

        return opt_info

    def coordinatewise_clip_grad_norm_(self, parameters):
        """Clips gradient norm of an iterable of parameters.

        The norm is computed over all gradients together, as if they were
        concatenated into a single vector. Gradients are modified in-place.

        Arguments:
            parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
                single Tensor that will have gradients normalized
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
                infinity norm.

        """
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        if self.grad_clip is None:
            self.grad_clip = [p.grad.data.abs() for p in parameters]
        else:
            self.grad_clip = [(self.clip_beta * self.grad_clip[i].abs().pow(self.alpha) + \
                               (1 - self.clip_beta) * p.grad.data.abs().pow(self.alpha)).pow(1. / self.alpha)
                              for i, p in enumerate(parameters)]
        # print(self.grad_clip)
        should_print = random.random() < 0.0000025
        for i, p in enumerate(parameters):
            div = torch.add(p.grad.data.abs(), 1e-6)
            ratio = torch.div(self.grad_clip[i], div)
            min = torch.min(ratio, torch.ones(ratio.size()).cuda())
            if should_print:
                print(min)
            p.grad.data.mul_(min)

        return torch.nn.utils.clip_grad_norm_(parameters, 1e12)

    def loss(self, agent_inputs, action, return_, advantage, old_value, valid, old_dist_info,
            init_rnn_state=None):
        """
        Compute the training loss: policy_loss + value_loss + entropy_loss
        Policy loss: min(likelhood-ratio * advantage, clip(likelihood_ratio, 1-eps, 1+eps) * advantage)
        Value loss:  0.5 * (estimated_value - return) ^ 2
        Calls the agent to compute forward pass on training data, and uses
        the ``agent.distribution`` to compute likelihoods and entropies.  Valid
        for feedforward or recurrent agents.
        """
        if init_rnn_state is not None:
            # [B,N,H] --> [N,B,H] (for cudnn).
            init_rnn_state = buffer_method(init_rnn_state, "transpose", 0, 1)
            init_rnn_state = buffer_method(init_rnn_state, "contiguous")
            dist_info, value, _rnn_state = self.agent(*agent_inputs, init_rnn_state)
        else:
            dist_info, value = self.agent(*agent_inputs)
        dist = self.agent.distribution

        ratio = dist.likelihood_ratio(action, old_dist_info=old_dist_info,
            new_dist_info=dist_info)
        surr_1 = ratio * advantage
        clipped_ratio = torch.clamp(ratio, 1. - self.ratio_clip,
            1. + self.ratio_clip)
        surr_2 = clipped_ratio * advantage
        surrogate = torch.min(surr_1, surr_2)
        pi_loss = - valid_mean(surrogate, valid)

        value_error1 = 0.5 * (value - return_) ** 2
        # clip by old_values
        clipped_value = old_value + torch.clamp(value - old_value, -self.ratio_clip, self.ratio_clip)
        value_error1 = 0.5 * (value - return_) ** 2
        value_error2 = 0.5 * (clipped_value - return_) ** 2
        # print('old', old_value)
        # print('new', value)
        # print('clipped', clipped_value)
        # print(valid_mean(value_error1, valid), valid_mean(value_error2, valid))
        value_loss = self.value_loss_coeff * torch.max(valid_mean(value_error1, valid),
                                                       valid_mean(value_error2, valid))

        entropy = dist.mean_entropy(dist_info, valid)
        entropy_loss = - self.entropy_loss_coeff * entropy

        loss = pi_loss + value_loss + entropy_loss

        perplexity = dist.mean_perplexity(dist_info, valid)
        return loss, pi_loss, value_loss, entropy, perplexity
