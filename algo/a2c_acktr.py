import torch
import torch.nn as nn
import torch.optim as optim

from .kfac import KFACOptimizer
from .kbfgs import KBFGSOptimizer


class A2C_ACKTR():
    def __init__(self,
                 actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 alpha=None,
                 max_grad_norm=None,
                 acktr=False,
                 kbfgs=False,
                 if_homo=False,
                 if_clip=True,
                 if_momentumGrad=False,
                 stat_decay=0.99,
                 stat_decay_A=0.99,
                 stat_decay_G=0.0,
                 if_decoupled_decay=False,
                 if_invert_A=False,
                 if_eigen=True):

        self.actor_critic = actor_critic
        
        self.acktr = acktr
        self.kbfgs = kbfgs

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        
        assert acktr + kbfgs < 2

        if acktr:
            self.optimizer = KFACOptimizer(actor_critic, lr=lr, stat_decay=stat_decay,
                                           damping=eps,
                                           if_homo=if_homo, if_eigen=if_eigen)
        elif kbfgs:
            
            print('stat_decay_A')
            print(stat_decay_A)
            
            print('stat_decay_G')
            print(stat_decay_G)
            
#             sys.exit()
            

#             self.optimizer = KBFGSOptimizer(actor_critic, lr=lr, damping=eps, if_homo=if_homo)
            self.optimizer = KBFGSOptimizer(actor_critic, lr=lr,
                                            stat_decay=stat_decay,
                                            stat_decay_A=stat_decay_A,
                                            stat_decay_G=stat_decay_G,
                                            if_decoupled_decay=if_decoupled_decay,
                                            damping=eps, if_homo=if_homo,
                                            if_clip=if_clip,
                                            if_momentumGrad=if_momentumGrad,
                                            if_invert_A=if_invert_A)
        else:
            # momentum = 0 (default), meaning that mini-batch grad is used, 
            # instead of moving average
            self.optimizer = optim.RMSprop(
                actor_critic.parameters(), lr, eps=eps, alpha=alpha)

    def update(self, rollouts):
        
#         print('torch.norm(rollouts.obs)')
#         print(torch.norm(rollouts.obs))
        
#         print('rollouts.obs.size()')
#         print(rollouts.obs.size())
        
#         print('rollouts.actions.size()')
#         print(rollouts.actions.size())
        
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()
        
        if self.kbfgs:
            self.optimizer.kbfgs_stats_cur = True
            self.optimizer.kbfgs_stats_next = False

        values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.recurrent_hidden_states[0].view(-1, self.actor_critic.recurrent_hidden_state_size),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape))

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(advantages.detach() * action_log_probs).mean()
        



#         print('need to remove kbfgs')

        if self.acktr and self.optimizer.steps % self.optimizer.Ts == 0:
#         if (self.acktr or self.kbfgs) and self.optimizer.steps % self.optimizer.Ts == 0:
            # Sampled fisher, see Martens 2014
            self.actor_critic.zero_grad()
            pg_fisher_loss = -action_log_probs.mean()

            value_noise = torch.randn(values.size())
            if values.is_cuda:
                value_noise = value_noise.cuda()

            sample_values = values + value_noise
            vf_fisher_loss = -(values - sample_values.detach()).pow(2).mean()

            fisher_loss = pg_fisher_loss + vf_fisher_loss
            self.optimizer.acc_stats = True
            fisher_loss.backward(retain_graph=True)
            self.optimizer.acc_stats = False
            
        if self.kbfgs:
            self.optimizer.acc_stats = False
            
            self.optimizer.kbfgs_stats = True
            
            

        self.optimizer.zero_grad()
        
        (value_loss * self.value_loss_coef + action_loss -
         dist_entropy * self.entropy_coef).backward()
        
#         print('self.optimizer.m_aa')
#         print(self.optimizer.m_aa)

#         if self.acktr == False:
        if self.acktr == False and self.kbfgs == False:
        # i.e. using RMSprop
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                     self.max_grad_norm)

        # this is the KBFGS / KFAC / RMSprop step
        self.optimizer.step()
        
        # post update
        if self.kbfgs:
            # perform another forward/backward pass
            
#             print('torch.norm(rollouts.obs) in post step')
#             print(torch.norm(rollouts.obs))
            
#             sys.exit()
            
            self.optimizer.kbfgs_stats_cur = False
            self.optimizer.kbfgs_stats_next = True
            
            values_next, action_log_probs_next, dist_entropy_next, _ = self.actor_critic.evaluate_actions(
                rollouts.obs[:-1].view(-1, *obs_shape),
                rollouts.recurrent_hidden_states[0].view(-1, self.actor_critic.recurrent_hidden_state_size),
                rollouts.masks[:-1].view(-1, 1),
                rollouts.actions.view(-1, action_shape))
            
            values_next = values_next.view(num_steps, num_processes, 1)
            action_log_probs_next = action_log_probs_next.view(num_steps, num_processes, 1)
            

        
            advantages_next = rollouts.returns[:-1] - values_next
            value_loss_next = advantages_next.pow(2).mean()
            
            action_loss_next = -(advantages_next.detach() * action_log_probs_next).mean()
            
#             print('torch.norm(list(self.actor_critic.parameters())[0].grad) before zero grad')
#             print(torch.norm(list(self.actor_critic.parameters())[0].grad))
            
            self.optimizer.zero_grad()
            
#             print('torch.norm(list(self.actor_critic.parameters())[0].grad) after zero grad')
#             print(torch.norm(list(self.actor_critic.parameters())[0].grad))
            
#             sys.exit()

            (value_loss_next * self.value_loss_coef + action_loss_next -
             dist_entropy_next * self.entropy_coef).backward()
            
#             print('torch.norm(list(self.actor_critic.parameters())[0].grad) after backward')
#             print(torch.norm(list(self.actor_critic.parameters())[0].grad))
            
#             sys.exit()
            
#             print('disable post update for now')
            post_step_output = self.optimizer.post_step()
            if post_step_output == -1:
                return [], [], [], -1

        return value_loss.item(), action_loss.item(), dist_entropy.item(), 0
