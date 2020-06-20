import torch

def critic_update(state_mb, return_mb, q, optim_q):
    val_loc = q(state_mb)
    critic_loss = (return_mb - val_loc).pow(2).mean()

    optim_q.zero_grad()
    critic_loss.backward()
    optim_q.step()

    del val_loc
    critic_loss_numpy = critic_loss.detach().cpu().numpy()
    del critic_loss

    yield critic_loss_numpy, 1

def get_advantage(next_value, reward_mat, value_mat, masks, gamma=0.99, tau=0.95):
    value_mat = torch.cat([value_mat, torch.FloatTensor([[next_value]])])
    gae = 0
    returns = []

    for step in reversed(range(len(reward_mat))):
        delta = reward_mat[step] + gamma * value_mat[step + 1] * masks[step] - value_mat[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + value_mat[step])
    return returns