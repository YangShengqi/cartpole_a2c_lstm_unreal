import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import gym
import pickle

# delete cart velocity state observation
# made a standard cartpole env as POMDP!!!!!!!!!!!!!!!!!!!
STATE_DIM = 4
ACTION_DIM = 2
STEP = 100
SAMPLE_NUMS = 1000
A_HIDDEN = 64
C_HIDDEN = 64
OPTIM_BATCH = 20
AUX_OPTIM_BATCH = 2
EPISODE_NUM = 100


def save_model(model, path):
    with open(path, 'wb') as file:
        pickle.dump(model, file)


def load_model(path):
    with open(path, 'rb') as file:
        model = pickle.load(file)
    return model


# actor using a LSTM + fc network architecture to estimate hidden states.
class ActorNetwork(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(ActorNetwork, self).__init__()
        self.lstm = nn.LSTM(in_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, out_size)
        self.aux_fc = nn.Linear(hidden_size, 128)
        self.aux_fc2 = nn.Linear(128, 1)

    def forward(self, x, hidden):
        x, hidden = self.lstm(x, hidden)
        pred = self.aux_fc(x)
        pred = self.aux_fc2(pred)
        x = self.fc(x)
        x = self.fc2(x)
        x = F.log_softmax(x, 2)
        return x, hidden, pred


# critic using a LSTM + fc network architecture to estimate hidden states.
class ValueNetwork(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(ValueNetwork, self).__init__()
        self.lstm = nn.LSTM(in_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, out_size)

    def forward(self, x, hidden):
        x, hidden = self.lstm(x, hidden)
        x = self.fc(x)
        x = self.fc2(x)
        return x, hidden


def roll_out(actor_network, task, sample_nums, value_network, init_state, device):
    states = []
    actions = []
    rewards = []
    trues = []
    is_done = False
    final_r = 0
    score = 0
    state = init_state
    a_hx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0).to(device)
    a_cx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0).to(device)
    c_hx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0).to(device)
    c_cx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0).to(device)

    for j in range(sample_nums):
        states.append(state)
        log_softmax_action, (a_hx, a_cx), pred = actor_network(torch.Tensor([state]).unsqueeze(0).to(device), (a_hx, a_cx))
        softmax_action = torch.exp(log_softmax_action)
        a = softmax_action.cpu().data.numpy()[0][0]
        action = np.random.choice(ACTION_DIM, p=softmax_action.cpu().data.numpy()[0][0])
        one_hot_action = [int(k == action) for k in range(ACTION_DIM)]
        next_state, reward, done, _ = task.step(action)
        trues.append(next_state[2])
        pred = pred.squeeze(0).squeeze(0).squeeze(0).to('cpu').data.numpy()
        next_state[2] = pred
        actions.append(one_hot_action)
        rewards.append(reward)
        final_state = next_state
        state = next_state
        if done:
            is_done = True
            state = task.reset()
            score = j+1
            #print score while training
            #print(score)
            break
    if not is_done:
        c_out, _ = value_network(torch.Tensor([final_state]), (c_hx, c_cx))
        final_r = c_out.cpu().data.numpy()
        score = sample_nums
    return states, actions, rewards, trues, final_r, state, score


def discount_reward(r, gamma, final_r):
    discounted_r = np.zeros_like(r)
    running_add = final_r
    for t in reversed(range(0, len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def main():
    device = torch.device('cuda', index=0) if torch.cuda.is_available() else torch.device('cpu')
    pre_model_exist = False

    # init a task generator for data fetching
    task = gym.make("CartPole-v1")
    init_state = task.reset()
    # init_state = np.delete(init_state, 1)

    # init value network and actor network
    if pre_model_exist:
        value_network = load_model('./value_aux')
        actor_network = load_model('./actor_aux')
    else:
        value_network = ValueNetwork(in_size=STATE_DIM, hidden_size=C_HIDDEN, out_size=1)
        actor_network = ActorNetwork(in_size=STATE_DIM, hidden_size=A_HIDDEN, out_size=ACTION_DIM)
    value_network.to(device)
    actor_network.to(device)
    value_network_optim = torch.optim.Adam(value_network.parameters(), lr=1e-3)
    actor_network_optim = torch.optim.Adam(actor_network.parameters(), lr=1e-4)

    for step in range(STEP):
        # sample
        score_sum = 0
        s_buff = []
        a_buff = []
        q_buff = []
        t_buff = []
        for samp_i in range(EPISODE_NUM):
            states, actions, rewards, trues, final_r, current_state, score = roll_out(actor_network, task, SAMPLE_NUMS, value_network, init_state, device)
            init_state = current_state
            score_sum += score
            qvalues = discount_reward(rewards, 0.99, final_r)
            s_buff.append(states)
            a_buff.append(actions)
            q_buff.append(qvalues)
            t_buff.append(trues)

        # preprocess
        score_avg = score_sum / EPISODE_NUM
        s_pad = []
        a_pad = []
        q_pad = []
        t_pad = []
        for element in s_buff:
            s_pad.append(torch.Tensor(element))
        s_pad = pad_sequence(s_pad, batch_first=True).to(device)
        for element in a_buff:
            a_pad.append(torch.Tensor(element))
        a_pad = pad_sequence(a_pad, batch_first=True).to(device)
        for element in q_buff:
            q_pad.append(torch.Tensor(element))
        q_pad = pad_sequence(q_pad, batch_first=True).to(device)
        for element in t_buff:
            t_pad.append(torch.Tensor(element))
        t_pad = pad_sequence(t_pad, batch_first=True).to(device)

        # train
        actor_loss_sum = 0
        value_loss_sum = 0
        aux_loss_sum = 0
        aux_sum = 0
        for epoch_i in range(5):
            # shuffle
            perm = np.arange(s_pad.shape[0])
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm).to(device)
            state = s_pad[perm].clone()
            action = a_pad[perm].clone()
            qvalue = q_pad[perm].clone()
            true = t_pad[perm].clone()

            aux_opt_sum = 0
            for opt_i in range(int(EPISODE_NUM/OPTIM_BATCH)):
                ind = slice(opt_i * OPTIM_BATCH, min((opt_i + 1) * OPTIM_BATCH, len(state)))
                batch = min((opt_i+1)*OPTIM_BATCH, len(state)) - opt_i*OPTIM_BATCH
                s_batch, a_batch, q_batch, t_batch = state[ind], action[ind], qvalue[ind], true[ind]

                # train actor network
                a_hx = torch.zeros(1, batch, A_HIDDEN).to(device)
                a_cx = torch.zeros(1, batch, A_HIDDEN).to(device)
                c_hx = torch.zeros(1, batch, C_HIDDEN).to(device)
                c_cx = torch.zeros(1, batch, C_HIDDEN).to(device)
                actor_network_optim.zero_grad()
                log_softmax_actions, _, _ = actor_network(s_batch, (a_hx, a_cx))
                vs, _ = value_network(s_batch, (c_hx, c_cx))
                vs.detach()
                qs = q_batch.unsqueeze(2)
                advantages = qs - vs
                # aa = log_softmax_actions * a_batch
                # bb = torch.sum(aa, 1).unsqueeze(1).expand(-1, advantages.shape[1], -1)
                # cc = bb * advantages
                # dd = - torch.mean(cc)
                log_prob = torch.sum(log_softmax_actions * a_batch, 1).unsqueeze(1).expand(-1, advantages.shape[1], -1)
                actor_network_loss = - torch.mean(log_prob * advantages)
                actor_network_loss.backward()
                # torch.nn.utils.clip_grad_norm_(actor_network.parameters(), 0.5)
                actor_network_optim.step()
                actor_network_loss = actor_network_loss.to('cpu')
                actor_loss_sum += actor_network_loss

                # train value network
                value_network_optim.zero_grad()
                target_values = qs
                c_hx = torch.zeros(1, batch, C_HIDDEN).to(device)
                c_cx = torch.zeros(1, batch, C_HIDDEN).to(device)
                values, _ = value_network(s_batch, (c_hx, c_cx))
                criterion = nn.MSELoss()
                value_network_loss = criterion(values, target_values)
                value_network_loss.backward()
                # torch.nn.utils.clip_grad_norm_(value_network.parameters(), 0.5)
                value_network_optim.step()
                value_network_loss = value_network_loss.to('cpu')
                value_loss_sum += value_network_loss

                # train aux network
                aux_opt_num = int(s_batch.shape[0] / AUX_OPTIM_BATCH)
                aux_opt_sum += aux_opt_num
                for aux_opt_i in range(aux_opt_num):
                    aux_ind = slice(aux_opt_i * AUX_OPTIM_BATCH,
                                    min((aux_opt_i + 1) * AUX_OPTIM_BATCH, s_batch.shape[0]))
                    aux_batch = min((aux_opt_i + 1) * AUX_OPTIM_BATCH, s_batch.shape[0]) - aux_opt_i * AUX_OPTIM_BATCH
                    s_b, t_b = s_batch[aux_ind], t_batch[aux_ind]
                    a_hx = torch.zeros(1, aux_batch, A_HIDDEN).to(device)
                    a_cx = torch.zeros(1, aux_batch, A_HIDDEN).to(device)
                    _, _, p_b = actor_network(s_b, (a_hx, a_cx))
                    p_b = p_b.view(-1, 1).squeeze(1)
                    t_b = t_b.view(-1, 1).squeeze(1)
                    actor_network_optim.zero_grad()
                    criterion = nn.MSELoss()
                    aux_loss = criterion(p_b, t_b)
                    aux_loss.backward()
                    actor_network_optim.step()
                    aux_loss = aux_loss.to('cpu')
                    aux_loss_sum += aux_loss

            aux_sum += aux_opt_sum

        actor_loss_avg = actor_loss_sum.data.numpy() / 25
        value_loss_avg = value_loss_sum.data.numpy() / 25
        aux_loss_avg = aux_loss_sum.data.numpy() / aux_sum
        print('step:', step, '| actor_loss:', actor_loss_avg, '| critic_loss:', value_loss_avg, '| aux_loss:', aux_loss_avg, '| score:', score_avg)
        actor_network.to('cpu')
        value_network.to('cpu')
        save_model(actor_network, './actor_aux')
        save_model(value_network, './value_aux')
        actor_network.to(device)
        value_network.to(device)

    torch.cuda.empty_cache()

    # # Testing
    # if (step + 1) % 50== 0:
    #         result = 0
    #         test_task = gym.make("CartPole-v0")
    #         for test_epi in range(10):
    #             state = test_task.reset()
    #             for test_step in range(200):
    #                 softmax_action = torch.exp(actor_network(Variable(torch.Tensor([state]))))
    #                 #print(softmax_action.data)
    #                 action = np.argmax(softmax_action.data.numpy()[0])
    #                 next_state,reward,done,_ = test_task.step(action)
    #                 result += reward
    #                 state = next_state
    #                 if done:
    #                     break
    #         print("step:",step+1,"test result:",result/10.0)
    #         steps.append(step+1)
    #         test_results.append(result/10)


if __name__ == '__main__':
    main()
