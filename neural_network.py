import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import os


class Net(nn.Module):
    def __init__(self, board_size):
        super(Net, self).__init__()
        self.board_size = board_size

        # common layers
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # action policy layers
        self.act_conv1 = nn.Conv2d(in_channels=128, out_channels=4, kernel_size=1)
        self.act_fc1 = nn.Linear(in_features=4*board_size*board_size,
                                out_features=board_size*board_size)

        # state value layers
        self.val_conv1 = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=1)
        self.val_fc1 = nn.Linear(2*board_size*board_size, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x_act = F.relu(self.act_conv1(x))
        x_act = torch.flatten(x_act, start_dim=1)
        x_act = torch.softmax(self.act_fc1(x_act), dim=-1)

        x_val = F.relu(self.val_conv1(x))
        x_val = torch.flatten(x_val, start_dim=1)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))
        return x_act, x_val


class PolicyNet:
    def __init__(self, board_size, model_file=None, use_gpu=True):
        self.board_size = board_size
        self.model_file = model_file

        self.buffer_size = 10000
        self.data_buffer = deque(maxlen=self.buffer_size)

        self.states = []
        self.mcts_probs = []
        self.players = []

        self.batch_size = 512
        self.epochs = 5

        if use_gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        self.policy_value_net = Net(board_size).to(self.device)

        self.optimizer = torch.optim.Adam(self.policy_value_net.parameters(), lr=1e-4, weight_decay=1e-4)

        if model_file:
            state_dict = torch.load(model_file)
            self.policy_value_net.load_state_dict(state_dict)
            print("model file is loaded")

    def get_policy_value(self, board):
        state = board.current_state()
        batch_state = np.expand_dims(state, axis=0)
        batch_state = torch.tensor(batch_state, dtype=torch.float32, device=self.device)
        act_probs, value = self.policy_value_net(batch_state)
        valid_actions = board.get_valid_move_pos()
        valid_actions_index = [pos[0]*self.board_size + pos[1] for pos in valid_actions]
        act_probs = act_probs.detach().numpy().flatten()
        return valid_actions, act_probs[valid_actions_index], value

    def collect_temp_data(self, board, actions_index, act_probs):
        self.states.append(board.current_state())
        all_action_probs = np.zeros(shape=(self.board_size*self.board_size, ))
        all_action_probs[actions_index] = act_probs
        self.mcts_probs.append(all_action_probs)
        self.players.append(board.get_turn())

    def clear_temp_data(self):
        self.states.clear()
        self.mcts_probs.clear()
        self.players.clear()

    def data_augment(self):
        pass


    def calc_and_collect_final_data(self, result):
        winners = np.zeros(shape=(len(self.players)))
        winners[np.array(self.players)==result] = 1.0
        self.data_buffer.extend(zip(self.states, self.mcts_probs, winners))
        ## 扩充数据
        # extend_data = self.data_augment()
        # self.data_buffer.extend(extend_data)
        self.clear_temp_data()

    def train_once(self):
        if self.batch_size > len(self.data_buffer):
            batch_data = self.data_buffer
        else:
            batch_data = random.sample(self.data_buffer, self.batch_size)
        batch_states = []
        batch_mcst_probs = []
        batch_results = []
        for data in batch_data:
            batch_states.append(data[0])
            batch_mcst_probs.append(data[1])
            batch_results.append(data[2])
        batch_states_tensor = torch.tensor(batch_states, dtype=torch.float32, device=self.device)
        batch_mcst_probs_tensor = torch.tensor(batch_mcst_probs, dtype=torch.float32, device=self.device)
        batch_results_tensor = torch.tensor(batch_results, dtype=torch.float32, device=self.device)
        # self.policy_value_net = self.policy_value_net.to(self.device)

        self.optimizer.zero_grad()
        act_probs, values = self.policy_value_net(batch_states_tensor)
        probs_loss = -(batch_mcst_probs_tensor * torch.log(act_probs)).mean()
        value_loss = F.mse_loss(values.flatten(), batch_results_tensor)
        loss = probs_loss + value_loss
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy()

    def train(self):
        print("training ...........")
        losses = []
        for epoch in range(self.epochs):
            print("epoch {}/{}".format(epoch, self.epochs))
            loss = self.train_once()
            losses.append(loss)
        print("loss: {}".format(np.mean(losses)))

    def save_model(self, model_name):
        model_path = "saved_model"
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        torch.save(self.policy_value_net.state_dict(),f=os.path.join(model_path, model_name))




if __name__ == '__main__':
    board_size = 15
    net = Net(board_size=board_size)

    input_data = np.random.random(size=(8, 4, board_size, board_size))
    input_tensor = torch.tensor(input_data, dtype=torch.float32)

    act_tenor, val_tensor = net(input_tensor)

    print(act_tenor.shape, val_tensor.shape)
