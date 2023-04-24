# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# from tqdm import tqdm

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class SudokuNet(nn.Module):
#     def __init__(self):
#         super(SudokuNet, self).__init__()
#         self.fc1 = nn.Linear(81, 128)
#         self.fc2 = nn.Linear(128, 81)
    
#     def forward(self, x):
#         x = x.view(-1, 81)
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x.view(-1, 9, 9)

# class Agent:
#     def __init__(self, net, optimizer, gamma=0.9):
#         self.net = net.to(device)
#         self.optimizer = optimizer
#         self.gamma = gamma
#         self.total_reward = 0
        
#     def get_action(self, state, epsilon=0.1):
#         if np.random.uniform() > epsilon:
#             state = torch.FloatTensor(state).to(device)
#             q_values = self.net(state).data.cpu().numpy()
#             return np.argmax(q_values)
#         else:
#             return np.random.randint(9)
    
#     def update(self, state, action, reward, next_state):
#         state = torch.FloatTensor(state).to(device)
#         next_state = torch.FloatTensor(next_state).to(device)
#         q_values = self.net(state).data.cpu().numpy()
#         next_q_values = self.net(next_state).data.cpu().numpy()
#         q_values[0][action] = reward + self.gamma * np.max(next_q_values)
#         q_values = torch.FloatTensor(q_values).to(device)
#         loss = nn.MSELoss()(self.net(state), q_values)
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
#         self.total_reward += reward

# class SudokuEnv:
#     def __init__(self):
#         self.board = np.zeros((9, 9), dtype=np.int32)
#         self.reset()
    
#     def reset(self):
#         self.board = np.zeros((9, 9), dtype=np.int32)
#         for i in range(10):
#             row = np.random.randint(9)
#             col = np.random.randint(9)
#             val = np.random.randint(1, 10)
#             self.board[row][col] = val
#         return self.board
    
#     def step(self, action):
#         row, col = action // 9, action % 9
#         if self.board[row][col] == 0:
#             reward = 1
#             self.board[row][col] = np.random.randint(1, 10)
#         else:
#             reward = -1
#         done = (self.board == 0).sum() == 0
#         return self.board, reward, done

# if __name__ == '__main__':
#     env = SudokuEnv()
#     net = SudokuNet().to(device)
#     optimizer = optim.Adam(net.parameters(), lr=0.001)
#     agent = Agent(net, optimizer)
#     for i in tqdm(range(1000)):
#         state = env.reset()
#         done = False
#         while not done:
#             action = agent.get_action(state)
#             next_state, reward, done = env.step(action)
#             agent.update(state, action, reward, next_state)
#             state = next_state
            
#         if i % 100 == 0:
#             tqdm.write("Iteration: {} | Total Reward: {}".format(i, agent.total_reward))
#             agent.total_reward = 0

import os
import datetime
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
from torchsummary import summary
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class SudokuModelA(nn.Module):
    def __init__(self):
        super(SudokuModelA, self).__init__()
        self.fc1 = nn.Linear(81, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 81)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SudokuModelB(nn.Module):
    def __init__(self):
        super(SudokuModelB, self).__init__()
        self.fc1 = nn.Linear(81, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 81)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SudokuModelC(nn.Module):
    def __init__(self):
        super(SudokuModelC, self).__init__()
        self.fc1 = nn.Linear(81, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 81)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SudokuModelD(nn.Module):
    def __init__(self):
        super(SudokuModelD, self).__init__()
        self.fc1 = nn.Linear(81, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 81)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class SudokuModel(nn.Module):
    def __init__(self, arch_type):
        super(SudokuModel, self).__init__()

        self.arch_type = arch_type

        # Fully connected
        if arch_type == 1:
            self.fc1 = nn.Linear(81, 128)
            self.fc2 = nn.Linear(128, 81)

        # Convolutional
        elif arch_type == 2:
            self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(128 * 9, 128)
            self.fc2 = nn.Linear(128, 81)

        # GRU
        elif arch_type == 3:
            self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.gru = nn.GRU(128, 256, batch_first=True)
            self.fc1 = nn.Linear(256, 128)
            self.fc2 = nn.Linear(128, 81)

        # LSTM
        elif arch_type == 4:
            self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.lstm = nn.LSTM(128, 256, batch_first=True)
            self.fc1 = nn.Linear(256, 128)
            self.fc2 = nn.Linear(128, 81)

    def forward(self, x):
        if self.arch_type == 1:
            x = x.view(-1, 81)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x.view(-1, 9, 9)

        elif self.arch_type == 2:
            x = x.view(-1, 1, 9, 9)
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = x.view(-1, 128 * 9)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x.view(-1, 9, 9)

        elif self.arch_type == 3:
            x = x.view(-1, 1, 9, 9)
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = x.view(-1, 9, 128)
            _, h_n = self.gru(x)
            x = h_n[-1]
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x.view(-1, 9, 9)

        elif self.arch_type == 4:
            x = x.view(-1, 1, 9, 9)
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = x.view(-1, 9, 128)
            # _, (h_n, _) = self.lstm(x)
            # x = h_n[-1]
            x, (h_n, _) = self.lstm(x)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x.view(-1, 9, 9)
            
            
class Agent:
    def __init__(self, net, optimizer, gamma=0.9):
        self.net = net.to(device)
        self.optimizer = optimizer
        self.gamma = gamma
        self.total_reward = 0
        
    # def get_action(self, state, epsilon=0.1):
    #     if np.random.uniform() > epsilon:
    #         state = torch.FloatTensor(state).to(device)
    #         q_values = self.net(state).data.cpu().numpy()
    #         valid_actions = np.argwhere(state == 0).cpu()
    #         valid_q_values = q_values.flatten()[valid_actions[:, 0], valid_actions[:, 1]]
    #         action_index = np.argmax(valid_q_values)
    #         action = tuple(valid_actions[action_index])
    #         return action
    #     else:
    #         valid_actions = np.argwhere(state == 0).cpu()
    #         action_index = np.random.randint(len(valid_actions))
    #         action = tuple(valid_actions[action_index])
    #         return action
    
    def get_action(self, state, epsilon=0.1):
        if np.random.uniform() > epsilon:
            state_tensor = torch.FloatTensor(state).to(device)
            q_values = self.net(state_tensor).data.cpu().numpy()
            valid_actions = np.argwhere(state == 0)
            valid_q_values = q_values[0, valid_actions[:, 0], valid_actions[:, 1]]
            action_index = np.argmax(valid_q_values)
            action = tuple(valid_actions[action_index])
            return action
        else:
            valid_actions = np.argwhere(state == 0)
            action_index = np.random.randint(len(valid_actions))
            action = tuple(valid_actions[action_index])
            return action


    
    # def update(self, state, action, reward, next_state):
    #     state = torch.FloatTensor(state).to(device)
    #     next_state = torch.FloatTensor(next_state).to(device)
    #     q_values = self.net(state).data.cpu().numpy()
    #     next_q_values = self.net(next_state).data.cpu().numpy()
    #     q_values[action] = reward + self.gamma * np.max(next_q_values)
    #     q_values = torch.FloatTensor(q_values).to(device)
    #     loss = nn.MSELoss()(self.net(state), q_values)
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
    #     self.total_reward += reward

    def update(self, state, action, reward, next_state):
        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        q_values = self.net(state).data.cpu().numpy()
        next_q_values = self.net(next_state).data.cpu().numpy()
        q_values[0, action[0], action[1]] = reward + self.gamma * np.max(next_q_values)
        q_values = torch.FloatTensor(q_values).to(device)
        loss = nn.MSELoss()(self.net(state), q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.total_reward += reward

class SudokuEnv:
    def __init__(self):
        self.board = np.zeros((9, 9), dtype=np.int32)
        self.reset()
    
    def reset(self):
        self.board = np.zeros((9, 9), dtype=np.int32)
        for i in range(10):
            row = np.random.randint(9)
            col = np.random.randint(9)
            val = np.random.randint(1, 10)
            self.board[row][col] = val
        return self.board
    
    def step(self, action):
        row, col = action
        if self.board[row][col] == 0:
            reward = 1
            self.board[row][col] = np.random.randint(1, 10)
        else:
            reward = -1
        done = (self.board == 0).sum() == 0
        return self.board, reward, done

def plot_and_save_graph(x, y, xlabel, ylabel, title, filename):
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename)
    plt.close()


if __name__ == '__main__':
    
    results_folder = "results"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    arch_type = 4
    
    env = SudokuEnv()
    net = SudokuModel(arch_type).to(device)
    input_size = (1, 9, 9)
    summary(net, input_size)

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    agent = Agent(net, optimizer)
    total_rewards = []
    losses = []  # Asumiendo que también quieres guardar las pérdidas por episodio

    for i in trange(1000, desc="Training", ncols=100):
        state = env.reset()
        done = False
        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            loss = agent.update(state, action, reward, next_state)  # Asegúrate de que la función update retorna la pérdida
            state = next_state
        total_rewards.append(agent.total_reward)
        losses.append(loss)  # Guarda la pérdida en la lista de pérdidas
        if (i+1) % 100 == 0:
            tqdm.write('Episode {}: Total reward = {}'.format(i+1, agent.total_reward))
            agent.total_reward = 0

    # Obtener la fecha y hora actuales
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    
    # Trazar y guardar el gráfico de rewards por época
    plot_and_save_graph(range(1, 1001), total_rewards, 'Episode', 'Total Reward', 'Total Reward per Episode',
                        os.path.join(results_folder, f'sudoku_RL_rewards_{current_time}.png'))

    # Trazar y guardar el gráfico de función de pérdida (Asumiendo que tienes una lista de pérdidas llamada `losses`)
    plot_and_save_graph(range(1, 1001), losses, 'Episode', 'Loss', 'Loss per Episode',
                        os.path.join(results_folder, f'sudoku_RL_losses_{current_time}.png'))

    print('\n\nGuardando el modelo...')
    

    # Guardar el modelo
    model_path = os.path.join(results_folder, f'sudoku_RL_{current_time}.pth')
    torch.save(net.state_dict(), model_path)

    # Guardar información relevante en un archivo de texto
    info_file_path = os.path.join(results_folder, f'sudoku_RL_info_{current_time}.txt')
    with open(info_file_path, 'w') as info_file:
        info_file.write(f'Model: SudokuNet\n')
        info_file.write(f'Date and time: {current_time}\n')
        info_file.write(f'Training episodes: 1000\n')
        info_file.write(f'Optimizer: Adam\n')
        info_file.write(f'Learning rate: 0.001\n')
        info_file.write(f'Gamma: 0.9\n')
        info_file.write(f'Total rewards: {total_rewards}\n')

        # Agregar información sobre la estructura seleccionada
        if arch_type == 1:
            structure = "Linear Layers Only"
        elif arch_type == 2:
            structure = "Convolutional Layers"
        elif arch_type == 3:
            structure = "CNN + GRU Layers"
        elif arch_type == 4:
            structure = "CNN + LSTM Layers"
        else:
            structure = "Unknown"

        info_file.write(f'Structure type: {structure}\n')



