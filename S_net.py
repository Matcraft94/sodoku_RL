# Creado por Lucy
# Fecha: 2023/04/08

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

import warnings

# Ignora las advertencias de tiempo de ejecución específicas
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class SudokuModel(nn.Module):
    def __init__(self, arch_type):
        super(SudokuModel, self).__init__()

        self.arch_type = arch_type

        # Fully connected
        if arch_type == 1:
            self.fc1 = nn.Linear(81, 128)
            self.fc2 = nn.Linear(128, 81 * 9)

        # Convolutional
        elif arch_type == 2:
            self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(128 * 9, 128)
            self.extra_layer = nn.Linear(128, 128)  # Agrega una capa adicional
            self.fc2 = nn.Linear(128, 81 * 9)

        # GRU
        elif arch_type == 3:
            self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.gru = nn.GRU(128, 256, batch_first=True)
            self.fc1 = nn.Linear(256, 128)
            self.fc2 = nn.Linear(128, 81 * 9)

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
            return x.view(-1, 9, 9, 9)

        elif self.arch_type == 2:
            x = x.view(-1, 1, 9, 9)
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = x.view(-1, 128 * 9)
            x = torch.relu(self.fc1(x))
            x = nn.functional.relu(self.extra_layer(x))
            x = self.fc2(x)
            return x.view(-1, 9, 9, 9)

        elif self.arch_type == 3:
            x = x.view(-1, 1, 9, 9)
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = x.view(-1, 9, 128)
            _, h_n = self.gru(x)
            x = h_n[-1]
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x.view(-1, 9, 9, 9)

        elif self.arch_type == 4:
            x = x.view(-1, 1, 9, 9)
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = x.view(-1, 9, 128)
            x, (h_n, _) = self.lstm(x)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x.view(-1, 9, 9, 9)
            
            
class Agent:
    def __init__(self, net, optimizer, gamma=0.9):
        self.net = net.to(device)
        self.optimizer = optimizer
        self.gamma = gamma
        self.total_reward = 0

    # def get_action(self, state, episode, epsilon=0.1):
    #     if not episode:
    #         episode = np.random.uniform(10, 20)
    #     if np.random.uniform() > epsilon:
    #         state_tensor = torch.FloatTensor(state).to(device)
    #         q_values = self.net(state_tensor).data.cpu().numpy()
    #         valid_actions = np.argwhere(state == 0)
    #         valid_q_values = []
    #         valid_action_value_pairs = []

    #         for r, c in valid_actions:
    #             for v in range(9):
    #                 valid_q_values.append(q_values[0, r, c, v])
    #                 valid_action_value_pairs.append(((r, c), v + 1))

    #         action_index = np.argmax(valid_q_values)
    #         action, value = valid_action_value_pairs[action_index]
    #         return action, value
    #     else:
    #         valid_actions = np.argwhere(state == 0)
    #         action_index = np.random.randint(len(valid_actions))
    #         action = tuple(valid_actions[action_index])
    #         value = np.random.randint(1, 10)
    #         return action, value

    # def get_action(self, state, episode):
    #     state = torch.FloatTensor(state).unsqueeze(0).to(device)
    #     q_values = self.net(state).data.cpu().numpy()

    #     valid_actions = self.get_valid_actions(state)
    #     valid_q_values = q_values[0, valid_actions[:, 0], valid_actions[:, 1], valid_actions[:, 2]]

    #     if len(valid_q_values) == 0:  # Verifica si hay acciones válidas
    #         return None, None

    #     if np.random.rand() < self.epsilon(episode):
    #         action_index = np.random.randint(len(valid_actions))
    #     else:
    #         action_index = np.argmax(valid_q_values)

    #     action = tuple(valid_actions[action_index])
    #     value = action[2] + 1

    #     return action, value

    def get_valid_actions(self, state):
        state = state.squeeze()
        empty_cells = np.argwhere(state == 0)
        valid_actions = []

        for cell in empty_cells:
            row, col = cell
            for value in range(9):
                if self.is_valid_move(state, row, col, value + 1):
                    valid_actions.append([row, col, value])

        return np.array(valid_actions)
    
    def get_action(self, state, episode):
        epsilon = max(0.1, 1.0 - 0.0001 * episode)
        valid_actions = self.get_valid_actions(state)

        if len(valid_actions) == 0:  # Agrega esta línea
            return None, None  # Agrega esta línea

        if np.random.rand() < epsilon:
            action_index = np.random.randint(len(valid_actions))
            action, value = valid_actions[action_index][:2], valid_actions[action_index][2]
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.net(state_tensor).data.cpu().numpy()
            valid_q_values = q_values[0, valid_actions[:, 0], valid_actions[:, 1], valid_actions[:, 2]]
            action_index = np.argmax(valid_q_values)
            action, value = valid_actions[action_index][:2], valid_actions[action_index][2]

        return action, value + 1



    # def update(self, state, action, value, reward, next_state):
    def update(self, state, action, value, reward, next_state, next_action, next_value):
        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        q_values = self.net(state).data.cpu().numpy()
        next_q_values = self.net(next_state).data.cpu().numpy()
        # q_values[0, action[0], action[1], value - 1] = reward + self.gamma * np.max(next_q_values)
        q_values[0, action[0], action[1], value - 1] = reward + self.gamma * next_q_values[0, next_action[0], next_action[1], next_value - 1]
        q_values = torch.FloatTensor(q_values).to(device)
        loss = nn.MSELoss()(self.net(state), q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.total_reward += reward
        return loss.item()
    
    @staticmethod
    def is_valid_move(board, row, col, value):
        box_row = (row // 3) * 3
        box_col = (col // 3) * 3

        if value in board[row] or value in board[:, col]:
            return False

        for i in range(3):
            for j in range(3):
                if board[box_row + i][box_col + j] == value:
                    return False

        return True


    
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

    # def update(self, state, action, reward, next_state):
    #     state = torch.FloatTensor(state).to(device)
    #     next_state = torch.FloatTensor(next_state).to(device)
    #     q_values = self.net(state).data.cpu().numpy()
    #     next_q_values = self.net(next_state).data.cpu().numpy()
    #     q_values[0, action[0], action[1]] = reward + self.gamma * np.max(next_q_values)
    #     q_values = torch.FloatTensor(q_values).to(device)
    #     loss = nn.MSELoss()(self.net(state), q_values)
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
    #     self.total_reward += reward
    
    # # def update(self, state, action, value, reward, next_state):
    # def update(self, state, action, value, reward, next_state, next_action, next_value):
    #     state = torch.FloatTensor(state).to(device)
    #     next_state = torch.FloatTensor(next_state).to(device)
    #     q_values = self.net(state).data.cpu().numpy()
    #     next_q_values = self.net(next_state).data.cpu().numpy()
    #     # q_values[0, action[0], action[1], value - 1] = reward + self.gamma * np.max(next_q_values)
    #     q_values[0, action[0], action[1], value - 1] = reward + self.gamma * next_q_values[0, next_action[0], next_action[1], next_value - 1]
    #     q_values = torch.FloatTensor(q_values).to(device)
    #     loss = nn.MSELoss()(self.net(state), q_values)
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
    #     self.total_reward += reward
    #     return loss.item()
    
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
    
    # def is_valid_move(self, row, col, value):
    #     # Check row
    #     if value in self.board[row]:
    #         return False

    #     # Check column
    #     if value in self.board[:, col]:
    #         return False

    #     # Check subgrid
    #     subgrid_row, subgrid_col = row // 3, col // 3
    #     for i in range(3):
    #         for j in range(3):
    #             if self.board[subgrid_row * 3 + i][subgrid_col * 3 + j] == value:
    #                 return False
    #     return True

    def step(self, action, value):
        row, col = action
        if self.board[row][col] == 0:
            if self.is_valid(self.board, row, col, value):
                reward = 1 #                                                        Semodifica este para el reward 5
            else:
                reward = -1
            self.board[row][col] = value
        else:
            reward = -1
        done = (self.board == 0).sum() == 0
        return self.board, reward, done

    @staticmethod
    def is_valid(grid, row, col, value):
        # Check row
        if value in grid[row]:
            return False
        # Check column
        if value in grid[:, col]:
            return False
        # Check 3x3 box
        box_row, box_col = row // 3 * 3, col // 3 * 3
        if value in grid[box_row:box_row + 3, box_col:box_col + 3]:
            return False
        return True
    
    # def step(self, action):
    #     row, col = action
    #     if self.board[row][col] == 0:
    #         reward = 1
    #         self.board[row][col] = np.random.randint(1, 10)
    #     else:
    #         reward = -1
    #     done = (self.board == 0).sum() == 0
    #     return self.board, reward, done

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

    arch_type = 2
    
    env = SudokuEnv()
    net = SudokuModel(arch_type).to(device)
    input_size = (1, 9, 9, 9)
    summary(net, input_size)

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    agent = Agent(net, optimizer)
    total_rewards = []
    losses = []  # Asumiendo que también quieres guardar las pérdidas por episodio

    for i in trange(1000, desc="Training", ncols=100):
        state = env.reset()
        done = False
        episode_losses = []
        # # # while not done:
        # # #     action, value = agent.get_action(state, i)
        # # #     next_state, reward, done = env.step(action, value)
        # # #     loss = agent.update(state, action, value, reward, next_state)
        # # #     episode_losses.append(loss)
        # # #     state = next_state
        # # while not done:
        # #     action, value = agent.get_action(state, i)
        # #     next_state, reward, done = env.step(action, value)
        # #     next_action, next_value = agent.get_action(next_state, i)  # Obtén la siguiente acción y el siguiente valor
        # #     agent.update(state, action, value, reward, next_state, next_action, next_value)
        # #     state = next_state
        # while not done:
        #     action, value = agent.get_action(state, i)
        #     next_state, reward, done = env.step(action, value)
        #     next_action, next_value = agent.get_action(next_state, i)  # Obtén la siguiente acción y el siguiente valor

        #     if next_action is not None and next_value is not None:
        #         agent.update(state, action, value, reward, next_state, next_action, next_value)
            
        #     state = next_state
        while not done:
            action, value = agent.get_action(state, i)
            if action is None and value is None:  # Agrega esta línea
                break  # Agrega esta línea
            next_state, reward, done = env.step(action, value)
            next_action, next_value = agent.get_action(next_state, i)

            if next_action is not None and next_value is not None:
                agent.update(state, action, value, reward, next_state, next_action, next_value)

            state = next_state

        total_rewards.append(agent.total_reward)
        mean_episode_loss = np.mean(episode_losses)
        losses.append(mean_episode_loss)
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



