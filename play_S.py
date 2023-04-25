# Creado por Lucy
# Fecha: 2023/04/29

import time
import torch
import numpy as np
import random
import tkinter as tk
from copy import deepcopy

# define the Sudoku game environment
class Sudoku:
    def __init__(self, difficulty):
        self.difficulty = difficulty
        self.grid = self.generate_grid()

    def generate_grid(self):
        # Modify the code below to assign a value to num_cells based on the difficulty
        if self.difficulty.lower() == 'easy':
            num_cells = 12
        elif self.difficulty.lower() == 'medium':
            num_cells = 24
        elif self.difficulty.lower() == 'hard':
            num_cells = 35
        else:
            raise ValueError("Invalid difficulty level")

        grid = [[0]*9 for _ in range(9)]
        for i in range(num_cells):
            row = random.randint(0, 8)
            col = random.randint(0, 8)
            while grid[row][col] != 0:
                row = random.randint(0, 8)
                col = random.randint(0, 8)
            value = random.randint(1, 9)
            if self.is_valid(grid, row, col, value):
                grid[row][col] = value
            else:
                i -= 1
        return grid

    # check if a value is valid in a particular cell
    def is_valid(self, grid, row, col, value):
        for i in range(9):
            if grid[row][i] == value or grid[i][col] == value:
                return False
        row_start = (row // 3) * 3
        col_start = (col // 3) * 3
        for i in range(row_start, row_start + 3):
            for j in range(col_start, col_start + 3):
                if grid[i][j] == value:
                    return False
        return True

    # solve the Sudoku using backtracking
    def solve(self, grid):
        for row in range(9):
            for col in range(9):
                if grid[row][col] == 0:
                    for value in range(1, 10):
                        if self.is_valid(grid, row, col, value):
                            grid[row][col] = value
                            if self.solve(grid):
                                return grid
                            else:
                                grid[row][col] = 0
                    return False
        return grid

    # display the Sudoku grid
    def display(self):
        for i in range(9):
            for j in range(9):
                print(self.grid[i][j], end=' ')
                if j == 2 or j == 5:
                    print('|', end=' ')
            print()
            if i == 2 or i == 5:
                print('---------------------')
        print()


# define the SUdokuNet class
class SUdokuNet:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_path is not None:
            self.model = self.load_model(model_path)
        else:
            self.model = self.build_model()
        self.model.to(self.device)

    def build_model(self):
        class SudokuModel(torch.nn.Module):
            def __init__(self):
                super(SudokuModel, self).__init__()
                self.fc1 = torch.nn.Linear(81, 128)
                self.fc2 = torch.nn.Linear(128, 81)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.sigmoid(self.fc2(x))
                x = x.view(-1, 9, 9)
                return x

        return SudokuModel()


    def load_model(self, model_path):
        model = self.build_model()
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        return model

    # play the Sudoku game
    def play_game(self):
        print("Welcome to Sudoku!")
        difficulty = input("Please select the difficulty level: Easy, Medium, or Hard: ")
        game = Sudoku(difficulty)

        while True:
            print("Current game:")
            game.display()

            # Get the model's prediction
            input_grid = torch.tensor(game.grid, dtype=torch.float32).flatten().to(self.device)
            predicted_grid = self.model(input_grid).detach().cpu().numpy().reshape(9, 9)
            row, col, value = self.get_prediction(predicted_grid, game.grid)

            if game.grid[row][col] != 0:
                print("That cell is already filled!")
            elif not game.is_valid(game.grid, row, col, value):
                print("Invalid move!")
            else:
                game.grid[row][col] = value
                if game.grid == game.solution:
                    print("Congratulations! The AI has solved the Sudoku!")
                    return
                elif all([all(row) for row in game.grid]):
                    print("Sorry, the AI has failed to solve the Sudoku.")
                    return

    # get the prediction from the model
    def get_sudoku_prediction(self, predicted_grid, current_grid):
        max_diff = 0
        row, col, value = -1, -1, -1

        for i in range(9):
            for j in range(9):
                if current_grid[i][j] == 0:
                    diff = predicted_grid[i][j] - current_grid[i][j]
                    if diff > max_diff:
                        max_diff = diff
                        row, col, value = i, j, int(round(predicted_grid[i][j]))

        return row, col, value

# CHATSUDOKU
class ChatSudokuSolver:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sudoku_solver = SUdokuNet(model_path=model_path)

    # def load_model(self, model_path):
    #     state_dict = torch.load(model_path, map_location=self.device)
    #     self.sudoku_solver.model.load_state_dict(state_dict)
    #     self.sudoku_solver.model.to(self.device)

    def chat_play_game(self):
        response = "Welcome to Sudoku Solver!\n"
        difficulty = input("Please select the difficulty level: Easy, Medium, or Hard: ")
        game = Sudoku(difficulty)

        while True:
            response += "Current game:\n" + self.format_grid(game.grid) + "\n"
            row, col, value = self.sudoku_solver.get_prediction(game.grid)

            if game.grid[row][col] != 0:
                response += "That cell is already filled!\n"
            elif not game.is_valid(game.grid, row, col, value):
                response += "Invalid move!\n"
            else:
                game.grid[row][col] = value
                if game.grid == game.solution:
                    response += "Congratulations! The AI has solved the Sudoku!\n"
                    break
                elif all([all(row) for row in game.grid]):
                    response += "Sorry, the AI has failed to solve the Sudoku.\n"
                    break

        return response

    def format_grid(self, grid):
        formatted = ""
        for i in range(9):
            for j in range(9):
                formatted += str(grid[i][j]) + ' '
                if j == 2 or j == 5:
                    formatted += '| '
            formatted += '\n'
            if i == 2 or i == 5:
                formatted += '---------------------\n'
        return formatted
    
    # def get_prediction(self, grid):
    #     input_grid = torch.tensor(grid, dtype=torch.float32).flatten().to(self.device)
    #     predicted_grid = self.sudoku_solver.model(input_grid).detach().cpu().numpy().reshape(9, 9)
    #     row, col, value = self.sudoku_solver.get_sudoku_prediction(predicted_grid, grid)
    #     return row, col, value
    def get_prediction(self, grid, game):
        input_grid = torch.tensor(grid, dtype=torch.float32).flatten().to(self.device)
        predicted_grid = self.sudoku_solver.model(input_grid).detach().cpu().numpy().reshape(9, 9)
        row, col, value = self.sudoku_solver.get_sudoku_prediction(predicted_grid, grid)

        # Verifica que el valor predicho sea válido antes de retornarlo
        while not game.is_valid(grid, row, col, value):
            value = (value % 9) + 1

        return row, col, value



class SudokuGUI:
    def __init__(self, chat_solver):
        self.chat_solver = chat_solver
        self.root = tk.Tk()
        self.root.title("Sudoku Solver")
        self.grid_cells = [[None for _ in range(9)] for _ in range(9)]
        self.grid_history = []
        self.current_step = -1

        self.build_gui()
        self.root.mainloop()

    def build_gui(self):
        frame = tk.Frame(self.root, padx=10, pady=10)
        frame.pack()

        for i in range(9):
            for j in range(9):
                cell = tk.Entry(frame, width=3, font=("Arial", 14), justify="center")
                cell.grid(row=i, column=j, padx=2, pady=2)

                if i % 3 == 2 and i != 8:
                    cell.grid_configure(pady=(2, 6))
                if j % 3 == 2 and j != 8:
                    cell.grid_configure(padx=(2, 6))

                self.grid_cells[i][j] = cell

        button_frame = tk.Frame(self.root, padx=10, pady=10)
        button_frame.pack()

        start_button = tk.Button(button_frame, text="Start", command=self.start_game)
        start_button.grid(row=0, column=0, padx=5)

        back_button = tk.Button(button_frame, text="Back", command=self.back_step)
        back_button.grid(row=0, column=1, padx=5)

    # def start_game(self):
    #     difficulty = input("Please select the difficulty level: Easy, Medium, or Hard: ")
    #     game = Sudoku(difficulty)
    #     self.game = game
    #     self.grid_history = [deepcopy(game.grid)]
    #     self.current_step = 0
    #     self.update_grid()

    #     while not self.is_solved():
    #         valid_move = False
    #         while not valid_move:
    #             row, col, value = self.chat_solver.get_prediction(game.grid)
    #             if game.is_valid(game.grid, row, col, value):  # Comprueba si el movimiento es válido
    #                 valid_move = True
    #             else:
    #                 # Cambia de estrategia: explora diferentes valores en la misma celda
    #                 game.grid[row][col] = (game.grid[row][col] % 9) + 1

    #         game.grid[row][col] = value
    #         self.grid_history.append(deepcopy(game.grid))
    #         self.current_step += 1
    #         self.update_grid()
    #         self.root.update_idletasks()
    #         self.root.update()
    #         time.sleep(0.1)
    def start_game(self):
        difficulty = input("Please select the difficulty level: Easy, Medium, or Hard: ")
        game = Sudoku(difficulty)
        self.game = game
        self.grid_history = [deepcopy(game.grid)]
        self.current_step = 0
        self.update_grid()

        while not self.is_solved():
            valid_move = False
            while not valid_move:
                row, col, value = self.chat_solver.get_prediction(game.grid, game)
                if game.is_valid(game.grid, row, col, value):  # Comprueba si el movimiento es válido
                    valid_move = True
                else:
                    # Cambia de estrategia: explora diferentes valores en la misma celda
                    game.grid[row][col] = (game.grid[row][col] % 9) + 1

            game.grid[row][col] = value
            self.grid_history.append(deepcopy(game.grid))
            self.current_step += 1
            self.update_grid()
            self.root.update_idletasks()
            self.root.update()
            time.sleep(0.1)


    def back_step(self):
        if self.current_step > 0:
            self.current_step -= 1
            self.update_grid()

    def update_grid(self):
        for i in range(9):
            for j in range(9):
                value = self.grid_history[self.current_step][i][j]
                self.grid_cells[i][j].delete(0, tk.END)

                if value != 0:
                    self.grid_cells[i][j].insert(tk.END, str(value))

                    if self.grid_history[0][i][j] != 0:
                        self.grid_cells[i][j].configure(fg="blue")
                    else:
                        self.grid_cells[i][j].configure(fg="black")

    def is_solved(self):
        return all(value != 0 for row in self.game.grid for value in row)


# Load the pre-trained model
model_path = "./results/sudoku_RL_2023-04-22_23-38-15.pth"

# Create an instance of the ChatSudokuSolver class
chat_solver = ChatSudokuSolver(model_path)

# Create the Sudoku GUI and start the main loop
sudoku_gui = SudokuGUI(chat_solver)
