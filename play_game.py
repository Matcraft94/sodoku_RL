# Creado por Lucy
# Fecha: 2023/04/08

# import necessary libraries
import torch
import numpy as np
import random

# define the Sudoku game environment
class Sudoku:
    def __init__(self, difficulty):
        self.difficulty = difficulty
        self.grid = self.generate_grid()
        self.solution = self.solve(self.grid)

    # generate the Sudoku grid
    def generate_grid(self):
        if self.difficulty == 'Easy':
            num_cells = random.randint(40, 50)
        elif self.difficulty == 'Medium':
            num_cells = random.randint(30, 40)
        elif self.difficulty == 'Hard':
            num_cells = random.randint(20, 30)

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
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.build_model().to(self.device)

    # # build the neural network model
    # def build_model(self):
    #     model = torch.nn.Sequential(
    #         torch.nn.Linear(81, 128),
    #         torch.nn.ReLU(),
    #         torch.nn.Linear(128, 81),
    #         torch.nn.Sigmoid()
    #     )
    #     return model
    # build the neural network model
    def build_model(self):
        # Load the trained model from a file
        model_file = "results/sudoku_RL_2023-04-22_23-38-15.pth"
        model = torch.load(model_file)
        return model
    
    # play the Sudoku game
    # def play_game(self):
    #     print("Welcome to Sudoku!")
    #     difficulty = input("Please select the difficulty level: Easy, Medium, or Hard: ")
    #     game = Sudoku(difficulty)

    #     while True:
    #         print("Current game:")
    #         game.display()
    #         row = int(input("Please enter the row number (1-9): ")) - 1
    #         col = int(input("Please enter the column number (1-9): ")) - 1
    #         value = int(input("Please enter the value (1-9): "))
    #         if game.grid[row][col] != 0:
    #             print("That cell is already filled!")
    #         elif not game.is_valid(game.grid, row, col, value):
    #             print("Invalid move!")
    #         else:
    #             game.grid[row][col] = value
    #             if game.grid == game.solution:
    #                 print("Congratulations! You have solved the Sudoku!")
    #                 return
    #             elif all([all(row) for row in game.grid]):
    #                 print("Sorry, you have failed to solve the Sudoku.")
    #                 return

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
    def get_prediction(self, predicted_grid, current_grid):
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

    
    
# create an instance of the AgentGPT class and play the game
agent = SUdokuNet()
agent.play_game()