import numpy as np


class Environment:
    def __init__(self):
        self.observation_space = 9
        self.action_space = 9
        self.player = 1
        self.empty = 0
        self.ai = -1
        self.draw = 0
        self.winning_sequence = self.draw
        self.board_row = 3
        self.board_col = 3
        self.board = np.zeros((self.board_row, self.board_col))
        self.turn = self.player

    def evaluate(self):
        "Check if player won the game and record the winning sequence"
        "return : (done , reward)"
        if np.all(self.board[0, :] == self.player) or np.all(self.board[1, :] == self.player) or np.all(
                self.board[2, :] == self.player):
            return (True, self.player * self.turn)
        elif np.all(self.board[:, 0] == self.player) or np.all(self.board[:, 1] == self.player) or np.all(
                self.board[:, 2] == self.player):
            return (True, self.player * self.turn)
        elif np.all(self.board.diagonal() == self.player) or np.all(np.fliplr(self.board).diagonal() == self.player):
            return (True, self.player * self.turn)

        elif np.all(self.board[0, :] == self.ai) or np.all(self.board[1, :] == self.ai) or np.all(
                self.board[2, :] == self.ai):
            return (True, self.ai * self.turn)
        elif np.all(self.board[:, 0] == self.ai) or np.all(self.board[:, 1] == self.ai) or np.all(
                self.board[:, 2] == self.ai):
            return (True, self.ai * self.turn)
        elif np.all(self.board.diagonal() == self.ai) or np.all(np.fliplr(self.board).diagonal() == self.ai):
            return (True, self.ai * self.turn)
        else:
            return (self.all_fill(), self.draw)

    def reset(self):
        self.board = np.zeros((self.board_row, self.board_col))
        return self.board

    def step(self, action):
        self.board[int(action / 3), int(action % 3)] = self.turn
        done, reward = self.evaluate()
        return self.board, reward, done, None

    def available(self):
        return (self.board == self.empty).reshape((self.observation_space))

    def all_fill(self):
        return self.board[self.board == self.empty].shape[0] == 0

    def print_board(self):
        print("   |   |   ")
        print(" " + str(int(self.board[0, 0])) + " | " + str(int(self.board[0, 1])) + " | " + str(
            int(self.board[0, 2])) + "  ")
        print("   |   |")
        print("---|---|---")
        print("   |   |")
        print(" " + str(int(self.board[1, 0])) + " | " + str(int(self.board[1, 1])) + " | " + str(
            int(self.board[1, 2])) + "  ")
        print("   |   |")
        print("---|---|---")
        print("   |   |")
        print(" " + str(int(self.board[2, 0])) + " | " + str(int(self.board[2, 1])) + " | " + str(
            int(self.board[2, 2])) + "  ")
        print("   |   |   ")
