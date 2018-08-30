import numpy as np
from copy import copy as copy

class ConnectFour():
    def __init__(self):
        self.game_field = np.zeros((6,7), dtype=int)
        self.game_history = np.array([self.game_field])
        self.player1 = 1
        self.player2 = 2

    def reset(self):
        self.game_field = np.zeros((6,7), dtype=int)

    def show_field(self):
        print(self.game_field)

    def return_field(self):
        #returns a copy of the game field
        return copy(self.game_field)

    def return_game_history(self):
        return self.game_history

    def add_field_to_history(self):
        self.game_history = np.append(self.game_history, np.array([self.game_field]),  axis=0)

    def mirror_field(self):
        mirrored_field = self.game_field - (self.game_field == 2) + (self.game_field == 1)  # make twos to ones; ones to twos
        return copy(mirrored_field)

    def mirror_history(self):
        mirrored_hist = copy(np.array([self.game_history[0,:,:]]))
        for i in range(1,self.game_history.shape[0]):
            game = copy(self.game_history[i,:,:])
            game = game - (game == 2) + (game == 1)
            game = np.array([game])
            mirrored_hist = np.append(mirrored_hist, game, axis=0)
        return mirrored_hist

    def move(self, player, column):
        # column in {0,...,6}
        column_entries = self.game_field[:,column][::-1]
        row = np.argmin(column_entries)
        if np.min(column_entries) != 0:
            raise IndexError
        else:
            column_entries[row] = player

    def check_horizontal(self, row, column):

        check_right = (column <= 3)
        check_left = (column >= 3)

        count = 1
        winner = 0

        a = self.game_field[row,:]

        if check_right:
            for i in range(3):
                winner = self.game_field[row, column + i]
                if self.game_field[row, column + i] == self.game_field[row, column + 1 + i]:
                    count += 1

        found_four = (count == 4 and winner != 0)

        if found_four:
            return found_four, winner


        count = 1

        if check_left:
            for i in range(3):
                winner = self.game_field[row, column - i]
                if self.game_field[row, column - i] == self.game_field[row, column - 1 - i]:
                    count += 1

        found_four = (count == 4 and winner != 0)


        return found_four, winner

    def check_diagonal(self, row, column):

        check_upright   = (row >= 3) and (column <= 3)
        check_upleft    = (row >= 3) and (column >= 3)

        check_downright = (row <= 2) and (column <= 3)
        check_downleft  = (row <= 2) and (column >= 3)

        count = 1
        winner = 0

        def found_four():
            return count == 4 and winner != 0

        if check_upright:
            for i in range(3):
                winner = self.game_field[row - i, column + i]
                if self.game_field[row - i, column + i] == self.game_field[row - i - 1, column + 1 + i]:
                    count += 1

        if found_four():
            return True, winner

        count = 1

        if check_upleft:
            for i in range(3):
                winner = self.game_field[row - i, column - i]
                if self.game_field[row - i, column - i] == self.game_field[row - i - 1, column - 1 - i]:
                    count += 1

        if found_four():
            return True, winner

        count = 1

        if check_downright:
            for i in range(3):
                winner = self.game_field[row + i, column + i]
                if self.game_field[row + i, column + i] == self.game_field[row + i + 1, column + 1 + i]:
                    count += 1

        if found_four():
            return True, winner

        count = 1

        if check_downleft:
            for i in range(3):
                winner = self.game_field[row + i, column - i]
                if self.game_field[row + i, column - i] == self.game_field[row + i + 1, column - 1 - i]:
                    count += 1

        if found_four():
            return True, winner

        count = 1

        return False, winner

    def check_vertical(self, row, column):
        check_down = (row <= 2)

        count = 1
        winner = 0
        found_four = False

        if check_down:
            for i in range(3):
                winner = self.game_field[row + i, column]
                if self.game_field[row + i, column] == self.game_field[row + i + 1, column]:
                    count += 1

        found_four = (count == 4 and winner != 0)

        return found_four, winner



    def check_four(self):

        check_list = [self.check_horizontal, self.check_vertical, self.check_diagonal]

        for row in range(6):
            for column in range(7):
                for direction in check_list:
                    found_four, winner = direction(row, column)
                    if found_four:
                        print("Winner is: " + str(winner) + "; Position: " + str((row,column)))
                        return found_four
        return False

    def request_move(self, player, action):
        try:
            self.move(player, action)
            self.add_field_to_history()
        except IndexError:
            print("invalid move")

    def run_game(self, agent1, agent2, print_steps=False):
        move_count = 0
        game_done = False
        while not game_done and move_count <= 41: # <= 41 since starting at 0
            player = move_count % 2 + 1

            if player == 1:
                agent1.get_observation(self.game_field)
                action = agent1.get_action()
            else:
                agent2.get_observation(self.mirror_field())
                action = agent2.get_action()

            self.request_move(player, action)

            move_count += 1

            game_done = self.check_four()

            print(self.game_field)

        print("Game finished")


if __name__ == "__main__":
    from ConnectFour.agent import RandomAgent

    Game = ConnectFour()
    Agent1 = RandomAgent(state=Game.game_field)
    Agent2 = RandomAgent(state=Game.game_field)
    Game.run_game(Agent1, Agent2)
    print(Game.game_history)
    print(Game.mirror_history())



