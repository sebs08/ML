# 4 wins engine
from netz import NeuroNet
import numpy as np
from connect4 import *
import time

class RecordGame():
    def __init__(self):
        self.game_history = np.array(()).reshape((0,42))


    def add_field(self, field):
        """

        :param field: vector which contains the field data
        :return: nothing
        """
        self.game_history = np.vstack([self.game_history, np.array(field)])

    def save_trace(self, winner):
        # add who won
        self.game_history = np.hstack([self.game_history, winner * np.ones((self.game_history.shape[0], 1))])

        output_name = time.strftime('%d_%m_%y_%H%M%S')
        np.savetxt(output_name, self.game_history.astype(int), fmt='%i', delimiter=',')

    def reset_game(self):
        self.game_history = np.array(()).reshape((0,42))


def play_game(game, net):
    game.newGame() # initialize new game
    record_game = RecordGame() # initialize recording

    while not game.finished: # play game until finished
        record_game.add_field(game.return_board_as_array()) # add field to record
        game.nextMove() # make next move

        game.findFours() # check if over


    record_game.add_field(game.return_board_as_array())

    if game.winner == game.players[0]:
        net_win = True
    else:
        net_win = False

    return record_game.game_history, net_win


def play_game_batch(game, net, size):
    count = 0
    games = 0
    won_games = []

    while count < size:
        print("game " + str(games) + "; won till now " + str(count))
        history, winner = play_game(game, net)
        if winner:
            count += 1
            won_games.append(history)
        games += 1

    return won_games

def play_game_batch2(game, net, size):
    games = []
    winner_of_game = []

    while len(games) < size:
        print("game  " + str(len(games)))
        history, winner = play_game(game, net)
        games.append(history)
        if winner:
            winner_of_game.append(1)
        else:
            winner_of_game.append(-0.2)

    return games, winner_of_game



def make_move(net,field, epsilon):
    # make move following epsilon greedy strategy
    move = net.get_move(field)
    if np.random.rand(1)[0] < epsilon:
        return np.random.randint(0,7,1)[0]
    else:
        return move

def run_engine(save=True):
    print("load net...")
    four_wins_net = NeuroNet(42, 7, 7, load_weights=True, save_weights=True)
    print("net loaded")

    print("start game...")
    game = Game(lambda x: make_move(four_wins_net, x, 0.2))
    print("game started")

    print("play games...")
    play_game(game,four_wins_net)

    """
    runs = 0
    while runs <= 100:
        back_prob_games, winner_of_games = play_game_batch2(game, four_wins_net, 50)
        print("doing backprob")
        for i in range(len(back_prob_games)-1):
            four_wins_net.train_multi(back_prob_games[i], winner_of_games[i])
        runs += 1
        print(str(100-runs) + " to go")
    """






if __name__ == "__main__":
    run_engine()