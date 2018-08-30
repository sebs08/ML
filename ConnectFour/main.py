from ConnectFour.agent import CNNAgent, RandomAgent
from ConnectFour.game import ConnectFour

def run():
    # initialize the Game
    Game = ConnectFour()

    # initialize Agents
    Agent1 = CNNAgent(state=Game.game_field)
    Agent2 = RandomAgent(state=Game.game_field)

    # run a game
    Game.run_game(Agent1, Agent2, print_steps=True)





if __name__ == "__main__":
    run()