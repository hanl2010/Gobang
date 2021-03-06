from board import Board
from mcts import MCTS
from mcts_with_policy_fn import MCTS_WITH_POLICY
import pygame
import random
from evaluate_method import Evaluate
from neural_network import PolicyNet
import numpy as np
import sys
import time


class Game:
    def __init__(self, board_size=15, n_dots=5, model_file=None, board_show=False):
        self.board = Board(board_size=board_size, n_dots=n_dots)
        self.board_show = board_show
        self.policy_network = PolicyNet(board_size=board_size, use_gpu=False, model_file=model_file)
        self.screen_size = (640, 640)
        self.evaluate_times = 10

        if board_show:
            # pygame.init()
            # pygame.display.set_caption("五子棋")
            # self.screen = pygame.display.set_mode(size=self.screen_size)
            # self.board.draw_board(self.screen, self.screen_size)
            # pygame.event.set_allowed((pygame.MOUSEBUTTONDOWN, pygame.QUIT))
            self.board.print_board()

    def human_play(self):
        # for event in pygame.event.get():
        while True:
            event = pygame.event.wait()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                x, y = event.pos
                action = (round((y-self.board.h_grid_size)/self.board.h_grid_size),
                          round((x-self.board.w_grid_size)/self.board.w_grid_size))
                self.board.move(action)
                break
            elif event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def random_play(self):
        action = random.choice(self.board.get_valid_move_pos())
        self.board.move(action)

    def mcts_play(self):
        mcts = MCTS(c_param=5, n_rollout=2000)
        actions_index, actions_prob = mcts.simulation(self.board)
        self.policy_network.collect_temp_data(self.board, actions_index, actions_prob)
        action = mcts.get_move()
        # mcts.print_first_level_node()
        print("pure mcts action", action)
        self.board.move(action)

    def network_play(self):
        ## 两种走棋方案：
        ## 1. 使用神经网络得到的action probability
        # actions, act_probs, value = self.policy_network.get_policy_value(self.board)
        # action_index = np.argmax(act_probs)
        # self.board.move(actions[action_index])
        ## 2. 使用神经网络 和 MCST 模拟若干次，通过MCST得到action probability
        mcts = MCTS_WITH_POLICY(policy_func=self.policy_network.get_policy_value, c_param=5, n_rollout=500)
        actions_index, act_probs = mcts.simulation(self.board)
        # print(act_probs)
        # actions_index, act_probs = mcts.simulation_parallel(self.board) # 并行方法有问题
        self.policy_network.collect_temp_data(board=self.board, actions_index=actions_index, act_probs=act_probs)
        action = mcts.get_move()
        print("policy mcts action", action)
        self.board.move(action)

    def evaluate_play(self):
        eva = Evaluate(self.board)
        action = eva.evaluate_and_get_move()
        print("evaluate action", action)
        self.board.move(action)

    def evaluate(self, black_strategy, white_strategy):
        black_wins = 0.0
        white_wins = 0.0
        draws = 0.0
        for i in range(self.evaluate_times):
            print("{} times evaluate".format(i))
            while self.board.get_game_result() == self.board.RESULT_NOT_OVER:
                if self.board.get_turn() == self.board.CELL_BLACK:
                    black_strategy()
                else:
                    white_strategy()
                if self.board_show:
                    self.board.print_board()
                print("evaluate times:{}, black wins:{}, white wins:{}, draws:{}".format(i, black_wins, white_wins,
                                                                                         draws))
            result = self.board.get_game_result()
            self.board.reset()
            if result==self.board.RESULT_BLACK_WIN:
                black_wins += 1
            elif result==self.board.RESULT_WHITE_WIN:
                white_wins += 1
            else:
                draws += 1
            print("evaluate times:{}, black wins:{}, white wins:{}, draws:{}".format(i, black_wins, white_wins, draws))
        return (black_wins+0.5*draws)/self.evaluate_times


    def play_game(self, black_strategy, white_strategy, i, results, use_network=False):

        while self.board.get_game_result() == self.board.RESULT_NOT_OVER:
            print("######## black wins: {}/{}, white_wins: {}/{}, draws: {} #########".format(
                                                sum(np.array(results)==-1), len(results),
                                                sum(np.array(results)==1), len(results),
                                                sum(np.array(results)==0)
            ))
            if self.board.get_turn() == self.board.CELL_BLACK:
                print("{} times play, black turn:".format(i))
                black_strategy()
            else:
                print("{} times play, white turn:".format(i))
                white_strategy()
            if self.board_show:
                # 更新子进程中的board
                # self.board.draw_piece(self.screen)
                # for event in pygame.event.get():
                #     if event.type == pygame.QUIT:
                #         pygame.quit()
                #         sys.exit()
                self.board.print_board()

        result = self.board.get_game_result()
        if use_network:
            self.policy_network.calc_and_collect_final_data(result)
            ## train network
            losses = self.policy_network.train()
            print("avg_loss: {}".format(np.mean(losses)))
        # 等待X秒之后，重置棋盘
        time.sleep(2)
        ## reset board
        self.board.reset()
        # if self.board_show:
        #     self.board.draw_board(self.screen, self.screen_size)
        return result


if __name__ == '__main__':
    board_size = 9
    n_dots = 5
    game = Game(board_size=board_size, n_dots=n_dots, model_file="saved_model/current_policy.model", board_show=True)
    results = []
    i = 0
    best_win_ratio = 0
    use_network = True
    while True:
        i += 1
        result = game.play_game(game.network_play, game.network_play, i, results, use_network=use_network)
        results.append(result)

        if i%10 == 0 and use_network==True:
            game.policy_network.save_model("current_policy.model")
            win_ratio = game.evaluate(game.network_play, game.mcts_play)
            if win_ratio > best_win_ratio:
                game.policy_network.save_model("best_policy_{}_{}.model".format(board_size,n_dots))


