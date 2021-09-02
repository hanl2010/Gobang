import numpy as np
import copy
import random
# import sys
import os
from multiprocessing import Pool, Manager
import time

# sys.setrecursionlimit(10000)

class Node:
    def __init__(self, parent, prior_prob):
        self.parent = parent
        self.children = {}
        self.n_wins = 0.0
        self.n_visits = 0.0
        self.P = prior_prob
        self.Q = 0.0

    def select_child(self, c_param):
        return max(self.children.items(), key=lambda action_node : action_node[1].get_UCB_value(c_param))

    def get_UCB_value(self, c_param):
        return self.Q + c_param*self.P*np.sqrt(self.parent.n_visits)/(1+self.n_visits)
        # return self.Q + np.sqrt(2*np.log(self.parent.n_visits)/(1+self.n_visits))

    def add_child(self, action, prob):
        if action not in self.children:
            self.children[action] = Node(self, prior_prob=prob)

    def update(self, result):
        self.n_visits += 1
        self.n_wins += result
        self.Q = self.n_wins/self.n_visits
        # self.Q += 1.0*(result - self.Q) / self.n_visits  #junxiaosong的计算方法

    def update_recursion(self, result):
        # If it is not root, this node's parent should be updated first.
        if self.parent:
            self.parent.update_recursion(-result)
        self.update(result)

    def is_leaf(self):
        if self.children == {}:
            return True
        else:
            return False

    def is_root(self):
        if self.parent is None:
            return True
        else:
            return False

    def __repr__(self):
        return "[n_children:" + str(len(self.children)) + " W/V:" + str(self.n_wins) + "/" + str(self.n_visits) + "]"



class MCTS:
    def __init__(self, c_param=5, n_rollout=2000):
        self.root = Node(parent=None, prior_prob=1.0)
        self.c_param = c_param
        self.n_rollout = n_rollout

    def simulation(self, board):
        print("pure msts simulation")
        start = time.time()
        for i in range(self.n_rollout):
            if i%10 == 0:
                print(".", end="")
            node = self.root
            board_copy = copy.deepcopy(board)
            current_player = board_copy.get_turn()
            ## selection
            while not node.is_leaf():
                action, node = node.select_child(self.c_param)
                board_copy.move(action)
            ## expand
            if board_copy.get_game_result() == board_copy.RESULT_NOT_OVER:
                valid_actions = board_copy.get_valid_move_pos()
                probs = np.ones(shape=(len(valid_actions), )) / len(valid_actions)
                for action, prob in zip(valid_actions, probs):
                    node.add_child(action, prob)
            ## rollout
            while board_copy.get_game_result() == board_copy.RESULT_NOT_OVER:
                valid_actions = board_copy.get_valid_move_pos()
                board_copy.move(random.choice(valid_actions))
            ## update
            if board_copy.get_game_result() == board_copy.RESULT_DRAW:
                result = 0
            else:
                result = 1 if board_copy.get_game_result() == current_player else -1
            while node is not None:
                node.update(result)
                node = node.parent
        end = time.time()
        print("Task runs %.2fs" % (end - start))

    def simulation_subprocess(self, board, root, queue, simulation_times, c_param):
        for i in range(simulation_times):
            if i%10 == 0:
                print(".", end="")
            node = root
            board_copy = copy.deepcopy(board)
            current_player = board_copy.get_turn()
            # print(current_player)
            # if node.children != {}:
            #     print(node.children.items())
            # selection
            # print(i, node)
            while not node.is_leaf():
                action, node = node.select_child(c_param)
                board_copy.move(action)
            # expand
            if board_copy.get_game_result() == board_copy.RESULT_NOT_OVER:
                ## 扩展出所有子结点
                valid_actions = board_copy.get_valid_move_pos()
                probs = np.ones(shape=(len(valid_actions),)) / len(valid_actions)
                for action, prob in zip(valid_actions, probs):
                    node.add_child(action, prob)
                ## 或者 随机扩展出一个子结点
                ## 存在的问题： 当只有根结点时，只扩展出一个子结点，下一次进行选择时，只能进入上一次随机扩展出的子结点，就不会去探索其他分支
                # action = random.choice(valid_actions)
                # node.add_child(action, prob=1.0/len(valid_actions))
                ## 使用神经网络对
            # rollout 实际上是从扩展结点的父节点开始往下执行
            # 可以到游戏结束，也可以规定最多执行n次
            while board_copy.get_game_result() == board_copy.RESULT_NOT_OVER:
            # for _ in range(100):
            #     if board_copy.get_game_result() != board_copy.RESULT_NOT_OVER:
            #         break
                valid_actions = board_copy.get_valid_move_pos()
                # if board_copy.get_turn() == board_copy.CELL_
                board_copy.move(random.choice(valid_actions))
            # update
            if board_copy.get_game_result() == board_copy.RESULT_DRAW:
                result = 0
            else:
                result = 1 if board_copy.get_game_result() == current_player else -1
            while node is not None:
                node.update(result)
                node = node.parent
                # result = -result
            # node.update_recursion(result) # junxiaosong的更新方法，从根结点开始更新
        if queue != None:
            queue.put(root)

    def simulation_parallel(self, board):
        print("pure mcst parallel simulation")
        start = time.time()
        processes_count = os.cpu_count() - 2
        pool = Pool(processes_count)
        queue = Manager().Queue()
        simulation_times_each_process = self.n_rollout // processes_count
        for i in range(processes_count):
            pool.apply_async(func=self.simulation_subprocess,
                             args=(board, self.root, queue, simulation_times_each_process, self.c_param, ))
        pool.close()
        pool.join()

        ## 合并第二层子结点
        while not queue.empty():
            sub_root = queue.get()
            # print("sub_root {}".format(sub_root))
            self.root.n_visits += sub_root.n_visits
            self.root.n_wins += sub_root.n_wins
            for key in sub_root.children:
                if key in self.root.children:
                    self.root.children[key].n_wins += sub_root.children[key].n_wins
                    self.root.children[key].n_visits += sub_root.children[key].n_wins
                else:
                    self.root.children[key] = sub_root.children[key]

        ## 根据获胜情况，选取最优结果
        # sub_root_list = []
        # while not queue.empty():
        #     sub_root = queue.get()
        #     sub_root_list.append(sub_root)
        # max_root = max(sub_root_list, key=lambda root:root.n_wins)
        # self.root = max_root

        end = time.time()
        print("Task runs %.2fs" % (end - start))

        # print("main process end {}".format(self.root))
        # print(max(self.root.children.items(), key=lambda action_node: action_node[1].n_wins))
        # print(max(self.root.children.items(), key=lambda action_node: action_node[1].n_visits))

    def get_move(self):
        return max(self.root.children.items(), key=lambda action_node: action_node[1].n_visits)[0]



