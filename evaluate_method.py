import numpy as np
import random

class Evaluate:
    def __init__(self, board):
        self.board = board
        self.weight = {
            self.board.CELL_BLACK: np.zeros_like(board.board),
            self.board.CELL_WHITE: np.zeros_like(board.board)
        }
        self.board_size = board.board_size

    def evaluate_and_get_move(self):
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board.board[i][j] == self.board.CELL_EMPTY:
                    # 获取当前位置为中心的前4个位置和后4个位置，共9个点
                    # start: 起始位置
                    start_row = i-(self.board.N_DOTS-1) if i-(self.board.N_DOTS-1) >=0 else 0
                    start_col = j-(self.board.N_DOTS-1) if j-(self.board.N_DOTS-1) >=0 else 0
                    # end: 结束位置（不包含）
                    end_row = i+(self.board.N_DOTS-1) if i+(self.board.N_DOTS-1) <=self.board_size else self.board_size
                    end_col = j+(self.board.N_DOTS-1) if j+(self.board.N_DOTS-1) <=self.board_size else self.board_size

                    # 检测白棋
                    self.calc_weight(start_row, start_col, end_row, end_col, player=self.board.CELL_WHITE,
                                     opponent=self.board.CELL_BLACK)

                    # 检测黑棋
                    self.calc_weight(start_row, start_col, end_row, end_col, player=self.board.CELL_BLACK,
                                     opponent=self.board.CELL_WHITE)
        current_player = self.board.get_turn()
        if current_player == self.board.CELL_WHITE:
            opponent = self.board.CELL_BLACK
        else:
            opponent = self.board.CELL_WHITE

        if np.max(self.weight[opponent]) >= 3:
            weight = self.weight[current_player] + self.weight[opponent]*5
        elif np.max(self.weight[current_player]) >= 3:
            weight = self.weight[current_player]*5 + self.weight[opponent]
        else:
            weight = self.weight[current_player]
        print(self.weight[current_player])
        print(self.weight[opponent])
        print(weight)

        rows_and_cols = np.where(weight==np.max(weight))
        pos = random.choice(list(zip(rows_and_cols[0], rows_and_cols[1])))
        return pos


    def calc_weight(self, start_row, start_col, end_row, end_col, player, opponent):
        for row_index in range(start_row, end_row):
            for col_index in range(start_col, end_col):
                ## 检测一行
                player_weight = 0
                for i in range(self.board.N_DOTS):
                    if col_index+i < end_col:
                        # 如果在五个位置中出现对手player的棋，则当前player棋必定不能连成五子，空位的权重设为0
                        if self.board.board[row_index][col_index+i] == opponent:
                            player_weight = 0
                            break
                        # 如果出现当前player的棋，则将权重增加，当前player的棋越多，连成五子的可能越大，则空位的权重越大
                        # 如果上面的条件满足，则此处的统计无效
                        if self.board.board[row_index][col_index+i] == player:
                            player_weight += 1
                # 更新weight, 新的weight大于原有weight，且所在位置为空位时，才更新
                for i in range(self.board.N_DOTS):
                    if col_index+i < end_col and self.board.board[row_index][col_index+i]==self.board.CELL_EMPTY \
                            and player_weight > self.weight[player][row_index][col_index+i]:
                        self.weight[player][row_index][col_index+i] = player_weight

                ## 检测一列
                player_weight = 0
                for i in range(self.board.N_DOTS):
                    if row_index+i < end_row:
                        # 如果在五个位置中出现对手player的棋，则当前player棋必定不能连成五子，空位的权重设为0
                        if self.board.board[row_index+i][col_index] == opponent:
                            player_weight = 0
                            break
                        # 如果出现当前player的棋，则将权重增加，当前player的棋越多，连成五子的可能越大，则空位的权重越大
                        # 如果上面的条件满足，则此处的统计无效
                        if self.board.board[row_index+i][col_index] == player:
                            player_weight += 1
                # 更新weight
                for i in range(self.board.N_DOTS):
                    if row_index+i < end_row and self.board.board[row_index+i][col_index]==self.board.CELL_EMPTY \
                            and player_weight > self.weight[player][row_index+i][col_index]:
                        self.weight[player][row_index+i][col_index] = player_weight

                ## 检测对角线
                player_weight = 0
                for i in range(self.board.N_DOTS):
                    if row_index+i < end_row and col_index+i < end_col:
                        # 如果在五个位置中出现对手player的棋，则当前player棋必定不能连成五子，空位的权重设为0
                        if self.board.board[row_index+i][col_index+i] == opponent:
                            player_weight = 0
                            break
                        # 如果出现当前player的棋，则将权重增加，当前player的棋越多，连成五子的可能越大，则空位的权重越大
                        # 如果上面的条件满足，则此处的统计无效
                        if self.board.board[row_index+i][col_index+i] == player:
                            player_weight += 1
                # 更新weight
                for i in range(self.board.N_DOTS):
                    if row_index+i < end_row and col_index+i <end_col \
                            and self.board.board[row_index + i][col_index+i] == self.board.CELL_EMPTY \
                            and player_weight > self.weight[player][row_index+i][col_index+i]:
                        self.weight[player][row_index+i][col_index+i] = player_weight

                ## 检测斜对角线
                player_weight = 0
                for i in range(self.board.N_DOTS):
                    if row_index+i < end_row and col_index-i >= start_col:
                        # 如果在五个位置中出现对手player的棋，则当前player棋必定不能连成五子，空位的权重设为0
                        if self.board.board[row_index + i][col_index - i] == opponent:
                            player_weight = 0
                            break
                        # 如果出现当前player的棋，则将权重增加，当前player的棋越多，连成五子的可能越大，则空位的权重越大
                        # 如果上面的条件满足，则此处的统计无效
                        if self.board.board[row_index + i][col_index - i] == player:
                            player_weight += 1
                # 更新weight
                for i in range(self.board.N_DOTS):
                    if row_index+i < end_row and col_index-i >= start_col \
                            and self.board.board[row_index+i][col_index-i]==self.board.CELL_EMPTY \
                            and player_weight > self.weight[player][row_index + i][col_index -i]:
                        self.weight[player][row_index+i][col_index-i] = player_weight


