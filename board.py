import pygame
import numpy as np


class Board:
    def __init__(self, board_size, n_dots):
        self.CELL_EMPTY = 0
        self.CELL_WHITE = 1
        self.CELL_BLACK = -1

        self.RESULT_WHITE_WIN = 1
        self.RESULT_BLACK_WIN = -1
        self.RESULT_DRAW = 0
        self.RESULT_NOT_OVER = 2

        self.N_DOTS = n_dots  #棋子连子个数

        self.board = np.ones(shape=(board_size, board_size))*self.CELL_EMPTY
        self.step_count = 0
        self.board_size = board_size

        self.last_move = None

        self.black_color = (0, 0, 0)
        self.white_color = (255, 255, 255)
        self.h_grid_size = 0
        self.w_grid_size = 0



    def reset(self):
        self.board = np.ones(shape=(self.board_size, self.board_size))*self.CELL_EMPTY
        self.step_count = 0
        self.last_move = None

    def move(self, action):
        valid_pos = self.get_valid_move_pos()
        if action not in valid_pos:
            print("not a valid move, please move again!")
            return False
        else:
            row, col = action
            self.board[row][col] = self.get_turn()
            self.step_count += 1
            self.last_move = action
            return True

    def get_game_result(self):
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row][col] != self.CELL_EMPTY:
                    col_line = True
                    row_line = True
                    diagonal_line = True
                    antidiagonal_line = True
                    for i in range(self.N_DOTS):
                        # 判断一行
                        if row+i < self.board_size and self.board[row][col] == self.board[row+i][col]:
                            pass
                        else:
                            col_line = False
                        # 判断一列
                        if col+i < self.board_size and self.board[row][col] == self.board[row][col+i]:
                            pass
                        else:
                            row_line = False
                        # 判断对角线
                        if col+i < self.board_size and row+i < self.board_size and \
                                self.board[row][col] == self.board[row+i][col+i]:
                            pass
                        else:
                            diagonal_line = False
                        # 判断斜对角线
                        if row+i < self.board_size and col-i >=0 and \
                            self.board[row][col] == self.board[row+i][col-i]:
                            pass
                        else:
                            antidiagonal_line = False
                    if col_line or row_line or diagonal_line or antidiagonal_line:
                        if self.board[row][col] == self.CELL_BLACK:
                            return self.RESULT_BLACK_WIN
                        else:
                            return self.RESULT_WHITE_WIN
        if self.CELL_EMPTY not in self.board:
            return self.RESULT_DRAW
        else:
            return self.RESULT_NOT_OVER

    def get_valid_move_pos(self):
        valid_pos = []
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row][col] == self.CELL_EMPTY:
                    valid_pos.append((row, col))
        return valid_pos

    def get_turn(self):
        if self.step_count % 2 == 0:
            return self.CELL_BLACK
        else:
            return self.CELL_WHITE

    def draw_board(self, screen, screen_size):
        # 给窗口填充颜色
        screen.fill(color=(125, 95, 24))

        self.h_grid_size = screen_size[0] // (self.board_size + 1)
        self.w_grid_size = screen_size[1] // (self.board_size + 1)
        # 画棋盘
        for index in range(self.board_size + 1):
            pygame.draw.line(screen,
                             self.black_color,
                             start_pos=(self.h_grid_size, index * self.w_grid_size),
                             end_pos=(screen_size[0] - self.h_grid_size, index * self.w_grid_size),
                             width=1)
            pygame.draw.line(screen,
                             self.black_color,
                             start_pos=(index * self.h_grid_size, self.w_grid_size),
                             end_pos=(index * self.h_grid_size, screen_size[1] - self.w_grid_size),
                             width=1)
        # 给棋盘加外框
        pygame.draw.rect(screen,
                         self.black_color,
                         (self.h_grid_size - 4, self.w_grid_size - 4, screen_size[0] - 2 * self.h_grid_size + 8,
                          screen_size[1] - 2 * self.w_grid_size + 8),
                         width=3)
        # 在棋盘上标出几个特殊点
        pygame.draw.circle(screen, self.black_color, center=(screen_size[0] // 2, screen_size[1] // 2), radius=5,
                           width=0)
        pygame.draw.circle(screen, self.black_color, center=(screen_size[0] // 4, screen_size[1] // 4), radius=5,
                           width=0)
        pygame.draw.circle(screen, self.black_color, center=(screen_size[0] // 4, screen_size[1] * 3 // 4), radius=5,
                           width=0)
        pygame.draw.circle(screen, self.black_color, center=(screen_size[0] * 3 // 4, screen_size[1] // 4), radius=5,
                           width=0)
        pygame.draw.circle(screen, self.black_color, center=(screen_size[0] * 3 // 4, screen_size[1] * 3 // 4),
                           radius=5, width=0)
        pygame.display.flip()

    def draw_piece(self, screen):
        # # 绘制棋子时，step已经在move中发生了变化，此时get_turn获取到的是下一步的执棋者，
        # # 因此当前棋子的颜色 与get_turn的值相反
        # color = self.white_color if self.get_turn() == self.CELL_BLACK else self.black_color
        # pos = [40 * (col + 1), 40 * (row + 1)]
        # pygame.draw.circle(screen, color=color, center=pos, radius=18, width=0)
        # 绘制棋子
        text = pygame.font.Font("C:\\Windows\\Fonts\\simsun.ttc", 50)

        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row][col] != self.CELL_EMPTY:
                    color = self.black_color if self.board[row][col] == self.CELL_BLACK else self.white_color
                    pos = [self.h_grid_size*(col+1), self.w_grid_size*(row+1)]
                    pygame.draw.circle(screen, color=color, center=pos, radius=18, width=0)

        result = self.get_game_result()
        result_text = ""
        if result == self.RESULT_DRAW:
            print("平局")
            result_text = "平局"
        elif result == self.RESULT_WHITE_WIN:
            print("白棋胜")
            result_text = "白棋胜"
        elif result == self.RESULT_BLACK_WIN:
            print("黑棋胜")
            result_text = "黑棋胜"
        text_fmt = text.render(result_text, True, (255, 0, 0))
        screen.blit(text_fmt, (300, 200))
        pygame.display.flip()

    def get_symbol(self, cell):
        if cell == self.CELL_BLACK:
            return 'X'
        if cell == self.CELL_WHITE:
            return 'O'
        return '-'

    def print_board(self):
        board_as_string = "-----------------------\n"
        for row in range(self.board_size):
            for col in range(self.board_size):
                symbol = self.get_symbol(self.board[row][col])
                if col == self.board_size - 1:
                    board_as_string += f" {symbol} \n"
                else:
                    board_as_string += f" {symbol}"
        board_as_string += "-----------------------\n"
        print(board_as_string)


    def current_state(self):
        state = np.zeros(shape=(4, self.board_size, self.board_size))
        # 黑子位置标记为1， 其他位置标记为0
        state[0] = (self.board == self.CELL_BLACK).astype(np.int)
        # 白子位置标记为1， 其他位置标记为0
        state[1] = (self.board == self.CELL_WHITE).astype(np.int)
        # 标记最后落子的位置
        if self.last_move:
            row, col = self.last_move
            state[2][row][col] = 1
        # 标记当前玩家是否为先手，如果是先手，则标记为1，否则标记为0
        if self.step_count % 2 == 0:
            state[3] = 1
        else:
            state[3] = 0
        return state
