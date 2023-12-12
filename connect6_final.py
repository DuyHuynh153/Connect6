import tkinter as tk
from enum import Enum
import tkinter.messagebox
import random
import copy
import time
import numpy as np
import cProfile
from functools import cache, lru_cache
from heapq import nlargest
import math

from numba import jit,njit

BOARD_SIZE = 19
directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

font_config = ("Helvetica", 10)


class Player(Enum):
    BLACK = "black"
    RED = "red"


class Connect6Game:
    def __init__(self):
        self.current_player = Player.BLACK
        self.board_state = [[""] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.stones_placed = 1
        self.game_over = False
        self.buttons = [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.game_states = []
        self.max_undo = 20
        self.redo_states = []
        self.turn = 0
        self.AI_MODE = False
        self.current_board = []
        self.playout =0

    def setup_ui(self, root):
        # Create the outer frame for the bigger board
        outer_frame = tk.Frame(root)
        outer_frame.grid(row=0, column=0, padx=100, pady=100)

        # Add a label to display the current player's turn
        self.next_turn_label = tk.Label(outer_frame, text=f"Next turn: {self.current_player.value}",
                                        font=("Helvetica", 16))
        self.next_turn_label.grid(row=0, column=0, columnspan=BOARD_SIZE, pady=(0, 20))

        # Add an error message label
        self.error_label = tk.Label(outer_frame, text="", fg="red", font=("Helvetica", 16))
        self.error_label.grid(row=BOARD_SIZE, columnspan=BOARD_SIZE)

        # Add a label to display win messages
        self.win_label = tk.Label(outer_frame, text="", fg="green", font=("Helvetica", 16))
        self.win_label.grid(row=BOARD_SIZE + 1, columnspan=BOARD_SIZE)

        # Create the inner frame for the Connect 6 board
        inner_frame = tk.Frame(outer_frame)
        inner_frame.grid(row=1, column=1)

        # Create buttons for the Connect 6 board and store them in the buttons list
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                button = tk.Button(inner_frame, width=4, height=1,
                                   command=lambda row=i, col=j: self.place_stone(row, col))
                button.grid(row=i, column=j)
                self.buttons[i][j] = button

            # Create a new frame for the "Save," "Load," "Back," and "Redo" buttons
        button_frame = tk.Frame(outer_frame)
        button_frame.grid(row=21, column=0, columnspan=19)  # Place the button frame below the board

        # Create a new frame for the restart button and place it in the outer frame
        restart_frame = tk.Frame(outer_frame)
        restart_frame.grid(row=2, column=1)

        # Add a "Restart" button to clear the board and start a new game
        restart_button = tk.Button(restart_frame, text="Restart", command=self.restart_game)
        restart_button.grid(row=0, column=0, padx=10)

        # Add a "Save" button to save the current board state
        save_button = tk.Button(restart_frame, text="Save", command=self.save_moves)
        save_button.grid(row=0, column=1, padx=10)

        # Add a "Load" button to load a saved game state
        load_button = tk.Button(restart_frame, text="Load", command=self.load_moves)
        load_button.grid(row=0, column=2, padx=10)

        # Add a "Load" button to load a saved game state
        # load_button_step_by_step = tk.Button(restart_frame, text="Load Step by step", command= self.load_moves_step_by_step)
        # load_button_step_by_step.grid(row=0, column=3, padx=10)

        # Add a "Back" button to go back to a previous game state
        back_button = tk.Button(restart_frame, text="Back", command=self.back_move)
        back_button.grid(row=0, column=4, padx=10)

        # Add a "Redo" button to redo a move
        redo_button = tk.Button(restart_frame, text="Redo", command=self.redo_move)
        redo_button.grid(row=0, column=5, padx=10)

        # Add a "Play With AI" button
        self.play_with_ai_button = tk.Button(restart_frame, text=f"Play With AI: {self.AI_MODE}",
                                             command=self.play_with_ai)
        self.play_with_ai_button.grid(row=0, column=6, padx=10)
        
        self.playout_button = tk.Button(restart_frame, text=f"Playout: {self.playout}")
        self.playout_button.grid(row=0, column=7, padx=10)

    def place_stone(self, row, col, text=None, winner=None, _10_node=None):
        if self.game_over:
            return

        # Check if the clicked button is empty
        if self.board_state[row][col] == "":
            # Clear the redo stack when a new move is made
            self.redo_states.clear()

            # Save the current game state before making a move
            prev_state = [row[:] for row in self.board_state]
            prev_player = self.current_player
            prev_stones_placed = self.stones_placed

            # Update the board state and display the stone color
            self.board_state[row][col] = self.current_player.value
            # Change the button color
            stone_color = self.current_player.value
            self.buttons[row][col].configure(bg=stone_color)
            # if text == None:
            #     self.buttons[row][col].configure (bg=stone_color)
            # elif text =='winner':
            #     self.buttons[row][col].configure (bg=stone_color,text=f'{winner.state.winScore}/{winner.state.visitCount}')
            #     for node, position in _10_node.items():
            #         self.buttons[position[0]][position[1]].configure (bg='gray',text=f'{node.state.winScore}/{node.state.visitCount}')
            self.stones_placed += 1

            # Save the previous state to the game_states stack so we can back,redo,save,load,...
            self.game_states.append((prev_state, prev_player, prev_stones_placed))

            # Check for a win
            if self.check_win(row, col):
                self.show_win_message()
                self.game_over = True
                # Change the color of the winning stones to green
                directions = [(1, 0), (0, 1), (1, 1), (-1, 1)]
                for dr, dc in directions:
                    winning_stones = [(row, col)]
                    for offset in range(1, 6):
                        new_row = row + dr * offset
                        new_col = col + dc * offset
                        if (0 <= new_row < 19 and 0 <= new_col < 19 and
                                self.board_state[new_row][new_col] == self.current_player.value):
                            winning_stones.append((new_row, new_col))
                        else:
                            break

                    for offset in range(1, 6):
                        new_row = row - dr * offset
                        new_col = col - dc * offset
                        if (0 <= new_row < 19 and 0 <= new_col < 19
                                and self.board_state[new_row][new_col] == self.current_player.value):
                            winning_stones.append((new_row, new_col))
                        else:
                            break

                    if len(winning_stones) >= 6:
                        for r, c in winning_stones:
                            self.buttons[r][c].configure(bg="green")

            elif self.check_draw():
                self.show_draw_message()
            elif self.stones_placed == 2:
                # Switch to the next player's turn
                # if next player is Player or AI
                if self.AI_MODE == True:
                    if self.current_player.value == "black":
                        self.make_ai_move(np.array((row, col), dtype=np.int8))

                elif self.AI_MODE == False:
                    self.stones_placed = 0
                    self.switch_player()

            self.error_label.config(text="")


        else:
            self.error_label.config(text="Invalid move. Cell is already occupied.")
            # self.root.after(100, self.clear_error_label)

    def update_button(self, row, col):
        stone_color = "black" if self.board_state[row][col] == Player.BLACK.value else "red"
        self.buttons[row][col].config(bg=stone_color)

    def clear_error_label(self):
        self.error_label.config(text="")

    def clear_win_label(self):
        self.win_label.config(text="")

    def show_win_message(self):
        self.win_label.config(text=f"{self.current_player.value} wins!")
        self.game_over = True

    def show_draw_message(self):
        self.win_label.config(text="It's a draw!")
        self.game_over = True

    def switch_player(self):

        self.current_player = Player.RED if self.current_player == Player.BLACK else Player.BLACK
        self.next_turn_label.config(text=f"Next turn: {self.current_player.value}")

    def checkStatus(self, row, col):
        """Check the status of the game at the given position."""
        if self.check_win(row, col):
            return 1  # Thắng
        elif self.check_draw():
            return 0  # Hòa
        else:
            return -1  # Tiếp tục

    def check_win(self, row, col):
        """Check if the current player has won."""
        directions = [(1, 0), (0, 1), (1, 1), (-1, 1)]
        for dr, dc in directions:
            count = 1  # Count the current player's stones
            for offset in range(1, 6):  # Check up to 5 stones in each direction
                new_row = row + dr * offset
                new_col = col + dc * offset
                if 0 <= new_row < 19 and 0 <= new_col < 19 and self.board_state[new_row][
                    new_col] == self.current_player.value:
                    count += 1
                else:
                    break

            for offset in range(1, 6):
                new_row = row - dr * offset
                new_col = col - dc * offset
                if 0 <= new_row < 19 and 0 <= new_col < 19 and self.board_state[new_row][
                    new_col] == self.current_player.value:
                    count += 1
                else:
                    break

            if count >= 6:
                return True

        return False

    def check_draw(self):
        """Check if the game ends in a draw."""
        return all(all(cell != "" for cell in row) for row in self.board_state)

    def clear_board(self):
        """Clear the board and reset the board state."""
        for row in range(19):
            for col in range(19):
                self.buttons[row][col].config(bg="SystemButtonFace")
                self.board_state[row][col] = ""

    def save_moves(self):
        # Implement saving the game state to a file
        try:
            with open("moves.txt", "w") as file:

                for row in range(BOARD_SIZE):
                    for col in range(BOARD_SIZE):
                        if self.board_state[row][col] != "":
                            # Write the move in the format: row col color
                            file.write(f"{row} {col} {self.board_state[row][col]}\n")

                self.win_label.config(text="success saving move")

        except Exception as e:
            self.error_label.config(text="Error saving move")
            print(f"Error saving moves: {e}")

    def load_moves(self):
        try:
            with open("moves.txt", "r") as file:
                # Prompt the user for confirmation before loading moves
                confirmation = tkinter.messagebox.askyesno("Confirm Load",
                                                           "Do you want to load the saved moves? This will clear the current board.")

                if confirmation:
                    # Clear the current board
                    self.clear_board()

                    load_stones = 0

                    for line in file:
                        parts = line.strip().split()
                        if len(parts) == 3:
                            row, col, color = int(parts[0]), int(parts[1]), parts[2]
                            if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
                                if color == Player.BLACK.value:
                                    self.board_state[row][col] = Player.BLACK.value
                                    self.update_button(row, col)
                                    load_stones += 1
                                elif color == Player.RED.value:
                                    self.board_state[row][col] = Player.RED.value
                                    self.update_button(row, col)
                                    load_stones += 1

                    # Set the number of stones placed based on the loaded state

                    print("Stones after load: " + str(load_stones))
                    if load_stones % 4 == 2:
                        self.current_player = Player.RED
                        self.stones_placed = 1
                    elif load_stones % 4 == 3:
                        self.current_player = Player.RED
                        self.switch_player()
                        self.stones_placed = 0
                    elif load_stones % 4 == 0:
                        self.current_player = Player.BLACK
                        self.stones_placed = 1
                    elif load_stones % 4 == 1:
                        self.current_player = Player.BLACK
                        self.switch_player()
                        self.stones_placed = 0





        except FileNotFoundError:
            # The file does not exist, so we create a new one
            with open("moves.txt", "w") as file:
                # Add any initial content to the new file if needed
                print("A new 'moves.txt' file has been created.")

    def load_moves_step_by_step(self):

        try:
            with open("moves.txt", "r") as file:
                # Prompt the user for confirmation before loading moves
                confirmation = tkinter.messagebox.askyesno("Confirm Load",
                                                           "Do you want to load step by step? This will clear the current board.")

                if confirmation:
                    # Clear the current board
                    self.clear_board()
                    load_stones = 0
                    for line in file:
                        parts = line.strip().split()
                        if len(parts) == 3:
                            row, col, color = int(parts[0]), int(parts[1]), parts[2]
                            if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
                                if color == Player.BLACK.value:
                                    self.board_state[row][col] = Player.BLACK.value
                                    self.update_button(row, col)

                                    load_stones += 1
                                elif color == Player.RED.value:
                                    self.board_state[row][col] = Player.RED.value
                                    self.update_button(row, col)
                                    load_stones += 1

                    # Set the number of stones placed based on the loaded state

                    print("Stones after load: " + str(load_stones))
                    if load_stones % 4 == 2:
                        self.current_player = Player.RED
                        self.stones_placed = 1
                    elif load_stones % 4 == 3:
                        self.current_player = Player.RED
                        self.switch_player()
                        self.stones_placed = 0
                    elif load_stones % 4 == 0:
                        self.current_player = Player.BLACK
                        self.stones_placed = 1
                    elif load_stones % 4 == 1:
                        self.current_player = Player.BLACK
                        self.switch_player()
                        self.stones_placed = 0

        except FileNotFoundError:
            # The file does not exist, so we create a new one
            with open("moves.txt", "w") as file:
                # Add any initial content to the new file if needed
                print("A new 'moves.txt' file has been created.")

    def back_move(self):
        if self.game_states:

            self.redo_states.append((self.board_state, self.current_player, self.stones_placed))

            # Restore the previous game state
            prev_state, prev_player, prev_stones_placed = self.game_states.pop()
            self.board_state, self.current_player, self.stones_placed = prev_state, prev_player, prev_stones_placed

            # Clear the buttons for the previous moves (set them to the default color)
            for row in range(BOARD_SIZE):
                for col in range(BOARD_SIZE):
                    self.buttons[row][col].configure(bg="SystemButtonFace")

            # Update the UI to match the restored state
            for row in range(BOARD_SIZE):
                for col in range(BOARD_SIZE):
                    stone_color = self.board_state[row][col]
                    if stone_color != "":
                        self.buttons[row][col].configure(bg=stone_color)

            # Switch the player label
            self.next_turn_label.config(text=f"Next turn: {self.current_player.value}")
            self.game_over = False
            self.win_label.config(text=f"")
            self.error_label.config(text=f"")


        else:
            self.error_label.config(text="player not allow to back anymore")
            # self.root.after(1000, self.clear_error_label())

    def redo_move(self):
        if self.redo_states:
            # Push the current game state to the game_states stack
            self.game_states.append((self.board_state, self.current_player, self.stones_placed))

            # Restore the next game state from the redo stack
            next_state, next_player, next_stones_placed = self.redo_states.pop()
            self.board_state, self.current_player, self.stones_placed = next_state, next_player, next_stones_placed

            # Update the UI to match the restored state
            for row in range(BOARD_SIZE):
                for col in range(BOARD_SIZE):
                    stone_color = self.board_state[row][col]
                    if stone_color != "":
                        self.buttons[row][col].configure(bg=stone_color)
                        if self.check_win(row, col):
                            self.show_win_message()
                            self.game_over = True
                            # Change the color of the winning stones to green
                            directions = [(1, 0), (0, 1), (1, 1), (-1, 1)]
                            for dr, dc in directions:
                                winning_stones = [(row, col)]
                                for offset in range(1, 6):
                                    new_row = row + dr * offset
                                    new_col = col + dc * offset
                                    if (0 <= new_row < 19 and 0 <= new_col < 19 and
                                            self.board_state[new_row][new_col] == self.current_player.value):
                                        winning_stones.append((new_row, new_col))
                                    else:
                                        break

                                for offset in range(1, 6):
                                    new_row = row - dr * offset
                                    new_col = col - dc * offset
                                    if (0 <= new_row < 19 and 0 <= new_col < 19
                                            and self.board_state[new_row][new_col] == self.current_player.value):
                                        winning_stones.append((new_row, new_col))
                                    else:
                                        break

                                if len(winning_stones) >= 6:
                                    for r, c in winning_stones:
                                        self.buttons[r][c].configure(bg="green")

            # Switch the player label
            self.next_turn_label.config(text=f"Next turn: {self.current_player.value}")

        else:
            self.error_label.config(text="Can not redo because this is the futhest move")
            self.root.after(100, self.clear_error_label)

    def restart_game(self):
        # Clear the board and reset the game state
        self.clear_board()
        self.current_player = Player.BLACK
        self.stones_placed = 1
        self.game_over = False
        self.clear_win_label()
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                self.buttons[i][j].configure(text='')
        self.playout = 0
        self.playout_button.config(text=f"Playout: {self.playout}")

    def play_with_ai(self):
        self.AI_MODE = not self.AI_MODE
        self.play_with_ai_button.config(text=f"Play With AI: {self.AI_MODE}")  # Update button text

        if self.AI_MODE:
            # Restart the board
            self.restart_game()

            # Player (black) plays first
            self.current_player = Player.BLACK
            self.stones_placed = 1

        else:
            self.restart_game()

    def make_ai_move(self, position):
        # print("AI turn")
        # if self.stones_placed == 2:
        #     self.stones_placed =0
        #     self.switch_player()

        self.stones_placed = 0
        self.switch_player()

        while self.stones_placed < 2:
            print("lược chơi hiện tại của AI: " , str(self.turn))

            tkinter.messagebox.showinfo("Confirm play", "AI play the " + str(self.stones_placed + 1) + " move")
            numpy_board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
            for i in range(BOARD_SIZE):
                for j in range(BOARD_SIZE):
                    if self.board_state[i][j] == 'black':
                        numpy_board[i, j] = 1
                    elif self.board_state[i][j] == 'red':
                        numpy_board[i, j] = 2

            for i in range(BOARD_SIZE):
                for j in range(BOARD_SIZE):
                    self.buttons[i][j].configure(text='')

            monte_carlo = MonteCarloTreeSearch()
            winner_node, _10_best_node_winCrore, count_dem = monte_carlo.findNextMove(numpy_board, self.turn, position)
            print("so playout: ", count_dem)
            self.playout = count_dem
            self.playout_button.config(text=f"Playout: {self.playout}")  # Update button text

            # print("lược chơi hiện tại: " , str(self.turn))
            new_lst = [[""] * BOARD_SIZE for _ in range(BOARD_SIZE)]
            for i in range(BOARD_SIZE):
                for j in range(BOARD_SIZE):
                    if winner_node.state.board[i, j] == 1:
                        new_lst[i][j] = "black"
                    elif winner_node.state.board[i, j] == 2:
                        new_lst[i][j] = "red"

            flag = False
            row = 0
            col = 0
            for i in range(BOARD_SIZE):
                for j in range(BOARD_SIZE):
                    if new_lst[i][j] != self.board_state[i][j]:
                        # self.stones_placed +=1
                        self.turn += 1
                        row = i
                        col = j
                        self.place_stone(row, col, text='winner')
                        self.buttons[row][col].configure(
                            text=f'{winner_node.state.winScore}_{winner_node.state.visitCount}')
                        # flag = True
                        break

            for child in _10_best_node_winCrore[1:11]:

                for i in range(BOARD_SIZE):
                    for j in range(BOARD_SIZE):
                        if self.board_state[i][j] == '' and child.state.board[i, j] != 0:
                            self.buttons[i][j].configure(
                                text='{}_{}'.format(child.state.winScore, child.state.visitCount))
                            break
            if self.stones_placed == 2:
                break

        if self.stones_placed == 2:
            self.stones_placed = 0
            self.switch_player()


class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.childArray = []


    def is_fully_expand(self):
        return len(self.childArray) == 361 - self.state.current_player



class Tree:
    def __init__(self):
        self.root = Node(None)

    def set_root(self, node):
        self.root = node


class State:
    def __init__(self, board, current_player):
        self.board = board
        self.current_player = current_player
        self.visitCount = 0
        self.winScore = 0
        self.position = None

    def randomPlay(self):
        return perform_random_play(self.board, self.current_player)

        # player = 1 if (self.current_player - 1) % 4 in {0, 3} else 2
        #
        #     empty_cells = np.transpose(np.where(self.board == 0))
        #     if empty_cells.size > 0:
        #         random_move = random.choice(empty_cells)
        #         self.board[random_move[0], random_move[1]] = player
        #         return self.board, tuple(random_move)


# @jit(nopython=True)
@njit(cache = True)
def perform_random_play(board, current_player):
    player = 1 if (current_player - 1) % 4 in {0, 3} else 2

    # empty_cells = np.transpose(np.where(board == 0))
    empty_cells = np.argwhere(board == 0)
    if empty_cells.size > 0:

        random_index = random.randint(0, empty_cells.shape[0] - 1)
        random_move = empty_cells[random_index]

        board[random_move[0], random_move[1]] = player
        return board, random_move

@njit(cache = True)
def within_bounds(row, col):
    return (0 <= row < BOARD_SIZE) and (0 <= col < BOARD_SIZE)

@njit(cache=True)

def check_win(row, col, board, playerNum):
    # This function is now decorated with Numba for optimization
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    # opponent = 3 - playerNum

    for dr, dc in directions:
        count = 1  # Count the current player's stones
        for offset in range(1, 6):  # Check up to 5 stones in each direction
            new_row = row + dr * offset
            new_col = col + dc * offset
           
            if within_bounds(new_row,new_col) and  board[new_row, new_col] == playerNum:

                count += 1
            else:
                break

        for offset in range(1, 6):
            new_row = row - dr * offset
            new_col = col - dc * offset
            if within_bounds(new_row,new_col) and  board[new_row, new_col] == playerNum:
                count += 1
            else:
                break

        if count >= 6:
            return playerNum
    return False







class MonteCarloTreeSearch:
    WIN_SCORE = 1
    DEM = 0

    def __init__(self):
        self.opponent = 0

        self.end = time.time() + 6

    

    def checkStatus(self, node, check_row, check_col):
        board = node.state.board
        playerNum = board[check_row,check_col]

        # if check_row is not None and check_col is not None:
        result = check_win(check_row, check_col, board, playerNum)
        if result:
            return result
        else:
            # Checking for empty cells without using np.any()
            if 0 in board:
                return -1
            return 0

    def findNextMove(self, board, playerNo, position):
        self.opponent = playerNo + 1
        tree = Tree()

        rootNode = tree.root  # mặc định parent = None
        state = State(board, self.opponent)
        rootNode.state = state
        rootNode.state.position = position
        # self.end = time.time() + 5
        dem = 0
        while time.time() < self.end:

            node = rootNode
            while node.is_fully_expand():
                node = self.selectPromisingNode(node)

            if self.checkStatus(node, check_row=node.state.position[0], check_col=node.state.position[1]) == -1:
                if not node.is_fully_expand():
                    self.expandNode(node)
            node_to_explore = node
            if node.childArray:
                node_to_explore = random.choice(node.childArray)
            playout_result = self.simulateRandomPlayout(node_to_explore)
            self.backPropogation(node_to_explore, playout_result)
            dem += 1

        max_visit = 0
        winnerNode = None
        top_10_nodes = []
        total_visit = 0
        for child in rootNode.childArray:
            total_visit += child.state.visitCount
            if child.state.visitCount > max_visit:
                winnerNode = child
                max_visit = child.state.visitCount
            # Store nodes with the best winning scores
            top_10_nodes.append(child)

        # Get the top 10 nodes based on winning scores
        top_10_nodes = nlargest(11, top_10_nodes, key=lambda node: node.state.winScore)

       
        tree.set_root(winnerNode)
        return winnerNode, top_10_nodes, dem

    def selectPromisingNode(self, rootNode):
        node = rootNode  # node thay đổi thì rootNode thay đổi
        # Kiểm tra node có con không ?
        while len(node.childArray) != 0:
            node = UCT.findBestNodeWithUCT(node)
        return node

    def expandNode(self, promising_node):
        possible_states = np.transpose(np.where(promising_node.state.board == 0))

        random_index = np.random.randint(len(possible_states))
        random_position = possible_states[random_index]

        playerNo = promising_node.state.current_player % 4
        if playerNo in {0, 3}:
            playerNo = 1
        else:
            playerNo = 2

        # Create a copy of the board only for modification
        temp_board = np.copy(promising_node.state.board)
        temp_board[random_position[0], random_position[1]] = playerNo

        state = State(temp_board, promising_node.state.current_player + 1)
        state.position = random_position
        child_node = Node(state, promising_node)

        promising_node.childArray.append(child_node)

   

    def simulateRandomPlayout(self, node):
        state = State(
            board=np.copy(node.state.board),
            current_player=node.state.current_player
        )
        temp_node = Node(state)

        boardStatus = self.checkStatus(temp_node, node.state.position[0], node.state.position[1])

        while boardStatus == -1:
            temp_node.state.board, position = temp_node.state.randomPlay()
            temp_node.state.current_player += 1
            # print("({},{}): {}".format(position[0], position[1], "đen" if temp_node.state.board[position[0]][position[1]] == 1 else "trắng"))

            boardStatus = self.checkStatus(temp_node, position[0], position[1])


        # check_board = temp_node.state.board
        return boardStatus

       
    def backPropogation(self, nodeToExplore, playerNo):
        tempNode = nodeToExplore

        while tempNode is not None:
            tempNode.state.visitCount += 1

            player_turn = (tempNode.state.current_player - 1) % 4
            if player_turn in {0, 3}:
                player_turn = 1
            else:
                player_turn = 2

            if player_turn == playerNo:
                tempNode.state.winScore += 1
            if playerNo == 0:
                tempNode.state.winScore += 0.5

            tempNode = tempNode.parent



@njit( cache = True)
def uctValue(totalVisit, nodeWinScore, nodeVisit):
    if nodeVisit == 0:
        return 1e9

    return (nodeWinScore / nodeVisit) + 1.41 * math.sqrt(math.log(totalVisit) / nodeVisit)


class UCT:

    

    @staticmethod
    def findBestNodeWithUCT(node):

        parentVisit = node.state.visitCount
        max_uct_value = -1e9
        best_child = Node(None)

        for child in node.childArray:
            uct_val = uctValue(parentVisit, child.state.winScore, child.state.visitCount)
            if uct_val > max_uct_value:
                max_uct_value = uct_val
                best_child = child
        return best_child










def main():
    root = tk.Tk()
    root.title("Connect 6 Game")

    game = Connect6Game()
    game.setup_ui(root)

    root.mainloop()


if __name__ == "__main__":
    main()
    # cProfile.run('main()', sort="tottime")
