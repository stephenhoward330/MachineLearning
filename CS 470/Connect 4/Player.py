import numpy as np


# Utils

def make_move(board, move, player_number):
    """
    This function will execute the move (integer column number) on the given board, 
    where the acting player is given by player_number
    """
    row = 0
    while row < 6 and board[row, move] == 0:
        row += 1
    board[row - 1, move] = player_number


def get_valid_moves(board):
    """
    This function will return a list with all the valid moves (column numbers)
    for the input board
    """
    valid_moves = []
    for c in range(7):
        if 0 in board[:, c]:
            valid_moves.append(c)
    return valid_moves


def is_winning_state(board, player_num):
    """
    This function will tell if the player_num player is
    winning in the board that is input
    """
    player_win_str = '{0}{0}{0}{0}'.format(player_num)
    to_str = lambda a: ''.join(a.astype(str))

    def check_horizontal(b):
        for row in b:
            if player_win_str in to_str(row):
                return True
        return False

    def check_vertical(b):
        return check_horizontal(b.T)

    def check_diagonal(b):
        for op in [None, np.fliplr]:
            op_board = op(b) if op else b

            root_diag = np.diagonal(op_board, offset=0).astype(np.int)
            if player_win_str in to_str(root_diag):
                return True

            for i in range(1, b.shape[1] - 3):
                for offset in [i, -i]:
                    diag = np.diagonal(op_board, offset=offset)
                    diag = to_str(diag.astype(np.int))
                    if player_win_str in diag:
                        return True

        return False

    return (check_horizontal(board) or
            check_vertical(board) or
            check_diagonal(board))


def score(board, player_num):
    # count how many wins this player would have if they filled the rest of the board with their pieces
    t_score = 0

    def horizontal_score(b, num):
        h_score = 0
        for i in range(len(b)):
            for j in range(len(b[i])-3):
                if b[i][j] in [0, num] and \
                        b[i][j + 1] in [0, num] and \
                        b[i][j + 2] in [0, num] and \
                        b[i][j + 3] in [0, num]:
                    h_score += 1
        return h_score

    # wins across the rows
    t_score += horizontal_score(board, player_num)
    # wins across the columns
    t_score += horizontal_score(board.T, player_num)

    def diagonal_score(b, num):
        d_score = 0
        for i in range(len(b)-3):
            for j in range(len(b[i])-3):
                if b[i][j] in [0, num] and \
                        b[i + 1][j + 1] in [0, num] and \
                        b[i + 2][j + 2] in [0, num] and \
                        b[i + 3][j + 3] in [0, num]:
                    d_score += 1
        for i in range(3, len(b)):
            for j in range(len(b[i])-3):
                if b[i][j] in [0, num] and \
                        b[i - 1][j + 1] in [0, num] and \
                        b[i - 2][j + 2] in [0, num] and \
                        b[i - 3][j + 3] in [0, num]:
                    d_score += 1
        return d_score

    # diagonal wins
    t_score += diagonal_score(board, player_num)

    return t_score

# The players!


class AIPlayer:
    def __init__(self, player_number):
        self.player_number = player_number  # This is the id of the player this AI is in the game
        self.type = 'ai'
        self.player_string = 'Player {}:ai'.format(player_number)
        self.other_player_number = 1 if player_number == 2 else 2  # This is the id of the other player
        self.depth_limit = 4  # how deep to allow the alpha-beta search to go

    def max_value(self, state, alpha, beta, depth):
        best_action = None
        best_d = np.inf  # best distance... tie-breaker for nodes of the same value
        if is_winning_state(state, self.player_number) \
                or is_winning_state(state, self.other_player_number) \
                or depth == self.depth_limit:
            return self.evaluation_function(state), depth, best_action
        v = -np.inf
        for a in get_valid_moves(state):
            new_state = np.copy(state)
            make_move(new_state, a, self.player_number)
            new_v, d, _ = self.min_value(new_state, alpha, beta, depth + 1)
            if new_v > v:
                v = new_v
                best_d = d
                best_action = a
            # be quick to win, slow to lose
            elif new_v == v and v > 0 and d < best_d:
                v = new_v
                best_d = d
                best_action = a
            elif new_v == v and v < 0 and d > best_d:
                v = new_v
                best_d = d
                best_action = a
            if v > beta:  # prune
                return v, best_d, best_action
            alpha = max(alpha, v)
        # this happens if there are only a few open spots left on the board
        if best_action is None:
            return self.evaluation_function(state), depth, best_action
        return v, best_d, best_action

    def min_value(self, state, alpha, beta, depth):
        best_action = None
        best_d = np.inf  # best distance... tie-breaker for nodes of the same value
        if is_winning_state(state, self.player_number) \
                or is_winning_state(state, self.other_player_number) \
                or depth == self.depth_limit:
            return self.evaluation_function(state), depth, best_action
        v = np.inf
        for a in get_valid_moves(state):
            new_state = np.copy(state)
            make_move(new_state, a, self.other_player_number)
            new_v, d, _ = self.max_value(new_state, alpha, beta, depth + 1)
            if new_v < v:
                v = new_v
                best_d = d
                best_action = a
            # be quick to win, slow to lose
            elif new_v == v and v < 0 and d < best_d:
                v = new_v
                best_d = d
                best_action = a
            elif new_v == v and v > 0 and d > best_d:
                v = new_v
                best_d = d
                best_action = a
            if v < alpha:  # prune
                return v, best_d, best_action
            beta = min(beta, v)
        # this happens if there are only a few open spots left on the board
        if best_action is None:
            return self.evaluation_function(state), depth, best_action
        return v, best_d, best_action

    def get_alpha_beta_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the alpha-beta pruning algorithm

        This will play against either itself or a human player

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """

        val, _, action = self.max_value(board, -np.inf, np.inf, 0)
        print("AI does action", action, "with value", val)
        if action is None:
            raise Exception("Action is None!!!")
        return action

    def expectimax_value(self, state, depth, max_turn=True):
        best_action = None
        if is_winning_state(state, self.player_number) \
                or is_winning_state(state, self.other_player_number) \
                or depth == self.depth_limit:  # terminal node or depth limit reached
            return self.evaluation_function(state), best_action
        if max_turn:  # max node
            v = -np.inf
            for a in get_valid_moves(state):
                new_state = np.copy(state)
                make_move(new_state, a, self.player_number)
                new_v, _ = self.expectimax_value(new_state, depth + 1, not max_turn)
                if new_v > v:
                    v = new_v
                    best_action = a
        else:  # chance node
            v = 0
            valid_moves = get_valid_moves(state)
            move_prob = 1 / len(valid_moves)
            for a in valid_moves:
                new_state = np.copy(state)
                make_move(new_state, a, self.other_player_number)
                new_v, _ = self.expectimax_value(new_state, depth + 1, not max_turn)
                v += move_prob * new_v
        return v, best_action

    def get_expectimax_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the expectimax algorithm.

        This will play against the random player, who chooses any valid move
        with equal probability

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        val, action = self.expectimax_value(board, 0, True)
        print("AI does action", action, "with value", round(val, 2))
        if action is None:
            raise Exception("Action is None!!!")
        return action

    def evaluation_function(self, board):
        """
        Given the current state of the board, return the scalar value that
        represents the evaluation function for the current player
       
        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The utility value for the current board
        """

        if is_winning_state(board, self.player_number):
            return 100
        if is_winning_state(board, self.other_player_number):
            return -100

        return score(board, self.player_number) - score(board, self.other_player_number)


class RandomPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'random'
        self.player_string = 'Player {}:random'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state select a random column from the available
        valid moves.

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        valid_cols = []
        for col in range(board.shape[1]):
            if 0 in board[:, col]:
                valid_cols.append(col)

        return np.random.choice(valid_cols)


class HumanPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'human'
        self.player_string = 'Player {}:human'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state returns the human input for next move

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """

        valid_cols = []
        for i, col in enumerate(board.T):
            if 0 in col:
                valid_cols.append(i)

        move = int(input('Enter your move: '))

        while move not in valid_cols:
            print('Column full, choose from:{}'.format(valid_cols))
            move = int(input('Enter your move: '))

        return move
