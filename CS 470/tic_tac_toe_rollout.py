import random
import numpy as np


# returns None if not a winning state, 1 if x wins, 2 if o wins
def is_winning_state(state):
    # check rows
    for row in state:
        if np.all(row == row[0]) and row[0] != 0:
            return row[0]

    # check columns
    for col in state.T:
        if np.all(col == col[0]) and col[0] != 0:
            return col[0]

    # check diagonals
    if state[0][0] == state[1][1] == state[2][2] and state[1][1] != 0:
        return state[1][1]
    if state[2][0] == state[1][1] == state[0][2] and state[1][1] != 0:
        return state[1][1]

    return None


# rolls out the game to completion, and returns 1 if x won, -1 is o won, or 0 if tie
def rollout(board, x_turn):
    # check if game has been won
    is_won = is_winning_state(board)
    if is_won is not None:
        return 1 if is_won == 1 else -1

    # check if game is full
    if 0 not in board:
        return 0

    # make a move and recurse
    result = np.where(board == 0)
    available_moves = list(zip(result[0], result[1]))
    loc = random.choice(available_moves)
    board[loc[0]][loc[1]] = 1 if x_turn else 2

    return rollout(board, not x_turn)


# simulate n games, with x moving first in location start_loc, return total score
# also returns a list with three values: number of games won, games tied, and games lost
def simulate(start_loc, n):
    score = 0
    counters = [0, 0, 0]
    for _ in range(n):
        # initialize state
        state = np.zeros((3, 3), np.int)

        # do x's first move
        if start_loc == 'center':
            state[1][1] = 1
        elif start_loc == 'corner':
            # place in a random corner
            options = [(0, 0), (0, 2), (2, 0), (2, 2)]
            loc = random.choice(options)
            state[loc[0]][loc[1]] = 1
        elif start_loc == 'side':
            # place in a random side space
            options = [(0, 1), (1, 0), (1, 2), (2, 1)]
            loc = random.choice(options)
            state[loc[0]][loc[1]] = 1
        else:
            raise ValueError("start_loc must be center, corner, or side")

        # rollout (finish) game and get result
        r = rollout(state, False)

        # update score and counters with rollout result
        score += r
        if r == 1:
            counters[0] += 1
        elif r == 0:
            counters[1] += 1
        elif r == -1:
            counters[2] += 1
        else:
            raise ValueError("Invalid return value from rollout()")

    return score, counters


def main():
    n = 10000

    print("CENTER")
    s_result, c = simulate('center', n)
    print("N:", n, " Score:", s_result, " Avg. Score:", s_result/n)
    print("Games won:", c[0], " Games tied:", c[1], " Games lost:", c[2])
    print()

    print("CORNER")
    s_result, c = simulate('corner', n)
    print("N:", n, " Score:", s_result, " Avg. Score:", s_result/n)
    print("Games won:", c[0], " Games tied:", c[1], " Games lost:", c[2])
    print()

    print("SIDE")
    s_result, c = simulate('side', n)
    print("N:", n, " Score:", s_result, " Avg. Score:", s_result/n)
    print("Games won:", c[0], " Games tied:", c[1], " Games lost:", c[2])


# simulate games of tic-tac-toe to see if going first in the center, corner, or side is best
# x is represented by 1, and o is 2. Empty spaces are 0.
if __name__ == '__main__':
    main()
