import matplotlib.pyplot as plt
import random
from time import time


DIE_SIDES = [1, 2, 3, 4, 5, 6]


def toss_coin():
    return 'h' if random.randint(0, 1) == 0 else 't'


def st_petersburg_paradox(games):
    total_winnings = 0
    max_winnings = 0

    t1 = time()
    for _ in range(games):
        toss_count = 1
        # continue until toss is tails
        while toss_coin() == 'h':
            toss_count += 1

        winnings = 2**toss_count

        total_winnings += winnings
        if winnings > max_winnings:
            max_winnings = winnings

    print(f"Games: {games}")
    print(f"Avg winnings: ${total_winnings / games}")
    print(f"Max winnings: ${max_winnings}")
    print(f"Time: {round(time()-t1, 4)} s")


def monty_hall(games, strategy="switch"):
    cars_won = 0
    plot_data = []

    for i in range(games):
        # set three goat doors
        doors = {1: 'g', 2: 'g', 3: 'g'}
        # set one randomly to car
        doors.update({random.randint(1, 3): 'c'})

        # guess one randomly
        guess = random.randint(1, 3)

        # random open another door with a goat
        options = [1, 2, 3]
        options.remove(guess)
        door_to_open = random.choice(options)
        if doors[door_to_open] == 'c':
            if door_to_open == options[0]:
                door_to_open = options[1]
            else:
                door_to_open = options[0]
        doors.update({door_to_open: 'og'})

        # apply strategy
        if strategy == "stay" and doors[guess] == 'c':
            cars_won += 1
        elif strategy == "switch" and doors[guess] == 'g':
            cars_won += 1

        # save plotting data
        plot_data.append(round(cars_won/(i+1), 2))

    print(f"Games: {games}")
    print("Strategy:", strategy)
    print(f"Win probability: {round(cars_won / games, 4)}")
    return plot_data


def run_and_plot_mh():
    switch_data = monty_hall(1000, "switch")
    stay_data = monty_hall(1000, "stay")
    x_range = range(0, 1000)
    plt.plot(x_range, switch_data, label="Switch")
    plt.plot(x_range, stay_data, label="Stay")
    plt.xlabel("Games Played")
    plt.ylabel("Win Probability")
    plt.ylim(0, 1)
    plt.title("Win Probability over Games Played of Monty Hall")
    plt.legend()
    plt.show()


def risk_stage(num_a_dice=3, num_d_dice=2):
    if num_a_dice > 3:
        num_a_dice = 3
    if num_d_dice > 2:
        num_d_dice = 2

    a_roll = sorted(random.choices(DIE_SIDES, k=num_a_dice))
    d_roll = sorted(random.choices(DIE_SIDES, k=num_d_dice))

    a_losses = 0
    d_losses = 0
    # highest dice fight
    if a_roll[-1] > d_roll[-1]:
        d_losses += 1
    else:
        a_losses += 1
    # second-highest dice fight
    if min(num_a_dice, num_d_dice) == 2:
        if a_roll[-2] > d_roll[-2]:
            d_losses += 1
        else:
            a_losses += 1

    return a_losses, d_losses


def risk_battle(a_armies, d_armies):
    while a_armies > 1 and d_armies > 0:
        a_deaths, d_deaths = risk_stage(a_armies-1, d_armies)
        a_armies -= a_deaths
        d_armies -= d_deaths
    if a_armies > 1:
        assert d_armies == 0
        return 'a', a_armies
    elif d_armies > 0:
        assert a_armies == 1
        return 'd', d_armies
    else:
        raise Exception("bad battle")


# If the attacker rolls na dice and the defender nd, what are the probabilities
#   of the different outcomes (number of armies lost by each player)?
# Repeat for the different possible combinations of na (1,2,3) and nd (1,2).
def risk_prob_1(games, num_a, num_d):
    a_loses_two = 0
    a_loses_one = 0
    each_lose_one = 0
    d_loses_one = 0
    d_loses_two = 0

    for _ in range(games):
        a_losses, d_losses = risk_stage(num_a, num_d)
        if a_losses == 2:
            a_loses_two += 1
        elif d_losses == 2:
            d_loses_two += 1
        elif a_losses == d_losses == 1:
            each_lose_one += 1
        elif a_losses == 1 and d_losses == 0:
            a_loses_one += 1
        elif a_losses == 0 and d_losses == 1:
            d_loses_one += 1
        else:
            raise Exception("bad results")

    print("Games:", games)
    print("A loses two:", a_loses_two/games)
    print("A loses one:", a_loses_one/games)
    print("Each lose one:", each_lose_one/games)
    print("D loses one:", d_loses_one/games)
    print("D Loses two:", d_loses_two/games)


# 5 defenders, 2-20 attackers
def risk_prob_2(games):
    attacker_win_probabilities = []
    for attackers in range(2, 21):
        wins = 0
        for _ in range(games):
            if risk_battle(attackers, 5)[0] == 'a':
                wins += 1
        attacker_win_probabilities.append(wins/games)

    print(attacker_win_probabilities)
    plt.plot(range(2, 21), attacker_win_probabilities)
    plt.xticks(range(2, 21, 2))
    plt.xlabel("Attacker Armies")
    plt.ylabel("Attacker-Win-Probability")
    plt.title("Attacker-Win-Probability per Number of Attacker Armies")
    plt.show()


# 10 attackers, 10 defenders
def risk_prob_3(games):
    a_wins = [0] * 10
    d_wins = [0] * 10
    for _ in range(games):
        winner, num_armies = risk_battle(10, 10)
        if winner == 'a':
            a_wins[num_armies-1] += 1
        else:
            d_wins[num_armies-1] += 1
    print([a/games for a in a_wins])
    print([d/games for d in d_wins])


if __name__ == '__main__':
    # st_petersburg_paradox(1000000)

    # run_and_plot_mh()

    # risk_prob_1(100000, 1, 1)
    # risk_prob_2(10000)
    risk_prob_3(100000)
