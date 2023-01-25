import random
from risktools import *
# For interacting with interactive GUI
from gui.aihelper import *
from gui.turbohelper import *
import numpy as np


# ## TEST AI ####
#
#  Complete description for all turn types:  TODO: update some of these
#    - Attack - Attacks in the easiest places, then places on the longest path
#    - PreAssign - Blocks the opponent from controlling a continent, then places in the most 'valuable' territory
#    - Place - Places troops next to desirable territories, then places defensively
#    - Occupy - Leaves max troops on the borders
#    - PrePlace, Fortify - Places max troops on the border, evenly spread out
#    - TurnInCards - Chooses the best set to turn in, or won't turn in randomly
#
# This results in a noob AI that exists only to get dominated


def getAction(state, time_left=None):
    """Main AI function.  It should return a valid AI action for this state."""

    # Get the possible actions in this state
    actions = getAllowedActions(state)

    # If only one option, do it
    if len(actions) == 1:
        return actions[0]

    # ####### useful things ########
    def is_border_territory(t_name, player_number):
        for neighbor in state.board.territories[state.board.territory_to_id[t_name]].neighbors:
            if state.owners[neighbor] != player_number:
                return True
        return False

    # calculate the border strength of the given state (explanation in PrePlace action below)
    def calc_border_strength(b_state):
        # list of the sizes of the armies on the borders
        border_armies = []
        # loop through all the territories
        for terr in b_state.board.territories:
            # loop through the territory's neighbors
            for neighbor in terr.neighbors:
                # if a neighbor territory is owned by the enemy, it is a border territory
                if b_state.owners[neighbor] != b_state.current_player:
                    border_armies.append(b_state.armies[terr.id])
                    break

        result = 1
        for num in border_armies:
            result *= num

        return result

    # Return all of the neighbor territories of territory_id that aren't owned by the current player
    def get_enemy_neighbors(terr_id, g_state):
        successors = []
        # list all adjacent territories not owned by current player
        potentials = state.board.territories[terr_id].neighbors
        for p in potentials:
            if g_state.owners[p] != state.current_player:
                successors.append(p)
        return successors

    # Select a Random Action (to use for unspecified turn types)
    my_action = random.choice(actions)

    if state.turn_type == 'Attack':

        # recursive, returns the longest list of territories that could be won from the starting territory
        def max_atk_path(to_territory_id, val, territories):
            # first see if I have enough value (armies) to take another territory
            enemy_armies = state.armies[to_territory_id]
            if val - enemy_armies <= 0:
                return territories, val
            # expand to this territory
            territories.append(to_territory_id)
            val -= enemy_armies
            best_path, r_val = None, None
            successors = get_enemy_neighbors(to_territory_id, state)
            # remove already visited territories (to avoid cycles)
            successors = [s for s in successors if s not in territories]
            for s in successors:
                path, p_val = max_atk_path(s, val, territories.copy())
                if best_path is None or len(path) > len(best_path):
                    best_path = path
                    r_val = p_val  # return value = path value
            if best_path is None:
                return territories, val
            return best_path, r_val

        # TODO: consider how many 'borders' I have, minimize it
        # TODO: also consider the value of a territory, as it applies to continent bonuses (mine or theirs)

        # First consider the head-to-head value of each possible attack action, to see where the easiest wins are
        best = -np.inf
        good_actions = []
        for a in actions:
            if a.from_territory is not None:
                # minus 1 since one army has to stay back
                value = state.armies[state.board.territory_to_id[a.from_territory]] - 1 \
                        - state.armies[state.board.territory_to_id[a.to_territory]]
            else:
                # this is the "don't attack" action
                if not state.players[state.current_player].conquered_territory:
                    # if I haven't conquered a territory, don't consider not attacking as much
                    value = -2
                else:
                    value = 1
            # track the highest value, cap it at 4
            if value >= 4:
                if best < 4:
                    # this is the first time here, restart the list
                    good_actions = [a]
                    best = 4
                else:
                    # grow the list
                    good_actions.append(a)
            elif value > best:
                best = value
                good_actions = [a]
            elif value == best:
                good_actions.append(a)

        # if one has a better value than the rest already, do it
        if len(good_actions) == 1:
            return good_actions[0]

        # otherwise, consider the amount of territories that could be conquered from this starting spot
        best_path_length = -1
        best_actions = None
        for a in good_actions:
            if a.to_territory is None:
                # ignore the "stop attacking" action
                continue
            # this returns the longest path that we can reach from the start territory
            max_path, _ = max_atk_path(state.board.territory_to_id[a.to_territory],
                                       state.armies[state.board.territory_to_id[a.from_territory]],
                                       [])
            if len(max_path) > best_path_length:
                best_path_length = len(max_path)
                best_actions = [a]
            elif len(max_path) == best_path_length:
                best_actions.append(a)

        my_action = random.choice(best_actions)

    elif state.turn_type == 'PreAssign':
        # first, look to block the enemy from controlling entire continents by getting the last territory of a continent
        # Then, consider the value of each potential territory and choose the most valuable
        #   the territory value is calculated as:
        #       the continent reward value that you get per territory
        #           (i.e., since Australia gives two armies, and there are 4 territories there, each is worth 0.5)
        #       minus the number of enemies already in the continent
        #       plus a factor of how many neighboring territories are owned by me

        # ######## block the enemy from getting the last territory of a continent

        # loop through the continents
        for c in state.board.continents.values():
            empty_territory = None
            for t in c.territories:
                # if I own a territory in the continent, proceed to next continent
                if state.owners[t] == state.current_player:
                    break
                # if empty territory...
                elif state.owners[t] is None:
                    # save the empty territory
                    if empty_territory is None:
                        # one empty territory found...
                        empty_territory = state.board.territories[t].name
                    # proceed to next continent if multiple empty territories
                    else:
                        # multiple empty territories found...
                        break
                # if territory owned by enemy, do nothing
            # this means that we found only one empty territory in the continent,
            #       and we don't own any of it
            # In other words, we need to block the opponent from controlling the whole continent
            else:  # in for... else, else means 'no break'
                if empty_territory is not None:
                    return RiskAction('PreAssign', empty_territory, None, None)

        # ######## place in the most 'valuable' territory

        # the number of enemies in each continent
        enemies_per_continent = {}
        for c in state.board.continents.values():
            enemies = 0
            for t in c.territories:
                if state.owners[t] == 1 - state.current_player:
                    # increment if the opposition owns the territory
                    enemies += 1
            enemies_per_continent[c.name] = enemies

        # this dictionary says the 'value' of each territory
        val_per_territory = {}
        for c in state.board.continents.values():
            for t in c.territories:
                if state.owners[t] is None:
                    # consider the value of the territory only if it's empty

                    # count number of owned neighbors
                    n_count = 0
                    for n in state.board.territories[t].neighbors:
                        if state.owners[n] == state.current_player:
                            n_count += 1

                    val_per_territory[state.board.territories[t].name] = c.reward / len(c.territories) + \
                                                                         (n_count * 0.01) - enemies_per_continent[
                                                                             c.name]

        best_val = -np.inf
        good_territories = []
        for territory, value in val_per_territory.items():
            if value > best_val:
                good_territories = [territory]
                best_val = value
            elif value == best_val:
                good_territories.append(territory)

        # randomly select one of these good territories
        # we know these are all valid actions, since these territories don't have owners
        my_action = RiskAction('PreAssign', random.choice(good_territories), None, None)

    elif state.turn_type == 'PrePlace' or state.turn_type == 'Fortify':
        # consider the "border strength" that we would get from each possible Fortify or PrePlace action
        # I define the "border strength" as the product of all the armies on the borders
        # this is more useful than the sum because it incentivizes evenly spread-out armies
        #       for example the product of [3, 3, 3, 3] > the product of [4, 3, 3, 2]

        if state.turn_type == 'Fortify':
            # start with the "don't fortify" action
            good_actions = [actions[-1]]
            # the border strength of this action is just the border strength of the original state
            best_bs = calc_border_strength(state)
        else:  # PrePlace
            # since there is no "do nothing" action, start with empty list
            good_actions = []
            best_bs = -np.inf

        for a in actions:
            # a.to_territory is None on the last fortify action (don't fortify)
            if a.to_territory is not None:
                new_state = state.copy_state()
                if state.turn_type == 'Fortify':
                    simulateFortifyAction(new_state, a)
                else:  # PrePlace
                    simulatePrePlaceAction(new_state, a)
                border_strength = calc_border_strength(new_state)

                if border_strength > best_bs:
                    good_actions = [a]
                    best_bs = border_strength
                elif border_strength == best_bs:
                    good_actions.append(a)

        # Randomly select one of these actions
        if len(good_actions) > 0:
            my_action = random.choice(good_actions)

    elif state.turn_type == 'Place':
        # First, it narrows down the available actions to border actions
        # Then, it considers the offensive value of the action (can I win a continent from this territory?
        # Next, consider the defensive value of the action (would the opponent owning this territory win a continent?)

        # Create a list of all actions that move troops to a territory bordering an opponent
        border_actions = [x for x in actions if is_border_territory(x.to_territory, state.current_player)]

        # loop over enemy border territories
        #       if I have a neighboring territory with at least max(2*e+1, 4) troops, continue
        #       otherwise, place in the highest

        # consider offensive value of the territory
        # the offensive value is the increase in continent reward if you conquer a certain territory from here
        best_o_value = 0
        offensive_actions = []
        for a in border_actions:
            a_neighbors = state.board.territories[state.board.territory_to_id[a.to_territory]].neighbors
            # o_value can also be thought of as the continent_troops part of my reinforcement num
            o_value = 0
            neighbor_territory_armies = None
            for c in state.board.continents.values():
                if state.board.territory_to_id[a.to_territory] not in c.territories:
                    # only consider the continent of the territory in question
                    continue
                # owned by the current_player
                owned = True
                for t in c.territories:
                    if state.owners[t] != state.current_player:
                        if neighbor_territory_armies is None and t in a_neighbors:
                            # ignore just one neighbor of the territory in question
                            # store the number of armies, so we know how many to place to be able to take it
                            neighbor_territory_armies = state.armies[t]
                        else:
                            # consider what the o_value would be IF I were to win a neighbor of 'a'
                            owned = False
                            break
                if owned and neighbor_territory_armies is not None:
                    # if neighbor_territory_armies is None, then I own the continent already without conquering anything
                    o_value = c.reward
            if o_value > best_o_value:
                offensive_actions = [(a, neighbor_territory_armies)]
                best_o_value = o_value
            elif o_value == best_o_value:
                offensive_actions.append((a, neighbor_territory_armies))

        # if this has already limited it to one action, do it
        if len(offensive_actions) == 1:
            return offensive_actions[0][0]

        if len(offensive_actions) < len(border_actions):
            # if I found some offensive actions, then limit it to one per continent
            #    We found one territory to take -> stack armies on just one territory to take it

            best_actions = []
            included_continents = []
            # reduce it to one territory per continent
            for a, v in reversed(offensive_actions):
                for c in state.board.continents.values():
                    if c.name in included_continents:
                        continue
                    for t in c.territories:
                        if t == state.board.territory_to_id[a.to_territory]:
                            best_actions.append((a, v))
                            included_continents.append(c.name)
                            break

            # see if these territories already have enough troops
            final_actions = []
            for a, v in best_actions:
                if state.armies[state.board.territory_to_id[a.to_territory]] < max(2 * v + 1, 4):
                    final_actions.append(a)

            if len(final_actions) > 0:
                return random.choice(final_actions)

        # the following will only be useful if the offensive action step did NOT lower the actions
        #     one territory can't be offensively valuable and defensibly valuable at the same time (as I define it)
        # consider defensive value of the territory (would the opponent owning this territory win a continent?)
        # place troops (defend) in territories that are most desirable for the opposition
        best_d_value = -np.inf
        defensive_actions = []
        for a in border_actions:
            # d_value can also be thought of as the continent_troops part of the opposition's reinforcement num
            d_value = 0
            for c in state.board.continents.values():
                if state.board.territory_to_id[a.to_territory] not in c.territories:
                    # only consider the continent of the territory in question
                    continue
                # owned by the opposition, not current_player
                owned = True
                for t in c.territories:
                    if state.owners[t] == state.current_player:
                        if t != state.board.territory_to_id[a.to_territory]:  # ignore the territory in question
                            # consider what the d_value would be IF the opposition were to win territory t
                            owned = False
                            break
                if owned:
                    d_value += c.reward
            if d_value > best_d_value:
                defensive_actions = [a]
                best_d_value = d_value
            elif d_value == best_d_value:
                defensive_actions.append(a)

        # if this has already limited it to one action, do it
        if len(defensive_actions) == 1:
            return defensive_actions[0]

        # TODO: maybe consider border strength a little, then stack up armies somewhere
        # as a last resort, consider border strength (balance armies on the borders)
        best_actions = []
        best_bs = -np.inf

        for a in defensive_actions:
            new_state = state.copy_state()
            simulatePlaceAction(new_state, a)
            border_strength = calc_border_strength(new_state)

            if border_strength > best_bs:
                best_actions = [a]
                best_bs = border_strength
            elif border_strength == best_bs:
                best_actions.append(a)

        # Randomly select one of these actions, if there were any
        if len(best_actions) > 0:
            my_action = random.choice(best_actions)

    elif state.turn_type == 'Occupy':
        if not is_border_territory(actions[0].from_territory, state.current_player):
            # if I attacked from an interior territory, occupy max troops
            my_action = actions[-1]
        else:
            # if I attacked from a border territory...
            if not is_border_territory(actions[0].to_territory, state.current_player):
                # ... and I attacked an interior territory, occupy the min allowed troops (normally 3)
                my_action = actions[0]
            else:
                # ... and I attacked a border territory, leave 2 troops back
                my_action = actions[-2]  # I am guaranteed to always have at least 2 actions

    elif state.turn_type == 'TurnInCards':
        # the idea here is to turn in sets with territories that I have on them
        # also, don't turn in cards if you are doing well (so their value increases)
        # if we are losing, favor turning in cards sooner

        best_sets = []  # list of sets that are all equally 'valuable'
        best_val = -100
        none_option = False

        for a in actions:
            value = 0

            # this is the option to not turn in a set, which we want to consider
            if a.from_territory is None:
                none_option = True
                continue

            # we want to use sets with our territories on the cards
            if state.owners[state.board.cards[a.from_territory].territory] == state.current_player \
                    or state.owners[state.board.cards[a.to_territory].territory] == state.current_player \
                    or state.owners[state.board.cards[a.troops].territory] == state.current_player:
                value += 2

            # we would rather not use a wildcard if we can do without
            if state.board.cards[a.from_territory].picture == 'Wildcard' \
                    or state.board.cards[a.to_territory].picture == 'Wildcard' \
                    or state.board.cards[a.troops].picture == 'Wildcard':
                value -= 1

            if value > best_val:
                best_sets = [a]
                best_val = value
            elif value == best_val:
                best_sets.append(a)

        # apply some randomness!!!
        my_action = random.choice(best_sets)

        # if we have the option of doing nothing...
        if none_option:
            # determine how much of the board we own
            total = 0.0
            mine = 0.0
            for t in state.board.territories:
                total += 1.0
                if state.owners[t.id] == state.current_player:
                    mine += 1.0
            owned_pct = mine / total
            owned_pct = 0.5 if owned_pct > 0.5 else owned_pct
            # if we don't own many territories, consider turning in cards more
            # if we are doing well, have a 50/50 chance of turning in cards
            if random.random() < owned_pct:
                my_action = actions[-1]

    # Return the chosen action
    return my_action


# def pre_assign_score(state):
#     return getReinforcementNum(state, state.current_player) - getReinforcementNum(state, 1-state.current_player)
#

# def pre_assign_max(state, alpha, beta, depth):
#     best_actions = []
#
#     if depth == 3:
#         return pre_assign_score(state), None
#
#     available_actions = getPreAssignActions(state)
#     if len(available_actions) == 0:
#         return pre_assign_score(state), None
#
#     best_v = -np.inf
#     for a in available_actions:
#         new_state = state.copy_state()
#         simulatePreAssignAction(new_state, a)
#         v, _ = pre_assign_min(new_state, alpha, beta, depth+1)
#         if v > best_v:
#             best_v = v
#             best_actions = [a]
#         if v == best_v:
#             best_actions.append(a)
#         if v > beta:  # prune
#             return v, None
#         alpha = max(alpha, v)
#     return best_v, best_actions
#
#
# def pre_assign_min(state, alpha, beta, depth):
#     best_actions = []
#
#     if depth == 3:
#         return pre_assign_score(state), None
#
#     available_actions = getPreAssignActions(state)
#     if len(available_actions) == 0:
#         return pre_assign_score(state), None
#
#     best_v = np.inf
#     for a in available_actions:
#         new_state = state.copy_state()
#         simulatePreAssignAction(new_state, a)
#         v, _ = pre_assign_max(new_state, alpha, beta, depth+1)
#         if v < best_v:
#             best_v = v
#             best_actions = [a]
#         if v == best_v:
#             best_actions.append(a)
#         if v < alpha:  # prune
#             return v, None
#         beta = min(beta, v)
#     return best_v, best_actions


# Code below this is the interface with Risk.pyw GUI version
# DO NOT MODIFY

def aiWrapper(function_name, occupying=None):
    game_board = createRiskBoard()
    game_state = createRiskState(game_board, function_name, occupying)
    action = getAction(game_state)
    return translateAction(game_state, action)


def Assignment(player):
    # Need to Return the name of the chosen territory
    return aiWrapper('Assignment')


def Placement(player):
    # Need to return the name of the chosen territory
    return aiWrapper('Placement')


def Attack(player):
    # Need to return the name of the attacking territory, then the name of the defender territory
    return aiWrapper('Attack')


def Occupation(player, t1, t2):
    # Need to return the number of armies moving into new territory
    occupying = [t1.name, t2.name]
    return aiWrapper('Occupation', occupying)


def Fortification(player):
    return aiWrapper('Fortification')
