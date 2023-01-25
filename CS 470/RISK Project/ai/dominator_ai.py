import random
from risktools import *
# For interacting with interactive GUI
from gui.aihelper import *
from gui.turbohelper import *
import numpy as np


# ## DOMINATOR AI ####
#
#  Complete description for all turn types:
#    - Attack - Attacks only when it has a good chance of winning, and in the most valuable path
#    - PreAssign - Blocks the opponent from controlling a continent, then places in the most 'valuable' territory
#    - Place - Considers all border actions, and places in the location that will set up the most valuable attacks
#    - Occupy - Leaves max troops on the borders
#    - PrePlace, Fortify - Places max troops on the border, evenly spread out
#    - TurnInCards - Chooses the best set to turn in, or won't turn in randomly
#
# This results in an AI that is super strong and will dominate you


def getAction(state, time_left=None):
    """Main AI function.  It should return a valid AI action for this state."""

    # Get the possible actions in this state
    actions = getAllowedActions(state)

    # If only one option, do it
    if len(actions) == 1:
        return actions[0]

    # ############## useful things ##############
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

    # return all of the neighbors of territory_id that aren't owned by the current player
    def get_enemy_neighbors(terr_id, g_state):
        successors = []
        # list all adjacent territories not owned by current player
        potentials = state.board.territories[terr_id].neighbors
        for p in potentials:
            if g_state.owners[p] != state.current_player:
                successors.append(p)
        return successors

    # recursive, returns the value of territories that could be won from the starting territory
    def find_max_path(a_state, current_territory_id, armies, depth_limit, depth, warpath):
        # stop if depth limit reached
        if depth == depth_limit:
            return (get_continent_num(a_state, state.current_player)
                    - get_continent_num(a_state, 1 - state.current_player)), \
                   armies, warpath  # consider the number of armies remaining

        # see if I have enough value (armies) to take another territory
        enemy_arms = state.armies[current_territory_id]
        if armies - 1 - enemy_arms <= 0:
            return (get_continent_num(a_state, state.current_player)
                    - get_continent_num(a_state, 1 - state.current_player)), \
                   armies, warpath  # consider the number of armies remaining

        warpath.append(current_territory_id)
        armies -= 1
        armies -= enemy_arms
        # copy state and pretend I own the current territory
        n_state = a_state.copy_state()
        n_state.owners[current_territory_id] = state.current_player

        successors = get_enemy_neighbors(current_territory_id, n_state)

        # stop if there is nowhere to go from the current territory
        if len(successors) == 0:
            return (get_continent_num(n_state, state.current_player)
                    - get_continent_num(n_state, 1 - state.current_player)), \
                   armies, warpath  # consider the number of armies remaining

        best_value = -np.inf
        best_armies = -np.inf
        best_path = None
        # loop over the neighbors of the current territory
        for s in successors:
            # recursive call
            return_value, return_armies, return_path = find_max_path(n_state, s, armies, depth_limit, depth + 1,
                                                                     warpath)
            if return_value > best_value:
                best_value = return_value
                best_armies = return_armies
                best_path = return_path
            if return_value == best_value and return_armies > best_armies:
                best_armies = return_armies
                best_path = return_path

        return best_value, best_armies, best_path

    # just the continent part of the reinforcement num
    def get_continent_num(c_state, player_id):
        # See if they own all of any continents
        continent_troops = 0
        for continent in state.board.continents.values():
            owned = True
            for terr in continent.territories:
                if c_state.owners[terr] != player_id:
                    owned = False
                    break
            if owned:
                continent_troops += continent.reward

        return continent_troops
    # ############## end useful things ##############

    # Select a Random Action (to use for unspecified turn types)
    my_action = random.choice(actions)

    if state.turn_type == 'Attack':
        # the strategy here is to search for an attack that will provide the most value, with the most armies remaining
        #   value is determined by the difference in reinforcement numbers between me and the opponent

        # the key is the # of enemy troops
        # the value is the # of allied troops I need to win 75+% of the time
        win_dict_75 = {1: 3, 2: 4, 3: 5, 4: 7, 5: 8, 6: 9, 7: 10, 8: 11, 9: 12, 10: 13}

        # consider the amount of territories that could be conquered from this starting spot
        do_nothing_v = (get_continent_num(state, state.current_player) -
                        get_continent_num(state, 1 - state.current_player))
        good_actions = []
        # iterative-deepening search
        # search up to depth 7
        for d_limit in range(1, 8):
            best_v = do_nothing_v
            best_a = -np.inf
            good_actions = [actions[-1]]
            for a in actions:
                if a.to_territory is None:
                    # ignore the "stop attacking" action
                    continue
                start_armies = state.armies[state.board.territory_to_id[a.from_territory]]
                enemy_armies = state.armies[state.board.territory_to_id[a.to_territory]]
                # don't consider attacks that can't be reasonably won
                if enemy_armies <= 10:
                    if start_armies < win_dict_75[enemy_armies]:  # max(2 * enemy_armies - 1, 3)
                        continue
                else:
                    if start_armies < enemy_armies + 4:
                        continue
                path_value, path_armies, _ = find_max_path(state, state.board.territory_to_id[a.to_territory],
                                                           start_armies, d_limit, 0, [])
                if path_value > best_v:
                    best_v = path_value
                    best_a = path_armies
                    good_actions = [a]
                elif path_value == best_v and path_armies > best_a:
                    best_a = path_armies
                    good_actions = [a]
                elif path_value == best_v and path_armies == best_a:
                    good_actions.append(a)
            # if I've found something good, stop searching
            if best_v > do_nothing_v:
                break

        assert len(good_actions) > 0
        # if one has a better value than the rest already, do it
        if len(good_actions) == 1:
            return good_actions[0]

        # consider the head-to-head value of each possible attack action, to see where the easiest wins are
        best = -np.inf
        best_actions = []
        for a in good_actions:
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
            if value > best:
                best = value
                best_actions = [a]
            elif value == best:
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
                        (n_count * 0.01) - enemies_per_continent[c.name]

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
            # Create a list of all actions that move troops to a territory bordering an opponent
            actions = [x for x in actions if is_border_territory(x.to_territory, state.current_player)]
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
        # look at all border actions, and place in the location that will set up the most valuable attacks

        # Create a list of all actions that move troops to a territory bordering an opponent
        border_actions = [x for x in actions if is_border_territory(x.to_territory, state.current_player)]

        # consider the value of territories that could be conquered from this starting spot
        original_v = (get_continent_num(state, state.current_player) -
                      get_continent_num(state, 1 - state.current_player))
        good_actions = []
        handled_territories = []
        # iterative-deepening search
        # search up to depth 7
        for d_limit in range(1, 8):
            best_v = original_v
            best_a = -np.inf
            good_actions = []
            for a in border_actions:
                neighbors = get_enemy_neighbors(state.board.territory_to_id[a.to_territory], state)
                # this returns the value of stuff we can achieve from the start territory
                for n in neighbors:
                    start_armies = state.armies[state.board.territory_to_id[a.to_territory]]
                    if n in handled_territories:
                        # this means that we already have a strong allied territory on the path of this one
                        continue
                    path_value, path_armies, path = find_max_path(state, n,
                                                                  (state.armies[
                                                                       state.board.territory_to_id[a.to_territory]]
                                                                   + state.players[state.current_player].free_armies),
                                                                  d_limit, 0, [])
                    armies_needed = 0
                    for p in path:
                        armies_needed += state.armies[p]
                    if path_value >= best_v and start_armies >= max(2 * armies_needed + len(path), 4):
                        # we already have enough troops here (to attack all territories on path), move on
                        [handled_territories.append(x) for x in path]
                        continue
                    if path_value > best_v:
                        best_v = path_value
                        best_a = path_armies
                        good_actions = [a]
                    elif path_value == best_v and path_armies > best_a:
                        best_a = path_armies
                        good_actions = [a]
                    elif path_value == best_v and path_armies == best_a:
                        good_actions.append(a)
            # if I've found something good, stop searching
            if best_v > original_v:
                break

        # remove duplicate elements
        good_actions = list(dict.fromkeys(good_actions))

        if len(good_actions) == 1:
            return good_actions[0]

        # as a last resort, consider border strength (balance armies on the borders)
        best_actions = []
        best_bs = -np.inf

        if len(good_actions) > 0:
            for a in good_actions:
                new_state = state.copy_state()
                simulatePlaceAction(new_state, a)
                border_strength = calc_border_strength(new_state)

                if border_strength > best_bs:
                    best_actions = [a]
                    best_bs = border_strength
                elif border_strength == best_bs:
                    best_actions.append(a)
        else:  # use all border_actions if there are no 'good' actions
            for a in border_actions:
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
            # for math purposes, cap owned_pct at 0.5
            owned_pct = 0.5 if owned_pct > 0.5 else owned_pct
            # if we don't own many territories, consider turning in cards more
            # if we are doing well, have a 50/50 chance of turning in cards
            if random.random() < owned_pct:
                my_action = actions[-1]

    # Return the chosen action
    return my_action


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
