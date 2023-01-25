import sys
import risktools
from queue import PriorityQueue


# ##### risk_search.py
#      This script will use Uniform Cost Search to determine if an attacking path exists in 
#      the game of RISK from a source territory (owned by the acting player)
#      to a destination territory (owned by an opposing player)
#      through territories owned by opposing players.  
#      A sequence of successful attacks could follow this path and occupy the destination territory
#      The students can explore different cost functions for the paths
#      It will do this for a given state from a logfile.
#      You can get the state number from the risk_game_viewer 
# #####

class SearchNode:
    """
    A data structure to store the nodes in our search tree
    """

    def __init__(self, territory_id, parent, step_cost):
        self.id = territory_id
        self.parent = parent
        self.cost = 0
        if parent:
            self.cost = parent.cost + step_cost
        else:
            self.cost = step_cost

    def __eq__(self, other):
        return self.cost == other.cost

    def __ne__(self, other):
        return self.cost != other.cost

    def __lt__(self, other):
        return self.cost < other.cost

    def __le__(self, other):
        return self.cost <= other.cost

    def __gt__(self, other):
        return self.cost > other.cost

    def __ge__(self, other):
        return self.cost >= other.cost


def find_best_source(dst, state):
    # get a list of possible sources
    start_territories = []
    for t in state.board.territories:
        if state.owners[t.id] == state.current_player:
            start_territories.append(t)

    # find best src node to dst
    best_node = None
    best_cost = 999999
    for src in start_territories:
        res = run_search(src.id, dst, state, verbose=False)
        if res is not None:
            if res.cost < best_cost:
                best_cost = res.cost
                best_node = res

    # print details
    if best_node is None:
        print("No suitable src node found!")
    else:
        cur = best_node
        path = []

        # Extract the path and print it out
        while cur is not None:
            path.append(cur.id)
            cur = cur.parent
        path.reverse()

        print('PATH FOUND! Cost =', best_cost)
        for p in path:
            print(' [', p, ']', state.board.territories[p].name)


# The actual search function
def search(fringe, goal, state):
    # List to store the expanded states on this search
    expanded = []

    # Continue until the fringe is empty
    while not fringe.empty():
        node = fringe.get()
        if node.id == goal:
            return node
        expanded.append(node.id)
        adj_states = get_successors(node.id, state)
        for a in adj_states:
            if a not in expanded:
                # new_node = SearchNode(a, node, 1.0)  # same cost for each territory
                new_node = SearchNode(a, node, state.armies[a])  # cost is equal to the armies in the new territory
                fringe.put(new_node)

    # If no path found, return none
    return None


# Set up and run a search from src territory to dst territory in state
def run_search(src, dst, state, verbose=True):
    root = SearchNode(src, None, state.armies[src] * -1)  # cost formerly 0
    fringe = PriorityQueue()
    fringe.put(root)
    goal = search(fringe, dst, state)
    if goal:
        cur = goal
        path = []
        # Extract the path and print it out 
        while cur is not None:
            path.append(cur.id)
            cur = cur.parent

        # Reverse the path
        path.reverse()

        if verbose:
            print('PATH FOUND! Cost =', goal.cost)
            for p in path:
                print(' [', p, ']', state.board.territories[p].name)
    else:
        if verbose:
            print('NO PATH FOUND!')

    return goal


def get_successors(territory_id, state):
    """Return all of the neighbor territories of territory_id that aren't owned by the 
       current player in the given state"""
    successors = []

    # list all adjacent territories not owned by current player
    potentials = state.board.territories[territory_id].neighbors
    for p in potentials:
        if state.owners[p] != state.current_player:
            successors.append(p)

    return successors


def print_usage():
    print('USAGE: python risk_search.py log_filename state_num source_territory destination_territory')


if __name__ == "__main__":
    # Get ais from command line arguments
    if len(sys.argv) != 5:
        print_usage()
        sys.exit()

    # get log file name
    log_filename = sys.argv[1]

    # Open the logfile
    logfile = open(log_filename, 'r')

    # Get the state number
    state_number = int(sys.argv[2])

    # Get a state that we can use
    risk_board = risktools.loadBoard('world.zip')
    search_state = risktools.getInitialState(risk_board)

    # Get the source territory
    try:
        source_territory = risk_board.territory_to_id[sys.argv[3]]
    except:
        print(sys.argv[3], "not recognized as a valid territory! Exiting.")
        sys.exit()

    # Get the destination territory
    try:
        destination_territory = risk_board.territory_to_id[sys.argv[4]]
    except:
        print(sys.argv[4], "not recognized as a valid territory! Exiting.")
        sys.exit()

    print('Risk_search is searching from ', risk_board.territories[source_territory].name, 'to',
          risk_board.territories[destination_territory].name, 'in logfile', log_filename, 'state', state_number)

    # Get the relevant state from the file
    state_counter = 0
    while state_counter < state_number + 1:
        newline = logfile.readline()
        splitline = newline.split('|')
        if not newline or splitline[0] == 'RISKRESULT':
            print('End of logfile reached, state not found!  Only', state_counter,
                  'states encountered.  State number was', state_number)
            sys.exit()
        if splitline[0] == 'RISKSTATE':
            if state_counter == state_number:
                search_state.from_string(newline, risk_board)
                break
            state_counter += 1

    # Close the logfile
    logfile.close()

    # Now plan from source to destination
    run_search(source_territory, destination_territory, search_state)
    # find_best_source(destination_territory, search_state)
