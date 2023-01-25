from riotwatcher import LolWatcher
from tqdm import tqdm

# global variables
api_key = 'RGAPI-66b7a1e5-8b26-4c11-921a-9cc59c4fda2e'
watcher = LolWatcher(api_key)
my_region = 'na1'

# Stephen does diamond, platinum
# Christopher does gold, silver
# Mason does bronze, iron
Queues = [["DIAMOND", "I"], ["DIAMOND", "II"], ["DIAMOND", "III"], ["DIAMOND", "IV"],
          ["PLATINUM", "I"], ["PLATINUM", "II"], ["PLATINUM", "III"], ["PLATINUM", "IV"]]


def get_games():
    old_game_ids = []
    with open('game_ids.txt', "r") as f:
        contents = f.readlines()
        for line in contents:
            id = line[:-1]
            old_game_ids.append(int(id))

    game_ids = []
    for q in tqdm(Queues):
        players = watcher.league.entries(my_region, "RANKED_SOLO_5x5", q[0], q[1], 1)
        for i, player in enumerate(players):
            summoner = watcher.summoner.by_id(my_region, player["summonerId"])
            acc_id = summoner['accountId']
            game_list = watcher.match.matchlist_by_account(my_region, acc_id)
            for j, game in enumerate(game_list['matches']):
                game_ids.append(game["gameId"])
                if j == 4:
                    break
            if i == 9:
                break

    final_game_ids = [ID for ID in game_ids if ID not in old_game_ids]

    with open('game_ids.txt', "w") as f:
        for game in final_game_ids:
            f.write("%s\n" % game)


if __name__ == '__main__':
    get_games()
