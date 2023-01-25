from riotwatcher import LolWatcher
import pandas as pd
import stephen
import mason
import christopher
from tqdm import tqdm

# global variables
api_key = 'RGAPI-66b7a1e5-8b26-4c11-921a-9cc59c4fda2e'
watcher = LolWatcher(api_key)
my_region = 'na1'
back_matches = 5


def get_players(match_detail):
    players = [
        match_detail['participantIdentities'][0]['player']['accountId'],
        match_detail['participantIdentities'][1]['player']['accountId'],
        match_detail['participantIdentities'][2]['player']['accountId'],
        match_detail['participantIdentities'][3]['player']['accountId'],
        match_detail['participantIdentities'][4]['player']['accountId'],
        match_detail['participantIdentities'][5]['player']['accountId'],
        match_detail['participantIdentities'][6]['player']['accountId'],
        match_detail['participantIdentities'][7]['player']['accountId'],
        match_detail['participantIdentities'][8]['player']['accountId'],
        match_detail['participantIdentities'][9]['player']['accountId']]

    return players


def get_player_games(acc_id):
    matches = watcher.match.matchlist_by_account(my_region, acc_id)

    games = []
    for i in range(back_matches):
        match = matches['matches'][i+1]
        detail = watcher.match.by_id(my_region, match['gameId'])
        games.append(detail)

    return games


# 3220617285
if __name__ == '__main__':
    # test = [{'thing1': 5, 'thing2': 10}]
    # test_pd = pd.DataFrame(test)
    # test_pd.to_csv('test.csv', index=False)
    #
    # test2 = [{'thing1': 15, 'thing2': 20}]
    # test2_pd = pd.DataFrame(test2)
    # test2_pd.to_csv('test.csv', mode='a', index=False, header=False)

    master_pd = None
    game_ids = []
    with open('game_ids.txt', "r") as f:
        contents = f.readlines()
        for line in contents:
            id = line[:-1]
            game_ids.append(int(id))

    num_failed = 0
    for game_id in tqdm(game_ids):
        try:
            # print("trying:", game_id)
            detail = watcher.match.by_id(my_region, game_id)
            players = get_players(detail)

            final = [[detail]]
            for player in players:
                final.append(get_player_games(player))

            # get_data.get_data_instance()
            final_data = [{**stephen.stephen(final), **christopher.christopher(final), **mason.mason(final)}]
            row_pd = pd.DataFrame(final_data)

            if master_pd is None:
                master_pd = pd.DataFrame(row_pd)
            else:
                master_pd = master_pd.append(row_pd)
        except Exception as e:
            num_failed += 1
            continue

    # print(final_data)
    print(f'Success: {len(game_ids)-num_failed}/{len(game_ids)}')
    master_pd.to_csv('dataset.csv', mode='a', header=False, index=False)
