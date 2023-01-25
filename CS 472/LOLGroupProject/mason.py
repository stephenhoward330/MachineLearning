import pandas as pd


def get_account_id_from_match_info(match):
    players = []
    identities = match["participantIdentities"]
    for player in identities:
        players.append(player["player"]["accountId"])
    return players


def mason(match_info):
    original_game = match_info[0][0]
    account_ids = get_account_id_from_match_info(original_game)
    team_account = []
    player_dfs = []
    for i, player_matches in enumerate(match_info[1:]):
        player_account_id = account_ids[i]
        player_stats = []
        team_id = None
        # print('on player ' + str(i))
        for j, match_detail in enumerate(player_matches):
            # print('on match ' + str(j))
            row = {}

            participant_ids = match_detail['participantIdentities']
            participantId = None
            for participant_id in participant_ids:
                if participant_id['player']['currentAccountId'] == player_account_id:
                    # print('found player in match!')
                    participantId = participant_id['participantId']
            participants = match_detail['participants']
            for participant in participants:
                if participant['participantId'] == participantId:
                    row['damageDealtToObjectives'] = participant['stats']['damageDealtToObjectives']
                    row['kills'] = participant['stats']['kills']
                    row['totalHeal'] = participant['stats']['totalHeal']
                    row['win'] = int(participant['stats']['win'])
                    team_id = participant['teamId']
                if participant['stats']['deaths'] > 0:
                    row['damage_per_death'] = participant['stats']['totalDamageTaken' ] /participant['stats']['deaths']
                else:
                    row['damage_per_death'] = participant['stats']['totalDamageTaken']
                player_stats.append(row)
        df = pd.DataFrame(player_stats)
        df = df.mean(axis=0)
        df['currentAccountId'] = player_account_id
        df['teamId'] = team_id
        player_dfs.append(df)
    combined_df = pd.DataFrame(player_dfs)
    combined_df = combined_df.groupby(['teamId']).mean()
    diff_series = (combined_df.loc[100] - combined_df.loc[200])
    return diff_series.to_dict()
