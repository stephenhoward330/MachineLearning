import pandas as pd


# returns the account IDs for the ten players in one match
def get_player_dict(match_detail):
    players = {'100': [match_detail['participantIdentities'][0]['player']['accountId'],
                       match_detail['participantIdentities'][1]['player']['accountId'],
                       match_detail['participantIdentities'][2]['player']['accountId'],
                       match_detail['participantIdentities'][3]['player']['accountId'],
                       match_detail['participantIdentities'][4]['player']['accountId']],
               '200': [match_detail['participantIdentities'][5]['player']['accountId'],
                       match_detail['participantIdentities'][6]['player']['accountId'],
                       match_detail['participantIdentities'][7]['player']['accountId'],
                       match_detail['participantIdentities'][8]['player']['accountId'],
                       match_detail['participantIdentities'][9]['player']['accountId']]
               }

    return players


# finds the participant ID and team ID for one player in one game
def find_participant_num(detail, acc_id):
    for participant in detail['participantIdentities']:
        if participant['player']['accountId'] == acc_id:
            num = participant['participantId']
            break
    for participant in detail['participants']:
        if participant['participantId'] == num:
            team_num = participant['teamId']
            break
    return num-1, team_num


# returns a dict of the team_id and average stats for one player over 'back_matches' games
def get_player_stats(acc_id, team_id, matches):

    stats = []
    for match in matches:
        p_num, team_num = find_participant_num(match, acc_id)
        stat = {'assists': match['participants'][p_num]['stats']['assists'],
                'timeCCingOthers': match['participants'][p_num]['stats']['timeCCingOthers'],
                'visionScore': match['participants'][p_num]['stats']['visionScore'],
                'towerKills': match['teams'][(0 if team_num == 100 else 1)]['towerKills'],
                'dragonKills': match['teams'][(0 if team_num == 100 else 1)]['dragonKills']
                }
        stats.append(stat)

    first_df = pd.DataFrame(stats)
    average = {
        'teamId': team_id,
        'ave_assists': first_df['assists'].mean(),
        'ave_timeCCingOthers': first_df['timeCCingOthers'].mean(),
        'ave_visionScore': first_df['visionScore'].mean(),
        'ave_towerKills': first_df['towerKills'].mean(),
        'ave_dragonKills': first_df['dragonKills'].mean()
    }

    return average


# returns one row of the data
def stephen(list_of_details):
    first_match = list_of_details[0][0]

    # get the ten players split into two teams
    players = get_player_dict(first_match)

    # get average stats for all ten players
    participants = []
    for i, (team, acc_ids) in enumerate(players.items()):
        for j, acc_id in enumerate(acc_ids):
            participants.append(get_player_stats(acc_id, team, list_of_details[(i * 5) + j + 1]))

    participants_df = pd.DataFrame(participants)

    final_dict = {
        'assists_diff': round(participants_df.groupby('teamId')['ave_assists'].mean()['100'] -
                              participants_df.groupby('teamId')['ave_assists'].mean()['200'], 4),
        'CC_diff': round(participants_df.groupby('teamId')['ave_timeCCingOthers'].mean()['100'] -
                         participants_df.groupby('teamId')['ave_timeCCingOthers'].mean()['200'], 4),
        'visionScore_diff': round(participants_df.groupby('teamId')['ave_visionScore'].mean()['100'] -
                                  participants_df.groupby('teamId')['ave_visionScore'].mean()['200'], 4),
        'towerKills_diff': round(participants_df.groupby('teamId')['ave_towerKills'].mean()['100'] -
                                 participants_df.groupby('teamId')['ave_towerKills'].mean()['200'], 4),
        'dragonKills_diff': round(participants_df.groupby('teamId')['ave_dragonKills'].mean()['100'] -
                                  participants_df.groupby('teamId')['ave_dragonKills'].mean()['200'], 4),
        'won?': (True if first_match['teams'][0]['win'] == 'Win' else False)
    }

    return final_dict
