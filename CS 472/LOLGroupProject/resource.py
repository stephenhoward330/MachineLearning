from riotwatcher import LolWatcher
import pandas as pd

# DON'T DO MORE THAN:
#     20 requests every 1 second
#     100 requests every 2 minutes

# global variables
api_key = 'RGAPI-69d3804c-9bce-47c5-a269-e3399c0f586e'
watcher = LolWatcher(api_key)
my_region = 'na1'

if __name__ == '__main__':

    me = watcher.summoner.by_name(my_region, 'Trombone3')
    # print(me)

    # Return the rank stats
    # my_ranked_stats = watcher.league.by_summoner(my_region, me['id'])
    # print(my_ranked_stats)

    my_matches = watcher.match.matchlist_by_account(my_region, me['accountId'])

    # fetch last match detail
    last_match = my_matches['matches'][0]
    # print(last_match['gameId'])
    match_detail = watcher.match.by_id(my_region, last_match['gameId'])

    # check league's latest version
    latest = watcher.data_dragon.versions_for_region(my_region)['n']['champion']
    # Lets get some champions static information
    static_champ_list = watcher.data_dragon.champions(latest, False, 'en_US')

    # champ static list data to dict for looking up
    champ_dict = {}
    for key in static_champ_list['data']:
        row = static_champ_list['data'][key]
        champ_dict[row['key']] = row['id']

    participants = []
    for row in match_detail['participants']:
        participants_row = {'teamID': row['teamId'],
                            'champion': champ_dict[str(row['championId'])],
                            # 'spell1': row['spell1Id'],
                            # 'spell2': row['spell2Id'],
                            # 'win': row['stats']['win'],
                            # 'kills': row['stats']['kills'],
                            # 'deaths': row['stats']['deaths'],
                            'assists': row['stats']['assists'],
                            # 'totalDamageDealt': row['stats']['totalDamageDealt'],
                            # 'goldEarned': row['stats']['goldEarned'],
                            # 'champLevel': row['stats']['champLevel'],
                            # 'totalMinionsKilled': row['stats']['totalMinionsKilled'],
                            # 'item0': row['stats']['item0'],
                            # 'item1': row['stats']['item1'],
                            # 'timeCCDealt': row['stats']['totalTimeCrowdControlDealt'],
                            'timeCCingOthers': row['stats']['timeCCingOthers'],
                            'visionScore': row['stats']['visionScore']
                            }
        participants.append(participants_row)
    participants_df = pd.DataFrame(participants)

    condensed_participants = [{
        'teamID': 100,
        'assists': participants_df.groupby('teamID')['assists'].sum()[100],
        'timeCCingOthers': participants_df.groupby('teamID')['timeCCingOthers'].sum()[100],
        'visionScore': participants_df.groupby('teamID')['visionScore'].sum()[100]
    }, {
        'teamID': 200,
        'assists': participants_df.groupby('teamID')['assists'].sum()[200],
        'timeCCingOthers': participants_df.groupby('teamID')['timeCCingOthers'].sum()[200],
        'visionScore': participants_df.groupby('teamID')['visionScore'].sum()[200]
    }]

    con_participants_df = pd.DataFrame(condensed_participants)

    teams = []
    for row in match_detail['teams']:
        teams_row = {'teamID': row['teamId'],
                     'win': True if row['win'] == 'Win' else False,
                     'dragonKills': row['dragonKills'],
                     'towerKills': row['towerKills']
                     }
        teams.append(teams_row)
    teams_df = pd.DataFrame(teams)

    match = [{
        'assistsDiff': con_participants_df['assists'][0] - con_participants_df['assists'][1],
        'CCDiff': con_participants_df['timeCCingOthers'][0] - con_participants_df['timeCCingOthers'][1],
        'visionScoreDiff': con_participants_df['visionScore'][0] - con_participants_df['visionScore'][1],
        'dragonKillsDiff': teams_df['dragonKills'][0] - teams_df['dragonKills'][1],
        'towerKillsDiff': teams_df['towerKills'][0] - teams_df['towerKills'][1],
        'Win': teams_df['win'][0]
    }]

    match_df = pd.DataFrame(match)

    print("done")

    # (MID_LANE, SOLO): MIDDLE
    # (TOP_LANE, SOLO): TOP
    # (JUNGLE, NONE): JUNGLE
    # (BOT_LANE, DUO_CARRY): BOTTOM
    # (BOT_LANE, DUO_SUPPORT): UTILITY