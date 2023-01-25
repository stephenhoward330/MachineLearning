# from riotwatcher import LolWatcher
import numpy as np
import pandas as pd

# api = LolWatcher("RGAPI-aac76f13-06ae-4a95-bced-496b845d4bc6")
# current_region = "na1"
# num_past_games = 5


# def make_dataframe_from_game_id(game_id):
#     original_game_info = get_match_info_for_match_ids([game_id])[0]
#     account_ids = get_account_id_from_match_info(original_game_info)
#     last_games = get_last_matches_from_player_list(account_ids)
#     match_info_per_player = []
#     for game_list in last_games:
#         match_info_per_player.append(get_match_info_for_match_ids(game_list))
#     return make_dataframe_from_match_info(match_info_per_player,account_ids)


def make_dataframe_from_all_match_info(match_info_per_player, account_ids):
    dataframes = []
    for i, info_list in enumerate(match_info_per_player):
        dataframes.append(make_dataframe_from_match_info(info_list, account_ids[i]))
    result_frames = list(dataframes[0])
    for i in range(1, len(dataframes)):
        current_frame = dataframes[i]
        for j in range(len(result_frames)):
            result_frames[j] = result_frames[j].append(current_frame[j])
    return result_frames


def get_account_id_from_match_info(match):
    players = []
    identities = match["participantIdentities"]
    for player in identities:
        players.append(player["player"]["accountId"])
    return players


# def get_last_matches_from_player_list(players):
#     last_matches = []
#     for player in players:
#         last_matches.append(get_past_x_game_ids(player, 0))
#     return last_matches


# def get_past_x_game_ids(account_id, start_match_id):
#     matches = api.match.matchlist_by_account(current_region, account_id, begin_index=start_match_id, end_index=start_match_id + num_past_games)
#     ids = []
#     for match in matches["matches"]:
#         ids.append(match["gameId"])
#     return ids


# def get_match_info_for_match_ids(matches):
#     match_info = []
#     for match_id in matches:
#         match_info.append(api.match.by_id(current_region, match_id))
#     return match_info


def make_dataframe_from_match_info(matches, account_id):
    player_identity_attributes = [["participantId"],
                                  ["player", "accountId"]]
    player_participant_attributes = [["participantId"],
                                     ["teamId"],
                                     ["stats", "champLevel"],
                                     ["stats", "goldSpent"],
                                     ["stats", "deaths"],
                                     ["stats", "totalDamageDealtToChampions"],
                                     ["stats", "totalMinionsKilled"]]
    team_attributes = [["teamId"], ["baronKills"]]
    players_data = []
    player_identity_data = []
    team_data = []
    for match in matches:
        participant_id = 0
        game_id = match["gameId"]
        player_identities = match["participantIdentities"]
        for player in player_identities:
            if account_id is None or player["player"]["accountId"] == account_id:
                current_data = get_info(player, player_identity_attributes)
                current_data["gameId"] = game_id
                player_identity_data.append(current_data)
                participant_id = player["participantId"]
        teams = match["teams"]
        for team in teams:
            current_data = get_info(team, team_attributes)
            current_data["gameId"] = game_id
            team_data.append(current_data)

        player_stats = match["participants"]
        for player in player_stats:
            if account_id is None or player["participantId"] == participant_id:
                current_data = get_info(player, player_participant_attributes)
                current_data["gameId"] = game_id
                players_data.append(current_data)

    players_dataframe = pd.DataFrame(players_data)
    player_identities_dataframe = pd.DataFrame(player_identity_data)
    team_dataframe = pd.DataFrame(team_data)
    return player_identities_dataframe, players_dataframe, team_dataframe


def get_info(dictionary, attributes):
    data = {}
    for attribute in attributes:
        current_info = dictionary
        for i in range(0, len(attribute)):
            current_info = current_info[attribute[i]]
        data[attribute[i]] = current_info
    return data


def get_training_row(original_match_dataframes, player_identities_dataframe, player_stats_dataframe, team_dataframe):
    """
    SELECT participantId, AVG(goldSpent), AVG(damageDealtToChampions), AVG(deaths), AVG(baronKills), AVG(totalMinionsKilled)
    FROM identities i
        JOIN stats s
            ON i.gameId = s.gameId AND i.participantId = s.participantId
        JOIN teams t
            ON t.gameId = i.gameId AND s.teamId = t.teamId
    WHERE gameId in all_matches_to_average
    GROUP BY participantId


    Args:
        all_matches_to_average:
        match_id:
        player_identities_dataframe:
        player_stats_dataframe:
        team_dataframe:

    Returns:

    """
    player_account_team_table = pd.merge(original_match_dataframes[0], original_match_dataframes[1], on=["participantId", "gameId"])
    player_account_team_table = player_account_team_table.drop(["champLevel", "goldSpent", "deaths", "totalDamageDealtToChampions", "totalMinionsKilled", "participantId", "gameId"], axis=1)
    joined_tables = pd.merge(player_identities_dataframe, player_stats_dataframe, on=["gameId", "participantId"])
    joined_tables = pd.merge(joined_tables, team_dataframe, on=["gameId", "teamId"])
    joined_tables = joined_tables.drop("teamId", axis=1)
    joined_tables = pd.merge(joined_tables, player_account_team_table, on=["accountId"])
    joined_tables = joined_tables.groupby("teamId", as_index=True)
    joined_tables = joined_tables.agg({"goldSpent": np.mean, "totalDamageDealtToChampions": np.mean, "deaths": np.mean, "baronKills": np.mean, "totalMinionsKilled": np.mean})
    joined_tables = joined_tables[["goldSpent", "totalDamageDealtToChampions", "deaths", "baronKills", "totalMinionsKilled"]]
    first_team = joined_tables.loc[100]
    second_team = joined_tables.loc[200]
    result = first_team - second_team
    return result

def christopher(match_info):
    original_game = match_info[0][0]
    account_ids = get_account_id_from_match_info(original_game)
    dataframes = make_dataframe_from_all_match_info(match_info[1:], account_ids)
    original_game_dataframes = make_dataframe_from_match_info([original_game], None)
    results = get_training_row(original_game_dataframes, dataframes[0], dataframes[1], dataframes[2])
    return results.to_dict()


# def main():
#     me = api.summoner.by_name("na1", "pcschuckjr")
#     game_id = get_past_x_game_ids(me["accountId"], 0)[0]
#     game_info = [get_match_info_for_match_ids([game_id])]
#     account_ids = get_account_id_from_match_info(game_info[0][0])
#     for account in account_ids:
#         game_ids = get_past_x_game_ids(account, 0)
#         game_info.append(get_match_info_for_match_ids(game_ids))
#     result = christopher(game_info)
#     print(result)