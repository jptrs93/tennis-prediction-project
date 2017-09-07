"""

Pre-processes original data to add additional columns with information required by models.

"""

import csv
import os
from tennismodelling import output as out
from tennismodelling import markovchain as mcm


def get_game_percentages(row):
    """Returns the percentage of games won by the winner and loser respectively"""
    try:
        winner_games = 0
        loser_games = 0
        score = row[27]
        sets = score.split(' ')
        for set_score in sets:
            if any(c.isalpha() for c in set_score):
                pass
            else:
                winner_games += float(set_score[0])
                loser_games += float(set_score[2])
        if winner_games == 0:
            return -1, -1
        else:
            prob = min(0.999,(winner_games / (winner_games + loser_games)))
            return prob, 1- prob
    except (ValueError, ZeroDivisionError, IndexError):
        return -1, -1


def get_service_game_percentages(row):
    """Returns the percentage of service games won by the winner and loser respectively"""
    try:
        w_game_prob = 1 - (float(row[39]) - float(row[38])) / float(row[37])
        l_game_prob = 1 - (float(row[48]) - float(row[47])) / float(row[46])
        w_game_prob = max(0.001, min(w_game_prob, 0.999))
        l_game_prob = max(0.001, min(l_game_prob, 0.999))
        return w_game_prob, l_game_prob
    except (ValueError, ZeroDivisionError):
        return -1, -1


def get_point_percentages(row):
    """Returns the percentage of points won by the winner and loser respectively"""
    try:
        w_points_on_serve = (int(row[35]) + int(row[36]))                   # Points won by winner on serve
        w_points_off_serve = int(row[42]) - (int(row[44]) + int(row[45]))   # Points won by winner returning
        w_points = w_points_on_serve + w_points_off_serve                   # Total points won by winner
        t_points = int(row[42]) + int(row[33])                              # Total point played
        point_prob = min(0.999,w_points / float(t_points))                  # % points won by winner
        return point_prob, 1- point_prob
    except (ValueError, ZeroDivisionError):
        return -1, -1


def get_service_point_percentages(row):
    """Returns the percentage of service points won by the winner and loser respectively"""
    try:
        wsp = (int(row[35]) + int(row[36])) / float(row[33])
        lsp = (int(row[44]) + int(row[45])) / float(row[42])
        wsp = max(0.001, min(wsp, 0.999))
        lsp = max(0.001, min(lsp, 0.999))
        return wsp, lsp
    except (ValueError, ZeroDivisionError):
        return -1, -1


def get_combined_percentages(row):
    """Returns a combined percentage of games won by both players based on inferring from point one and averaging
    with the actual games won"""
    w_point_percentage, l_point_percentage = get_point_percentages(row)
    w_game_percentage, l_game_percentage = get_game_percentages(row)
    if w_point_percentage == -1 and w_game_percentage != -1:
        return w_game_percentage, l_game_percentage
    elif w_game_percentage == -1:
        return -1, -1
    else:
        w_game_inferred = mcm.game_chance(w_point_percentage)
        l_game_inferred = mcm.game_chance(l_point_percentage)
        w_combined_percentage = 0.5*(w_game_inferred + w_game_percentage)
        l_combined_percentage = 0.5*(l_game_inferred + l_game_percentage)
        return w_combined_percentage, l_combined_percentage


def get_combined_service_percentages(row):
    """Returns a combined percentage of service games won by both players based on inferring from point one and
    averaging with the actual games won"""
    w_point_percentage, l_point_percentage = get_service_point_percentages(row)
    w_game_percentage, l_game_percentage = get_service_game_percentages(row)
    if w_point_percentage == -1 and w_game_percentage != -1:
        return w_game_percentage, l_game_percentage
    elif w_game_percentage == -1:
        return -1, -1
    else:
        w_game_inferred = mcm.game_chance(w_point_percentage)
        l_game_inferred = mcm.game_chance(l_point_percentage)
        w_combined_serve_percentage = 0.5*(w_game_inferred + w_game_percentage)
        l_combined_serve_percentage = 0.5*(l_game_inferred + l_game_percentage)
        return w_combined_serve_percentage, l_combined_serve_percentage


file1 = 'atp_and_challenger_dataPoint.csv'
file2 = 'atp_and_challenger_data.csv'
input_file = os.path.join(os.environ['ML_DATA_DIR'], file2)
rows = [row for row in csv.reader(open(input_file))]


# Add extra headers
rows[0] = rows[0][:49] + ['w_games_perc.','l_games_perc.','w_service_games_perc.','l_service_games_perc.','w_points_perc.','l_points_perc.',
            'w_service_points_perc.','l_service_points_perc.','w_combined_perc.','l_combined_perc.','w_combined_service_perc.','l_combined_service_perc.']

# Add contents of extra columns
for row in rows[1:]:
    w_games_perc, l_games_perc = get_game_percentages(row)
    w_service_games_perc, l_service_games_perc = get_service_game_percentages(row)
    w_points_perc, l_points_perc = get_point_percentages(row)
    w_service_points_perc, l_service_points_perc = get_service_point_percentages(row)
    w_combined_percentage, l_combined_percentage = get_combined_percentages(row)
    w_combined_serve_percentage, l_combined_serve_percentage = get_combined_service_percentages(row)
    row = row[:49] + [w_games_perc,l_games_perc,w_service_games_perc,l_service_games_perc,w_points_perc,l_points_perc,w_service_points_perc,l_service_points_perc,
            w_combined_percentage, l_combined_percentage,w_combined_serve_percentage, l_combined_serve_percentage]

# re-write file with extra information
os.remove(input_file)
out.append_csv_rows(rows, input_file)
