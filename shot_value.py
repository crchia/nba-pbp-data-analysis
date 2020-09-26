import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


def clean_data(data, team='ALL'):
    """
    Condenses full game data down to data that matters (shots and rebounds).
    Can filter data to only contain a single team.

    Returns the cleaned data.
    """
    filtered = data[['Shooter', 'ShotType', 'ShotOutcome', 'ShotDist', 'Rebounder', 'ReboundType', 'SecLeft']]
    filtered = filtered.dropna(thresh=2)
    if team != 'ALL':
        filtered = filtered[(filtered['Shooter'].str.contains(team, na=False)) 
                            | (filtered['Rebounder'].str.contains(team, na=False))]
    return filtered    


def calc_ppp(data):
    """
    Calculates league wide points per possession using the following
    formula for possessions:
    Total Possessions = FGAs + FT Trips (excluding and 1s) + TOs - OREBs
    Note: Calculation for OREBs excludes end of quarter team rebounds
    and team rebounds that occur on a missed FT (when the FT isn't the 
    final one of the trip).

    Returns the calculated ppp.
    """
    pts = 0
    pts_data = data[data['AwayPlay'] == 'End of Game']
    pts = pts_data['AwayScore'].sum()
    pts += pts_data['HomeScore'].sum()
    shot = ~pd.isnull(data['Shooter'])
    turnover = ~pd.isnull(data['TurnoverType'])
    ft_1_of_2 = data['FreeThrowNum'] == '1 of 2'
    ft_1_of_3 = data['FreeThrowNum'] == '1 of 3'
    ft_2_of_3 = data['FreeThrowNum'] == '2 of 3'
    ft_miss = data['FreeThrowOutcome'] == 'miss'
    poss_data = data[shot | turnover | ft_1_of_2 | ft_1_of_3]
    orb = data['ReboundType'] == 'offensive'
    not_end_of_q = data['SecLeft'] != 0
    orbs_data = data[orb & not_end_of_q]
    ft_orbs_data = data[ft_miss & (ft_1_of_2 | ft_1_of_3 | ft_2_of_3)]
    total_orbs = len(orbs_data) - len(ft_orbs_data)
    total_poss = len(poss_data) - total_orbs
    ppp = pts / total_poss
    return ppp


def calc_shots(data):
    """
    Calculates the number of attempts and makes for each
    different type of shot:
    layup: 2pt shot <= 5 ft
    long_two: 2pt shot > 5 ft
    short_three: 3pt shot <= 30 ft
    long_three: 3pt shot > 30 ft

    Returns a dict mapping each type of shot to a np array
    whose 0th index contains attempts and 1st index contains makes.
    """
    shot_data = []
    for year in data:
        shots = {}
        shots['2-pt 0-5 ft'] = np.zeros(2, dtype=int)
        shots['2-pt 6-10 ft'] = np.zeros(2, dtype=int)
        shots['2-pt 11-15 ft'] = np.zeros(2, dtype=int)
        shots['2-pt >15 ft'] = np.zeros(2, dtype=int)
        shots['3-pt <26 ft'] = np.zeros(2, dtype=int)
        shots['3-pt 26-30 ft'] = np.zeros(2, dtype=int)
        shots['3-pt >30 ft'] = np.zeros(2, dtype=int)
        two_pt_zero_to_five_ft = year[year['ShotDist'] <= 5]
        shots['2-pt 0-5 ft'][0] = len(two_pt_zero_to_five_ft)
        shots['2-pt 0-5 ft'][1] = len(two_pt_zero_to_five_ft[two_pt_zero_to_five_ft['ShotOutcome'] == 'make'])
        two_pt_six_to_ten_ft = year[(year['ShotDist'] > 5) & (year['ShotDist'] <= 10)]
        shots['2-pt 6-10 ft'][0] = len(two_pt_six_to_ten_ft)
        shots['2-pt 6-10 ft'][1] = len(two_pt_six_to_ten_ft[two_pt_six_to_ten_ft['ShotOutcome'] == 'make'])
        two_pt_eleven_to_fifteen_ft = year[(year['ShotDist'] > 10) & (year['ShotDist'] <= 15)]
        shots['2-pt 11-15 ft'][0] = len(two_pt_eleven_to_fifteen_ft)
        shots['2-pt 11-15 ft'][1] = len(two_pt_eleven_to_fifteen_ft[two_pt_eleven_to_fifteen_ft['ShotOutcome'] == 'make'])
        two_pt_fifteen_plus_ft = year[(year['ShotType'].str.contains('2-pt')) & (year['ShotDist'] > 15)]
        shots['2-pt >15 ft'][0] = len(two_pt_fifteen_plus_ft)
        shots['2-pt >15 ft'][1] = len(two_pt_fifteen_plus_ft[two_pt_fifteen_plus_ft['ShotOutcome'] == 'make'])
        three_pt_twenty_five_or_less_ft = year[(year['ShotType'].str.contains('3-pt')) & (year['ShotDist'] <= 25)]
        shots['3-pt <26 ft'][0] = len(three_pt_twenty_five_or_less_ft)
        shots['3-pt <26 ft'][1] = len(three_pt_twenty_five_or_less_ft[three_pt_twenty_five_or_less_ft['ShotOutcome'] == 'make'])
        three_pt_twenty_six_to_thirty_ft = year[(year['ShotDist'] > 25) & (year['ShotDist'] <= 30)]
        shots['3-pt 26-30 ft'][0] = len(three_pt_twenty_six_to_thirty_ft)
        shots['3-pt 26-30 ft'][1] = len(three_pt_twenty_six_to_thirty_ft[three_pt_twenty_six_to_thirty_ft['ShotOutcome'] == 'make'])
        three_pt_thirty_plus_ft = year[year['ShotDist'] > 30]
        shots['3-pt >30 ft'][0] = len(three_pt_thirty_plus_ft)
        shots['3-pt >30 ft'][1] = len(three_pt_thirty_plus_ft[three_pt_thirty_plus_ft['ShotOutcome'] == 'make'])
        shot_data.append(shots)
    return shot_data

    
def calc_rebs(data):
    """
    Calculates the number of offensive rebounds for each
    type of shot.

    Returns a dict mapping each type of shot to the number
    of offensive rebounds off of that type of shot
    """
    rebs_data = []
    for year in data:
        rebs = {}
        rebs['2-pt 0-5 ft'] = 0
        rebs['2-pt 6-10 ft'] = 0
        rebs['2-pt 11-15 ft'] = 0
        rebs['2-pt >15 ft'] = 0
        rebs['3-pt <26 ft'] = 0
        rebs['3-pt 26-30 ft'] = 0 
        rebs['3-pt >30 ft'] = 0
        for i in range(len(year)):
            if year.iloc[i, 2] == 'miss':
                shot_type = year.iloc[i, 1]
                shot_dist = year.iloc[i, 3]
                off_reb = 0
                if year.iloc[i + 1, 5] == 'offensive' and year.iloc[i + 1, 6] != 0:
                    off_reb = 1
                if shot_dist <= 5:
                    rebs['2-pt 0-5 ft'] += off_reb
                elif shot_dist <= 10:
                    rebs['2-pt 6-10 ft'] += off_reb
                elif shot_dist <= 15:
                    rebs['2-pt 11-15 ft'] += off_reb
                elif '2-pt' in shot_type:
                    rebs['2-pt >15 ft'] += off_reb
                elif shot_dist <= 25:
                    rebs['3-pt <26 ft'] += off_reb
                elif shot_dist <= 30:
                    rebs['3-pt 26-30 ft'] += off_reb
                else:
                    rebs['3-pt >30 ft'] += off_reb
        rebs_data.append(rebs)
    return rebs_data


def calc_exp_pts(shots):
    """
    Calculates the expected points generated for
    each type of shot taken without factoring in
    offensive rebounding.

    Returns the expected points.
    """
    exp_pts_data = []
    for year in shots:
        exp_pts = {}
        attempts = np.array([year[shot_type][0] for shot_type in year])
        makes = np.array([year[shot_type][1] for shot_type in year])
        make_percentages = makes / attempts
        exp_pts['2-pt 0-5 ft'] = 2 * make_percentages[0]
        exp_pts['2-pt 6-10 ft'] = 2 * make_percentages[1]
        exp_pts['2-pt 11-15 ft'] = 2 * make_percentages[2]
        exp_pts['2-pt >15 ft'] = 2 * make_percentages[3]
        exp_pts['3-pt <26 ft'] = 3 * make_percentages[4]
        exp_pts['3-pt 26-30 ft'] = 3 * make_percentages[5]
        exp_pts['3-pt >30 ft'] = 3 * make_percentages[6]
        exp_pts_data.append(exp_pts)
    return exp_pts_data


def calc_exp_pts_w_orb(shots, rebs, ppp):
    """
    Calculates the expected points generated for
    each type of shot taken, factoring in offensive
    rebounding that occurs on missed shots.

    Returns the expected points.
    """
    exp_pts_data = []
    for i in range(len(shots)):
        exp_pts = {}
        attempts = np.array([shots[i][shot_type][0] for shot_type in shots[i]])
        makes = np.array([shots[i][shot_type][1] for shot_type in shots[i]])
        make_percentages = makes / attempts
        orbs = np.array([rebs[i][shot_type] for shot_type in rebs[i]])
        orb_percentages = orbs / attempts
        exp_pts['2-pt 0-5 ft'] = 2 * make_percentages[0] + orb_percentages[0] * ppp[i]
        exp_pts['2-pt 6-10 ft'] = 2 * make_percentages[1] + orb_percentages[1] * ppp[i]
        exp_pts['2-pt 11-15 ft'] = 2 * make_percentages[2] + orb_percentages[2] * ppp[i]
        exp_pts['2-pt >15 ft'] = 2 * make_percentages[3] + orb_percentages[3] * ppp[i]
        exp_pts['3-pt <26 ft'] = 3 * make_percentages[4] + orb_percentages[4] * ppp[i]
        exp_pts['3-pt 26-30 ft'] = 3 * make_percentages[5] + orb_percentages[5] * ppp[i]
        exp_pts['3-pt >30 ft'] = 3 * make_percentages[6] + orb_percentages[6] * ppp[i]
        exp_pts_data.append(exp_pts)
    return exp_pts_data


def gen_bar_shots_attempts(shots):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    for i in range(len(ax)):
        for j in range(len(ax[0])):
            shot_types = shots[i * 2 + j].keys()
            attempts = np.array([shots[i * 2 + j][shot_type][0] for shot_type in shots[i * 2 + j]])
            ax[i][j].bar(shot_types, attempts, color=['b', 'b', 'b', 'b', 'm', 'm', 'm'])
            ax[i][j].set_xticklabels(shot_types, rotation=30)
            ax[i][j].set_xlabel('Shot Type')
            ax[i][j].set_ylabel('# Shots Attempted')
            ax[i][j].set_title(str(2019 - i * 2 - j) + '-' + str(2019 - i * 2 - j + 1) + \
                            ' League Wide \n Shot Type vs. # Shots Attempted')
            plt.tight_layout()
    fig.savefig('shot_attempts.png')

    
def gen_bar_shot_percentages(shots):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    for i in range(len(ax)):
        for j in range(len(ax[0])):
            shot_types = shots[i * 2 + j].keys()
            attempts = np.array([shots[i * 2 + j][shot_type][0] for shot_type in shots[i * 2 + j]])
            makes = np.array([shots[i * 2 + j][shot_type][1] for shot_type in shots[i * 2 + j]])
            make_percentages = makes / attempts * 100
            ax[i][j].bar(shot_types, make_percentages, color=['b', 'b', 'b', 'b', 'm', 'm', 'm'])
            ax[i][j].set_xticklabels(shot_types, rotation=30)
            ax[i][j].set_xlabel('Shot Type')
            ax[i][j].set_ylabel('Field Goal %')
            ax[i][j].set_title(str(2019 - i * 2 - j) + '-' + str(2019 - i * 2 - j + 1) + \
                            ' League Wide \n Shot Type vs. Field Goal %')
            plt.tight_layout()
    fig.savefig('shot_percentages.png')



def gen_bar_orb_percentages(shots, rebs):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    for i in range(len(ax)):
        for j in range(len(ax[0])):
            shot_types = shots[i * 2 + j].keys()
            attempts = np.array([shots[i * 2 + j][shot_type][0] for shot_type in shots[i * 2 + j]])
            orbs = np.array([rebs[i * 2 + j][shot_type] for shot_type in rebs[i * 2 + j]])
            orb_percentages = orbs / attempts * 100
            ax[i][j].bar(shot_types, orb_percentages, color=['b', 'b', 'b', 'b', 'm', 'm', 'm'])
            ax[i][j].set_xticklabels(shot_types, rotation=30)
            ax[i][j].set_xlabel('Shot Type')
            ax[i][j].set_ylabel('Offensive Rebounding %')
            ax[i][j].set_title(str(2019 - i * 2 - j) + '-' + str(2019 - i * 2 - j + 1) + \
                            ' League Wide \n Shot Type vs. Offensive Rebounding %')
            plt.tight_layout()
    fig.savefig('orb_percentage_of_attempts.png')


def gen_bar_orbp_off_miss(shots, rebs):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    for i in range(len(ax)):
        for j in range(len(ax[0])):
            shot_types = shots[i * 2 + j].keys()
            attempts = np.array([shots[i * 2 + j][shot_type][0] for shot_type in shots[i * 2 + j]])
            makes = np.array([shots[i * 2 + j][shot_type][1] for shot_type in shots[i * 2 + j]])
            misses = attempts - makes
            orbs = np.array([rebs[i * 2 + j][shot_type] for shot_type in rebs[i * 2 + j]])
            orb_percentages = orbs / misses * 100
            ax[i][j].bar(shot_types, orb_percentages, color=['b', 'b', 'b', 'b', 'm', 'm', 'm'])
            ax[i][j].set_xticklabels(shot_types, rotation=30)
            ax[i][j].set_xlabel('Shot Type')
            ax[i][j].set_ylabel('Offensive Rebounding % Off Miss')
            ax[i][j].set_title(str(2019 - i * 2 - j) + '-' + str(2019 - i * 2 - j + 1) + \
                            ' League Wide \n Shot Type vs. Offensive Rebounding % Off Miss')
            plt.tight_layout()
    fig.savefig('orb_percentage_of_misses.png')


def gen_bar_exp_pts(exp_pts):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    for i in range(len(ax)):
        for j in range(len(ax[0])):
            shot_types = exp_pts[i * 2 + j].keys()
            exp_pts_data = exp_pts[i * 2 + j].values()
            ax[i][j].bar(shot_types, exp_pts_data, color=['b', 'b', 'b', 'b', 'm', 'm', 'm'])
            ax[i][j].set_xticklabels(shot_types, rotation=30)
            ax[i][j].set_xlabel('Shot Type')
            ax[i][j].set_ylabel('Expected Points Generated')
            ax[i][j].set_title(str(2019 - i * 2 - j) + '-' + str(2019 - i * 2 - j + 1) + \
                            ' League Wide \n Shot Type vs. Expected Points Generated')
            plt.tight_layout()
    fig.savefig('exp_pts.png')


def gen_bar_exp_pts_w_orb(exp_pts_w_orb):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    for i in range(len(ax)):
        for j in range(len(ax[0])):
            shot_types = exp_pts_w_orb[i * 2 + j].keys()
            exp_pts_data = exp_pts_w_orb[i * 2 + j].values()
            ax[i][j].bar(shot_types, exp_pts_data, color=['b', 'b', 'b', 'b', 'm', 'm', 'm'])
            ax[i][j].set_xticklabels(shot_types, rotation=30)
            ax[i][j].set_xlabel('Shot Type')
            ax[i][j].set_ylabel('Expected Points Generated')
            ax[i][j].set_title(str(2019 - i * 2 - j) + '-' + str(2019 - i * 2 - j + 1) + \
                            ' League Wide \n Shot Type vs. Expected Points Generated')
            plt.tight_layout()
    fig.savefig('exp_pts_w_orb.png')


def main():
    data_19_20 = pd.read_csv('NBA-PBP-2019-2020.csv')
    data_18_19 = pd.read_csv('NBA-PBP-2018-2019.csv')
    data_17_18 = pd.read_csv('NBA-PBP-2017-2018.csv')
    data_16_17 = pd.read_csv('NBA-PBP-2016-2017.csv')
    data = [data_19_20, data_18_19, data_17_18, data_16_17]
    # small_data_1 = pd.read_csv('small.txt')
    # small_data_2 = pd.read_csv('small2.txt')
    # small_data_3 = pd.read_csv('small3.txt')
    # small_data_4 = pd.read_csv('small4.txt')
    # data = [small_data_1, small_data_2, small_data_3, small_data_4]
    ppp_data = [calc_ppp(year) for year in data]
    print(ppp_data)
    # cleaned_data = [clean_data(year) for year in data]
    # shot_data = calc_shots(cleaned_data)
    # rebs_data = calc_rebs(cleaned_data)
    # exp_pts_data = calc_exp_pts(shot_data)
    # exp_pts_w_orb_data = calc_exp_pts_w_orb(shot_data, rebs_data, ppp_data)
    # gen_bar_shots_attempts(shot_data)
    # gen_bar_shot_percentages(shot_data)
    # gen_bar_orb_percentages(shot_data, rebs_data)
    # gen_bar_orbp_off_miss(shot_data, rebs_data)
    # gen_bar_exp_pts(exp_pts_data)
    # gen_bar_exp_pts_w_orb(exp_pts_w_orb_data)


if __name__ == '__main__':
    main()