url = 'https://www.oddsportal.com/rugby-union/world/world-cup/results/'
file = 'data/odds.mhtml'

import requests
from bs4 import BeautifulSoup
import pandas as pd
import geopy
import os
import re
from geopy.geocoders import Nominatim
import time
from collections import defaultdict
import numpy as np

years = [2023, 2019, 2015, 2011]
data = [0 for i in range(len(years))]
rows = [0 for i in range(len(years))]

def remove_tags(text):
    pattern = re.compile(r'<[^>]*>')
    return re.sub(pattern, '', text)
    
def import_odds():
    '''
    will import from files the odds, and create a df
    '''
    #2023
    parsed_data = []

    pattern = re.compile(r'\d{2}:\d{2}')

    for i in range(len(years)):  
        with open(f'data/odds{years[i]}.txt', 'r', encoding='utf-8-sig') as file:
            data[i] = file.read()
            rows[i] = re.split(pattern, data[i].strip())

# Iterate through each row to parse the data
    for i in range(len(rows[0])):
        parts = rows[0][i].split('\n')
        parts = [element.replace('from list', '').strip() for element in parts if element.strip() != '' and '–' not in element]
        if len(parts) != 0:
        # Extract relevant information
            team1 = parts[0]
            score1 = parts[1]
            odd1 = parts[6]
            team2 = parts[4]
            score2 = parts[3]
            odd2 = parts[7]
            odd3 = parts[8]
            # Append the parsed data to the list of dictionaries
            parsed_data.append({
                'Team A': team1,
                 'Team B': team2,
                'Score A': score1,
                'Score B': score2,
                'Odds A': odd1,
                'Odds D': odd2,
                'Odds B': odd3
            })
    columns_to_replace = ['Score A', 'Score B', 'Odds A', 'Odds D', 'Odds B']
    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(parsed_data)
    df[columns_to_replace]= df[columns_to_replace].astype(float)
    
    df['Proba_win'] = 1/df['Odds A']
    df['Proba_draw'] = 1/df['Odds D']
    df['Proba_loss'] = 1/df['Odds B']
    df['sum'] = df['Proba_win']+df['Proba_draw'] +df['Proba_loss'] 
    df['score_gap'] = np.abs(df['Score A']-df['Score B'])
    df['Year'] = 2023
    df_2023 = df

    parsed_data_2019 = []
    
    # Iterate through each row to parse the data
    for i in range(len(rows[1])):
        parts = rows[1][i].split('\n')
        parts = [remove_tags(part) for part in parts]
        parts = [element.replace('from list', '').strip() for element in parts if element.strip() != '' and '–' not in element ]
        if len(parts) != 0:
        # Extract relevant information
                team1 = parts[0]
                score1 = parts[2]
                odd1 = parts[6]
                team2 = parts[4]
                score2 = parts[3]
                odd2 = parts[7]
                odd3 = parts[8]
                # Append the parsed data to the list of dictionaries
                parsed_data_2019.append({
                    'Team A': team1,
                     'Team B': team2,
                    'Score A': score1,
                    'Score B': score2,
                    'Odds A': odd1,
                    'Odds D': odd2,
                    'Odds B': odd3
                })
    columns_to_replace = ['Score A', 'Score B', 'Odds A', 'Odds D', 'Odds B']
    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(parsed_data_2019)
    df = df[df['Team B']!='canc.']
    df[columns_to_replace]= df[columns_to_replace].astype(float)
    
    df['Proba_win'] = 1/df['Odds A']
    df['Proba_draw'] = 1/df['Odds D']
    df['Proba_loss'] = 1/df['Odds B']
    df['sum'] = df['Proba_win']+df['Proba_draw'] +df['Proba_loss'] 
    df['score_gap'] = np.abs(df['Score A']-df['Score B'])
    df['Year'] = 2019
    df_2019 = df

    parsed_data_2015 = []

    # Iterate through each row to parse the data-2011
    for i in range(len(rows[2])):
        parts = rows[2][i].split('\n')
        parts = [remove_tags(part) for part in parts]
        parts = [element.replace('from list', '').strip() for element in parts if element.strip() != '' and '–' not in element ]
        if len(parts) != 0:
        # Extract relevant information
                team1 = parts[0]
                score1 = parts[2]
                odd1 = parts[6]
                team2 = parts[4]
                score2 = parts[3]
                odd2 = parts[7]
                odd3 = parts[8]
                # Append the parsed data to the list of dictionaries
                parsed_data_2015.append({
                    'Team A': team1,
                     'Team B': team2,
                    'Score A': score1,
                    'Score B': score2,
                    'Odds A': odd1,
                    'Odds D': odd2,
                    'Odds B': odd3
                })
    columns_to_replace = ['Score A', 'Score B', 'Odds A', 'Odds D', 'Odds B']
    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(parsed_data_2015)
    df = df[df['Team B']!='canc.']
    df[columns_to_replace]= df[columns_to_replace].astype(float)
    
    df['Proba_win'] = 1/df['Odds A']
    df['Proba_draw'] = 1/df['Odds D']
    df['Proba_loss'] = 1/df['Odds B']
    df['sum'] = df['Proba_win']+df['Proba_draw'] +df['Proba_loss'] 
    df['score_gap'] = np.abs(df['Score A']-df['Score B'])
    df['Year'] = 2015
    df_2015 = df
    # Display the DataFrame
    parsed_data_2011 = []

    # Iterate through each row to parse the data
    for i in range(len(rows[3])):
        parts = rows[3][i].split('\n')
        parts = [remove_tags(part) for part in parts]
        parts = [element.replace('from list', '').strip() for element in parts if element.strip() != '' and '–' not in element ]
        if len(parts) != 0:
        # Extract relevant information
                team1 = parts[0]
                score1 = parts[2]
                odd1 = parts[6]
                team2 = parts[4]
                score2 = parts[3]
                odd2 = parts[7]
                odd3 = parts[8]
                # Append the parsed data to the list of dictionaries
                parsed_data_2011.append({
                    'Team A': team1,
                     'Team B': team2,
                    'Score A': score1,
                    'Score B': score2,
                    'Odds A': odd1,
                    'Odds D': odd2,
                    'Odds B': odd3
                })
    columns_to_replace = ['Score A', 'Score B', 'Odds A', 'Odds D', 'Odds B']
    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(parsed_data_2011)
    df[columns_to_replace]= df[columns_to_replace].astype(float)
    
    df['Proba_win'] = 1/df['Odds A']
    df['Proba_draw'] = 1/df['Odds D']
    df['Proba_loss'] = 1/df['Odds B']
    df['sum'] = df['Proba_win']+df['Proba_draw'] +df['Proba_loss'] 
    df['score_gap'] = np.abs(df['Score A']-df['Score B'])
    df['Year'] = 2011
    df_2011 = df
    # Display the DataFrame
    wc_odds_df = pd.concat([df_2023,df_2019, df_2015, df_2011])
    return wc_odds_df
