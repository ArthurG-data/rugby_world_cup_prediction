import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from sklearn import decomposition
from sklearn import datasets
import matplotlib.pyplot as plt

def create_rb_wc_df(save=False, percentage=True):
    '''
    Create a df from the rugbycology dataset,the team in split in team A and B
    if percentage is true, will comput % of successful rucks, etc.
    '''
    rub_wc_df = pd.read_csv(current_dir+'/data/'+rubicology_file )
    variables_start = list(rub_wc_df.columns)
    variable_end = []
    for variable in variables_start:
         if re.search(r'\sA(?!\S)', variable):
            new_variable = re.sub(r'\sA(?!\S)', ' ', variable)
            variable_end.append(new_variable.strip())
    variable_a = ['Date', 'Game ID']
    variable_b =  ['Date', 'Game ID']
    
    for variable in variable_end:
        if 'Possession' not in variable:
            variable_a.append(f'{variable} A')
        else:
            one, two = variable.split('  - ')
            one = f'{one} A'
            variable = ' - '.join([one, two])
            variable_a.append(variable)
    
    for variable in variable_end:
        if 'Possession' not in variable:
            variable_b.append(f'{variable} B')
        else:
            one, two = variable.split('  - ')
            one = f'{one} B'
            variable = ' - '.join([one, two])
            variable_b.append(variable)
            
    
    rub_wc_df_a = rub_wc_df[variable_a ]
    rub_wc_df_b = rub_wc_df[variable_b ]
    
    rub_wc_df_a.columns = rub_wc_df_a.columns.str.replace(r'\sA(?!\S)', '', regex=True)
    rub_wc_df_b.columns = rub_wc_df_b.columns.str.replace(r'\sB(?!\S)', '', regex=True)
    split_df = pd.concat([rub_wc_df_a,rub_wc_df_b ])
    if percentage:
        split_df['% Rucks Successful'] = split_df['Rucks Successful']/split_df['Total Rucks']
        split_df['% Goal kicks Successful'] = split_df['Goal Kicks Successful']/split_df['Goal Kicks Attempted']
        split_df['% Scrums Successful'] = split_df['Scrums Successful']/split_df['Scrums Total']
        split_df['% Tackle Successful'] = 1-(split_df['Tackles Missed'])/(split_df['Tackles Made']+split_df['Tackles Missed'])
    if save:
        split_df.to_csv(current_dir+folder+'split_df_teams.csv', index=False)
    return split_df


   

def create_relative_ratio_df(df):
    '''
    input: the split_df, with teams as row
    output: df where each frequence was normalize by the total frequence dor each variable
    '''
    ids = df['Game ID'].unique()
    excluded_columns = ['Game ID', 'Date', 'Team', '% Rucks Successful', '% Goal kicks Successful', '% Scrums Successful', '% Tackle Successful']
   
    for unique_id in ids:
        rows = df[df['Game ID'] == unique_id].copy()  # Use .copy() to avoid SettingWithCopyWarning
        # Sum for each variable across the rows
        sum_row = rows.drop(columns=excluded_columns).sum(axis=0)
        # Divide each row that has the Game ID = id by the sum_row
        rows.loc[:, ~rows.columns.isin(excluded_columns)] = rows.drop(columns=excluded_columns).div(sum_row, axis=1)
        # Update the original DataFrame with the scaled values
        df.loc[df['Game ID'] == unique_id, ~df.columns.isin(excluded_columns)] = rows.drop(columns=excluded_columns).values
    
    df.sort_values(by='Game ID', inplace=True)
    df.fillna(0, inplace=True)
    df['Outcome'] = df['Score'].apply(lambda x: 1 if x > 0.5 else (0.5 if x == 0.5 else 0))
    
    return df

def create_relative_diff_df(df):
    '''
    input: the split_df, with teams as row
    output: df where each frequence is the difference of the home - away
    '''
    ids = df['Game ID'].unique()
    series = []
    for unique_id in ids:
        rows = df[df['Game ID'] == unique_id]
        row_1 = rows.iloc[0].drop(['Game ID', 'Date', 'Team'])
        row_2 = rows.iloc[1].drop(['Game ID', 'Date', 'Team'])
        new_1 = row_1 - row_2
        new_2 = row_2 - row_1
        row_1_header= rows.iloc[0][['Game ID', 'Date', 'Team']]
        row_2_header= rows.iloc[1][['Game ID', 'Date', 'Team']]
        new_1 = pd.concat([row_1_header,new_1 ])
        new_2 = pd.concat([row_2_header,new_2 ])
        
        series.append(new_1)
        series.append(new_2)
        # update the original DataFrame with the scaled values
    df = pd.concat(series, axis=1)
    df.fillna(0,inplace=True)
    #df['Outcome'] = df['Score'].apply(lambda x: 1 if x > 0.5 else (0.5 if x == 0.5 else 0))
    return df


#for each team, get in what percentil amongst the teams it lies. create a dict for each team
def get_rank(relative_df):
    '''
    for each team , compute the average performance, then rank the tams for each variables
    '''
    relative_df_ratio_average = relative_df.drop(['Game ID', 'Date'], axis=1).groupby('Team').median().reset_index()
    dico = {}
    #for each team
    for col in relative_dfdrop('Team',axis=1).columns:
        sorted = relative_df[[col, 'Team']].sort_values(ascending=False, by=col)
        dico[col] =list(sorted['Team'].values )
          #for ind in grouped_median_dict.index:
        #country = grouped_median_dict['Team'][ind]  
    #for each variable
    return dico

def create_subplots(relative_df_ratio_average):
    '''
    input: the relative ratio df with the mean for each team
    output: radar plot for each team
    '''
    removed_columns = ['Red Cards', 'Yellow Cards', 'Outcome', 'Score']
    filtered_df = relative_df_ratio_average.drop(removed_columns, axis=1)
    
    rows = 4
    cols = 5
    horizontal_spacing=0.1
    vertical_spacing= 0.01
    teams = filtered_df ['Team'].values
    fig = make_subplots(rows=rows, cols=cols, specs=[[{'type': 'polar'}]*cols]*rows, subplot_titles=teams,horizontal_spacing=horizontal_spacing,  
        vertical_spacing=vertical_spacing
                       )
    
    
    
    for i in range(rows):
        for j in range(cols):
            index = i * cols + j
            if index < len(teams):
                # Select team
                df =filtered_df[filtered_df['Team'] == teams[index]]
                r = df.drop(columns=['Team']).values.flatten()
                theta = df.drop(columns=['Team']).columns
    
                fig.add_trace(
                    go.Barpolar(r=r, theta=theta, name=teams[index]),
                    row=i+1, col=j+1,
                )
    
                fig.update_layout(
        title='Polar Bar Chart Subplots',
        height=2200,  # Increase height for larger subplots
        width=3000,   # Increase width for larger subplots
        showlegend=False
    )
    
    # Update polar layout for each subplot
    for i in range(1, rows+1):
        for j in range(1, cols+1):
            fig.update_polars(
                row=i, col=j,
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickfont=dict(
                        family="Impact",  # Change the font family
                        size=10,  # Change the font size
                        color="Black"  # Change the font color
                    )
                ),
                angularaxis=dict(
                    tickfont=dict(
                        size=10,
                        # Reduce font size for variables
                    ),
                   tickangle=0,
                )
            )
    
    # Show figure
    fig.show()
    return fig

def create_match_summary(relative_df_ratio, game_id):
    '''
    create a plot with two radar plots, the score, the dates?the elo, proba if winning and predicted score
    '''
    selected = relative_df_ratio[relative_df_ratio['Game ID']==game_id]
    teams = selected['Team'].values
    date= selected['Date'].values
    Score = selected['Score'].values
    rows = 1
    cols = 3
    locations = [0,2]
    fig = make_subplots(rows=rows, cols=cols, specs=[[{'type': 'polar'}]*cols]*rows ,horizontal_spacing=horizontal_spacing,  
        vertical_spacing=vertical_spacing
                       )
    annotations = []
    for i in range(2):
        annotations.append(dict(
            x=0.5, y=0.5 - i * 0.1,
            xref='paper', yref='paper',
            text=f"Team: {teams[i]}<br>Date: {date[0]}<br>Score: {Score[i]}",
            showarrow=False,
            font=dict(
                size=12,
                color="Black"
            ),
            align="center"
        ))
    
    
    for i in range(2):
            index = i 
           
            # Select team
            df =selected [selected ['Team'] == teams[index]]
            df = df.drop(['Team', 'Game ID', 'Date','Outcome'], axis=1)
            r = df.values.flatten()
            theta = df.columns
    
            fig.add_trace(
                go.Barpolar(r=r, theta=theta, name=teams[index]),
                row=1, col=locations[i]+1,
            )
    
            fig.update_layout(
                title='Polar Bar Chart Subplots',
                height=800,  # Increase height for larger subplots
                width=800,   # Increase width for larger subplots
                showlegend=False,
                annotations=annotations
            )
    
    
    for i in range(2):
            fig.update_polars(
                row=1, col=locations[i]+1,
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickfont=dict(
                        family="Impact",  # Change the font family
                        size=8,  # Change the font size
                        color="Black"  # Change the font color
                    )
                ),
                angularaxis=dict(
                    tickfont=dict(
                        size=5,
                        # Reduce font size for variables
                    ),
                   tickangle=0,
                )
            )
    fig.show()


def load_data(split_filename, value='diff', draw=True):
    '''
    First step is to remove draws from the dataset , then
    '''
    non_numeric = ['Game ID', 'Date', 'Team']
    split_df = pd.read_csv(split_filename)
    numeric = [column for column in list(split_df.columns) if column not in non_numeric]
    #remove one the second entry
    
    if value == 'diff':
        df = create_relative_diff_df(split_df)
        df = df.transpose().reset_index(drop=True)
        df['Outcome'] = df.apply(lambda row: 0 if row['Score'] < 0 else (0.5 if row['Score'] == 0 else 1), axis=1)
    elif value == 'relative':
        df = create_relative_ratio_df(split_df)
        df.reset_index(inplace=True, drop=True)
        df['Outcome'] = df.apply(lambda row: 0 if row['Score'] < 0.5 else (0.5 if row['Score'] == 0.5 else 1), axis=1)
    else:
        df = split_df
    if draw:
        try:
            df= df[df['Outcome']!=0.5]
        except Exception as e:
            print('error')
            
    df[numeric] = df[numeric] .astype(float)      
    return df 

'''
look at correlation between variables and score (here represent the gap)
'''
def def_finc_correlation(diff_df,standardize=True, cutoff=0, verbose=True):
    variables_to_remove = ['Game ID', 'Date', 'Team','Outcome']
    to_add = np.asarray(diff_df[variables_to_remove])
    df =  diff_df.drop(columns=variables_to_remove)
    if standardize:
        scaler = StandardScaler()
        columns = list(df.columns)
        X= np.array(df)
        X = scaler.fit_transform(X)
    df = pd.DataFrame(data=X, columns=(columns))
    corr = df.corr()
    if verbose:
        print(corr['Score'])
    filtered_corr = corr[np.abs(corr)>=cutoff]['Score']
    kept_variables  = list(filtered_corr[filtered_corr.notna()].index)
    filterd_df = df[kept_variables]
    filterd_df[variables_to_remove] = to_add
    if verbose:
        print(kept_variables)
    return filterd_df
    
def filter_multicollinearity(diff_df, tolerance = 1, vif=10, verbose=True) :   
    '''
    pca can be an other method to deal with multicol
    '''
    variables_to_remove = ['Game ID', 'Date', 'Team','Outcome', 'Score']
    to_add = diff_df[variables_to_remove ]
    target_df = to_add['Score']
    features_df = diff_df.drop(columns=variables_to_remove)
    X = features_df
    Y = target_df
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 5)
    
    model = sm.OLS(Y_train, sm.add_constant(X_train)).fit()
    Y_pred = model.predict(sm.add_constant(X_test))
    print_model = model.summary()
    if verbose:
        print(print_model)
        
    x_temp = sm.add_constant(X_train)
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(x_temp.values, i) for i in range(x_temp.values.shape[1])]
    vif["features"] = x_temp.columns
    vif_df = vif.round(1)
    if verbose:
        print(vif.round(1))
    selected_variables = list(vif_df[vif_df['VIF Factor']<=vif]['features'].values)
    selected_variables = selected_variables[1:]
    if verbose:
        print(selected_variables)
    if verbose:
        selected_variables
    df = diff_df[selected_variables]
    df[variables_to_remove] = to_add
    df['Score'] = Y 
    return df

def create_random_split(data, win = 24):
    '''
    input: the df where each row is a team perfromance
    output: a df, without 2 perfromance from the same game, of win wins and 47-wins losses
    '''
    win_df = data[data['Score'] >0 ].reset_index(drop=True)
    loss_df = data[data['Score'] <0 ].reset_index(drop=True)
    ids = list(win_df['Game ID'].values)
    win = win
    loss = 47-win
    random_selected = []
    features = list(win_df.columns)
    while ids:
        selected = random.randint(0,len(ids)-1)
        id_ = ids.pop(selected)
        rand = random.randint(0,1)
        if  (rand == 0 and win >0):
            row = list(win_df[win_df['Game ID']== id_ ].values.flatten())
            
            random_selected.append(row)
            win -=1
        elif rand == 1 and loss >0:
            row = list(loss_df[loss_df['Game ID']== id_ ].values.flatten())
            random_selected.append(row)
            loss -=1
        elif rand == 0 and win  == 0:
            row = list(loss_df[loss_df['Game ID']== id_ ].values.flatten())
            random_selected.append(row)
            loss -=1
        else:
            row = list(loss_df[loss_df['Game ID']== id_ ].values.flatten())
            random_selected.append(row)
            loss -=1
    
    data = pd.DataFrame(data=random_selected, columns=features)
    return data
