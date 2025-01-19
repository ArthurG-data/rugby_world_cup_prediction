from sklearn import discriminant_analysis
import sklearn
import math
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 

def analyse_result(model,x,y, n_comp='all',reg='Logistic',test_split=0.3):
    '''
    create a confusion matrix with the classification results for logit or reg
    not output, print a figure with results from testing and training set
    '''
    fig, ax = plt.subplots(1, 2)
    fig.suptitle(f'Classification results for {reg} regression and {n_comp} components')
    #on the test
    y =  [1 if y >0.5 else 0 for y in y]
    X_train_log, X_test_log, Y_train_log, Y_test_log = train_test_split(x , y, test_size = test_split, random_state = 6)
    Y_pred_logit = model.predict(X_test_log)
    Y_pred_logit = [1 if y >0.5 else 0 for y in Y_pred_logit]
    cm = sklearn.metrics.confusion_matrix(Y_test_log, Y_pred_logit)
    displ = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
    displ.plot(ax=ax[0], colorbar=False)
    ax[0].set_title("Test")
    #train
    Y_pred_logit = model.predict(X_train_log)
    Y_pred_logit = [1 if y >0.5 else 0 for y in Y_pred_logit]
    cm = sklearn.metrics.confusion_matrix(Y_train_log, Y_pred_logit)
    displ = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
    displ.plot(ax=ax[1], colorbar=False)
    ax[1].set_title("Train")


def create_summary_log(result, target=0.05, postive_odd=True ):
    '''
    input the res dataframe and output the dataframe with odds
    '''
    pvalues = pd.DataFrame(result.pvalues, columns=['pvalues'])# Reviewing the results and creating a new column called Features
    pvalues.index.name = 'Features'
    pvalues.reset_index()# Creating new data frame using the coefficients of each summary

    params = pd.DataFrame(result.params, columns=['params'])

    params.index.name = 'Features'
    params.reset_index()

    params.index.name = 'Features'
    params.reset_index()
    
    results = pd.merge(params, pvalues, how = "left", on = "Features",suffixes=("params","pvalues")).fillna(0).reset_index()
    final = results.loc[(results['pvalues'] < target)].reset_index(drop=True)
    final['Odds'] = np.exp(final['params'])# Creating a new column called percent using the log odds of the feature
    final['Percent'] = (final['Odds'] - 1)*100

    final[['0.025', '0.975']] = result.conf_int().values
    final['0.025'] =np.exp(final['0.025'])
    final['0.975'] = np.exp(final['0.975'])
    return final
    
def create_forest_plot(result_df_glm):
    '''
    input: a df created by create summary log
    output: a forest plot with the features as y and odd rations with confidence interval as x
    '''
    plt.figure(figsize=(6, 4), dpi=150)
    
    ci = [result_df_glm.iloc[::-1]['Odds'] - result_df_glm.iloc[::-1]['0.025'].values, result_df_glm.iloc[::-1]['0.975'].values - result_df_glm.iloc[::-1]['Odds']]
    plt.errorbar(x=result_df_glm.iloc[::-1]['Odds'], y=result_df_glm.iloc[::-1].index.values, xerr=ci,
                color='black',  capsize=3, linestyle='None', linewidth=1,
                marker="o", markersize=5, mfc="black", mec="black")
    plt.axvline(x=1, linewidth=0.8, linestyle='--', color='black')
    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.xlabel('Odds Ratio and 95% Confidence Interval', fontsize=8)
    plt.ylabel('Components', fontsize=8)
    labels = [f'PC{i}' for i in range(result_df_glm.shape[0]+1)]
    
    plt.gca().set_yticklabels(labels)
    plt.tight_layout()
    plt.savefig('figures/raw_forest_plot.png')
    plt.show()
    
def cumulative_variance_explained(pca, save=True):
    variance_explained = (np.sum(pca.explained_variance_ratio_)).round(2)

    plt.rcParams["figure.figsize"] = (8,5)
    
    fig, ax = plt.subplots()
    xi = np.arange(1, len(pca.explained_variance_ratio_)+1, step=1)
    y = np.cumsum(pca.explained_variance_ratio_)
    
    plt.ylim(0.0,1.1)
    plt.plot(xi, y, marker='o', linestyle='-', color='black')
    
    plt.xlabel('Number of Components')
    plt.xticks(np.arange(1, len(pca.explained_variance_ratio_), step=1)) 
    plt.ylabel('Cumulative variance (%)')
    plt.title('Variance explained by the selectect number of components')
    
    plt.axhline(y=variance_explained, color='grey', linestyle='--')
    plt.text(1.1, 1, f'{variance_explained*100}% variance explained', color = 'black', fontsize=16)
    
    ax.grid(axis='x')
    plt.tight_layout()
    if save:
        plt.savefig('figures/pcavisualize_1.png', dpi=300)
  
def single_component_variance(pca, save=True): 
    fig, ax = plt.subplots()
    xi =pca.explained_variance_ratio_  
    y = [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))]
    colors = plt.cm.tab10(range(len(y)))
    #plt.ylim(0.0,1.1)
    plt.barh(y, xi, color=colors)
    
    plt.xlabel('Explained variance(%)')
    #plt.xticks(np.arange(1, len(pca.explained_variance_ratio_), step=1)) 
    plt.ylabel('Components')
    plt.title('Variance explained by the selectect number of components')
    
    #plt.axhline(y=variance_explained, color='grey', linestyle='--')
    #plt.text(1.1, 1, f'{variance_explained*100}% variance explained', color = 'black', fontsize=16)
    sns.despine()
    plt.tight_layout()
    plt.savefig('figures/single_component_variance.png')


def top_N(cumulative_sum, percentage):
    '''
    take the cimulative sum of he variance explained by each component, a target percentage, and return a number of compoenent
    '''
    topN = np.where(cumulative_sum > percentage)[0][0]
    print(f'{percentage*100}% in ' + '%d' % (topN+1) + ' components')
    return topN
def make_pca_component_plot(loading_matrix,components='all', cutoff=0,save=True, filename='figure_feature_loading.png'):

    if components=='all':
        f, ax = plt.subplots(figsize=(4, 6))
        loading_matrix = loading_matrix[(loading_matrix['highest_pc']>=cutoff) | (loading_matrix['highest_pc']<=-cutoff)]
        sns.barplot(x="highest_pc", y="features", data=loading_matrix,
                    hue="highest_pc_column",palette="tab10", color="b")
    else:
        f, ax = plt.subplots(figsize=(2, 3))
        loading_matrix = loading_matrix[(loading_matrix[components]>=cutoff) | (loading_matrix[components]<=-cutoff)]
        sns.barplot(x=components, y="features", data=loading_matrix)
    ax.legend(ncol=1, loc="upper left", frameon=False, bbox_to_anchor=(0.95, 1))
    ax.set_title(f"{components}", fontsize=12)
    ax.set( ylabel="Features",
           xlabel="Estimated Correlations")
    
    sns.despine(left=True, bottom=True)
    if save:
        f.savefig('figures/'+filename, bbox_inches='tight')
    plt.show()
    return f, ax

def summary_loading(fitted_pca,features, save=False):
    '''
    takes the input of a PCA, creates a tables with the components as variables and the correlation between variables and the component as row
    '''
    pc_feature = [f'PC{i+1}' for i in range(len(fitted_pca.components_))]
    loadings = fitted_pca.components_.T * np.sqrt(fitted_pca.explained_variance_)

    loading_matrix = pd.DataFrame(loadings, columns=pc_feature, index=features)
    # Find the column with the highest absolute value
    loading_matrix['highest_pc_column'] = loading_matrix.abs().idxmax(axis=1)
    
    # Retrieve the original values for the highest_pc
    loading_matrix['highest_pc'] = loading_matrix.apply(lambda row: row[row['highest_pc_column']], axis=1)
    
    # Sort the DataFrame by the highest_pc_column
    loading_matrix = loading_matrix.sort_values(by='highest_pc_column').reset_index(names = 'features')
    if save:
        loading_matrix.to_csv('figures/loading_all.csv')
    return loading_matrix

def do_pca_team(df, n_components=2):
    '''
    take a df where each team is on a row, selects n_components and make a graph
    '''
    # Assuming target_names is a list of your categorical target names
    label_encoder = LabelEncoder()
    target_names =df['Team']
    y_encoded = label_encoder.fit_transform(df['Team'])

    pca = decomposition.PCA(n_components)

    X=split_df.drop(columns=['Date','Game ID', 'Team'])

    X_r = pca.fit(X).transform(X)
 
    # Percentage of variance explained for each components
    print(
        "explained variance ratio (first two components): %s"
        % str(pca.explained_variance_ratio_)
    )


    plt.figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
              '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
              '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5']
    lw = 2
    
    for color, i, target_name in zip(colors, range(20), target_names):
        plt.scatter(
            X_r[y_encoded == i, 0], X_r[y_encoded == i, 1], color=color, alpha=0.8, lw=lw, label=target_name
        )
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("PCA of IRIS dataset")
    plt.show()
    
    return pca

def do_lda_team(df, n_components=2):
    '''
    do lda for the teams
    '''
    label_encoder = LabelEncoder()
    target_names =df['Team']
    y_encoded = label_encoder.fit_transform(df['Team'])
    lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=n_components)
    X_r = lda.fit(X, y_encoded).transform(X)
    
    
    plt.figure()
    for color, i, target_name in zip(colors, range(20), target_names):
        plt.scatter(
            X_r[y_encoded == i, 0], X_r[y_encoded == i, 1], alpha=0.8, color=color, label=target_name
        )
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("LDA of IRIS dataset")
    
    plt.show()
