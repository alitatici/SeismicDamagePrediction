import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os
from Preprocess4ML import *
import seaborn as sns

def preprocessDataFrame(preprocess_df4ml):

    preprocess_df4ml['Column Area'] = preprocess_df4ml['Column Depth'] * preprocess_df4ml['Column Width']

    # Drop columns that is not related with ML analyses.
    preprocess_df4ml.drop(['Run ID', 'Scale Factor', 'Building ID', 'Beam Width', 'Beam Depth', 'Column Width',
                           'Column Depth', 'Steel Strength', 'First Mode Period', 'Period Class', 'Earthquake URL',
                           'PGA', 'PGV', 'Sa(T1)', 'Sd(T1)', 'CAV'], inplace=True, axis=1)

    ## If 'First Storey - Commercial Use' == 'Yes' then 1; else 0.
    preprocess_df4ml["First Storey - Commercial Use"].replace(['Yes', 'No'], [1, 0], inplace=True)

    preprocess_df4ml['MIDR'] = preprocess_df4ml['MIDR'].\
        replace(preprocess_df4ml.MIDR[preprocess_df4ml.MIDR > 0.02].values, 0.02)

    ## MIDR Classification
    # MIDR_limits = np.arange(0,0.021,0.001)
    MIDR_limits = [0, 0.001, 0.002, 0.005, 0.01, np.inf] # SECED
    MIDR_categorized = pd.cut(preprocess_df4ml['MIDR'], bins=MIDR_limits, labels=False)
    preprocess_df4ml['MIDR'] = MIDR_categorized

    # counts = preprocess_df4ml['MIDR Categorized'].value_counts()

    ## Normalizations
    preprocess_df4ml['Number of Storey'] = \
        (preprocess_df4ml['Number of Storey'].values - min(preprocess_df4ml['Number of Storey'].values)) / \
        (max(preprocess_df4ml['Number of Storey'].values) - min(preprocess_df4ml['Number of Storey'].values))

    preprocess_df4ml['Number of Span'] = \
        (preprocess_df4ml['Number of Span'].values - min(preprocess_df4ml['Number of Span'].values)) / \
        (max(preprocess_df4ml['Number of Span'].values) - min(preprocess_df4ml['Number of Span'].values))

    preprocess_df4ml['Span Length'] = \
        (preprocess_df4ml['Span Length'].values - min(preprocess_df4ml['Span Length'].values)) / \
        (max(preprocess_df4ml['Span Length'].values) - min(preprocess_df4ml['Span Length'].values))

    preprocess_df4ml['Storey Height'] = \
        (preprocess_df4ml['Storey Height'].values - min(preprocess_df4ml['Storey Height'].values)) / \
        (max(preprocess_df4ml['Storey Height'].values) - min(preprocess_df4ml['Storey Height'].values))

    preprocess_df4ml['Column Area'] = \
        (preprocess_df4ml['Column Area'].values - min(preprocess_df4ml['Column Area'].values)) / \
        (max(preprocess_df4ml['Column Area'].values) - min(preprocess_df4ml['Column Area'].values))

    preprocess_df4ml['Concrete Strength'] = \
        (preprocess_df4ml['Concrete Strength'].values - min(preprocess_df4ml['Concrete Strength'].values)) / \
        (max(preprocess_df4ml['Concrete Strength'].values) - min(preprocess_df4ml['Concrete Strength'].values))

    preprocess_df4ml['Soil Condition'] = \
        (preprocess_df4ml['Soil Condition'].values - min(preprocess_df4ml['Soil Condition'].values)) / \
        (max(preprocess_df4ml['Soil Condition'].values) - min(preprocess_df4ml['Soil Condition'].values))

    preprocess_df4ml['PGD'] = \
        (preprocess_df4ml['PGD'].values - min(preprocess_df4ml['PGD'].values)) / \
        (max(preprocess_df4ml['PGD'].values) - min(preprocess_df4ml['PGD'].values))

    return preprocess_df4ml

def visualizing(ml_df):
    # =============================================================================
    # incelendi, doldurmak için kullanılMAdı.
    # =============================================================================
    # g = sns.FacetGrid(data_input, row='First Storey - Commercial Use')
    # g.map(sns.pointplot, 'Storey Height', 'Number of Storey')
    # g.add_legend()
    # plt.show()

    sns.heatmap(ml_df.corr(), annot=True, fmt=".2f")
    plt.show()

    # g1 = sns.factorplot(x='Structural_System', y='Slab_Material', data=df, kind='box')
    # g1.add_legend()
    # g1.set_xticklabels(['Hybrid', 'Moment Frame', 'Masonry'])
    # g1.set(xlabel='', ylabel="Slab_Material", title='Box Plot of Number of Story and Structural System(before filling)')
    # plt.show()
    #
    # g1 = sns.factorplot(x='NumberOfStorey', y='Slab_Material', data=df, kind='box')
    # g1.add_legend()
    # g1.set(xlabel ='', ylabel = "Slab_Material", title ='Box Plot of Number of Story and Structural System(before filling)')
    # plt.show()

    # plt.hist(ml_df['Concrete Strength'], bins=5)
    # plt.show()

    # ml_df.boxplot(column="Concrete Strength", by="First Storey - Commercial Use")
    # plt.show()

    # plt.hist(ml_df['Span Length'], bins=3)
    # plt.show()

    # sns.jointplot(data=ml_df, x="Concrete Strength", y="Span Length")
    # plt.show()

    # sns.jointplot(
    #     data=ml_df,
    #     x="Concrete Strength", y="Span Length", hue="First Storey - Commercial Use"
    # )
    # plt.show()

    # sns.pairplot(ml_df)
    # plt.show()
    #
    # g = sns.PairGrid(ml_df)
    # g.map_upper(sns.histplot)
    # g.map_lower(sns.kdeplot, fill=True)
    # g.map_diag(sns.histplot, kde=True)

    # g = sns.FacetGrid(ml_df, col="Concrete Strength",  row="Span Length")
    # g.map_dataframe(sns.histplot, x="Soil Condition")
    # plt.show()

    # g = sns.FacetGrid(ml_df, col = 'Concrete Strength')
    # g.map(sns.distplot, "PGV", bins = 5)
    # plt.show()

    # sns.histplot(data=ml_df, x="PGV", hue="Concrete Strength")
    # plt.show()

    # ml_df['Concrete Strength'] = ml_df['Concrete Strength'].astype(str)
    ml_df_3 = ml_df[ml_df['Number of Storey'] == 3]
    ml_df_5 = ml_df[ml_df['Number of Storey'] == 5]
    # sns.kdeplot(ml_df_3, x = 'PGV', hue = 'Concrete Strength', shade=True)
    # sns.kdeplot(ml_df, x = 'PGV', hue = 'Concrete Strength', shade=True, cumulative=True, common_norm=False, common_grid=True)
    sns.kdeplot(ml_df_5, x='PGV', hue='Concrete Strength', shade=True)
    # sns.kdeplot(ml_df, x = 'PGV', hue = 'Span Length', shade=True, label="Data 2")
    # Add a title and legend to the plot
    plt.title("Probability Density Functions")
    # plt.legend()
    # Show the plot
    plt.show()

    # g = sns.catplot(x = "Concrete Strength", y = "Span Length", data = ml_df, kind = "bar")
    # g.set_ylabels("Fail Probability")
    # plt.show()
    #
    # g = sns.FacetGrid(ml_df, col = 'Survived')
    # g.map(sns.distplot, "Age", bins = 25)
    # plt.show()


def mistakenlyFailed(data_input, data_input_failed):

    failed_run_id = data_input_failed['Run ID'][data_input_failed['NumberOfStep'] - data_input_failed['Total Step'] == 0]
    failed_MIDR = data_input_failed['MIDR'][data_input_failed['NumberOfStep'] - data_input_failed['Total Step'] == 0]
    data_input.loc[data_input['Run ID'].isin(failed_run_id), 'MIDR'] = failed_MIDR.values

    return data_input

def collapsedRuns(data_input):

    maxMIDR = max(data_input['MIDR'])
    data_input['MIDR'] = data_input['MIDR'].\
        replace(data_input.MIDR[data_input.MIDR == -999].values, maxMIDR)

    return data_input


def preprocessDataAnalysis(preprocess_df4ml):

    preprocess_df4ml['Column Area'] = preprocess_df4ml['Column Depth'] * preprocess_df4ml['Column Width']

    # Drop columns that is not related with ML analyses.
    preprocess_df4ml.drop(['Run ID', 'Scale Factor', 'Building ID', 'Beam Width', 'Beam Depth', 'Column Width',
                           'Column Depth', 'Steel Strength', 'First Mode Period', 'Period Class', 'Earthquake URL'
                           ], inplace=True, axis=1)

    ## If 'First Storey - Commercial Use' == 'Yes' then 1; else 0.
    preprocess_df4ml["First Storey - Commercial Use"].replace(['Yes', 'No'], [1, 0], inplace=True)

    preprocess_df4ml['MIDR'] = preprocess_df4ml['MIDR'].\
        replace(preprocess_df4ml.MIDR[preprocess_df4ml.MIDR > 0.02].values, 0.02)

    # ## MIDR Classification
    # MIDR_limits = [0, 0.001, 0.002, 0.005, 0.01, np.inf] # SECED
    # MIDR_categorized = pd.cut(preprocess_df4ml['MIDR'], bins=MIDR_limits, labels=False)
    # preprocess_df4ml['MIDR'] = MIDR_categorized

    return preprocess_df4ml