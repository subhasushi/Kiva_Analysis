import pandas as pd
import collections

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import pickle

import json

import requests

n_clusters = 4
            
def describe_numerical(df):
    grouped = df.groupby('class').agg(['median']).transpose().round(2)
    grouped = grouped.reset_index(level=1)
    grouped = grouped.drop('level_1',axis=1)
    numerical_stats_json = json.dumps(grouped.transpose()[['loan_amount','female_count', 
                            'male_count','population_in_mpi',
                            'lender_term', 'num_lenders_total', 
                            'posted_raised_hours','num_journal_entries']].to_dict(orient='records'))
    
    return numerical_stats_json

def describe_categorical(df):

    cluster_frequency = []

    for k in range(n_clusters):
        kth_cluster = df[df['class']==k]

        category_values = {}

        for c in ['country','sector_name','activity_name','status']:
            top_values = {str(c):[{}]}
             
            cat_values = kth_cluster[c].value_counts().head(5)

            for record in cat_values.items():
                top_values[str(c)][0].update({str(record[0]):int(record[1])})

            category_values.update(top_values)

        cluster_frequency.append(category_values)
        
    return cluster_frequency


def single_cluster_numerical(df,n):
    grouped = df.groupby('class').agg(['median']).transpose().round(2)
    grouped = grouped.reset_index(level=1)
    grouped = grouped.drop('level_1',axis=1)
    numerical_stats_json = grouped.transpose()[['loan_amount','female_count', 
                            'male_count','population_in_mpi',
                            'lender_term', 'num_lenders_total', 
                            'posted_raised_hours','num_journal_entries']].to_dict(orient='records')

    return numerical_stats_json[int(n)]

def single_cluster_categorical(df,k):
    kth_cluster = df[df['class']==int(k)]

    category_values = {}

    for c in ['country','sector_name','activity_name','status']:
        top_values = {c:[{}]}

        cat_values = kth_cluster[c].value_counts().head(5)

        for record in cat_values.items():
            top_values[c][0].update({str(record[0]):int(record[1])})

        category_values.update(top_values)

    return category_values

def getLatestLoans():
    response = requests.get('http://api.kivaws.org/v1/loans/newest.json').json()
    loan_items = response['loans']

    print('Getting Response !')

    new_loans_dictionary = []

    for loan in loan_items:
        new_item = {
            'loan_amount': loan['loan_amount'],
            'status': loan['status'],
            'activity_name': loan['activity'],
            'sector_name': loan['sector'],
            'borrower_count': loan['borrower_count'],
            'country_name': loan['location']['country'].lower()
        }

        new_loans_dictionary.append(new_item)

    new_loans_df = pd.DataFrame(new_loans_dictionary)

    new_loans_df = new_loans_df.rename(columns={'country_name':'country'})

    mpi = pd.read_csv('mpi.csv',encoding='latin-1')

    mpi_reduced = mpi[['country','mpi','population_in_mpi','Education','Health','Living standards','population_in_severe_mpi']]

    mpi_reduced['country'] = [x.lower() for x in mpi_reduced['country']]
    merged = new_loans_df.merge(mpi_reduced,on='country')

    print('Getting Response MPI!')

    return merged
    
def applyPCA(n,features):
    pca = PCA(n_components=n) #n-dimensional PCA
    principal_df = pd.DataFrame(pca.fit_transform(features))
    
    # Identifying the significant features of final PCA components
    
    PCA_stats = pd.DataFrame(pca.components_,columns=features.columns,index = range(0,n)).transpose()
    print(f"PCA transformation applied with {n} components:")
    print("\nTop 5 features in each principal component")
    print([PCA_stats[c].nlargest(5) for c in PCA_stats])
    print('-----------------------------------------------')
    
    return principal_df


def getFeatureVector(dataset):
    
    X = dataset[['loan_amount', 'status', 'activity_name', 'sector_name','borrower_count','mpi', 'population_in_mpi',
                 'Education', 'Health', 'Living standards', 'population_in_severe_mpi']]

    dataset_dummies = pd.get_dummies(X)

    scaler = StandardScaler()

    scaler.fit(dataset_dummies)

    data_scaled = scaler.transform(dataset_dummies)

    feature = pd.DataFrame(data_scaled)

    feature.columns = dataset_dummies.columns
    
    return feature