#!/usr/bin/env python
# coding: utf-8

# In[56]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import warnings; warnings.simplefilter('ignore')


# In[57]:


def convert_to_float(data_fm, features):
    for column in features:
        data_fm[column].fillna(method='bfill', inplace=True)
        data_fm[column].fillna(method='ffill', inplace=True)
        data_fm[column].fillna(method='backfill', inplace=True)
        
        if data_fm[column].dtype == 'object':
            try:
                data_fm[column] = data_fm[column].astype(np.float64)
            except:
                data_fm[column] = data_fm[column].str.replace(',', '').astype(np.float64)
                continue
            
        data_fm.fillna(method='pad', inplace=True)
        df.fillna(0, inplace=True)

    return data_fm

def encode_label(edited_df):
    for df in [edited_df]:

        X = df.select_dtypes(include=['object'])
        print("Columns size = ", (X.shape[1]))
        if (X.shape[1]) >= 1 and "fund_id" not in X.columns:
            le = preprocessing.LabelEncoder()
            X_2 = X.astype(str).apply(le.fit_transform)

            for col in X_2.columns:
                df[col] = X_2[col]

        df.fillna(method='pad', inplace=True)
        df.fillna(0, inplace=True)
        
    return df

def cleanse_data(data_to_cleanse, label_to_float):
    data_to_cleanse = convert_to_float(data_to_cleanse, label_to_float)
    data_to_cleanse = encode_label(data_to_cleanse)
    
    print("cleanse_data ", data_to_cleanse.info())
    #print(data_to_cleanse.describe(include='all'))
    
    return data_to_cleanse


# In[58]:


def encode_label_without_dropna(edited_df):
    for df in [edited_df]:
        X = df.select_dtypes(include=['object'])
        print("Columns size = ", (X.shape[1]))
        if (X.shape[1]) >= 1:
            le = preprocessing.LabelEncoder()
            X_2 = X.astype(str).apply(le.fit_transform)

            for col in X_2.columns:
                df[col] = X_2[col]

        df.fillna(method='pad', inplace=True)
        df.fillna(0, inplace=True)
        
    return df

def cleanse_data_without_dropna(data_to_cleanse, label_to_float):
    data_to_cleanse = convert_to_float(data_to_cleanse, label_to_float)
    data_to_cleanse = encode_label_without_dropna(data_to_cleanse)

    return data_to_cleanse


# In[172]:


min_max_correlation_cutoff_point = 0.005

def drop_unused_columns(data, columns):
    for column in columns:
        if column in data.columns:
            data.drop(columns=[column], inplace=True)
            
    return data

def join_file(data_1, data_2):
    joint_file = []
    if "tag" in data_1.columns:
        data_2 = drop_unused_columns(data_2, ['fund_id'])
        joint_file = data_1.merge(data_2, how='left', on='tag')
    else:
        data_2 = drop_unused_columns(data_2, ['tag'])
        joint_file = data_1.merge(data_2, how='left', on='fund_id')
        
    return joint_file

def drop_low_correlation_columns(x):
    if x.values[0] < min_max_correlation_cutoff_point and x.values[0] > -(min_max_correlation_cutoff_point):
        if x.name == "tag" or x.name == 'fund_id':
            join_1.drop(columns=[x.name], inplace = True) 

def print_correlation(data):
    join_1 = []
    tag_rating = pd.read_csv('tag_rating.csv')
    tag_rating = encode_label(tag_rating)
    
    if 'greatstone_rating' in data.columns:
        data.drop(columns='greatstone_rating', inplace=True)
    
    if 'fund_id' in data.columns:
        join_1 = drop_unused_columns(data, ['tag'])
        join_1 = tag_rating.merge(data, how='left', on='fund_id')
    else:
        join_1 = drop_unused_columns(data, ['fund_id'])
        join_1 = tag_rating.merge(data, how='left', on='tag')
    
    print("1.1", [column for column in join_1.columns])
    list_of_correlation = join_1.corr(method='pearson')[['greatstone_rating']].sort_values(by=['greatstone_rating'], ascending=False)
    join_1 = drop_unused_columns(join_1, ['greatstone_rating'])
    
    print("top list of correlation ", list_of_correlation[:20])
    print("bottom list of correlation ", list_of_correlation[-5:])
    #list_of_correlation.apply(lambda x: print("start", x.values[0], x.name), axis=1)
    
    list_of_correlation.apply(lambda x: join_1.drop(columns=[x.name], inplace = True) if ((x.values[0] < min_max_correlation_cutoff_point and x.values[0] > -(min_max_correlation_cutoff_point))) else 0, axis=1)
    #list_of_correlation.apply(drop_low_correlation_columns(join_1), axis=1)
    #list_of_correlation.apply(lambda x: x if list_of_correlation[x.name] > 0.01 else 0, axis=1)
    #list_of_correlation.apply(lambda x: join_1.drop(columns=[x.name]) if ), axis=1)
    
    join_1['fund_id'] = tag_rating['fund_id']
    join_1['tag'] = tag_rating['tag']
    
    print("join_1 ", join_1.info())
    
    return join_1
    


# # Cleanse Data

# ## fund_config

# In[173]:


fund_config = pd.read_csv('fund_config.csv')
fund_config.info()

fund_config = encode_label(fund_config)
fund_config.describe()
fund_config = fund_config.drop(columns=['fund_name', 'parent_company'])


# In[174]:


fund_config.describe()
fund_config.to_csv("file_1.csv")


# # fund_ratios

# In[175]:


fund_ratios = pd.read_csv('fund_ratios.csv')

fund_ratios = cleanse_data(fund_ratios, ['ps_ratio', 'mmc', 'pc_ratio'])


# In[176]:


fund_ratios = print_correlation(fund_ratios)
fund_ratios.to_csv("file_2.csv")


# In[177]:


fund_ratios.hist(stacked=False, bins=20, figsize=(12, 12))


# In[178]:


join_1 = join_file(fund_config, fund_ratios)
join_1.head()
join_1.info()


# ## bond_ratings

# In[179]:


bond_ratings = pd.read_csv('bond_ratings.csv')
bond_ratings.info()
bond_ratings = cleanse_data(bond_ratings, [])


# In[180]:


bond_ratings = print_correlation(bond_ratings)
bond_ratings.to_csv("file_3.csv")


# In[181]:


bond_ratings.hist(stacked=False, bins=20, figsize=(12, 12))


# In[182]:


join_2 = join_file(join_1, bond_ratings)
join_2.head()
join_2.info()


# ## fund_allocations

# In[183]:


fund_allocations = pd.read_csv('fund_allocations.csv')
fund_allocations['tag'] = fund_allocations['id']
fund_allocations = cleanse_data(fund_allocations, [])


# In[184]:


fund_allocations = print_correlation(fund_allocations)
fund_allocations.to_csv("file_4.csv")


# In[185]:


fund_allocations.hist(stacked=False, bins=20, figsize=(12, 12))


# In[186]:


join_3 = join_file(join_2, fund_allocations)
join_3.head()
join_3.info()


# ## fund_specs

# In[187]:


fund_specs = pd.read_csv('fund_specs.csv')
fund_specs = cleanse_data(fund_specs, [])


# In[188]:


fund_specs = print_correlation(fund_specs)
fund_specs.to_csv("file_5.csv")


# In[189]:


fund_specs.hist(stacked=False, bins=20, figsize=(12, 12))


# In[190]:


join_4 = join_file(join_3, fund_specs)
join_4.head()
join_4.info()


# ## other_specs

# In[191]:


other_specs = pd.read_csv('other_specs.csv')
other_specs = cleanse_data(other_specs, [])


# In[192]:


other_specs = print_correlation(other_specs)
other_specs.to_csv("file_6.csv")


# In[193]:


other_specs.hist(stacked=False, bins=20, figsize=(12, 12))


# In[194]:


join_5 = join_file(join_4, other_specs)
join_5.head()
join_5.info()


# ## return_3year

# In[195]:


return_3year = pd.read_csv('return_3year.csv')
return_3year = cleanse_data(return_3year, [])
return_3year = print_correlation(return_3year)
return_3year.to_csv("file_7.csv")
join_6 = join_file(join_5, return_3year)


# In[196]:


return_5year = pd.read_csv('return_5year.csv')
return_5year = cleanse_data(return_5year, [])
return_5year = print_correlation(return_5year)
return_5year.to_csv("file_8.csv")
print("join file")
join_7 = join_file(join_6, return_5year)


# In[197]:


return_10year = pd.read_csv('return_10year.csv')
return_10year = cleanse_data(return_10year, [])
return_10year = print_correlation(return_10year)
return_10year.to_csv("file_9.csv")
join_8 = join_file(join_7, return_10year)

print("+++++++++++++++++++++++++++++++++++++++")
join_8.info()


# In[198]:


join_8 = print_correlation(join_8)
print("----------- header----------------")


# # Final Test

# # Train and Evaluate Data

# In[199]:


complete_predict_final_data_with_object.shape


# In[200]:


joint_9.hist(stacked=False, bins=20, figsize=(12, 20))


# In[212]:


from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

def remove_header(data, header_to_remove):
    if header_to_remove in data.columns:
        data = data.drop(columns=[header_to_remove])
    return data

#complete_predict_final_data_with_object = final_data.copy(deep=True)
tag_rating = pd.read_csv("tag_rating.csv")
joint_9 = join_8.merge(tag_rating[['greatstone_rating', 'fund_id']],  how='left', on='fund_id')

features_to_log_transform = ['fund_return_3years_x', 'fund_return_3years_y', '5_years_return_fund', '5yrs_sharpe_ratio_fund', 
                             '3yrs_sharpe_ratio_fund', '3_years_alpha_fund', '5_years_alpha_fund', '10_years_return_fund', 
                             '10yrs_sharpe_ratio_fund', '5_years_return_mean_annual_fund', '3_years_return_mean_annual_fund', 
                             '1_year_return_fund', '10_years_alpha_fund', '2018_return_fund', 
                             '10_years_return_mean_annual_fund', '2015_return_fund', '2014_return_fund', 
                             '1_month_fund_return', '10yrs_sharpe_ratio_category']

for feature in features_to_log_transform:
    if np.min(joint_9[feature])>0: # Look for integer data series
        joint_9[feature] = np.log1p(joint_9[feature])

complete_predict_final_data_with_object = joint_9.copy(deep=True)
joint_9.to_csv("complete_predict_final_data_with_object.csv")

complete_predict_final_data_without_object = joint_9.copy(deep=True)
complete_predict_final_data_without_object = cleanse_data_without_dropna(complete_predict_final_data_without_object, [])
complete_predict_final_data_without_object.to_csv("complete_predict_final_data_without_object.csv")


final_data = joint_9.copy(deep=True)
final_data = final_data.dropna(axis=0)
#print(")))))))))))))))))))))))))))))) is null ", final_data.isnull())

final_data.info()
print("===================================")
final_data = remove_header(final_data, 'tag')
final_data = remove_header(final_data, 'fund_id')
final_data = remove_header(final_data, 'fund_id_x')
final_data = remove_header(final_data, 'fund_id_y')
final_data = remove_header(final_data, 'fund_return_3years_x')
#final_data = remove_header(final_data, 'us_govt_bond_rating')
final_data = remove_header(final_data, 'inception_date')
final_data = remove_header(final_data, 'Unnamed')

final_data = cleanse_data(final_data, [])
final_data.to_csv("test_1.csv")
#final_data.to_csv("test_2.csv")
#pca = PCA(n_components=6)
#pca_fit = pca.fit(X_train)
#print(pca_fit.explained_variance_ratio_)
#X_train = pca_fit.transform(X_train)

#X_test = pca_fit.transform(X_test)
print("#####################################")

X_df = final_data.loc[:, final_data.columns != 'greatstone_rating']
y_df = final_data['greatstone_rating']

X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size = 0.30, random_state=0)


# In[210]:


final_data[['fund_return_3years_y', 'inception_date']].describe()


# In[166]:


X_df = final_data.loc[:, final_data.columns != 'greatstone_rating']
y_df = final_data['greatstone_rating']

X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size = 0.30, random_state=0)


# In[152]:


complete_predict_final_data_without_object.info()


# # Classifier

# In[202]:


from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
from sklearn.metrics import r2_score

d = {'data type': ['train'], 'search' : ['random search'], 'name': ['0'], 'estimator': [0], 'r2 score': [0]}
model_performance = pd.DataFrame(data=d)

# User Random Search model
def random_search_classifier(model_type, model_name, distributions, model_performance):
    #pipe_svc = Pipeline([('scl', StandardScaler()),  ('pca', PCA(n_components=12)), (model_name, model_type)]) 
    pipe_svc = Pipeline([('scl', StandardScaler()), (model_name, model_type)]) 
    #pipe_svc = Pipeline([('scl', StandardScaler()), (model_name, model_type)]) 
    grid = RandomizedSearchCV(pipe_svc , distributions, cv = 5) 
    grid_fit = grid.fit(X_train, y_train) 
    y_train_pred = grid.predict(X_train)
    y_test_pred = grid.predict(X_test)
    
    print("Test Recall score : {:.3f}". format(recall_score(y_test, y_test_pred, average="micro"))) 
    #print("Accuracy score : {:.3f}". format(accuracy_score(y_train, y_train_pred, average="micro")))  
    #print("Test Precision score : {:.3f}". format(precision_score(y_test, y_test_pred, average="micro")))
    print("Test Precision score : {:.3f}". format(precision_score(y_test, y_test_pred, average="micro")))
    
    print("Test Recall score : {:.3f}". format(recall_score(y_test, y_test_pred, average="macro"))) 
    #print("Accuracy score : {:.3f}". format(accuracy_score(y_train, y_train_pred, average="micro")))  
    #print("Test Precision score : {:.3f}". format(precision_score(y_test, y_test_pred, average="micro")))
    print("Test Precision score : {:.3f}". format(precision_score(y_test, y_test_pred, average="macro")))
    
    print("Train Recall score : {:.3f}". format(recall_score(y_train, y_train_pred, average="micro"))) 
    #print("Accuracy score : {:.3f}". format(accuracy_score(y_train, y_train_pred, average="micro")))  
    print("Train Precision score : {:.3f}". format(precision_score(y_train, y_train_pred, average="micro"))) 
 
    print(f"Best parameters are {grid.best_params_}")
    
    d = {'data type': ['train'], 'search' : ['random search'], 'name': [str(model_type)], 'estimator': [str(grid.best_params_)], 'r2 score': ["{:.3f}". format(r2_score(y_train, y_train_pred))]}
    pd_params = pd.DataFrame(data=d)

    model_performance = pd.concat([model_performance, pd_params])
    
    d = {'data type': ['test'], 'search' : ['random search'], 'name': [str(model_type)], 'estimator': [str(grid.best_params_)], 'r2 score': ["{:.3f}". format(r2_score(y_test, y_test_pred))]}
    pd_params = pd.DataFrame(data=d)
    model_performance = pd.concat([model_performance, pd_params]).reset_index(drop=True)
    
    return model_performance, grid


# In[213]:


param_grid = {'rf__max_depth': [25], 'rf__bootstrap':[False], 'rf__ccp_alpha':[0], 'rf__verbose':[5], 'rf__min_samples_leaf':[2], 'rf__warm_start':[False], 'rf__class_weight':[None]} 
model_performance, rf_classifier_model = random_search_classifier(RandomForestClassifier(), 'rf', param_grid, model_performance)


# # Gradient Boosting

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
param_grid = {'rf__n_estimators': [100, 1000], 'rf__learning_rate':[0, 0.3], 'rf__max_depth':[5,10], 'rf__verbose':[5]} 
model_performance, gb_classifier_model = random_search_classifier(GradientBoostingClassifier(), 'rf', param_grid, model_performance)


# In[46]:


from sklearn.neural_network import MLPClassifier

param_grid = {'rf__hidden_layer_sizes':[(500,)], 'rf__alpha':[1e-05], 'rf__max_iter':[500], 'rf__verbose':[True]} 
model_performance, gb_classifier_model = random_search_classifier(MLPClassifier(), 'rf', param_grid, model_performance)


# # Predict Result

# In[53]:


complete_predict_final_data_with_object.shape


# In[54]:


complete_predict_final_data_without_object.shape


# In[55]:


X_df.shape


# In[ ]:


X_train.columns


# In[ ]:


complete_predict_final_data_without_object.info()


# In[171]:


result_pd = pd.DataFrame()
result_pd['greatstone_rating'] = rf_classifier_model.predict(complete_predict_final_data_without_object[X_train.columns])
result_pd['fund_id'] = complete_predict_final_data_with_object['fund_id']
result_pd['fund_id'] = result_pd['fund_id'].astype(object)
result_pd.to_csv("predicted.csv")

sample_submission = pd.read_csv('sample_submission.csv')

result_1 = sample_submission.drop(columns=['greatstone_rating']).merge(result_pd, how='left', on='fund_id')
result_1.to_csv("to_submit.csv")


# # Results

# In[ ]:


sample_submission = pd.read_csv('sample_submission.csv')
sample_submission.info()

sample_submission.head()


# In[ ]:


other_specs = pd.read_csv('other_specs.csv')
other_specs_submission  = other_specs[['tag' , 'greatstone_rating']]

print("other specs------------------")
print(other_specs_submission.head())

######################################
fund_ratios = pd.read_csv('fund_ratios.csv')
fund_ratios_partial = fund_ratios[['tag', 'fund_id']]

final_1 = fund_ratios_partial.merge(other_specs_submission, how='inner', on='tag')
print("fund_ratios------------------")
print(final_1.head())

######################################
final_2 = sample_submission.merge(final_1, how='inner', on='fund_id')
print("sample_submission------------")
print(final_2.head())

final_2.to_csv("submit.csv")

