'''
CS 5228 Knowledge Discovery and Data Mining
AY21/22 Semester 2
Final Project 

Chan Pei Shan        / A0243461L	
Chua Mei Yun         / A0131114A	
Goh Chee Hoe         / A0227330R	
Lim Jun Yang Leonard / A0086792M

Support Routines

'''
# --------------------------
# versions used
# --------------------------
# python          3.95
# numpy           1.22.0
# pandas          1.3.5
# scikit-learn    1.0.2
# matplotlib      3.5.1
# seaborn         0.11.2
# geopandas       0.10.2
# --------------------------

import os
import datetime
import inspect
import re 

import numpy as np
import pandas as pd 
from pandas.api.types import is_string_dtype, is_numeric_dtype

from matplotlib import rcdefaults, pyplot as plt
import seaborn as sns
import geopandas as gpd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, mean_squared_error, pairwise_distances
from sklearn.metrics.pairwise import haversine_distances
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor

#---------------------------------------------------------------------
#  Repeated definition of folder structure
#---------------------------------------------------------------------
datapath   = 'input'
auxpath    = os.path.join(datapath,'auxiliary-data')
outputpath = 'output'
submitpath = 'submissions'

#---------------------------------------------------------------------
#  current year when used in URA dataset merging
#---------------------------------------------------------------------
this_year = datetime.datetime.now().year

#---------------------------------------------------------------------
#  Saves dataframe as csv file in the output path
#  Creates output path if it does not exists
#  index_label toggles if index column is included in cs file
#---------------------------------------------------------------------
#
def save_df(df,fname=None, outpath='',index_label=None, hops=1):

    # test existence of outputpath, create if don't exists
    if ( outpath!='' and not os.path.exists(outpath) ): os.mkdir(outpath)
    
    # get the caller's actual arguments
    # find the original argument's name 
    func_back = inspect.currentframe()
    for i in range(hops): func_back = func_back.f_back
    vars = func_back.f_locals.items()
#    vars = inspect.currentframe().f_back.f_locals.items()
    for vname, value in vars: 
        if value is df: df_name = vname
        
    # use caller supplied fname if exists
    if fname: df_name = fname

    out_fname = os.path.join(outpath,df_name+'.csv')
    if index_label:
        df.to_csv(out_fname,index_label=index_label)
    else:
        df.to_csv(out_fname,index=False)


#---------------------------------------------------------------------
# Drops columns from dataframe
# Only drops columns that are present even if lst of columns is larger set
#---------------------------------------------------------------------
#
def drop_cols(df, cols):

    df           = df.copy()
    cols_df      = list(df.columns)
    cols_to_drop = list( set(cols) & set(cols_df) )
    df           = df.drop(columns=cols_to_drop)
    return df
# 
#---------------------------------------------------------------------

#---------------------------------------------------------------------
# function to fill NaN's with the mode for the categorical attribute
#---------------------------------------------------------------------
#
# def fillmode_orig(df, groupby:list, colname, df_summary=None):
    
    # df = df.copy()
    # df_mode = df.dropna(subset=[colname]).groupby(groupby)
    # df_mode = df_mode[colname].agg(lambda dummy:dummy.value_counts(sort=True,ascending=False).index[0])
    # df_mode = df_mode.reset_index()[groupby+[colname]]
    # df = df.merge(df_mode,left_on=groupby, right_on=groupby, how='left', suffixes=(None,'_y'))
    # df[colname] = df[colname+'_y']
    # df = df.drop(columns=[colname+'_y'])
    # if df_summary is not None: df_summary[colname] = df.isna().sum()
    # return df
# 
#---------------------------------------------------------------------

#---------------------------------------------------------------------
# function to fill NaN's with the mode for the categorical attribute
# more precise than fillmode, which fills all in the group
# fillnanmode fills only the NaN's
#---------------------------------------------------------------------
#
def fillmode(df, groupby:list, colname, df_summary=None):
    
    df = df.copy()
    df_mode = df.dropna(subset=[colname]).groupby(groupby)
    df_mode = df_mode[colname].agg(lambda dummy:dummy.value_counts(sort=True,ascending=False).index[0])
    df_mode = df_mode.reset_index()[groupby+[colname]]

    df_nan = df[df[colname].isna()][groupby+[colname]]
    df_nan = df_nan.reset_index().merge( df_mode,on=groupby,how='left',suffixes=['','_xtab'] ).set_index(df_nan.index)
    df_nan[colname] = df_nan[colname+'_xtab']
    df_nan = df_nan.drop(['index']+groupby+[colname+'_xtab'], axis=1)
    
    df.fillna(value=df_nan,axis=0,inplace=True)

    if df_summary is not None: df_summary[colname] = df.isna().sum()

    return df

#---------------------------------------------------------------------
# function to fillmode by pairs of (groupby, col)
# groupby is a list of the group cols
# col is a single column name (string)
# specific function for cleaning validation test set
# uses fillmode() method
# application: zone order may be important
#---------------------------------------------------------------------
#
def fillmode_by_pairs(df, group_col_pairs):
    
    df = df.copy()
    for groupby, col in group_col_pairs: df = fillmode(df,groupby,col)
    
    return df
    
#---------------------------------------------------------------------
# function to fillmode by URA zoning
# specific function for cleaning validation test set
# uses fillmode() method
# application: zone order may be important
#---------------------------------------------------------------------
#
def fillmode_by_zone(df, cols, zones):

    df = df.copy()
    for col in cols:
        for zone in zones:
            if df[col].isna().any(): df = fillmode(df,[zone],col)
    
    return df
# 
#---------------------------------------------------------------------


# ........................................................................................................ #
#                                                                                                          #
#                       ____        _           ____ _                  _                                  #
#                      |  _ \  __ _| |_ __ _   / ___| | ___  __ _ _ __ (_)_ __   __ _                      #
#                      | | | |/ _` | __/ _` | | |   | |/ _ \/ _` | '_ \| | '_ \ / _` |                     #
#                      | |_| | (_| | || (_| | | |___| |  __| (_| | | | | | | | | (_| |                     #
#                      |____/ \__,_|\__\__,_|  \____|_|\___|\__,_|_| |_|_|_| |_|\__, |                     #
#                                 |  _ \ ___  _   _| |_(_)_ __   ___ ___        |___/                      #
#                                 | |_) / _ \| | | | __| | '_ \ / _ / __|                                  #
#                                 |  _ | (_) | |_| | |_| | | | |  __\__ \                                  #
#                                 |_| \_\___/ \__,_|\__|_|_| |_|\___|___/                                  #
#                                                                                                          #
# ........................................................................................................ #

                                                              
#---------------------------------------------------------------------
# clean up districts, some incorrectly coded
#---------------------------------------------------------------------
#
def clean_districts(df):
    
    df = df.copy()
    filter_3 = (df['region']=='central region') & (df['subszone']=='sentosa')
    df.loc[filter_3,'district'] = int(df[filter_3]['district'].mode())
    filter_3 = (df['region']=='north region') & (df['subszone']=='sembawang springs')
    df.loc[filter_3,'district'] = int(df[filter_3]['district'].mode())
    filter_3 = (df['region']=='central region') & (df['subszone']=='anson')
    df.loc[filter_3,'district'] = int(df[filter_3]['district'].mode())
    filter_3 = (df['region']=='central region') & (df['subszone']=='pasir panjang 2')
    df.loc[filter_3,'district'] = int(df[filter_3]['district'].mode())
    filter_3 = (df['region']=='north-east region') & (df['subszone']=='seletar hills')
    df.loc[filter_3,'district'] = int(df[filter_3]['district'].mode())
    filter_3 = (df['region']=='east region') & (df['subszone']=='changi west')
    df.loc[filter_3,'district'] = int(df[filter_3]['district'].mode())
    filter_3 = (df['region']=='central region') & (df['subszone']=='chatsworth')
    df.loc[filter_3,'district'] = int(df[filter_3]['district'].mode())
    filter_3 = (df['region']=='central region') & (df['subszone']=='clark quay')
    df.loc[filter_3,'district'] = 9

    return df
    
#---------------------------------------------------------------------
# clean up planning area for 2 specific properties
#---------------------------------------------------------------------
#
def clean_reg_pa_subzones(df):
    
    df = df.copy()
    filter_1 = (df['name']=='heritage east')
    filter_2 = filter_1 & (df['region']=='east region')
    df.loc[filter_1,'region'] = df.loc[filter_2,'region'].mode().squeeze()
    df.loc[filter_1,'planning_area'] = df.loc[filter_2,'planning_area'].mode().squeeze()
    df.loc[filter_1,'subszone'] = df.loc[filter_2,'subszone'].mode().squeeze()

    filter_3 = (df['name']=='kent residences')
    df.loc[filter_3,'region'] = df.loc[filter_3,'region'].mode().squeeze()
    df.loc[filter_3,'planning_area'] = df.loc[filter_3,'planning_area'].mode().squeeze()
    df.loc[filter_3,'subszone'] = df.loc[filter_3,'subszone'].mode().squeeze()
    
    return df

#---------------------------------------------------------------------
# clean up planning market segment coding
# original dataset had market segemnt for all records set to 'ocr'
# defintion use here is based upon URA's definition
#---------------------------------------------------------------------
#
def clean_mkt_segment_ura(df):
    
    df = df.copy()
    filter_0 = (df['region']=='central region')
    filter_1 = filter_0 & ( (df['district']==9) |  
                            (df['district']==10)|  
                            (df['district']==11)    )
    filter_2 = filter_0 & ( (df['subszone']=='sentosa') |
                            (df['planning_area']=='downtown core') )
    df.loc[:       ,'market_segment'] = 'ocr'
    df.loc[filter_0,'market_segment'] = 'rcr'
    df.loc[filter_1,'market_segment'] = 'ccr'
    df.loc[filter_2,'market_segment'] = 'ccr'
    
    return df

#---------------------------------------------------------------------
# separate bedrooms N+M coding by additon `addrooms` column
#---------------------------------------------------------------------
#
def clean_bedrooms_N_plus_M_orig(df):
    
    df = df.copy()
    if is_string_dtype(df['bedrooms']):
        df = df[(df['bedrooms'].notna())]
        df['addrooms'] = df['bedrooms'].str.split('+').str[1]
#         df['addrooms'] = df['addrooms'].fillna(value=0)
        df['addrooms'].fillna(value=0,inplace=True)
        df['bedrooms'] = df['bedrooms'].str.split('+').str[0]                   #.astype(int)
    return df

def clean_bedrooms_N_plus_M(df):
    
    df = df.copy()
    filter_1 = (df['bedrooms'].notna())

    if is_string_dtype(df['bedrooms']):
        df.loc[filter_1,'addrooms'] = df['bedrooms'].str.split('+').str[1]
        df['addrooms'].fillna(value=0,inplace=True)
        df.loc[filter_1,'bedrooms'] = df['bedrooms'].str.split('+').str[0]
    
    return df

#---------------------------------------------------------------------
# mark known exclusive residential areas
#---------------------------------------------------------------------
#
def mark_excl_residential(df):
    
    df = df.copy()
    df['excl_residential'] = 0
    filter_2 = (df['region']=='central region') & (df['planning_area']=='newton')
    df.loc[filter_2,'excl_residential'] = 1
    filter_2 = (df['region']=='central region') & (df['planning_area']=='river valley')
    df.loc[filter_2,'excl_residential'] = 1
    filter_2 = (df['region']=='central region') & (df['planning_area']=='orchard')
    df.loc[filter_2,'excl_residential'] = 1
    filter_2 = (df['region']=='central region') & (df['planning_area']=='downtown core')
    df.loc[filter_2,'excl_residential'] = 1
    filter_2 = (df['region']=='central region') & (df['planning_area']=='sentosa')
    df.loc[filter_2,'excl_residential'] = 1
    # filter_2 = (df['region']=='central region') & (df['planning_area']=='maritime square')
    # df.loc[filter_2,'excl_residential'] = 1
    
    return df

#---------------------------------------------------------------------
# Merge 'scraped' URA sales transaction 
#---------------------------------------------------------------------
#
def merge_ura_dataset(df, df_ura):
    
    df = df.copy()
    df_u = df_ura.copy()
    
    ura_cols = ['street','type','market_segment','type_of_sale', 'price', 'nett_price', 'no_of_units',
                'area_size', 'type_of_area', 'floor_level', 'pricesqft', 'date_of_sale']
    df_u = drop_cols(df_u,ura_cols)
    df_u = fillmode(df_u,['name'],'district')
    df_u = fillmode(df_u,['name'],'tenure')
    df_u = df_u.drop_duplicates(subset=['name'])
    df_u = df_u.sort_values('name', axis=0)
    df = df.merge( df_u,how='left',on='name',suffixes=['','_ura'] )

    # extract tenure and lease start year from URA dataset
    df = df.assign(lease_start = lambda dummy: dummy['tenure_ura'].str.extract('(\d+)$'))
    df.tenure = df.tenure.fillna(value=99)                     # default NaN's to 99 years

    cols = list(df.columns[df.columns.str.endswith('_ura')])
    df = drop_cols(df ,cols)

    #save a copy
    save_df(df_u,fname='df_ura',outpath=outputpath)

    return df

#---------------------------------------------------------------------
# Merge 'scraped' URA sales transaction 
#---------------------------------------------------------------------
#
def fill_lease_and_ura_info(df):

    df = df.copy()
    
    df_latlng = df.copy()
    df_latlng = df_latlng[(df_latlng.lease_start.isna()) & (df_latlng.built_year.isna())]
    df_latlng = df_latlng[ (df_latlng.street==df_latlng.name) | (df_latlng.street.str.contains(str(df_latlng.name))) ]
    df_latlng = df_latlng[['listing_id','name','street','lat','lng']].drop_duplicates()

    #
    # fill NaN's and match by lat/lng
    #
    for idx, geoLoc in df_latlng.iterrows():
        
        listing_id, name, street, lat, lng = tuple(geoLoc)

        filter1       = ( df.lat==lat) & (df.lng==lng )
        filter2notnan = ( filter1 & df.built_year.notna() )
        filter2nan    = ( filter1 & df.built_year.isna() )
        df_notnan     = df[filter2notnan]
        df_nan        = df[filter2nan]

        if len(df_notnan) != 0:
            built_year  = df_notnan.built_year.mode().squeeze()
            no_of_units = df_notnan.no_of_units.mode().squeeze()
            name        = df_notnan.name.mode().squeeze()
            prop_type   = df_notnan.type.mode().squeeze()
            model       = df_notnan.model.mode().squeeze()
            tenure      = df_notnan.tenure.mode().squeeze()
            lease_start = df_notnan.lease_start.mode().squeeze()

            df.loc[filter2nan,'built_year']  = built_year
            df.loc[filter2nan,'no_of_units'] = no_of_units
            df.loc[filter2nan,'name']        = name
            df.loc[filter2nan,'type']        = prop_type
            df.loc[filter2nan,'model']       = model
            df.loc[filter2nan,'tenure']      = tenure
            df.loc[filter2nan,'lease_start'] = lease_start            
            
    return df
    
def compute_zprices(df):

    df = df.copy()
    
    gp_name     = df.groupby(['name'])
    gp_subzone  = df.groupby(['subszone'])
    gp_planarea = df.groupby(['planning_area'])
    gp_mktseg   = df.groupby(['market_segment'])


    df['zprice_subzone'] = (gp_name['pricesqft'].transform('mean') - gp_subzone['pricesqft'].transform('mean')) /  \
                            gp_subzone['pricesqft'].transform('std')
    df['zprice_subzone'].fillna(value=0,inplace=True)

    df['zprice_pa'] = (gp_subzone['pricesqft'].transform('mean') - gp_planarea['pricesqft'].transform('mean')) /  \
                       gp_planarea['pricesqft'].transform('std')
    df['zprice_pa'].fillna(value=0,inplace=True)

    df['zprice_ms'] = (gp_planarea['pricesqft'].transform('mean') - gp_mktseg['pricesqft'].transform('mean')) /  \
                       gp_mktseg['pricesqft'].transform('std')
    df['zprice_ms'].fillna(value=0,inplace=True)
    
    return df
    
#---------------------------------------------------------------------
# Calculate the lease start year
#---------------------------------------------------------------------
#
def compute_lease_left(df):
    
    df = df.copy()
    
    df['built_year']  = df['built_year'].fillna(value=df.lease_start)      # fill NaN's with lease_start first
    df['built_year']  = df['built_year'].fillna(value=this_year)           # if still NaN, fill with this year
    df['lease_start'] = df['lease_start'].fillna(value=df.built_year)      # fill NaN's with built_year

    filter2 = (df.built_year.astype(int) < df.lease_start.astype(int))     # if lease start later than built_year ...
    df.loc[filter2,'built_year'] = df.lease_start                          # ... than take the later date
    
    df['lease_left'] = 99                                                  # defaults to 99 years (most condo are)
    filter2_5 = df.tenure.notna()
    df.loc[filter2_5,'lease_left'] = df.tenure                             # take lease left as the tenure
    filter3 = (df.tenure=='freehold')                                      # bump up to 10 * 9999 years (there are such propeties !!!)
    df.loc[filter3,'lease_left'] = 999                                     # for freehold (do not overload)

    df.lease_left = df.lease_left.astype(int) - this_year + df.built_year.astype(int)
    
    return df


# ........................................................................................................ #
#                      ____                                            _                                   #
#                     |  _ \ _ __ ___ _ __  _ __ ___   ___ ___ ___ ___(_)_ __   __ _                       #
#                     | |_) | '__/ _ | '_ \| '__/ _ \ / __/ _ / __/ __| | '_ \ / _` |                      #
#                     |  __/| | |  __| |_) | | | (_) | (_|  __\__ \__ | | | | | (_| |                      #
#                     |_|   |_|  \___| .__/|_|  \___/ \___\___|___|___|_|_| |_|\__, |                      #
#                                 ___|_|           _   _                       |___/                       #
#                                |  _ \ ___  _   _| |_(_)_ __   ___ ___                                    #
#                                | |_) / _ \| | | | __| | '_ \ / _ / __|                                   #
#                                |  _ | (_) | |_| | |_| | | | |  __\__ \                                   #
#                                |_| \_\___/ \__,_|\__|_|_| |_|\___|___/                                   #
#                                                                                                          #
# ........................................................................................................ #


#---------------------------------------------------------------------
# vectorized Haversine formulation to count nbr of closest locations
#  --- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.haversine_distances.html
#---------------------------------------------------------------------
#
def countaux(df, dfAux, colname, dist):

    R          = 6371.01                                                 # radius of Earth in km
    c0         = np.radians( list(df.iloc[0][['lat','lng']]) )           # need just the first coords because grouped lat/lng 
    c1         = np.radians( dfAux[['lat','lng']].values.tolist() )
    delta      = c1 - c0
    sineDelLat = (np.sin(delta/2)**2)[:,0]
    sineDelLng = (np.sin(delta/2)**2)[:,1]
    cosineLats = (np.cos(c0)*np.cos(c1))[:,0]
    a = sineDelLat + cosineLats * sineDelLng
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    df[colname] = int(np.sum(R * c < dist))
    return df
# 
#---------------------------------------------------------------------

#=======================================================================================================
# preprocessing step: add count of nearby locations
#=======================================================================================================
#
def preproc_auxilliary(df):

    #.................................................................................
    # load auxillary data files
    #.................................................................................
    df_comm_centre  = pd.read_csv(os.path.join(auxpath,'sg-commerical-centres.csv'))
    df_market       = pd.read_csv(os.path.join(auxpath,'sg-gov-markets-hawker-centres.csv'))
    df_pri_sch      = pd.read_csv(os.path.join(auxpath,'sg-primary-schools.csv'))
    df_sec_sch      = pd.read_csv(os.path.join(auxpath,'sg-secondary-schools.csv'))
    df_mall         = pd.read_csv(os.path.join(auxpath,'sg-shopping-malls.csv'))
    df_mrt          = pd.read_csv(os.path.join(auxpath,'sg-train-stations.csv'))
    #.................................................................................
    # customised distance/range as each there are differences in 
    # expectations of 'nearness' for different locations
    #.................................................................................
    coord = ['lat','lng']
    df = df.groupby(coord).apply(countaux, dfAux=df_comm_centre, colname='num_comm_centre'  ,dist=5.0)
    df = df.groupby(coord).apply(countaux, dfAux=df_market     , colname='num_market_hawker',dist=0.4)
    df = df.groupby(coord).apply(countaux, dfAux=df_pri_sch    , colname='num_pri_sch'      ,dist=1.0)
    df = df.groupby(coord).apply(countaux, dfAux=df_sec_sch    , colname='num_sec_sch'      ,dist=1.0)
    df = df.groupby(coord).apply(countaux, dfAux=df_mall       , colname='num_mall'         ,dist=2.0)
    df = df.groupby(coord).apply(countaux, dfAux=df_mrt        , colname='num_mrt'          ,dist=0.4)
    df = df.reset_index(drop=True)
    return df
#
#=======================================================================================================

#=======================================================================================================
# categorizer to convert categoricals to ordinals (order is important)
#=======================================================================================================
#
def categorize(df,col,order=None,keep_col=False):
    
    df1 = df.copy()

    if not order: order = [pd.Categorical(df1[col]).categories]
    enc = OrdinalEncoder(categories=order)
    enc.fit( df1[[col]] )
    categorized = enc.transform(df1[[col]]).astype(int)
    if keep_col: df1[col+'_1'] = categorized
    else: df1[col] = categorized
        
    return df1
#
#=======================================================================================================

#=======================================================================================================
# preprocessing step: convert categoricals to ordinals (order is important)
#=======================================================================================================
#
def preproc_categoriesll(df,col_list):
    
    df = df.copy()
    
    segment_order = [['ocr', 'rcr', 'ccr']]
    region_order  = [['central region','east region','west region','north-east region','north region']]

    if 'market_segment'      in col_list: df = categorize(df,'market_segment',segment_order)
    if 'region'              in col_list: df = categorize(df,'region',region_order)
    if 'planning_area'       in col_list: df = categorize(df,'planning_area')
    if 'subszone'            in col_list: df = categorize(df,'subszone')
    if 'name'                in col_list: df = categorize(df,'name')
    if 'priceband'           in col_list: df = categorize(df,'priceband')
    if 'sqftband'            in col_list: df = categorize(df,'sqftband')
    if 'pricesqftband'       in col_list: df = categorize(df,'pricesqftband')
    if 'pricesqftband_pred'  in col_list: df = categorize(df,'pricesqftband_pred')
    
    df = df[col_list]
    
    return df
#
#=======================================================================================================

#=======================================================================================================
# preprocessing step: createlinear regression dummies 
#=======================================================================================================
def preproc_lrvariables(df):

    df   = pd.concat([pd.get_dummies(df, columns=['market_segment']), df['market_segment'] ], axis=1)
    df   = pd.get_dummies(df, columns=['pricesqftband'])
    df   = pd.get_dummies(df, columns=['sqftband'])
    
    #.................................................................................
    # vars tested below to be not influential singly possibly due to these 
    # information being encoded in the clusters, sq ft price bands and segment.
    #.................................................................................
    # df = pd.get_dummies(df, columns=['model'])
    # df = pd.get_dummies(df, columns=['district'])
    # df = pd.get_dummies(df, columns=['planning_area'])
    # df = pd.get_dummies(df, columns=['tenure'])
    #.................................................................................
    return df
#
#=======================================================================================================

#=======================================================================================================
# display linear regression results
#=======================================================================================================
def display_lr_parameters(lr_results,overallRMSE):
    
    # print overall RMSE
    print('Overall RMSE : {:>8,.0f}\n'.format(np.round(overallRMSE,1)))
    
    results = []
    fmt0    = '{:>30} : '
    fmt1    = '{:>30} : '
    for idx, (regressor, rmse) in enumerate(lr_results): 
        results.append([idx+1,
                        rmse,
                        regressor.intercept_.item(),
                        *list(regressor.coef_.squeeze())
                       ])
        fmt0 += '{:>10,.0f}'
        fmt1 += '{:>10,.1f}'
    results.insert(0,['Regressor',
                      'RSME',
                      'intercept',
                      *list(regressor.feature_names_in_.squeeze())
                     ])

    for idx, line in enumerate(zip(*results)):
        fmt = fmt0 if idx<=1 else fmt1
        print(fmt.format(*line))
        if idx==1: print('')
#
#=======================================================================================================

#=======================================================================================================
# creates mapping dataframe for psf bands probability 
# - calculates the probability distribution from transactions in the row indices
# - for rows that shows zero probability, fill will mean of [groupby] column
# returns df with rows summing to probability of 1.0
#=======================================================================================================
#
def create_psfbprob_mapping( df_eda,colname,rows,cols,groupby,val):
    
    # length of [rows] indices
    k = len( rows )
    
    # create the pivot, and put index as a col
    df = pd.pivot_table(df_eda,index=rows,columns=cols,values=val ,aggfunc='count')
    df = pd.DataFrame(df.to_records()).reset_index().drop(['index'],axis=1)
    
    # compute probability distribution across cols
    rowsum = df.sum(axis=1,numeric_only=True)
    rowsum += ( rowsum==0 )                                                     # ensure no div_0
    df['rowsum'] = df.sum(axis=1,numeric_only=True)
    
    # rename columns to drop the [val] prefix
    colnames = df.columns[k:].to_list()
    r = re.compile('(\([\d.]+, [\d.]+)')                                        # col name matching pattern
    names = [colname+m[0]+']' for s in colnames if (m:=re.findall('(\([\d.]+, [\d.]+)', s))]
    df.rename(columns=dict(zip(colnames,names)), inplace=True)
    df = pd.concat([df.iloc[:,:k],df.iloc[:,k:].div(rowsum, axis=0)],axis=1)

    # compute groupby mean and insert to rows that sums to zero (probability cannot be zero)
    df_grpmean = df[df['rowsum']!=0].groupby(groupby).mean(numeric_only=True).reset_index()
    df_mean    = df.copy()
    df_mean    = df.merge(df_grpmean,on=groupby,suffixes=['_x',''])
    df_mean    = df_mean[df_mean.columns[~df_mean.columns.str.endswith('_x')]]
    filter1    = (df.rowsum==0)
    df.loc[filter1,names] = df_mean[filter1]
    df.drop(columns=['rowsum'],inplace=True)
    
    return df
    
#=======================================================================================================
# fill probability distribution of psfbands by zones
# - first creates the zone mapping dataframe
# - filters for NaN's in the first mapped cols/merge cols
# - populate by locating
# returns df with ALL rows summing to probability of 1.0
#=======================================================================================================
#
def fillprob_by_zone(df,df_orig,colname,zones,other_index):

    df = df.copy()
    
    cols     = ['pricesqftband']
    val      = ['listing_id']
            
    for zone in zones:
        rows                  = [zone,other_index]
        df_map_zone           = create_psfbprob_mapping( df_orig,colname,rows,cols,[zone],val )
        names                 = list(df_map_zone.columns[df_map_zone.columns.str.startswith(colname)])
        df_prob               = df[rows].copy()
        df_prob               = df_prob.merge(df_map_zone,how='left',on=rows)
        filter1               = (df[names[0]].isna())
        df.loc[filter1,names] = df_prob[filter1]
    
    return df

# .................................................................................................... #
#                                                                                                      #
#       ____               _ _      _   _               ____             _   _                         #
#      |  _ \ _ __ ___  __| (_) ___| |_(_) ___  _ __   |  _ \ ___  _   _| |_(_)_ __   ___ ___          #
#      | |_) | '__/ _ \/ _` | |/ __| __| |/ _ \| '_ \  | |_) / _ \| | | | __| | '_ \ / _ / __|         #
#      |  __/| | |  __| (_| | | (__| |_| | (_) | | | | |  _ | (_) | |_| | |_| | | | |  __\__ \         #
#      |_|   |_|  \___|\__,_|_|\___|\__|_|\___/|_| |_| |_| \_\___/ \__,_|\__|_|_| |_|\___|___/         #
#                                                                                                      #
#                                                                                                      #
# .................................................................................................... #


#=======================================================================================================
# full dataset splitting using (train,test) set
#=======================================================================================================
#
#    X can be a list of datasets to train, y are corresponding response datsets
#    1. dataset = tuple (train, test) 
#    2. splits  = list of dataset splitting filters: tuples of pd.Series of boolean matching
#                 the input dataset, i.e. must be generated from the dataset.
#    3. predictors - a list of predictors names (string)
#    4. response variable - response variable name; must be a column in the dataset set
#    5. rscaler  - variable/column name of data to scale response variable after regression
#
#    Returns list of split datasets. Each items in the list is a tuple of containing
#            ( X_train, y_train, X_test, y_test, rscaler )
#            list will directly feed into  mutliple regression function()
#
def x_split( dataset, splits, predictors, response, rscaler ):

    split_list = []
    train, test = dataset

    if not splits: 
        splits = [ ( pd.Series([True] * len(train), dtype=bool, index=train.index),
                     pd.Series([True] * len(test) , dtype=bool, index=test.index ) ) ]
        
    for split in splits:

        train_split, test_split = split

        data_split = []
        data_split.append( train.loc[train_split,predictors] )           # X-train
        data_split.append( train.loc[train_split,[response]] )           # y_train
        data_split.append(  test.loc[test_split ,predictors] )           # X_test
        data_split.append(  test.loc[test_split ,[response]] )           # y_test
        data_split.append(  test.loc[test_split ,[rscaler] ] )           # scaler

        split_list.append( tuple(data_split) )
            
    return split_list

#=======================================================================================================
# multiple regressor
#=======================================================================================================
#
#   split_dataset   : object returned from x-split method
#   regressor_class : scikit-learn regressor class
#   reg_args        : keyword arguments for regressor_class
#
#   Returnns tuple of regression results list, and overall RMSE
#
def multiple_regressor( split_dataset, regressor_class, **reg_args ):

    results = []
    y_test_stacked = np.array([],dtype=np.float64)              # stack up actual and predictions
    y_pred_stacked = np.array([],dtype=np.float64)              # for overall RMSE computation
    
    # for each sub/split-dataset, generate one linear regressor
    for X_train, y_train, X_test,  y_test, rscaler in split_dataset:
        
        regressor = regressor_class(**reg_args)
        regressor.fit(X_train,y_train.values.ravel())
        
        y_pred = regressor.predict(X_test)

        y_test_ = y_test.squeeze() * rscaler.squeeze()
        y_pred_ = y_pred.squeeze() * rscaler.squeeze()
        RMSE = np.sqrt(mean_squared_error(y_test_, y_pred_))
        
        y_test_stacked = np.concatenate( (y_test_stacked, y_test_), axis=0 )
        y_pred_stacked = np.concatenate( (y_pred_stacked, y_pred_), axis=0 )
        
        results.append( (regressor,RMSE) )

    RMSE_overall = np.sqrt(mean_squared_error(y_test_stacked, y_pred_stacked))
    
    return results, RMSE_overall


#=======================================================================================================
# multiple prediction
#=======================================================================================================
#
#   regressors : regression results part returned by multiple_regressor (exclude overall RMSE returned)
#   df         : validation datset dataframe cleaned and preprocessed
#   rscaler    : the rescaling factor column
#   df_save    : optional df to write results to (for checking purposes)
#
#
def multi_prediction( regressors, dataset, splits, rscaler, df_save=None, prefix=''):
    
    # initialize results dataframe with correct length and index
    results = pd.DataFrame(dataset[rscaler],index=dataset.index)

    # predict price per square foot
    for split, (regressor,_) in zip(splits,regressors):
        test = dataset[list(regressor.feature_names_in_)]
        results.loc[split,'pred_pricesqft'] = regressor.predict(test[split]).squeeze()  

    # get final property price prediction accouting for square footage
    results['Predicted'] = (results.pred_pricesqft) * (results[rscaler])

    # save copy for checking
    if df_save is not None:
        df_save[prefix+'_rscaler']        = results[rscaler]
        df_save[prefix+'_pricesqft_pred'] = results.pred_pricesqft
        df_save[prefix+'_price_pred']     = results.Predicted
        save_df(df_save,outpath=outputpath,hops=2)

    return results[['Predicted']]



# .................................................................................................... #
#                                                                                                      #
#           ____                 _     _             ____             _   _                            #
#          / ___|_ __ __ _ _ __ | |__ (_) ___ ___   |  _ \ ___  _   _| |_(_)_ __   ___ ___             #
#         | |  _| '__/ _` | '_ \| '_ \| |/ __/ __|  | |_) / _ \| | | | __| | '_ \ / _ / __|            #
#         | |_| | | | (_| | |_) | | | | | (__\__ \  |  _ | (_) | |_| | |_| | | | |  __\__ \            #
#          \____|_|  \__,_| .__/|_| |_|_|\___|___/  |_| \_\___/ \__,_|\__|_|_| |_|\___|___/            #
#                         |_|                                                                          #
#                                                                                                      #
#                                                                                                      #
# .................................................................................................... #

#---------------------------------------------------------------------
# Frequency and boxplot by custom bins 
# Line plots (supports 2 y axes)
# data correlation plot using seaborn
#---------------------------------------------------------------------
#
def freq_plot(data,bins,title,xlabel,ylabel,barcolor):

    hist, bin_edges = np.histogram(data,bins)             # make the histogram
    fig,ax = plt.subplots()
    ax.set_title(title)
    pps = ax.bar(range(len(hist)),hist,width=1,color=barcolor,edgecolor='black' ) 
    for p in pps:
        height = p.get_height()
        ax.annotate('{}'.format(height),
                    xy=(p.get_x()+p.get_width()/2, height),
                    xytext=(0, 10),                                   # 10 points above the top
                    textcoords="offset points",
                    ha='center', va='top')
    ax.set_xticks([i for i,j in enumerate(hist)])
    ax.set_xticklabels(['{} - {}'.format(bins[i],bins[i+1]) for i,j in enumerate(hist)])
    plt.ylim([0, hist.max()*1.1])
    plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=True)  
    ax.set_xlabel(xlabel,fontsize=8)
    ax.set_ylabel(ylabel,fontsize=8)
    ax = plt.tick_params(axis="x", labelsize=8)
    plt.tight_layout()
    plt.show()

def boxplot_bin(df, col, scale, col_bin, col_cut, bins, title, xlab, ylab):
    
    df = df.copy()
    df[col_cut] = pd.cut(df[col_bin]*scale, bins=bins)
    ax = df.boxplot(column=col, by=col_cut, grid=False)
    plt.suptitle('')
    ax.set_title(title)
    ax.set_xlabel(xlab,fontsize=12)
    ax.set_ylabel(ylab,fontsize=12)
    ax = plt.tick_params(axis="x", labelsize=8)
    ax = plt.tight_layout()
    ax = plt.show()
    df = None
    
def line_plot(data,xlab,ylab,lw,data2=None,y2lab=None,lw2=None):
    # color specs - https://xkcd.com/color/rgb/

    plt.figure()
    x1,y1 = data
    plt.plot(x1, y1, marker='o', lw=lw, color='xkcd:deep rose')
    if data2: 
        x2,y2 = data2
        if len(x1)==len(x2): 
            ax = plt.gca()    # Get current axis
            ax2 = ax.twinx()  # make twin axis based on x
            ax2.set_ylabel(y2lab)
            ax2.plot(x1, y2, marker='+', lw=lw2 if lw2 else 2, color='xkcd:bright blue')
    plt.xlabel(xlab, fontsize=12)
    plt.ylabel(ylab, fontsize=12)
    plt.tick_params(axis="x", labelsize=10)
    plt.tick_params(axis="y", labelsize=10)
    plt.tight_layout()
    plt.show()
    
def plot_correlation(df):
    df = df.corr()
    sns.set_style(style = 'white')
    sns.set(font_scale = 0.5)
    f, ax = plt.subplots(figsize=(6, 5))
    ax = plt.tick_params(axis="x", labelsize=6)
    plt.tight_layout()
    sns.heatmap(df, cmap='magma', square=True, linewidth=.5, cbar_kws={"shrink": 0.95 }, ax=ax)
    rcdefaults()                # reset to matplotlib defaults after plotting
    
#---------------------------------------------------------------------
# Topographical map ploting (lat/lng)
#---------------------------------------------------------------------
def plot_scatter_map(data,title,cbarlab):
    
    df, gmap, lat, lng, cbar, cmap = data
    fig, ax = plt.subplots(figsize=(8,4))
    ax.tick_params(axis='both', which='major', labelsize=8)
    gmap.plot(color="lightgrey", ax=ax)
    df.plot(x=lng, y=lat,kind='scatter',xlabel='longitude',ylabel='latitude',
            s=5,c=cbar,colormap=cmap, title=title, ax=ax)
    cax = fig.get_axes()[1]
    cax.set_ylabel(cbarlab,fontsize=8)
    cax.tick_params(labelsize=8)

#---------------------------------------------------------------------
# Kmeans cluster plotting methods
# - 2D method adapted from CS5228 lecture notebook
#---------------------------------------------------------------------
def plot_clusters(kmeans, data):

    plt.figure()
 
    # Plot all the data points a color-code them w.r.t. to their cluster label/id
    plt.scatter(data[:, 1], data[:, 2], c=kmeans.labels_, s=50, cmap=plt.cm.tab10)
    
    # Plot the cluster centroids as fat plus signs
    for cluster_id, centroid in enumerate(kmeans.cluster_centers_):
        plt.scatter(centroid[0], centroid[1], marker='+', color='k', s=250, lw=5)

    plt.tight_layout()
    plt.show()
    
# Kmeans cluster 3D plotting methods
def plot_clusters_3d(kmeans, data):

    ax = plt.figure()
    ax = plt.axes(projection='3d')
    xdata = data[:,2]
    ydata = data[:,1]
    zdata = data[:,0]
    ax.scatter3D(xdata, ydata, zdata, c=kmeans.labels_, cmap=plt.cm.tab10);
    plt.tight_layout()
    plt.show()
# 
#---------------------------------------------------------------------