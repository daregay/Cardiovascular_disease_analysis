# libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import math

def cleaning_df(df, col, value_greater, value_less):
    '''uses the dataframe and column to locate rows that contain certain values and drops those rows'''
    lst= ((df[df[col] > value_greater]).index).tolist()
    df.drop(lst, axis=0, inplace= True)
    lst2= ((df[df[col] < value_less]).index).tolist()
    df.drop(lst2, axis=0, inplace= True)
    return df


def get_percentage(df, col, target_col):
    '''given a dataframe and column name, it returns a new dataframe that groups by the column 
    and percentage of the column values that are in the positive class for the classifier target variable'''
    df2= df.groupby(col).sum()[target_col]*100/df[col].value_counts()
    return df2.reset_index()


def percent_barplot (df, col, xlabel, title, color= 'cmk'): 
    '''returns barplot of percentage of certain catogorical columns that is in target class '''
    label= ['normal','above normal','well above normal']
    fig= plt.figure(figsize=(15,8))
    plt.bar(label, df[col], color= color)
    plt.ylabel('percent', size= 18)
    plt.title(title, size= 22)
    plt.xticks(size= 15)
    plt.yticks(size= 15)
    plt.xlabel(xlabel, size= 18)
    plt.style.use('seaborn')
    return None


def year_count_barplot(df, col, x, hue):
    '''takes dataframe, counts the number of postive target class for all unique values in a column'''
    df[col] = (df['age'] / 365).round().astype('int')
    data1= df.rename(columns={'cardio': 'CVD'})
    data1['CVD'] = data1['CVD'].map({0: 'No', 1: 'Yes'})
    # plot of counts 
    plt.figure(figsize=(18,10))
    sns.countplot(x=x, hue=hue, data = data1, palette="Set2")
    plt.ylabel('Number of people', size= 22)
    plt.title(label= "Count of People Who Have/Don't Have CVD for Each Age Group", size= 30)
    plt.xticks(size= 22)
    # plt.xlim(False)
    plt.yticks(size= 15)
    plt.xlabel('Age', size= 22)
    plt.style.use('seaborn')
    plt.xlim ()
    plt.legend(fontsize= 22, title= 'CVD', loc= 2, title_fontsize= 22)
    return None


def multcolumn_groupby(df,col1, col2 ):
    '''returns two dataframes, one groupedby two columns and counted, the other a new dataframe with col1
    and how many times each of its values accur'''
    percent= df.groupby([col1, col2]).count().reset_index()
    total= df.groupby(col1).count()[col2]
    total_df= pd.DataFrame(total.index, total).reset_index()
    return percent, total_df


def groupby_columns(col1, col2, df, name):
    '''given dataframe, groups by col1 and counts by col2 and  returns new dataframe using the given name'''
    df_= df.groupby('col1').count()[col2]
    name= pd.DataFrame(df_.index, df).reset_index()
    return name


def create_dictionary (df, value_col, key_col):
    '''given a dataframe, value_col and key_col are the colummns that will be used to create a 
    dictionary with key_col as key and value_col as key'''
    dic= {}
    value= df[value_col].tolist()
    key= df[key_col].tolist()
    for k,v in zip (key, value):
        dic[k]= v
    return dic 


def add_list_column(col, dic1, df):
    '''for all values in a col given a dataframe, the function will see if the values are in a dictionary and 
    return a list of all those values'''
    lst= []
    for i in df[col]:
        for k, v in dic1.items():
            if i == k:
                lst.append(v)
    return lst


def add_column(col, dic1, df, col_name, column_perc, column):
    ''' takes the function add_list_column which return a list, adds the returned list to a dataframe
    and creates another list that takes percentage of two columns
    df= dataframe
    col, dic1,df = the column, dictionary and dataframe that will be passed to add_column
    column= the column used to create percentage 
    '''
    df[col_name]= add_list_column(col, dic1, df)
    df[column_perc]= df[column]*100/df[col_name]
    return df


def count_percent_barplot(cardio_):
    "finds the percentage "
    percent_0=  cardio_[cardio_['cardio']== 0]
    percent0_lst= percent_0['total_percent'].tolist()
    percent_1= cardio_[cardio_['cardio'] == 1]
    percent1_lst= percent_1['total_percent'].tolist()
    percent1_lst.insert(0,0)
    indices = percent_0['years']
    #plots the percentage
    width = np.min(np.diff(indices))/3
    fig = plt.figure(figsize= (20,12))
    ax = fig.add_subplot(111)
    ax.bar(indices-width,percent0_lst,width,color='b',label='no CVD')
    ax.bar(indices,percent1_lst,width,color='r',label='CVD')
    ax.set_xlabel('Age', size= 30)
    ax.set_ylabel('Percent', size = 30)
    ax.set_title("Percent of People Who have/ don't have CVD for each Age Group", size=30)
    ax.legend(fontsize= 22)
    ax.tick_params(axis='both', which='major', labelsize=22)
    return None


def lifestyle_df(df, rename_column, column, rename_values):
    ''' it renames a columns in the dataframe=df, given dictionary called rename_column,
    it also changes the catergorical value names of another column given dictionary called rename_values
    returns dataframe '''
    data2= df.rename(columns=rename_column)
    data2['CVD'] = data2['CVD'].map({0: 'No', 1: 'Yes'})
    data2[column]= data2[column].map(rename_values)
    return data2


def lifestyle_barplot(data, x, hue, xlabel, title):
    '''plots a barplot given data as a dataframe, x= the column to plot
    hue= color encoding column'''
    plt.figure(figsize=(18,8))
    sns.countplot(x=x, hue=hue, data=data, palette="Set2")
    plt.ylabel('Number of people', size= 30)
    plt.title(label= title, size= 30)
    plt.xticks(size= 22)
    # plt.xlim(False)
    plt.yticks(size= 15)
    plt.xlabel(xlabel, size= 22)
    plt.style.use('seaborn')
    plt.xlim ()
    plt.legend(fontsize= 22, title= 'CVD', loc= 1, title_fontsize= 22)
    return None


def bmi_barplot(df):
    '''returns a counted barplot of the BMI column'''
    data = df.rename(columns={'cardio': 'CVD'})
    data['CVD'] = data['CVD'].map({0: 'No', 1: 'Yes'})
    plt.figure(figsize=(20,14))
    plot_= sns.countplot(x='BMI', hue='CVD', data = data, palette="Set2")
    plt.ylabel('Number of People', size= 22)
    plt.title(label= 'Count of people who have CVD for each Body Mass Index', size= 30)
    plt.xticks(size= 22)
    plt.xlim()
    plt.yticks(size= 15)
    plt.xlabel('Body Mass Index (BMI range - kg/m2)', size= 22)
    plt.style.use('seaborn')
    plt.xlim ()
    plt.legend(fontsize= 22, title= 'CVD', loc= 1, title_fontsize= 22)
    #fig.canvas.draw()
    new_ticks = [i.get_text() for i in plot_.get_xticklabels()]
    plt.xticks(range(0, len(new_ticks), 10), new_ticks[::10])
    return None


def count_barplot(df, col,title, xlabel):
    '''returns a counted barplot of the col values that are in target class'''
    data = df.rename(columns={'cardio': 'CVD'})
    data['CVD'] = data['CVD'].map({0: 'No', 1: 'Yes'})
    # make countplot
    plt.figure(figsize=(20,14))
    plot_= sns.countplot(x=col, hue='CVD', data = data, palette="Set2")
    plt.ylabel('Number of People', size= 22)
    plt.title(label= title, size= 30)
    plt.xticks(size= 22)
    plt.xlim()
    plt.yticks(size= 15)
    plt.xlabel(xlabel, size= 22)
    plt.style.use('seaborn')
    plt.xlim()
    plt.legend(fontsize= 22, title= 'CVD', loc= 1, title_fontsize= 22)
    #fig.canvas.draw()
    new_ticks = [i.get_text() for i in plot_.get_xticklabels()]
    plt.xticks(range(0, len(new_ticks), 10), new_ticks[::10])
    return None


def create_dummies(df, column):
    '''creates dummies variables using the column and adds them to the dataframe
     delets the original column'''
    dummy= pd.get_dummies(df[column], prefix= column)
    df= df.join(dummy, lsuffix="_left")
    del df[column]
    return df


if __name__ == "__main__":
    # import dataframe 
    unclean_df= pd.read_csv('data/cardio_train.csv', ';')
    # clean the dataframe
    clean=cleaning_df(unclean_df, 'ap_lo', 200, 0)
    cvd=cleaning_df(clean, 'ap_hi', 300, 0)
    # EDA
    # take percentage of how many people have cvd for each glucose classifications and make barplot
    gluc_df = get_percentage(cvd, 'gluc', 'cardio')
    percent_barplot(gluc_df, 0, 'Glucose classification', 'Percent of People Who Have CVD for Each Glucose Classification')
    # take percentage of how many people have cvd for each cholestrol classifications and make barplot
    chole_df= get_percentage(cvd, 'cholesterol', 'cardio')
    percent_barplot(chole_df, 0, 'Cholesterol Classification', 'Percent of People Who Have CVD for Each Cholesterol Classification')
    # barplot of count of cvd as age increases 
    year_count_barplot(cvd, 'years', 'years', 'CVD')
    # makes percentage of how many peopl in each age group have cvd/dont have cvd and plots barplot
    cardio_percent, cardio_total= multcolumn_groupby(cvd,'years', 'cardio' )
    dic= create_dictionary(cardio_total, 'cardio', 'years')
    cardio_ = add_column('years',dic,cardio_percent, 'total_cardio', 'total_percent', 'id')
    count_percent_barplot(cardio_)
    # cleans the column to make a plot and plots cvd amoung number of people who smoke/dont smoke
    smoke= lifestyle_df(cvd, {'cardio': 'CVD'}, 'smoke', {0: "Don't Smoke", 1: 'Smoke'})
    lifestyle_barplot(smoke, 'smoke', 'CVD', "People Who Smoke and Don't Smoke", "Smoking and Cardiovascular Disease (CVD)")
    # cleans the column to make a plot and plots cvd amoung number of people who drink/dont drink alcohol
    alcohol= lifestyle_df(cvd, {'cardio': 'CVD'}, 'alco', {0: "Don't Drink", 1: "Drink"})
    lifestyle_barplot(alcohol, 'alco', 'CVD', 'Alcohol', "Alcohol Consumption and Cardiovascular Disease (CVD)")
    # cleans the column to make a plot and plots cvd amoung number of people who drink/dont drink alcohol
    excercise= lifestyle_df(cvd, {'cardio': 'CVD'}, 'active', {0: 'Not Active', 1: "Active"})
    lifestyle_barplot(excercise, 'active', 'CVD', 'Excercise', "Excercise and Cardiovascular Disease (CVD)")
    # barplot of count of people for each  BMI group who have cvd/ don't have cvd 
    cvd['BMI']= np.round(cvd.weight /((cvd.height/100)**2),1)
    bmi_barplot(cvd)
    # barplot of cvd and diastolic blood pressure and systolic blood pressure 
    count_barplot(cvd,'ap_lo', "Diastolic blood pressure and Number of People who have/ don't have CVD",'Diastolic blood pressure (mmHg)')
    count_barplot(cvd,'ap_hi', "Systolic blood pressure and Number of People who have/ don't have CVD",'Systolic blood pressure (mmHg)')
    # further data engineering for model selection and save the final dataframe 
    cvd['gender']= cvd['gender'].map({1: 0, 2: 1})
    del cvd['height']
    del cvd['weight']
    del cvd ['id']
    del cvd['years']
    cvd= create_dummies(cvd, 'cholesterol')
    cvd= create_dummies(cvd, 'gluc')
    # # cvd.to_csv('final_df.csv')
    plt.show()
    