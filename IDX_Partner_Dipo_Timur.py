


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns



# Proses ini merupakan eksplorasi awal untuk memahami struktur, kualitas, dan pola dalam data. Ini mencakup identifikasi missing values, outlier, distribusi variabel


#melakukan load pada datset
url = 'loan_data_2007_2014 (2).csv'
df = pd.read_csv('loan_data_2007_2014 (2).csv')
df.head()



#membuat salinan data untuk dimanipulasi tanpa mengubah data aslinyaaa
df_copy = df.copy()


df_copy.dtypes

#statistika deskriptif
df_copy.describe()



#melihat jumlah missing value pada datasets
missing_data = df_copy.isnull()
for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print("")




df_copy.columns



#data yang memiliki missing value lebih dari 40 persen akan terhapus
threshold = 0.4
df_clean = df_copy.dropna( axis = 1, thresh=(len(df_copy)*(1-threshold)))

#sebelum dan sesudah di drop
print(df_copy.shape)
print(df_clean.shape)



df_clean.columns


#Melihat data yang hilang setelah di threshold
missing_data = pd.DataFrame({
    'Data Type': df_clean.dtypes,
    'Missing Value': df_clean.isnull().sum(),
    'Percentage missing': df_clean.isnull().sum() / df_clean.shape[0] * 100 })
print(missing_data)




#mencari nilai duplikat 
df_clean.duplicated()




#Impiutasi missing value
numeric_value = df_clean.select_dtypes(include = ['float64','int64']).columns
categoric_value = df_clean.select_dtypes(include = ['object']).columns

#numeric value
for col in numeric_value : 
    if df_clean[col].isnull().sum() > 0:
        median = df_clean[col].median()
        df_clean[col].fillna(median,inplace = True)

#categoric
for col in categoric_value : 
    if df_clean[col].isnull().sum() > 0:
        mode = df_clean[col].mode()[0]
        df_clean[col].fillna(mode, inplace = True)



#mengecek kembali datanya
df_clean.isnull().sum()



#mendefinisikan variabel dependennya 
df_clean.loan_status.value_counts()


goodstat_loan = ['Current','Fully Paid','Does not meet the credit policy. Status:Fully Paid']
badstat_loan = ['Charged Off','Late (31-120 days)','In Grace Period','Late (16-30 days)','Default','Does not meet the credit policy. Status:Charged Off']

df_clean['loan_stat_category'] = df_clean['loan_status'].apply(lambda x: 'good loan' if x in goodstat_loan else 'bad loan' )


df_clean.loan_stat_category.value_counts()



#bar plot untuk melihat distribusi
df_clean['loan_stat_category'].value_counts().plot(kind='bar', color=['#40E0D0','red'])
plt.title('distribusi status pinjaman')
plt.xlabel('status')
plt.ylabel('jumlah pinjaman')
plt.show()




df_clean.columns
for column in df_clean.columns:
    value = df_clean[column].value_counts()
    print('Value Count',column,'sebesar',value)
    print ('')



drop_column = ['Unnamed: 0','id','member_id','emp_title', 'loan_status', 'zip_code','title','funded_amnt','funded_amnt_inv','issue_d',
    'pymnt_plan','url','earliest_cr_line','out_prncp','out_prncp_inv','total_pymnt','total_pymnt_inv','total_rec_prncp',
    'total_rec_int','total_rec_late_fee',
    'recoveries','collection_recovery_fee','last_pymnt_d','last_pymnt_amnt',
    'last_credit_pull_d','policy_code']

df_clean.drop(columns=drop_column, axis=1, inplace=True)



df_clean.head()
df_clean.columns


df_clean.dtypes



#mengecek multikolinearitas / korelasi

#numeric value
numerical_value = df_clean.select_dtypes(include=['int64', 'float64'])


corelation_matrix = numerical_value.corr()

#heatmap
plt.figure(figsize =(12,10))
sns.heatmap(corelation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True, linewidths=0.5)
plt.title('Heatmap Correlation')
plt.show()


df_clean.select_dtypes(include='object').columns


drop_column2 = ['installment','total_acc','total_rev_hi_lim','addr_state','sub_grade','dti']
df_clean.drop(columns=drop_column2, axis=1, inplace=True)

#heatmap setelah di drop ke 2 kali
numerical_value = df_clean.select_dtypes(include=['int64', 'float64'])


corelation_matrix = numerical_value.corr()

#heatmap
plt.figure(figsize =(12,10))
sns.heatmap(corelation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True, linewidths=0.5)
plt.title('Heatmap Correlation')
plt.show()


df_clean.dtypes


# 1. TERM (ubah menjadi numerik)
df_clean['term'] = df_clean['term'].str.extract('(\d+)').astype(float)

# 2. megubah nilai grade (A-G -> 1-7)
grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
df_clean['grade'] = df_clean['grade'].map(grade_map)

#(mapping tahun kerja)
emp_length_map = {
    '10+ years': 10,
    '9 years': 9,
    '8 years': 8,
    '7 years': 7,
    '6 years': 6,
    '5 years': 5,
    '4 years': 4,
    '3 years': 3,
    '2 years': 2,
    '1 year': 1,
    '< 1 year': 0.5
}
df_clean['emp_length'] = df_clean['emp_length'].replace(emp_length_map).astype(float)

# 4. label encoding untuk kolom dengan 2 kategori
label_cols = ['initial_list_status', 'application_type', 'loan_stat_category']
for col in label_cols:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col])

# 5. one hot encoding untuk kolom kategorikal nominal
df_clean = pd.get_dummies(df_clean, columns=['home_ownership', 'verification_status', 'purpose'], drop_first=True,dtype=int)


df_clean.describe()


df_clean.info()


# mendefinisikan variabel dependen dan independen
X = df_clean.drop('loan_stat_category', axis=1)
y = df_clean['loan_stat_category']

# 🔀 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Logistic Regression
logreg = LogisticRegression(max_iter=1000, class_weight='balanced')
logreg.fit(X_train, y_train)
ypred_logreg = logreg.predict(X_test)
print("=== Logistic Regression ===")
print("Akurasi:", accuracy_score(y_test, ypred_logreg))
print(confusion_matrix(y_test, ypred_logreg))
print(classification_report(y_test, ypred_logreg))

# 2. Decision Tree
dectree = DecisionTreeClassifier(random_state=42, class_weight='balanced')
dectree.fit(X_train, y_train)
ypred_tree = dectree.predict(X_test)
print("=== Decision Tree ===")
print("Akurasi:", accuracy_score(y_test, ypred_tree))
print(confusion_matrix(y_test, ypred_tree))
print(classification_report(y_test, ypred_tree))

# 3. Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
ypred_rf = rf.predict(X_test)
print("=== Random Forest ===")
print("Akurasi:", accuracy_score(y_test, ypred_rf))
print(confusion_matrix(y_test, ypred_rf))
print(classification_report(y_test, ypred_rf))



# membuat perbandingan akurasi setiap model

acc_logreg = accuracy_score(y_test, ypred_logreg)
acc_dectree = accuracy_score(y_test, ypred_tree)
acc_rf = accuracy_score(y_test, ypred_rf)

model_names = ['Logistic Regression', 'Decision Tree', 'Random Forest']
accuracies = [acc_logreg, acc_dectree, acc_rf]

# Buat bar plot
plt.figure(figsize=(10, 5))
plt.bar(model_names, accuracies, color=['palegreen', 'limegreen', 'forestgreen'])
plt.title('Akurasi Setiap Model')
plt.ylabel('Akurasi')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Tampilkan nilai akurasi di atas bar
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.01, f"{acc:.3f}", ha='center', fontsize=10)

plt.tight_layout()
plt.show()

