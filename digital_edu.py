import pandas as pd   
import matplotlib.pyplot as plt

df = pd.read_csv("train.csv")

#temp = df[df['result']==1]['city'].value_counts().head()
#temp.plot(kind = 'pie', label = '')
#plt.show()
#print(df.info())

df.drop(['has_photo', 'life_main', 'people_main', 'city', 'followers_count', 
'last_seen','occupation_name', 'id', 'bdate', 'has_mobile', 'graduation',
'career_start', 'relation', 'career_end'], axis=1, inplace = True)
#1

def sex_apply(sex):
    if sex == 2:
        return 0
    return 1
df['sex'] = df['sex'].apply(sex_apply)
      
#2               

df['education_form'].fillna('Full-time', inplace = True)
df[list(pd.get_dummies(df['education_form']).columns)] = pd.get_dummies(df['education_form'])
df.drop(['education_form'], axis=1, inplace = True)

#3

def edu_status_form(edu_status):
    if edu_status == 'Undergraduate applicant':
        return 0
    elif edu_status == 'Student (Specialist)' or edu_status == "Student (Bachelor's)" or edu_status == "Student (Master's)":
        return 1
    elif edu_status == "Alumnus (Specialist)" or edu_status == "Alumnus (Bachelor's)" or edu_status == "Alumnus (Master's)":
        return 2
    else:
        return 3
df['education_status'] = df['education_status'].apply(edu_status_form)

#4

def langs_apply(langs):
    if langs.find('Русский') != -1 and langs.find('English') != -1:
        return 0
    else:
        return 1
df['langs'] = df['langs'].apply(langs_apply)

#5

df['occupation_type'].fillna('university', inplace = True)
def occupation_type_apply(ocu_type):
    if ocu_type == 'university':
        return 0
    return 1
df['occupation_type'] = df['occupation_type'].apply(occupation_type_apply)
df.info()

#model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

x = df.drop('result', axis = 1)
y = df['result']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.40)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
print(y_test)
print(y_pred)
print('Точность:', round(accuracy_score(y_test, y_pred)*100, 2), '%')