#############################################################
#                 KREDI ONAY MAKINESI
#############################################################

# Konu:
# Kredi skor kartları, finans sektöründe yaygın olarak kullanılan bir
# risk kontrol yöntemidir. Gelecekteki temerrüt ve kredi kartı borçlarının
# olasılığını tahmin etmek için kredi kartı başvuru sahipleri tarafından sunulan
# kişisel bilgileri ve verileri kullanır. Banka, başvuru sahibine kredi kartı
# verilip verilmeyeceğine karar verebilir. Kredi puanları, riskin büyüklüğünü
# objektif olarak ölçebilir.

# Genel olarak konuşursak, kredi puan kartları geçmiş verilere dayanır. Bir
# zamanlar büyük ekonomik dalgalanmalarla karşılaşıldığında. Geçmiş modeller
# orijinal tahmin gücünü kaybedebilir. Lojistik model, kredi puanlaması için
# yaygın olarak kullanılan bir yöntemdir. Çünkü Lojistik ikili sınıflandırma
# görevleri için uygundur ve her özelliğin katsayılarını hesaplayabilir.

# Şu anda, makine öğrenmesi algoritmalarının geliştirilmesiyle; Boosting,
# Random Forests ve Support Vector Machines gibi daha tahmine dayalı yöntemler,
# kredi kartı puanlamasına dahil edilmiştir. Bununla birlikte, bu yöntemler
# genellikle iyi bir şeffaflığa sahip değildir. Müşterilere ve düzenleyicilere
# ret veya kabul için bir neden sağlamak zor olabilir.

# Görev:
# Bir başvuru sahibinin 'iyi' veya 'kötü' bir müşteri olup olmadığını tahmin
# etmek için bir makine öğrenimi modeli oluşturun, diğer görevlerden farklı
# olarak 'iyi' veya 'kötü' tanımı verilmez (0, 1 kullanılır). Etiketinizi
# oluşturmak için vintage analizi gibi bazı teknikler kullanmalısınız.
# Ayrıca, dengesiz veri sorunu bu görevde büyük bir sorundur.


# application_record değişkenleri:

# ID - Müşteri numarası
# CODE_GENDER - Cinsiyet
# FLAG_OWN_CAR - Aracı olup-olmaması
# FLAG_OWN_REALTY - Taşınmaz varlığı olup-olmaması
# CNT_CHILDREN - Çocuk sayısı
# AMT_INCOME_TOTAL - Yıllık geliri
# NAME_INCOME_TYPE - Gelir kategorisi
# NAME_EDUCATION_TYPE - Eğitim seviyesi
# NAME_FAMILY_STATUS - Medeni hali
# NAME_HOUSING_TYPE	- Yaşam alanı
# DAYS_BIRTH - Doğum günü (geriye doğru bugünden (0) sayılır, -1 dün demek)
# DAYS_EMPLOYED	İşe başlama tarihi (geriye doğru bugünden (0) sayıslır; pozitif ise kişinin işsiz olduğu anlamına gelir)
# FLAG_MOBIL - Cep telefonu olup-olmaması
# FLAG_WORK_PHONE - İş telefon hattı olup olmaması
# FLAG_PHONE - Ev telefonu olup-olmaması
# FLAG_EMAIL - Email adresi var mı
# OCCUPATION_TYPE - Mesleği
# CNT_FAM_MEMBERS - Aile büyüklüğü


# credit_record.csv değişkenleri:

# ID - Müşteri numarası

# MONTHS_BALANCE - Kayıt ayı - Müşteri verilerinin ilk kabul edildiği ay tarihi
#                              Geriye doğru hesaplanır. 0 bulunduğumuz ay, -1 geçen ay,
#                              -2 de 2 ay öncesi...

# STATUS Statü  - 0: 1-29 gün geçti
#                 1: vadesi 30-59 gün geçti
#                 2: 60-89 gün gecikmiş
#                 3: 90-119 gün gecikti
#                 4: 120-149 gün gecikti
#                 5: Vadesi geçmiş veya şüpheli borçlar, 150 günden fazla iptaller
#                 C: o ay ödedi
#                 X: Ay için kredi yok



import helpers.angels
from helpers.angels import *
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder,MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df_credit = pd.read_csv("datasets/credit_record.csv")
df_app = pd.read_csv("datasets/application_record.csv")
df_credit.head(20)
def target_df(dataframe1,dataframe2):
    # Credit df de bulunan Status değişkeninin sınıfları incelenir
    # ve müşterilerin davranışlarının örüntü barındırıp barındırmadığı incelenir
    #########################################################################
    credit_grouped = pd.get_dummies(data=dataframe1, columns=['STATUS'], prefix='', prefix_sep=''
                                    ).groupby('ID')[sorted(dataframe1['STATUS'].unique().tolist())].sum()
    overall_pastdue = ['0', '1', '2', '3', '4', '5']
    credit_grouped['number_of_months'] = dataframe1.groupby('ID')['MONTHS_BALANCE'].count()
    credit_grouped['over_90'] = credit_grouped[['3', '4', '5']].sum(axis=1)
    credit_grouped['less_90'] = credit_grouped[['0', '1', '2']].sum(axis=1)
    credit_grouped['overall_pastdue'] = credit_grouped[overall_pastdue].sum(axis=1)
    credit_grouped['paid_pastdue_diff'] = credit_grouped['C'] - credit_grouped['overall_pastdue']
    for col in credit_grouped.columns:
        credit_grouped[col] = credit_grouped[col].astype(int)
    # incelemelere göre target oluşturma işlemi
    # (3 aydan az geciktirme sayısı 12 den az) ve (3 aydan az geciktirme sayısı, 3 aydan fazla geciktirme sayısından fazla olan müşteriler 1 sınıfına atanır)
    # kredi kartı borçları tam zamanında ödenen aylar 90 gün den az gecikerek ödenen aylardan fazla olan müşteriler 1 sınıfına atanır
    # 90 gün den az gecikerek ödenen borçların oranı takip edilen tüm aylar içerisinde %70 oranından fazla ise, müşteriler 1 sınıfına atanır
    # kredi kartı borçları tam zamanında ödenen ayların oranı takip edilen tüm aylar içerisinde %70 oranından fazla ve eşit  ise, müşteriler 1 sınıfına atanır
    #######################################################################
    target = []
    for index, row in credit_grouped.iterrows():
        if (((row["less_90"]) < 12) > (row["over_90"])):
            target.append(1)
        elif (row["C"] > row["less_90"]):
            target.append(1)
        elif (((row["less_90"] * 100) / (row["number_of_months"])) > 70):
            target.append(1)
        elif (((row['C'] * 100) / (row["number_of_months"])) >= 70):
            target.append(1)
        else:
            target.append(0)
    credit_grouped['target'] = target
    # credit df de bizim için önemli olan featurelar ile müşteri bilgileri tutulan df birleştirilir
    #########################################################################
    features = ["X", 'number_of_months', 'over_90', 'less_90', 'overall_pastdue', 'paid_pastdue_diff', 'target']
    most_important_features = credit_grouped.loc[:, features]
    most_important_features.reset_index(inplace=True)
    customers_df = pd.merge(dataframe2, most_important_features, on='ID')
    customers_df.index = customers_df['ID']
    customers_df = customers_df.drop('ID', axis=1)
    customers_df["OCCUPATION_TYPE"].replace([np.nan], "No_Occupation_info", inplace=True)
    return customers_df
df_credit.shape
df_app.shape
df_ = target_df(df_credit,df_app)
df = df_.copy()

df.head()





#########################################################################
# KEŞİFÇİ VERİ ANALİZİ / VERİ ÖN İŞLEME
#########################################################################
df.shape # (36457, 24)
# duplicated drop
bool_series = df.duplicated(keep="last")
df = df[~bool_series]
df.shape # (30475, 24)
# cat_cols, num_cols, cat_but_car YAKALANMASI
cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=7, car_th=20)

# Describe Analysis
df.describe(percentiles=[0.5, 0.75, 0.80, 0.90, 0.95, 0.97, 0.98, 0.99])

# Ayrıkırı değerler için değerlendirilmeyecek değişkenlerimiz
spider_df = ['X', 'number_of_months', 'over_90', 'less_90', 'overall_pastdue', 'paid_pastdue_diff']
num_cols = [col for col in num_cols if col not in spider_df]

# Outlier Control
check_outlier(df, num_cols, q1=0.01, q3=0.99)

# Catch Outlier
for col in num_cols: grab_outliers(df, col, q1=0.01, q3=0.99, index=False)

#  Outlier for Process
for col in num_cols: replace_with_thresholds(df, col, q1=0.01, q3=0.99)

check_outlier(df, num_cols, q1=0.01, q3=0.99)
cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=7, car_th=20)


# Örüntü barındırmayan Müşterilerin Drop edilmesi
no_credit_customer = df[df["X"] == df["number_of_months"]]
df.drop(no_credit_customer.index, inplace=True)


# CNT_CHILDREN değişkeni integer olarak atıyoruz
df["CNT_CHILDREN"] = df["CNT_CHILDREN"].astype(int)

##############################################################
#                      FEATURE ENGINEERING
##############################################################

# Müşteri kaç gün çalışmış --->>> Müşteri Kaç yıl çalışmış
df["YEARS_WORKED"] = (df["DAYS_EMPLOYED"]/365)*(-1)
df.loc[df["YEARS_WORKED"] < 0, 'YEARS_WORKED'] = -1

df["YEARS_WORKED"] = df["YEARS_WORKED"].astype(int)
df["CNT_FAM_MEMBERS"] = df["CNT_FAM_MEMBERS"].astype(int)

# Müşteri kaç gün yaşamış --->>> Müşteri Kaç yıl yaşamış
df["AGE"] = (df["DAYS_BIRTH"]/365)*(-1)
df["AGE"] = df["AGE"].astype(int)



# ['FLAG_WORK_PHONE','FLAG_PHONE','FLAG_EMAIL'] ---> kategorize edilmiş hali

df.loc[((df["FLAG_WORK_PHONE"] == 1) & (df["FLAG_MOBIL"] == 1) & (df["FLAG_PHONE"] == 1) & (df["FLAG_EMAIL"] == 1)),
       "COMMUNICATION_TOOL"] = "FULL_COMMUNICATION"

df.loc[((df["FLAG_WORK_PHONE"] == 0) & (df["FLAG_MOBIL"] == 1) & (df["FLAG_PHONE"] == 0) & (df["FLAG_EMAIL"] == 0)),
       "COMMUNICATION_TOOL"] = "LOW_COMMUNICATION"

df["COMMUNICATION_TOOL"].replace([np.nan], "MIDDLE_COMMUNICATION", inplace=True)
df.isnull().sum()
# KAÇ TANE İLETİŞİM ARACI KULLANIYOR VAR OLAN İLETİŞİM ARAÇLARI TOPLANIR AYRI BİR DEĞİŞKEN

df["COMMUNICATION_TOOL_COUNT"] =  df["FLAG_WORK_PHONE"].astype(int) + df["FLAG_PHONE"].astype(int) \
                                            + df["FLAG_EMAIL"].astype(int) + df["FLAG_MOBIL"].astype(int)


# CAT_INCOME

df.loc[df["AMT_INCOME_TOTAL"] <= 50000, "CAT_INCOME"] = "income_less50"
df.loc[(df["AMT_INCOME_TOTAL"] > 50000) & (df["AMT_INCOME_TOTAL"] <= 100000), "CAT_INCOME"] = "income_50_100"
df.loc[(df["AMT_INCOME_TOTAL"] > 100000) & (df["AMT_INCOME_TOTAL"] <= 130000), "CAT_INCOME"] = "income_100_130"
df.loc[(df["AMT_INCOME_TOTAL"] > 130000) & (df["AMT_INCOME_TOTAL"] <= 170000), "CAT_INCOME"] = "income_130_170"
df.loc[(df["AMT_INCOME_TOTAL"] > 170000) & (df["AMT_INCOME_TOTAL"] <= 225000), "CAT_INCOME"] = "income_170_225"
df.loc[(df["AMT_INCOME_TOTAL"] > 225000) & (df["AMT_INCOME_TOTAL"] <= 400000), "CAT_INCOME"] = "income_225_400"
df.loc[df["AMT_INCOME_TOTAL"] > 400000, "CAT_INCOME"] = "income_high400"


# INCOME_PER_FAM_MEMBER - Aile üyesi başına düşen geliri ifade eden değişken

df["INCOME_PER_FAM_MEMBER"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]

# Calisma hayatinin, toplam yasamindaki orani:
df["WORK_MEAN"] = (df["YEARS_WORKED"]*100) / df["AGE"]
df["WORK_MEAN"] = df["WORK_MEAN"].astype(int) # integera ceviriyoruz
df.loc[df["WORK_MEAN"] < 0, 'WORK_MEAN'] = -1 # dead olanlari tanimliyoruz

# Bir basari segmentasyonu:

df["SUCCESS"] = df["AMT_INCOME_TOTAL"] / df["WORK_MEAN"]
df.loc[df["SUCCESS"] < 0, 'SUCCESS'] = -1     # dead olanlar
df.loc[df["SUCCESS"] == np.inf, 'SUCCESS'] = 0  # hayatinda hic calismamis ama yillik geliri olanlar
df["SUCCESS"] = df["SUCCESS"].astype(int)


# Kategorik basari degiskeni:

df.loc[df["SUCCESS"]== -1, "CAT_SUCCESS"] = "dead_person"
df.loc[df["SUCCESS"]== 0, "CAT_SUCCESS"] = "never_worked"
df.loc[(df["SUCCESS"] > 0) & (df["SUCCESS"] <=2200), "CAT_SUCCESS"] = "low_success"
df.loc[(df["SUCCESS"] > 2200) & (df["SUCCESS"] <=8700), "CAT_SUCCESS"] = "mid_success"
df.loc[(df["SUCCESS"] > 8700) & (df["SUCCESS"] <=21000), "CAT_SUCCESS"] = "high_success"
df.loc[(df["SUCCESS"] > 21000) & (df["SUCCESS"] <=50000), "CAT_SUCCESS"] = "very_high_success"
df.loc[(df["SUCCESS"] > 50000), "CAT_SUCCESS"] = "superior_success"


# YAŞA GÖRE KREDİ ALABİLME SKORU Yaşa göre kredi önceliklerdirme

df.loc[df['AGE'] >= 65, 'AGE_PRIVILEDGE'] = 1
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 21), 'AGE_PRIVILEDGE'] = 2
df.loc[(df['AGE'] >= 50) & (df['AGE'] < 65), 'AGE_PRIVILEDGE'] = 3
df.loc[(df['AGE'] >= 21) & (df['AGE'] < 30), 'AGE_PRIVILEDGE'] = 4
df.loc[(df['AGE'] >= 30) & (df['AGE'] < 50), 'AGE_PRIVILEDGE'] = 5


# MESLEK GRUPLARINI INCOME KIRILIMINDA İNDİRGEME & lABEL
### OCCUPATION_TYPE & AMT_INCOME_TOTAL ###

# min 27000.000
# 30% 135000.000
# 50% 157500.000
# 80% 247500.000
# max 1575000.000

# 4 kategoriye ayrılacak
# Low_skill  1
# Staff      2
# Owners     3
# Managers   4


# df.groupby(["OCCUPATION_TYPE"]).agg({"AMT_INCOME_TOTAL": "mean"}).sort_values(by="AMT_INCOME_TOTAL",ascending=False)

df['OCCUPATION_TYPE'] = pd.cut(x=df['AMT_INCOME_TOTAL'],
                               bins=[0,130000, 170000, 230000, df['AMT_INCOME_TOTAL'].max()],
                               labels=["Low_skill","Staff", "Owners", "Managers"])

df.loc[(df['OCCUPATION_TYPE'].isnull() == True, "AMT_INCOME_TOTAL")]
df['OCCUPATION_TYPE'].replace(to_replace=["Low_skill", "Staff", "Owners", "Managers"], value=[1, 2, 3, 4], inplace=True)


# FLOAT DEĞİŞKENLERİ İNT YAPMAK && BINARY SINIF DÖNÜŞÜMÜ
df["AMT_INCOME_TOTAL"] = df["AMT_INCOME_TOTAL"].astype(int)
df["INCOME_PER_FAM_MEMBER"] = df["INCOME_PER_FAM_MEMBER"].astype(int)

df["CODE_GENDER"].replace("M", "1", inplace=True)
df["CODE_GENDER"].replace("F", "0", inplace=True)
df["CODE_GENDER"] = df["CODE_GENDER"].astype(int)

df["FLAG_OWN_CAR"].replace("Y", "1", inplace=True)
df["FLAG_OWN_CAR"].replace("N", "0", inplace=True)
df["FLAG_OWN_CAR"] = df["FLAG_OWN_CAR"].astype(int)

df["FLAG_OWN_REALTY"].replace("Y", "1", inplace=True)
df["FLAG_OWN_REALTY"].replace("N", "0", inplace=True)
df["FLAG_OWN_REALTY"] = df["FLAG_OWN_REALTY"].astype(int)




# Gereksiz değişkenler drop edildi
ll = ["FLAG_MOBIL","DAYS_BIRTH","DAYS_EMPLOYED","FLAG_WORK_PHONE","FLAG_PHONE","FLAG_EMAIL","X","number_of_months","over_90",
      "less_90","overall_pastdue"]
df.drop(ll, inplace = True, axis = 1)
df.drop(["NAME_INCOME_TYPE"] ,axis= 1, inplace=True)
df.head()
df.info()




#########################################################################
# ONE-HOT ENCODING
#########################################################################
cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=7, car_th=20)



cat_cols = [col for col in cat_cols if "target" not in col]
df = helpers.angels.one_hot_encoder(df, cat_cols, drop_first=True)
#########################################################################
# STANDARD
#########################################################################
df.head()
df[num_cols] = StandardScaler().fit_transform(df[num_cols])
#customers_df[num_cols] = pd.DataFrame(X_scaled, columns=customers_df[num_cols].columns)

#########################################################################
#  BASE MODEL
#########################################################################
y = df["target"]
X = df.drop(["target"], axis=1)

def base_models(X, y, scoring=["accuracy","precision","recall","f1","roc_auc"]):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('LightGBM', LGBMClassifier()),
                   ]
    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=10, scoring=scoring)
        print(f"########## {name} ##########")
        print(" ")
        def cv_mean(model_results):
            print(f"Fit Time: {round(model_results['fit_time'].mean(), 4)}")
            print(f"Score Time: {round(model_results['score_time'].mean(), 4)}")
            print(f"Test Accuracy: {round(model_results['test_accuracy'].mean(), 4)}")
            print(f"Test Precision: {round(model_results['test_precision'].mean(), 4)}")
            print(f"Test Recall: {round(model_results['test_recall'].mean(), 4)}")
            print(f"Test F1: {round(model_results['test_f1'].mean(), 4)}")
            print(f"Test ROC AUC: {round(model_results['test_roc_auc'].mean(), 4)}")
        cv_mean(cv_results)
        print(" ")

base_models(X, y)
import sklearn
sklearn.metrics.SCORERS.keys()




#########################################################################
# HYPERPARAMETER OPTIMIZATION
#########################################################################

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 20)}
rf_params = {'max_depth': range(1, 20),
             "max_features": [5, 7, "auto"],
             "min_samples_split": range(1, 20),
             "n_estimators": [200, 300]}
lightgbm_params = {'max_depth': range(1, 20),
                   "min_samples_split": range(1, 20),
                   "learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500,10000],
                   "colsample_bytree": [0.7, 1]
                    }
classifiers = [("CART", DecisionTreeClassifier(), cart_params),
               ("RF", RandomForestClassifier(), rf_params),
               ('LightGBM', LGBMClassifier(), lightgbm_params)]
def hyperparameter_optimization(X, y, cv=10, scoring=["accuracy","precision","recall","f1","roc_auc"]):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(" (Before) ")
        def cv_mean(model_results):
            print(f"Fit Time: {round(model_results['fit_time'].mean(), 4)}")
            print(f"Score Time: {round(model_results['score_time'].mean(), 4)}")
            print(f"Test Accuracy: {round(model_results['test_accuracy'].mean(), 4)}")
            print(f"Test Precision: {round(model_results['test_precision'].mean(), 4)}")
            print(f"Test Recall: {round(model_results['test_recall'].mean(), 4)}")
            print(f"Test F1: {round(model_results['test_f1'].mean(), 4)}")
            print(f"Test ROC AUC: {round(model_results['test_roc_auc'].mean(), 4)}")
        cv_mean(cv_results)
        print(" ")
        print("Applying GridSearchCV....")
        print(" ")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False,).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        def cv_mean(model_results):
            print(f"Fit Time: {round(model_results['fit_time'].mean(), 4)}")
            print(f"Score Time: {round(model_results['score_time'].mean(), 4)}")
            print(f"Test Accuracy: {round(model_results['test_accuracy'].mean(), 4)}")
            print(f"Test Precision: {round(model_results['test_precision'].mean(), 4)}")
            print(f"Test Recall: {round(model_results['test_recall'].mean(), 4)}")
            print(f"Test F1: {round(model_results['test_f1'].mean(), 4)}")
            print(f"Test ROC AUC: {round(model_results['test_roc_auc'].mean(), 4)}")
        cv_mean(cv_results)
        print(" ")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models


best_models = hyperparameter_optimization(X, y)



######################################################
# Stacking & Ensemble Learning
######################################################
def voting_classifier(best_models, X, y):
    print("Voting Classifier...")
    voting_clf = VotingClassifier(estimators=[('CART', best_models["CART"]), ('RF', best_models["RF"]),
                                              ('LightGBM', best_models["LightGBM"])],
                                  voting='soft').fit(X, y)
    cv_results = cross_validate(voting_clf, X, y, cv=10, scoring=["accuracy","precision","recall","f1","roc_auc"])
    def cv_mean(model_results):
        print(f"Fit Time: {round(model_results['fit_time'].mean(), 4)}")
        print(f"Score Time: {round(model_results['score_time'].mean(), 4)}")
        print(f"Test Accuracy: {round(model_results['test_accuracy'].mean(), 4)}")
        print(f"Test Precision: {round(model_results['test_precision'].mean(), 4)}")
        print(f"Test Recall: {round(model_results['test_recall'].mean(), 4)}")
        print(f"Test F1: {round(model_results['test_f1'].mean(), 4)}")
        print(f"Test ROC AUC: {round(model_results['test_roc_auc'].mean(), 4)}")
    cv_mean(cv_results)
    return voting_clf

voting_clf = voting_classifier(best_models, X, y)






# Sadece LightGBM MODEL


# Hold Out Yöntemi
#----------------------------------------------------------------------------------------
# Veri setinin train-test olarak ayrılması:
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20, random_state=17)

# Modelin train setine kurulması:
lgbmH_model = LGBMClassifier().fit(X_train, y_train)

# Test setinin modele sorulması:
y_pred = lgbmH_model.predict(X_test)

# AUC Score için y_prob (1. sınıfa ait olma olasılıkları)
y_prob = lgbmH_model.predict_proba(X_test)[:, 1]



# Confusion Matrix
def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

plot_confusion_matrix(y_test, y_pred)
# Classification report
print(classification_report(y_test, y_pred))

# ROC Curve
plot_roc_curve(lgbmH_model, X_test, y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()

roc_auc_score(y_test, y_prob)

def plot_importance(model, features, save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[:15], color="navy")
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')
plot_importance(lgbmH_model, X_test)
# K-katlı Cv
#----------------------------------------------------------------------
lightgbm_model = LGBMClassifier()

lgbm_params = {"learning_rate": [0.01, 0.1, 0.001],
               "n_estimators": [100, 300, 500, 10000],
               "colsample_bytree": [0.5, 0.7, 1],
               "early_stopping_round":1}

first_cv_results = cross_validate(lightgbm_model, X, y, cv=10, scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

lightgbm_best_grid = GridSearchCV(lightgbm_model, lgbm_params, cv=10, n_jobs=-1, verbose=False).fit(X, y)

lightgbm_final = lightgbm_model.set_params(**lightgbm_best_grid.best_params_).fit(X, y)
final_cv_results = cross_validate(lightgbm_final, X, y, cv=10, scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])


def cv_mean(model_results):
    print(f"Fit Time: {round(model_results['fit_time'].mean(), 4)}")
    print(f"Score Time: {round(model_results['score_time'].mean(), 4)}")
    print(f"Test Accuracy: {round(model_results['test_accuracy'].mean(), 4)}")
    print(f"Test Precision: {round(model_results['test_precision'].mean(), 4)}")
    print(f"Test Recall: {round(model_results['test_recall'].mean(), 4)}")
    print(f"Test F1: {round(model_results['test_f1'].mean(), 4)}")
    print(f"Test ROC AUC: {round(model_results['test_roc_auc'].mean(), 4)}")
cv_mean(first_cv_results)
cv_mean(final_cv_results)


plot_importance(lightgbm_model, X)
plot_importance(lightgbm_final, X)
######################################################
# 6. Prediction for a New Observation
######################################################
X.columns
random_user = X.sample(1, random_state=45)
voting_clf.predict(random_user)

df.head()
df.loc[df.index ==5105488,"target"]





















