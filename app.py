import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


class App:
    def __init__(self):
        self.df = None
        self.df_cleaned = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.data_file = None

        self.classifier_name = None
        self.Init_Streamlit_Page()

    def Init_Streamlit_Page(self):
        st.title('YZUE Modül 1')

        self.data_file = st.sidebar.file_uploader("Choose a dataset from local", type=["csv", "xlsx"])

        self.classifier_name = st.sidebar.selectbox(
            'Select classifier',
            ('KNN', 'SVM', 'Naive Bayes')
        )
    def load_data(self):
        """
        Verisetini oku ve ilk 10 terimini göster.
        """

        # Verisetini okuyabildiysek self.df variable'ına atayabiliriz.
        if self.data_file is not None:
            if self.data_file.name.endswith('.csv'):
                self.df = pd.read_csv(self.data_file)
            else:
                self.df = pd.read_excel(self.data_file)

            # Assignment işleminden sonra datasetin ilk 10 değerini gösteriyoruz.
            st.write("First 10 rows of the dataset:")
            st.dataframe(self.df.head(10))

    def preprocess_data(self):
        """
        Verisetimize önişleme adımını uygula.
        """
        if self.df is not None:
            # Verisetimiz varsa, data temizle işlemini gerçekleştiriyoruz.
            # Bu aşama veriseti yüklenince otomatik olarak gerçekleşiyor.

            self.clean_data()  # Gereksiz sütunları temizle ve verisetinin son 10 satırını göster.

            self.change_diagnosis()  # Diagnosis sütunundaki verileri değiştir.

            self.split_data_and_visualize()

    def clean_data(self):
        """
        Verisetinden gereksiz sütunları kaldır.
        """
        # Index ve patient_id sütunları gereksiz olduğu için bu sütunları kaldırabiliriz.
        self.df.drop(['id'], axis=1,
                     inplace=True)  # id sütununu kaldır
        self.df.dropna(axis=1, how='all', inplace=True)  # boş olan sütunları kaldır.

        st.title("After preprocessing")
        st.write("Last 10 rows of the dataset:")
        st.dataframe(self.df.tail(10))

    def change_diagnosis(self):
        """
        Diagnosis sütunundaki M değerini 1, B değerini 0 olacak şekilde değiştir.
        Bu sütunu Y etiket verisi olarak kullan, geri kalan sütunları ise X öznitelik verisi olarak kullan.
        """
        if 'diagnosis' in self.df.columns:
            # 'M' değerini 1'e, 'B' değerini 0'a dönüştürme
            self.df['diagnosis'] = self.df['diagnosis'].map({'M': 1, 'B': 0})  # M'yi 1, B'yi de 0 olarak mapliyoruz.

    def split_data_and_visualize(self):
        """
        Verisetini training ve test olarak böl.
        """

        # Yüksek korelasyona sahip verileri kaldırıyorum.
        self.remove_correlated_features(threshold=0.99)

        X = self.df.drop('diagnosis', axis=1)  # X öznitelik verisinde diagnosis dışındaki sütunlar kullanılıyor.
        y = self.df['diagnosis']  # Diagnosis sütunu Y için etiket verisi olarak kullanılıyor.

        # Korelasyon matris çizdiriyoruz.
        st.title("Correlation matrix for selected dataset:")
        self.visualize_correlation_matrix()

        # Malignant ve bening olacak şekilde veriyi ayırıp radius_mean ve texture_mean olacak şekilde çizdiriyoruz.
        st.title("Radius_mean v Texture_Mean")
        self.visualize_graph()

        # Verileri 80-20 oranın ayırıyoruz.
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # StandartScaler uygulayarak accuracy arttırıyoruz.
        # Normalziasyon işlemini gerçekleştirmiş oluyoruz.
        scaler = StandardScaler()

        # X_train üzerinde fit_transform() uygula
        self.X_train = scaler.fit_transform(self.X_train)

        # X_test üzerinde transform() uygula
        self.X_test = scaler.transform(self.X_test)

    def visualize_correlation_matrix(self):
        """
        Korelasyon matrisini çizdiriyoruz.
        """
        if self.df is not None:
            plt.figure(figsize=(20, 16))
            sns.heatmap(self.df.corr(), annot=True, linewidths=.5, fmt=".2f")
            st.pyplot(plt)

    def visualize_graph(self):
        """
        Ödevde belirtilen grafiği çizdiriyoruz. radius_mean ve texture_mean birbirleri arasındaki korelasyonu görüyoruz.
        :return:
        """
        # Malignant (kötü) ve Benign (iyi) verilerini yeniden ayırma
        malignant = self.df[self.df['diagnosis'] == 1]
        benign = self.df[self.df['diagnosis'] == 0]

        # 'radius_mean' ve 'texture_mean' özniteliklerine göre seaborn kullanarak scatter grafiği çizdiriyoruz.
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='radius_mean', y='texture_mean', data=malignant, color='red', alpha=0.5, label='kotu')
        sns.scatterplot(x='radius_mean', y='texture_mean', data=benign, color='green', alpha=0.5, label='iyi')
        plt.title('radius_mean ve texture_mean comparison')
        plt.xlabel('radius_mean')
        plt.ylabel('texture_mean')
        st.pyplot(plt)

    def choose_model(self):
        """
        Eğitmek için model seç ve parametreleri belirle.
        """
        if self.classifier_name == 'KNN':
            self.model = KNeighborsClassifier()
            self.params = {
                'n_neighbors': list(range(1, 31)),
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        elif self.classifier_name == 'SVM':
            self.model = SVC()
            self.params = {
                'C': [0.1, 1, 10, 100],  # Düzenlileştirme parametresi
                'gamma': [1, 0.1, 0.01, 0.001],  # RBF kernel için gamma parametresi
                'kernel': ['rbf', 'sigmoid']  # Kernel tipi
            }
        elif self.classifier_name == 'Naive Bayes':
            self.model = GaussianNB()
            self.params = {}  # Naive Bayes için genellikle parametre ayarlaması gerekmez
        else:
            raise ValueError(f"Unknown classifier: {self.classifier_name}")

    def train_model(self):
        """
        Seçilen modeli GridSearchCV kullanarak en iyi parametrelerle eğit.
        """
        if self.model is not None and self.X_train is not None:
            # GridSearchCV objesini oluştur
            grid_search = GridSearchCV(self.model, param_grid=self.params, cv=5, scoring='accuracy')

            # Modeli eğit
            grid_search.fit(self.X_train, self.y_train)

            # En iyi parametreleri ve modeli al
            self.model = grid_search.best_estimator_
            st.write("Best parameters found:", grid_search.best_params_)
            st.write("Best score found:", grid_search.best_score_)

    def evaluate_model(self):
        """
        Eğitilmiş modeli değerlendir.
        """
        if self.model is not None and self.X_test is not None:
            y_pred = self.model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            cm = confusion_matrix(self.y_test, y_pred)

            st.write("Model Evaluation Metrics:")
            st.write(f"Accuracy: {accuracy:.2f}")
            st.write(f"Precision: {precision:.2f}")
            st.write(f"Recall: {recall:.2f}")
            st.write(f"F1 Score: {f1:.2f}")

            st.write("Confusion Matrix:")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", ax=ax)
            ax.set_xlabel('y_predicted')
            ax.set_ylabel('y_true')
            st.pyplot(fig)

    def run(self):
        """
        Streamlit uygulamasını çalıştır.
        """
        st.title('Data Set Analysis')

        self.load_data()  # Veriyi bilgisayardan seç ve yükle. (Görev 1)

        self.preprocess_data()  # Veriyi önişlemeden geçir ve diğer işlemleri yap. (Görev 2)

        if st.sidebar.button('Train Model'):
            self.choose_model()
            self.train_model()
            self.evaluate_model()


    def drop_unnecessary_columns(self):
        """
        Manuel olarak kaldırmak istenirse bu kullanılabilir.
        """
        drop_list1 = ['perimeter_mean', 'radius_mean', 'compactness_mean', 'concave points_mean', 'radius_se',
                      'perimeter_se', 'radius_worst', 'perimeter_worst', 'compactness_worst', 'concave points_worst',
                      'compactness_se', 'concave points_se', 'texture_worst', 'area_worst']
        # Birbirleriyle korelasyonu yüksek olan sütunları kaldırdım.
        self.df.drop(drop_list1, axis=1, inplace=True)

        # Diagnosis ile negatif korelasyon sağlayan değerleri kaldırdım.
        self.df.drop(['fractal_dimension_mean', 'texture_se', 'smoothness_se'],
                     axis=1, inplace=True)

    def remove_correlated_features(self, threshold=0.9):
        """
        Otomatik olarak korelasoynlara göre kaldırmak için bu fonksiyon kullanılabilir.
        """
        corr_matrix = self.df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] >= threshold)]
        self.df.drop(to_drop, axis=1, inplace=True)

        # Diagnosis ile negatif korelasyon sağlayan değerleri kaldırdım.
        self.df.drop(['fractal_dimension_mean', 'texture_se', 'smoothness_se'],
                     axis=1, inplace=True)

        return self.df, to_drop
