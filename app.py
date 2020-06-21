import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.title("streamlit project")
wrio = st.write("loading dataset ...")

def load_data():
	data = datasets.load_iris()
	X = pd.DataFrame(data.data,columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
	Y = data.target
	return X,Y
X,Y = load_data()

wrio = st.write("loading datasets done")

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

if st.checkbox("For see train data"):
	st.write(X_train)

model_select = st.sidebar.selectbox("Select ML model", ("KNN", "Naive bayes"))

st.subheader(f"ML model is {model_select}")

def hyperparameter(model_select):
	dict1 = {}
	if model_select == "KNN":
		default_value = st.sidebar.slider('K', 1, 10,3)
		dict1['K'] = default_value
	return dict1

dict1 = hyperparameter(model_select)

if model_select == "KNN":
	neigh = KNeighborsClassifier(n_neighbors=dict1['K'])
	neigh.fit(X_train, y_train)
	pred = neigh.predict(X_test)
	acc = accuracy_score(y_test,pred)
	st.write(f'accuracy_score = {acc}')

else:
	clf = GaussianNB()
	clf.fit(X_train, y_train)
	pred = clf.predict(X_test)
	acc = accuracy_score(y_test,pred)
	st.write(f'accuracy_score = {acc}')

pca = PCA(n_components=2)
pl = pca.fit_transform(X)
x_projection = pl[:,0]
y_projection = pl[:,1]

fig = plt.figure()
plt.scatter(x_projection,y_projection,c=Y,alpha = 0.8,cmap = "viridis")
plt.xlabel = ("x_projection")
plt.ylabel = ("y_projection")
plt.colorbar()

st.pyplot()



