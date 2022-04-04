import pandas as pd
import numpy as np

df = pd.read_csv('pizza.csv')
X=df.drop(['LIKEPIZZA'],axis=1)
y=df['LIKEPIZZA']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20, random_state = 42 )


from sklearn.neighbors import KNeighborsClassifier
models = KNeighborsClassifier(n_neighbors=2)
models.fit(X,y)

import pickle
pickle.dump(models, open('model_knn.pkl','wb'))
model_knn = pickle.load(open('model_knn.pkl','rb'))
model_knn= pickle.load(open('model_knn.pkl','rb'))





# In[ ]:




