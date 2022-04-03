from sklearn import linear_model
import pandas as pd
import pickle

df = pd.read_csv('prices.csv')

y = df['Value'] # dependent variable
X = df[['Rooms', 'Distance']] # independent variable

lm = linear_model.LinearRegression()
lm.fit(X, y) # fitting the model
pickle.dump(lm, open('model.pkl','wb')) # save the model

print(lm.predict([[15, 61]]))  # format of input
print(f'score: {lm.score(X, y)}')
