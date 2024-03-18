import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

from gtts import gTTS
from playsound import playsound

#read the data
data = pd.read_csv(r"C:\Users\M V S Akhil Teja\innov-tionX\weatherdata.csv")

#creating dataframe
df = pd.DataFrame(data)

#label encoding name
df['name']=LabelEncoder().fit_transform(df['name'])

#label encoding datetime
df['datetime']=LabelEncoder().fit_transform(df['datetime'])

#label encoding preciptype
df['preciptype']=LabelEncoder().fit_transform(df['preciptype'])

#label encoding sunrise
df['sunrise']=LabelEncoder().fit_transform(df['sunrise'])

#label encoding sunset
df['sunset']=LabelEncoder().fit_transform(df['sunset'])

#label encoding conditions
df['conditions']=LabelEncoder().fit_transform(df['conditions'])

#label encoding descriptions
df['description']=LabelEncoder().fit_transform(df['description'])

#label encoding icon
df['icon']=LabelEncoder().fit_transform(df['icon'])

#label encoding stations
df['stations']=LabelEncoder().fit_transform(df['stations'])

#train parameter temperature
features = ["name","datetime","tempmax","tempmin","temp","feelslikemax","feelslikemin","feelslike","dew","humidity","precip","precipprob","precipcover","preciptype","snow","snowdepth","windgust","windspeed","winddir","sealevelpressure","cloudcover","visibility","solarradiation","solarenergy","uvindex","severerisk","sunrise","sunset","moonphase","conditions","description","icon","stations"]
target = 'temp'

#preparing the data for training
X=df[features]
y=df['temp']
train_X, test_X, train_y, test_y = train_test_split(X, y,random_state = 0)

#training and predicting the mean the data
model1=DecisionTreeRegressor(random_state=1)
model1.fit(train_X, train_y)
pred1=model1.predict(test_X)
#StandardScaler(*, copy=False, with_mean=True, with_std=False)
print("Mean_absolute_error : %f" %(mean_absolute_error(test_y, pred1)))

# Make predictions for temp
y_pred = model1.predict(test_X)

#mean value of the temp predictions
Average = sum(y_pred)/len(y_pred)
print("Mean : %f" %round(Average,2))

# It is a text value that we want to convert to audio  
text_val = ("the average temperature today is %f" %Average)

# Here are converting in English Language  
language = 'en'

# Passing the text and language to the engine, 
obj = gTTS(text=text_val, lang=language, slow=False)

#Here we are saving the transformed audio in a mp3 file named temp.mp3  
obj.save("temp.mp3")

# Play the temp.mp3 file
playsound("temp.mp3")

