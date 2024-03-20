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
y=df[target]
train_X, test_X, train_y, test_y = train_test_split(X, y,random_state = 0)

#training and predicting temp from the training data
model1=DecisionTreeRegressor(random_state=1)
model1.fit(train_X, train_y)
y_pred_temp=model1.predict(test_X)

#mean absolute error of the temp predictions
print("Mean absolute error of temp: %f" %(mean_absolute_error(test_y, y_pred_temp)))

#mean value of the temp predictions
Average_temp = sum(y_pred_temp)/len(y_pred_temp)
print("Mean of temp: %f" %round(Average_temp,2))

# It is a text value that we want to convert to audio  
text_val = ("the average temperature today is %f" %Average_temp)

# Here are converting in English Language  
language = 'en'

# Passing the text and language to the engine, 
obj = gTTS(text=text_val, lang=language, slow=False)

#Here we are saving the transformed audio in a mp3 file named temp.mp3  
obj.save("temp.mp3")

# Play the temp.mp3 file
playsound("temp.mp3")



#train parameter rainfall
features = ["name","datetime","tempmax","tempmin","temp","feelslikemax","feelslikemin","feelslike","dew","humidity","precip","precipprob","precipcover","preciptype","snow","snowdepth","windgust","windspeed","winddir","sealevelpressure","cloudcover","visibility","solarradiation","solarenergy","uvindex","severerisk","sunrise","sunset","moonphase","conditions","description","icon","stations"]
target = 'precip'

#preparing the data for training
X=df[features]
y=df[target]
train_X, test_X, train_y, test_y = train_test_split(X, y,random_state = 0)

#training and predicting rainfall from the training data
model1=DecisionTreeRegressor(random_state=1)
model1.fit(train_X, train_y)
y_pred_rainfall=model1.predict(test_X)

#mean absolute error of the rainfall predictions
print("Mean absolute error of rainfall: %f" %(mean_absolute_error(test_y, y_pred_rainfall)))

#mean value of the rainfall predictions
Average_rainfall = sum(y_pred_rainfall)/len(y_pred_rainfall)
print("Mean of rainfall : %f" %round(Average_rainfall,2))

# It is a text value that we want to convert to audio  
text_val = ("the average rainfall today is %f" %Average_rainfall)

# Here are converting in English Language  
language = 'en'

# Passing the text and language to the engine, 
obj = gTTS(text=text_val, lang=language, slow=False)

#Here we are saving the transformed audio in a mp3 file named rainfall.mp3  
obj.save("rainfall.mp3")

# Play the rainfall.mp3 file
playsound("rainfall.mp3")



#train parameter humidity
features = ["name","datetime","tempmax","tempmin","temp","feelslikemax","feelslikemin","feelslike","dew","humidity","precip","precipprob","precipcover","preciptype","snow","snowdepth","windgust","windspeed","winddir","sealevelpressure","cloudcover","visibility","solarradiation","solarenergy","uvindex","severerisk","sunrise","sunset","moonphase","conditions","description","icon","stations"]
target = 'humidity'

#preparing the data for training
X=df[features]
y=df[target]
train_X, test_X, train_y, test_y = train_test_split(X, y,random_state = 0)

#training and predicting humidity from the training data
model1=DecisionTreeRegressor(random_state=1)
model1.fit(train_X, train_y)
y_pred_humidity=model1.predict(test_X)

#mean absolute error of the humidity predictions
print("Mean absolute error of humidity : %f" %(mean_absolute_error(test_y, y_pred_humidity)))

#mean value of the humidity predictions
Average_humidity = sum(y_pred_humidity)/len(y_pred_humidity)
print("Mean of humidity : %f" %round(Average_humidity,2))

# It is a text value that we want to convert to audio  
text_val = ("the humidity today is %f" %Average_humidity)

# Here are converting in English Language  
language = 'en'

# Passing the text and language to the engine, 
obj = gTTS(text=text_val, lang=language, slow=False)

#Here we are saving the transformed audio in a mp3 file named humidity.mp3  
obj.save("humidity.mp3")

# Play the humidity.mp3 file
playsound("humidity.mp3")




#train parameter UV Index
features = ["name","datetime","tempmax","tempmin","temp","feelslikemax","feelslikemin","feelslike","dew","humidity","precip","precipprob","precipcover","preciptype","snow","snowdepth","windgust","windspeed","winddir","sealevelpressure","cloudcover","visibility","solarradiation","solarenergy","uvindex","severerisk","sunrise","sunset","moonphase","conditions","description","icon","stations"]
target = 'uvindex'

#preparing the data for training
X=df[features]
y=df[target]
train_X, test_X, train_y, test_y = train_test_split(X, y,random_state = 0)

#training and predicting UV Index from the training data
model1=DecisionTreeRegressor(random_state=1)
model1.fit(train_X, train_y)
y_pred_uvindex=model1.predict(test_X)

#mean absolute error of the UV Index predictions
print("Mean absolute error of UV Index : %f" %(mean_absolute_error(test_y, y_pred_uvindex)))

#mean value of the UV Index predictions
Average_uvindex = sum(y_pred_uvindex)/len(y_pred_uvindex)
print("Mean of UV Index : %f" %round(Average_uvindex,2))

# It is a text value that we want to convert to audio  
text_val = ("the UV index today is %f" %Average_uvindex)

# Here are converting in English Language  
language = 'en'

# Passing the text and language to the engine, 
obj = gTTS(text=text_val, lang=language, slow=False)

#Here we are saving the transformed audio in a mp3 file named uvindex.mp3  
obj.save("uvindex.mp3")

# Play the uvindex.mp3 file
playsound("uvindex.mp3")




#train parameter wind speed
features = ["name","datetime","tempmax","tempmin","temp","feelslikemax","feelslikemin","feelslike","dew","humidity","precip","precipprob","precipcover","preciptype","snow","snowdepth","windgust","windspeed","winddir","sealevelpressure","cloudcover","visibility","solarradiation","solarenergy","uvindex","severerisk","sunrise","sunset","moonphase","conditions","description","icon","stations"]
target = 'windspeed'

#preparing the data for training
X=df[features]
y=df[target]
train_X, test_X, train_y, test_y = train_test_split(X, y,random_state = 0)

#training and predicting windspeed from the training data
model1=DecisionTreeRegressor(random_state=1)
model1.fit(train_X, train_y)
y_pred_windspeed=model1.predict(test_X)

#mean absolute error of the wind speed predictions
print("Mean absolute error of wind speed : %f" %(mean_absolute_error(test_y, y_pred_windspeed)))

#mean value of the wind speed predictions
Average_windspeed = sum(y_pred_windspeed)/len(y_pred_windspeed)
print("Mean of wind speed : %f" %round(Average_windspeed,2))

# It is a text value that we want to convert to audio  
text_val = ("the wind speed today is %f" %Average_windspeed)

# Here are converting in English Language  
language = 'en'

# Passing the text and language to the engine, 
obj = gTTS(text=text_val, lang=language, slow=False)

#Here we are saving the transformed audio in a mp3 file named windspeed.mp3  
obj.save("windspeed.mp3")

# Play the windspeed.mp3 file
playsound("windspeed.mp3")



#train parameter visibility
features = ["name","datetime","tempmax","tempmin","temp","feelslikemax","feelslikemin","feelslike","dew","humidity","precip","precipprob","precipcover","preciptype","snow","snowdepth","windgust","windspeed","winddir","sealevelpressure","cloudcover","visibility","solarradiation","solarenergy","uvindex","severerisk","sunrise","sunset","moonphase","conditions","description","icon","stations"]
target = 'visibility'

#preparing the data for training
X=df[features]
y=df[target]
train_X, test_X, train_y, test_y = train_test_split(X, y,random_state = 0)

#training and predicting visibility from the training data
model1=DecisionTreeRegressor(random_state=1)
model1.fit(train_X, train_y)
y_pred_visibility=model1.predict(test_X)

#mean absolute error of the visibility predictions
print("Mean absolute error of visibility : %f" %(mean_absolute_error(test_y, y_pred_visibility)))

#mean value of the visibility predictions
Average_visibility = sum(y_pred_visibility)/len(y_pred_visibility)
print("Mean of visibility : %f" %round(Average_visibility,2))

# It is a text value that we want to convert to audio  
text_val = ("the visibility today is %f" %Average_visibility)

# Here are converting in English Language  
language = 'en'

# Passing the text and language to the engine, 
obj = gTTS(text=text_val, lang=language, slow=False)

#Here we are saving the transformed audio in a mp3 file named visibility.mp3  
obj.save("visibility.mp3")

# Play the visibility.mp3 file
playsound("visibility.mp3")

#TTS for Umbrellla
if (Average_rainfall>0 or Average_uvindex>8):
    # It is a text value that we want to convert to audio  
    text_val = ("We would suggest you to carry an umbrella")
    print(text_val)

    # Here are converting in English Language  
    language = 'en'

    # Passing the text and language to the engine, 
    obj = gTTS(text=text_val, lang=language, slow=False)

    #Here we are saving the transformed audio in a mp3 file named umbrella.mp3  
    obj.save("umbrella.mp3")

    # Play the umbrella.mp3 file
    playsound("umbrella.mp3")
else:
    # It is a text value that we want to convert to audio  
    text_val = ("Carrying an umbrella might not be necessary")
    print(text_val)

    # Here are converting in English Language  
    language = 'en'

    # Passing the text and language to the engine, 
    obj = gTTS(text=text_val, lang=language, slow=False)

    #Here we are saving the transformed audio in a mp3 file named umbrella.mp3  
    obj.save("umbrella.mp3")

    # Play the umbrella.mp3 file
    playsound("umbrella.mp3")

#TTS for driving
if (Average_visibility<=3):
    # It is a text value that we want to convert to audio  
    text_val = ("Driving might not be advised")
    print(text_val)

    # Here are converting in English Language  
    language = 'en'

    # Passing the text and language to the engine, 
    obj = gTTS(text=text_val, lang=language, slow=False)

    #Here we are saving the transformed audio in a mp3 file named driving.mp3  
    obj.save("driving.mp3")

    # Play the driving.mp3 file
    playsound("driving.mp3")
    
elif (Average_visibility>3 and Average_visibility<10):
    # It is a text value that we want to convert to audio  
    text_val = ("Please be cautious while driving");
    print(text_val)

    # Here are converting in English Language  
    language = 'en'

    # Passing the text and language to the engine, 
    obj = gTTS(text=text_val, lang=language, slow=False)

    #Here we are saving the transformed audio in a mp3 file named driving.mp3  
    obj.save("driving.mp3")

    # Play the driving.mp3 file
    playsound("driving.mp3")

#TTS for clothes 
if (Average_temp<=15):
    # It is a text value that we want to convert to audio  
    text_val = ("You should wear a sweater or a jacket")
    print(text_val)

    # Here are converting in English Language  
    language = 'en'

    # Passing the text and language to the engine, 
    obj = gTTS(text=text_val, lang=language, slow=False)

    #Here we are saving the transformed audio in a mp3 file named clothes.mp3  
    obj.save("clothes.mp3")

    # Play the clothes.mp3 file
    playsound("clothes.mp3")

elif(Average_temp>15 and Average_temp<=25):
    # It is a text value that we want to convert to audio  
    text_val = ("You should wear full clothes")
    print(text_val)

    # Here are converting in English Language  
    language = 'en'

    # Passing the text and language to the engine, 
    obj = gTTS(text=text_val, lang=language, slow=False)

    #Here we are saving the transformed audio in a mp3 file named clothes.mp3  
    obj.save("clothes.mp3")

    # Play the clothes.mp3 file
    playsound("clothes.mp3")

else:
    # It is a text value that we want to convert to audio  
    text_val = ("Please wear light clothes")
    print(text_val)

    # Here are converting in English Language  
    language = 'en'

    # Passing the text and language to the engine, 
    obj = gTTS(text=text_val, lang=language, slow=False)

    #Here we are saving the transformed audio in a mp3 file named clothes.mp3  
    obj.save("clothes.mp3")

    # Play the clothes.mp3 file
    playsound("clothes.mp3")

#TTS for sunscreen
if (Average_uvindex>8 or Average_temp>30):
    # It is a text value that we want to convert to audio  
    text_val = ("Please apply sunscreen")
    print(text_val)
    
    # Here are converting in English Language  
    language = 'en'

    # Passing the text and language to the engine, 
    obj = gTTS(text=text_val, lang=language, slow=False)

    #Here we are saving the transformed audio in a mp3 file named sunscreen.mp3  
    obj.save("sunscreen.mp3")

    # Play the sunscreen.mp3 file
    playsound("sunscreen.mp3")

#TTS for general suggestions
if (Average_temp<=15 and (Average_rainfall>0 or Average_uvindex>8) and Average_visibility<=3):
    # It is a text value that we want to convert to audio  
    text_val = ("Today the temparature is :%f. Rainfall is expected today. Driving is not safe today. You should carry an umbrella. Please wear a sweater or a jacket" %Average_temp)
    print(text_val)

    # Here are converting in English Language  
    language = 'en'

    # Passing the text and language to the engine, 
    obj = gTTS(text=text_val, lang=language, slow=False)

    #Here we are saving the transformed audio in a mp3 file named condition1.mp3  
    obj.save("condition1.mp3")

    # Play the condition1.mp3 file
    playsound("condition1.mp3")

elif (Average_temp>15 and Average_temp<=25 and (Average_rainfall>0 or Average_uvindex>8) and Average_visibility<=3):
    # It is a text value that we want to convert to audio  
    text_val = ("Today the temparature is :%f. Rainfall is expected today. Driving is not safe today. You should carry an umbrella. You should wear full clothes" %Average_temp)
    print(text_val)

    # Here are converting in English Language  
    language = 'en'

    # Passing the text and language to the engine, 
    obj = gTTS(text=text_val, lang=language, slow=False)

    #Here we are saving the transformed audio in a mp3 file named condition2.mp3  
    obj.save("condition2.mp3")

    # Play the condition2.mp3 file
    playsound("condition2.mp3")

elif (Average_temp>15 and Average_temp<=25 and (Average_rainfall>0 or Average_uvindex>8) and Average_visibility>3 and Average_visibility<=10):
    # It is a text value that we want to convert to audio  
    text_val = ("Today the temparature is :%f. Rainfall is expected today. Please be careful while driving. You should carry an umbrella. You should wear full clothes" %Average_temp);
    print(text_val)

    # Here are converting in English Language  
    language = 'en'

    # Passing the text and language to the engine, 
    obj = gTTS(text=text_val, lang=language, slow=False)

    #Here we are saving the transformed audio in a mp3 file named condition3.mp3  
    obj.save("condition3.mp3")

    # Play the condition3.mp3 file
    playsound("condition3.mp3")

elif (Average_temp>15 and Average_temp<=25 and (Average_rainfall==0 or Average_uvindex<=8) and Average_visibility>3 and Average_visibility<=10):
    # It is a text value that we want to convert to audio  
    text_val = ("Today the temparature is :%f. Rainfall is not expected today. Please be careful while driving. You should wear full clothes" %Average_temp)
    print(text_val)

    # Here are converting in English Language  
    language = 'en'

    # Passing the text and language to the engine, 
    obj = gTTS(text=text_val, lang=language, slow=False)

    #Here we are saving the transformed audio in a mp3 file named condition4.mp3  
    obj.save("condition4.mp3")

    # Play the condition4.mp3 file
    playsound("condition4.mp3")

elif (Average_temp>25 and (Average_rainfall==0 or Average_uvindex<=8) and Average_visibility>10):
    # It is a text value that we want to convert to audio  
    text_val = ("Today the temparature is :%f. Rainfall is not expected today. Driving is safer today. You should wear light clothes" %Average_temp)
    print(text_val)

    # Here are converting in English Language  
    language = 'en'

    # Passing the text and language to the engine, 
    obj = gTTS(text=text_val, lang=language, slow=False)

    #Here we are saving the transformed audio in a mp3 file named condition5.mp3  
    obj.save("condition5.mp3")

    # Play the condition5.mp3 file
    playsound("condition5.mp3")
