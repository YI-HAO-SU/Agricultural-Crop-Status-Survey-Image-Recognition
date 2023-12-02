import pandas as pd
import os

DATASET_PATH = "D:/ML_exercise/tag_locCoor.csv"

species = ['asparagus', 'bambooshoots', 'betel', 'broccoli', 'cauliflower', 'chinesecabbage', 'chinesechives', 'custardapple', 'grape', 'greenhouse', 'greenonion', 'kale', 'lemon', 'lettuce', 'litchi', 'longan', 'loofah', 'mango', 'onion', 'others', 'papaya', 'passionfruit', 'pear', 'pennisetum', 'redbeans', 'roseapple', 'sesbania', 'soybeans', 'sunhemp', 'sweetpotato', 'taro', 'tea', 'waterbamboo']
names = []

df = pd.read_csv(DATASET_PATH, encoding='big5')

for i in range(len(df)):
    names.append(df.at[i, "Img"])

df = df.set_index('TARGET_FID')
df.insert(1, column="Species", value="")
print(df)



'''
for K in species:
    print(K)
    for root, dirs, files in os.walk("D:/111_Su_Kevin/AI_CUP/Datasets/{}".format(K)):
        for file in files:
            for i in range(len(names)):
                if file == names[i]:
                    df.at[i, "Species"] = K
                   
print(df)
df.to_csv("D:/111_Su_Kevin/AI_CUP/tt_1.csv")
'''
print("done")
