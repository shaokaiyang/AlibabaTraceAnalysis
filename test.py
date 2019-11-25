
import pandas as pd

df = pd.DataFrame({"id": [1, 2, 3, 4, 5,6,7,8,9]})

for i in range(3):
    print(i)

print(df.iloc[[0], [0]])
print(df.at[0, 'id'])
print(df)
