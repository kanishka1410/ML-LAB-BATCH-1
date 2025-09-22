from google.colab import files

uploaded = files.upload()  # This will prompt you to select file(s)

# After upload, you can read the file as:
import pandas as pd
data = pd.read_csv('kan.csv', sep='\s+')  # \s+ means one or more whitespace characters


df=pd.DataFrame(data)
print(df)

