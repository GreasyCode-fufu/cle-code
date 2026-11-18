import pandas as pd
import os
import shutil

# 读取xlsx文件
df = pd.read_excel('E:\edge download\c5yvn32dzg-2\Annotations\data.xlsx')

# 创建类别文件夹
categories = df['Multiclass.Label'].unique()
for category in categories:
    os.makedirs(category, exist_ok=True)

# 遍历xlsx文件的每一行
for index, row in df.iterrows():
    image_name = row['File']
    category = row['Multiclass.Label']
    source_path = os.path.join('E:\edge download\c5yvn32dzg-2\Photos', image_name)
    destination_path = os.path.join(category, image_name)

    # 将图片移动到对应的类别文件夹中
    shutil.move(source_path, destination_path)
