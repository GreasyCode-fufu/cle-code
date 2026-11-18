#引入相关库

import pandas as pd
import os
import shutil #用于移动文件

#输入表格所在路径+名称并读取，如D:/data/label.csv,文件后缀不要忘记
csv_path = '表格所在路径+名称'
list=pd.read_csv(csv_path)
# print(list) #此命令可查看csv文件内容

path = '图片所在路径/.png' #若图像格式为其他，则将后缀更改为其他

# list["File ID"] = input_path

list["File ID1"]=list["File_ID"] #File_ID为csv文件中图片文件名，含后缀，即与path路径的图片名完全一致

# # #
#label为0,1，2
for i in range(3):
    #即例如指定输出路径分别为D:/data/val/0,1,2
    output_path = '图片所在路径/val/'+str(i)
    #输出路径是否存在，若不存在则创建
    isExists = os.path.exists(output_path)
    if not isExists:
        os.makedirs(output_path)
    #label为csv文件中标签列的列名
    listnew = list[list["label"] == i]
    l = listnew["File ID1"].tolist()
    # 对于File ID中的每一个，执行移动指令
    for each in l:
        shutil.move(each, output_path)

#

