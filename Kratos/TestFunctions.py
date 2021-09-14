from pandas.compat import to_str
#
from Kratos import TextRankProject,Classifications
# sentences, context = po.text_rank_output("../Yogesh_Test_data.csv")

# Classifications.naive_bayes()

# print(sentences,context)
# with open('output.csv', 'w') as f:
#     f.write("text, context\n")
#     for i in range(4):
#         line=sentences[i]+", "+to_str(context[i])+"\n"
#         f.write(line)
#         # print(type(line))
# # import pandas as pd
# # ss=pd.read_csv("output.csv")
# # print(ss)

# --------PREPROCESSING-----------------
# from Kratos import preprocess as po
# import pandas
# data=pandas.read_csv("../new_dataset.csv", encoding="latin")
# po.pre_precess_text(data)


#---------TRAINING DATA---------------------
# from Kratos import NN_Classification
# X_train, X_test, y_train, y_test=NN_Classification.split_data("output.csv")
# NN_Classification.training_NN(X_train, X_test, y_train, y_test)

# -------NURAL NETWOEK------------------
# from Kratos import NN_Classification
# sentences=['AI', 'IOT','technology']
# contex=NN_Classification.get_context(sentences)
# print(contex)

# ------NURAL NETWORK------------
from Kratos import NN_Classification
NN_Classification.analyze_NN('output.csv')
# -------Mapping test------
# from Kratos import mapping
# str=[1,2,4,5]
# p=mapping.map(str)
# print(p)

#------naive baiys ---------
# from Kratos import Classifications
# from Kratos import TextRankProject
# import pandas as pd
# data=pd.read_csv("../Test_data.csv",encoding="latin")
# print(data.head(10))
# text_data=data['text']
# context=Classifications.get_context("stored_model/naive_bayes.pkl",text_data)
# print(context)
# Classifications.get_context("stored_model/naive_bayes.pkl",text_data)
# Classifications.get_context()
#
# from Kratos import TextRankProject
# TextRankProject.text_rank_output('../Test_data.csv')