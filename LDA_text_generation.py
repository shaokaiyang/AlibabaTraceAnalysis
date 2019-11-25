"""
@file   : 010-希拉里邮件进行主题建立之主题模型.py
@author : xiaolu
@time1  : 2019-05-11
"""
import numpy as np
import pandas as pd
import re
from gensim.models import doc2vec, ldamodel
from gensim import corpora


def clean_email_text(text):
    # 数据清洗
    text = text.replace('\n', " ")  # 新行，我们是不需要的
    text = re.sub(r"-", " ", text)  # 把 "-" 的两个单词，分开。（比如：july-edu ==> july edu）
    text = re.sub(r"\d+/\d+/\d+", "", text)  # 日期，对主体模型没什么意义
    text = re.sub(r"[0-2]?[0-9]:[0-6][0-9]", "", text)  # 时间，没意义
    text = re.sub(r"[\w]+@[\.\w]+", "", text)  # 邮件地址，没意义
    text = re.sub(r"/[a-zA-Z]*[:\//\]*[A-Za-z0-9\-_]+\.+[A-Za-z0-9\.\/%&=\?\-_]+/i", "", text)  # 网址，没意义
    pure_text = ''
    # 以防还有其他特殊字符（数字）等等，我们直接把他们loop一遍，过滤掉
    for letter in text:
        # 只留下字母和空格
        if letter.isalpha() or letter == ' ':
            pure_text += letter
    # 再把那些去除特殊字符后落单的单词，直接排除。
    # 我们就只剩下有意义的单词了。
    text = ' '.join(word for word in pure_text.split() if len(word) > 1)  # 而且单词长度必须是2以上
    return text


def remove_stopword():
    stopword = []
    with open('./data/stop_words.utf8', 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace('\n', '')
            stopword.append(line)
    return stopword


if __name__ == '__main__':
    # 加载数据
    df = pd.read_csv('./data/HillaryEmails.csv')
    df = df[['Id', 'ExtractedBodyText']].dropna()  # 这两列主要有空缺值，这条数据就不要了。
    print(df.head())
    print(df.shape)  # (6742, 2)

    docs = df['ExtractedBodyText']  # 获取邮件
    docs = docs.apply(lambda s: clean_email_text(s))  # 对邮件清洗

    print(docs.head(1).values)
    doclist = docs.values  # 直接将内容拿出来
    print(docs)

    stop_word = remove_stopword()

    texts = [[word for word in doc.lower().split() if word not in stop_word] for doc in doclist]
    print(texts[0])  # 第一个文本现在的样子

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    print(corpus[0])  # [(36, 1), (505, 1), (506, 1), (507, 1), (508, 1)]

    lda = ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=20)
    print(lda.print_topic(10, topn=5))  # 第10个主题最关键的五个词
    print(lda.print_topics(num_topics=20, num_words=5))  # 把所有的主题打印出来看看

    # 保存模型
    lda.save('zhutimoxing.model')

    # 加载模型
    lda = ldamodel.LdaModel.load('zhutimoxing.model')

    # 新鲜数据，判读主题
    text = 'I was greeted by this heartwarming display on the corner of my street today. ' \
           'Thank you to all of you who did this. Happy Thanksgiving. -H'
    text = clean_email_text(text)
    texts = [word for word in text.lower().split() if word not in stop_word]
    bow = dictionary.doc2bow(texts)
    print(lda.get_document_topics(bow))  # 最后得出属于这三个主题的概率为[(4, 0.6081926), (11, 0.1473181), (12, 0.13814318)]
