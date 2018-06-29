import numpy as np
from scipy import spatial

class Sentence_similarity():
    def __init__(self,mecab, model, feature_dimension=300):
        self.feature_dimension=feature_dimension
        self.model=model
        self.mecab=mecab

    def avg_feature_vector(self, sentence, error_words=False):
        words = self.mecab.parse(sentence).replace(' \n', '').split() # mecabの分かち書きでは最後に改行(\n)が出力されてしまうため、除去
        feature_vec = np.zeros((self.feature_dimension,), dtype="float32") # 特徴ベクトルの入れ物を初期化
        error_words_list=[]
        for word in words:
            try:
                feature_vec = np.add(feature_vec, self.model[word])
            except:
                error_words_list.append(word)
        if len(words) > 0:
            feature_vec = np.divide(feature_vec, len(words))

        if error_words==True: return feature_vec, error_words_list
        elif error_words==False: return feature_vec

    def sentence_similarity(self, sentence_1, sentence_2):
        # 今回使うWord2Vecのモデルは300次元の特徴ベクトルで生成されているので、feature_dimension
        sentence_1_avg_vector = self.avg_feature_vector(sentence=sentence_1)
        sentence_2_avg_vector = self.avg_feature_vector(sentence=sentence_2)
        # １からベクトル間の距離を引いてあげることで、コサイン類似度を計算
        return 1 - spatial.distance.cosine(sentence_1_avg_vector, sentence_2_avg_vector)


