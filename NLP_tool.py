import numpy as np
import jaconv
from scipy import spatial
from sklearn.feature_extraction.text import TfidfVectorizer


class Sentence_similarity():
    def __init__(self, wakati_tagger, model, feature_dimension=300):
        self.feature_dimension = feature_dimension
        self.model = model
        self.mecab = wakati_tagger

    def _wakati_rank(self, sentence, nbest=1):
        # 分かち書きの候補を生成
        if nbest >= 2:
            wacati_rank = list(dict.fromkeys(
                self.mecab.parseNBest(nbest, sentence).strip().split(' \n')))
            wacati_rank_ls = [i.split(' ') for i in wacati_rank]
        # mecab.parseNBestは遅いので、nbest=1の時はmecab.parseを使うようにする
        elif nbest == 1:
            wacati_rank = self.mecab.parse(sentence).strip().split(' ')
            wacati_rank_ls = [wacati_rank]
        return wacati_rank_ls

    def avg_feature_vector(self, sentence, error_words=False, nbest=1):
        # 分かち書きの候補を生成
        wacati_rank_ls = self._wakati_rank(sentence, nbest)
        for words in wacati_rank_ls:
            # 特徴ベクトルの入れ物を初期化
            feature_vec = np.zeros((self.feature_dimension,), dtype="float32")
            error_words_list = []
            # 文章を平均化する
            for word in words:
                try:
                    feature_vec = np.add(feature_vec, self.model[word])
                except:
                    error_words_list.append(word)
            feature_vec = np.divide(feature_vec, len(words))
            # 全ての単語がerrorだった場合、ループが続く（wakati_rank_lsの次の候補に行く）
            if len(error_words_list) != len(words):
                break
        # 値を返す
        # print ('used wakati sentence:', words)
        if error_words is True:
            return feature_vec, error_words_list
        elif error_words is False:
            return feature_vec

    def sentence_similarity(self, sentence_1, sentence_2, nbest=5):
        sentence_1_avg_vector = self.avg_feature_vector(
            sentence=sentence_1, nbest=nbest)
        sentence_2_avg_vector = self.avg_feature_vector(
            sentence=sentence_2, nbest=nbest)
        # １からベクトル間の距離を引いてあげることで、コサイン類似度を計算
        similarity = 1 - spatial.distance.cosine(
            sentence_1_avg_vector, sentence_2_avg_vector)
        return similarity


class One_HotVector():
    def __init__(self, mecabrc_tagger, mecabrc_tagger2=None):
        self.mecab = mecabrc_tagger
        self.mecab2 = mecabrc_tagger2

    def wakati_noun(self, text, nbest=10, double_dict=False, letter_conv=True):
        """
            文章から単語を抽出
        """
        text = jaconv.h2z(text)  # 半角を全角に
        out_words = []
        node = self.mecab.parseNBest(nbest, text).split('\n')
        # neologd辞書と通常辞書の両方を使いたい時
        if ((double_dict is True) & (self.mecab2 is not None)):
            node2 = self.mecab2.parse(text).split('\n')
            node.extend(node2)
        if letter_conv is True:  # ひらがなをカタカナへ
            node3 = list(map(jaconv.hira2kata, node))
            node.extend(node3)
        node = list(set(node))  # 重複を削除

        for i in range(len(node)):
            if (((node[i].find('名詞') != -1) | (node[i].find('形容詞') != -1) |
                 (node[i].find('動詞') != -1)) & (node[i].find('地域') == -1)):
                out_words.append(node[i].split('	')[0])
        return list(set(out_words))

    def to_onehot(self, documents):
        """
        各文章における重み付け
        """
        docs = np.array(documents)
        vectorizer = TfidfVectorizer(
            analyzer=self.wakati_noun,
            stop_words='|',
            min_df=1,
            use_idf=True,
            lowercase=False,
            # max_df=6
            # token_pattern='(?u)\\b\\w+\\b' #文字列長が1の単語を処理対象に含める
        )
        vecs = vectorizer.fit_transform(docs)

        tango = []
        for k, v in sorted(vectorizer.vocabulary_.items(),
                           key=lambda x: x[1]): tango.append(k)
        return vecs.toarray(), tango


class NLP_tool(Sentence_similarity, One_HotVector):
    def __init__(self, mecab, model):
        super(Sentence_similarity, self).__init__()
