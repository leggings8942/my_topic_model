import re
import unicodedata
import demoji
import neologdn
import pandas as pd
import numpy as np
import MeCab
from scipy.special import gamma, digamma


def remove_url(text:str) -> str:
    t = re.sub(r"(https?|ftp)(:\/\/[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+\$,%#]+)", " ", text)
    return t

def normalization_string(pd_text:pd.DataFrame, colname:str) -> pd.DataFrame:
    for idx in pd_text.index:
        text = pd_text.at[idx, colname]
        if type(text) is not str:
            print(f"type(text) = {type(text)}")
            print("エラー：：文字列型である必要があります。")
            raise
        
        text = unicodedata.normalize("NFKC", text) # UNICODE正規化
        text = neologdn.normalize(text)            # NEOLOGD正規化
        text = remove_url(text)                    # URL削除
        text = demoji.replace(text, ' ')           # 絵文字削除
        text = text.lower()                        # 小文字化
        
        pd_text.at[idx, colname] = text
    
    return pd_text

class Mixture_Of_Unigram_Models_In_EM:
    def __init__(self, train_data, tol:float=1e-6, topic_num:int=10, max_iterate:int=300000, random_state=None) -> None:
        if type(train_data) is pd.core.frame.DataFrame:
            train_data = train_data.to_numpy()
        
        if type(train_data) is list:
            train_data = np.array(train_data)
        
        if type(train_data) is not np.ndarray:
            print(f"type(train_data) = {type(train_data)}")
            print("エラー：：Numpy型である必要があります。")
            raise
        
        if train_data.ndim != 2:
            print(f"train_data dims = {train_data.ndim}")
            print("エラー：：次元数が一致しません。")
            raise
        
        self.train_data  = train_data
        self.tol         = tol
        self.topic_num   = topic_num
        self.max_iterate = max_iterate
        
        self.topic_θ     = np.full(self.topic_num, 0)
        self.W2I         = {}
        
        vocab_count = 0
        word_count  = [{} for _ in range(0, train_data.shape[0])]
        wakati      = MeCab.Tagger("-Owakati")
        for idx in range(0, train_data.shape[0]):
            doc  = train_data[idx][0]
            node = wakati.parseToNode(doc)
            while node:
                word   = node.surface
                hinshi = node.feature.split(',')[0]
                if (word not in self.W2I) and (hinshi == '名詞'):
                    self.W2I[word]  = vocab_count
                    vocab_count    += 1
                
                if (word in word_count[idx]) and (hinshi == '名詞'):
                    word_count[idx][word] += 1
                elif hinshi == '名詞':
                    word_count[idx][word]  = 1
                else:
                    pass
                
                node = node.next
        
        self.word_distr = np.full([self.topic_num, vocab_count], 0)
        self.word_count = np.zeros([len(word_count), vocab_count])
        for idx in range(0, len(word_count)):
            for key, val in word_count[idx].items():
                self.word_count[idx, self.W2I[key]] = val
        
        self.random_state = random_state
        if random_state != None:
            self.random = np.random.default_rng(seed=self.random_state)
        else:
            self.random = np.random.default_rng()
    
    def fit(self):
        doc_num = self.word_count.shape[0]
        θ       = self.random.random(self.topic_θ.shape)
        Φ       = self.random.random(self.word_distr.shape)
        N_doc   = self.word_count
        while True:
            θ_new = np.zeros_like(θ)
            Φ_new = np.zeros_like(Φ)
            for idx_doc in range(0, doc_num):
                # 負担率の計算
                q_dt = [θ[idx] * np.prod(Φ[idx, :] ** N_doc[idx_doc, :]) for idx in range(0, self.topic_num)]
                q_dt = q_dt / np.sum(q_dt)
                
                # トピック分布の計算
                θ_new = θ_new + q_dt
                
                # 単語分布の計算
                Φ_new = Φ_new + np.outer(q_dt, N_doc[idx_doc, :])
            
            # 各分布の正規化
            θ = θ_new / np.sum(θ_new)
            Φ = Φ_new / np.sum(Φ_new, axis=1).reshape(self.topic_num, 1)
            
            # 終了条件
            if np.sum(np.square(self.topic_θ - θ)) + np.sum(np.square(self.word_distr - Φ)) < self.tol:
                break
            
            # 各分布の更新
            self.topic_θ    = θ
            self.word_distr = Φ
        
        # 各分布の更新
        self.topic_θ    = θ
        self.word_distr = Φ

        return True


class Mixture_Of_Unigram_Models_In_VB:
    def __init__(self, train_data, tol:float=1e-6, topic_num:int=10, max_iterate:int=300000, random_state=None) -> None:
        if type(train_data) is pd.core.frame.DataFrame:
            train_data = train_data.to_numpy()
        
        if type(train_data) is list:
            train_data = np.array(train_data)
        
        if type(train_data) is not np.ndarray:
            print(f"type(train_data) = {type(train_data)}")
            print("エラー：：Numpy型である必要があります。")
            raise
        
        if train_data.ndim != 2:
            print(f"train_data dims = {train_data.ndim}")
            print("エラー：：次元数が一致しません。")
            raise
        
        self.train_data  = train_data
        self.tol         = tol
        self.topic_num   = topic_num
        self.max_iterate = max_iterate
        
        self.topic_α     = np.full(self.topic_num, 1 / self.topic_num)
        self.topic_α_k   = np.full(self.topic_num, 1 / self.topic_num)
        self.W2I         = {}
        
        vocab_count = 0
        word_count  = [{} for _ in range(0, train_data.shape[0])]
        wakati      = MeCab.Tagger("-Owakati")
        for idx in range(0, train_data.shape[0]):
            doc  = train_data[idx][0]
            node = wakati.parseToNode(doc)
            while node:
                word   = node.surface
                hinshi = node.feature.split(',')[0]
                if (word not in self.W2I) and (hinshi == '名詞'):
                    self.W2I[word]  = vocab_count
                    vocab_count    += 1
                
                if (word in word_count[idx]) and (hinshi == '名詞'):
                    word_count[idx][word] += 1
                elif hinshi == '名詞':
                    word_count[idx][word]  = 1
                else:
                    pass
                
                node = node.next
        
        self.word_β     = np.full([self.topic_num, vocab_count], 1 / vocab_count)
        self.word_β_k   = np.full([self.topic_num, vocab_count], 1 / vocab_count)
        self.word_count = np.zeros([len(word_count), vocab_count])
        for idx in range(0, len(word_count)):
            for key, val in word_count[idx].items():
                self.word_count[idx, self.W2I[key]] = val
        
        self.random_state = random_state
        if random_state != None:
            self.random = np.random.default_rng(seed=self.random_state)
        else:
            self.random = np.random.default_rng()
    
    def fit(self):
        doc_num        = self.word_count.shape[0]
        N_doc          = self.word_count
        self.topic_α_k = self.random.random(self.topic_α.shape)
        self.word_β_k  = self.random.random(self.word_β.shape)
        while True:
            α_new = self.topic_α.copy()
            β_new = self.word_β.copy()
            for idx_doc in range(0, doc_num):
                def diff_dig_α(idx:int):
                    return digamma(self.topic_α_k[idx]) - digamma(np.sum(self.topic_α_k))
                def diff_dig_β(idx:int):
                    dig1 = np.sum(N_doc[idx_doc, idx2] * digamma(self.word_β_k[idx, idx2]) for idx2 in range(0, N_doc.shape[1]))
                    dig2 = np.sum(N_doc[idx_doc, :]) * digamma(np.sum(self.word_β_k[idx, :]))
                    return dig1 - dig2
                
                # 負担率の計算
                q_dt = [diff_dig_α(idx) + diff_dig_β(idx) for idx in range(0, self.topic_num)]
                q_dt = np.exp(q_dt)
                q_dt = q_dt / np.sum(q_dt)
                
                # トピック分布の計算
                α_new = α_new + q_dt
                
                # 単語分布の計算
                β_new = β_new + np.outer(q_dt, N_doc[idx_doc, :])
            
            # 終了条件
            if np.sum(np.square(self.topic_α_k - α_new)) + np.sum(np.square(self.word_β_k - β_new)) < self.tol:
                break
            
            # 各分布の更新
            self.topic_α_k = α_new
            self.word_β_k  = β_new
        
        # 各分布の更新
        self.topic_α_k = α_new
        self.word_β_k  = β_new

        return True

class Mixture_Of_Unigram_Models_In_CGS:
    def __init__(self, train_data, tol:float=1e-6, topic_num:int=10, max_iterate:int=300000, random_state=None) -> None:
        if type(train_data) is pd.core.frame.DataFrame:
            train_data = train_data.to_numpy()
        
        if type(train_data) is list:
            train_data = np.array(train_data)
        
        if type(train_data) is not np.ndarray:
            print(f"type(train_data) = {type(train_data)}")
            print("エラー：：Numpy型である必要があります。")
            raise
        
        if train_data.ndim != 2:
            print(f"train_data dims = {train_data.ndim}")
            print("エラー：：次元数が一致しません。")
            raise
        
        self.train_data  = train_data
        self.data_num    = train_data.shape[0]
        self.tol         = tol
        self.max_iterate = max_iterate
        
        self.topic_num   = topic_num
        self.topic_α     = 3
        self.topic_k     = np.full(self.topic_num,       0)
        self.z_d         = np.full(train_data.shape[0], -1)
        self.W2I         = {}
        
        vocab_count = 0
        word_dic    = [{} for _ in range(0, self.data_num)]
        wakati      = MeCab.Tagger("-Owakati")
        for idx in range(0, train_data.shape[0]):
            doc  = train_data[idx][0]
            node = wakati.parseToNode(doc)
            while node:
                word   = node.surface
                hinshi = node.feature.split(',')[0]
                if (word not in self.W2I) and (hinshi == '名詞'):
                    self.W2I[word]  = vocab_count
                    vocab_count    += 1
                
                if (word in word_dic[idx]) and (hinshi == '名詞'):
                    word_dic[idx][word] += 1
                elif hinshi == '名詞':
                    word_dic[idx][word]  = 1
                else:
                    pass
                
                node = node.next
        
        self.word_num = vocab_count
        self.word_β   = 3
        self.word_d   = np.zeros([self.data_num,  vocab_count])
        self.word_k   = np.zeros([self.topic_num, vocab_count])
        for idx in range(0, self.data_num):
            for key, val in word_dic[idx].items():
                self.word_d[idx, self.W2I[key]] = val
        
        self.random_state = random_state
        if random_state != None:
            self.random = np.random.default_rng(seed=self.random_state)
        else:
            self.random = np.random.default_rng()
    
    def fit(self):
        doc_n   = self.data_num
        topic_n = self.topic_num
        word_n  = self.word_num
        D_k     = self.topic_k
        N_kv    = self.word_k
        Z_d     = self.z_d
        while True:
            for idx_doc in range(0, doc_n):
                if Z_d[idx_doc] != -1:
                    D_k[Z_d[idx_doc]]     = D_k[Z_d[idx_doc]]     - 1
                    N_kv[Z_d[idx_doc], :] = N_kv[Z_d[idx_doc], :] - self.word_d[idx_doc, :]
                
                # サンプリング確率の計算
                factor_1      = (D_k + self.topic_α) / (np.sum(D_k) + self.topic_α * topic_n)
                factor_2_coef = (np.sum(N_kv, axis=1) + self.word_β * word_n) / (np.sum(N_kv, axis=1) + np.sum(self.word_d[idx_doc, :]) + self.word_β * word_n)
                factor_2_amou = np.where(self.word_d[idx_doc, :] > 0, gamma(N_kv + self.word_d[idx_doc, :] + self.word_β) / gamma(N_kv + self.word_β), 1)
                factor_2      = factor_2_coef * np.prod(factor_2_amou, axis=1)
                factor_2      = factor_2 / np.sum(factor_2)
                categorical   = factor_1 * factor_2
                categorical   = categorical / np.sum(categorical)
                
                if np.any(np.isnan(categorical)):
                    print(categorical)
                
                # トピックのサンプリング
                rnd_c = self.random.multinomial(n=1, pvals=categorical, size=1)
                Z_d[idx_doc] = np.where(rnd_c == 1)[1]
                
                # トピック分布の計算
                D_k[Z_d[idx_doc]]     = D_k[Z_d[idx_doc]]     + 1
                N_kv[Z_d[idx_doc], :] = N_kv[Z_d[idx_doc], :] + self.word_d[idx_doc, :]
            
            # トピックのディリクレ分布のハイパーパラメータの更新
            topic_ratio  = (np.sum(digamma(D_k + self.topic_α)) - topic_n * digamma(self.topic_α)) / (topic_n * digamma(np.sum(D_k) + self.topic_α * topic_n) - topic_n * digamma(self.topic_α * topic_n))
            self.topic_α = self.topic_α * topic_ratio
            
            # 単語のディリクレ分布のハイパーパラメータの更新
            word_ratio  = (np.sum(digamma(N_kv + self.word_β)) - topic_n * word_n * digamma(self.word_β)) / (word_n * np.sum(digamma(np.sum(N_kv, axis=1) + word_n * self.word_β)) - topic_n * word_n * digamma(word_n * self.word_β))
            self.word_β = self.word_β * word_ratio
            
            # 終了条件
            if np.sum(np.square(topic_ratio - 1)) + np.sum(np.square(word_ratio - 1)) < self.tol:
                break

        return True

class LDA_In_EM:
    def __init__(self, train_data:pd.DataFrame, tol:float=1e-6, topic_num:int=10, max_iterate:int=300000, random_state=None) -> None:
        if type(train_data) is list:
            train_data = pd.DataFrame(data=train_data, columns=['text'])
        
        if type(train_data) is not pd.DataFrame:
            print(f"type(train_data) = {type(train_data)}")
            print("エラー：：Pandas DataFrameである必要があります。")
            raise
        
        self.doc_num     = len(train_data)
        self.tol         = tol
        self.topic_num   = topic_num
        self.max_iterate = max_iterate
        
        vocab_count = 0
        word_count  = [{} for _ in range(0, self.doc_num)]
        wakati      = MeCab.Tagger("-Owakati -d /opt/homebrew/lib/mecab/dic/mecab-ipadic-neologd")
        self.W2I    = {}
        self.DI2W   = [{} for _ in range(0, self.doc_num)]
        DEL_IDX     = []
        for idx in range(0, self.doc_num):
            doc  = train_data.iat[idx, 0]
            node = wakati.parseToNode(doc)
            doc_w_count = 0
            while node:
                word   = node.surface
                hinshi = node.feature.split(',')[0]
                if (word not in self.W2I) and (hinshi == '名詞'):
                    self.W2I[word]  = vocab_count
                    vocab_count    += 1
                
                if (word in word_count[idx]) and (hinshi == '名詞'):
                    self.DI2W[idx][doc_w_count] = word
                    word_count[idx][word] += 1
                    doc_w_count += 1
                elif hinshi == '名詞':
                    self.DI2W[idx][doc_w_count] = word
                    word_count[idx][word]  = 1
                    doc_w_count += 1
                else:
                    pass
                
                node = node.next
            
            # 空の文書を登録
            if doc_w_count == 0:
                DEL_IDX.append(idx)
        
        # 空の文書を削除
        self.train_data  = train_data.drop(train_data.index[DEL_IDX]).reset_index(drop=True)
        self.doc_num     = len(self.train_data)
        self.DI2W        = [elem for elem in self.DI2W  if elem != {}]
        word_count       = [elem for elem in word_count if elem != {}]
        
        self.random_state = random_state
        if random_state != None:
            self.random = np.random.default_rng(seed=self.random_state)
        else:
            self.random = np.random.default_rng()
        
        self.vocab_num = vocab_count
        self.doc_w_num = np.array([np.sum([val for val in word_count[idx].values()]) for idx in range(0, self.doc_num)], dtype=int)
        self.topic_θ   = np.array([self.random.random((self.topic_num,)) for _ in range(0, self.doc_num)])
        self.topic_θ   = self.topic_θ / np.sum(self.topic_θ, axis=1).reshape(self.doc_num,   1)
        self.word_Φ    = np.array([self.random.random((self.vocab_num,)) for _ in range(0, self.topic_num)])
        self.word_Φ    = self.word_Φ  / np.sum(self.word_Φ,  axis=1).reshape(self.topic_num, 1)
    
    def fit(self) -> bool:
        # 学習開始
        for idx in range(0, self.max_iterate):
            θ_new = np.zeros_like(self.topic_θ)
            Φ_new = np.zeros_like(self.word_Φ)
            for idx_doc in range(0, self.doc_num):
                for idx_doc_w in range(0, self.doc_w_num[idx_doc]):
                    # 負担率の計算
                    q_dn = self.topic_θ[idx_doc, :] * self.word_Φ[:, self.W2I[self.DI2W[idx_doc][idx_doc_w]]]
                    q_dn = q_dn / np.sum(q_dn)
                    
                    # トピック分布の計算
                    θ_new[idx_doc, :] = θ_new[idx_doc, :] + q_dn
                
                    # 単語分布の計算
                    Φ_new[:, self.W2I[self.DI2W[idx_doc][idx_doc_w]]] = Φ_new[:, self.W2I[self.DI2W[idx_doc][idx_doc_w]]] + q_dn
            
            # 各分布の正規化
            θ = θ_new / np.sum(θ_new, axis=1).reshape(self.doc_num, 1)
            Φ = Φ_new / np.sum(Φ_new, axis=1).reshape(self.topic_num, 1)
            
            # デバッグ出力
            error = np.sum(np.square(self.topic_θ - θ)) + np.sum(np.square(self.word_Φ - Φ))
            if idx % 100 == 0:
                print(f"学習回数：{idx}")
                print(f"誤差：{error}")
                print()
            
            # 終了条件
            if error < self.tol:
                break
            
            # 各分布の更新
            self.topic_θ = θ
            self.word_Φ  = Φ
        
        # 各分布の更新
        self.topic_θ = θ
        self.word_Φ  = Φ

        return True
    
    def stats_info(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        I2W       = {val: key for key, val in self.W2I.items()}
        doc_idx   = [f"文書{i + 1}"           for i in range(0, self.doc_num)]
        topic_idx = [f"トピック{i + 1}"        for i in range(0, self.topic_num)]
        word_idx  = [f"単語{i + 1}:{I2W[i]}"  for i in range(0, self.vocab_num)]
        
        # トピック数とは違い、単語数は事前に把握することができないため四捨五入を行わない
        pd_θ = pd.DataFrame(data=np.round(self.topic_θ, 4), index=doc_idx,   columns=topic_idx)
        pd_Φ = pd.DataFrame(data=self.word_Φ,               index=topic_idx, columns=word_idx)
        
        return pd_θ, pd_Φ

class LDA_In_VB:
    def __init__(self, train_data:pd.DataFrame, tol:float=1e-6, topic_num:int=10, max_iterate:int=300000, random_state=None) -> None:
        if type(train_data) is list:
            train_data = pd.DataFrame(data=train_data, columns=['text'])
        
        if type(train_data) is not pd.DataFrame:
            print(f"type(train_data) = {type(train_data)}")
            print("エラー：：Pandas DataFrameである必要があります。")
            raise
        
        self.train_data  = train_data
        self.doc_num     = len(train_data)
        self.tol         = tol
        self.topic_num   = topic_num
        self.max_iterate = max_iterate
        
        vocab_count = 0
        word_count  = [{} for _ in range(0, self.doc_num)]
        wakati      = MeCab.Tagger("-Owakati -d /opt/homebrew/lib/mecab/dic/mecab-ipadic-neologd")
        self.W2I    = {}
        self.DI2W   = [{} for _ in range(0, self.doc_num)]
        DEL_IDX     = []
        for idx in range(0, self.doc_num):
            doc  = train_data.iat[idx, 0]
            node = wakati.parseToNode(doc)
            doc_w_count = 0
            while node:
                word   = node.surface
                hinshi = node.feature.split(',')[0]
                if (word not in self.W2I) and (hinshi == '名詞'):
                    self.W2I[word]  = vocab_count
                    vocab_count    += 1
                
                if (word in word_count[idx]) and (hinshi == '名詞'):
                    self.DI2W[idx][doc_w_count] = word
                    word_count[idx][word] += 1
                    doc_w_count += 1
                elif hinshi == '名詞':
                    self.DI2W[idx][doc_w_count] = word
                    word_count[idx][word]  = 1
                    doc_w_count += 1
                else:
                    pass
                
                node = node.next
            
            # 空の文書を登録
            if doc_w_count == 0:
                DEL_IDX.append(idx)
        
        # 空の文書を削除
        self.train_data  = train_data.drop(train_data.index[DEL_IDX]).reset_index(drop=True)
        self.doc_num     = len(self.train_data)
        self.DI2W        = [elem for elem in self.DI2W  if elem != {}]
        word_count       = [elem for elem in word_count if elem != {}]
        
        self.random_state = random_state
        if random_state != None:
            self.random = np.random.default_rng(seed=self.random_state)
        else:
            self.random = np.random.default_rng()
        
        self.vocab_num  = vocab_count
        self.doc_w_num  = np.array([np.sum([val for val in word_count[idx].values()]) for idx in range(0, self.doc_num)], dtype=int)
        self.topic_θ_αk = np.array([self.random.random((self.topic_num,)) for _ in range(0, self.doc_num)])
        self.word_Φ_βv  = np.array([self.random.random((self.vocab_num,)) for _ in range(0, self.topic_num)])
        self.topic_θ_α  = 0.001
        self.word_Φ_β   = 0.001
    
    def fit(self) -> bool:
        # 学習開始
        for idx in range(0, self.max_iterate):
            θ_new = np.zeros_like(self.topic_θ_αk) + self.topic_θ_α
            Φ_new = np.zeros_like(self.word_Φ_βv)  + self.word_Φ_β
            for idx_doc in range(0, self.doc_num):
                for idx_doc_w in range(0, self.doc_w_num[idx_doc]):
                    # 負担率の計算
                    avg_qθ = digamma(self.topic_θ_αk[idx_doc]) - digamma(np.sum(self.topic_θ_αk[idx_doc]))
                    avg_qΦ = digamma(self.word_Φ_βv[:, self.W2I[self.DI2W[idx_doc][idx_doc_w]]]) - digamma(np.sum(self.word_Φ_βv, axis=1))
                    q_dn   = np.exp(avg_qθ + avg_qΦ)
                    q_dn   = q_dn / np.sum(q_dn)
                    
                    # トピック分布のハイパーパラメータの計算
                    θ_new[idx_doc, :] = θ_new[idx_doc, :] + q_dn
                
                    # 単語分布のハイパーパラメータの計算
                    Φ_new[:, self.W2I[self.DI2W[idx_doc][idx_doc_w]]] = Φ_new[:, self.W2I[self.DI2W[idx_doc][idx_doc_w]]] + q_dn
            
            # デバッグ出力
            error = np.sum(np.square(self.topic_θ_αk - θ_new)) + np.sum(np.square(self.word_Φ_βv - Φ_new))
            if idx % 100 == 0:
                print(f"学習回数：{idx}")
                print(f"誤差：{error}")
                print()
            
            # 終了条件
            if error < self.tol:
                break
            
            # 各分布の更新
            self.topic_θ_αk = θ_new
            self.word_Φ_βv  = Φ_new
        
        # 各分布の更新
        self.topic_θ_αk = θ_new
        self.word_Φ_βv  = Φ_new

        return True
    
    def stats_info(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        I2W       = {val: key for key, val in self.W2I.items()}
        doc_idx   = [f"文書{i + 1}"           for i in range(0, self.doc_num)]
        topic_idx = [f"トピック分布θ_α{i + 1}"     for i in range(0, self.topic_num)]
        word_idx  = [f"単語分布Φ_β{i + 1}:{I2W[i]}"  for i in range(0, self.vocab_num)]
        
        # 点推定への変換
        topic_θ = (self.topic_θ_αk + self.topic_θ_α) / np.sum(self.topic_θ_αk + self.topic_θ_α, axis=1).reshape(self.doc_num,   1)
        word_Φ  = (self.word_Φ_βv  + self.word_Φ_β)  / np.sum(self.word_Φ_βv  + self.word_Φ_β,  axis=1).reshape(self.topic_num, 1)
        
        # トピック数とは違い、単語数は事前に把握することができないため四捨五入を行わない
        pd_θ = pd.DataFrame(data=np.round(topic_θ, 4), index=doc_idx,   columns=topic_idx)
        pd_Φ = pd.DataFrame(data=word_Φ,               index=topic_idx, columns=word_idx)
        
        return pd_θ, pd_Φ

class LDA_In_CGS: # Collapsed Gibbs Sampling
    def __init__(self, train_data:pd.DataFrame, tol:float=1e-6, topic_num:int=10, max_iterate:int=1000, random_state=None) -> None:
        if type(train_data) is list:
            train_data = pd.DataFrame(data=train_data, columns=['text'])
        
        if type(train_data) is not pd.DataFrame:
            print(f"type(train_data) = {type(train_data)}")
            print("エラー：：Pandas DataFrameである必要があります。")
            raise
        
        self.train_data  = train_data
        self.doc_num     = len(train_data)
        self.tol         = tol
        self.topic_num   = topic_num
        self.max_iterate = max_iterate
        
        vocab_count = 0
        word_count  = [{} for _ in range(0, self.doc_num)]
        wakati      = MeCab.Tagger("-Owakati -d /opt/homebrew/lib/mecab/dic/mecab-ipadic-neologd")
        self.W2I    = {}
        self.DI2W   = [{} for _ in range(0, self.doc_num)]
        DEL_IDX     = []
        for idx in range(0, self.doc_num):
            doc  = train_data.iat[idx, 0]
            node = wakati.parseToNode(doc)
            doc_w_count = 0
            while node:
                word   = node.surface
                hinshi = node.feature.split(',')[0]
                if (word not in self.W2I) and (hinshi == '名詞'):
                    self.W2I[word]  = vocab_count
                    vocab_count    += 1
                
                if (word in word_count[idx]) and (hinshi == '名詞'):
                    self.DI2W[idx][doc_w_count] = word
                    word_count[idx][word] += 1
                    doc_w_count += 1
                elif hinshi == '名詞':
                    self.DI2W[idx][doc_w_count] = word
                    word_count[idx][word]  = 1
                    doc_w_count += 1
                else:
                    pass
                
                node = node.next
        
            # 空の文書を登録
            if doc_w_count == 0:
                DEL_IDX.append(idx)
        
        # 空の文書を削除
        self.train_data  = train_data.drop(train_data.index[DEL_IDX]).reset_index(drop=True)
        self.doc_num     = len(self.train_data)
        self.DI2W        = [elem for elem in self.DI2W  if elem != {}]
        word_count       = [elem for elem in word_count if elem != {}]
        
        self.random_state = random_state
        if random_state != None:
            self.random = np.random.default_rng(seed=self.random_state)
        else:
            self.random = np.random.default_rng()
        
        self.vocab_num  = vocab_count
        self.doc_w_num  = np.array([np.sum([val for val in word_count[idx].values()]) for idx in range(0, self.doc_num)], dtype=int)
        self.N_dk       = np.zeros([self.doc_num, self.topic_num])
        self.N_kv       = np.zeros([self.topic_num, self.vocab_num])
        self.topic_θ_α  = 0.001
        self.word_Φ_β   = 0.001
    
    def fit(self) -> bool:
        # 学習開始
        Z_dn = np.zeros([self.doc_num, self.doc_w_num.max()], dtype=int) - 1
        for idx in range(0, self.max_iterate):
            for idx_doc in range(0, self.doc_num):
                for idx_w_num in range(0, self.doc_w_num[idx_doc]):
                    # 各総量値からサンプリング対象を除外
                    if Z_dn[idx_doc, idx_w_num] != -1:
                        self.N_dk[idx_doc, Z_dn[idx_doc, idx_w_num]] -= 1
                        self.N_kv[Z_dn[idx_doc, idx_w_num], self.W2I[self.DI2W[idx_doc][idx_w_num]]] -= 1
                    
                    # サンプリング確率の計算
                    sampling_prob = (self.N_dk[idx_doc, :] + self.topic_θ_α) * (self.N_kv[:, self.W2I[self.DI2W[idx_doc][idx_w_num]]] + self.word_Φ_β) / (np.sum(self.N_kv, axis=1) + self.vocab_num * self.word_Φ_β)
                    sampling_prob = sampling_prob / np.sum(sampling_prob)
                    
                    # 各単語のトピックのサンプリング
                    rnd_c = self.random.multinomial(n=1, pvals=sampling_prob, size=1)
                    Z_dn[idx_doc, idx_w_num] = np.where(rnd_c == 1)[1]
                    
                    # 各総量値の更新
                    self.N_dk[idx_doc, Z_dn[idx_doc, idx_w_num]] += 1
                    self.N_kv[Z_dn[idx_doc, idx_w_num], self.W2I[self.DI2W[idx_doc][idx_w_num]]] += 1
            
            # トピックのディリクレ分布のハイパーパラメータの更新
            topic_ratio    = (np.sum(digamma(self.N_dk + self.topic_θ_α)) - self.doc_num * self.topic_num * digamma(self.topic_θ_α)) / (self.topic_num * np.sum(digamma(np.sum(self.N_dk, axis=1) + self.topic_num * self.topic_θ_α)) - self.doc_num * self.topic_num * digamma(self.topic_num * self.topic_θ_α))
            self.topic_θ_α = self.topic_θ_α * topic_ratio
            
            # 単語のディリクレ分布のハイパーパラメータの更新
            word_ratio    = (np.sum(digamma(self.N_kv + self.word_Φ_β)) - self.topic_num * self.vocab_num * digamma(self.word_Φ_β)) / (self.vocab_num * np.sum(digamma(np.sum(self.N_kv, axis=1) + self.vocab_num * self.word_Φ_β)) - self.topic_num * self.vocab_num * digamma(self.vocab_num * self.word_Φ_β))
            self.word_Φ_β = self.word_Φ_β * word_ratio
            
            # デバッグ出力
            error = np.sum(np.abs(topic_ratio - 1)) + np.sum(np.abs(word_ratio - 1))
            if idx % 100 == 0:
                print(f"学習回数：{idx}")
                print(f"誤差：{error}")
                print()
            
            # 終了条件
            if error < self.tol:
                break

        return True
    
    def stats_info(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        I2W       = {val: key for key, val in self.W2I.items()}
        doc_idx   = [f"文書{i + 1}"                 for i in range(0, self.doc_num)]
        topic_idx = [f"トピック分布θ_α{i + 1}"        for i in range(0, self.topic_num)]
        word_idx  = [f"単語分布Φ_β{i + 1}:{I2W[i]}"  for i in range(0, self.vocab_num)]
        
        # 点推定への変換
        topic_θ = (self.N_dk + self.topic_θ_α) / np.sum(self.N_dk + self.topic_θ_α, axis=1).reshape(self.doc_num, 1)
        word_Φ  = (self.N_kv + self.word_Φ_β)  / np.sum(self.N_kv + self.word_Φ_β,  axis=1).reshape(self.topic_num, 1)
        
        # トピック数とは違い、単語数は事前に把握することができないため四捨五入を行わない
        pd_θ = pd.DataFrame(data=np.round(topic_θ, 4), index=doc_idx,   columns=topic_idx)
        pd_Φ = pd.DataFrame(data=word_Φ,               index=topic_idx, columns=word_idx)
        
        return pd_θ, pd_Φ