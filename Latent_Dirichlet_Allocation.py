import re
import unicodedata
import emoji
import neologdn
import pandas as pd
import numpy as np
from scipy.special import gamma, digamma, polygamma
from sudachipy import tokenizer
from sudachipy import dictionary


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
        # text = demoji.replace(text, ' ')           # 絵文字削除
        text = emoji.demojize(text)                # 絵文字をテキストに変換
        text = text.lower()                        # 小文字化
        
        pd_text.at[idx, colname] = text
    
    return pd_text

def create_stop_word(pd_text:pd.DataFrame, colname:str, stop_word:list[str], threshold:int=10) -> frozenset[str]:
    tokenizer_obj = dictionary.Dictionary(dict_type='full').create()
    tokenize_mode = tokenizer.Tokenizer.SplitMode.C
    
    word_count = {}
    for idx in pd_text.index:
        text = pd_text.at[idx, colname]
        if type(text) is not str:
            print(f"type(text) = {type(text)}")
            print("エラー：：文字列型である必要があります。")
            raise
        
        tokens = tokenizer_obj.tokenize(text, tokenize_mode)
        for node in tokens:
            word = node.surface()
            
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word]  = 1
    
    custom_list = stop_word + [key for key, val in word_count.items() if val < threshold]
    stop_word   = frozenset(custom_list)
    return stop_word
        

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
            
            # 各分布の更新
            self.topic_θ    = θ
            self.word_distr = Φ
            
            # 終了条件
            if np.sum(np.square(self.topic_θ - θ)) + np.sum(np.square(self.word_distr - Φ)) < self.tol:
                break

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
            
            # 各分布の更新
            self.topic_α_k = α_new
            self.word_β_k  = β_new
            
            # 終了条件
            if np.sum(np.square(self.topic_α_k - α_new)) + np.sum(np.square(self.word_β_k - β_new)) < self.tol:
                break

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
    def __init__(self, train_data:pd.DataFrame, stop_word:frozenset[str], tol:float=1e-6, topic_num:int=10, max_iterate:int=300000, random_state=None) -> None:
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
        self.target_POS  = ['名詞', '動詞', '形容詞', '感動詞', '助動詞', '形状詞']
        
        tokenizer_obj = dictionary.Dictionary(dict_type='full').create()
        tokenize_mode = tokenizer.Tokenizer.SplitMode.C
        
        vocab_count = 0
        word_count  = [{} for _ in range(0, self.doc_num)]
        self.W2I    = {}
        self.DI2W   = [{} for _ in range(0, self.doc_num)]
        DEL_IDX     = []
        for idx in range(0, self.doc_num):
            doc    = train_data.iat[idx, 0]
            tokens = tokenizer_obj.tokenize(doc, tokenize_mode)
            doc_w_count = 0
            for node in tokens:
                word   = node.surface()
                hinshi = node.part_of_speech()[0]
                
                # ストップワードの除去
                if word in stop_word:
                    continue
                
                if (word not in self.W2I) and (hinshi in self.target_POS):
                    self.W2I[word]  = vocab_count
                    vocab_count    += 1
                
                if (word in word_count[idx]) and (hinshi in self.target_POS):
                    self.DI2W[idx][doc_w_count] = word
                    word_count[idx][word] += 1
                    doc_w_count += 1
                elif hinshi in self.target_POS:
                    self.DI2W[idx][doc_w_count] = word
                    word_count[idx][word]  = 1
                    doc_w_count += 1
            
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
            
            # 各分布の更新
            self.topic_θ = θ
            self.word_Φ  = Φ
            
            # 終了条件
            if error < self.tol:
                break

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
    def __init__(self, train_data:pd.DataFrame, stop_word:frozenset[str], tol:float=1e-6, topic_num:int=10, max_iterate:int=300000, random_state=None) -> None:
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
        self.target_POS  = ['名詞', '動詞', '形容詞', '感動詞', '助動詞', '形状詞']
        
        tokenizer_obj = dictionary.Dictionary(dict_type='full').create()
        tokenize_mode = tokenizer.Tokenizer.SplitMode.C
        
        vocab_count = 0
        word_count  = [{} for _ in range(0, self.doc_num)]
        self.W2I    = {}
        self.DI2W   = [{} for _ in range(0, self.doc_num)]
        DEL_IDX     = []
        for idx in range(0, self.doc_num):
            doc    = train_data.iat[idx, 0]
            tokens = tokenizer_obj.tokenize(doc, tokenize_mode)
            doc_w_count = 0
            for node in tokens:
                word   = node.surface()
                hinshi = node.part_of_speech()[0]
                
                # ストップワードの除去
                if word in stop_word:
                    continue
                
                if (word not in self.W2I) and (hinshi in self.target_POS):
                    self.W2I[word]  = vocab_count
                    vocab_count    += 1
                
                if (word in word_count[idx]) and (hinshi in self.target_POS):
                    self.DI2W[idx][doc_w_count] = word
                    word_count[idx][word] += 1
                    doc_w_count += 1
                elif hinshi in self.target_POS:
                    self.DI2W[idx][doc_w_count] = word
                    word_count[idx][word]  = 1
                    doc_w_count += 1
            
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
        self.topic_θ_α  = 1 / self.vocab_num
        self.word_Φ_β   = 1 / self.vocab_num
    
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
            
            # 各分布の更新
            self.topic_θ_αk = θ_new
            self.word_Φ_βv  = Φ_new
            
            # 終了条件
            if error < self.tol:
                break

        return True
    
    def stats_info(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        I2W       = {val: key for key, val in self.W2I.items()}
        doc_idx   = [f"文書{i + 1}"          for i in range(0, self.doc_num)]
        topic_idx = [f"トピック{i + 1}"       for i in range(0, self.topic_num)]
        word_idx  = [f"単語{i + 1}:{I2W[i]}" for i in range(0, self.vocab_num)]
        
        # 点推定への変換
        topic_θ = (self.topic_θ_αk + self.topic_θ_α) / np.sum(self.topic_θ_αk + self.topic_θ_α, axis=1).reshape(self.doc_num,   1)
        word_Φ  = (self.word_Φ_βv  + self.word_Φ_β)  / np.sum(self.word_Φ_βv  + self.word_Φ_β,  axis=1).reshape(self.topic_num, 1)
        
        # トピック数とは違い、単語数は事前に把握することができないため四捨五入を行わない
        pd_θ = pd.DataFrame(data=np.round(topic_θ, 4), index=doc_idx,   columns=topic_idx)
        pd_Φ = pd.DataFrame(data=word_Φ,               index=topic_idx, columns=word_idx)
        
        return pd_θ, pd_Φ

class LDA_In_CGS: # Collapsed Gibbs Sampling
    def __init__(self, train_data:pd.DataFrame, stop_word:frozenset[str], tol:float=1e-6, topic_num:int=10, max_iterate:int=500, random_state=None) -> None:
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
        self.target_POS  = ['名詞', '動詞', '形容詞', '感動詞', '助動詞', '形状詞']
        
        tokenizer_obj = dictionary.Dictionary(dict_type='full').create()
        tokenize_mode = tokenizer.Tokenizer.SplitMode.C
        
        vocab_count = 0
        word_count  = [{} for _ in range(0, self.doc_num)]
        self.W2I    = {}
        self.DI2W   = [{} for _ in range(0, self.doc_num)]
        DEL_IDX     = []
        for idx in range(0, self.doc_num):
            doc    = train_data.iat[idx, 0]
            tokens = tokenizer_obj.tokenize(doc, tokenize_mode)
            doc_w_count = 0
            for node in tokens:
                word   = node.surface()
                hinshi = node.part_of_speech()[0]
                
                # ストップワードの除去
                if word in stop_word:
                    continue
                
                if (word not in self.W2I) and (hinshi in self.target_POS):
                    self.W2I[word]  = vocab_count
                    vocab_count    += 1
                
                if (word in word_count[idx]) and (hinshi in self.target_POS):
                    self.DI2W[idx][doc_w_count] = word
                    word_count[idx][word] += 1
                    doc_w_count += 1
                elif hinshi in self.target_POS:
                    self.DI2W[idx][doc_w_count] = word
                    word_count[idx][word]  = 1
                    doc_w_count += 1
        
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
        self.topic_θ_α  = np.array([1 / self.vocab_num for _ in range(0, self.topic_num)])
        self.word_Φ_β   = 1 / self.vocab_num
    
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
            topic_ratio    = (np.sum(digamma(self.N_dk + self.topic_θ_α), axis=0) - self.doc_num * digamma(self.topic_θ_α)) / (np.sum(digamma(np.sum(self.N_dk + self.topic_θ_α, axis=1))) - self.doc_num * digamma(np.sum(self.topic_θ_α)))
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
        doc_idx   = [f"文書{i + 1}"          for i in range(0, self.doc_num)]
        topic_idx = [f"トピック{i + 1}"       for i in range(0, self.topic_num)]
        word_idx  = [f"単語{i + 1}:{I2W[i]}" for i in range(0, self.vocab_num)]
        
        # 点推定への変換
        topic_θ = (self.N_dk + self.topic_θ_α) / np.sum(self.N_dk + self.topic_θ_α, axis=1).reshape(self.doc_num, 1)
        word_Φ  = (self.N_kv + self.word_Φ_β)  / np.sum(self.N_kv + self.word_Φ_β,  axis=1).reshape(self.topic_num, 1)
        
        # トピック数とは違い、単語数は事前に把握することができないため四捨五入を行わない
        pd_θ = pd.DataFrame(data=np.round(topic_θ, 4), index=doc_idx,   columns=topic_idx)
        pd_Φ = pd.DataFrame(data=word_Φ,               index=topic_idx, columns=word_idx)
        
        return pd_θ, pd_Φ


# X(旧:Twitter) のポスト(旧:ツイート)から感情分析を行うモデルを構築する
# 解析する感情ラベルは具体的に「喜び, 悲しみ, 期待, 驚き, 怒り, 恐れ, 嫌悪, 信頼」の8つのラベルを想定している
# このモデルはノイズあり対応トピックモデルの亜種である

# 生成過程は下記の通り
# 1. ハイパーパラメータの設定
#   (a) α:Float64 定数
#   (b) β:Float64 定数
#   (c) γ:Float64 定数
#   (d) η:[Float64, Float64] 定数
#   (e) ν:array[Float64] 所与の定数
#   (f) C:array[Float64] 所与の定数
# 2. 一般単語分布 φ_0 〜 Dirichlet(β)
# 3. 関係性分布 λ 〜 Beta(η)
# 4. For トピック k = 1, ... K
#       (a) 感情分布 ψ_k 〜 Dirichlet(γ)
# 5. For 感情トピック l = 1, ... 8
#       (a) 感情単語分布 φ_l 〜 Dirichlet(β)
# 6. For 文書 d = 1, ... D
#       (a) トピック分布 θ_d 〜 Dirichlet(α)
#       (b) For 感情サンプル m = 1, ... M_d
#               i.  トピックラベル y_dm 〜 Categorical(θ_d)
#               ii. 感情分布 x_dm 〜 Dirichlet(ψ_(y_dm))
#       (c) 文書感情尤度 ν_d 〜 Nagino(Σ_k θ_dk ψ_k, C)
#       (d) 文書感情分布 x_d = x_d1 * x_d2 * ... * x_d(M_d)
#       (e) For 各単語 n = 1, ... N_d
#               i.   感情トピック z_dn 〜 Categorical(x_d)
#               ii.  関係性 r_dn 〜 Bernoulli(λ)
#               iii. 単語 w_dn 〜 Categorical(φ_(z_dn)) if r_dn == 1 else Categorical(φ_0)

# この感情分析モデルを構築するにあたり、新しい確率分布を定義した
# Σ_i p_i = 1
# C_i : 非負の実数
# x_i ∈ [0, C_i]
# Nagino(x | p, C) = Π_i (p_i / C_i^(p_i)) x_i^(p_i - 1)
# この確率分布はベータ分布の多変量への拡張である。そのような確率分布としてはディリクレ分布があげられる。
# ディリクレ分布と似ている部分も多く見られるが、制約条件的な部分で違いがある
# この確率分布の期待値と分散を以下に示す
# E(x_i) = (p_i / (1 + p_i)) C_i
# V(x_i) = (p_i / ((1 + p_i)^2 (2 + p_i))) C_i^2

# 感情分析用のトピックモデルの先行研究例として以下のようなトピックモデルが存在する
# ・SLDA(Supervised Latent Dirichlet Allocation)
#       LDAモデルの拡張
#       特徴的な処理として、各文書ごとのトピック割り当て分布を算出する
#       トピック割り当て分布を説明変数にして特定の感情ラベル(目的変数)に線形回帰する
# ・JST(Joint Sentiment Topic)
#       LDAモデルの拡張
#       特徴的な処理として、一つの文書に対して以下の仮定を行う
#           ・各単語は単一の感情ラベルを持つ
#           ・各単語は単一のトピックラベルを持つ
#           ・各トピックラベルは単一の感情ラベルと関連する
#           ・各感情ラベルは他の感情ラベルと重複しないように同数のトピックラベルを持つ

# これらの先行研究モデルにはそれぞれ以下のような弱点が存在する
# ・SLDA
#   結合トピックモデルと同じ弱点を持ちうる
#   すなわち、感情ラベルと単語分布の間に関連性がない
#   LDAが切り分けるトピックの基準によっては、感情ラベルと相反する単語が関連づけられる可能性がある
# ・JST
#   必要メモリ量が多い
#   各トピックが単一の感情しか表現できない

# これらの先行研究に比較して実装モデルは以下の特徴を持つ
# ・対応トピックモデルの亜種であるため、感情ラベルと単語分布の間に関連性がある
# ・ノイズありモデルの亜種であるため、感情ラベルと直接関係ない単語分布を推定できる
# ・JSTモデルと比較して、一つの文書に対して以下の仮定を行う
#       ・各文書は単一の感情分布を持つ
#       ・各文書は複数のトピックラベルを持つ
#       ・各単語は複数のトピックラベルを持つ
#       ・各単語は単一の感情トピックを持つ
#       ・各トピックは複数の感情トピックと関連する
#       ・各感情トピックは複数のトピックと関連する
#       ・各トピックは他のトピックと感情トピックが重複することを許す
#       ・各感情トピックは他の感情トピックとトピックが重複することを許す
# ・必要メモリ量がJSTモデルと比較して少ない

# 変分対象のパラメータがそれぞれ以下のような独立性を持つと仮定する
#   q(R, Z, X, Y, θ, Φ, Ψ, Λ) = q(R, X, θ)q(Y, Z, Λ)q(Ψ, Φ)

# ベイズ自由エネルギー最適化を以下の数式を最小化することによって行う
# Ω = {R, X, θ}
# Ξ = {Y, Z, Λ}
# χ = {Ψ, Φ}
# F = ∫∫∫∫ Σ_R Σ_Z Σ_Y q(R, Z, X, Y, θ, Φ, Ψ, Λ) log(p(W, ν, R, Z, X, Y, θ, Φ, Ψ, Λ) / q(R, Z, X, Y, θ, Φ, Ψ, Λ)) dXdθdΦdΨdΛ
#   = ∫∫∫ q(Ω)q(Ξ)q(χ) log(p(W, ν, Ω, Ξ, χ) / q(Ω)q(Ξ)q(χ)) dΩdΞdχ
#   = ∫∫∫ q(Ω)q(Ξ)q(χ) {logp(W, ν, Ω, Ξ, χ) - logq(Ω) - logq(Ξ) - logq(χ)} dΩdΞdχ

# 参考：
# 平均値の算出式が離散な分布：Bernoulli、Categorical
# 平均値の算出式が連続な分布：Beta、Dirichlet、Nagino

# p(W, ν, Ω, Ξ, χ)
# = p(W, ν, R, Z, X, Y, θ, Φ, Ψ, Λ)
# = p(W | Z, R, Φ) p(Z | X) p(R | Λ) p(Λ | η) p(ν | θ, Ψ) p(X | Y, Ψ) p(Y | θ) p(θ | α) p(Ψ | γ) p(Φ | β)

# logq(Ω) ∝ E_q(Ξ)q(χ)[logp(W, Ω, Ξ, χ)]
# logq(Ξ) ∝ E_q(Ω)q(χ)[logp(W, Ω, Ξ, χ)]
# logq(χ) ∝ E_q(Ω)q(Ξ)[logp(W, Ω, Ξ, χ)]

# logq(Ω) ∝ E_q(Ξ)q(χ)[logp(W | Z, R, Φ) p(R | Λ)] + E_q(Ξ)q(χ)[logp(Z | X) p(X | Y, Ψ)] + E_q(Ξ)q(χ)[logp(ν | θ, Ψ) p(Y | θ) p(θ | α)]
#         = E_q(Z, Λ)q(Φ)[logp(W | Z, R, Φ) p(R | Λ)] + E_q(Y, Z)q(Ψ)[logp(Z | X) p(X | Y, Ψ)] + E_q(Y)q(Ψ)[logp(ν | θ, Ψ) p(Y | θ) p(θ | α)]
#         = logq(R) + logq(X) + logq(θ)
# logq(Ξ) ∝ E_q(Ω)q(χ)[logp(X | Y, Ψ) p(Y | θ)] + E_q(Ω)q(χ)[logp(W | Z, R, Φ) p(Z | X)] + E_q(Ω)q(χ)[logp(R | Λ) p(Λ | η)]
#         = E_q(X, θ)q(Ψ)[logp(X | Y, Ψ) p(Y | θ)] + E_q(R, X)q(Φ)[logp(W | Z, R, Φ) p(Z | X)] + E_q(R)[logp(R | Λ) p(Λ | η)]
#         = logq(Y) + logq(Z) + logq(Λ)
# logq(χ) ∝ E_q(Ω)q(Ξ)[logp(ν | θ, Ψ) p(X | Y, Ψ) p(Ψ | γ)] + E_q(Ω)q(Ξ)[logp(W | Z, R, Φ) p(Φ | β)]
#         = E_q(X, θ)q(Y)[logp(ν | θ, Ψ) p(X | Y, Ψ) p(Ψ | γ)] + E_q(R)q(Z)[logp(W | Z, R, Φ) p(Φ | β)]
#         = logq(Ψ) + logq(Φ)

# q(Λ) ∝ Beta((Σ_d Σ_v q_dv + η[0] - 1) + 1, (Σ_d Σ_v η[1] - q_dv) + 1)  q_dv 〜 q(R)
# サイズ : 2
# 形状  : ベータ分布
# 連続確率分布

# q(Φ) ∝ Dirichlet_0((Σ_d Σ_n:(v=W_dn) (1 - q_dv)) + β)  q_dv 〜 q(R)
    #    Dirichlet_l((Σ_d Σ_n:(v=W_dn) q_dv q_dnl) + β)  q_dv 〜 q(R)  q_dnl 〜 q(Z)
# サイズ : (1 + 単語トピック数L) × 語彙数V
# 形状  : ディリクレ分布
# 連続確率分布

# q(R=1) ∝ exp(digamma(q_a) - digamma(q_a + q_b)) exp(Σ_n:(v=W_dn) Σ_l q_dnl (digamma(q_lv) - digamma(Σ_v q_lv)))   q_a, q_b 〜 q(Λ)  q_dnl 〜 q(Z)  q_lv 〜 q(Φ)
# q(R=0) ∝ exp(digamma(q_b) - digamma(q_a + q_b)) exp(Σ_n:(v=W_dn) (digamma(q_0v) - digamma(Σ_v q_0v)))   q_a, q_b 〜 q(Λ)  q_dnl 〜 q(Z)  q_0v 〜 q(Φ0)
# サイズ : 文書数D × 語彙数V
# 形状  : 不明
# 離散確率分布

# q(X) ∝ Dirichlet(X_dm | Σ_n q_dnl + Σ_k q_dmk {q_kl / Σ_l q_kl - 1} + 1)  q_dnl 〜 q(Z)  q_dmk 〜 q(Y)  q_kl 〜 q(Ψ)
# サイズ : 文書数D × 感情サンプル数M_d × 単語トピック数L
# 形状  : ディリクレ分布
# 連続確率分布

# q(Ψ) ∝ exp(Σ_k Σ_l Σ_d Σ_m q_dmk ψ_kl (digamma(q_dml) - digamma(Σ_l q_dml))) exp(Σ_k Σ_l Σ_d (q_dk / (Σ_k q_dk)) ψ_kl logν_dl) exp(Σ_k Σ_l logψ_kl^(γ - 1))  q_dmk 〜 q(Y)  q_dml 〜 q(X)  q_dk 〜 q(θ)
# ブラックボックス変分推定 対象関数
# δF / δq_kl = Σ_d Σ_m q_dmk (1 - q_kl / (Σ_l q_kl)) / (Σ_l q_kl) (digamma(q_dml) - digamma(Σ_l q_dml)) - Σ_(l'≠l) q_dmk (q_kl' / (Σ_l q_kl) ** 2) (digamma(q_dml') - digamma(Σ_l q_dml'))
            # + Σ_d q_dk / (Σ_k q_dk) logν_dl (1 - q_kl / (Σ_l q_kl)) / (Σ_l q_kl) - Σ_(l\'≠l) q_dk / (Σ_k q_dk) logν_dl' (q_kl' / (Σ_l q_kl) ** 2)
            # + (γ - q_kl) (polygamma(1, q_kl) - polygamma(1, Σ_l q_kl)) - (digamma(q_kl) - digamma(Σ_l q_kl))
            #   q_dk 〜 q(θ)  q_dmk 〜 q(Y)  q_dml 〜 q(X)  q_kl 〜 q(Ψ)
# サイズ : 感情トピック数Κ × 単語トピック数L
# 形状  : ディリクレ分布
# 連続確率分布

# q(θ) ∝ exp(Σ_d Σ_k Σ_l K θ_dk (q_kl / (Σ_l q_kl)) logν_dl) exp(Σ_d Σ_k logθ_dk^{Σ_m q_dmk + α - 1})  q_kl 〜 q(Ψ)  q_dmk 〜 q(Y)
# ブラックボックス変分推定 対象関数
# δlogF / δq_dk = Σ_l K (1 - q_dk / (Σ_k q_dk)) / (Σ_k q_dk) q_kl / (Σ_l q_kl) logν_dl - Σ_l Σ_k\'≠k K q_dk' / ((Σ_k q_dk) ** 2) q_k'l / (Σ_l q_k'l) logν_dl
            #    + (Σ_m q_dmk + α - q_dk) (polygamma(1, q_dk) - polygamma(1, Σ_k q_dk)) - (digamma(q_dk) - digamma(Σ_k q_dk))
                #  q_kl 〜 q(Ψ)  q_dmk 〜 q(Y)  q_dk 〜 q(θ)
# サイズ : 文書数D × 感情トピック数Κ
# 形状  : ディリクレ分布
# 連続確率分布

# q(Y) ∝ exp(digamma(q_dk) - digamma(Σ_k q_dk)) exp(Σ_l (q_kl / (Σ_l q_kl) - 1) (digamma(q_dml) - digamma(Σ_l q_dml)))  q_dk 〜 q(θ)  q_kl 〜 q(Ψ)  q_dml 〜 q(X)
# サイズ : 文書数D × 感情サンプル数M_d × 感情トピック数Κ
# 形状  : 不明
# 離散確率分布

# q(Z) ∝ exp(Σ_m digamma(q_dml) - digamma(Σ_l q_dml)) exp(q_d(w_dn) (digamma(q_l(w_dn)) - digamma(Σ_v q_lv)) + (1 - q_d(w_dn)) (digamma(q_0(w_dn)) - digamma(Σ_v q_0v)))  q_dml 〜 q(X)  q_dv 〜 q(R)  q_lv 〜 q(Φ)
# サイズ : 文書数D × 単語数N_d × 単語トピック数L
# 形状  : 不明
# 離散確率分布


class Harmonized_Sentiment_Topic_Model_In_VB:
    def __init__(self, train_data:pd.DataFrame, train_labels:pd.DataFrame, stop_word:frozenset[str], label_max_value:float=4, tol:float=1e-6, topic_num:int=10, max_iterate:int=300000, random_state=None) -> None:
        if type(train_data) is list:
            train_data = pd.DataFrame(data=train_data, columns=['text'])
        
        if type(train_data) is not pd.DataFrame:
            print(f"type(train_data) = {type(train_data)}")
            print("エラー：：Pandas DataFrameである必要があります。")
            raise
        
        if type(train_labels) is not pd.DataFrame:
            print(f"type(train_labels) = {type(train_labels)}")
            print("エラー：：Pandas DataFrameである必要があります。")
            raise
        
        # index部分の初期化
        train_data   = train_data.reset_index(drop=True)
        train_labels = train_labels.reset_index(drop=True)
        
        # ラベルデータに対する前処理
        train_labels[train_labels < 0] = 0
        train_labels[train_labels > label_max_value] = label_max_value
        
        if len(train_data) != len(train_labels):
            print(f"len(train_data)   = {len(train_data)}")
            print(f"len(train_labels) = {len(train_labels)}")
            print("エラー：：与えられたデータ列に誤りがあります。")
            raise
        
        # DataFrameの整形
        train_data = train_data['text']
        train_data = pd.concat([train_data, train_labels], axis=1)
        
        self.train_data  = train_data
        self.doc_num     = len(train_data)
        self.label_num   = len(train_labels.columns)
        self.label       = train_labels.columns
        self.stop_word   = stop_word
        self.label_max   = label_max_value
        self.tol         = tol
        self.topic_num   = topic_num
        self.max_iterate = max_iterate
        self.target_POS  = ['名詞', '動詞', '形容詞', '感動詞', '助動詞', '形状詞']
        
        tokenizer_obj = dictionary.Dictionary(dict_type='full').create()
        tokenize_mode = tokenizer.Tokenizer.SplitMode.C
        
        vocab_count = 0
        self.W2I    = {}
        self.DI2W   = [{} for _ in range(0, self.doc_num)]
        self.DW2I   = [{} for _ in range(0, self.doc_num)]
        self.DCNT   = [{} for _ in range(0, self.doc_num)]
        self.DLBL   = []
        DEL_IDX     = []
        for idx in range(0, self.doc_num):
            doc    = train_data.iat[idx, 'text']
            tokens = tokenizer_obj.tokenize(doc, tokenize_mode)
            doc_w_count = 0
            for node in tokens:
                word   = node.surface()
                hinshi = node.part_of_speech()[0]
                
                # ストップワードの除去
                if word in stop_word:
                    continue
                
                if (word not in self.W2I) and (hinshi in self.target_POS):
                    self.W2I[word]  = vocab_count
                    vocab_count    += 1
                
                if (word in self.DCNT[idx]) and (hinshi in self.target_POS):
                    self.DI2W[idx][doc_w_count] = word
                    self.DW2I[idx][word].append(doc_w_count)
                    self.DCNT[idx][word] += 1
                    doc_w_count += 1
                elif hinshi in self.target_POS:
                    self.DI2W[idx][doc_w_count] = word
                    self.DW2I[idx][word] = [doc_w_count]
                    self.DCNT[idx][word] = 1
                    doc_w_count += 1
            
            # 空の文書を登録
            if doc_w_count == 0:
                DEL_IDX.append(idx)
            else:
                tmp = [train_data.iat[idx, lbl] for lbl in self.label]
                self.DLBL.append(tmp)
        
        # 空の文書を削除
        self.train_data = train_data.drop(train_data.index[DEL_IDX]).reset_index(drop=True)
        self.doc_num    = len(self.train_data)
        self.DI2W       = [elem for elem in self.DI2W if elem != {}]
        self.DW2I       = [elem for elem in self.DW2I if elem != {}]
        self.DCNT       = [elem for elem in self.DCNT if elem != {}]
        
        # 解析結果の保存
        self.vocab_num  = vocab_count
        self.doc_v_num  = np.array([len(self.DCNT[idx])                   for idx in range(0, self.doc_num)], dtype=int)
        self.doc_w_num  = np.array([np.sum(list(self.DCNT[idx].values())) for idx in range(0, self.doc_num)], dtype=int)
        self.relat_R_η  = 1
        self.topic_θ_α  = 1 / self.vocab_num
        self.word_Φ_β   = 1 / self.vocab_num
        self.senti_Ψ_γ  = 1 / self.vocab_num
        
        # 乱数の設定
        self.random_state = random_state
        if random_state != None:
            self.random = np.random.default_rng(seed=self.random_state)
        else:
            self.random = np.random.default_rng()
        
        # 各種変分事後分布の初期化
        self.Λ  = self.random.random(2)
        self.Φ0 = self.random.random(size=(1,              self.vocab_num))
        self.Φ1 = self.random.random(size=(self.label_num, self.vocab_num))
        self.Ψ  = self.random.random(size=(self.topic_num, self.label_num))
        self.θ  = self.random.random(size=(self.doc_num,   self.topic_num))
        self.Y  = self.random.random(size=(self.doc_num, max(self.doc_w_num), self.topic_num))
        self.X  = self.random.random(size=(self.doc_num, max(self.doc_w_num), self.label_num))
        self.Z  = self.random.random(size=(self.doc_num, max(self.doc_w_num), self.label_num))
        self.R  = self.random.random(size=(self.doc_num, self.vocab_num))
        
        # 離散分布の正規化
        self.Y = self.Y / np.sum(self.Y)
        self.Z = self.Z / np.sum(self.Z)
        self.R = self.R / (self.doc_num * self.vocab_num)
    
    def fit(self) -> bool:
        # 学習開始
        for idx in range(0, self.max_iterate):
            # 初期化
            Λ_new  = np.zeros_like(self.Λ)
            Φ0_new = np.zeros_like(self.Φ0) + self.word_Φ_β
            Φ1_new = np.zeros_like(self.Φ1) + self.word_Φ_β
            Ψ_new  = np.zeros_like(self.Ψ)
            θ_new  = np.zeros_like(self.θ)
            Y_new  = np.zeros_like(self.Y)
            X_new  = np.zeros_like(self.X)
            Z_new  = np.zeros_like(self.Z)
            R_new  = np.zeros_like(self.R)
            
            # q(Λ) ∝ Beta((Σ_d Σ_v q_dv + η[0] - 1) + 1, (Σ_d Σ_v η[1] - q_dv) + 1)  q_dv 〜 q(R)
            Λ_new[0] = np.sum([ self.R[d, self.W2I[self.DI2W[d][doc_idx]]] + self.relat_R_η - 1 for d in range(0, self.doc_num) for doc_idx in range(0, self.doc_v_num[d])]) + 1
            Λ_new[1] = np.sum([-self.R[d, self.W2I[self.DI2W[d][doc_idx]]] + self.relat_R_η     for d in range(0, self.doc_num) for doc_idx in range(0, self.doc_v_num[d])]) + 1
            
            # 文書dごとにループ
            for idx_doc in range(0, self.doc_num):
                # q(R=1) ∝ exp(digamma(q_a) - digamma(q_a + q_b)) exp(Σ_n:(v=W_dn) Σ_l q_dnl (digamma(q_lv) - digamma(Σ_v q_lv)))   q_a, q_b 〜 q(Λ)  q_dnl 〜 q(Z)  q_lv 〜 q(Φ)
                # q(R=0) ∝ exp(digamma(q_b) - digamma(q_a + q_b)) exp(Σ_n:(v=W_dn) (digamma(q_0v) - digamma(Σ_v q_0v)))   q_a, q_b 〜 q(Λ)  q_dnl 〜 q(Z)  q_0v 〜 q(Φ0)
                for word in self.DW2I[idx_doc]:
                    q_1Λ  = digamma(Λ_new[0]) - digamma(Λ_new[0] + Λ_new[1])
                    q_1ZΦ = np.sum([self.Z[idx_doc, n, l] * (digamma(self.Φ1[l, self.W2I[word]]) - digamma(np.sum(self.Φ1, axis=1))) for n in self.DW2I[idx_doc][word] for l in range(0, self.label_num)])
                    
                    q_0Λ  = digamma(Λ_new[1]) - digamma(Λ_new[0] + Λ_new[1])
                    q_0ZΦ = np.sum((digamma(self.Φ0[0, self.W2I[word]]) - digamma(np.sum(self.Φ0, axis=1))) * len(self.DW2I[idx_doc][word]))
                    
                    R_new[idx_doc, self.W2I[word]] = np.exp(q_1Λ + q_1ZΦ) / (np.exp(q_1Λ + q_1ZΦ) + np.exp(q_0Λ + q_0ZΦ))
                
                # 感情サンプルごとにループ
                # 感情サンプル数は文書dごとの語彙数と等しいとする
                for idx_doc_s in range(0, self.doc_v_num[idx_doc]):
                    # q(Y) ∝ exp(digamma(q_dk) - digamma(Σ_k q_dk)) exp(Σ_l (q_kl / (Σ_l q_kl) - 1) (digamma(q_dml) - digamma(Σ_l q_dml)))  q_dk 〜 q(θ)  q_kl 〜 q(Ψ)  q_dml 〜 q(X)
                    for idx_doc_k in range(0, self.topic_num):
                        q_Ψ = self.Ψ[idx_doc_k, :] / np.sum(self.Ψ[idx_doc_k, :])
                        q_X = digamma(self.X[idx_doc, idx_doc_s, :]) - digamma(np.sum(self.X[idx_doc, idx_doc_s, :]))
                        q_θ = digamma(self.θ[idx_doc, idx_doc_k]) - digamma(np.sum(self.θ[idx_doc, :]))
                        
                        Y_new[idx_doc, idx_doc_s, idx_doc_k] = np.exp(np.sum(q_X * q_Ψ) + q_θ)
                    Y_new[idx_doc, idx_doc_s, :] = Y_new[idx_doc, idx_doc_s, :] / np.sum(Y_new[idx_doc, idx_doc_s, :])
                    
                    
                    # q(Ψ) ∝ exp(Σ_k Σ_l Σ_d Σ_m q_dmk ψ_kl (digamma(q_dml) - digamma(Σ_l q_dml))) exp(Σ_k Σ_l Σ_d (q_dk / (Σ_k q_dk)) ψ_kl logν_dl) exp(Σ_k Σ_l logψ_kl^(γ - 1))  q_dmk 〜 q(Y)  q_dml 〜 q(X)  q_dk 〜 q(θ)
                    # ブラックボックス変分推定 対象関数
                    # δF / δq_kl = Σ_d Σ_m q_dmk (1 - q_kl / (Σ_l q_kl)) / (Σ_l q_kl) (digamma(q_dml) - digamma(Σ_l q_dml)) - Σ_(l'≠l) q_dmk (q_kl' / (Σ_l q_kl) ** 2) (digamma(q_dml') - digamma(Σ_l q_dml'))
                                # + Σ_d q_dk / (Σ_k q_dk) logν_dl (1 - q_kl / (Σ_l q_kl)) / (Σ_l q_kl) - Σ_(l\'≠l) q_dk / (Σ_k q_dk) logν_dl' (q_kl' / (Σ_l q_kl) ** 2)
                                # + (γ - q_kl) (polygamma(1, q_kl) - polygamma(1, Σ_l q_kl)) - (digamma(q_kl) - digamma(Σ_l q_kl))
                                #   q_dk 〜 q(θ)  q_dmk 〜 q(Y)  q_dml 〜 q(X)  q_kl 〜 q(Ψ)
                    # サイズ : 感情トピック数Κ × 単語トピック数L
                    # 形状  : ディリクレ分布
                    # 連続確率分布
                    for idx in range(0, 500):
                        q_xy = np.zeros(size=(self.topic_num, self.label_num))
                        q_θ  = np.zeros(size=(self.topic_num, self.label_num))
                        q_p  = np.zeros(size=(self.topic_num, self.label_num))
                        for k in range(0, self.topic_num):
                            for l in range(0, self.label_num):
                                for d in range(0, self.doc_num):
                                    for m in range(0, self.doc_v_num[d]):
                                        q_xy[k, l] += self.Y[d, m, k] * (digamma(self.X[d, m, l]) - digamma(np.sum(self.X[d, m, :])))
                                    
                                    q_θ[k, l] += self.θ[d, k] / np.sum(self.θ[d, :]) * np.log(self.LBL[d, l])
                                
                                q_p[k, l] = (self.senti_Ψ_γ - self.Ψ[k, l]) * (polygamma(1, self.Ψ[k, l]) - polygamma(1, np.sum(self.Ψ[k, :]))) - (digamma(self.Ψ[k, l]) - digamma(np.sum(self.Ψ[k, :])))
                        q_diff = q_xy + q_θ
                        q_diff = q_diff * (1 - self.Ψ / np.sum(self.Ψ, axis=1)) / np.sum(self.Ψ, axis=1) - q_diff * self.Ψ / (np.sum(self.Ψ, axis=1) ** 2) + np.sum(q_diff * self.Ψ / (np.sum(self.Ψ, axis=1) ** 2), axis=1)
                        q_diff = q_diff + q_p
                        self.Ψ += 0.001 * q_diff
                                

                    # q(θ) ∝ exp(Σ_d Σ_k Σ_l K θ_dk (q_kl / (Σ_l q_kl)) logν_dl) exp(Σ_d Σ_k logθ_dk^{Σ_m q_dmk + α - 1})  q_kl 〜 q(Ψ)  q_dmk 〜 q(Y)
                    # ブラックボックス変分推定 対象関数
                    # δlogF / δq_dk = Σ_l K (1 - q_dk / (Σ_k q_dk)) / (Σ_k q_dk) q_kl / (Σ_l q_kl) logν_dl - Σ_l Σ_k\'≠k K q_dk' / ((Σ_k q_dk) ** 2) q_k'l / (Σ_l q_k'l) logν_dl
                                #    + (Σ_m q_dmk + α - q_dk) (polygamma(1, q_dk) - polygamma(1, Σ_k q_dk)) - (digamma(q_dk) - digamma(Σ_k q_dk))
                                    #  q_kl 〜 q(Ψ)  q_dmk 〜 q(Y)  q_dk 〜 q(θ)
                    # サイズ : 文書数D × 感情トピック数Κ
                    # 形状  : ディリクレ分布
                    # 連続確率分布
                    for idx in range(0, 500):
                        q_ψ = np.zeros(size=(self.doc_num, self.topic_num))
                        q_y = np.zeros(size=(self.doc_num, self.topic_num))
                        for d in range(0, self.doc_num):
                            for k in range(0, self.topic_num):
                                for l in range(0, self.label_num):
                                    q_ψ[d, k] += self.topic_num * self.Ψ[k, l] / np.sum(self.Ψ[k, :]) * np.log(self.LBL[d, l])
                                
                            q_y[d, k] = (np.sum(self.Y[d, :, k]) + self.topic_θ_α - self.θ[d, k]) * (polygamma(1, self.θ[d, k]) - polygamma(1, np.sum(self.θ[d, :]))) - (digamma(self.θ[d, k]) - digamma(np.sum(self.θ[d, :])))
                        q_diff = q_ψ
                        q_diff = q_diff * (1 - self.θ / np.sum(self.θ, axis=1)) / np.sum(self.θ, axis=1) - q_diff * self.θ / (np.sum(self.θ, axis=1) ** 2) + np.sum(q_diff * self.θ / (np.sum(self.θ, axis=1) ** 2), axis=1)
                        q_diff = q_diff + q_y
                        self.θ += 0.001 * q_diff
                    
                    # q(X) ∝ Dirichlet(X_dm | Σ_n q_dnl + Σ_k q_dmk {q_kl / Σ_l q_kl - 1} + 1)  q_dnl 〜 q(Z)  q_dmk 〜 q(Y)  q_kl 〜 q(Ψ)
                    q_Z  = np.sum(self.Z[idx_doc, :, :], axis=0)
                    q_YΨ = np.sum([Y_new[idx_doc, idx_doc_s, k] * (Ψ_new[k, :] / np.sum(Ψ_new[k, :]) - 1) for k in range(0, self.topic_num)], axis=0)
                    X_new[idx_doc, idx_doc_s, :] = q_Z + q_YΨ
                    X_new[idx_doc, idx_doc_s, :] = X_new[idx_doc, idx_doc_s, :] / np.sum(X_new[idx_doc, idx_doc_s, :])
                
                # 単語ごとにループ
                for idx_doc_w in range(0, self.doc_w_num[idx_doc]):
                    # q(Z) ∝ exp(Σ_m digamma(q_dml) - digamma(Σ_l q_dml)) exp(q_d(w_dn) (digamma(q_l(w_dn)) - digamma(Σ_v q_lv)) + (1 - q_d(w_dn)) (digamma(q_0(w_dn)) - digamma(Σ_v q_0v)))  q_dml 〜 q(X)  q_dv 〜 q(R)  q_lv 〜 q(Φ)
                    # サイズ : 文書数D × 単語数N_d × 単語トピック数L
                    q_X  = np.sum([digamma(self.X[idx_doc, m, :]) - digamma(np.sum(self.X[idx_doc, m, :])) for m in range(0, self.doc_v_num[idx_doc])], axis=0)
                    q_1Φ = R_new[idx_doc, self.W2I[self.DI2W[idx_doc_w]]]       * digamma(self.Φ1[:, self.W2I[self.DI2W[idx_doc_w]]]) - digamma(np.sum(self.Φ1, axis=1))
                    q_0Φ = (1 - R_new[idx_doc, self.W2I[self.DI2W[idx_doc_w]]]) * digamma(self.Φ0[0, self.W2I[self.DI2W[idx_doc_w]]]) - digamma(np.sum(self.Φ0, axis=1))
                    Z_new[idx_doc, idx_doc_w, :] = np.exp(q_X + q_1Φ + q_0Φ)
                    Z_new[idx_doc, idx_doc_w, :] = Z_new[idx_doc, idx_doc_w, :] / np.sum(Z_new[idx_doc, idx_doc_w, :])
                    
                    
                    # q(Φ) ∝ Dirichlet_0((Σ_d Σ_n:(v=W_dn) (1 - q_dv)) + β)  q_dv 〜 q(R)
                    #        Dirichlet_l((Σ_d Σ_n:(v=W_dn) q_dv q_dnl) + β)  q_dv 〜 q(R)  q_dnl 〜 q(Z)
                    Φ0_new[0, self.W2I[self.DI2W[idx_doc_w]]] += 1 - R_new[idx_doc, self.W2I[self.DI2W[idx_doc_w]]]
                    Φ1_new[:, self.W2I[self.DI2W[idx_doc_w]]] += R_new[idx_doc, self.W2I[self.DI2W[idx_doc_w]]]     * Z_new[idx_doc, idx_doc_w, :]
            
            # デバッグ出力
            error_Λ = np.sum(np.abs(self.Λ - Λ_new))
            error_Φ = np.sum(np.abs(self.Φ0 - Φ0_new)) + np.sum(np.abs(self.Φ1 - Φ1_new))
            error_Ψ = np.sum(np.abs(self.Ψ - Ψ_new))
            error_θ = np.sum(np.abs(self.θ - θ_new))
            error_Y = np.sum(np.abs(self.Y - Y_new))
            error_X = np.sum(np.abs(self.X - X_new))
            error_Z = np.sum(np.abs(self.Z - Z_new))
            error_R = np.sum(np.abs(self.R - R_new))
            error = error_Λ + error_Φ + error_Ψ + error_θ + error_Y + error_X + error_Z + error_R
            if idx % 100 == 0:
                print(f"学習回数：{idx}")
                print(f"誤差：{error}")
                print()
            
            # 各分布の更新
            self.Λ = Λ_new
            self.Φ0 = Φ0_new
            self.Φ1 = Φ1_new
            self.Ψ = Ψ_new
            self.θ = θ_new
            self.Y = Y_new
            self.X = X_new
            self.Z = Z_new
            self.R = R_new
            
            # 終了条件
            if error < self.tol:
                break

        return True
    
    def stats_info(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        # I2W       = {val: key for key, val in self.W2I.items()}
        # doc_idx   = [f"文書{i + 1}"          for i in range(0, self.doc_num)]
        # topic_idx = [f"トピック{i + 1}"       for i in range(0, self.topic_num)]
        # word_idx  = [f"単語{i + 1}:{I2W[i]}" for i in range(0, self.vocab_num)]
        
        # # 点推定への変換
        # topic_θ = (self.topic_θ_αk + self.topic_θ_α) / np.sum(self.topic_θ_αk + self.topic_θ_α, axis=1).reshape(self.doc_num,   1)
        # word_Φ  = (self.word_Φ_βv  + self.word_Φ_β)  / np.sum(self.word_Φ_βv  + self.word_Φ_β,  axis=1).reshape(self.topic_num, 1)
        
        # # トピック数とは違い、単語数は事前に把握することができないため四捨五入を行わない
        # pd_θ = pd.DataFrame(data=np.round(topic_θ, 4), index=doc_idx,   columns=topic_idx)
        # pd_Φ = pd.DataFrame(data=word_Φ,               index=topic_idx, columns=word_idx)
        
        # return pd_θ, pd_Φ
        pass