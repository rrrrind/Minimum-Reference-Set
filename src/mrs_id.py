import numpy as np
from sklearn.neighbors import KNeighborsClassifier

class MRSIdentifier(object):
    
    def run(self, fs, c_label, mrs_limit=None):
        ca, cb = self._split_plots_by_class(fs, c_label)
        if (mrs_limit == None) or (mrs_limit > len(fs)):
            mrs_limit = len(fs) # MRSの最大値は全プロット数
        plot_dist = self._measurement_by_euclid(ca, cb)
        selected_samples = self._calc_mrs(ca, cb, plot_dist, fs, c_label, mrs_limit)
        return selected_samples
        
    def _split_plots_by_class(self, fs, c_label):
        ca = []
        cb = []
        for i, c in enumerate(c_label):
            if c == 0: # 0を"クラスA"とする
                ca.append(fs[i])
            elif c == 1: # 1を"クラスB"とする
                cb.append(fs[i])
        ca = np.array(np.array(ca).tolist())
        cb = np.array(np.array(cb).tolist())
        return ca, cb
    
    def _measurement_by_euclid(self, ca, cb):
        p_dist = np.zeros([len(ca),len(cb)])
        for i, ca_vec in enumerate(ca):
            for j, cb_vec in enumerate(cb):
                p_dist[i,j] = np.linalg.norm(ca_vec-cb_vec)
        return p_dist
    
    def _select_sorted_pairs(self, p_dist):
        p_dist_sort = np.sort(list(set(p_dist.flatten())))
        
        sorted_ca_idx = []
        sorted_cb_idx = []
        for _, i_dist in enumerate(p_dist_sort):
            pair_list = np.where(p_dist == i_dist)
            pair_list = np.array(pair_list).tolist()
            sorted_ca_idx.append(pair_list[0])
            sorted_cb_idx.append(pair_list[1])
        return sum(sorted_ca_idx, []), sum(sorted_cb_idx, [])

    def _create_train_dataset(self, ca, cb, ca_idx_list, cb_idx_list):
        train_ca_idx = list(set(ca_idx_list)) # プロットの重複を削除
        train_cb_idx = list(set(cb_idx_list)) # プロットの重複を削除
        X_ca = ca[train_ca_idx]
        y_ca = np.zeros(len(X_ca), dtype=np.int64)
        X_cb = cb[train_cb_idx]
        y_cb = np.ones(len(X_cb), dtype=np.int64)
        return np.vstack([X_ca,X_cb]), np.hstack([y_ca,y_cb])
    
    def _calc_accuracy_of_1nn(self, train_x, train_y, fs, c_label):
        # 1NNの学習
        neigh = KNeighborsClassifier(n_neighbors=1)
        neigh.fit(train_x, train_y)
        return neigh.score(fs, c_label)
    
    def _calc_mrs(self, ca, cb, p_dist, fs, c_label, mrs_limit):
        ca_idx_list = []
        cb_idx_list = []

        sorted_ca_idx, sorted_cb_idx = self._select_sorted_pairs(p_dist)
        
        count = 0
        while True:
            # ----- 最も近い2プロット(クラスAから1プロット，クラスBから1プロット)を選択する -----
            ca_idx_list.append(sorted_ca_idx[count])
            cb_idx_list.append(sorted_cb_idx[count])
            
            # ----- 選択した2プロットで全データを分類する -----
            # 学習用データの作成
            train_x, train_y = self._create_train_dataset(ca, cb, ca_idx_list, cb_idx_list)
            # 1nnの分類精度
            acc = self._calc_accuracy_of_1nn(train_x, train_y, fs, c_label)
            
            nor_num = len(list(set(ca_idx_list))) #重複を削除した，選択された正常データの総和
            ano_num = len(list(set(cb_idx_list)))#重複を削除した，選択された異常データの総和
            mrs_now = nor_num + ano_num
            
            # ----- 分類する際に選択したプロット数で特徴量空間の評価を行う(選択されたプロットが少ない空間を良いとする) -----
            # accuracyが"1"ならば終了，"1"でなければ続行
            if (acc == 1) or (mrs_now >= mrs_limit):
                return mrs_now
            
            count += 1