import numpy as np
from sklearn.neighbors import KNeighborsClassifier

class MinimumReferenceSet(object):
    
    def run(self, fs, c_label, limit=650):
        ca, cb = self._split_plots_by_class(fs, c_label)
        plot_dist = self._measurement_by_euclid(ca, cb)
        eval_val = self._calc_mrs(ca, cb, plot_dist, fs, c_label, limit)
        return eval_val
        
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
    
    def _sort_outputs_of_measurement(self, p_dist):
        return np.sort(p_dist.flatten())
    
    def _select_2plots_closer(self, p_dist, p_dist_sort, ca_idx_list, cb_idx_list, pair_num):
        # プロット間の距離の近さがn番目のペアのlistを取得[caのindex,cbのindex]
        pair_list = np.where(p_dist == p_dist_sort[pair_num])
        pair_list = np.array(pair_list).tolist()
        
        # 同じ距離のペアが複数いる場合，順番に代入されるようにする
        lap_num = len(pair_list[0])
        if lap_num != 1:
            for i in range(len(pair_list[0])):
                check_pair = [pair_list[0][i],pair_list[1][i]]
                if not ((pair_list[0][i] in ca_idx_list[-lap_num:]) and (pair_list[1][i] in cb_idx_list[-lap_num:])):
                    # 同率順位なので，まだ選択されていないプロットを優先的に選択する
                    return pair_list[0][i], pair_list[1][i]
                else:
                    # 重複しているプロットを返り値とする(後でsetで削除する)
                    return pair_list[0][0], pair_list[1][0]
        else :
            return pair_list[0][0], pair_list[1][0]
            
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
    
    def _calc_mrs(self, ca, cb, p_dist, fs, c_label, limit):
        ca_idx_list = []
        cb_idx_list = []
        p_dist_sort = self._sort_outputs_of_measurement(p_dist)
        
        for pair_num in range(limit):
            # ----- 最も近い2プロット(クラスAから1プロット，クラスBから1プロット)を選択する -----
            ca_idx, cb_idx = self._select_2plots_closer(p_dist, p_dist_sort, ca_idx_list, cb_idx_list, pair_num)
            
            ca_idx_list.append(ca_idx)
            cb_idx_list.append(cb_idx)
            
            # ----- 選択した2プロットで全データを分類する -----
            # 学習用データの作成
            train_x, train_y = self._create_train_dataset(ca, cb, ca_idx_list, cb_idx_list)
            # 1nnの分類精度
            acc = self._calc_accuracy_of_1nn(train_x, train_y, fs, c_label)
            
            # ----- 分類する際に選択したプロット数で特徴量空間の評価を行う(選択されたプロットが少ない空間を良いとする) -----
            # accuracyが"1"ならば終了，"1"でなければ続行
            if acc == 1:
                return pair_num + 1
            elif (pair_num + 1) == limit :
                return pair_num + 1