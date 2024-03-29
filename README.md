# Minimum Reference Set
## Contents
Minimum Reference Setのモジュール  

- mrs_id.py  
入力された特徴集合(特徴量空間)の評価が可能．  
多次元・2クラスに対応． 

- mrs_fs.py  
全体の特徴量から構築可能なk次元特徴集合(特徴量空間)の評価が可能．  
多次元・2クラスに対応． 


## Algorithm  
### MRS overview  
![MRS_overview](https://github.com/rrrrind/Minimum-Reference-Set/blob/main/img/MRS%E6%A6%82%E8%A6%81.jpg)  

### MRS_ID  
1. クラス毎にプロットを分割し，ユークリッドでクラス毎のプロット間の距離を測る  
1. 最も近い2プロットを選択する  
1. 選択した2プロットで全データを分類する(最近傍法を使用)  
1. 分類誤差が0の場合は終了し、0でない場合は更に2プロット選択し全データを分類する  
1. 分類する際に選択したプロット数で特徴量空間の評価を行う(選択されたプロットが少ない空間を良いとする)  

### MRS_FSA  
1. 入力された『全体の特徴量』から，初期集合を決定する  
    - mode = 'random'の場合は，k次元の特徴集合をランダムで作成し，それを初期集合とする  
    - mode = 'specific'の場合は，kに入力された特徴集合を初期集合とする  
1. 初期集合に対してMRS_IDを適用し，特徴集合の特徴量を別のに入れ替えMRS_IDの適用を繰り返す  
1. その際に，評価値(MRS)が小さくなった場合は，特徴量の入れ替えを承認する  
1. 最終的に，最も評価値(MRS)が小さくなった特徴集合を，最も良い特徴集合と判断する  

※アルゴリズムが間違えてる等ございましたら，ご指摘いただけると幸いです．  


## How to use
### MRS_ID
1. from mrs import MinimumReferenceSet
1. mrs = MinimumReferenceSet()
1. selected_samples = mrs.run(fs, c_label, mrs_limit)
    - selected_samples(int型): 入力された特徴集合(特徴量空間)の評価値
    - fs(np.array型, [サンプル, 特徴量]): 評価したい特徴集合(特徴量空間)
    - c_label(list or np.array型, [0 or 1]): クラスラベル
    - mrs_limit(int型): 探索上限，指定しない場合は全探索，指定を推奨(100とか？)

### MRS_FSA
1. from modules.mrs_fs import MRSFeatureSelection
1. mrs_fs = MRSFeatureSelection()
1. size_of_mrs, feature_set = mrs_fs.run(features, labels, mode=None, k=None)  
    - size_of_mrs(int型): 最良とされた特徴集合(特徴量空間)の評価値
    - feature_set(list型): 最良とされた特徴集合(特徴量空間)
    - features(np.array or pd.DataFrame, [サンプル, 特徴量]): 全サンプルと全特徴量
    - labels(list or np.array型, [0 or 1]): クラスラベル 
    - mode(string型, 'random' or 'specific') :  
        - 'random'の場合: k次元の特徴集合をランダムで作成し，それを初期集合とする  
        - 'specific'の場合: kにlist型で特徴集合を入力し，それを初期集合とする  
          (既に初期集合が決まっている場合は'specific'の使用を，決まっていない場合は'random'の使用を推奨)  


## Paper
[Minimum reference set based feature selection for small sample classifications](https://dl.acm.org/doi/abs/10.1145/1273496.1273516)
