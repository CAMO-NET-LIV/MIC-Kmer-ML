from recursive.dataset.file_label import FileLabel
from recursive.dataset.loader import Loader
from recursive.features.extender import Extender
from recursive.model.xgb import XGBoost
from recursive.features import lookup as lp

k = 8
extension = 2
target = 70

label_file = 'labels-cleaned.csv'
data_dir = '../volatile/e_coli_mic/'

file_label = FileLabel(label_file, data_dir)
lp.add_all_kmer(k)
loader = Loader(file_label, num_workers=76)
train_kmer, test_kmer, train_labels, test_labels = loader.get_kmer_dataset(k)
extender = Extender()

while True:
    if k > target:
        break

    xgb = XGBoost()
    results_df, importance_df = xgb.run(train_kmer, test_kmer, train_labels, test_labels)

    print(importance_df)

    index = list(map(int, importance_df['Feature'].str.replace('f', '').values))[:1000]
    lp.select(index)
    extender.extend_all_seq_in_lookup(extension)
    train_kmer, test_kmer, train_labels, test_labels = loader.get_extended_dataset()

    k += extension
