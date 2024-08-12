from recursive.dataset.file_label import FileLabel
from recursive.dataset.loader import Loader
from recursive.segment.extender import Extender
from recursive.model.xgb import XGBoost
from recursive.segment import seg_manager
from recursive.etc.config import config

k = 8
extension = 2
target = 70

SAVE_FILE = 'recursive-70.txt'

file_label = FileLabel(config['label_file'], config['data_dir'])
extender = Extender()
loader = Loader(file_label, num_workers=76)

# check if the save file exists
try:
    seg_manager.load(SAVE_FILE)
    train_kmer, test_kmer, train_labels, test_labels = loader.get_extended_dataset()
except FileNotFoundError:
    seg_manager.add_all_kmer(k)
    train_kmer, test_kmer, train_labels, test_labels = loader.get_kmer_dataset(k)

while True:
    if k > target:
        break

    xgb = XGBoost()
    results_df, importance_df = xgb.run(train_kmer, test_kmer, train_labels, test_labels)

    print(importance_df)

    index = list(map(int, importance_df['Feature'].str.replace('f', '').values))[:1500]
    seg_manager.use_subset(index)
    extender.extend_all_segs(extension)
    seg_manager.segments_pruning()
    seg_manager.save(SAVE_FILE)
    train_kmer, test_kmer, train_labels, test_labels = loader.get_extended_dataset()

    k += extension
