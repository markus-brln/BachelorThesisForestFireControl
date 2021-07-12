# BachelorThesisForestFireControl


Hello on the data collection and CNN training branch.

Pipeline
1. Collect training data manually (gui/main.py with NN_control = False)
2. Translate raw data (from gui/data/runs/) according to CNN 
   variant (CNN/data_translator_new_architectures.py)
3. Train several models per architecture (CNN/train_models.py)
4. Transfer saved .json and .h5 files to the results_gathering branch