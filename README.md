# BachelorThesisForestFireControl


Welcome on the data collection and CNN training branch.

Folder description:
- CNN: data translation + CNN training
- Documentation: I/O representations of current and past ideas for network architectures
- gui: manual data collection
- PastPapersAndNotes: previous work and summaries of it

Pipeline
1. Collect training data manually (gui/main.py with NN_control = False)
   - Spacebar: progress simulation
   - Left Mouse Button: assign digging waypoint to agent marked as yellow
   - Right Mouse Button: non-digging waypoint (twice the maximum travelling distance)
   - Enter: save data points from that run
   - Backspace: discard data points from that run
   - Close window: save all data points to file, type in name in command line
2. Translate raw data (from gui/data/runs/) according to CNN variant 
   - CNN/data_translator_new_architectures.py
3. Train several models per architecture
   - CNN/train_models.py
4. Check visually whether your models work 
   - gui/main.py with NN_control = True
5. Transfer saved .json and .h5 files to the results_gathering branch 
   - also under CNN/saved_models/