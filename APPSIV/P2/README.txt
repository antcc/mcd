To test the code, run (FROM THE ROOT DIRECTORY!):

  $ python code/video_classification.py

Alternatively, one can perform the process interactively from the Jupyter
notebook attached (in Google Colab), or visualize it in the already compiled
HTML version.

Note that the provided original scripts have been slightly modified to account
for the different versions of the data (5, 10, 15 and 20 classes), so there is
no need to treat them separately.

The directory structure is the following:

root
----
  - Lab2.pdf: lab report of this assignment.
  - Video_Classification.html: compiled HTML version of the Jupyter notebook
    developed with the main code and results.

code
----
  - {data.py, processor.py}: common Python scripts for both sessions.
  - S1: directory with Python scripts for Session 1.
  - S2: directory with Python scripts for Session 2.
  - video_classification.py: Python script to run all the code developed.
  - Video_Classification.ipynb: Jupyer notebook to run all the code developed and
    visualize the results easily. It is designed to run in Google Colab but can
    be adapted effortlessly.

data
----
  - checkpoints: model checkpoints for each of the 4 versions of the CNN.
  - logs: training logs for each of the 4 versions of the CNN.
  - sequences: empty.
  - test: empy (*).
  - train: empty (*).
  - ucfTrainTestlist: specification of several train/test splits for the data.
  - {1_move_files.py, 2_extract_files.py}: helper Python scripts to process data.
  - data_file.csv: empty (it is populated after succesfully running the helper
    scripts).

(*) The 'train' and 'test' directories should be filled with the UCF-20 data
following the procedure outlined in the class slides.
