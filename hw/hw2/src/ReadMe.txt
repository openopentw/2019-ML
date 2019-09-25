#########
# README
#########

q_09_10.py, q_11_12.py 和 q_13_16.py 是第 9 題到第 16 題的程式。

q_09_10.py 和 q_11_12.py 的執行方法如下：

  usage: q_09_10.py / q_11_12.py [-h] data

  positional arguments:
    data        The path to the data. (hw2_lssvm_all.dat)

  optional arguments:
    -h, --help  show this help message and exit

而 q_13_16.py 的執行方法如下：

  usage: q_13_16.py [-h] [-o] train test

  positional arguments:
    train             The path to the training data. (hw2_adaboost_train.dat)
    test              The path to the testing data. (hw2_adaboost_test.dat)

  optional arguments:
    -h, --help        show this help message and exit
    -o, --output_fig  Output image to a file. (default is output to screen)

預設輸出圖片的路徑是同一層資料夾，第 13 題的圖檔名會是 q_13.png，14 ~ 16
依此類推。

需要安裝的 python package 都列在 requirements.txt 中。
