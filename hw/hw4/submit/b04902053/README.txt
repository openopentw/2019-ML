#########
# README
#########

q_11_12.py 、 q_13_14.py 和 q_15_16.py 是第 11 題到第 16 題的程式。

都是使用 python 3.6。

- q_11_12.py 和 q_13_14.py 的執行方法如下：

  usage: q_11_12.py / q_13_14.py [-h] [-o] train test

  positional arguments:
    train                hw4_train.dat
    test                 hw4_test.dat

  optional arguments:
    -h, --help           show this help message and exit
    -o, --output_to_png  Output image to files. (default is display on screen)

- 而 q_15_16.py 的執行方法如下：

  usage: q_15_16.py [-h] [-t TRIAL] [-o] data

  positional arguments:
    data                  hw4_nolabel_train.dat

  optional arguments:
    -h, --help            show this help message and exit
    -t TRIAL, --trial TRIAL
                          experiment times (default = 500)
    -o, --output_to_png   Output image to files. (default is display on screen)

也就是說，預設會輸出圖片到螢幕上，如果 -o 的話，會輸出圖片到同一層資料夾。

第 11 題的圖檔名會是 q_11.png，12 ~ 16 依此類推。

需要安裝的 python package 都列在 requirements.txt 中。
