#########
# README
#########

q_11_13.py 和 q_14_16.py 是第 11 題到第 16 題的程式。

q_11_13.py 的執行方法如下：

	usage: q_11_13.py [-h] [-o] train test

	positional arguments:
	  train                The path to the training data. (hw3_train.dat)
	  test                 The path to the testing data. (hw3_test.dat)

	optional arguments:
	  -h, --help           show this help message and exit
	  -o, --output_to_png  Output image to a file. (default is output to screen)

而 q_14_16.py 的執行方法如下：

	usage: q_14_16.py [-h] [-o] [--num_tree NUM_TREE] [--boots_rate BOOTS_RATE]
					  train test

	positional arguments:
	  train                 The path to the training data. (hw3_train.dat)
	  test                  The path to the testing data. (hw3_test.dat)

	optional arguments:
	  -h, --help            show this help message and exit
	  -o, --output_to_png   Output image to a file. (default is output to screen)
	  --num_tree NUM_TREE, --nt NUM_TREE
							Number of trees in the random forest. (default 30000)
	  --boots_rate BOOTS_RATE, --bt BOOTS_RATE
							Bootstrap ratio in the random forest. (default 0.8)

也就是說，預設會輸出圖片到螢幕上，如果 -o 的話，會輸出圖片到同一層資料夾，第 13 題的圖檔名會是 q_13.png，14 ~ 16 依此類推。

需要安裝的 python package 都列在 requirements.txt 中。
