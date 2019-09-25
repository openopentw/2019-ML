# Records

### Idea

- Model
  - 先做 PCA 降維
  - 或是先用 encoder-decoder 降維
  - DNN (記得先 normalize data)
- Multi-target prediciton
  - (1) 先算出第 1 個 y，再把這個 y 加到 x 裡面，再算出第 2 個 y，依此類推算出第 3 個 y。
    - (1-a) y 的順序可以變動？→ensemble
  - (2) 更好的作法：先算出前 2 個 y，再用所有的資料預測第 3 個 y，換掉第 3 個 y 重複做 3 次，最後都取第 3 個 y 接起來。
- Feature Engineering
  - (1) first normalize data, and then xgboost or linear regression or else 
  - (2) 用 random forest 的 OOB 來挑 feature
  - (3) 用 linear regression 的 weight 挑 feature，或是 leave one out 來挑
- custom objective function
- Parameter Tuning
  - (cv) cross validation
  - 微調 random forest 的樹的數量，從 performance 的穩定度來看目前樹的數量夠不夠多

### Results

- See `Results.csv`

### Not Using Machines

- 

### 