# Curfless-Blood-Pressure-Prediction-Python


### Dataset:

Dataset :  [Link](https://archive.ics.uci.edu/ml/machine-learning-databases/00340/)

### Xử lý dữ liệu

Các file matlab xử lý dữ liệu đã được chuyển thành code python.

### Machine learning models

Đã thêm 2 model để thử nghiệm là Gaussian Process Regression và Stacked Ensemble.

### Chạy training các models

Link đến notebook để train 2 model mới được thêm vào: [Link](https://colab.research.google.com/drive/1mKiSDRO9sr4ud5C0udRj5zhx3Dq9rSgT?usp=sharing)

```bash
cd models_ML
python rf.py
python gpr.py
python ensemble.py
```

### Kết quả chạy 2 model mới được thêm

#### Gaussian Process Regression

- Train: MAE - 0.0046    |    RMSE - 0.006
- Test: MAE - 10.392    |    RMSE - 13.499

#### Stacked Ensemble

- Train: MAE - 9.877    |    RMSE - 12.674
- Test: MAE - 10.865    |    RMSE - 13.775






