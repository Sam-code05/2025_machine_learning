import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- 資料準備 ---

def parse_data(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    namespace = {'cwa': 'urn:cwa:gov:tw:cwacommon:0.1'}
    content_text = root.find('.//cwa:Content', namespace).text
    
    # 使用 np.loadtxt 處理包含換行符的字串
    grid_data = np.loadtxt(io.StringIO(content_text), delimiter=',')
    
    if grid_data.shape != (120, 67):
        raise ValueError(f"預期資料形狀為 (120, 67)，但讀取到 {grid_data.shape}")

    # 建立經緯度座標
    start_lon, start_lat = 120.00, 21.88
    lon_step, lat_step = 0.03, 0.03
    lon_points = start_lon + np.arange(67) * lon_step
    lat_points = start_lat + np.arange(120) * lat_step

    data_list = []
    for i, lat in enumerate(lat_points):
        for j, lon in enumerate(lon_points):
            temp = grid_data[i, j]
            data_list.append({'lon': lon, 'lat': lat, 'temp': temp})
            
    return pd.DataFrame(data_list)


class GaussianDiscriminantAnalysis:
    def __init__(self):
        self.phi = None
        self.mu_0 = None
        self.mu_1 = None
        self.sigma_0_inv = None
        self.sigma_1_inv = None
        self.sigma_0_det_log = None
        self.sigma_1_det_log = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # 計算 P(y=1) 的先驗機率 phi
        self.phi = np.mean(y)

        # 計算 mu_0 和 mu_1
        X_0, X_1 = X[y == 0], X[y == 1]
        self.mu_0 = np.mean(X_0, axis=0)
        self.mu_1 = np.mean(X_1, axis=0)

        # 計算各類別獨立的協方差矩陣 Sigma_0 和 Sigma_1
        # 加上一個小的正則化項 (epsilon * I) 避免矩陣奇異
        epsilon = 1e-6 
        sigma_0 = np.cov(X_0, rowvar=False) + epsilon * np.identity(n_features)
        sigma_1 = np.cov(X_1, rowvar=False) + epsilon * np.identity(n_features)
        
        # 計算並儲存逆矩陣和行列式的對數
        self.sigma_0_inv = np.linalg.inv(sigma_0)
        self.sigma_1_inv = np.linalg.inv(sigma_1)
        self.sigma_0_det_log = np.log(np.linalg.det(sigma_0))
        self.sigma_1_det_log = np.log(np.linalg.det(sigma_1))

    def predict_proba(self, X):
        if self.phi is None:
            raise RuntimeError("模型尚未訓練，請先調用 fit() 方法。")

        # 計算每個點到 mu_0 和 mu_1 的馬氏距離平方
        diff_0 = X - self.mu_0
        mahalanobis_0 = np.sum((diff_0 @ self.sigma_0_inv) * diff_0, axis=1)
        
        diff_1 = X - self.mu_1
        mahalanobis_1 = np.sum((diff_1 @ self.sigma_1_inv) * diff_1, axis=1)

        # 計算對數機率分數 (忽略常數項)
        log_prob_0 = -0.5 * (mahalanobis_0 + self.sigma_0_det_log) + np.log(1 - self.phi)
        log_prob_1 = -0.5 * (mahalanobis_1 + self.sigma_1_det_log) + np.log(self.phi)
        
        return log_prob_1 - log_prob_0

    def predict(self, X):
        scores = self.predict_proba(X)
        return (scores > 0).astype(int)

def plot_decision_boundary(model, X, y, title):
    plt.figure(figsize=(8, 10))
    
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette={0: 'blue', 1: 'red'}, alpha=0.5)
    
    # 建立網格
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # 在網格上進行預測
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid_points)
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.coolwarm)
    plt.title(title)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend(['Class 0 ', 'Class 1 '])
    plt.savefig('gda_decision_boundary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("GDA 決策邊界圖已儲存至 gda_decision_boundary.png")


def combined_model_predict(X, classification_model, regression_model):
    # 使用分類模型預測標籤
    labels = classification_model.predict(X)
    
    # 初始化結果陣列為 -999
    predictions = np.full(len(X), -999.0)
    
    # 找出預測為有效 (label=1) 的資料點
    valid_indices = np.where(labels == 1)[0]
    
    if len(valid_indices) > 0:
        valid_X = X[valid_indices]
        temp_predictions = regression_model.predict(valid_X)
      
        predictions[valid_indices] = temp_predictions
        
    return predictions

def plot_combined_result(df, predictions, title):
    df['h_pred'] = predictions
    
    # 只繪製有效的預測點 (溫度不為 -999 的點)
    valid_predictions_df = df[df['h_pred'] != -999].copy()
    
    plt.figure(figsize=(8, 10))
    
    # 使用散點圖，x=經度, y=緯度, c=預測溫度
    scatter = plt.scatter(
        valid_predictions_df['lon'], 
        valid_predictions_df['lat'], 
        c=valid_predictions_df['h_pred'], 
        cmap='viridis', 
        marker='s',  # 使用方形標記來模擬像素感
        s=10       # 調整標記大小以填滿空隙
    )
    
  
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.colorbar(scatter, label='Predicted Temperature (°C)')
    plt.title(title)
    plt.xlabel('Longitude ')
    plt.ylabel('Latitude ')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('combined_model_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("組合模型地理分佈圖已儲存至 combined_model_geographic_plot.png")


# --- 主程式執行區 ---

if __name__ == '__main__':
    full_df = parse_data('O-A0038-003.xml')
    
    classification_df = full_df.copy()
    classification_df['label'] = np.where(classification_df['temp'] == -999, 0, 1)
    regression_df = full_df[full_df['temp'] != -999].copy()
    regression_df = regression_df.rename(columns={'temp': 'value'})

    print("--- 執行 GDA 分類任務 ---")
    X_cls = classification_df[['lon', 'lat']].values
    y_cls = classification_df['label'].values
 
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
        X_cls, y_cls, test_size=0.3, random_state=42, stratify=y_cls
    )

    gda_model = GaussianDiscriminantAnalysis()
    gda_model.fit(X_train_cls, y_train_cls)
    
    y_pred_cls = gda_model.predict(X_test_cls)
    accuracy = accuracy_score(y_test_cls, y_pred_cls)
    cm = confusion_matrix(y_test_cls, y_pred_cls)
    report = classification_report(y_test_cls, y_pred_cls, target_names=['Class 0 (無效)', 'Class 1 (有效)'])

    print(f"GDA 模型在測試集上的準確率 (Accuracy): {accuracy:.4f}")
    print("\n混淆矩陣 (Confusion Matrix):")
    print(cm)
    print("\n分類報告 (Classification Report):")
    print(report)
    
    plot_decision_boundary(gda_model, X_train_cls, y_train_cls, 'GDA Decision Boundary on Training Data')


    print("\n--- 執行組合迴歸任務 ---")
    X_reg = regression_df[['lon', 'lat']].values
    y_reg = regression_df['value'].values
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_reg, y_reg) # 在所有有效資料上訓練以獲得最佳模型
    
    full_grid_X = full_df[['lon', 'lat']].values
    combined_predictions = combined_model_predict(full_grid_X, gda_model, rf_model)
    
    num_predicted_invalid = np.sum(combined_predictions == -999)
    num_gda_invalid = np.sum(gda_model.predict(full_grid_X) == 0)
    print(f"GDA 預測為無效的點數: {num_gda_invalid}")
    print(f"組合模型輸出為 -999 的點數: {num_predicted_invalid}")
    if num_predicted_invalid == num_gda_invalid:
        print("驗證成功：組合模型正確地將 GDA 預測為 0 的點設為 -999。")

    plot_combined_result(full_df, combined_predictions, 'Combined Model h(x) Prediction Heatmap')