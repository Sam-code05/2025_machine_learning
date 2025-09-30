import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score


def parse_and_transform_data(xml_file):
    """
    解析 XML 檔案，並轉換為指定的分類和回歸資料集。
    """
    # 解析 XML
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    namespace = {'cwa': 'urn:cwa:gov:tw:cwacommon:0.1'}
    content_text = root.find('.//cwa:Content', namespace).text
    grid_data = np.loadtxt(io.StringIO(content_text), delimiter=',')
    print(f"成功讀取網格資料，形狀為: {grid_data.shape}")

    start_lon, start_lat = 120.00, 21.88
    lon_step, lat_step = 0.03, 0.03
    lon_points = start_lon + np.arange(67) * lon_step
    lat_points = start_lat + np.arange(120) * lat_step

    data_list = []
    for i, lat in enumerate(lat_points):
        for j, lon in enumerate(lon_points):
            temp = grid_data[i, j]
            data_list.append({'lon': lon, 'lat': lat, 'temp': temp})
            
    full_df = pd.DataFrame(data_list)
    
    classification_df = full_df.copy()
    classification_df['label'] = np.where(classification_df['temp'] == -999, 0, 1)
    classification_df = classification_df[['lon', 'lat', 'label']]
    regression_df = full_df[full_df['temp'] != -999].copy()
    regression_df = regression_df[['lon', 'lat', 'temp']].rename(columns={'temp': 'value'})
    
    print("資料轉換完成。")
    print(f"總網格點數量: {len(full_df)}")
    print(f"分類資料集筆數: {len(classification_df)} (有效: {classification_df['label'].sum()}, 無效: {len(classification_df) - classification_df['label'].sum()})")
    print(f"回歸資料集筆數: {len(regression_df)}")
    
    return classification_df, regression_df, grid_data

def visualize_data(grid_data):
    """
    將原始網格資料視覺化，以熱力圖顯示溫度分佈。
    """
    plt.figure(figsize=(10, 12))
    plot_data = np.where(grid_data == -999, np.nan, grid_data)
    sns.heatmap(np.flipud(plot_data), cmap='viridis', square=False)
    plt.title('Temperature Distribution Heatmap (°C)')
    plt.xlabel('Longitude Index')
    plt.ylabel('Latitude Index')
    plt.savefig('temperature_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("溫度分佈熱力圖已儲存至 temperature_heatmap.png")


def train_classification_model(df):
    """
    訓練並評估分類模型。(修正版：使用隨機森林)
    """
    print("\n--- 開始訓練分類模型 ---")
    X = df[['lon', 'lat']]
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print("分類模型評估結果：")
    print(f"準確率 (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
    print("混淆矩陣 (Confusion Matrix):")
    print(confusion_matrix(y_test, y_pred))
    print("分類報告 (Classification Report):")
    print(classification_report(y_test, y_pred))
    
    return model

def train_regression_model(df):
    """
    訓練並評估回歸模型。
    """
    print("\n--- 開始訓練回歸模型 ---")
    X = df[['lon', 'lat']]
    y = df['value']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print("回歸模型評估結果：")
    print(f"均方誤差 (MSE): {mean_squared_error(y_test, y_pred):.4f}")
    print(f"R-squared (R²): {r2_score(y_test, y_pred):.4f}")
    
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red', lw=2)
    plt.title('Regression Model: True vs. Predicted Temperatures')
    plt.xlabel('True Temperature (°C)')
    plt.ylabel('Predicted Temperature (°C)')
    plt.grid(True)
    plt.savefig('regression_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("回歸結果圖已儲存至 regression_results.png")
    
    return model


if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['Heiti TC'] # Mac
    # plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] # Windows
    plt.rcParams['axes.unicode_minus'] = False

    xml_filename = 'O-A0038-003.xml'
    
    classification_data, regression_data, original_grid = parse_and_transform_data(xml_filename)
    visualize_data(original_grid)

    classification_model = train_classification_model(classification_data)
    regression_model = train_regression_model(regression_data)
    
    print("\n所有任務執行完畢。")