# Unity 虛擬環境車子自動控制實作教學

## 📦 相關資源 (Related Resources)
| 工具/專案 | 連結 |
|:---|:---|
| **Docker** | [Install Docker Desktop on Windows](https://docs.docker.com/desktop/setup/install/windows-install/) |
| **VS Code** | [Download Visual Studio Code](https://code.visualstudio.com/download) |
| **Foxglove** | [Download Foxglove Studio](https://foxglove.dev/download) |
| **Unity 專案檔** | [Google Drive](https://drive.google.com/file/d/1NoTt-6TKbsZpL9VsCNiBgbYlBSgeDJNY/view?usp=sharing) |
| **ROS2 車輛控制程式碼** (`pros_car`) | [GitHub](https://github.com/asd56585452/pros_car) |
| **SLAM & NAV2 應用程式** (`pros_app`) | [GitHub](https://github.com/asd56585452/pros_app) |
| **YOLO 整合套件** (`ros2_yolo_integration`) | [GitHub](https://github.com/asd56585452/ros2_yolo_integration) |
| **WSL 官方指南** | [WSL \| Microsoft Learn](https://learn.microsoft.com/zh-tw/windows/wsl/) |

---

## 🛠️ 環境建立 (Environment Setup)

### 1. 安裝 WSL & Ubuntu 22.04
開啟 **Windows PowerShell**（建議以系統管理員身分執行）並輸入：
```bash
wsl --install -d Ubuntu-22.04
```
安裝完成後重啟電腦，進入 Ubuntu 終端機完成初始帳號設定。

### 2. 設定 VS Code
1. 安裝 **Visual Studio Code**。
2. 安裝擴充套件：`Remote Development`。
3. 開啟 VS Code，使用 **Remote-WSL** 連線至 Ubuntu 環境。

### 3. 設定 Docker
1. 安裝 **Docker Desktop for Windows**。
2. 開啟設定，勾選 `Use WSL 2 based engine` 以啟用 WSL 2 引擎。
3. 確認 Docker Desktop 已與 WSL Ubuntu 成功整合。

### 4. 建立工作空間與下載專案
在 VS Code 的 WSL 終端機中執行：
```bash
mkdir -p ~/workspace/pros
cd ~/workspace/pros
git clone https://github.com/asd56585452/pros_app.git
git clone https://github.com/asd56585452/pros_car.git
git clone https://github.com/asd56585452/ros2_yolo_integration.git
```

---

## 🗺️ 實作教學：SLAM 與地圖掃描

### 1. 啟動 SLAM
切換至 `pros_app` 目錄並執行控制腳本：
```bash
cd ~/workspace/pros/pros_app
python3 ./control.py -s
# 或直接執行: ./slam_unity.sh
```

### 2. 設定 Foxglove 視覺化
1. 開啟 **Foxglove Studio** → 點擊 **Open connection**。
2. 選擇 **Rosbridge**，輸入 URL：`ws://localhost:9090` 並連線。
3. 建議新增以下 Panel 與 Topic：
   - **Fixed Frame / Display Frame**: `map`
   - **Transform / TF Frames**: `map`, `base_footprint`, `camera`, `laser`, `arm_ik_base`
   - **Topics**:
     - `/camera/color/camera_info`
     - `/camera/image/compressed`
     - `/map`
     - `/scan`
     - `/plan`

### 3. 鍵盤開車掃描地圖
1. 開啟另一個 VS Code 終端機視窗。
2. 進入 `pros_car` 目錄並啟動車輛控制節點：
   ```bash
   cd ~/workspace/pros/pros_car
   ./car_control.sh
   # 或手動執行: ros2 run pros_car_py robot_control
   ```
3. 操作選單與鍵盤：
   - 輸入 `r` 進入 **Main Menu**
   - 選擇 `Control Vehicle`
   - 使用鍵盤控制：
     - `w`：前進
     - `s`：後退
     - `e` / `d`：轉向
   - 駕駛車輛繞行虛擬場景，觀察 Foxglove 中 `/scan` 與 `/map` 即時建圖。

---

## 📍 Localization 與路徑規劃

### 1. 儲存已掃描地圖
在 `pros_app` 終端機執行：
```bash
cd ~/workspace/pros/pros_app
./store_map.sh
```
*(依終端機提示輸入地圖檔名)*

### 2. 啟動定位模式 (Localization)
```bash
./localization_unity.sh
```

### 3. 設定初始位置與目標點
1. 切換至 Foxglove。
2. 使用 **Publish** 工具發布 ROS2 訊息：
   - `/initialpose`：點擊地圖設定車輛當前實際位置（定位初始化）。
   - `/goal_pose`：點擊地圖設定導航終點。
3. 車輛將自動計算路徑並開始移動。

---

## 🚀 啟動 NAVIGATION (自動導航)

### 1. 透過終端機切換模式
在 `pros_car` 終端機中依序操作：
```bash
cd ~/workspace/pros/pros_car
# 1. 選擇 Auto Navigation
# 2. 輸入指令: manual_auto_nav
```

### 2. Foxglove 互動導航
- 確保已發布 `/initialpose`。
- 在 Foxglove 畫面中點擊目標位置發布 `/goal_pose`。
- 觀察 `/plan` 與車輛實際移動軌跡，驗證 Nav2 導航功能。

---

## ⚠️ 其他注意事項
1. **Unity 場景設定**：確保已正確載入 `AI ROS2 Car Model`，且場景內的 ROS2 Topic Publish 設定已開啟。
2. **Nav2 參數**：路徑規劃 (`/plan`) 依賴 Nav2 套件。若車輛無法移動或路徑計算失敗，請檢查 `costmap` 障礙物膨脹半徑與機器人輪廓設定。
3. **環境一致性**：所有 `ros2 run` 與 Python 指令需在對應的 Docker 容器或 WSL Ubuntu 終端機中執行。
4. **除錯建議**：可使用 `ros2 topic echo <topic_name>` 或 `ros2 topic list` 檢查節點與訊息是否正常發布。

> 💡 **實作目標提示**：本作業整合 Unity 模擬、ROS2 通訊、SLAM 建圖與 Nav2 路徑規劃，最終目標為實現車輛在虛擬環境中的 **全自動導航 (Auto Navigation)**。請確實完成各階段指令驗證後再提交作業。

---

# Unity 虛擬環境車子自動控制實作教學

## 📋 目錄
1. [環境檢查與系統啟動](#1-環境檢查與系統啟動)
2. [自動導航實作 (Auto Navigation)](#2-自動導航實作-auto-navigation)
3. [影像辨識整合 (YOLO)](#3-影像辨識整合-yolo)
4. [Foxglove 視覺化設定](#4-foxglove-視覺化設定)
5. [相機畫面取得與 YOLO 訓練資料](#5-相機畫面取得與-yolo-訓練資料)
6. [夾爪操作 (手動與自動)](#6-夾爪操作手動與自動)
7. [URDF 機器人描述格式介紹](#7-urdf-機器人描述格式介紹)
8. [系統 URDF 結構與 ROS2 運作流程](#8-系統-urdf-結構與-ros2-運作流程)
9. [URDF 關鍵零組件說明](#9-urdf-關鍵零組件說明)
10. [URDF 與 Final 實作關聯](#10-urdf-與-final-實作關聯)

---

## 1. 環境檢查與系統啟動
- 啟動定位腳本：`localization_unity.sh`
- 系統注意事項：
  - 執行前請確認 Docker 容器與 ROS2 環境已正確載入。
  - 注意 `Nav2` 路徑規劃 (`plan nav2`) 狀態。
  - 確保所有終端機指令在對應的工作區目錄下執行。

## 2. 自動導航實作 (Auto Navigation)
### 啟動步驟
1. 開啟 Terminal，進入車輛控制目錄：
   ```bash
   cd ~/workspace/pros/pros_car
   ```
2. 啟動控制腳本與 ROS2 節點：
   ```bash
   ./car_control.sh
   ros2 run pros_car_py robot_control
   ```
3. 於主選單中選擇：`Auto Navigation` → `manual_auto_nav` 並按 `Enter`。

### Foxglove 導航操作
- **設定初始位置**：Publish `/initialpose`
- **設定目標位置**：Publish `/goal_pose`
- 點擊 Publish 發送導航指令，觀察車子自動行駛。

## 3. 影像辨識整合 (YOLO)
### 啟動步驟
1. 執行環境初始化：`localization_unity.sh`
2. 進入 YOLO 整合目錄並啟動節點：
   ```bash
   cd ~/workspace/pros/ros2_yolo_integration
   ./yolo_activate.sh
   ros2 run yolo_example_pkg yolo_node
   ```
- 確保 YOLO 節點正常啟動且無錯誤訊息。

## 4. Foxglove 視覺化設定
1. 開啟 **Foxglove** 軟體。
2. 點擊 `Open connection` → 選擇 `Rosbridge`。
3. 輸入 WebSocket 位址：`ws://localhost:9090` 並連線。
4. **影像顯示設定**：
   - 新增 Panel → Topic 選擇 `/yolo/detection/compressed`
   - `Calibration` 設為 `None`
   - 命名該 Panel 為 `yolo`
5. 確認相機畫面與偵測框正常顯示。

## 5. 相機畫面取得與 YOLO 訓練資料
1. 進入 `pros_car` → `Main Menu`。
2. 選擇 `Control Vehicle`。
3. 使用鍵盤控制車輛：
   - `w`, `s`, `e`, `r`：控制前後移動與轉向。
4. 按下 `c` 鍵擷取當前相機畫面。
5. 影像預設儲存路徑：
   ```
   pros_car/src/pros_car_py/images
   ```

## 6. 夾爪操作 (手動與自動)
### 前置作業
- 執行 `localization_unity.sh`
- Unity 介面切換至 `Arm Mode` → `AI ROS2`

### 手動控制 (`Manual Arm Control`)
1. 開啟 VSCode 與 Terminal。
2. 啟動車輛控制節點（同第 2 節指令）。
3. 於 Main Menu 選擇 `Manual Arm Control`。
4. 鍵盤對應控制：
   - `i` → 動作指令 A
   - `k` → 動作指令 B
   - `b` → 動作指令 C
   - `q` → 退出/重置
   - 數字鍵 `0`, `1`, `2` → 對應關節或夾爪狀態切換

### 自動控制 (`Automatic Arm Mode`)
1. 執行 `localization_unity.sh`
2. 切換至 `Automatic Arm Mode`，輸入指令：`auto_arm_human`
3. 在 Foxglove 畫面中點擊 `click_point` 設定抓取目標。
4. 於 VSCode 中調整 `click_point` 相關參數以微調行為。

## 7. URDF 機器人描述格式介紹
- **全名**：Unified Robot Description Format
- **格式**：基於 XML 的機器人結構描述檔。
- **核心元素**：
  - `Link` (連桿)：定義機器人的實體部分，包含 `Visual` (外觀)、`Collision` (碰撞)、`Inertial` (慣性)。
  - `Joint` (關節)：連接兩個 Link，類型包含 `fixed`、`revolute`、`continuous`。
- **在系統中的角色**：
  - 建立 `TF Tree` (座標轉換樹)。
  - Foxglove 顯示關鍵座標系：`map`、`base_footprint`、`camera`、`laser`、`arm_ik_base`。
  - 作為 `SLAM` 與 `Nav2` 導航的基礎框架（定義 `laser` 與 `base_footprint` 相對關係）。

## 8. 系統 URDF 結構與 ROS2 運作流程
### 目前系統結構
- **URDF 檔案路徑**：`pros_app/docker/compose/demo/v6_unity.urdf`
- **結構組成**：
  - `Base`：`base_footprint`
  - `Sensors`：透過 `Fixed Joint` 連接 `laser` 與 `camera`
  - `Arm`：透過 `Fixed Joint` 連接 `arm_ik_base`

### ROS2 運作流程
1. 執行 `./localization_unity.sh` 與 `slam_unity.sh`
2. 載入 Docker Compose：`docker-compose_robot_unity.yml`
3. `Robot State Publisher` 讀取 URDF 檔，發布 `/tf` 與 `/tf_static` 訊息。

## 9. URDF 關鍵零組件說明
| 類別 | 說明 | 範例/備註 |
|:---|:---|:---|
| **車體與其他 (Body & Others)** | TF Tree 根節點 | `base_footprint` (Root)<br>`base_link` 相對於 `base_footprint`<br>`upper_blue_v1_1` / `upper_silver_v1_1` |
| **感測器 (Sensors)** | 相機與雷達 | `camera` → `camera_1` (Link)<br>`camera_optical_frame` (Z 軸朝前) |
| **車輪 (Wheels)** | `Continuous` Joint | `new_wheel_v1_1` (1號)<br>`new_wheel_v1__1__1` (2號)<br>`new_wheel_v1__2__1` (3號)<br>`new_wheel_v1__2___1__1` (4號) |
| **機械臂與夾爪 (Arm & Gripper)** | 逆運動學 (IK) 基礎 | `arm_ik_base` (IK 參考點)<br>`new_motor_v2_1`<br>`big_U_self_v2_1` (關節 0)<br>`big_U_self_v2__1__1` (關節 1)<br>夾爪：`grap2_v2_1`, `grap_base_v1_1`, `grap2_v2_Mirror__1` |

## 10. URDF 與 Final 實作關聯
- **視覺導引**：結合 YOLO 偵測結果進行空間目標定位。
- **座標計算**：透過 `click_point` (點擊座標) 轉換為機器人可理解的空間座標。
- **控制範例**：
  - 腳本路徑：`pros_car/src/pros_car_py/pros_car_py/arm_controller_2D.py`
  - 座標系轉換邏輯：`map` → `arm_ik_base`
- 利用 URDF 定義的 TF 關係，將 2D 畫面點擊點轉換為 3D 空間中的機械臂目標姿態。

---
💡 **實作提醒**：操作過程中請隨時監控 Terminal 輸出，確認 ROS2 Topic 是否正常發布，並檢查 Foxglove 連線狀態與 TF Tree 是否完整載入。

---

# Unity 虛擬環境車子自動控制實作教學

## 📡 Unity 和 ROS2 的 Topic 通訊

---

### 🔹 發布 (PUBLISH) - Unity → ROS2

這些是由 Unity 端生成並傳送給 ROS2 的數據：

#### 📊 `/scan_tmp`
| 屬性 | 說明 |
|------|------|
| **資料型別** | `sensor_msgs/LaserScan` |
| **用途** | 傳遞虛擬環境中光達掃描到的距離 (range) 與強度 (intensity) 數據 |
| **數值範圍** | 限制在 `0.15m` 到 `16.0m` 之間 |

#### 📷 `/camera/image/compressed`
| 屬性 | 說明 |
|------|------|
| **資料型別** | `sensor_msgs/msg/CompressedImage` |
| **用途** | 傳遞車載 RGB 相機拍攝的畫面，影像以 JPG 格式壓縮後發布 |

#### 📐 `/camera/image/camera_info`
| 屬性 | 說明 |
|------|------|
| **資料型別** | `sensor_msgs/msg/CameraInfo` |
| **用途** | 傳遞 RGB 相機的內參矩陣（包含 K、D、R、P 等畸變與投影參數），提供給 ROS2 系統進行視覺校正與空間換算 |

#### 🗺️ `/camera/depth/compressed`
| 屬性 | 說明 |
|------|------|
| **資料型別** | `sensor_msgs/msg/CompressedImage` |
| **用途** | 傳遞深度相機取得的深度圖，單通道 RFloat 深度數據經過 EXR (ZIP 壓縮) 後轉成 Base64 字串傳送 |

#### 🗺️ `/camera/depth/camera_info`
| 屬性 | 說明 |
|------|------|
| **資料型別** | `sensor_msgs/msg/CameraInfo` |
| **用途** | 傳遞深度相機的內參矩陣（包含 K、D、R、P 等畸變與投影參數），提供給 ROS2 系統進行視覺校正與空間換算 |

---

### 🔹 訂閱 (SUBSCRIBE) - ROS2 → Unity

這些是 Unity 接收來自 ROS2 控制演算法的指令：

#### 🚗 `/car_C_front_wheel` & `/car_C_rear_wheel`
| 屬性 | 說明 |
|------|------|
| **資料型別** | `std_msgs/msg/Float32MultiArray` |
| **用途** | 接收包含各 2 浮點數的陣列，分別對應虛擬車輛 4 個車輪的目標轉速，並套用到 Unity 的 `ArticulationBody` 來驅動車子 |

#### 🦾 `/robot_arm`
| 屬性 | 說明 |
|------|------|
| **資料型別** | `trajectory_msgs/msg/JointTrajectoryPoint` |
| **用途** | 接收機械臂各個關節的目標旋轉角度 (Target Angle)，系統會計算當前角度與目標角度的差值，決定馬達要正轉或反轉來達到指定位置 |

---

## 🎯 YOLO Topic 整合

### YOLO 相關 Topics

| Topic 名稱 | 說明 |
|------------|------|
| `/out/compressed` | 相機拍到的原始影像 |
| `/yolo/detection/compressed` | 處理後的影像（含偵測結果） |
| `/yolo/target_info` | 是否找到目標 + 距離資訊 |

#### `/yolo/target_info` 資料結構
```
Index 0 (found)    : 1 代表有找到目標
Index 1 (distance) : 目標深度 (單位：m)
Index 2 (delta_x)  : 目標與影像中心像素偏移量
```

### 深度分割設定
- **Topic**: `/camera/x_multi_depth_values`
- **功能**: 每個等分點的深度值
- **預設設定**: 切分成 20 等距水平點
- **調整位置**: `ros2_yolo_integration/src/yolo_example_pkg/yolo_example_pkg/object_detect.py`
  ```python
  self.x_num_splits = 20  # 可在此調整分割數量
  ```

---

## 🚙 pros_car Topic 架構

### 核心檔案位置
```
workspace/pros/pros_car/src/pros_car_py/pros_car_py/
├── ros_communicator.py    # Topic 統一管理
├── car_controller.py      # 車輛控制邏輯
├── car_models.py          # 車輛模型定義
└── object_detect.py       # 物件偵測整合
```

### 主要匯入模組 (`ros_communicator.py`)
```python
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped, PoseStamped, Point
from std_msgs.msg import String, Header, Float32MultiArray, Bool
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan, Imu, CompressedImage
from trajectory_msgs.msg import JointTrajectoryPoint
from visualization_msgs.msg import Marker
from nav2_msgs.srv import ClearEntireCostmap
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from cv_bridge import CvBridge
import orjson
```

### RosCommunicator 類別架構
```python
class RosCommunicator(Node):
    def __init__(self):
        super().__init__("RosCommunicator")
        
        # Subscribe AMCL Pose
        self.latest_amcl_pose = None
        self.subscriber_amcl = self.create_subscription(
            PoseWithCovarianceStamped, 
            "/amcl_pose", 
            self.subscriber_amcl_callback, 
            10
        )
        
        # Subscribe Goal Pose
        self.latest_goal_pose = None
        self.target_pose = None
        self.subscriber_goal = self.create_subscription(
            PoseStamped, 
            "/goal_pose", 
            self.subscriber_goal_callback, 
            10
        )
        # ... 其他訂閱與發布設定
```

> 💡 **提示**: 控制輪子、機械臂等功能皆集中在 `ros_communicator.py`，建議自行研究 `ACTION_MAPPINGS` 設定。

---

## 📱 pros_app Topic 與 Docker 設定

### 架構說明
- 大多為現有工具整合，可透過 `docker-compose` 查看服務調用關係
- 建議先確認調用了什麼工具，再搜尋對應的使用方式

### Docker Compose 設定範例
```yaml
# 檔案: workspace/pros/pros_app/docker/compose/docker-compose_robot_unity.yml
services:
  robot_bringup:
    image: ghcr.io/screamlab/pros_jetson_driver_image:0.1.0
    env_file:
      - .envV6
    volumes:
      - ./demo:/workspace/demo:ro
    networks:
      - my_bridge_network
    command: "ros2 launch /workspace/demo/robot_bringup_unity.xml"

networks:
  my_bridge_network:
    driver: bridge
```

### ROS2 Launch 設定範例
```xml
<!-- 檔案: workspace/pros/pros_app/docker/compose/demo/robot_bringup_unity.xml -->
<launch>
    <node pkg="robot_state_publisher" exec="robot_state_publisher" name="robot_state_publisher">
        <param name="robot_description" value="$(command 'cat /workspace/demo/v6')"/>
    </node>
    
    <node name="scan_matcher" pkg="ros2_laser_scan_matcher" exec="laser_scan_matcher_node">
        <param name="base_frame" value="base_footprint"/>
        <param name="publish_tf" value="true"/>
        <param name="publish_odom" value="true"/>
    </node>
</launch>
```

---

## 🔧 開發建議與注意事項

1. **Topic 名稱確認**: 所有 Topic 名稱需與 ROS2 端保持一致，避免通訊失敗
2. **資料型別匹配**: 確保 Unity 發布/訂閱的資料型別與 ROS2 定義相符
3. **壓縮格式**: 影像傳輸建議使用壓縮格式以減少頻寬消耗
4. **深度數據處理**: 深度圖轉換為 Base64 時需注意編解碼一致性
5. **控制頻率**: 車輛控制指令建議保持固定頻率發布，確保平穩驅動
6. **錯誤處理**: 在 `RosCommunicator` 中加入適當的例外處理機制

---

> 📌 **參考路徑總覽**
> ```
> workspace/
> ├── pros/
> │   ├── pros_car/          # 車輛控制核心
> │   │   └── src/pros_car_py/
> │   ├── pros_app/          # 應用層整合
> │   │   └── docker/compose/
> │   └── ros2_yolo_integration/  # YOLO 視覺整合
> ```

# Unity 虛擬環境車子自動控制實作教學
*(Unity Virtual Environment Car Autonomous Control Implementation Tutorial)*

## 📡 Foxglove Topic & Raw Messages
- **Foxglove Topic**: Used for data visualization & debugging
- **Raw Messages**: Direct ROS2 message inspection

---

## 🎯 FINAL 實作功能方向建議 (Final Implementation Feature Suggestions)

### 🗺️ Map Randomization
#### 🔹 Basic Explanation
To encourage students to use robust algorithms and models, this year’s map will be **fully randomized**. The bridge, bears, and path will be randomized.
- `Pre-randomization` vs `Post Randomization`

#### 🔹 Path Randomization Explanation
The path randomization is achieved by a combination of **Wave Function Collapse (WFC)** and the **A\* algorithm**:
- **A\* Algorithm**: Ensures a guaranteed path exists from the starting points to the target door.
- **WFC Algorithm**: Iterates through remaining tiles, selects the tile with the lowest possible number of valid options, collapses it, and propagates constraints to neighboring tiles based on predefined adjacency rules.
- 🔗 Reference: [mxgmn/WaveFunctionCollapse](https://github.com/mxgmn/WaveFunctionCollapse)

#### 🔹 Map Versions
| Version | Features |
|---------|----------|
| **Current (2 Maps)** | • `Final Project`: Randomized paths, bridge placement, and bears.<br>• `Racing2026`: Road & bridge positions are fixed, but bear placements are randomized. |
| **Future Versions** | Additional pickup objects + dynamic moving obstacles. |

---

### 👁️ 視覺與感知進階 (Vision & Perception)
| Module | Description |
|--------|-------------|
| **視覺伺服 (Visual Servoing)** | YOLO detection + PID controller (Error → `cmd_vel`) / SLAM integration |
| **語意分割 (Semantic Segmentation)** | YOLO (Bounding Box) vs. `YOLOv8-Seg` (Instance Mask) |
| **3D Bounding Box / 6D Pose Estimation** | 3D position + `Roll, Pitch, Yaw` estimation using Depth sensors & ROS2 |

### 🧭 導航與建圖進階 (Navigation & SLAM)
| Module | Description |
|--------|-------------|
| **視覺 SLAM (Visual SLAM)** | 2D/3D (RGB-D) using `RTAB-Map` or `ORB-SLAM3` (CPU/GPU optimized) |
| **動態 SLAM (Dynamic SLAM)** | Integrating YOLO pipeline for real-time dynamic object masking & tracking |
| **3D Navigation / OctoMap** | Replacing 2D Costmap with 3D `OctoMap`, integrated with `Nav2` or `MoveIt2` for volumetric path planning |

### 🎛️ 控制、決策與系統整合 (Control & Decision)
| Module | Description |
|--------|-------------|
| **動態避障 (Dynamic Obstacle Avoidance)** | `Nav2` Local Planner (`TEB` or `MPPI`). Focus on parameter tuning & avoiding the *Freezing Robot Problem* |
| **進階車子驅動模型 (Advanced Driving Model)** | Kinematic/Dynamic models. Fusing `cmd_vel` with **EKF**, IMU, and Odometry to handle covariance matrices & blind navigation in `Nav2` |
| **主動探索 (Active / Frontier Exploration)** | Using `explore_lite` for frontier-based mapping, optionally combined with YOLO for targeted area scanning |

---

## 📊 評分方式 (Scoring)

### 🔑 登入方式 (Login)
| Credential | Value |
|------------|-------|
| **Username** | `username` |
| **Key** | `key` |

### 📝 Task Breakdown & Points
| Task | Objective | Points |
|------|-----------|--------|
| **Task 1** | Locate & Observe | 10 pts |
| | Recovery | 20 pts |
| **Task 2** | Ascent | 10 pts |
| | Descent | 10 pts |
| | Recovery | 10 pts |
| **Task 3** | Locate & Observe | 10 pts |
| | Unlock | 20 pts |
| | Clear | 10 pts |
| **Total** | | **100 pts** |

> 💡 *Note: Tasks 2 & 3 follow the same structured evaluation criteria. Ensure proper logging & recovery mechanisms to maximize scoring.*