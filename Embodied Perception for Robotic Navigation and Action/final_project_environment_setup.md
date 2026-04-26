# UNITY 虛擬環境車自動控制實作教學

## 相關資源

| 工具 | 連結 |
|------|------|
| **Docker** | https://docs.docker.com/desktop/setup/install/windows-install/ |
| **VS Code** | https://code.visualstudio.com/download |
| **Foxglove** | https://foxglove.dev/download |
| **Unity** | https://drive.google.com/drive/folders/13juV_QX70JGf63GbHUtsG4nQGBDRMlNY?usp=sharing |

## GitHub 專案

| 專案 | 連結 |
|------|------|
| **Pros Car**（主專案） | https://github.com/asd56585452/pros_car |
| **SLAM NAV2 應用** | https://github.com/asd56585452/pros_app |
| **YOLO 整合** | https://github.com/asd56585452/ros2_yolo_integration |
| **WSL 安裝教學** | https://learn.microsoft.com/zh-tw/windows/wsl/ |

## 教學內容大綱

```
FINAL
├── URDF
├── Unity Topic Publish
├── Code Implementation
└── 環境檢查與系統教學
```

---

# 一、環境建置

## 1. WSL & Ubuntu 22.04 安裝

```powershell
# 在 Windows PowerShell 中執行
wsl --install -d Ubuntu-22.04
```

> ✅ 安裝完成後，進入 Ubuntu 終端機進行後續設定

## 2. VS Code 設定

1. 下載並安裝 [VS Code](https://code.visualstudio.com/download)
2. 安裝 **Remote Development** 擴充功能
3. 透過 WSL 連接 Ubuntu 環境

## 3. Docker 設定

1. 安裝 [Docker Desktop for Windows](https://docs.docker.com/desktop/setup/install/windows-install/)
2. 啟用 WSL 2 引擎：
   - 開啟 Docker Desktop → Settings → General
   - ✅ 勾選 **Use WSL 2 based engine**
3. 確保 Docker 可於 WSL 中正常運作

## 4. 環境驗證清單

| 項目 | 狀態 |
|------|------|
| Windows + WSL + Ubuntu 22.04 | ☐ |
| VS Code + Remote WSL | ☐ |
| Docker Desktop + WSL 2 整合 | ☐ |
| 工作目錄：`/home/{user}`（WSL 內） | ☐ |

## 5. 建議工作目錄結構

```
/home/{user}/
├── pros_car/                   # 主專案
├── pros_app/                   # SLAM/NAV2 應用
├── ros2_yolo_integration/      # YOLO 整合
└── unity_assets/               # Unity 相關資源
```

> 💡 **小提示**：所有開發建議在 WSL 的 Ubuntu 22.04 環境中進行，以確保與 ROS 2 的相容性。

---

# 二、系統啟動教學

## 1. 啟動 SLAM

進入 `pros_app` 目錄並執行：

```bash
cd ~/workspace/pros/pros_app/
python3 ./control.py -s
./slam_unity.sh
```

*(系統架構涉及：Windows / Unity Car Mode / AI / ROS2)*

## 2. Foxglove 設定與連線

1. 開啟 **Foxglove Studio**
2. 點擊 **Open Connection** → 選擇 **Rosbridge WebSocket**
3. 設定 WebSocket URL：`ws://localhost:9090`
4. 點擊 **Open Connection**

### Transform Frames 設定

設定以下 Frames：`map`、`base_footprint`、`camera`、`laser`、`arm_ik_base`

### 顯示 Topics 設定

| Topic | 說明 |
|-------|------|
| `/camera/color/camera_info` | 相機資訊 |
| `/camera/image/compressed` | 相機影像 |
| `/map` | 地圖 |
| `/scan` | 光達掃描 |
| `/plan` | 路徑規劃 |

設定 **Display frame** 為 `map`。

## 3. 鍵盤開車與掃描地圖

### 步驟一：啟動車輛控制節點

```bash
cd ~/workspace/pros/pros_car
./car_control.sh
ros2 run pros_car_py robot_control
```

### 步驟二：操控車輛掃描

1. 進入 **Main Menu**
2. 選擇 **Control Vehicle**
3. 使用鍵盤按鍵控制：

| 按鍵 | 功能 |
|:---:|:---|
| `w` | 前進 |
| `s` | 後退 |
| `e` | 右轉 |
| `r` | 左轉 |
| `c` | 儲存當前畫面至 `pros_car/src/pros_car_py/images/` |

*(可同步於 Foxglove 與 Terminal 中監控掃描狀態)*

## 4. LOCALIZATION 與路徑規劃

在 `pros_app` 的 Terminal 中依序執行：

1. 執行腳本 `b`
2. 儲存地圖：`./store_map.sh`
3. 執行 `b`
4. 執行 `d`
5. 啟動定位：`./localization_unity.sh`

## 5. 啟動 NAVIGATION（導航）

1. 回到 **Foxglove** 介面
2. 使用工具發佈訊息：
   - `/initialpose`（設定初始位置）
   - `/goal_pose`（設定目標點）
3. 點擊 **Publish** 啟動自動導航

---

# 三、自動導航

## Main Menu 選項

```
< Control Vehicle
< Manual Arm Control
< Manual Crane Control
< Auto Navigation
< Automatic Arm Mode
< Exit
```

## 導航模式選擇

```
[manual_auto_nav]   target_auto_nav   custom_nav
```

## 啟動步驟

1. 回到 `pros_car` Terminal
2. 選擇 **Auto Navigation**
3. 選擇 **manual_auto_nav**
4. 導航車輛至指定位置

### 導航所需 Topics

```
map  base_link  lidar  odom
```

## ⚠️ 注意事項

> **Note 1:** 導航過程中可能與橋樑或障礙物發生碰撞，此為預期行為，因為光達只能偵測較高的障礙物。

> **Note 2:** 若路徑規劃失敗，請手動控制車輛遠離牆壁與障礙物，避免 Nav2 誤判車輛位置在牆壁內部。

---

# 四、YOLO 影像辨識

## 步驟一：維持 Localization

```bash
./localization_unity.sh
```

## 步驟二：啟動 YOLO 節點

```bash
# 開啟新 Terminal，進入 ros2_yolo_integration，再進入 Docker
cd ~/workspace/pros/ros2_yolo_integration
./yolo_activate.sh

# 執行 YOLO 節點
ros2 run yolo_example_pkg yolo_node
```

## 步驟三：啟動機器人控制

```bash
# 開啟新 Terminal，進入 pros_car，再進入 Docker
cd ~/workspace/pros/pros_car
./car_control.sh

# 選擇：Auto Navigation -> manual_auto_nav，按 Enter
ros2 run pros_car_py robot_control
```

✅ **成功標誌**：出現數字輸出，且車輛開始自動搜尋球。

## Foxglove 影像面板設定

1. 點擊右上角 **+** 圖示，新增 **Image** 面板

| 設定項目 | 值 |
|---------|---|
| **Topic** | `/yolo/detection/compressed` |
| **Calibration** | `None` |
| **Rectify** | `Off` / `On` |
| **Sync annotations** | `Off` / `On` |

✅ **預期結果**：場景中放置網球時，YOLO 偵測結果會出現在影像面板中。

## YOLO 訓練資料：相機影像擷取

1. 回到 `pros_car` 的 **Main Menu**
2. 選擇 **Control Vehicle**
3. 使用鍵盤操控車輛掃描環境（`w` / `s` / `e` / `r`）
4. 按 `c` 儲存當前畫面至 `pros_car/src/pros_car_py/images/`

---

# 五、夾爪操作

## 手動夾爪控制

### 啟動步驟

1. 回到 VSCode，開啟新的 Terminal：
   ```bash
   cd ~/workspace/pros/pros_car
   ./car_control.sh r
   ros2 run pros_car_py robot_control
   ```
   > 💡 成功啟動後需出現閃電圖案。

**Terminal 啟動輸出範例：**

```text
jetson@ubuntu：~/workspace/pros_car$ ./car_control.sh
Testing Docker run with GPU...
Detected OS： Linux， Architecture： aarch64
GPU Flags： --runtime=nvidia
Detected architecture： arm64
root@aa118cf4bf96:/workspaces$
Starting >>> pros_car_py
Starting >>> robot_description
Finished <<< pros_car_py [3.00s]
Finished <<< robot_description [3.02s]
Summary： 2 packages finished [3.69s]
+ root@aa118cf4bf96:/workspaces$ ros2 run pros_car_py robot_control
```

### 操控方式

1. 正確啟動後進入 **Main Menu**
2. 選擇 **Manual Arm Control**
3. 選擇操控軸：`0`、`1`、`2`

| 按鍵 | 功能說明 |
|:---:|:---|
| `i` | 角度增加 |
| `k` | 角度減少 |
| `b` | 回到預設角度 |
| `q` | 返回上一層 |

## 自動夾爪操作

### 啟動步驟

1. 維持 `localization_unity.sh` 運行
2. 開車移動到目標（熊）的前方
3. 切換成 **Automatic Arm Mode** → `auto_arm_human`
4. 在 Foxglove 中使用 `click_point` 點擊熊的位置
5. 回到 VSCode 按下 `g`，自動夾取 `click_point` 標記的物件

### Unity 設定面板

```
SETTINGS
├─ Connected: [close]
├─ CAR: [O]
├─ ARM: [@]
├─ SENSOR: [C]
└─ CAMERA: [CONTROL]

ARM CONFIGURATION
├─ Mode: AI
├─ Finger: 136°
├─ Wrist: 30°
├─ Elbow: 90°
├─ Shoulder: 0°
└─ Base: 0°
```

> **注意**：在 Unity 介面中，將 **Arm → Mode** 切換為 **AI** 以啟用 ROS2 控制。

---

# 六、URDF 介紹

## URDF 簡介

**定義**：URDF（`Unified Robot Description Format`）是一種基於 XML 格式的語言，用於描述機器人的物理結構與運動學模型。

**核心組成元素：**

- **Link（連桿）**：描述機器人的剛體部分（如車體、輪子、機械臂連桿）。可定義其 `Visual`（視覺外觀）、`Collision`（碰撞邊界）與慣性參數（`Inertial`）。
- **Joint（關節）**：描述兩個 Link 之間的連接關係與運動方式（如固定 `fixed`、旋轉 `revolute`、連續 `continuous`）。

## URDF 在系統中的角色

- **建立 TF Tree（座標轉換樹）**：告訴系統機器人各個部位的相對位置。
- **視覺化與對齊**：在 Foxglove 中，能將 `map`、`base_footprint`、`camera`、`laser` 與 `arm_ik_base` 等座標系正確對齊顯示。
- **SLAM 與導航的基礎**：Nav2 與 SLAM 演算法必須知道雷達（`laser`）掃描到的點相對於車體中心（`base_footprint`）的精確距離與角度，才能正確建圖與避障。
- **手臂自動夾取**：需要知道物件位置相對於手臂的位置，才能自動計算機械臂的夾取角度。

## 目前系統的 URDF 結構

- **檔案位置**：`pros_app/docker/compose/demo/v6_unity.urdf`
- **Base（底盤）**：系統以 `base_footprint` 為基準原點。
- **Sensors（感測器）**：透過 `Fixed Joint` 將光達（`laser`）與相機（`camera`）固定在車體上的特定位置。
- **Arm（機械臂）**：透過 `Fixed Joint` 描述機械臂的底座位置 `arm_ik_base`。

## URDF 在 ROS2 的運作流程

1. **啟動節點**：當執行 `./localization_unity.sh` 或 `slam_unity.sh` 時，系統會啟動對應的 `docker-compose` 設定檔。
2. **Robot State Publisher**：ROS2 內建節點讀取 URDF 檔案。
3. **廣播 TF**：計算所有 Link 的位置後，向全域廣播 `/tf_static` 主題。

**啟動腳本範例（`slam_unity.sh`）：**

```bash
#!/bin/bash
source "./utils.sh"
main "./docker/compose/docker-compose_robot_unity.yml" "./docker/compose/docker-compose_slam_unity.yml"
```

## URDF 與 FINAL 實作關聯

當相機畫面經過 YOLO 辨識出目標（如網球），並計算出空間點位（`click_point`）後，系統需要依賴 URDF 提供的座標轉換，將「相機座標系」下的目標位置轉換為「機械臂基座座標系」下的目標位置，才能讓機械臂精準移動到該點進行夾取。

## URDF 關鍵零件整理

### 感測器（SENSORS）

系統中定義了實體感測器零件，以及供 ROS2 與 Foxglove 對齊使用的虛擬座標系。由於實體中心不等於計算需要的中心，通常會再生成虛擬 Link 表達實際使用的 Link：

| Link | 說明 |
|------|------|
| `camera` | 相機的基準座標系，由 `camera_1` 轉換而來 |
| `camera_1` | 實際相機的實體模型 |
| `camera_optical_frame` | 相機光學座標系，用於影像處理（Z 軸朝前） |

### 車輪（WHEELS）

四個獨立的車輪 Link，透過 Continuous Joint 與車體支架連接：

| Link | 說明 |
|------|------|
| `new_wheel_v1_1` | 車輪 1 |
| `new_wheel_v1_1_1` | 車輪 2 |
| `new_wheel_v1_2_1` | 車輪 3 |
| `new_wheel_v1_2_1__1` | 車輪 4 |

### 機械臂與夾爪（ARM & GRIPPER）

| Link | 說明 |
|------|------|
| `arm_ik_base` | 機械臂 IK 計算基準點，固定在馬達 `new_motor_v2_1` 上 |
| `big_U_self_v2_1` | 第 0 臂（對應第 0 軸旋轉中心），與 `arm_ik_base` 位置重合 |
| `big_U_self_v2_1_1` | 第 1 臂（對應第 1 軸旋轉中心） |
| `grap2_v2_1` | 夾爪其中一側（旋轉中心），連接在 `grap_base_v1_1` 上 |
| `grap2_v2_Mirror__1` | 夾爪另一側（鏡像旋轉中心），連接在 `grap_base_v1_1` 上 |

### 車體與其他（BODY & OTHERS）

| Link | 說明 |
|------|------|
| `base_footprint` | 車體在地面的投影原點，是整個 TF Tree 的 Root（根節點），也是系統導航與定位的基準 |
| `base_link` | 實際車體底盤的中心點，與 `base_footprint` 存在偏差 |
| `upper_blue_v1_1` / `upper_silver_v1_1` | 車體上方的藍色與銀色外殼零件 |

## 程式碼範例：座標轉換（`arm_controller_2D.py`）

程式碼路徑：`pros_car/src/pros_car_py/pros_car_py/arm_controller_2D.py`

以下為自動夾爪實作中，將目標點從 `map` 座標系轉換到 `arm_ik_base` 座標系的程式片段：

```python
if key == "g":
    try:
        # 建立目標的 PointStamped（原本在 map 座標系）
        target_map = PointStamped()
        target_map.header.frame_id = target_marker.header.frame_id  # 通常是 'map'
        target_map.header.stamp = self.ros_communicator.get_clock().now().to_msg()
        target_map.point = target_marker.pose.position

        # 將 map 上的網球，轉換到手臂基準座標系
        transform = self.tf_buffer.lookup_transform(
            self.base_link_name,
            target_map.header.frame_id,
            rclpy.time.Time()
        )
        target_base = tf2_geometry_msgs.do_transform_point(target_map, transform)
        x_target = target_base.point.x
        z_target = target_base.point.z

        print(f"目標相對基座座標：X={x_target:.3f}, Z={z_target:.3f}")

        # 開啟背景執行緒，執行「抓取與緩慢歸位」的完整排程
        threading.Thread(
            target=self.execute_grab_sequence,
            args=(x_target, z_target),
            daemon=True
        ).start()

    except Exception as e:
        print(f"▲座標轉換或TF失敗：{e}")

elif key == "b":
    # 重置手臂：讀取 init 裡面設定的初始角度
    self.joint_angles = [joint["init"] for joint in self.joint_limits]
    self._clamp_and_publish()
    self.visualize_arm_lines()
    print("手臂已重置為初始角度。")

elif key == "q":
    self.target_marker = None
    return
else:
    print(f"按鍵'{key}'無效，請使用 'g'(抓取)，'b'(重置)，或 'q'(取消)。")
    return
```

---

# 七、Unity 與 ROS2 的 Topic 通訊規範

## 發布（PUBLISH）—— Unity → ROS2

| Topic | 資料型別 | 用途 |
|-------|---------|------|
| `/scan_tmp` | `sensor_msgs/LaserScan` | 傳遞虛擬環境中光達掃描到的距離與強度數據，數值限制在 0.15m～16.0m 之間 |
| `/camera/image/compressed` | `sensor_msgs/msg/CompressedImage` | 傳遞車載 RGB 相機拍攝的畫面（JPG 格式壓縮） |
| `/camera/image/camera_info` | `sensor_msgs/msg/CameraInfo` | 傳遞 RGB 相機的內參矩陣（K、D、R、P 等畸變與投影參數） |
| `/camera/depth/compressed` | `sensor_msgs/msg/CompressedImage` | 傳遞深度圖，將 RFloat 深度數據經 EXR（ZIP 壓縮）後轉成 Base64 字串傳送 |
| `/camera/depth/camera_info` | `sensor_msgs/msg/CameraInfo` | 傳遞深度相機的內參矩陣 |

## 訂閱（SUBSCRIBE）—— ROS2 → Unity

| Topic | 資料型別 | 用途 |
|-------|---------|------|
| `/car_C_front_wheel` & `/car_C_rear_wheel` | `std_msgs/msg/Float32MultiArray` | 接收包含各 2 浮點數的陣列，對應虛擬車輛 4 個車輪的目標轉速，套用至 Unity 的 ArticulationBody 驅動車子 |

## YOLO Topics

| Topic | 說明 |
|-------|------|
| `/out/compressed` | 原始影像輸出 |
| `/yolo/detection/compressed` | YOLO 偵測結果（視覺化） |
| `/yolo/target_info` | 偵測目標的元資料 |
| `/camera/x_multi_depth_values` | 多點深度數值（20 個採樣點） |

### `/yolo/target_info` 訊息結構

```python
Index0 (found):    int    # 偵測旗標（1 = 偵測到）
Index1 (distance): float  # 與目標的距離（單位：m）
Index2 (delta_x):  float  # 相對畫面中心的水平偏移量
```

## Robot Arm Topic

| Topic | 資料型別 | 用途 |
|-------|---------|------|
| *(arm joint control)* | `trajectory_msgs/msg/JointTrajectoryPoint` | 機械臂各關節的目標角度控制 |

## 套件結構

```
ros2_yolo_integration/
└── src/
    └── yolo_example_pkg/
        └── yolo_example_pkg/
            └── object_detect.py    # 包含：self.x_num_splits
```

## 資料流示意圖

```mermaid
graph LR
    A[Camera Feed] --> B[YOLO Detection]
    B --> C[/yolo/target_info]
    C --> D[Trajectory Planner]
    D --> E[trajectory_msgs/JointTrajectoryPoint]
    E --> F[Robot Arm Controller]
    G[Depth Camera] --> H[/camera/x_multi_depth_values]
    H --> D
```

---

# 八、地圖隨機化說明

## 基本說明

為鼓勵學生使用更強健的演算法與模型，本年度地圖將完全隨機化：

- 橋樑、熊、路徑均會隨機化

| 隨機化前 | 隨機化後 |
|---------|---------|
| *（原始地圖）* | *（隨機地圖）* |

## 路徑隨機化原理

路徑隨機化透過 **Wave Function Collapse（WFC）** 與 **A\* 演算法**的組合實現：

| 組件 | 功能 |
|------|------|
| **A\* Algorithm** | 確保從起點到終點存在一條固定路徑 |
| **Wave Function Collapse（WFC）** | 遍歷剩餘的 Tile，選取可能選項最少的 Tile 進行 Collapse，並根據預定義規則將變化傳播至其他 Tile |

> 🔗 **WFC 參考**：https://github.com/mxgmn/WaveFunctionCollapse

## 版本說明

| 版本 | 說明 |
|------|------|
| **目前版本** | 2 張地圖，具有隨機化的路徑、橋樑位置和熊 |
| **未來版本** | 新增需拾取的物件 + 移動障礙物 |

| 地圖 | 說明 |
|------|------|
| **Final Project** | 路徑、橋樑位置和熊均隨機化 |
| **Racing2026** | 道路與橋樑位置固定，熊的位置隨機化 |

---

# 九、FINAL 實作功能方向建議

## 視覺與感知進階（Vision & Perception）

### Visual Servoing

```
YOLO → Error Calculation → PID Controller → cmd_vel (/cmd_vel)
```

目標：30 FPS

### Semantic Segmentation

| 方法 | 輸出 | 建議模型 |
|------|------|---------|
| YOLO | Bounding Box | YOLOv8 |
| YOLOv8-Seg | Mask | YOLOv8-Seg |

### 3D Bounding Box 偵測 / 6D Pose Estimation

- 估計：**Roll, Pitch, Yaw**
- 輸入：Depth images
- 框架：**ROS2**

## 導航與建圖進階（Navigation & SLAM）

### Visual SLAM

| 類型 | 輸入 | 推薦方案 |
|------|------|---------|
| **2D Visual SLAM** | RGB-D | RTAB-Map、ORB-SLAM3 |
| **3D Visual SLAM** | RGB-D + IMU | ORB-SLAM3 |

> 💡 需考慮 CPU/GPU 資源分配以確保即時效能

### Dynamic SLAM（動態 SLAM）

```
YOLO Detection → Dynamic Object Masking → SLAM Pipeline → Real-time Mapping
```

## 控制、決策與系統整合（Control & Decision）

### 3D Navigation / OctoMap

| 組件 | 說明 |
|------|------|
| **2D Costmap** | 傳統 2D 導航成本表示 |
| **3D OctoMap** | 3D Voxel-based 佔用建圖，適用於複雜環境 |
| **Nav2** | ROS2 導航框架（主要為 2D） |
| **MoveIt2** | ROS2 3D 操作與運動規劃框架 |

### Dynamic Obstacle Avoidance（動態避障）

**Nav2 Local Planner 選項：**
- `TEB Local Planner`：Timed Elastic Band，適合動態環境
- `MPPI`：Model Predictive Path Integral，基於採樣的優化

**調整建議：**
- 解決 *Freezing Robot Problem*（機器人被障礙物包圍時停止移動）
- 調整 costmap inflation、障礙物閾值與規劃器參數

### Advanced Driving Model（進階車輛驅動模型）

- 以真實車輛動力學取代基本的 `ROS cmd_vel`
- 整合 `EKF`（Extended Kalman Filter）進行感測器融合：
  - IMU + Odometry → 穩健的狀態估計
  - 處理 `Blind Navigation` 情境（協方差感知濾波）
- 為 `Nav2` 配置精確的 `Odometry` 與 `Covariance` 參數

### Active Exploration（主動探索）

- **Frontier-Based Exploration**：使用 `explore_lite` 套件進行高效未知區域建圖
- **視覺整合**：結合 `YOLO` 進行物件感知探索與語意建圖

---

# 十、評分方式

## 登入方式

```
Username + Key-based Authentication（EASY）
```

## 任務評分表

| 任務 | 子任務 | 分數 | 備註 |
|------|--------|------|------|
| **Task 1** | Locate & Observe | 10 分 | 5 分部分給分 |
| | Recovery | 20 分 | 故障恢復能力 |
| **Task 2** | Ascent | 10 分 | 斜坡地形導航 |
| | Descent | 10 分 | 受控下坡移動 |
| | Recovery | 10 分 | 斜坡上的錯誤處理 |
| **Task 3** | Locate & Observe | 10 分 | 目標識別 |
| | Unlock | 20 分 | 互動/操作任務 |
| | Clear | 10 分 | 區域清除或目標完成 |

> Task 2 與 Task 3 可能涉及複雜情境執行的合併評估標準。

---

# 十一、Release Notes

## 4/20 更新

### UNITY 環境發布 Linux 版本

**下載檔案**：`pros_twin_unity_linux_V3.zip`（Vulkan 支援）

**安裝與啟動指令**：

```bash
# 賦予執行權限
chmod +x pros_twin_unity_tsai_run_linux.x86_64

# 啟動應用程式（強制使用 Vulkan）
./pros_twin_unity_tsai_run_linux.x86_64 -force-vulkan
```

### 環境支援說明

| 組件 | 說明 |
|------|------|
| `pros_app`、`pros_car`、`yolo` | Linux 環境支援 |
| Docker | 支援 Linux OS / WSL |
| Docker Image | `pull docker image` |
| Unity Vulkan | 圖形渲染後端 |
| Unity OS | 支援 Windows / Linux |

> **注意**：Windows 使用者可透過 WSL 執行 Unity Linux 版本；Linux 使用者直接執行原生版本。

### ROS2_YOLO_INTEGRATION 顯卡支援

```bash
# CUDA 12.8 版本（推薦用於 NVIDIA 50 系列顯卡）
source yolo_activate_cu128.sh

# 一般 CUDA 版本
source yolo_activate.sh
```

### PROS_APP 相關腳本

```bash
./localization_unity.sh    # 定位功能
./slam_unity.sh            # SLAM 建圖
./rosbridge_server.sh      # ROSBridge 伺服器
```

Foxglove 整合支援 `foxglove_bridge` 與 `Foxglove WebSocket`，資料流：Unity ↔ Rosbridge ↔ Foxglove Topic。

## 4/27 更新

### PROS_APP 更新

- 新增 **Navigation2（NAV2）** 支援
- SLAM 補充使用說明：
  - `initpose`：初始位置設定
  - `final project slam`：期末專案 SLAM 應用
  - `racing2026`：競賽場景標籤

### V4.1 場景更新

| 項目 | 說明 |
|------|------|
| Unity racing2026 | 競賽場景整合 |
| ROSBridge | Unity ↔ ROS2 通訊橋接 |
| 平台支援 | Linux / Windows，V4 → V4.1 升級 |

### 預設帳號密碼

```
帳號: test
密碼: TEST1234
備註: easy
```

---

# 附錄：快速指令參考

```bash
# 啟動 SLAM
cd ~/workspace/pros/pros_app/
./slam_unity.sh

# 啟動定位
./localization_unity.sh

# 儲存地圖
./store_map.sh

# 啟動車輛控制
cd ~/workspace/pros/pros_car
./car_control.sh
ros2 run pros_car_py robot_control

# 啟動 YOLO
cd ~/workspace/pros/ros2_yolo_integration
./yolo_activate.sh
ros2 run yolo_example_pkg yolo_node

# Foxglove Rosbridge URL
ws://localhost:9090

# YOLO 偵測 Topic
/yolo/detection/compressed
```
