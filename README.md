# 2D 目标检测实战（nuScenes mini × YOLO11）

这是"2D / 3D 目标检测渐进教程"的 **2D 半区**。它把 nuScenes 的相机图像转成 YOLO 数据集，
用单阶段检测器 YOLO11n 完成训练与评估，最后渲染成一段行车检测视频。

> 3D 半区（激光雷达点云 + CenterPoint）是另一个独立仓库 `YXYCDWX/3d`，两份代码完全独立，可分别运行。
> 2D 与 3D **共用同一套类别顺序与场景划分**，方便你横向对照两条技术路线。

---

## 1. 这一半在解决什么问题

给一张车载相机照片，让计算机回答两件事：

1. **图里有什么？** —— 分类（汽车 / 行人 / 自行车……）
2. **它们在哪里？** —— 定位（用矩形框 `(x1, y1, x2, y2)` 标出范围）

分类 + 定位 = 检测。2D 检测输出的是物体**在画面中的位置**；要得到物体**在真实空间里的位置与朝向**，
需要 3D 检测（见 3D 半区仓库 `YXYCDWX/3d`）。

nuScenes 检测任务统一为 **10 个类别**（2D / 3D 共用同一顺序）：

```
car, truck, construction_vehicle, bus, trailer,
barrier, motorcycle, bicycle, pedestrian, traffic_cone
```

---

## 2. 目录结构

```
2d/
├─ README.md               # 本文档
├─ nuscenes_common.py      # 【库】解析 nuScenes JSON + 坐标系变换（全局↔传感器）
├─ convert_2d.py           # 数据：nuScenes → YOLO 2D 数据集（nu2d/）
├─ train_yolo.py           # 训练：YOLO11n
├─ eval2d.py               # 【库】统一 2D 评估器（COCO 风格 mAP）
├─ yolo11n.pt              # YOLO11n 预训练起始权重（Ultralytics 官方）
├─ tools/                  # 常用工具
│   ├─ eval_unified.py     #   统一口径 mAP + 预测可视化
│   ├─ report_confusion.py #   固定工作点下的查准 / 查全与混淆矩阵
│   ├─ audit_2d.py         #   标签审计：全量重算比对 + 巨框统计
│   └─ make_video_2d.py    #   出片：CAM_FRONT 逐帧检测行车视频
├─ v1.0-mini/              # 原始数据集（自行下载，不入库）
├─ nu2d/                   # 转换后的 YOLO 数据集（脚本生成，不入库）
└─ outputs/                # 训练 / 评估产物（脚本生成，不入库）
```

> **命名约定**：标注 `【库】` 的文件不单独运行，供其他脚本 `import` 使用。

---

## 3. 快速开始（从零到出片）

下面命令都假定**已经位于本仓库根目录**。全部用相对路径，换机器、换盘符都不用改代码。

### Step 0 · 环境

需要 Python 3.11 + PyTorch(CUDA) + ultralytics + numpy + opencv + pillow。实测通过的版本：

| 包 | 版本 |
|---|---|
| python | 3.11.16 |
| torch | 2.9.0+cu128 |
| ultralytics | 8.4.138 |
| numpy | 2.4.4 |
| opencv-contrib-python | 5.0.0.93 |
| pillow | 12.2.0 |

```powershell
conda create -n dl python=3.11 -y
conda activate dl
pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
pip install ultralytics==8.4.138 numpy==2.4.4 opencv-contrib-python==5.0.0.93 pillow==12.2.0
```

验证：

```powershell
python -c "import torch, ultralytics, cv2; print(torch.__version__, torch.cuda.is_available())"
```

### Step 1 · 准备数据

到 [nuScenes 官网](https://www.nuscenes.org/download) 下载 `v1.0-mini.tgz`，解压到**本仓库根目录**下，
使结构为：

```
（本仓库根目录）/
└─ v1.0-mini/            # 注意：v1.0-mini 直接放在仓库根目录
   ├─ v1.0-mini/         # 元数据 JSON（scene / sample / sample_annotation / ...）
   ├─ samples/           # 6 路相机关键帧 jpg + LIDAR_TOP 关键帧 bin
   ├─ sweeps/            # 非关键帧点云 bin
   └─ maps/  demo/  LICENSE  .v1.0-mini.txt
```

> 路径由脚本里的 `DATAROOT` 决定（= 本仓库根目录下的 `v1.0-mini/`），请勿改动层级。

### Step 2 · 转换数据

```powershell
python convert_2d.py
```

- 输入：`v1.0-mini/`
- 输出：`nu2d/images/{train,val}`、`nu2d/labels/{train,val}`、`nu2d/data.yaml`
- 做法：遍历 6 路相机的关键帧，把 3D 标注框投影到图像，得到 2D 框（YOLO 归一化格式）

### Step 3 · 训练

```powershell
python train_yolo.py yolo11                      # 默认 60 轮
python train_yolo.py yolo11 --aug strong         # 打开更强的空间/混合增强
python train_yolo.py yolo11 --aug strong --kfold 4   # 场景级 4 折交叉验证
```

- 起点：官方预训练权重 `yolo11n.pt`
- 常用参数：`--epochs / --imgsz / --batch / --aug / --tag / --kfold`
- 产物：`outputs/yolo11/train/weights/best.pt`、`last.pt`，以及 `outputs/yolo11_results.json`
- **断点续训**：中途停掉再跑同一条命令，会从 `last.pt` 自动接上未完成的轮次

### Step 4 · 评估

```powershell
python tools/eval_unified.py outputs/yolo11/train/weights/best.pt
```

- 终端打印 `mAP@0.5` 与 `mAP@0.5:0.95`，并逐类给出 AP 与 GT 框数
- 结果写入 `outputs/yolo11/unified_results.json`
- 同时在 `outputs/viz_check/` 输出可视化（**绿框=真值，红框=预测**）

### Step 5 · 出片

```powershell
python tools/make_video_2d.py            # 训练集前 8 场景
python tools/make_video_2d.py 12 0.3 0 --split val   # 验证集后 2 场景
```

- 输出：`outputs/videos/yolo11_cam_front.mp4`（验证集为 `..._val.mp4`）
- 参数：`fps`（默认 12，nuScenes 原生帧率）、`thr`（默认 0.3）、`max_frames`（0=全部）

---

## 4. 关键概念（本半区最值得弄懂的地方）

### 4.1 从 3D 框投影出 2D 框

nuScenes 的标注框是**全局坐标系**下的 3D 框。要把它画到某张相机照片上，需要走一遍坐标变换：

```
全局坐标 ──(取逆)──► 传感器坐标 ──(相机内参 K 投影)──► 像素坐标 (u, v)
```

对应代码：`nuscenes_common.py` 的 `frames_for_sample_data()`（组装"全局→传感器"矩阵与内参）与
`convert_2d.py` 的 `project_box_to_image()`（角点投影 + 取包围盒）。

**一个必须牢记的约定（nuScenes 官方规定）**：3D 框尺寸写作 `(w, l, h)`，其中

- 长 `l` 沿物体局部 x 轴（即它的朝向 / heading 方向）
- 宽 `w` 沿局部 y 轴（横向）
- 高 `h` 沿 z 轴

把 `w` 和 `l` 写反，框就会"横过来"，这是新手最常见的错误之一。代码里 `box_corners()` 把这个约定落到了实处。

### 4.2 为什么要丢掉"贴着相机"的框

`project_box_to_image()` 要求 3D 框的 **8 个角点全部位于相机前方**（深度 `z > 0.5 m`）。
原因很直观：若某角点贴近相机平面，深度接近 0，投影坐标 `u = x / z` 会"爆炸"，算出横跨整张图的错误巨框。
代码的做法是**直接丢弃**这类框，而不是硬算。

### 4.3 为什么"按场景"划分训练/验证

mini 共 10 个场景，脚本按场景名排序后取**前 8 个训练、后 2 个验证**。

> 同一场景里相邻帧内容高度相似。若按"图片"随机划分，训练集与验证集会混入几乎一样的图，
> 指标会虚高——这叫**数据泄漏**。按场景切分才能反映真实的泛化能力。
> `train_yolo.py --kfold K` 更是严格做到"每折留一整个场景做验证"。

### 4.4 YOLO 标签格式

`nu2d/labels/*.txt` 每行一个目标：`类别 cx cy w h`，后四个都是 **0~1 归一化**值
（相对图像宽高的中心点与宽高）。`data.yaml` 里只写 `train: images/train` 这类**相对路径**，
让 Ultralytics 以 `data.yaml` 所在目录为数据集根——因此整个 `2d/` 目录可以任意搬移。

### 4.5 数据增强小课（`--aug strong`）

- **空间类**：随机旋转、缩放、平移、剪切、透视——让模型"物体转个角度也认得出"
- **混合类**：mosaic（四图拼接）、mixup（两图叠加）——增加样本多样性
- **颜色类**：只做常规 HSV 抖动（亮度/饱和度/色相），**不做反色、不做灰度**
  ——颜色是判断类别的重要线索，破坏颜色会让标签变得不可信

---

## 5. 命令速查

| 用途 | 命令 |
|---|---|
| 转数据 | `python convert_2d.py` |
| 训练（60 轮） | `python train_yolo.py yolo11` |
| 断点续训 | 同一条训练命令，自动续训 |
| 统一评估 | `python tools/eval_unified.py <weights> [viz_n]` |
| 查准/查全 + 混淆矩阵 | `python tools/report_confusion.py` |
| 标签审计 | `python tools/audit_2d.py` |
| 出片 | `python tools/make_video_2d.py [fps] [thr] [max_frames] [--split train\|val]` |

---

## 6. Windows / 中文路径注意事项

- **DataLoader 固定单进程**：所有训练脚本都写死 `num_workers=0`。Windows 上多进程数据加载容易死锁卡住，
  教学环境稳定优先：宁可慢一点，也要一次跑通。
- **只用相对路径**：脚本内部一律用 `os.path.dirname(__file__)` 定位，`data.yaml` 不写绝对 `path:`，
  因此目录放到中文路径、换盘符都能正常运行。
- **显存不够**：下调 `train_yolo.py --batch`（默认 16）即可。
- **视频写入**：出片中途 `Ctrl+C` 硬停会破坏 mp4 索引，请等它正常收尾；若 mp4v 不可用，脚本会自动回退 avi。

---

## 7. 常见问题（FAQ）

**Q：为什么不直接用官方 `nuscenes-devkit`？**
A：为了教学透明——直接读 JSON 并自己写坐标变换，你能完整看到"数据是怎么被解析、被变换的"，不依赖黑盒封装。

**Q：训练到一半能停吗？**
A：可以。重跑训练命令会自动从 `last.pt` 接续未完成的轮次。

**Q：评估为什么要"统一口径"？**
A：`eval2d.py` 提供统一的 GT 读取与 AP 计算（COCO 风格 `mAP@0.5` / `mAP@0.5:0.95`，全点插值），
`tools/eval_unified.py` 直接用同一把尺子量权重，保证不同实验之间可横向比较。

**Q：想换模型 / 加实验怎么办？**
A：模型清单集中在 `train_yolo.py` 顶部的 `MODELS` 常量里；`--tag` 可给输出目录加后缀，
方便多组实验并存对比。

---

## 8. 下一步

- 想理解"无锚框检测"是怎么工作的？去看 3D 半区仓库 `YXYCDWX/3d` 的 CenterPoint（手写实现，注释对齐 OpenPCDet 口径）。
- 想读懂损失函数？打开该仓库的 `models3d.py`，逐行对照 `CenterHead.loss` 与其中 focal 公式的注释。
