# 工程整理与回归验证

本次保留 One Euro 算法、参数和全部已有 One Euro 测试；未安装、升级或卸载任何依赖。三个核心模块继续分离，没有合并成大型文件。Transformer 模型、预处理公式、关键点顺序、阈值和推理间隔未改动。

## 文件修改与删除依据

| 文件 | 整理内容 |
| --- | --- |
| `models/fall_detector.py` | 移除低可见度冻结分支、旧多类型分类和规则报警及无人调用的比例辅助函数；保留 One Euro 与人工特征；过期和关闭共用 `_clear_person_state()`。 |
| `fall_detection_system.py` | 删除无效的旧规则阈值参数和赋值；Transformer 事件与连续状态去重语义保持不变。 |
| `dashboard/dashboard_app.py` | 删除旧阈值控件、无效灵敏度控件和具体类型选项；仅保留二分类统计、固定 0.9 阈值说明及原报警冷却。 |
| `experiments/analyze_false_positive.py` | 保留 One Euro 人工特征记录，移除旧门控配置；仅新 Transformer 结果启动误报分析窗口，并保持连续摔倒状态去重。 |
| `tests/test_pose_lifecycle.py` | 保留所有原测试；删除已不存在状态的断言，新增每人/每轴 One Euro 独立性、低可见度持续更新和无跟踪条目时的辅助状态清理检查。 |
| `tests/test_transformer_integration.py` | 保留原始/平滑支路核对及所有原测试；覆盖低可见度下仍执行 One Euro；新增误报分析事件去重测试。 |
| `tests/validate_pose_runtime.py` | 移除测试启动参数中的旧规则阈值，保留生命周期、启停与历史对照验证能力。 |
| `tests/validate_transformer_runtime.py` | 更新清理检查；结果写入新目录，并逐次比较整理前每人推理时机和概率，容差 1e-6。 |
| `README.md`、`fall_detection_algorithm.md` | 更新为当前双支路、Transformer 二分类、模块职责和生命周期说明，删除旧算法描述与启动参数。 |

删除三个源码文件：

1. `fall_detection.py`：删除前搜索主程序、Dashboard、测试、配置、实验脚本和文档；没有代码引用。只有历史验证报告提及这个旧文件，属于文字记录，不是运行依赖。
2. `utils/visibility_filter.py`：先移除检测器及误报分析脚本的引用，再删除其专用对照实验后，已无剩余运行调用。
3. `experiments/evaluate_fall_detection.py`：内容专用于已删除的坐标门控开关对照；引用检索未发现其他代码、配置或测试依赖该脚本。

清除了工程自己的旧 `.pyc` 缓存，没有触碰虚拟环境。没有删除视频、模型、实验 CSV 或历史验证证据。没有合并文件；仅将人员清理操作集中到同一私有方法。

`utils/one_euro_filter.py`、`models/transformer_fall_classifier.py` 和模型文件在本次整理中未修改。One Euro 的 `min_cutoff=1.0`、`beta=0.007`、`d_cutoff=1.0` 保持不变。

## 整理后的结构

```text
Human-Fall-new/
├─ fall_detection_system.py
├─ models/
│  ├─ fall_detector.py
│  ├─ transformer_fall_classifier.py
│  ├─ fall_detection_transformer.tflite
│  └─ Fall-Detection-LICENSE.txt
├─ utils/
│  ├─ one_euro_filter.py
│  └─ utils.py
├─ dashboard/dashboard_app.py
├─ tests/
│  ├─ test_pose_lifecycle.py
│  ├─ test_transformer_integration.py
│  ├─ validate_pose_runtime.py
│  └─ validate_transformer_runtime.py
├─ experiments/
│  ├─ analyze_false_positive.py
│  ├─ analyze_visibility_distribution.py
│  ├─ analyze_visibility_jump.py
│  ├─ evaluate_pose_smoothing.py
│  ├─ evaluate_tracking.py
│  ├─ tracking_test.py
│  ├─ tracking_logger.py
│  └─ logs/
├─ validation_outputs/
│  ├─ transformer_real_runtime/       # 整理前比较基准
│  └─ transformer_cleanup_runtime/    # 本次结果
├─ fall_snapshots/
├─ README.md
├─ fall_detection_algorithm.md
└─ requirements.txt
```

目录图省略已有模型权重、IDE 文件和其他未变更文件。保留原始 visibility 分布、位移分析脚本，因为它们测量 MediaPipe 数据，不是坐标过滤器；One Euro 平滑实验也保留。

## One Euro 实际数据流

```text
每人独立 MediaPipe Pose → 原始 33 点 (x,y,z,visibility)
├─ 不可变原始副本 → 指定 17 点 → 同边界完整帧坐标恢复 → 部署归一化
│                 → 该人的 30 帧窗口 → Transformer → 最终判断
└─ pose_filters[person_id][landmark_id][axis]
   → x/y/z 独立 One Euro；visibility 原样传递
   → calculate_pose_features() → pose_histories[person_id]
   → 角度、速度、加速度、jerk、宽高比、髋/肩/足位置等辅助分析
```

即使当前 visibility 很低，当前坐标也会进入 One Euro，不冻结上次坐标。Transformer 始终读取滤波前保存的原始点。人员过期和整体关闭均清理 Pose、One Euro、人工历史、序列、概率、计数和报警状态，关闭失败的 Pose 仍可重试；处理和关闭继续使用同一生命周期锁。

## 全部验证结果

- 环境：`D:\anaconda\envs\human-fall\python.exe`，Python 3.10.21，Windows x64，CPU。
- 原有 34 项测试均保留；新增 3 项后共 **37 项测试全部通过**。没有删除 One Euro 测试。
- 全工程 **18 个 Python 源文件语法检查通过**，包括保留的实验脚本。
- 当前源码、测试、IDE/应用配置及当前算法/使用文档的旧过滤器、具体类型和旧规则标识扫描命中 **0**。扫描细节见 `audit.json`。
- **历史验证报告、运行日志和实验 CSV 作为不可改写的历史证据保留，其中可能仍出现旧术语。这些文字不是现存功能或运行代码；本报告不声称历史证据中的旧名称也被抹除。**
- 工程 Transformer 模型 SHA-256 与上游本地模型一致：`c66659c3ac6d37c60469bee4d6f88cb2ac4976928bde6d46b19e7f851e8980b0`。

真实 LiteRT：`ai_edge_litert.interpreter.Interpreter` 创建并分配张量成功，仍通过 `model_content` 兼容中文路径。输入 `(1,30,51)`、签名 `(-1,30,51)`、float32；输出 `(1,1)`、签名 `(-1,1)`、float32。未 resize，所有调用 batch=1。

有限模拟输入 `np.linspace(-1,1,1530,dtype=np.float32).reshape(1,30,51)` 的真实模型输出为 **0.7568454146385193**。这属于独立真实模型调用，不是视频识别结果。

完整真实视频使用：

```text
D:\study\大创\测试数据\HumanPose\wave\wave\Bubble_Gum_Club_2_wave_h_nm_np2_fr_med_0.avi
```

原视频未修改，320×240，实际顺序解码并处理 78 帧（容器标称 79 帧）。运行实际 CLI 链路及报警消费者，只屏蔽窗口显示和键盘轮询。

| ID | 更新数 | Pose 成功数 | 真实 Transformer 次数 | 概率范围 | 与整理前最大绝对差 |
| --- | ---: | ---: | ---: | --- | ---: |
| 1 | 78 | 78 | 17 | 0.0286559425 ～ 0.0427256413 | 0.0 |
| 2 | 78 | 78 | 17 | 0.0489337668 ～ 0.0512008294 | 0.0 |

两人均在各自第 30、33、36、39……78 次更新进行推理，其他帧也进入队列。每人序列、Pose、概率和计数独立。逐次概率与整理前完全一致。真实运行完成 156 次原始点来源检查、156 次人工特征平滑来源检查。资源释放与重复关闭全部通过。

本视频没有概率达到 0.9，没有真实报警事件或快照；没有制造阳性结果。阳性报警和缓存阳性不重复报警由模拟测试覆盖，不能将其称为真实视频阳性验证。本视频没有缺失姿态或自然离场超过 5 秒，相关行为由保留的模拟测试覆盖。

另执行真实 CLI 两轮、Qt offscreen Dashboard 两轮启停、窗口 closeEvent 和重复退出回调，共 5 个检测器，错误列表为空。GUI 检查没有显示窗口或触发声音，也没有据此宣称完成真实阳性报警。

最终没有推理、数据核对或释放异常。仍有原工程已有 `attempt_download` 导入提示，捕获后正常加载本地 YOLO 权重；本次没有调整下载逻辑或依赖。

证据：`results.json`（逐帧 fall_data、每次概率、基准比较和释放状态）、`regression.log`、`audit.json`、`application_restart.json`、`applications.log`，以及上级目录 `transformer_cleanup_runtime.log`。
