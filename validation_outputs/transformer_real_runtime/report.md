# Transformer CPU 真实推理验证

本次使用 `D:\anaconda\envs\human-fall\python.exe`（Python 3.10.21，64 位 Windows），未安装、升级或卸载任何依赖。未调整阈值、batch、采样方式、滤波开关或推理间隔。

## 实际模型加载与独立调用

解释器：`from ai_edge_litert.interpreter import Interpreter`。模型为当前工程 `models/fall_detection_transformer.tflite`。解释器创建和 `allocate_tensors()` 成功。

| 张量 | shape | shape_signature | dtype |
| --- | --- | --- | --- |
| 输入 `serving_default_input_features:0` | `(1,30,51)` | `(-1,30,51)` | float32 |
| 输出 `StatefulPartitionedCall_1:0` | `(1,1)` | `(-1,1)` | float32 |

签名的 batch 维标记为动态；本次所有调用固定为 batch=1，没有调用 resize，也没有验证多人 batch。

独立输入为 `np.linspace(-1,1,1530,dtype=np.float32).reshape(1,30,51)`，所有元素有限。实际执行 `set_tensor → invoke → get_tensor`，得到：

```text
[[0.7568454146385193]]
shape=(1,1), dtype=float32
```

输出是有限的合法概率。此项是模拟输入上的真实模型调用，不是视频识别结果。

## 完整真实视频链路

原始视频：

```text
D:\study\大创\测试数据\HumanPose\wave\wave\Bubble_Gum_Club_2_wave_h_nm_np2_fr_med_0.avi
```

分辨率 320×240。视频容器元数据标称 79 帧，独立顺序解码实际为 78 帧。检测程序处理全部 78 个可解码帧后正常退出。没有重复、遮挡或修改视频帧。

通过实际 `fall_detection_system.run_cli_mode()` 运行：本地 YOLOv12 权重 `yolov12n1.pt`（CPU）→ ByteTrack → 每人独立 MediaPipe Pose → 原始关键点预处理 → 每人 30 帧缓存 → LiteRT Transformer → 实际 CLI 报警事件/快照出口。

为无窗口验证，仅屏蔽 `cv2.imshow` 与键盘轮询；不替换检测器、跟踪器、姿态模型或分类模型的推理结果。记录包装器调用原实现，并检查中间数据。CLI 的快照功能保持启用，若产生真实报警会调用原 `save_frame` 保存到本报告目录下的 `alarm_snapshots`。

| person_id | 序列更新 | 成功姿态输出 | 是否积满 30 帧 | 真实 Transformer 调用 | 最小 Fall 概率 | 最大 Fall 概率 |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| 1 | 78 | 78 | 是 | 17 | 0.0286559425 | 0.0427256413 |
| 2 | 78 | 78 | 是 | 17 | 0.0489337668 | 0.0512008294 |

两人的实际调用均发生在各自第 `30,33,36,39,42,45,48,51,54,57,60,63,66,69,72,75,78` 次更新，共 34 次视频 Transformer 调用。两人恰好全程同时被跟踪，因此本视频中这些更新次数也等于视频帧号；不同入场时间下的独立计数另由回归模拟测试覆盖。

| 更新次数 | ID 1 概率 | ID 2 概率 |
| --- | ---: | ---: |
| 30 | 0.0415509082 | 0.0511877500 |
| 33 | 0.0417122170 | 0.0510464236 |
| 36 | 0.0424646586 | 0.0511505827 |
| 39 | 0.0427256413 | 0.0510792583 |
| 78 | 0.0286559425 | 0.0489337668 |

## 运行中通过的核对

- 每个 ID 对应不同的 Pose 对象，同一 ID 跨帧复用同一对象。
- 156 次成功姿态分析逐一比对真实 MediaPipe 输出、滤波前原始副本和 Transformer 特征来源。
- 156 次人工姿态特征调用接收实际 One Euro 输出；其中 154 次滤波输出与原始点不同，Transformer 仍读取原始支路。本次 Visibility Filter 保持默认关闭；开启分支由回归模拟测试覆盖。
- 每人 deque 独立；每帧追加的特征与该 ID 的参考窗口逐项一致。处理某人时，其他人的概率及间隔计数保持不变。
- 非推理帧也更新窗口，保留上次概率，不调用分类器。推理时输入为 float32 `(1,30,51)`；模型返回概率写入对应 ID 的 `fall_data`。
- `fallen_ids` 与阈值判断一致，`fall_event_ids` 与“本帧新推理且概率达到 0.9”一致。
- 视频结束后 Pose 各关闭一次，视频句柄释放，所有人员状态清空，分类器及解释器引用置空；额外重复调用 `close()` 无错误。

本视频两人概率始终小于 0.9。因此 `fall_event_ids` 始终为空，实际 CLI 报警消费者没有产生报警或快照。没有伪造概率或降低阈值。**阳性结果进入报警出口、阳性旧概率不重复报警的行为，本次仍只有模拟测试证据，不能称为真实视频阳性报警验证通过。** 本视频覆盖了真实阴性结果及其缓存不会触发报警。

本次没有运行 Dashboard 可见窗口或声音报警；其事件和生命周期回归测试通过不等于真实 GUI 报警实测。视频没有自然离场超过 5 秒的情况，过期清理由现有回归测试覆盖。

## 发现的问题与修改

1. 初次实际加载发现：LiteRT 通过纯英文相对路径加载成功，但通过含中文目录的绝对 `model_path` 加载失败，报 `ValueError: Could not open ...`。已在 `models/transformer_fall_classifier.py` 改用 `self.model_path.read_bytes()` + `Interpreter(model_content=...)`。模型路径仍依据模块位置解析，字节内容保持原文件不变。修复后实际初始化、完整视频推理均成功。
2. 验证脚本最初直接把容器标称帧数作为断言，因 79/78 差异失败；改为独立顺序解码后核对实际帧数。最终复跑完整通过，退出码 0。
3. 启动仍出现工程已有的 `attempt_download` 导入失败提示，原代码捕获后成功加载本地 YOLO 权重。没有执行下载或修改相关逻辑。

最终运行中没有检测、模型调用、数据核对或资源释放异常。

本次修改/新增：

- `models/transformer_fall_classifier.py`：中文绝对路径加载修复。
- `tests/test_transformer_integration.py`：模拟解释器接收模型字节，并核对内容与模型文件一致。
- `tests/validate_transformer_runtime.py`：可复跑的真实 CPU 模型及完整 CLI 视频验证脚本。
- 本目录结果、画面样例、报告，以及上级目录的运行日志和回归日志。

## 回归与复现

现有 **34 项测试全部通过**，包含原 Pose 隔离、复用、过期、并发关闭、主程序/Dashboard 生命周期及 Transformer 接入测试。这些测试使用模拟推理接口，与上述 34 次真实视频模型调用是两个不同计数。

前次接入的 6 个 Python 文件及本次新增的验证脚本，共 **7 个文件语法检查通过**。

```powershell
& 'D:\anaconda\envs\human-fall\python.exe' tests/validate_transformer_runtime.py
& 'D:\anaconda\envs\human-fall\python.exe' -m unittest discover -s tests -v
```

运行目录为当前工程根目录。复跑验证脚本会覆盖本目录的结果文件和示例图，不安装依赖。

逐帧 `fall_data`、每人计数和每次真实概率见 `results.json`；实际运行日志见上级 `transformer_real_runtime.log`；回归日志见上级 `transformer_regression.log`；语法记录见 `syntax.txt`。
