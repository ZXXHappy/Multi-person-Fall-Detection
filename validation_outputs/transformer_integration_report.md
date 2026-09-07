# Transformer 接入修正与验证

日期：2026-09-07。代码接入和模拟验证已完成；未安装、升级或卸载依赖。真实 TFLite 模型调用与完整真实视频推理尚未完成。

## 修改范围

复用现有 `models/transformer_fall_classifier.py` 和 `models/fall_detection_transformer.tflite`，没有新增功能重复的特征模块。修改的 Python 文件为：

1. `models/fall_detector.py`：原始关键点支路、每人推理计数、缓存概率、报警事件标志、清理及 CPU 模式。
2. `models/transformer_fall_classifier.py`：stride 配置、异常数值清理、解释器逐项导入/模型加载回退、输入输出契约校验、路径解析。
3. `dashboard/dashboard_app.py`：仅新推理阳性结果进入报警冷却和事件记录，清理过期 ID 的报警记录状态。
4. `fall_detection_system.py`：CLI 仅记录新推理产生的新摔倒状态，缓存概率不产生新事件；阴性帧正确清除当前摔倒状态。
5. `tests/test_pose_lifecycle.py`：沿用 Pose 隔离和并发释放测试，补充新增状态清理检查。
6. `tests/test_transformer_integration.py`：新增原始/平滑支路、stride、概率缓存、报警消费者、解释器契约等测试。

## 最终数据流

```text
YOLOv12（CPU）→ ByteTrack person_id → 有效人物框裁剪 → 每人独立 MediaPipe Pose
                                                        ├─ 原始 33 点独立副本
                                                        │  → 指定 17 点
                                                        │  → 完整帧归一化坐标
                                                        │  → 部署骨架归一化 + nan_to_num
                                                        │  → 每人 deque(maxlen=30)
                                                        │  → 单解释器、batch=1、逐人推理
                                                        │  → Fall 概率与报警
                                                        └─ One Euro / 可选 Visibility Filter
                                                           → 原有角度、速度等人工特征
```

原始关键点以独立 tuple 保存，提取 Transformer 特征时不读取平滑返回值。缺失姿态会清除该 ID 的旧原始点，避免复用上一帧姿态。只有有效 ID、有效裁剪框且进入姿态分析的人员才追加序列元素；姿态分析无结果时追加全零 float32 51 维向量。无效框直接跳过。

裁剪与恢复坐标使用相同的实际边界。保持部署模型要求的索引顺序：

```python
[27, 7, 13, 2, 23, 25, 11, 15, 0, 28, 8, 14, 5, 24, 26, 12, 16]
```

逐点排列 `x,y,visibility`。骨架归一化保持上游 Raspberry Pi 代码的公式和参考点边界行为；按本次要求，最终使用 `np.nan_to_num(..., nan=0, posinf=0, neginf=0)` 清理异常数值。正常有限值与上游函数逐值一致；NaN/Inf 清理是本次明确要求的补充。

## 推理时机及报警语义

`TRANSFORMER_INFERENCE_STRIDE = 3`，计数按每个 ID 新增的序列元素独立维护：

| 该人员序列更新次数 | 队列操作 | 推理 |
| --- | --- | --- |
| 1–29 | 每次追加 | 否 |
| 30 | 追加，首次满 30 | 是；重置计数 |
| 31、32 | 每次追加并淘汰最旧元素 | 否；保留旧概率 |
| 33、36、39… | 每次均更新队列 | 每新增 3 元素执行一次 |

零向量也参与计数。已判断为摔倒的人员继续更新和推理。模型输入固定 `(1,30,51)` / float32，输出校验为 `(1,1)` / float32；不调用 `resize_tensor_input()`。

`fall_data` 中按 ID 提供：

- `transformer_probabilities`：最近一次 Fall 概率；首次推理前为 None。
- `transformer_sequence_lengths`：当前缓存长度。
- `transformer_has_inferred`：是否完成首次推理。
- `inferred_this_frame`：本帧是否产生新推理结果。

`fallen_ids` 和 `fall_detected` 表示当前最近一次概率是否达到 0.9，可在未推理帧保持阳性显示。

`fall_event_ids` 只包含本帧新推理达到 0.9 的人员。它是报警事件候选，Dashboard 再按原有每人冷却控制声音、记录和快照；CLI 保留新摔倒状态的去重。沿用旧概率不会产生新事件，冷却不阻止推理。二分类报警类型统一为 `transformer_fall`。

人员超过 5 秒未出现时，清除序列、计数、首次推理标记、概率、原始点、Pose、One Euro、历史和报警状态；通过 `expired_ids` 同步清理 Dashboard 的对应冷却记录。检测器关闭时清空全部状态并置空解释器引用。处理与关闭仍由同一生命周期锁保护。

## 验证结果及边界

在 `D:\anaconda\envs\human-fall\python.exe` 下执行：

```powershell
& D:\anaconda\envs\human-fall\python.exe -m unittest discover -s tests -v
```

34 项测试全部通过，包括原有 Pose 隔离、复用、过期和并发释放测试。

覆盖两条支路的数据来源（Visibility Filter 开/关）、逐点排列、坐标恢复、上游归一化边界、异常值清理、零向量、独立 ID 的 30/33/36 推理时机、逐元素滑窗、缓存概率、阴性结果清除报警、Dashboard 冷却和 CLI 事件去重、解释器加载失败后尝试下一种接口、张量 shape/dtype 校验和资源清理。6 个修改的 Python 文件语法检查全部通过。

这些属于模拟验证：外部推理接口使用假对象，报警概率为模拟值。未把模拟概率作为模型分类结果。原始/平滑支路测试使用生产 One Euro 和 Visibility Filter 代码，MediaPipe 关键点由测试构造。GUI 报警逻辑测试提取实际方法运行，没有打开可见窗口。

已再次核对工程模型与本地上游模型，SHA-256 均为：

```text
C66659C3AC6D37C60469BEE4D6F88CB2AC4976928BDE6D46B19E7F851E8980B0
```

## 当前环境及手动安装建议

- Python：3.10.21（Anaconda CPython，MSC v.1942，64 bit AMD64）。
- Python 路径：`D:\anaconda\envs\human-fall\python.exe`。
- Windows 架构：x86-64 / AMD64；系统版本号 10.0.26200；Python 指针宽度 64 bit。
- 未新增或调用 CUDA 检查；检测器明确使用 CPU。

实际导入结果：

| 导入方式 | 当前错误 |
| --- | --- |
| `from ai_edge_litert.interpreter import Interpreter` | `ModuleNotFoundError: No module named 'ai_edge_litert'` |
| `from tflite_runtime.interpreter import Interpreter` | `ModuleNotFoundError: No module named 'tflite_runtime'` |
| `import tensorflow as tf; tf.lite.Interpreter` | `ModuleNotFoundError: No module named 'tensorflow'` |

当前初始化会明确抛出 `TFLiteUnavailableError`，不会静默回退到规则报警。因此本次未完成真实模型 invoke，也未完成包含真实 Transformer 的视频推理。

建议用户手动安装 LiteRT CPU 推理接口（本次未执行）：

```powershell
& "D:\anaconda\envs\human-fall\python.exe" -m pip install "ai-edge-litert==2.1.4"
```

已核对 [PyPI 的 2.1.4 发布文件](https://pypi.org/project/ai-edge-litert/2.1.4/)：提供 `ai_edge_litert-2.1.4-cp310-cp310-win_amd64.whl`，对应当前 CPython 3.10、Windows x64。命令不选择 GPU 或 NPU 扩展，也不需要安装 TensorFlow 训练组件。轮子的平台和 Python 标签兼容已确认，但具体模型加载及真实输出仍须安装后实测。
