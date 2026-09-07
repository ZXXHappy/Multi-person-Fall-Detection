# Pose 实例隔离：运行验证记录

## 结论

已使用现有 `D:\anaconda\envs\human-fall\python.exe` 完成真实 YOLO、ByteTrack、MediaPipe 推理，以及 CLI / 无窗口 Qt Dashboard 启停验证。未安装依赖，未修改业务代码或旧版 `fall_detection.py`。

Pose 实例隔离、超过 5 秒后的释放、重新入场创建实例、停止后重启均按预期运行。**动态视频的“两人每帧持续输出姿态”未完全通过**：第二人的姿态间歇缺失；共享基线与独立实例模式表现相同。

## 环境与输入

- Python：`D:\anaconda\envs\human-fall\python.exe`
- PyTorch：`2.14.0+cpu`；CUDA 不可用；PyTorch CPU 线程数 8。
- MediaPipe：`0.10.9`；OpenCV：`5.0.0`。
- 本地权重：`yolov12n1.pt`；分析置信度 0.5；检测置信度 0.3。
- 原有 ByteTrack、One Euro、报警判定保留；Visibility Filter 为工程默认关闭状态。
- 视频：`D:\study\大创\测试数据\HumanPose\wave\wave\Bubble_Gum_Club_2_wave_h_nm_np2_fr_med_0.avi`
- 分辨率 320×240，标称 30 FPS；OpenCV 本次实际解码 78 帧。
- 启动存在原有 `attempt_download` 导入失败提示，但异常被工程原有代码捕获，随后成功加载本地权重；未发生推理或资源释放异常。

## 场景结果

| 场景 | 实测结果 | 判定及边界 |
| --- | --- | --- |
| 两人同时出现 | 未修改动态视频中，ID 1、2 各对应一个独立 Pose，四轮测试每轮计时区间均有 68 个双人帧。独立模式 ID 1 输出 68/68，ID 2 输出 46/68，最长连续 13 帧无姿态。共享基线计数相同。 | 实例隔离通过；动态视频双人连续输出未完全通过，不能将“有独立实例”视为“始终有姿态”。 |
| 一人离开超过 5 秒 | 真实视频帧的左侧人物区域被遮住 6.547 秒，期间处理 46 帧。ID 2 的实例关闭并移除；ID 1 原实例保持不变，输出 46/46。 | 受控输入、真实推理验证通过；没有伪造 ID、推理结果或时间。并非自然拍摄的离场视频。 |
| 人员重新进入 | 恢复原画面后，ByteTrack 为返回者分配 ID 3，新建 Pose；旧 ID 2 的实例不再使用。ID 1 输出 25/25；ID 3 在重新确认跟踪后处理 24 帧、输出 24/24。 | 受控输入、真实推理验证通过。重新确认跟踪的第一帧未进行返回者的姿态处理，未使用共享实例兜底。 |
| 停止后重新启动 | CLI 连续创建、处理视频、退出两次；Dashboard 真实 QTimer 驱动两轮启停，另验证窗口关闭及重复应用退出回调。5 个检测器的所有 Pose 均恰好关闭一次。 | 无资源释放报错。Dashboard 使用 Qt offscreen，无可见窗口；CLI 仅替换显示/按键接口，模拟按 q 退出，检测与释放路径真实运行。 |

离场测试前，对选定真实视频帧重复处理 20 次，两人均输出 20/20。该结果用于验证固定输入下的复用，不替代动态视频的连续输出结果。

所有运行中的 Pose 均通过轻量记录包装器转发到真实 MediaPipe 实例。包装器会在关闭后处理或重复关闭时抛出异常；实际验证未触发这些错误。生产代码未替换为模拟推理。

## 多人处理帧率

使用相同的 78 帧原始视频，以“共享、独立、独立、共享”的 ABBA 顺序运行。共享模式仅在测试子类中恢复单个共享 MediaPipe 时序实例，其余检测、跟踪、滤波及报警计算保持一致。

每轮排除前 10 帧，计时剩余 68 帧的 `process_frame()`。计时包含 YOLO、ByteTrack、姿态推理和后续计算，不含视频解码、显示、模型加载及预热。

| 轮次 | Pose 模式 | FPS | 平均帧耗时 | P95 帧耗时 | 双人同时输出姿态 |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | 共享 | 7.36 | 135.94 ms | 178.61 ms | 46/68 |
| 2 | 独立 | 7.90 | 126.65 ms | 144.19 ms | 46/68 |
| 3 | 独立 | 7.32 | 136.68 ms | 163.26 ms | 46/68 |
| 4 | 共享 | 8.04 | 124.42 ms | 150.73 ms | 46/68 |

按总帧数 / 总处理时间合并两轮：

| 模式 | 合并 FPS | 平均帧耗时 |
| --- | ---: | ---: |
| 共享 Pose | 7.68 | 130.18 ms |
| 每 ID 独立 Pose | 7.60 | 131.66 ms |

独立模式本次合并吞吐量低约 **1.13%**，每帧平均增加约 **1.49 ms**，小于轮次间波动；这组短片测试没有显示明确的稳态吞吐量下降。仅验证 CPU、低分辨率、两人场景，不能外推至更多人、更高分辨率或 GPU，也不是统计显著性结论。

当前约 7.6 FPS 低于输入视频的 30 FPS。首帧初始化耗时已单独记录在 JSON / CSV，但受进程首次预热、权重和缓存顺序影响，不用于比较实例创建成本。没有测量内存占用。

## 文件与复现

- `tests/validate_pose_runtime.py`：真实推理验证脚本，入口分别为 `lifecycle`、`benchmark`、`applications`。
- `validation_outputs/source.txt`：原始视频完整路径。
- `validation_outputs/candidate.jpg`、`controlled_absence.jpg`：受控离场测试的原始参考帧和遮挡输入。
- `validation_outputs/lifecycle.json`：逐帧 ID、Pose 实例编号、姿态输出和耗时，以及离场/重入断言结果。
- `validation_outputs/benchmark.json`：四轮逐帧数据、环境及计时口径。
- `validation_outputs/benchmark.csv`：四轮统计表。
- `validation_outputs/application_restart.json`：CLI / Qt 启停结果。

在工程根目录运行：

```powershell
& D:\anaconda\envs\human-fall\python.exe tests\validate_pose_runtime.py lifecycle
& D:\anaconda\envs\human-fall\python.exe tests\validate_pose_runtime.py benchmark
& D:\anaconda\envs\human-fall\python.exe tests\validate_pose_runtime.py applications
```

脚本使用已保存的 source.txt 和参考帧，不会安装依赖或启用摄像头。复跑会覆盖相应验证输出。此前 13 项模拟测试也在本次开始时重新运行通过，与上述真实推理验证分开记录。
