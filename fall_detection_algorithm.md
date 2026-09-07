# 当前摔倒检测算法

## 人员和关键点

YOLOv12 在 CPU 上检测人物，ByteTrack 分配 person_id。有效 ID 和非空裁剪框才进入姿态分析。人物框先裁到图像范围内；每人首次分析时创建 MediaPipe Pose，之后复用，static_image_mode=False。

MediaPipe 原始 (x,y,z,visibility) 33 点先保存为独立副本，再送入该人的 One Euro。原始支路用于 Transformer；平滑支路用于辅助姿态特征。One Euro 平滑 x/y/z，visibility 原样传递，不按可见度冻结坐标。

保留角度、速度、加速度、jerk、宽高比、髋部和肩部位置。pose_histories[person_id] 保存最近 15 次人工特征，仅供辅助分析，不触发最终报警。

## Transformer 特征

关键点顺序保持：

```python
[27, 7, 13, 2, 23, 25, 11, 15, 0, 28, 8, 14, 5, 24, 26, 12, 16]
```

每点连续保存 x、y、visibility，构成 float32 (51,)。使用与裁剪相同的实际边界恢复完整帧坐标：

```python
x_full = (x1 + x_crop * (x2 - x1)) / frame_width
y_full = (y1 + y_crop * (y2 - y1)) / frame_height
```

骨架归一化保持配套部署公式：髋部中心为原点，肩髋中心纵向距离为尺度；参考点有效条件为 visibility > 0.3。双侧有效取中心，单侧有效使用单侧；髋部参考无效时保留未归一化特征，尺度无效或小于 1e-5 时仅平移。visibility 不改变，最后用 np.nan_to_num 清理异常值。

## 调度与报警

每人独立 deque(maxlen=30)、更新计数和最近概率。每次姿态分析追加一个特征；无姿态时追加 float32 51 维零向量。无效 ID 或无效框直接跳过。

第 30 次更新立即推理，之后每新增 3 个元素推理一次。间隔帧仍更新队列。单解释器串行处理人员，固定输入 (1,30,51) float32，输出 (1,1) float32。

最近概率 >= 0.9 时加入 fallen_ids 并设置 fall_detected。仅本帧新推理达到阈值的 ID 进入 fall_event_ids，类型统一为 transformer_fall。CLI 对连续状态去重，Dashboard 沿用每人冷却；冷却只限制发送，不影响队列和推理。

fall_data 提供每人概率、序列长度、首次推理状态、inferred_this_frame、人员/摔倒/事件 ID 和过期 ID。

## 生命周期与模块

models/fall_detector.py 负责多人检测、Pose、One Euro 和时序调度；models/transformer_fall_classifier.py 负责特征、模型加载及推理；utils/one_euro_filter.py 只负责算法和参数。

过期和关闭共用 _clear_person_state()。超过 5 秒未出现时清理该人状态并关闭 Pose。close() 持有与处理相同的锁，清理剩余人员并置空解释器引用。Pose 关闭失败时保留对象供后续重试，其他人员仍正常释放。
