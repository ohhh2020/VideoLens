# 本地模型与 CPU / GPU 切换

项目只部署以下三个本地模型：

| 模型 | 用途 | 设置字段 |
| --- | --- | --- |
| faster-whisper-small | 无平台字幕时执行语音转写 | `asr.device` |
| BGE-M3 | 文本与查询向量，1024 维 | `embedding.text_device` |
| SigLIP2 base | 图像和图文查询向量，768 维 | `embedding.image_device` |

三个模型为必需项，首次执行 `python scripts/download_models.py --verify`。缺少权重、加载失败或推理异常会报错，不使用哈希向量、伪转写或其他本地模型替代。

## 网页热切换

打开“设置 → 本地模型运行设备”，分别选择 CPU 或 GPU 并保存，可以任意组合。先等待正在执行的推理，再加载并验证候选模型，验证及保存全部成功才替换；失败保留原模型与设置。仅重载发生变化的模型，不重新分析已有视频，也无需重启服务。

设备选项表示目标设备，“本次服务已生效”读取实际权重设备；Whisper 显示实际加载设备与精度。GPU 版向量编码使用 FP16，Whisper 的 GPU/CPU 选项分别使用 float16/int8。

## NVIDIA GPU 环境

在已激活 Conda 环境中安装兼容硬件和驱动的 CUDA 版 PyTorch。例如本次 RTX 5060 实测使用：

```bash
conda activate VideoLens
python -m pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

替换库前先停止服务。其他设备按 [PyTorch 官方指引](https://pytorch.org/get-started/locally/)选择版本。能检测显卡不等于模型已可推理，启动预热会进一步确认。

Whisper 的 CUDA DLL 从当前 Conda 环境、NVIDIA Python 包及 PyTorch 库目录定位，无需把 CUDA 库加入系统 PATH。目标 GPU 不可用时明确失败，不悄悄退回 CPU。FFmpeg / FFprobe 仍由系统 PATH 定位。

## Windows CUDA 转写隔离

Windows上选择GPU运行Whisper时，模型由同一Conda环境中的受控子进程加载，网页进程持有代理，不重复加载一份Whisper权重。子进程随服务管理，连续视频复用同一份模型；切换CPU、卸载模型或正常关闭服务时回收。CPU转写保持原来的进程内执行方式。

如果CTranslate2的原生调用长时间没有返回，普通Python线程无法安全终止它。现在可以回收卡住的转写子进程，使任务明确失败并释放本服务的推理锁。用户之后手动重试会创建新的CUDA上下文，不自动重新分析或改用CPU，不生成空转写冒充成功。

`configs/app_config.yaml`的`asr`段提供以下配置，修改后需重新启动VideoLens：

```yaml
isolate_cuda: true
worker_timeout_s: 90        # 等待准备完成或下一个语音片段的最长秒数
worker_start_timeout_s: 60  # 严格加载模型的最长秒数
```

超时按准备状态及实际片段重置，不是整部视频只能分析90秒。模型、精度、beam size、词级时间戳和初始batch均保持原设置；OOM仍缩批，不切换设备。隔离与超时设置不作废有效转写缓存，也不自动重算旧视频。独立进程不报告未经测量的GPU峰值，日志会注明未测量。

此次真实样本及异常恢复测试见[Whisper原生调用卡住修复](../更新日志/2026-11-02Whisper原生卡住修复.md)。

## 报告视图与 Markdown

用户模式只展示内容概览和关键结论，详细模式保留完整报告、统计与转写入口。切换立即预览，保存后同步浏览器其他工作台，不删除完整数据，也不重新分析视频。

报告和聊天使用本地 marked、DOMPurify 渲染 Markdown，并清理危险脚本、事件属性和链接。SigLIP 配置兼容层只修正上游打印配置的方式，不修改权重或实际 token 参数，也不屏蔽真正的错误。
