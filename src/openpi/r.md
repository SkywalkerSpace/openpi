这个包的核心定位很明确：它不是一个通用机器人框架，而是围绕 openpi 的三类策略模型做的完整闭环，包含模型定义、数据变换、训练配置、推理封装和 websocket 部署。总的控制链路是：训练配置选型 -> 载入模型和归一化统计 -> 组装输入/输出 transforms -> 生成 policy -> 通过 websocket server 对外服务。

**结构梳理**
- src/openpi/models/model.py 定义了最底层的抽象：Observation、Actions、BaseModel、BaseModelConfig，以及参数恢复和观测预处理。
- src/openpi/models/pi0_config.py 和 src/openpi/models/pi0.py 是 π0 / π0.5 的实现，负责 flow matching 风格的动作生成。
- src/openpi/models/pi0_fast.py 和 src/openpi/models/tokenizer.py 是 π0-FAST 路线，核心区别是把动作离散化成 token，自回归解码。
- src/openpi/policies/policy.py 是运行时 policy 包装器，统一处理输入 transform、模型推理、输出 transform，还带有录制器。
- src/openpi/policies/policy_config.py 负责把训练配置、checkpoint、norm stats、transform 链拼成可直接推理的 policy。
- src/openpi/policies/aloha_policy.py、src/openpi/policies/droid_policy.py 是平台适配层，把不同机器人/数据集的字段映射到统一格式。
- src/openpi/training/config.py 是训练配置中心，里面注册了所有预置配置，比如 pi0_aloha、pi05_droid、pi05_libero、pi0_fast_droid 等。
- src/openpi/transforms.py 是整个数据管道的“积木库”，负责 repack、归一化、tokenize、delta action、pad state/action 等操作。
- src/openpi/serving/websocket_policy_server.py 提供 websocket 推理服务，支持 health check 和连续动作流。

**功能上怎么理解**
- 模型层做的是“看图、看状态、看语言，输出动作”。
- policies 层做的是“把不同平台的数据格式翻译成模型能吃的格式，再把模型输出翻译回机器人格式”。
- training 层做的是“定义数据集、归一化统计、权重加载、冻结策略、优化器和训练超参”。
- serving 层做的是“把训练好的 policy 变成一个能被机器人 runtime 调用的服务”。

**操作方法**
1. 安装和环境准备以 README 为准，核心是 uv sync 和可编辑安装。
2. 选一个训练配置名，例如 pi05_droid 或 pi05_libero。
3. 训练时先算 norm stats，再跑训练脚本。
4. 推理时用训练配置名和 checkpoint 目录创建 policy，然后可本地调用或起 websocket server。

最关键的使用入口都已经在 README.md 和 scripts/serve_policy.py 里串好了。比如 serve_policy 的默认路径就是读取训练配置、自动选择默认 checkpoint，然后起一个 websocket server。

如果你要，我下一步可以继续做两种更细的整理之一：
1. 按“数据流”把 pi0 / pi0-fast / pi0.5 的输入输出链路逐步画出来。
2. 按“可运行方式”把训练、离线推理、远程推理、服务端接口整理成一份可执行指南。

已创建 3 个待办事项