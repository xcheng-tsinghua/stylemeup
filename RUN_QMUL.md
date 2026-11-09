# 本地 QMUL FG-SBIR 运行说明

用户已确认分别在 shoe / chair 上做训练实例与测试实例不相交的 FG-SBIR，而不是类别级 ZS-SBIR。

## 环境与入口

已验证 `D:\miniconda\envs\dp\python.exe`：PyTorch 2.4.1+cu121、torchvision 0.19.1+cu121、RTX 4060 Ti 16 GB。新增入口使用 PyTorch 原生 functional_call 实现一阶 MAML，不需要 learn2learn，也不需要下载预训练权重。原始代码保留。

在项目根目录 PowerShell 中运行一个完整轮次：

```powershell
& D:\miniconda\envs\dp\python.exe -u Code\run_qmul.py --category chair --output runs\chair --epochs 1
& D:\miniconda\envs\dp\python.exe -u Code\run_qmul.py --category shoe --output runs\shoe --epochs 1
```

默认数据路径为 `D:\document\DataSet\sketch_retrieval\qmul_v2_fit`，可用 `--root` 改写。默认 meta batch 为 4。`--max-steps` 只用于冒烟测试；正式完整轮次不传该参数。

继续训练到累计 200 轮（这是继续训练命令，不表示已执行）：

```powershell
& D:\miniconda\envs\dp\python.exe -u Code\run_qmul.py --category chair --output runs\chair --epochs 200 --resume runs\chair\last.pt
& D:\miniconda\envs\dp\python.exe -u Code\run_qmul.py --category shoe --output runs\shoe --epochs 200 --resume runs\shoe\last.pt
```

独立重载权重评估：

```powershell
& D:\miniconda\envs\dp\python.exe Code\run_qmul.py --category chair --output runs\chair_verify --resume runs\chair\last.pt --evaluate-only
& D:\miniconda\envs\dp\python.exe Code\run_qmul.py --category shoe --output runs\shoe_verify --resume runs\shoe\last.pt --evaluate-only
```

## 数据与评估

| 数据集 | 训练实例 | 训练草图 | 三风格训练 episode | 测试实例/图库 | 测试草图 |
|---|---:|---:|---:|---:|---:|
| shoe | 1800 | 5982 | 1541 | 200 | 666 |
| chair | 300 | 952 | 233 | 100 | 323 |

读取原始 `sketch_s3/{train,test}/class/*.txt` 的绝对坐标与笔划结束标记，不使用已拟合/截断笔划版本。栅格化遵循原实现白线黑底约定，Pillow 画线并以 3×3 最大值滤波加粗；与原 Bresenham + 十字形 scipy 膨胀存在像素级差异。输入统一为 256×256，归一化至 [-1,1]。

照片 train/test 文件夹都包含该类别的所有照片。训练正负样本严格限制在训练草图对应的实例；图库只使用测试草图对应实例。初始化会检查实例划分不重叠。训练遵循仓库的编号 1/2/3 三草图配对；其他编号不用于训练 episode，测试保留所有草图。

使用编码器 128 维归一化 invariant 特征和欧氏距离。Acc@1/5/10 为真值实例排名不超过相应阈值的查询比例；MRR 为倒数排名均值。距离相同时使用保守的 <= 排名。每个查询只有一个真值照片。

输出：`manifest.json` 为环境与数据审计，`history.jsonl` 为各轮指标，`retrieval_epoch_001.json` 为全部查询排名及 top10，`last.pt` 含模型、正则器、优化器与随机状态。默认仅保存最后一轮，不通过测试集指标选模型。设置固定种子但不承诺不同 CUDA 环境逐位一致。

## 实现修复与复现边界

新入口复用 `Networks.py` 的 VAE 与 Regulariser，保持原仓库的四输入重建、跨风格/跨域重建、KL 与双 triplet 目标：

- 每个 meta batch 清零梯度，按实际 episode 数归一化。
- 将原代码已计算但遗漏的 self reconstruction 加入总损失。
- triplet 使用正确的二维 [样本, 特征]，修复原 stack 三维张量导致的距离维度问题。
- 对 VAE 使用一阶 MAML；对 weighted-L1 正则器保留解析超梯度，避免直接 detach 内循环梯度导致正则器无法学习。运行时检查其梯度存在且有限。
- 使用梯度裁剪、有限值检查、完整断点与无梯度批量评估。

这不是完整论文精度复现。原论文使用 InceptionV3、64 维特征、FT 层、20 轮预热和不同损失调度；本仓库发布入口是 U-Net 式 VAE、128 维、无 FT。这里复用发布代码的网络，未擅自换成另一个主干，也未补写论文缺失模块。交叉域重建保留发布代码的写法。正则器参数也保留原实现的无约束形式。

本地 shoe 划分 1800/200 与论文测试图库 275 不同；本地 chair 测试草图数也不等同于论文设置。论文的源实例 meta-train/meta-validation 细分未在本地目录提供，本入口沿用发布代码的实例内风格 episode。一个完整轮次用于端到端验证，不能声称已收敛。

论文来源：https://arxiv.org/html/2103.15706

## 根目录训练入口（SwanLab）

安装依赖后启动：

```powershell
pip install -r requirements-qmul.txt
python train.py --dataset chair --epoch 200 --batch_size 4 --seed 42 --swanlab_api_key YOUR_KEY --swanlab_project StyleMeUp --resume False
```

继续训练至累计 300 轮：

```powershell
python train.py --dataset chair --epoch 300 --batch_size 4 --seed 42 --swanlab_project StyleMeUp --resume True
```

密钥也可通过 `SWANLAB_API_KEY` 环境变量或 SwanLab 已有登录提供，不写入训练配置、检查点或日志。`--dataset` 支持 `shoe` / `chair`；数据路径使用 `--root` 指定，默认值同原入口。`--batch_size` 是 meta batch 的 episode 数量。

`--resume` 默认 False，也支持单独写 `--resume`（等价 True）。False 创建独立运行目录；True 在 `--output`（默认项目的 `runs`）下递归查找 `.pt`，按修改时间优先使用数据路径、类别和 SwanLab 项目兼容且可加载的检查点。恢复模型、正则器、优化器、已完成轮次及随机状态；未找到时打印警告并从头开始。恢复以完整 epoch 为单位，中途打断会重跑未完成的 epoch。恢复优化器后使用检查点中的学习率。

新检查点保存 SwanLab 实验 ID，通过 `resume="must"` 续写同一实验，云端实验不存在或无法认证时会报错。旧入口生成的检查点没有实验 ID，可恢复训练，但会警告并创建新的 SwanLab 实验。每轮上传 `Acc@1`、`Acc@5`（0–100 百分比）、`loss`、`epoch`，step 使用累计 epoch。检查点原子替换，保存后再上传；恢复时重放最后一轮指标以补偿上传中断。

`--max_steps 1 --epoch 1` 可用于冒烟检查，但评估仍遍历全部测试数据。可用 `--eval_batch` 控制评估批大小。
