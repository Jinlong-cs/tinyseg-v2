# TinySeg v2

YOLO26n-sem 二类可行驶区域分割：数据准备、训练、验证、静态 ONNX 导出与 RDK X5 INT8 PTQ。
替代旧 TinySeg 的实例分割训练/导出路径；旧仓库保留作历史参考。

- 训练与导出输入固定 RGB `NCHW [N,3,352,640]`，值域 `0–1`。
- 原图 `640×512` 整帧 LINEAR resize 到 `640×352`，不裁剪、不 letterbox。
- 类别 `0=non_drivable`、`1=drivable`，`255=ignore`。
- 导出只输出主头 logits `[1,2,44,80]`，opset11；不内嵌 NMS、ArgMax 或全分辨率 resize。
- 下游先双线性放大两个 logits，再比较 `fg > bg`，平局为 0；需要时 NEAREST 恢复原图 mask 尺寸。

## 安装与源码

使用 Python 3.11+。训练和导出环境与 Horizon 编译环境分开。
模型结构直接来自 [Ultralytics 官方源码](https://github.com/ultralytics/ultralytics)，
固定提交 `3d75949c41f78a2c6763eb347ac7a94793098e89`（8.4.175）。
这是独立的轻量适配仓库，不是 GitHub fork；不复制或修改官方模型库。

```bash
python3.11 -m venv venv
source venv/bin/activate
export TINY_WORK_DIR="$PWD/work"
export PIP_CACHE_DIR="$PWD/work/cache/pip"
export TMPDIR="$PWD/work/tmp"
mkdir -p "$TMPDIR" work/weights
python -m pip install -r requirements.txt
```

所有生成文件放在 `work/`；数据、权重、ONNX、BIN 不纳入 Git。仅加载可信来源的 `.pt`。
依赖来源与许可见 [UPSTREAM.md](UPSTREAM.md)，协作规范见 [AGENTS.md](AGENTS.md)。

## 数据准备

原始 JSONL 每行一个样本，路径相对清单所在目录。`domain` 格式 `product/view`，
用于产品均衡采样和验证的分域统计；仅一个产品时正常使用。
`group` 是采集序列标识，提供时检查它不能跨 train/val。检测重复像素跨集合时直接报错。

```json
{"id":"front_0001","split":"train","domain":"robot/front","group":"sequence_A","image":"images/0001.png","label":"labels/0001.txt"}
{"id":"rear_0002","split":"val","domain":"robot/rear","group":"sequence_B","image":"images/0002.png","label":"labels/0002.txt"}
```

`label` 是归一化 YOLO 多边形：`0=drivable`、`1=stairs/排除区域`，排除区域覆盖重叠的 drivable；
在目标分辨率栅格化，保持已验证的旧标注转换规则。空文件是合法的全背景样本。
也可把 `label` 换成 `mask`，提供与原图同尺寸的单通道 PNG，像素只能为 0/1/255；此时 NEAREST resize。
不要把 legacy polygon 类别编号与新语义 mask 像素编号混用。

```bash
python prepare_data.py --manifest /path/to/source.jsonl --out work/data
```

生成 PNG、相对路径 `manifest.jsonl`、文件哈希和 split 计数。训练前重新核对准备后文件的哈希。

## 训练与验证

先下载官方完整语义预训练权重，不从随机初始化开始：

```bash
curl -fL https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-sem-ade20k.pt \
  -o work/weights/yolo26n-sem-ade20k.pt
python train.py --manifest work/data/manifest.jsonl \
  --pretrained work/weights/yolo26n-sem-ade20k.pt --out work/train --device cuda:0
```

官方预训练 SHA256：`68ac71bef2868c987fff7cd7c49cb656922f6d8447ad21af3893305577435000`。
下载后核对哈希；实际使用的权重哈希会写入运行配置。

保留已验证配方：AdamW lr=3e-4、weight decay=1e-4、batch=32、200 epochs、5 epoch warmup + cosine；
主头 CE + 0.5 Dice，辅助头 CE 总权重 0.4；AMP、clip norm=10、无 EMA。
保留这段小训练适配器是为了明确固定损失、采样、图像几何和恢复状态；模型结构直接使用官方实现。
以产品内各 view IoU 平均，再对产品平均选模，同分优先较低 false-drivable。
验证统一使用 OpenCV 双线性 logits，与部署解码规则一致；不重新实现官方网络。
早停最早第 100 epoch，patience=40，min_delta=0.0005；每 5 epoch 保存一次完整 checkpoint。

`--config recipe.json` 可覆盖 `train.py` 的 `DEFAULTS`；未知配置键报错。
`last.pt`、`best.pt`、`epochNNN.pt` 均保留 FP32 权重、optimizer、scheduler、AMP scaler、
随机数、epoch/step、采样及早停状态。恢复必须保持数据/配置/预训练文件不变，并使用相同设备环境。

```bash
python train.py --manifest work/data/manifest.jsonl \
  --pretrained work/weights/yolo26n-sem-ade20k.pt --out work/train \
  --resume work/train/last.pt --device cuda:0
python evaluate.py --checkpoint work/train/best.pt --manifest work/data/manifest.jsonl \
  --out work/validation.json --device cuda:0
python export.py --checkpoint work/train/best.pt --check-manifest work/data/manifest.jsonl --out work/export
```

导出默认检查 64 张 val 图 + 一个随机输入，记录 logits 误差和最终 mask 翻转比例；失败即停止。
本次还验证了 CPU 连续训练与 epoch checkpoint 恢复后的参数逐项一致。CUDA reduction 不承诺逐位确定性。

## X5 INT8 PTQ 与编译

校准只读取 manifest 的 `train` 样本，默认按 domain 轮转选择 512 张，不重复样本。
需先按场景、采集序列划分 train/val，再冻结清单；同一序列不要跨集合。
RGB **0–255 float32 CHW** 校准输入不预先除以 255，mapper 统一做一次归一化。

```bash
python x5.py calibrate --manifest work/data/manifest.jsonl --count 512 --out work/calibration
```

在 x86 Linux 的 OpenExplorer **v1.2.8** CPU 容器中执行编译；Mac 的 Python 环境只用于训练代码检查、导出或配置生成。
不要在 SDK 容器安装上述训练依赖，保留镜像自带的 mapper/onnx/numpy。

```bash
IMAGE=registry.d-robotics.cc/deliver/hub.hobot.cc/aitools/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8
docker run --rm --cpus=4 --memory=16g \
  -v "$PWD:/workspace" -w /workspace \
  -e TINY_WORK_DIR=/workspace/work -e PYTHONDONTWRITEBYTECODE=1 \
  -e OMP_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 \
  "$IMAGE" python3 x5.py compile \
  --onnx work/export/model.onnx --calibration work/calibration/rgb \
  --out work/ptq --name MODEL_640x352_int8_nv12 --run
```

将 `MODEL` 换成实际模型名称。省略 `--run` 仅生成配置与输入摘要；后续实际执行应使用新的输出目录。
每次输出目录必须不存在，避免覆盖旧产物。镜像标签可能被更新，发布时还要记录实际 image ID。

固定 `march=bayes-e`、NV12、**`input_space_and_range: bt601_video`**、`scale=1/255`、
`set_all_nodes_int8`、latency/O3。输入 ONNX 必须匹配同目录中通过数值检查的 `export.json`。
`compiled.json` 保存 BIN/ONNX/config SHA256、实际 BPU/CPU 放置、节点精度与回退列表；
公开输出可能为 FLOAT32，不能据此称内部计算回退。
任何命令、节点报告解析或输入色彩范围检查失败都会退出，不生成成功结论。

编译成功后仍需用 `HB_ONNXRuntime` 和真实 X5 回放核对精度、NV12 输入、输出布局和延迟。
C++/ROS 节点留在部署仓库，本仓库不复制节点实现。每个新 BIN 都必须重新验证，历史 BIN 的成功不替代验证。

## 本次迁移的验证范围

已在本地 CPU 检查合成数据准备、短程训练、固定输入尺寸、校准数据和 PTQ 配置生成；
使用实际权重完成 PyTorch→ONNX 数值检查。合成数据只检查代码路径，不代表产品精度。
本次未重新执行 Horizon 编译、GPU 正式训练或 X5 测速；编译命令和节点审计来自此前运行过的链路。
迁移后的新 BIN、GPU AMP 恢复及目标场景精度仍需各自验证。生产发布须保留数据版本、split、配置、
软件版本、checkpoint/ONNX/BIN SHA256，以及 float、quant、board 三层证据。
