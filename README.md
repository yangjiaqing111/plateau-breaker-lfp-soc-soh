# PlateauBreaker: AFFRLS-DAEKF LFP SOC/SOH 联合估计

面向混合动力车辆复杂工况的磷酸铁锂电池 SOC/SOH 双时间尺度自适应联合估计方法，含 Fisher 信息引导的平台区偏差记忆补偿。

## 核心方法

- **建模**：含滞后电压状态的二阶 RC 等效电路模型
- **参数辨识**：AFFRLS（遗忘因子随残差自适应调整 + 激励不足冻结）
- **状态估计**：DAEKF 双时间尺度（1s SOC 快尺度 + 循环级容量慢尺度）
- **偏差补偿**：Fisher 信息引导的分区偏差记忆补偿（四维查找表）

## 文件说明

| 文件 | 说明 |
|------|------|
| `realtime_demo.py` | 车端主循环演示（AFFRLS + DAEKF 完整实现） |
| `realtime_demo_clean.py` | 精简版演示脚本 |
| `gen_paper_figures.py` | 论文图表生成脚本 |
| `figure/` | 论文图表 |
| `Readme.txt` | 补充说明 |

## 数据集说明

本仓库代码使用以下公开数据集，因体积原因未随仓库上传，请从官方地址下载：

- CALCE Battery Research Data: https://calce.umd.edu/batterydata
- Borealis A123 LFP 老化数据集: https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/RPCWBY
- Oxford Battery Degradation Dataset: https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac
- UCCS A123 数据集: https://data.mendeley.com/datasets/p8kf893yv3/1

## 复现环境

- Python ≥ 3.10
- 依赖：`numpy`、`matplotlib`
- 随机种子固定为 42

## 引用

若本项目对你的研究有帮助，请引用相关论文（待发表）。

## 许可

MIT License
