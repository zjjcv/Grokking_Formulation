# Grokking Formulation - Scripts

重组后的脚本目录结构。

## 目录结构

```
scripts/
├── training/              # 训练脚本
│   ├── train.py                # 统一训练脚本（支持四种运算）
│   ├── train_addition.py       # x + y mod 97（独立脚本）
│   ├── train_subtraction.py    # x - y mod 97（独立脚本）
│   ├── train_multiplication.py # x * y mod 97（独立脚本）
│   └── train_division.py       # x ÷ y mod 97（独立脚本）
│
├── analysis/              # 数据收集分析脚本
│   ├── 2d_dft/            # 2D DFT 分析
│   │   ├── gini_ipr.py
│   │   ├── f_alignment.py
│   │   ├── phase_coherence.py
│   │   ├── fourier_projection.py
│   │   ├── spectral_attribution.py
│   │   ├── circuit_competition.py
│   │   ├── qk_circuit.py
│   │   └── qk_circuit_full.py
│   ├── effective_rank.py
│   ├── flatness_slt.py
│   ├── group_representation.py
│   └── circuit_competition_full.py
│
├── plotting/              # 绘图脚本
│   ├── 2d_dft/            # 2D DFT 绘图
│   ├── training_curves.py
│   ├── effective_rank.py
│   ├── flatness_slt.py
│   ├── group_representation.py
│   └── circuit_competition_full.py
│
├── lib/                   # 共享库
│   ├── __init__.py
│   ├── config.py          # 配置和样式
│   └── utils.py           # 工具函数
│
└── run_pipeline.py        # 主入口（运行所有分析）
```

## 使用方法

### 训练模型

**推荐：使用统一训练脚本**

```bash
# 从项目根目录运行
cd /root/data1/zjj/Grokking_Formulation

# 训练加法
python scripts/training/train.py --operation add

# 训练减法
python scripts/training/train.py --operation sub

# 训练乘法
python scripts/training/train.py --operation mul

# 训练除法
python scripts/training/train.py --operation div
```

**或者使用独立脚本**（已保留）

```bash
python scripts/training/train_addition.py
python scripts/training/train_subtraction.py
python scripts/training/train_multiplication.py
python scripts/training/train_division.py
```

### 运行分析

```bash
# 运行所有分析
python scripts/run_pipeline.py --all

# 运行单个运算的分析
python scripts/run_pipeline.py --operation x+y

# 单独运行某个分析
python scripts/analysis/2d_dft/gini_ipr.py --operation x+y
```

### 生成图表

```bash
# 单独运行绘图脚本
python scripts/plotting/2d_dft/gini_ipr.py --operation x+y
python scripts/plotting/training_curves.py --operation x+y
```

## 导入模块

```python
# 在脚本中导入配置
import sys
sys.path.insert(0, 'scripts')
from lib.config import OPERATIONS, COLORS, setup_style
from lib.utils import get_checkpoint_dir, get_figures_dir
```

## 注意事项

1. 所有路径在 `lib/config.py` 中配置
2. 使用 `--operation` 参数选择运算类型
3. 原始脚本已归档到 `archive/legacy/` 目录
