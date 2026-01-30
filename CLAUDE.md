# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

研究项目，探索神经网络中的 Grokking 现象及其理论解释。Grokking 是指神经网络在长期训练后突然从记忆模式泛化到理解模式的现象。

## Project Structure

```
├── src/                    # 源代码 (Python)
├── experiments/
│   ├── figures/           # 实验生成的图表
│   └── logs/              # 实验运行日志
├── data/                  # 数据集
├── configs/               # 配置文件
└── writing/
    ├── paper/             # 论文手稿
    └── notes/             # 研究笔记
```

## Development

### 设置环境
```bash
# 创建虚拟环境 (推荐)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 运行实验
```bash
# 运行主实验脚本
python src/main.py --config configs/default.yaml
```

### 权限
以下 bash 命令已预授权:
- `python:*` - Python 执行
- `ls:*` - 目录列表
- `find:*` - 文件搜索
