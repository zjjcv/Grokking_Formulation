#!/usr/bin/env python3
"""
统一执行脚本 - 为所有四种运算执行所有分析和绘图

执行顺序：
1. 数据收集分析（Gini/IPR, 对齐, 相位, 投影, 归因, 电路）
2. 绘图生成（训练曲线, Gini/IPR, 对齐, 投影, 电路, 归因）

使用方法:
    python run_all_unified.py --operation x+y     # 单个运算
    python run_all_unified.py --all               # 所有运算
    python run_all_unified.py --all --skip-analysis  # 跳过分析，只绘图
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# 添加 src 目录到路径
# 使用 resolve() 获取绝对路径
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent  # 项目根目录
SRC_DIR = SCRIPT_PATH.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(SRC_DIR))

from lib.config import OPERATIONS, get_all_operations


# ==================== 分析脚本 ====================
ANALYSIS_SCRIPTS = {
    'gini_ipr': SRC_DIR / '2D_DFT_unified/gini_ipr.py',
    'f_alignment': SRC_DIR / '2D_DFT_unified/f_alginment.py',
    'phase_coherence': SRC_DIR / '2D_DFT_unified/phase_coherence.py',
    'fourier_projection': SRC_DIR / '2D_DFT_unified/fourier_projection.py',
    'spectral_attribution': SRC_DIR / '2D_DFT_unified/spectral_attribution.py',
    'circuit_competition': SRC_DIR / '2D_DFT_unified/circuit_competition.py',
}

# ==================== 绘图脚本 ====================
PLOT_SCRIPTS = {
    'training_curves': SRC_DIR / 'plot_unified/training_curves.py',
    'gini_ipr': SRC_DIR / 'plot_unified/gini_ipr.py',
    'f_alignment': SRC_DIR / 'plot_unified/f_alignment.py',
    'fourier_projection': SRC_DIR / 'plot_unified/fourier_projection.py',
    'circuit_competition': SRC_DIR / 'plot_unified/circuit_competition.py',
    'spectral_attribution': SRC_DIR / 'plot_unified/spectral_attribution.py',
}


def run_script(script_path, operation, script_type='analysis'):
    """运行单个脚本"""
    if not script_path.exists():
        print(f"    警告: 脚本不存在 - {script_path}")
        return False

    # 计算相对于项目根目录的脚本路径
    rel_path = script_path.relative_to(PROJECT_ROOT)

    cmd = [sys.executable, str(rel_path), '--operation', operation]

    # 设置环境变量
    env = {**os.environ, 'PYTHONPATH': str(PROJECT_ROOT / 'src')}

    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=3600,
            env=env
        )
        if result.returncode == 0:
            return True
        else:
            if result.stderr:
                print(f"    错误: {result.stderr[-300:] if len(result.stderr) > 300 else result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"    超时")
        return False
    except Exception as e:
        print(f"    异常: {e}")
        return False


def run_analyses(operation):
    """运行所有分析"""
    print(f"\n  {'='*50}")
    print(f"  运行数据收集分析")
    print(f"  {'='*50}")

    results = {}
    for name, script_path in ANALYSIS_SCRIPTS.items():
        print(f"\n  [{name}]")
        success = run_script(script_path, operation, 'analysis')
        results[name] = success

    success_count = sum(1 for s in results.values() if s)
    print(f"\n  分析完成: {success_count}/{len(results)} 成功")
    return results


def run_plots(operation):
    """运行所有绘图"""
    print(f"\n  {'='*50}")
    print(f"  生成图表")
    print(f"  {'='*50}")

    results = {}
    for name, script_path in PLOT_SCRIPTS.items():
        print(f"\n  [{name}]")
        success = run_script(script_path, operation, 'plot')
        results[name] = success

    success_count = sum(1 for s in results.values() if s)
    print(f"\n  绘图完成: {success_count}/{len(results)} 成功")
    return results


def process_operation(operation, skip_analysis=False):
    """处理单个运算"""
    op_name = OPERATIONS[operation]['name']
    print(f"\n\n{'#'*70}")
    print(f"# 处理: {op_name} ({operation})")
    print(f"# 数据目录: {OPERATIONS[operation]['data_dir']}")
    print(f"# 图表目录: {OPERATIONS[operation]['figures_dir']}")
    print(f"{'#'*70}")

    all_results = {}

    if not skip_analysis:
        analysis_results = run_analyses(operation)
        all_results['analysis'] = analysis_results
    else:
        print("\n  跳过数据收集分析")

    plot_results = run_plots(operation)
    all_results['plots'] = plot_results

    return all_results


def main():
    parser = argparse.ArgumentParser(description='统一执行所有分析和绘图')
    parser.add_argument('--operation', type=str, default='all',
                        help='模运算类型 (x+y, x-y, x*y, x_div_y, all)')
    parser.add_argument('--skip-analysis', action='store_true',
                        help='跳过数据收集分析，仅生成图表')

    args = parser.parse_args()

    op = args.operation
    if op == 'all':
        operations = get_all_operations()
    elif op in get_all_operations():
        operations = [op]
    else:
        print(f"错误: 未知的运算类型 '{op}'")
        print(f"可选: {get_all_operations()}")
        return

    print("=" * 70)
    print("Grokking 多运算统一分析系统")
    print("=" * 70)
    print(f"运算数量: {len(operations)}")
    print(f"分析类型: {len(ANALYSIS_SCRIPTS)}")
    print(f"绘图类型: {len(PLOT_SCRIPTS)}")
    if args.skip_analysis:
        print("模式: 仅生成图表")
    else:
        print("模式: 完整分析 + 绘图")
    print("=" * 70)

    all_results = {}

    for operation in operations:
        results = process_operation(operation, args.skip_analysis)
        all_results[operation] = results

    # 打印最终汇总
    print("\n\n" + "=" * 70)
    print("全部完成")
    print("=" * 70)

    for op, results in all_results.items():
        print(f"\n{op} ({OPERATIONS[op]['name']}):")
        if 'analysis' in results:
            a_success = sum(1 for s in results['analysis'].values() if s)
            a_total = len(results['analysis'])
            print(f"  分析: {a_success}/{a_total}")
        if 'plots' in results:
            p_success = sum(1 for s in results['plots'].values() if s)
            p_total = len(results['plots'])
            print(f"  绘图: {p_success}/{p_total}")

    print("\n" + "=" * 70)
    print("图表保存位置:")
    for op in operations:
        print(f"  {op}: {OPERATIONS[op]['figures_dir']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
