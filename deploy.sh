#!/bin/bash

# Grokking_Formulation 自动部署脚本
# 用法: ./deploy.sh "commit message"

set -e

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 配置 - 请修改为您的实际仓库地址
REPO_URL="YOUR_GITHUB_REPO_URL_HERE"  # 例如: https://github.com/username/Grokking_Formulation.git
BRANCH="main"

# 检查参数
if [ -z "$1" ]; then
    COMMIT_MSG="Update $(date '+%Y-%m-%d %H:%M:%S')"
else
    COMMIT_MSG="$1"
fi

echo -e "${YELLOW}=== Grokking_Formulation 部署脚本 ===${NC}"
echo ""

# 检查是否是 git 仓库
if [ ! -d .git ]; then
    echo -e "${YELLOW}初始化 Git 仓库...${NC}"
    git init
    git branch -M $BRANCH
fi

# 检查 remote 是否已配置
if ! git remote get-url origin &>/dev/null; then
    echo -e "${YELLOW}配置远程仓库...${NC}"
    if [ "$REPO_URL" = "YOUR_GITHUB_REPO_URL_HERE" ]; then
        echo -e "${RED}错误: 请先在脚本中设置 REPO_URL${NC}"
        echo -e "${YELLOW}编辑 $0，将 YOUR_GITHUB_REPO_URL_HERE 替换为您的仓库地址${NC}"
        exit 1
    fi
    git remote add origin $REPO_URL
    echo -e "${GREEN}远程仓库已配置: $REPO_URL${NC}"
fi

# 显示当前状态
echo -e "${YELLOW}当前状态:${NC}"
git status --short
echo ""

# 添加所有更改
echo -e "${YELLOW}添加文件...${NC}"
git add .

# 提交
echo -e "${YELLOW}提交更改...${NC}"
git commit -m "$COMMIT_MSG" || echo -e "${YELLOW}没有新的更改需要提交${NC}"

# 推送
echo -e "${YELLOW}推送到 GitHub ($BRANCH 分支)...${NC}"
git push -u origin $BRANCH

echo ""
echo -e "${GREEN}=== 部署完成! ===${NC}"
