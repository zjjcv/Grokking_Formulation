#!/bin/bash

# Git 仓库初始化脚本
# 首次使用时运行此脚本

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}=== 初始化 Git 仓库 ===${NC}"
echo ""

# 输入您的 GitHub 仓库 URL
read -p "请输入您的 GitHub 仓库 URL (例如: https://github.com/username/repo.git): " REPO_URL

if [ -z "$REPO_URL" ]; then
    echo -e "${RED}错误: 仓库 URL 不能为空${NC}"
    exit 1
fi

# 初始化 git
echo -e "${YELLOW}初始化 Git...${NC}"
git init
git branch -M main

# 添加 remote
echo -e "${YELLOW}添加远程仓库...${NC}"
git remote add origin $REPO_URL

# 更新 deploy.sh 中的 REPO_URL
echo -e "${YELLOW}更新 deploy.sh 配置...${NC}"
sed -i "s|REPO_URL=\"YOUR_GITHUB_REPO_URL_HERE\"|REPO_URL=\"$REPO_URL\"|" deploy.sh

# 首次提交
echo -e "${YELLOW}首次提交...${NC}"
git add .
git commit -m "Initial commit: Grokking Formulation research project"

# 推送
echo -e "${YELLOW}推送到 GitHub...${NC}"
git push -u origin main

echo ""
echo -e "${GREEN}=== 初始化完成! ===${NC}"
echo -e "${GREEN}以后可以使用 ./deploy.sh 来快速提交更改${NC}"
