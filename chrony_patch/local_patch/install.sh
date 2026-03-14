#!/bin/bash

# 设置工作目录
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PATCH_DIR="$SCRIPT_DIR"

# 检查是否以root权限运行
if [ "$(id -u)" -ne 0 ]; then
    echo "错误：此脚本必须以root权限运行！"
    echo "请使用 'sudo $0' 或切换到root用户后运行"
    exit 1
fi

# 日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# 步骤1: 安装必要软件
log "步骤1: 安装sshpass和chrony"

# 更新软件包列表
log "更新软件包列表..."
apt-get update > /dev/null

# 安装sshpass
if ! command -v sshpass &> /dev/null; then
    log "安装sshpass..."
    apt-get install -y sshpass
    if [ $? -eq 0 ]; then
        log "sshpass安装成功"
    else
        log "错误：sshpass安装失败"
        exit 1
    fi
else
    log "sshpass已安装，跳过安装"
fi

# 安装chrony
if ! command -v chronyc &> /dev/null; then
    log "安装chrony..."
    apt-get install -y chrony
    if [ $? -eq 0 ]; then
        log "chrony安装成功"
    else
        log "错误：chrony安装失败"
        exit 1
    fi
else
    log "chrony已安装，跳过安装"
fi

# 步骤2: 备份并替换chrony配置文件
log "步骤2: 配置chrony服务"

# 检查patch目录是否存在
if [ ! -d "$PATCH_DIR" ]; then
    log "错误：patch目录不存在！"
    log "请确保脚本与patch目录在同一位置"
    exit 1
fi

# 检查配置文件是否存在
CHRONY_CONF="$PATCH_DIR/chrony.conf"
if [ ! -f "$CHRONY_CONF" ]; then
    log "错误：配置文件 $CHRONY_CONF 不存在！"
    exit 1
fi

# 创建备份目录
BACKUP_DIR="/etc/chrony/backups"
mkdir -p "$BACKUP_DIR"

# 生成带时间戳的备份文件名
BACKUP_FILE="$BACKUP_DIR/chrony.conf.backup.$(date +%Y%m%d_%H%M%S)"

# 备份现有配置文件
if [ -f "/etc/chrony/chrony.conf" ]; then
    log "备份现有配置文件到 $BACKUP_FILE"
    cp "/etc/chrony/chrony.conf" "$BACKUP_FILE"
    if [ $? -ne 0 ]; then
        log "错误：配置文件备份失败"
        exit 1
    fi
else
    log "警告：未找到原始配置文件，将直接创建新配置"
fi

# 替换配置文件
log "应用新的chrony配置..."
cp "$CHRONY_CONF" "/etc/chrony/chrony.conf"
if [ $? -ne 0 ]; then
    log "错误：配置文件替换失败"
    exit 1
fi

chown root:root "/etc/chrony/chrony.conf"
chmod 644 "/etc/chrony/chrony.conf"

# 步骤3: 重启服务并验证
log "步骤3: 重启服务并验证"

log "重启chrony服务..."
systemctl restart chrony
if [ $? -ne 0 ]; then
    log "错误：chrony服务重启失败"
    exit 1
fi

log "检查服务状态..."
systemctl status chrony --no-pager

log ""
log "验证时间同步状态："
chronyc tracking

log ""
log "验证时间源："
chronyc sources

log ""
log "安装与配置完成！"
log "chrony 已成功安装并配置"
log "配置文件已更新并备份在 $BACKUP_DIR"
