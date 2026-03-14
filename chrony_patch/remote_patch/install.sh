#!/bin/bash

# Chrony 安装与配置自动化脚本
# 功能：
# 1. 安装指定的 chrony deb 包
# 2. 备份并替换 chrony 配置文件
# 3. 部署 chrony 预同步脚本

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PATCH_DIR="$SCRIPT_DIR"

# 检查是否以root权限运行
if [ "$(id -u)" -ne 0 ]; then
    echo "错误：此脚本必须以root权限运行！"
    echo "请使用 'sudo $0' 或切换到root用户后运行"
    exit 1
fi

# 检查patch目录是否存在
if [ ! -d "$PATCH_DIR" ]; then
    echo "错误：patch目录不存在！"
    echo "请确保脚本与patch目录在同一位置"
    exit 1
fi

# 定义文件路径
CHRONY_DEB="$PATCH_DIR/chrony_4.2-2ubuntu2_arm64.deb"
NTPDATE_DEB="$PATCH_DIR/ntpdate_1_3a4.2.8p15+dfsg-1ubuntu2_arm64.deb"
CHRONY_CONF="$PATCH_DIR/chrony.conf"
PRESYNC_SCRIPT="$PATCH_DIR/chrony-presync.sh"
BACKUP_DIR="/etc/chrony/backups"

CHRONY_SERVICE="$PATCH_DIR/chrony.service"
SERVICE_BACKUP_DIR="/lib/systemd/system/backups"

# 函数：显示步骤标题
print_step() {
    echo ""
    echo "========================================"
    echo " $1"
    echo "========================================"
    echo ""
}

# 函数：检查文件是否存在
check_file_exists() {
    if [ ! -f "$1" ]; then
        echo "错误：文件 $1 不存在！"
        exit 1
    fi
}

# 步骤1: 安装chrony deb包
print_step "步骤1: 安装chrony软件包"
check_file_exists "$CHRONY_DEB"

echo "正在安装 $CHRONY_DEB ..."
if dpkg -i "$CHRONY_DEB"; then
    echo "安装成功！"
else
    echo "安装过程中出现错误，尝试修复依赖关系..."
    apt-get install -f -y
    echo "修复完成，重新安装..."
    dpkg -i "$CHRONY_DEB"
    echo "安装完成！"
fi

print_step "安装ntpdate软件包"
check_file_exists "$NTPDATE_DEB"

echo "正在安装 $NTPDATE_DEB ..."
if dpkg -i "$NTPDATE_DEB"; then
    echo "安装成功！"
else
    echo "安装过程中出现错误，尝试修复依赖关系..."
    apt-get install -f -y
    echo "修复完成，重新安装..."
    dpkg -i "$NTPDATE_DEB"
    echo "安装完成！"
fi

# 步骤2: 备份并替换chrony配置文件
print_step "步骤2: 配置chrony服务"
check_file_exists "$CHRONY_CONF"

# 创建备份目录
mkdir -p "$BACKUP_DIR"

# 生成带时间戳的备份文件名
BACKUP_FILE="$BACKUP_DIR/chrony.conf.backup.$(date +%Y%m%d_%H%M%S)"

# 备份现有配置文件
if [ -f "/etc/chrony/chrony.conf" ]; then
    echo "备份现有配置文件到 $BACKUP_FILE"
    cp "/etc/chrony/chrony.conf" "$BACKUP_FILE"
else
    echo "警告：未找到原始配置文件，将直接创建新配置"
fi

# 替换配置文件
echo "应用新的chrony配置..."
cp "$CHRONY_CONF" "/etc/chrony/chrony.conf"
chown root:root "/etc/chrony/chrony.conf"
chmod 644 "/etc/chrony/chrony.conf"

# 步骤: 备份并替换chrony service文件
check_file_exists "$CHRONY_SERVICE"

# 创建备份目录
mkdir -p "$SERVICE_BACKUP_DIR"

# 生成带时间戳的备份文件名
BACKUP_FILE="$SERVICE_BACKUP_DIR/chrony.service.backup.$(date +%Y%m%d_%H%M%S)"

# 备份现有配置文件
if [ -f "/lib/systemd/system/chrony.service" ]; then
    echo "备份现有配置文件到 $BACKUP_FILE"
    cp "/lib/systemd/system/chrony.service" "$BACKUP_FILE"
else
    echo "警告：未找到原始文件，将直接创建新配置"
fi

# 替换配置文件
echo "应用新的chrony service..."
cp "$CHRONY_SERVICE" "/lib/systemd/system/chrony.service"
chown root:root "/lib/systemd/system/chrony.service"
ln -s "/lib/systemd/system/chrony.service" "/etc/systemd/system/chronyd.service"
#chmod 777 "/etc/systemd/system/chronyd.service"
systemctl daemon-reload
# 步骤3: 部署预同步脚本
print_step "步骤3: 部署预同步脚本"
check_file_exists "$PRESYNC_SCRIPT"

echo "复制预同步脚本到 /usr/local/bin..."
cp "$PRESYNC_SCRIPT" "/usr/local/bin/chrony-presync.sh"
chown root:root "/usr/local/bin/chrony-presync.sh"
chmod 755 "/usr/local/bin/chrony-presync.sh"

mkdir -p /var/lib/chrony
chown _chrony:_chrony /var/lib/chrony
chmod 755 /var/lib/chrony
/usr/local/bin/chrony-presync.sh

# 创建运行时目录
sudo mkdir -p /run/chrony
sudo chown _chrony:_chrony /run/chrony


# 步骤4: 重启服务并验证
print_step "步骤4: 重启服务并验证"

echo "重启chrony服务..."
systemctl restart chrony

echo "检查服务状态..."
systemctl status chrony --no-pager

echo ""
echo "验证时间同步状态："
chronyc tracking

echo ""
echo "验证时间源："
chronyc sources

# 完成信息
print_step "安装与配置完成！"
echo "chrony 已成功安装并配置"
echo "配置文件已更新并备份在 $BACKUP_DIR"
echo "预同步脚本已安装到 /usr/local/bin/chrony-presync.sh"
echo ""
echo "您可以通过以下命令手动测试预同步脚本："
echo "sudo /usr/local/bin/chrony-presync.sh"
