import re
import matplotlib.pyplot as plt

# ====================== 配置区域 ======================
LOG_FILE_PATH = "loginfo.txt"  # 替换为你的日志路径
# =======================================================

# 【修改点1】更宽松的正则表达式：允许任意数量的空白字符
LOSS_PATTERN = re.compile(r"===>\s+Epoch\s+(\d+)\s+Complete:\s+Avg\.\s+Loss:\s+(\d+\.\d+)")
PSNR_PATTERN = re.compile(r"===>\s+Epoch\s+(\d+)\s+Complete:\s+Avg\.\s+PSNR:\s+(\d+\.\d+)")


def main():
    # 1. 读取日志
    try:
        with open(LOG_FILE_PATH, 'r', encoding='utf-8') as f:
            log_lines = f.readlines()
    except UnicodeDecodeError:
        with open(LOG_FILE_PATH, 'r', encoding='gbk') as f:
            log_lines = f.readlines()

    epoch_data = {}

    # 2. 解析日志（添加调试打印）
    print("-" * 50)
    print("开始解析日志，匹配到的有效行如下：")
    print("-" * 50)

    for line_idx, line in enumerate(log_lines):
        line = line.strip()

        # 匹配 Loss
        # 【修改点2】使用 search 代替 match，允许行首有其他字符
        loss_match = LOSS_PATTERN.search(line)
        if loss_match:
            epoch = int(loss_match.group(1))
            avg_loss = float(loss_match.group(2))
            if epoch not in epoch_data:
                epoch_data[epoch] = {}
            epoch_data[epoch]["loss"] = avg_loss
            print(f"[行 {line_idx + 1}] 匹配到 Loss -> Epoch {epoch}, Loss: {avg_loss}")

        # 匹配 PSNR
        psnr_match = PSNR_PATTERN.search(line)
        if psnr_match:
            epoch = int(psnr_match.group(1))
            avg_psnr = float(psnr_match.group(2))
            if epoch not in epoch_data:
                epoch_data[epoch] = {}
            epoch_data[epoch]["psnr"] = avg_psnr
            print(f"[行 {line_idx + 1}] 匹配到 PSNR -> Epoch {epoch}, PSNR: {avg_psnr}")

    print("-" * 50)

    # 3. 数据整理
    sorted_epochs = sorted(epoch_data.keys())
    valid_epochs = []
    valid_loss = []
    valid_psnr = []

    for epoch in sorted_epochs:
        data = epoch_data[epoch]
        if "loss" in data and "psnr" in data:
            valid_epochs.append(epoch)
            valid_loss.append(data["loss"])
            valid_psnr.append(data["psnr"])

    # 4. 结果检查
    if not valid_epochs:
        print("\n❌ 错误：未检测到任何完整的 Epoch 数据！")
        print("请检查上方的调试打印，确认是否有匹配到数据。")
        print("如果没有匹配到，请复制日志中的几行 'Complete' 内容发给我，我来帮你调整正则。")
        return

    print(f"\n✅ 成功提取 {len(valid_epochs)} 个 Epoch 数据：")
    print(f"   Epoch 范围: {valid_epochs[0]} ~ {valid_epochs[-1]}")
    print(f"   初始 Loss: {valid_loss[0]:.4f}, 最终 Loss: {valid_loss[-1]:.4f}")
    print(f"   初始 PSNR: {valid_psnr[0]:.4f}, 最终 PSNR: {valid_psnr[-1]:.4f}")

    # 5. 绘图（和之前一样）
    plt.rcParams["font.size"] = 14
    plt.rcParams["figure.dpi"] = 120
    line_color = "#1f77b4"
    grid_alpha = 0.7

    # Loss 图
    plt.figure(figsize=(12, 8))
    plt.plot(valid_epochs, valid_loss, color=line_color, linewidth=1)
    plt.title("Training Loss", fontsize=16)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.grid(True, linestyle="-", alpha=grid_alpha)
    plt.tight_layout()
    plt.savefig("training_loss.png")
    plt.show()

    # PSNR 图
    plt.figure(figsize=(12, 8))
    plt.plot(valid_epochs, valid_psnr, color=line_color, linewidth=1)
    plt.title("Eval PSNR", fontsize=16)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("PSNR", fontsize=14)
    plt.grid(True, linestyle="-", alpha=grid_alpha)
    plt.tight_layout()
    plt.savefig("eval_psnr.png")
    plt.show()


if __name__ == "__main__":
    main()