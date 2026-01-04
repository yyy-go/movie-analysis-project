import matplotlib.font_manager
import os

print("检查系统中文字体...")
print("=" * 60)

# 列出所有可用字体
fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist]

# 查找包含中文关键词的字体
chinese_keywords = ['hei', 'song', 'kai', 'yahei', 'pingfang', 'sim', 'msyh', 'microsoft']
chinese_fonts = []

for font in fonts:
    font_lower = font.lower()
    for keyword in chinese_keywords:
        if keyword in font_lower:
            chinese_fonts.append(font)
            break

print(f"找到 {len(chinese_fonts)} 个可能的中文字体:")
for font in sorted(set(chinese_fonts)):
    print(f"  • {font}")

# 检查Windows常用字体路径
print("\n检查Windows字体文件:")
font_paths = [
    'C:/Windows/Fonts/simhei.ttf',    # 黑体
    'C:/Windows/Fonts/msyh.ttc',      # 微软雅黑
    'C:/Windows/Fonts/simsun.ttc',    # 宋体
    'C:/Windows/Fonts/simkai.ttf',    # 楷体
]

for path in font_paths:
    exists = os.path.exists(path)
    status = "✓ 存在" if exists else "✗ 不存在"
    print(f"  {status}: {path}")

print("=" * 60)