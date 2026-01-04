import os

print("验证数据文件是否存在...")
print("当前工作目录:", os.getcwd())

# 检查文件
file_path = "data/raw/movies.csv"
abs_path = os.path.abspath(file_path)

print(f"\n检查文件: {file_path}")
print(f"绝对路径: {abs_path}")

if os.path.exists(file_path):
    print("✓ 文件存在!")
    size = os.path.getsize(file_path)
    print(f"文件大小: {size} 字节 ({size/1024:.2f} KB)")
    
    # 显示文件内容
    print("\n文件内容预览:")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i in range(6):  # 显示前6行
                line = f.readline()
                if not line:
                    break
                print(f"第{i+1}行: {line.strip()}")
    except:
        try:
            with open(file_path, 'r', encoding='gbk') as f:
                for i in range(6):
                    line = f.readline()
                    if not line:
                        break
                    print(f"第{i+1}行: {line.strip()}")
        except Exception as e:
            print(f"无法读取文件: {e}")
else:
    print("✗ 文件不存在")
    
    # 列出data/raw目录内容
    raw_dir = "data/raw"
    if os.path.exists(raw_dir):
        print(f"\n{raw_dir} 目录内容:")
        files = os.listdir(raw_dir)
        if files:
            for file in files:
                full_path = os.path.join(raw_dir, file)
                if os.path.isfile(full_path):
                    size = os.path.getsize(full_path)
                    print(f"  - {file} ({size} 字节)")
                else:
                    print(f"  - {file}/ (目录)")
        else:
            print("  目录为空")
    else:
        print(f"\n{raw_dir} 目录不存在")

print("\n" + "=" * 50)