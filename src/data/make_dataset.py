import pandas as pd
import os

def load_movie_data():
    """
    加载电影数据集 - 固定绝对路径
    """
    # 你的绝对路径
    filepath = r"D:\desktop\wenjianjia\buxiangbiancheng\lianxi\movie-analysis-project\data\raw\movies.csv"
    
    # 验证文件存在
    if not os.path.exists(filepath):
        print(f"错误: 文件不存在于 {filepath}")
        print("请检查:")
        print("1. 文件路径是否正确")
        print("2. 文件名是否正确 (movies.csv)")
        print("3. 文件是否被移动或删除")
        return None
    
    try:
        # 尝试读取文件
        df = pd.read_csv(filepath)
        
        print("=" * 50)
        print("数据加载成功!")
        print("=" * 50)
        print(f"文件路径: {filepath}")
        print(f"数据形状: {df.shape[0]} 行 × {df.shape[1]} 列")
        print(f"内存使用: {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
        print("\n数据概览:")
        print(df.head())
        
        return df
        
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")
        print("\n可能的原因:")
        print("1. 文件编码问题 - 尝试用记事本另存为UTF-8格式")
        print("2. 文件损坏 - 重新下载数据")
        print("3. 文件格式不是标准CSV")
        return None


if __name__ == "__main__":
    # 加载数据
    movie_data = load_movie_data()
    
    if movie_data is not None:
        print("\n🎬 数据分析:")
        print("-" * 30)
        print(f"数据列名: {list(movie_data.columns)}")
        print(f"数据类型:")
        print(movie_data.dtypes)
        
        # 检查缺失值
        missing = movie_data.isnull().sum()
        if missing.sum() > 0:
            print(f"\n缺失值统计:")
            for col, count in missing.items():
                if count > 0:
                    percent = (count / len(movie_data)) * 100
                    print(f"  {col}: {count} 个 ({percent:.1f}%)")