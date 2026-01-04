# test_genres.py
import pandas as pd
import json
import ast  # 可能使用ast.literal_eval

def test_genres_parsing(filepath):
    # 加载数据
    df = pd.read_csv(filepath)
    
    print("数据集信息:")
    print(f"形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    
    if 'genres' not in df.columns:
        print("没有genres列")
        return
    
    # 查看genres列信息
    print(f"\ngenres列信息:")
    print(f"数据类型: {df['genres'].dtype}")
    print(f"非空数量: {df['genres'].notna().sum()}")
    print(f"空值数量: {df['genres'].isna().sum()}")
    
    # 获取非空样本
    non_null = df['genres'].dropna()
    
    if len(non_null) == 0:
        print("没有非空的genres数据")
        return
    
    # 分析前10个样本
    print(f"\n分析前10个样本:")
    
    for i in range(min(10, len(non_null))):
        sample = non_null.iloc[i]
        print(f"\n--- 样本 {i} ---")
        print(f"类型: {type(sample)}")
        print(f"原始值: {repr(sample)}")
        
        # 尝试不同的解析方法
        parsed_data = None
        
        # 方法1: 如果是字符串且看起来像JSON
        if isinstance(sample, str):
            # 去除可能的引号问题
            clean_str = sample.strip()
            
            # 尝试json.loads
            try:
                parsed = json.loads(clean_str)
                print(f"JSON解析成功: {type(parsed)}")
                parsed_data = parsed
            except json.JSONDecodeError:
                print(f"JSON解析失败")
                
                # 方法2: 尝试ast.literal_eval (处理Python字面量)
                try:
                    parsed = ast.literal_eval(clean_str)
                    print(f"ast.literal_eval成功: {type(parsed)}")
                    parsed_data = parsed
                except:
                    print(f"ast.literal_eval失败")
                    
            # 方法3: 如果是其他格式
            if parsed_data is None:
                print(f"尝试直接处理字符串")
                # 这里可以根据实际格式添加处理逻辑
        
        # 方法4: 如果是列表
        elif isinstance(sample, list):
            print(f"直接是列表，长度: {len(sample)}")
            parsed_data = sample
        
        # 提取类别名称
        if parsed_data:
            print(f"解析后的结构: {type(parsed_data)}")
            if isinstance(parsed_data, list):
                print(f"列表内容:")
                for j, item in enumerate(parsed_data):
                    print(f"  元素{j}: {type(item)}, 值: {repr(item)}")
                    if isinstance(item, dict):
                        print(f"    字典键: {list(item.keys())}")
                        if 'name' in item:
                            print(f"    类别名称: {item['name']}")

if __name__ == "__main__":
    filepath = r"D:\desktop\wenjianjia\buxiangbiancheng\lianxi\movie-analysis-project\data\processed\movies_cleaned.csv"
    test_genres_parsing(filepath)