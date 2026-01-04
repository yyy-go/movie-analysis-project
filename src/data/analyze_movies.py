import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import os

warnings.filterwarnings('ignore')

def data_quality_check(df):
    """
    数据质量检查
    """
    print("=" * 60)
    print("数据质量检查报告")
    print("=" * 60)
    
    # 基本信息
    print("1. 数据集基本信息:")
    print(f"   总行数: {df.shape[0]}")
    print(f"   总列数: {df.shape[1]}")
    print(f"   内存使用: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
    
    # 缺失值分析
    print("\n2. 缺失值分析:")
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame({
        '缺失数量': missing_values,
        '缺失比例(%)': missing_percent
    }).sort_values('缺失数量', ascending=False)
    
    # 只显示有缺失值的列
    missing_df = missing_df[missing_df['缺失数量'] > 0]
    
    if len(missing_df) > 0:
        print("   发现缺失值的列:")
        print(missing_df.to_string())
    else:
        print("   没有发现缺失值")
    
    # 数据类型检查
    print("\n3. 数据类型分布:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"   {dtype}: {count} 列")
    
    # 重复值检查
    print("\n4. 重复行检查:")
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"   发现 {duplicates} 个重复行")
        print("   重复行示例:")
        print(df[df.duplicated()].head())
    else:
        print("   没有发现重复行")
    
    return missing_df

def handle_missing_values(df, strategy='median'):
    """
    处理缺失值
    """
    print("\n" + "=" * 60)
    print("缺失值处理")
    print("=" * 60)
    
    df_clean = df.copy()
    missing_before = df.isnull().sum().sum()
    
    if missing_before == 0:
        print("没有缺失值需要处理")
        return df_clean
    
    print(f"处理前总缺失值: {missing_before}")
    
    # 针对不同类型列采用不同策略
    for column in df_clean.columns:
        missing_count = df_clean[column].isnull().sum()
        
        if missing_count > 0:
            col_type = df_clean[column].dtype
            
            if col_type in ['float64', 'int64']:
                # 数值型列
                if strategy == 'median':
                    fill_value = df_clean[column].median()
                elif strategy == 'mean':
                    fill_value = df_clean[column].mean()
                elif strategy == 'mode':
                    fill_value = df_clean[column].mode()[0]
                else:  # zero
                    fill_value = 0
                    
                df_clean[column].fillna(fill_value, inplace=True)
                print(f"   {column}: 数值型，用{strategy}({fill_value:.2f})填充 {missing_count} 个缺失值")
                
            elif col_type == 'object':
                # 文本型列
                if df_clean[column].nunique() < 20:  # 类别数少用众数
                    fill_value = df_clean[column].mode()[0] if not df_clean[column].mode().empty else 'Unknown'
                else:  # 类别数多用Unknown
                    fill_value = 'Unknown'
                    
                df_clean[column].fillna(fill_value, inplace=True)
                print(f"   {column}: 文本型，用'{fill_value}'填充 {missing_count} 个缺失值")
                
            elif 'datetime' in str(col_type):
                # 时间型列
                df_clean[column].fillna(pd.NaT, inplace=True)
                print(f"   {column}: 时间型，用NaT填充 {missing_count} 个缺失值")
    
    missing_after = df_clean.isnull().sum().sum()
    print(f"\n处理后总缺失值: {missing_after}")
    print(f"清理了 {missing_before - missing_after} 个缺失值")
    
    return df_clean

def detect_outliers(df, column):
    """
    使用 IQR方法（箱线图原理） 来检测数值型数据列的异常值。
    """
    if df[column].dtype not in ['float64', 'int64']:
        return None, None, None
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    return outliers, lower_bound, upper_bound

def outlier_analysis(df):
    """
    异常值分析
    """
    print("\n" + "=" * 60)
    print("异常值检测分析")
    print("=" * 60)
    
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    outlier_report = []
    
    for col in numeric_cols:
        outliers, lower, upper = detect_outliers(df, col)
        
        if outliers is not None and len(outliers) > 0:
            outlier_count = len(outliers)
            outlier_percent = (outlier_count / len(df)) * 100
            
            outlier_report.append({
                '列名': col,
                '异常值数量': outlier_count,
                '异常值比例(%)': outlier_percent,
                '下界': lower,
                '上界': upper,
                '最小值': df[col].min(),
                '最大值': df[col].max()
            })
    
    if outlier_report:
        outlier_df = pd.DataFrame(outlier_report)
        print("发现异常值的列:")
        print(outlier_df.to_string(index=False))
        
        # 显示异常值详情
        print("\n异常值详情:")
        for report in outlier_report:
            print(f"\n{report['列名']}:")
            print(f"   异常值范围: < {report['下界']:.2f} 或 > {report['上界']:.2f}")
            print(f"   实际范围: {report['最小值']:.2f} - {report['最大值']:.2f}")
            print(f"   异常值占比: {report['异常值比例(%)']:.1f}%")
            
    else:
        print("没有发现明显的异常值")
    
    return outlier_report if outlier_report else None

def basic_distribution_analysis(df):
    """
    基本分布分析
    """
    print("\n" + "=" * 60)
    print("基本分布分析")
    print("=" * 60)
    
    # 1. 评分分布分析
    print("1. 评分分布分析:")
    if 'vote_average' in df.columns or 'rating' in df.columns:
        rating_col = 'vote_average' if 'vote_average' in df.columns else 'rating'
        
        rating_stats = df[rating_col].describe()
        print(f"   评分列: {rating_col}")
        print(f"   平均评分: {rating_stats['mean']:.2f}")
        print(f"   中位数: {rating_stats['50%']:.2f}")
        print(f"   标准差: {rating_stats['std']:.2f}")
        print(f"   范围: {rating_stats['min']:.2f} - {rating_stats['max']:.2f}")
        
        # 评分分布统计
        print(f"   25%分位数: {df[rating_col].quantile(0.25):.2f}")
        print(f"   75%分位数: {df[rating_col].quantile(0.75):.2f}")
        print(f"   偏度: {df[rating_col].skew():.2f}")
        print(f"   峰度: {df[rating_col].kurtosis():.2f}")
    
    # 2. 类型分布分析
    print("\n2. 类型分布分析:")
    genre_cols = ['genres', 'genre', 'genre_ids']
    genre_col = next((col for col in genre_cols if col in df.columns), None)
    
    if genre_col:
        # 处理类型数据（可能有多类型用逗号分隔）
        try:
            if df[genre_col].dtype == 'object':
                # 分割类型并统计
                all_genres = []
                for genres in df[genre_col].dropna():
                    if isinstance(genres, str):
                        # 假设类型用逗号、分号或|分隔
                        split_chars = [',', ';', '|', '/']
                        for char in split_chars:
                            if char in genres:
                                all_genres.extend([g.strip() for g in genres.split(char)])
                                break
                        else:
                            all_genres.append(genres.strip())
                
                genre_counts = pd.Series(all_genres).value_counts().head(15)
                
                print(f"   最受欢迎的15种电影类型:")
                for genre, count in genre_counts.items():
                    percentage = (count / len(df)) * 100
                    print(f"   {genre}: {count} 部 ({percentage:.1f}%)")
                
        except Exception as e:
            print(f"   类型分析出错: {e}")
    
    # 3. 时间分布分析
    print("\n3. 时间分布分析:")
    date_cols = ['release_date', 'year', 'release_year']
    date_col = next((col for col in date_cols if col in df.columns), None)
    
    if date_col:
        try:
            # 提取年份
            if df[date_col].dtype == 'object':
                # 尝试解析日期
                df['release_year'] = pd.to_datetime(df[date_col], errors='coerce').dt.year
            elif 'int' in str(df[date_col].dtype):
                df['release_year'] = df[date_col]
            
            year_counts = df['release_year'].dropna().astype(int).value_counts().sort_index()
            
            print(f"   电影数量最多的5个年份:")
            top_years = year_counts.head()
            for year, count in top_years.items():
                print(f"   {int(year)}年: {count} 部")
            
            print(f"\n   电影数量最少的5个年份:")
            bottom_years = year_counts.tail()
            for year, count in bottom_years.items():
                print(f"   {int(year)}年: {count} 部")
            
        except Exception as e:
            print(f"   时间分析出错: {e}")

def correlation_analysis(df):
    """
    相关性分析
    """
    print("\n" + "=" * 60)
    print("相关性分析")
    print("=" * 60)
    
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    if len(numeric_cols) > 1:
        correlation_matrix = df[numeric_cols].corr()
        
        print("数值列的相关性矩阵:")
        
        # 只显示相关性较强的部分
        strong_corr_threshold = 0.7
        
        strong_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > strong_corr_threshold:
                    strong_corr_pairs.append((
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        corr_value
                    ))
        
        if strong_corr_pairs:
            print(f"\n发现强相关性 (|r| > {strong_corr_threshold}):")
            for col1, col2, corr in strong_corr_pairs:
                correlation_type = "正相关" if corr > 0 else "负相关"
                print(f"   {col1} 和 {col2}: {corr:.3f} ({correlation_type})")
        else:
            print(f"\n未发现强相关性 (|r| > {strong_corr_threshold})")

def categorical_analysis(df):
    """
    分类变量分析
    """
    print("\n" + "=" * 60)
    print("分类变量分析")
    print("=" * 60)
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    if len(categorical_cols) > 0:
        print(f"分类列总数: {len(categorical_cols)}")
        
        for col in categorical_cols:
            unique_count = df[col].nunique()
            missing_count = df[col].isnull().sum()
            
            print(f"\n{col}:")
            print(f"   唯一值数量: {unique_count}")
            print(f"   缺失值数量: {missing_count} ({missing_count/len(df)*100:.1f}%)")
            
            if unique_count <= 10:
                # 对于类别数少的列，显示分布
                value_counts = df[col].value_counts()
                print(f"   值分布:")
                for value, count in value_counts.items():
                    percentage = (count / len(df)) * 100
                    print(f"     {value}: {count} ({percentage:.1f}%)")
            elif unique_count <= 20:
                # 对于中等类别数的列，只显示前5个
                top_values = df[col].value_counts().head(5)
                print(f"   前5个值:")
                for value, count in top_values.items():
                    percentage = (count / len(df)) * 100
                    print(f"     {value}: {count} ({percentage:.1f}%)")
            else:
                # 对于类别数多的列，只显示基本信息
                print(f"   (唯一值过多，不显示具体分布)")
    
    else:
        print("没有分类列")

def summary_statistics(df):
    """
    汇总统计信息
    """
    print("\n" + "=" * 60)
    print("汇总统计信息")
    print("=" * 60)
    
    print(f"数据集维度: {df.shape[0]} 行 × {df.shape[1]} 列")
    
    # 数值列统计
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 0:
        print(f"\n数值列 ({len(numeric_cols)}个):")
        numeric_summary = df[numeric_cols].describe().T
        numeric_summary['范围'] = numeric_summary['max'] - numeric_summary['min']
        numeric_summary['变异系数(%)'] = (numeric_summary['std'] / numeric_summary['mean']).abs() * 100
        
        print(numeric_summary[['mean', 'std', 'min', '25%', '50%', '75%', 'max', '范围', '变异系数(%)']].round(2))
    
    # 分类列统计
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print(f"\n分类列 ({len(categorical_cols)}个):")
        cat_summary = pd.DataFrame({
            '唯一值数量': [df[col].nunique() for col in categorical_cols],
            '缺失值数量': [df[col].isnull().sum() for col in categorical_cols],
            '缺失比例(%)': [(df[col].isnull().sum() / len(df) * 100).round(1) for col in categorical_cols]
        }, index=categorical_cols)
        
        print(cat_summary)

def analyze_movies(df):
    """
    完整的电影数据分析流程
    """
    print("开始电影数据分析")
    print("=" * 70)
    
    # 1. 数据质量检查
    missing_df = data_quality_check(df)
    
    # 2. 处理缺失值
    df_clean = handle_missing_values(df)
    
    # 3. 异常值检测
    outlier_analysis(df_clean)
    
    # 4. 基本分布分析
    basic_distribution_analysis(df_clean)
    
    # 5. 相关性分析
    correlation_analysis(df_clean)
    
    # 6. 分类变量分析
    categorical_analysis(df_clean)
    
    # 7. 汇总统计
    summary_statistics(df_clean)
    
    print("\n" + "=" * 70)
    print("分析完成！")
    print("=" * 70)
    
    return df_clean

# 使用示例
if __name__ == "__main__":
    # 导入数据加载函数
    from make_dataset import load_movie_data
    
    # 加载数据
    print("加载电影数据...")
    df = load_movie_data()
    
    if df is not None:
        # 执行完整分析
        df_clean = analyze_movies(df)
        
        # 保存清理后的数据
        output_path = 'data/processed/movies_cleaned.csv'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_clean.to_csv(output_path, index=False)
        print(f"\n清理后的数据已保存到: {output_path}")
        
        # 生成分析报告摘要
        print("\n" + "=" * 70)
        print("分析报告摘要")
        print("=" * 70)
        print(f"原始数据大小: {df.shape}")
        print(f"清理后数据大小: {df_clean.shape}")
        print(f"处理缺失值数量: {df.isnull().sum().sum() - df_clean.isnull().sum().sum()}")
        print(f"数值列数量: {len(df_clean.select_dtypes(include=['float64', 'int64']).columns)}")
        print(f"分类列数量: {len(df_clean.select_dtypes(include=['object']).columns)}")
        print("=" * 70)
    else:
        print("无法加载数据，请检查数据文件")