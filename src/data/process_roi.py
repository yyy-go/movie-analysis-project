import json
from matplotlib import category
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
from collections import Counter
warnings.filterwarnings('ignore')

# 设置中文字体（如果图表需要显示中文）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号




def load_and_preprocess_data(filepath):
    # 加载数据，预处理
    df = pd.read_csv(filepath)
    print(f"数据集形状：{df.shape}")
    print(f"列名：{df.columns.tolist()}")

    #查看数据基本信息
    print(df.info())
    #缺失值
    print(df.isnull().sum())

    #创建衍生指标
    df['budget'] = pd.to_numeric(df['budget'], errors='coerce')

    #过滤0或nan
    df = df[(df['budget'] > 0) & (df['revenue'] > 0)]
    df = df.dropna(subset=['budget','revenue'])

    #计算ROI和profit
    df['ROI'] = df['revenue'] / df['budget']
    df['profit'] = df['revenue'] - df['budget']

    #添加log变换的ROI（使分布更接近正态）
    df['log_ROI'] = np.log1p(df['ROI'])
    
    #其他数值特征
    df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce')
    df['vote_count'] = pd.to_numeric(df['vote_count'],errors='coerce')
    df['vote_average'] = pd.to_numeric(df['vote_average'],errors='coerce')

    #处理release_date
    df['release_year'] = pd.to_datetime(df['release_year'], errors='coerce')

    print(f"\n处理后数据形状：{df.shape}")
    print(f"ROI统计信息")
    print(df['ROI'].describe())

    return df

df = load_and_preprocess_data(r"D:\desktop\wenjianjia\buxiangbiancheng\lianxi\movie-analysis-project\data\processed\movies_cleaned.csv")


def exploratory_analysis(df):
    #探索性数据分析
    print("\n=== 探索性数据分析 ===")
    #ROI分布
    plt.figure(figsize=(15,10))
    plt.subplot(2,3,1)
    plt.hist(df['ROI'], bins=50, edgecolor='black',alpha=0.7)
    plt.title('ROI分布')
    plt.xlabel('ROI')
    plt.ylabel('频数')

    #ROI与vote_average的关系
    plt.subplot(2,3,2)
    plt.scatter(df['vote_average'], df['ROI'], alpha=0.5)
    plt.title('ROI vs 时长')
    plt.xlabel('runtime')
    plt.ylabel('ROI')

    #ROI与budget的关系
    plt.subplot(2, 3, 4)
    plt.scatter(np.log1p(df['budget']), np.log1p(df['ROI']), alpha=0.5)
    plt.title('ROI vs 预算（对数变换）')
    plt.xlabel('log(budget)')
    plt.ylabel('log(ROI)')

    #按年份的ROI趋势
    plt.subplot(2, 3, 5)
    yearly_roi = df.groupby('release_year')['ROI'].mean()
    yearly_roi.plot()
    plt.title('按年份平均ROI')
    plt.xlabel('年份')
    plt.ylabel('平均ROI')

    #ROI与vote_count的关系
    plt.subplot(2, 3, 6)
    plt.scatter(np.log1p(df['vote_count']), df['ROI'], alpha=0.5)
    plt.title('ROI vs 投票数')
    plt.xlabel('log(vote_count)')
    plt.ylabel('ROI')

    plt.tight_layout()
    #plt.savefig('eda_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

    #类别分析
    print("\n=== 类别分析 ===")
    
    # 处理genres
    if 'genres' in df.columns:
        genres_list = []
        for genres in df['genres'].dropna():
            try:
                # 尝试解析JSON字符串
                if isinstance(genres, str):
                    try:
                        genres_data = json.loads(genres)
                        if isinstance(genres_data, list):
                            for genre_item in genres_data:
                                if isinstance(genre_item, dict) and 'name' in genre_item:
                                    genres_list.append(genre_item['name'])
                    except json.JSONDecodeError:
                        # 如果不是JSON，尝试直接处理
                        pass
                # 如果是列表直接处理
                elif isinstance(genres, list):
                    for genre_item in genres:
                        if isinstance(genre_item, dict) and 'name' in genre_item:
                            genres_list.append(genre_item['name'])
            except Exception as e:
                print(f"处理类别时出错: {e}")
                continue
        
        if genres_list:  # 确保列表不为空
            genre_counts = Counter(genres_list)
            print(f"最常见出现的类别（前10）:")
            for genre, count in genre_counts.most_common(10):
                print(f"  {genre}: {count}次")
            
            # 可选：可视化展示
            top_10_genres = dict(genre_counts.most_common(10))
            categories = list(top_10_genres.keys())
            counts = list(top_10_genres.values())
            plt.figure(figsize=(12, 6))
            plt.bar(categories, counts)
            plt.title('Top 10 电影类别')
            plt.xlabel('类别')
            plt.ylabel('电影数量')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        else:
            print("未找到有效的类别数据")
    else:
        print("数据集中没有 'genres' 列")

    #相关性分析
    print("\n=== 数值特征相关性 ===")
    numeric_cols = ['budget', 'revenue', 'ROI', 'profit', 'runtime', 'vote_average', 'vote_count', 'release_year']
    numeric_df = df[numeric_cols].select_dtypes(include=[np.number])

    plt.figure(figsize=(10,8))
    corr_matrix = numeric_df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('特征相关性热图')
    plt.tight_layout()
    #plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\nROI与各特征的相关性")
    print(corr_matrix['ROI'].sort_values(ascending=False))

def feature_engineering(df):
    print("\n=== 特征工程 ===")
    df_featured = df.copy()
    if 'genres' in df_featured.columns:
        all_genres = set()
        for genres in df_featured['genres'].dropna():
            try:
                if isinstance(genres, str):
                     genre_list = json.loads(genres)
                elif isinstance(genres, list):
                    genre_list = genres
                else:
                    continue
                #提取类别名称         
                for genre_item in genre_list:
                    if isinstance(genre_item, dict) and 'name' in genre_item:
                        all_genres.add(genre_item['name'])
            
            except (json.JSONDecodeError,TypeError):
                continue
        
        #创建二进制特征
        for genre in all_genres:
            df_featured[f'genre_{genre.replace(" ","_").replace("-","_")}'] = df_featured['genres'].apply(
                lambda x: extract_genre_flag(x,genre)
            )
        
        print(f"创建了{len(all_genres)} 个类型特征")
        print(f"类别包括：{sorted(list(all_genres))[:10]}{'...' if len(all_genres) > 10 else ''}")

    #处理production_companies
    if 'production_companies' in df_featured.columns:
        all_companies = set()
        for companies in df_featured['production_companies'].dropna():
            try:
                if isinstance(companies,str):
                    company_list = json.loads(companies)
                elif isinstance(companies,list):
                    company_list = companies
                else:
                    continue
                
                for company_item in company_list:
                    if isinstance(company_item,dict) and 'name' in company_item:
                        all_companies.add(company_item['name'])
            except(json.JSONDecodeError, TypeError):
                continue
        df_featured['main_company'] = df_featured['production_companies'].apply(
            lambda x: extract_main_company(x)
        )

        if 'main_company' in df_featured.columns:
            company_counts = df_featured['main_company'].value_counts()
            print(f"\n制作公司统计（前10）：")
            print(company_counts.head(10))

            min_threshold = max(5,len(df_featured) * 0.01)
            top_companies = company_counts[company_counts >= min_threshold].index
            df_featured['main_company'] = df_featured['main_company'].apply(
                lambda x: x if x in top_companies else 'Other'
            )
            print(f"保留了 {len(top_companies)} 个主要公司，其他归为'Other'")

    #处理original_language
    if 'original_language' in df_featured.columns:
        lang_counts = df_featured['original_language'].value_counts()
        print(f'\n原始语言统计（前10）：')
        print(lang_counts.head(10))
        top_langs = lang_counts[lang_counts >= 20].index
        df_featured['original_language'] = df_featured['original_language'].apply(
            lambda x: str(x) if pd.notnull(x) and str(x) in top_langs else 'Other'
        )
        print(f"保留了 {len(top_langs)} 种主要语言，其他归为'Other'")
    #添加月份特征
    if 'release_date' in df_featured.columns:
        df_featured['release_month'] = pd.to_datetime(
            df_featured['release_date'], errors='coerce'
        ).dt.month
        print(f"\n上映月份分布:")
        print(df_featured['release_month'].value_counts().sort_index())
    #添加预算级别分布
    if 'budget' in df_featured.columns:
        df_featured['budget_level'] = pd.cut(df_featured['budget'],bins=[0, 1e6, 1e7, 1e8, np.inf],labels=['Low', 'Medium', 'High', 'Very High'])
        print(f"\n预算级别分布：")
        print(df_featured['budget_level'].value_counts())

    #对数变换数值特征
    numeric_features_to_log = ['budget','vote_count','revenue']
    for feature in numeric_features_to_log:
        if feature in df_featured.columns:
            if(df_featured[feature] < 0).any():
                print(f"{feature}列有负值，设为0")
                df_featured[feature] = df_featured[feature].clip(lower=0)
            df_featured[f'log_{feature}'] = np.log1p(df_featured[feature])
            print(f"对{feature}进行了对数变换 ")

    print(f"\n特征工程后数据形状：{df_featured.shape}")
    print(f"新增特征列：{[col for col in df_featured.columns if col not in df.columns][:10]}...")

    return df_featured

def extract_genre_flag(genres_data,target_genre):
    #从genres数据里提取特定类别的标志
    if pd.isnull(genres_data):
        return 0
    
    try:
        if isinstance(genres_data,str):
            genres_list = json.loads(genres_data)
        elif isinstance(genres_data,list):
            genres_list = genres_data
        else:
            return 0
        
        for genre_item in genres_list:
            if isinstance(genre_item,dict) and genre_item.get('name') == target_genre:
                return 1
    except (json.JSONDecodeError,TypeError):
        return 0
    
    return 0

def extract_main_company(companies_data):
    if pd.isnull(companies_data):
        return 'Unknown'
    
    try:
        if isinstance(companies_data,str):
            company_list = json.loads(companies_data)
        elif isinstance(companies_data,list):
            company_list = companies_data
        else:
            return 'Unknown'
        
        if company_list and len(company_list)>0:
            if isinstance(company_list[0],dict) and 'name' in company_list[0]:
                return company_list[0]['name']
            else:
                return str(company_list[0])
        else:
            return 'Unknown'
    except (json.JSONDecodeError,TypeError):
        return 'Unknown'


if __name__ == "__main__":
    # 最基本的调用方式
    filepath = r"D:\desktop\wenjianjia\buxiangbiancheng\lianxi\movie-analysis-project\data\processed\movies_cleaned.csv"
    
    # 顺序执行三个主要函数
    df = load_and_preprocess_data(filepath)
    exploratory_analysis(df)
    df_featured = feature_engineering(df)
    
    # 保存结果
    output_path = filepath.replace('.csv', '_with_features.csv')
    df_featured.to_csv(output_path, index=False)




                    








    
