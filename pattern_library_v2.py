# 规律库预测系统 v2.0 - 近期加权版
# 基于滑动窗口特征匹配的智能预测，近期数据权重更高
import pandas as pd
from collections import Counter, defaultdict
import sys
sys.stdout.reconfigure(encoding='utf-8')

# ============ 基础数据 ============
zm = {1:'马',13:'马',25:'马',37:'马',49:'马',2:'蛇',14:'蛇',26:'蛇',38:'蛇',3:'龙',15:'龙',27:'龙',39:'龙',4:'兔',16:'兔',28:'兔',40:'兔',5:'虎',17:'虎',29:'虎',41:'虎',6:'牛',18:'牛',30:'牛',42:'牛',7:'鼠',19:'鼠',31:'鼠',43:'鼠',8:'猪',20:'猪',32:'猪',44:'猪',9:'狗',21:'狗',33:'狗',45:'狗',10:'鸡',22:'鸡',34:'鸡',46:'鸡',11:'猴',23:'猴',35:'猴',47:'猴',12:'羊',24:'羊',36:'羊',48:'羊'}
zs_all = ['鼠','牛','虎','兔','龙','蛇','马','羊','猴','鸡','狗','猪']

# ============ 数据加载 ============
def load(fp, sc=6, ec=13):
    df = pd.read_excel(fp, header=None, engine='openpyxl')
    data = []
    for i in range(len(df)):
        row = df.iloc[i]
        nums = []
        for col in range(sc, ec):
            v = row[col]
            if pd.notna(v):
                try:
                    n = int(float(v))
                    if 1 <= n <= 49:
                        nums.append(n)
                except:
                    pass
        if len(nums) == 7:
            data.append({'nums': nums, 'zs': [zm[n] for n in nums]})
    return data

# ============ 遗漏计算 ============
def calc_miss(train):
    last = train[-1]['zs']
    miss = {}
    for z in zs_all:
        if z in last:
            miss[z] = 0
        else:
            m = 1
            for i in range(len(train)-2, -1, -1):
                if z in train[i]['zs']:
                    break
                m += 1
            miss[z] = m
    return miss

# ============ 策略函数 ============
def s1(train):  # 近5热
    c = Counter()
    for d in train[-5:]: c.update(d['zs'])
    return c.most_common(1)[0][0]

def s2(train):  # 近10热
    c = Counter()
    for d in train[-10:]: c.update(d['zs'])
    return c.most_common(1)[0][0]

def s3(train):  # 近5热+漏>=2
    c = Counter()
    for d in train[-5:]: c.update(d['zs'])
    miss = calc_miss(train)
    for z,_ in c.most_common():
        if miss[z] >= 2:
            return z
    return c.most_common(1)[0][0]

def s4(train):  # 近10热+漏>=2
    c = Counter()
    for d in train[-10:]: c.update(d['zs'])
    miss = calc_miss(train)
    for z,_ in c.most_common():
        if miss[z] >= 2:
            return z
    return c.most_common(1)[0][0]

def s5(train):  # 近5热+在上期
    last = set(train[-1]['zs'])
    c = Counter()
    for d in train[-5:]: c.update(d['zs'])
    for z,_ in c.most_common():
        if z in last:
            return z
    return c.most_common(1)[0][0]

def s6(train):  # 近10热+在上期
    last = set(train[-1]['zs'])
    c = Counter()
    for d in train[-10:]: c.update(d['zs'])
    for z,_ in c.most_common():
        if z in last:
            return z
    return c.most_common(1)[0][0]

def s7(train):  # 近5热+不在上期
    last = set(train[-1]['zs'])
    c = Counter()
    for d in train[-5:]: c.update(d['zs'])
    for z,_ in c.most_common():
        if z not in last:
            return z
    return c.most_common(1)[0][0]

def s8(train):  # 近10热+不在上期
    last = set(train[-1]['zs'])
    c = Counter()
    for d in train[-10:]: c.update(d['zs'])
    for z,_ in c.most_common():
        if z not in last:
            return z
    return c.most_common(1)[0][0]

strategies = [
    ('S1', s1, '近5期最热'),
    ('S2', s2, '近10期最热'),
    ('S3', s3, '近5期最热+遗漏>=2'),
    ('S4', s4, '近10期最热+遗漏>=2'),
    ('S5', s5, '近5期最热+上期出现'),
    ('S6', s6, '近10期最热+上期出现'),
    ('S7', s7, '近5期最热+不在上期'),
    ('S8', s8, '近10期最热+不在上期'),
]
strategy_funcs = {name: func for name, func, _ in strategies}
strategy_descs = {name: desc for name, _, desc in strategies}

# ============ 特征提取 ============
def extract_features(train):
    c = Counter()
    for d in train: c.update(d['zs'])
    miss = calc_miss(train)
    
    # 重号率
    repeat_count = 0
    for i in range(1, len(train)):
        last_zs = set(train[i-1]['zs'])
        curr_zs = set(train[i]['zs'])
        repeat_count += len(last_zs & curr_zs)
    avg_repeat = repeat_count / (len(train) - 1)
    
    # 热度集中度
    top3_count = sum(cnt for z, cnt in c.most_common(3))
    total_count = sum(c.values())
    concentration = top3_count / total_count
    
    # 遗漏波动
    miss_std = sum((miss[z] - sum(miss.values())/12)**2 for z in zs_all) ** 0.5 / 12
    
    # 热度比值
    hot1 = c.most_common(1)[0][1]
    hot2 = c.most_common(2)[1][1] if len(c) > 1 else 0
    hot_ratio = hot1 / hot2 if hot2 > 0 else 3
    
    # 近期热度
    c5 = Counter()
    for d in train[-5:]: c5.update(d['zs'])
    hot5_top = c5.most_common(1)[0][1]
    
    # 上期在热度中
    last_zs = set(train[-1]['zs'])
    last_in_hot5 = sum(1 for z in last_zs if z in [x[0] for x in c5.most_common(3)])
    
    features = {
        'repeat': 1 if avg_repeat < 2.0 else (2 if avg_repeat < 2.5 else (3 if avg_repeat < 3.0 else 4)),
        'concentration': 1 if concentration < 0.35 else (2 if concentration < 0.40 else 3),
        'miss_volatility': 1 if miss_std < 0.3 else (2 if miss_std < 0.5 else 3),
        'hot_ratio': 1 if hot_ratio < 1.3 else (2 if hot_ratio < 1.5 else 3),
        'hot5_level': 1 if hot5_top < 5 else (2 if hot5_top < 7 else 3),
        'last_in_hot': 1 if last_in_hot5 >= 3 else (2 if last_in_hot5 >= 2 else 3),
    }
    
    return features, avg_repeat, concentration

# ============ 规律库类 ============
class PatternLibrary:
    def __init__(self, recent_weight=3.0):
        """
        recent_weight: 近期数据权重倍数（越大越重视近期）
        """
        self.library = {}
        self.strategy_map = {}
        self.recent_weight = recent_weight
    
    def build(self, data):
        """构建规律库 - 近期加权"""
        n = len(data)
        
        for window_start in range(0, n - 11):
            window_end = window_start + 10
            train = data[window_start:window_end]
            actual = set(data[window_end]['zs'])
            
            features, _, _ = extract_features(train)
            feature_key = tuple(sorted(features.items()))
            
            # 计算时间权重（近期权重更高）
            time_weight = 1.0 + (window_start / (n - 20)) * (self.recent_weight - 1)
            
            # 找最佳策略
            for name, func in strategy_funcs.items():
                pred = func(train)
                hit = 1 if pred in actual else 0
                
                if feature_key not in self.library:
                    self.library[feature_key] = {
                        'samples': [],
                        'strategy_scores': defaultdict(float)
                    }
                
                self.library[feature_key]['samples'].append('{}-{}期'.format(window_start+1, window_end))
                self.library[feature_key]['strategy_scores'][name] += hit * time_weight
        
        # 生成策略映射
        for key, info in self.library.items():
            best = max(info['strategy_scores'].items(), key=lambda x: x[1])
            self.strategy_map[key] = {
                'strategy': best[0],
                'weighted_score': best[1],
                'count': len(info['samples'])
            }
    
    def predict(self, train):
        """预测"""
        features, avg_repeat, concentration = extract_features(train)
        feature_key = tuple(sorted(features.items()))
        
        if feature_key in self.strategy_map:
            return self.strategy_map[feature_key]['strategy'], '精确', feature_key
        else:
            # 模糊匹配
            best_similar = None
            best_diff = float('inf')
            for key in self.strategy_map:
                diff = sum(1 for a, b in zip(sorted(features.items()), key) if a[1] != b[1])
                if diff < best_diff:
                    best_diff = diff
                    best_similar = key
            return self.strategy_map[best_similar]['strategy'], '模糊({})'.format(best_diff), best_similar
    
    def get_prediction(self, train):
        """获取预测生肖"""
        strategy_name, match_type, key = self.predict(train)
        func = strategy_funcs[strategy_name]
        return func(train), strategy_name, match_type, key

# ============ 主程序 ============
if __name__ == '__main__':
    import os
    
    data_file = r'D:\Desktop\数据2026_correct.xlsx'
    
    print('='*70)
    print('规律库预测系统 v2.0 - 近期加权版')
    print('='*70)
    print()
    
    if not os.path.exists(data_file):
        print('错误: 数据文件不存在')
        sys.exit(1)
    
    data = load(data_file)
    print('加载数据: {}期'.format(len(data)))
    
    # 构建规律库（近期权重=3倍）
    print('构建规律库（近期权重=3倍）...')
    library = PatternLibrary(recent_weight=3.0)
    library.build(data)
    print('规律库: {}个特征组合'.format(len(library.library)))
    
    # 预测最新一期
    train = data[-10:]
    prediction, strategy, match_type, key = library.get_prediction(train)
    
    print()
    print('='*70)
    print('第89期预测结果')
    print('='*70)
    print()
    print('数据范围: 第{}期 - 第{}期'.format(len(data)-9, len(data)))
    print('特征键值: {}'.format(key))
    print('匹配类型: {}'.format(match_type))
    print('选定策略: {} ({})'.format(strategy, strategy_descs[strategy]))
    print()
    print('预测生肖: 【{}】'.format(prediction))
    
    # 显示热度排名
    print()
    print('='*70)
    print('热度分析')
    print('='*70)
    c = Counter()
    for d in train: c.update(d['zs'])
    miss = calc_miss(train)
    
    print()
    print('近10期热度:')
    for z, cnt in c.most_common():
        print('  {}: {}次 (遗漏{})'.format(z, cnt, miss[z]))
    
    print()
    print('近5期热度:')
    c5 = Counter()
    for d in train[-5:]: c5.update(d['zs'])
    for z, cnt in c5.most_common():
        print('  {}: {}次 (遗漏{})'.format(z, cnt, miss[z]))
    
    # 验证预测
    print()
    print('='*70)
    print('策略验证')
    print('='*70)
    print()
    for name, func, desc in strategies:
        pred = func(train)
        print('{}: {}'.format(name, pred))
