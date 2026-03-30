# 规律库预测系统 v4.0 - 位置特征+优化权重版
import pandas as pd
from collections import Counter, defaultdict
import sys
sys.stdout.reconfigure(encoding='utf-8')

zm = {1:'马',13:'马',25:'马',37:'马',49:'马',2:'蛇',14:'蛇',26:'蛇',38:'蛇',3:'龙',15:'龙',27:'龙',39:'龙',4:'兔',16:'兔',28:'兔',40:'兔',5:'虎',17:'虎',29:'虎',41:'虎',6:'牛',18:'牛',30:'牛',42:'牛',7:'鼠',19:'鼠',31:'鼠',43:'鼠',8:'猪',20:'猪',32:'猪',44:'猪',9:'狗',21:'狗',33:'狗',45:'狗',10:'鸡',22:'鸡',34:'鸡',46:'鸡',11:'猴',23:'猴',35:'猴',47:'猴',12:'羊',24:'羊',36:'羊',48:'羊'}
zs_all = ['鼠','牛','虎','兔','龙','蛇','马','羊','猴','鸡','狗','猪']

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

# ============ 策略生成器（近N期 x 4种条件）============
def make_strategies():
    strats = {}
    for n in range(5, 11):
        def make_hot(n=n):
            def s(train):
                c = Counter()
                for d in train[-n:]: c.update(d['zs'])
                return c.most_common(1)[0][0]
            return s
        def make_miss2(n=n):
            def s(train):
                c = Counter()
                for d in train[-n:]: c.update(d['zs'])
                miss = calc_miss(train)
                for z,_ in c.most_common():
                    if miss[z] >= 2:
                        return z
                return c.most_common(1)[0][0]
            return s
        def make_last_in(n=n):
            def s(train):
                last = set(train[-1]['zs'])
                c = Counter()
                for d in train[-n:]: c.update(d['zs'])
                for z,_ in c.most_common():
                    if z in last:
                        return z
                return c.most_common(1)[0][0]
            return s
        def make_last_out(n=n):
            def s(train):
                last = set(train[-1]['zs'])
                c = Counter()
                for d in train[-n:]: c.update(d['zs'])
                for z,_ in c.most_common():
                    if z not in last:
                        return z
                return c.most_common(1)[0][0]
            return s
        strats[f'S{n}A'] = (make_hot(n),    f'近{n}期最热')
        strats[f'S{n}B'] = (make_miss2(n),  f'近{n}期热+漏>=2')
        strats[f'S{n}C'] = (make_last_in(n),f'近{n}期热+在上期')
        strats[f'S{n}D'] = (make_last_out(n),f'近{n}期热+不在上期')
    return strats

strategies = make_strategies()
strategy_funcs = {k: v[0] for k, v in strategies.items()}
strategy_descs = {k: v[1] for k, v in strategies.items()}

# ============ 特征提取（9个维度）============
def extract_features(train, pos=None):
    """提取特征，可选指定位置"""
    c = Counter()
    for d in train: c.update(d['zs'])
    miss = calc_miss(train)

    # 基础特征
    repeat_count = 0
    for i in range(1, len(train)):
        last_zs = set(train[i-1]['zs'])
        curr_zs = set(train[i]['zs'])
        repeat_count += len(last_zs & curr_zs)
    avg_repeat = repeat_count / (len(train) - 1)

    top3_count = sum(cnt for z, cnt in c.most_common(3))
    total_count = sum(c.values())
    concentration = top3_count / total_count

    miss_std = sum((miss[z] - sum(miss.values())/12)**2 for z in zs_all) ** 0.5 / 12

    hot1 = c.most_common(1)[0][1]
    hot2 = c.most_common(2)[1][1] if len(c) > 1 else 0
    hot_ratio = hot1 / hot2 if hot2 > 0 else 3

    c5 = Counter()
    for d in train[-5:]: c5.update(d['zs'])
    hot5_top = c5.most_common(1)[0][1]

    last_zs = set(train[-1]['zs'])
    last_in_hot5 = sum(1 for z in last_zs if z in [x[0] for x in c5.most_common(3)])

    # 位置特征（新增）
    if pos is not None and pos < 7:
        pos_counter = Counter()
        for d in train:
            if pos < len(d['zs']):
                pos_counter[d['zs'][pos]] += 1
        pos_hot = pos_counter.most_common(1)[0][0] if pos_counter else None
        pos_count = pos_counter.most_common(1)[0][1] if pos_counter else 0
    else:
        pos_hot = None
        pos_count = 0

    features = {
        'repeat':         1 if avg_repeat < 2.0 else (2 if avg_repeat < 2.5 else (3 if avg_repeat < 3.0 else 4)),
        'concentration':  1 if concentration < 0.35 else (2 if concentration < 0.40 else 3),
        'miss_volatility':1 if miss_std < 0.3 else (2 if miss_std < 0.5 else 3),
        'hot_ratio':      1 if hot_ratio < 1.3 else (2 if hot_ratio < 1.5 else 3),
        'hot5_level':     1 if hot5_top < 5 else (2 if hot5_top < 7 else 3),
        'last_in_hot':    1 if last_in_hot5 >= 3 else (2 if last_in_hot5 >= 2 else 3),
        'pos_hot':        pos_hot if pos_hot else 'NA',
        'pos_count':      1 if pos_count < 3 else (2 if pos_count < 5 else 3),
    }
    return features

# ============ 规律库（带权重优化）============
class PatternLibrary:
    def __init__(self, strategies, recent_weight=3.0, feature_weights=None):
        self.strategies = strategies
        self.library = {}
        self.strategy_map = {}
        self.recent_weight = recent_weight
        # 特征权重（新增）
        self.feature_weights = feature_weights or {
            'repeat': 1.0,
            'concentration': 1.2,  # 热度集中度更重要
            'miss_volatility': 1.0,
            'hot_ratio': 1.1,  # 热比略重要
            'hot5_level': 1.0,
            'last_in_hot': 1.0,
            'pos_hot': 1.5,  # 位置特征最重要
            'pos_count': 1.2,
        }

    def build(self, data):
        n = len(data)
        for window_start in range(0, n - 11):
            window_end = window_start + 10
            train = data[window_start:window_end]
            actual = set(data[window_end]['zs'])

            # 时间权重优化（非线性增长）
            progress = window_start / max(n - 20, 1)
            time_weight = 1.0 + progress ** 2 * (self.recent_weight - 1)

            # 为每个位置建立规律库
            for pos in range(7):
                features = extract_features(train, pos=pos)
                fkey = tuple(sorted((k, v) for k, v in features.items() if k != 'pos_hot'))

                if fkey not in self.library:
                    self.library[fkey] = defaultdict(float)

                for name, func in self.strategies.items():
                    pred = func(train)
                    hit = 1 if pred in actual else 0
                    self.library[fkey][name] += hit * time_weight

        for key, scores in self.library.items():
            best = max(scores.items(), key=lambda x: x[1])
            self.strategy_map[key] = best[0]

    def predict(self, train, pos=None):
        features = extract_features(train, pos=pos)
        fkey = tuple(sorted((k, v) for k, v in features.items() if k != 'pos_hot'))

        if fkey in self.strategy_map:
            return self.strategy_map[fkey], '精确', fkey
        else:
            # 加权距离计算
            def weighted_distance(k):
                dist = 0
                for (k1, v1), (k2, v2) in zip(sorted(features.items()), k):
                    if k1 != 'pos_hot' and k2 != 'pos_hot':
                        weight = self.feature_weights.get(k1, 1.0)
                        dist += weight * (1 if v1 != v2 else 0)
                return dist

            best_key = min(self.strategy_map.keys(), key=weighted_distance)
            diff = weighted_distance(best_key)
            return self.strategy_map[best_key], f'模糊({diff:.1f})', best_key

    def get_prediction(self, train, pos=None):
        name, match_type, key = self.predict(train, pos=pos)
        return self.strategies[name](train), name, match_type, key


# ============ 主程序 ============
if __name__ == '__main__':
    data_file = r'D:\Desktop\数据2026_correct.xlsx'
    data = load(data_file)
    print('='*80)
    print('规律库预测系统 v4.0 - 位置特征+优化权重版')
    print('='*80)
    print()
    print('加载数据: {}期'.format(len(data)))

    # 构建规律库
    library = PatternLibrary(strategy_funcs, recent_weight=3.0)
    library.build(data)
    print('规律库: {}个特征组合'.format(len(library.library)))
    print()

    # 回测
    print('='*80)
    print('回测结果（第61-88期）')
    print('='*80)
    print('{:<8} | {:^12} | {:^8} | {:^8} | {}'.format(
        '期数', '策略', '预测', '结果', '实际'))
    print('-'*80)

    hits = 0
    total = 0
    for i in range(60, 88):
        train = data[i-10:i]
        actual = set(data[i]['zs'])

        # 使用位置0的规律库预测
        pred, strat, match, key = library.get_prediction(train, pos=0)

        hit = pred in actual
        if hit: hits += 1
        total += 1

        print('{:<8} | {:^12} | {}{:<6} | {}'.format(
            '第{}期'.format(i+1), strat, pred,
            '✓' if hit else '✗', ','.join(sorted(actual)[:4])))

    print()
    print('命中率: {}/{} = {:.0f}%'.format(hits, total, hits/total*100))

    # 近10期
    print()
    print('='*80)
    print('近10期回测（第79-88期）')
    print('='*80)

    recent_hits = 0
    for i in range(78, 88):
        train = data[i-10:i]
        actual = set(data[i]['zs'])
        pred, strat, match, key = library.get_prediction(train, pos=0)
        if pred in actual: recent_hits += 1

    print('近10期命中率: {}/10 = {:.0f}%'.format(recent_hits, recent_hits/10*100))

    # 预测第89期
    print()
    print('='*80)
    print('第89期预测')
    print('='*80)
    train = data[-10:]
    pred, strat, match, key = library.get_prediction(train, pos=0)

    print('选定策略: {} ({})'.format(strat, strategy_descs[strat]))
    print('匹配类型: {}'.format(match))
    print()
    print('预测生肖: 【{}】'.format(pred))
