import jieba.posseg as pseg
from collections import Counter

# 定义停用词
stop_words = ["的", "了", "和", "在", "但是", "其", "那么", "不仅", "而且"]

# 给定的中文文本
text = ("人民生活不断改善，深入贯彻以人民为中心的发展思想，一大批惠民举措落地实施。"
        "人民获得感显著增强。脱贫攻坚战取得决定性进展，六千多万贫困人口稳定脱贫，"
        "贫困发生率从百分之十一点二下降到百分之二点三以下。教育事业全面发展，"
        "意识形态教育明显加强。就业状况持续改善，城镇新增就业年均一千三百万人以上。"
        "城乡居民收入增速超过经济增速，中等收入群体持续扩大，覆盖城乡居民的社会保障体系基本建立，"
        "人民健康和医疗卫生水平大幅提高，保障性住房建设稳步推进。社会治理体系更加完善，社会大局保持稳定，国家安全全面加强。")

# 使用jieba的pseg.lcut进行分词和词性标注
words = pseg.lcut(text)

# 移除停用词并提取名词
nouns = [word for word, flag in words if word not in stop_words and flag.startswith('n')]

# 统计词频
counter = Counter(nouns)
top_nouns = counter.most_common(5)

# 打印结果
for word, freq in top_nouns:
    print(f"{word}: {freq}")