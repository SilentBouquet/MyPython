import jieba.posseg as pseg

stop_words = ["的", "了", "和", "在", "但是", "其", "那么", "不仅", "而且"]

rocket_senses = {"weapon": "火箭（武器）", "spacecraft": "火箭（航天）", "marvel_hero": "火箭（超级英雄）"}


def disambiguate_rocket(sentence):
    words = pseg.lcut(sentence)
    related_words = [word for word, flag in words if word not in stop_words and flag.startswith('n')]
    if any(word in related_words for word in ["复仇者联盟", "超级英雄", "格鲁特", "灭霸"]):
        return rocket_senses["marvel_hero"]
    if any(word in related_words for word in ["航天", "卫星", "运载", "长征"]):
        return rocket_senses["spacecraft"]
    if any(word in related_words for word in ["NBA", "篮球", "老鹰", "休斯敦"]):
        return "休斯敦火箭（篮球队）"
    return rocket_senses["weapon"]


test_sentences = [
    "灭霸降临后，火箭和其他的复联成员联手阻止灭霸未果。灭霸打响指后，火箭目睹格鲁特的消失悲痛万分。",
    "在复仇者联盟决定进行量子领域的时空逆转计划后，火箭和浩克一起说服雷神托尔回归。",
    "北京时间12月1日，NBA常规赛继续进行，休斯敦火箭主场迎战亚特兰大老鹰。",
    "从1992年开始研制的长征二号F型火箭，是中国航天史上技术最复杂、可靠性和安全性指标最高的运载火箭。"
]

for sentence in test_sentences:
    print(f"句子: {sentence}\n消歧后的含义: {disambiguate_rocket(sentence)}\n")