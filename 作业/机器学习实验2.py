from itertools import chain, combinations
from collections import defaultdict
from tqdm import tqdm
import pandas as pd


class Apriori:

    def subsets(self, arr):
        return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])

    def returnItemsWithMinSupport(self, itemSet, transactionList, minSupport, freqSet, count=100000):
        _itemSet = set()
        localSet = defaultdict(int)

        n = 0
        for item in tqdm(itemSet):
            for transaction in transactionList:
                if item.issubset(transaction):
                    freqSet[item] += 1
                    localSet[item] += 1
            n += 1
            if n >= count:
                print('\n total length is  {}, break down at {} '.format(len(itemSet), count))
                break

        for item, count in list(localSet.items()):
            support = float(count) / len(transactionList)
            if support >= minSupport:
                _itemSet.add(item)
        return _itemSet

    def joinSet(self, itemSet, length):
        print('\nstart generator itemSet ,from length {} to n-grams {} '.format(len(itemSet), length))
        if length > 3:
            print('\nmaybe joinSet process will take long time...')
            return set([i.union(j) for i in itemSet for j in tqdm(itemSet) if len(i.union(j)) == length])
        else:
            return set([i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length])

    def getItemSetTransactionList(self, data_iterator):
        transactionList = list()
        itemSet = set()
        print('\n Generate 1-itemSets ... ')
        for record in data_iterator:
            transaction = frozenset(record)
            transactionList.append(transaction)
            for item in transaction:
                itemSet.add(frozenset([item]))  # Generate 1-itemSets
        return itemSet, transactionList

    def runApriori(self, data_iter, minSupport, minConfidence, minLift=0, tuples=2, count=100000):
        itemSet, transactionList = self.getItemSetTransactionList(data_iter)
        freqSet = defaultdict(int)
        largeSet = dict()
        assocRules = dict()
        oneCSet = self.returnItemsWithMinSupport(itemSet,
                                                 transactionList,
                                                 minSupport,
                                                 freqSet,
                                                 count=count)

        currentLSet = oneCSet
        k = 2
        largeSet[k - 1] = currentLSet

        while (currentLSet != set([])) and k <= tuples:
            currentLSet = self.joinSet(currentLSet, k)
            print('\n the tuple n-grams is {} , length is {} , time cost maybe : {} min... '.format(k, len(currentLSet),
                                                                                                    len(currentLSet) * 0.0015 / 60))
            currentCSet = self.returnItemsWithMinSupport(currentLSet,
                                                         transactionList,
                                                         minSupport,
                                                         freqSet,
                                                         count=count)
            currentLSet = currentCSet
            largeSet[k] = currentLSet
            k = k + 1

        def getSupport(item):
            return float(freqSet[item]) / len(transactionList)

        print('\nCalculation the tuple words and support ... ')
        toRetItems = []
        for key, value in list(largeSet.items()):
            toRetItems.extend([(tuple(item), getSupport(item))
                               for item in value])

        toRetRules = []
        print('\nCalculation the pretuple words and confidence ... ')
        for key, value in list(largeSet.items())[1:]:
            for item in value:
                if len(item) <= tuples:
                    _subsets = map(frozenset, [x for x in self.subsets(item)])
                    for element in _subsets:
                        remain = item.difference(element)
                        if len(remain) > 0:
                            confidence = getSupport(item) / getSupport(element)
                            lift = confidence / getSupport(remain)
                            self_support = getSupport(item)
                            if self_support >= minSupport:
                                if confidence >= minConfidence:
                                    if confidence >= minLift:
                                        toRetRules.append(((tuple(element), tuple(remain)), tuple(item),
                                                           self_support, confidence, lift))
        return toRetItems, toRetRules

    def printResults(self, items, rules):
        for item, support in sorted(items):
            print("item: %s , %.3f" % (str(item), support))
        print("\n------------------------ RULES:")
        for rule, remain, support, confidence, lift in sorted(rules):
            pre, post = rule
            print("Rule: %s ==> %s , %.3f" % (str(pre), str(post), confidence))

    def dataFromFile(self, fname, extra=False):
        if not extra:
            file_iter = open(fname, encoding='utf-8')
            for line in file_iter:
                line = line.strip().rstrip(',')
                record = frozenset(line.split(','))
                yield record
        else:
            for n in range(len(fname)):
                record = frozenset(fname.ix[n, :])
                yield record

    def dataFromList(self, data_list):
        for dl in data_list:
            yield frozenset(dl)

    def transferDataFrame(self, items, rules, removal=True):
        items_data = pd.DataFrame(items)
        items_data.columns = ['word', 'support']
        items_data['len'] = list(map(len, items_data.word))

        rules_data = pd.DataFrame(rules)
        rules_data.columns = ['word', 'item', 'support', 'confidence', 'lift']
        rules_data['word_x'] = list(map(lambda x: x[0][0] if len(x[0]) == 1 else x[0], rules_data.word))
        rules_data['word_y'] = list(map(lambda x: x[1][0] if len(x[1]) == 1 else x[1], rules_data.word))
        rules_data['item_len'] = list(map(len, rules_data['item']))

        if removal:
            rules_data['word_xy'] = list(map(lambda x: ''.join(list({x[0][0], x[1][0]})), rules_data.word))
            rules_data = rules_data.drop_duplicates(['word_xy'])

        return items_data, rules_data[['word_x', 'word_y', 'item', 'item_len', 'support', 'confidence', 'lift']]


if __name__ == "__main__":
    apr = Apriori()

    inFile = apr.dataFromFile('INTEGRATED-DATASET.csv', extra=False)
    minSupport = 0.15
    minConfidence = 0.3
    items, rules = apr.runApriori(inFile, minSupport, minConfidence, tuples=2)

    apr.printResults(items, rules)

    items_data, rules_data = apr.transferDataFrame(items, rules)
