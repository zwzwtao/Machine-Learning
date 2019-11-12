def load_dataset():
    # dataset contains all the transactions
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def create_c1(dataset):
    c1 = []
    for transaction in dataset:
        for item in transaction:
            if not [item] in c1:
                c1.append([item])
    c1.sort()

    return list(map(frozenset, c1))


def scan_d(dataset, itemset, min_support):
    # item is actually combinations of items in n-itemset
    # item_count is like {frozenset({2, 3, 5}): 2, frozenset({2, 5, 6}): 5}
    # in the example, means the number of appearances of 3-itemset frozenset({2, 3, 5})
    # is 2 and for frozenset({2, 5, 6}) is 5
    # returned_list is like: [frozenset({2, 5, 6}), frozenset({3, 5, 6})]
    # supports is like: {frozenset({1, 3}): 0.5, frozenset({2, 5}): 0.75}
    item_count = {}
    for transaction in dataset:
        # item is actually combinations of items in n-itemset
        for item in itemset:
            # e.g. {1, 3, 5} is subset of a transaction like {1, 2, 3, 5, 7}
            if item.issubset(transaction):
                if item not in item_count:
                    item_count[item] = 1
                else:
                    item_count[item] += 1
    num_transactinos = float(len(list(dataset)))
    returned_list = []
    # stores all the supports for itemsets with a support more than min_support
    supports = {}
    for key in item_count:
        # compute support
        support = item_count[key] / num_transactinos
        if support >= min_support:
            returned_list.insert(0, key)
        supports[key] = support

    return returned_list, supports


def apriori_gen(itemset, k):
    """
    k: starts from 2
    :return: combinations of items which construct a k-itemset
             e.g. [frozenset({2, 3, 5}), frozenset({6, 8, 10})]
    """
    ret_list = []
    len_itemset = len(itemset)
    for i in range(len_itemset):
        for j in range(i + 1, len_itemset):
            l1 = list(itemset[i])[: k - 2]
            l2 = list(itemset[j])[: k - 2]
            # if k = 2, then is a '[]  == []', still works
            if l1 == l2:
                ret_list.append(itemset[i] | itemset[j])

    return ret_list


def apriori(dataset, min_support=0.5):
    """
    :return: itemset_list(also called frequent itemsets):
             a list contains all the k-itemsets whose support is bigger than min_support
             e.g. [[frozenset({2, 3, 5}), frozenset({6, 8, 10})], [frozenset({2, 3, 5, 7}), frozenset({6, 8, 10, 11})]]
             supports: the supports for all the k-itemsets
    """
    # 1-itemset
    c1 = create_c1(dataset)
    l1, supports = scan_d(dataset, c1, min_support)
    itemset_list = [l1]
    # starts from 2-itemset
    k = 2
    # if I want to make a k-itemset, I have to make sure there are enough
    # itemsets in (k-1)-itemset(k-2 because index starts from 0)
    while (len(itemset_list[k - 2]) > 0):
        k_itemset = apriori_gen(itemset_list[k - 2], k)
        l_k, supports_k = scan_d(dataset, k_itemset, min_support)
        supports.update(supports_k)
        itemset_list.append(l_k)
        k += 1

    return itemset_list, supports


def generate_rules(frequent_itemsets, supports, min_confidence=0.6):
    rule_list = []
    for i in range(1, len(frequent_itemsets)):
        # frequent_itemset[i] means (i+1)-itemset
        for freq_set in frequent_itemsets[i]:
            # e.g. frozenset({2, 3, 5}) -> [frozenset({2}), frozenset({3}), frozenset({5})]
            h1 = [frozenset([]) for item in freq_set]
            rules_from_conseq(freq_set, h1, supports, rule_list, min_confidence)


def rules_from_conseq(freq_set, h, supports, rule_list, min_confidence=0.6):
    """
    :param h: a frozenset list contains all the single items from a k-itemset
              e.g. [frozenset({2}), frozenset({3}), frozenset({5})]
    :return:
    """
    m = len(h[0])
    while len(freq_set) > m:  # if there are at least 2 items in the itemset
        h = cal_conf(freq_set, h, supports, rule_list, min_confidence=0.6)
        if len(h) > 1:  # if
            apriori_gen(h, m + 1)
            m += 1
        else:
            break


def cal_conf(freq_set, h, supports, rule_list, min_confidence=0.6):
    pruned_h = []
    # p(Y|X) = p(XY) / p(X), here conseq is Y
    for conseq in h:
        conf = supports[freq_set] / supports[freq_set - conseq]
        if conf > min_confidence:
            print(freq_set - conseq, '-->', conseq, 'confidence', conf)
            rule_list.append((freq_set - conseq, conseq, conf))
            pruned_h.append(conseq)

    return pruned_h


if __name__ == '__main__':
    dataset = load_dataset()
    frequent_itemsets, supports = apriori(dataset)
    i = 0
    for freq_itemset in frequent_itemsets:
        print('item size: ', i + 1, ' ', freq_itemset)
        i += 1

    generate_rules(frequent_itemsets, supports, min_confidence=0.5)
