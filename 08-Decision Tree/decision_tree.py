from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from math import log
import operator


def create_dataset():
    dataset = [[0, 0, 0, 0, 'no'],
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    labels = ['F1-AGE', 'F2-WORK', 'F3-HOME', 'F4-LOAN']
    return dataset, labels


def create_tree(dataset, labels, feature_labels):
    """
    dataset: a dataset the algorithm do classification on
    labels: in this case, labels are 'yes's and 'no's
    feature_labels: store the feature label in order according to node we assign
    """
    # obtain different classes
    class_list = [sample[-1] for sample in dataset]

    # check if all the labels are the same, if same, return label
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]

    # check if all the features have been used, if so,
    # stop and return a class
    if len(dataset[0]) == 1:
        return majority_cnt(class_list)

    best_feature_idx = choose_best_feature_to_split(dataset)
    best_feature_label = labels[best_feature_idx]

    feature_labels.append(best_feature_label)
    my_tree = {best_feature_label: {}}
    del labels[best_feature_idx]

    feature_values = [sample[best_feature_idx] for sample in dataset]
    # only select the unique value under the feature, in this example,
    # could be 0,1,2...
    unique_values = set(feature_values)
    for value in unique_values:
        # just want to clarify the labels are different
        # bust since the used labels are already deleted
        # here we use labels[:]
        sub_labels = labels[:]
        my_tree[best_feature_label][value] = \
            create_tree(split_dataset(dataset, best_feature_idx, value), sub_labels, feature_labels)

    return my_tree


def majority_cnt(class_list):
    """
    find the labels that account for majority and return a class label
    (in this case is 'yes' or 'no')
    """
    class_count = {}
    for vote in class_list:
        if vote not in class_list.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    # in this example, we only have 2 different classes,
    # but may be more in other problems
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)

    # sorted_class_count will look like: [('yes', 3), ('no', 2)]
    return sorted_class_count[0][0]


def choose_best_feature_to_split(dataset):
    """
    choose the best feature that gives the max information gain
    """
    # the number of remaining features
    num_features = len(dataset[0]) - 1

    # compute entropy
    base_entropy = calc_base_entropy(dataset)

    best_info_gain = 0
    best_feature = -1
    for i in range(num_features):
        feature_list = [sample[i] for sample in dataset]
        unique_values = set(feature_list)
        new_entropy = 0
        for val in unique_values:
            sub_dataset = split_dataset(dataset, i, val)
            prob = len(sub_dataset) / float(len(dataset))
            new_entropy += prob * calc_base_entropy(sub_dataset)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i

    return best_feature


def split_dataset(dataset, axis, val):
    """
    split the whole dataset to a sub dataset with one feature selected
    """
    sub_dataset = []
    for feature_vector in dataset:
        if feature_vector[axis] == val:
            # delete the feature column of that feature
            reduced_feature_vector = feature_vector[:axis]
            reduced_feature_vector.extend(feature_vector[axis + 1:])
            sub_dataset.append(reduced_feature_vector)

    return sub_dataset


def calc_base_entropy(dataset):
    """
    calculate the entropy of the label and return it
    """
    num_sample = len(dataset)
    label_count = {}
    for feature_vector in dataset:
        current_label = feature_vector[-1]
        if current_label not in label_count.keys():
            label_count[current_label] = 0
        label_count[current_label] += 1

    entropy = 0
    for key in label_count:
        prop = float(label_count[key]) / num_sample
        entropy -= prop * log(prop, 2)

    return entropy


def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    arrow_args = dict(arrowstyle="<-")
    font = FontProperties(fname=r"c:\windows\fonts\simsunb.ttf", size=14)
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args, FontProperties=font)


def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


def plotTree(myTree, parentPt, nodeTxt):
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")
    leafNode = dict(boxstyle="round4", fc="0.8")
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = next(iter(myTree))
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')                       # createfig
    fig.clf()  													 # empty fig
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)  # get rid of x, y axis
    plotTree.totalW = float(getNumLeafs(inTree))       			 # get number of decision tree leaf nodes
    plotTree.totalD = float(getTreeDepth(inTree))  	 			 # get tree depth
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()


if __name__ == '__main__':
    dataset, labels = create_dataset()
    feature_labels = []
    myTree = create_tree(dataset, labels, feature_labels)
    createPlot(myTree)
