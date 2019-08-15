def loadData(filename):

    data = []
    label = []
    with open(filename) as f:
        for line in f.readlines():
            lines = line.strip().split(',')

            data.append([int(num) for num in lines[1:]])
            label.append(int(lines[0]))

    return data, label

