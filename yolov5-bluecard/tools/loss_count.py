f = open('../loss','r')
lines = f.read()
for line in lines:
    ind = line.index('total:')
    total = line[ind:ind+7]
    print(total)