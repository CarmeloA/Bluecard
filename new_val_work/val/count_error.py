f = open('../exp3_result_count.txt','r')
lines = f.readlines()

for ind,line in enumerate(lines):
    if ind % 2 == 0:
        i = line.index('[')
        j = line.index(']')
        target = line[i+1:j]
        l = target.split(' ')
        k = target.index(':')
        print(k)

