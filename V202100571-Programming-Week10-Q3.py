def countSubstrings(seq):
    pal = [[None] * len(seq) for _ in range(len(seq))]
    res = 0
    for i in range(len(seq)):
        pal[i][i] = 1
        res += 1
    for i in range(len(seq) - 1, -1, -1):
        for j in range(len(seq) - 1, i, -1):
            if seq[i] == seq[j] and pal[i + 1][j - 1] != 0:
                pal[i][j] = 1
                res += 1
            else:
                pal[i][j] = 0
    return res
seq = str(input(""))
print(countSubstrings(seq))