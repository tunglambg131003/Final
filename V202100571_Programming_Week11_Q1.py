m,n= map(int, input("").split())
seats = []
for i in range(m):
    seats.append(list(map(int, input("").split())))

match = [[-1] * n for _ in range(m)]
visit = [[-1] * n for _ in range(m)]

def dfs(i, j, v):
    for x, y in ((i-1,j-1),(i,j-1),(i+1,j-1),(i-1,j+1),(i,j+1),(i+1,j+1)):
        if 0 <= x < m and 0 <= y < n and seats[x][y] == 1 and visit[x][y] != v:
            visit[x][y] = v
            x_1, y_1 = divmod(match[x][y], n)
            if match[x][y] == -1 or dfs(x_1, y_1, v) == 1:
                match[x][y] = i*n+j
                match[i][j] = x*n+y
                return 1
    return 0

t = 0
v = 0
for i in range(m):
    for j in range(n):
        if seats[i][j] == 1 and match[i][j] == -1:
            visit[i][j] = v
            t += dfs(i, j, v)
            v += 1

print(sum(seats[i][j] == 1 for i in range(m) for j in range(n)) - t)

