def Pascal(n):
    result = [1]
    if n == 1:
        return result
    prev = Pascal(n - 1)
    for i in range(1, len(prev)):
        curr = prev[i - 1] + prev[i]
        result.append(curr)
    result.append(1)
    return result
n = int(input(""))
res = Pascal(n)
for i in range(len(res)):
   if i == len(res) - 1:
      print(res[i])
   else:
      print(res[i], end=" ")


