cpu = list(map(int, input("").split()))
k = int(input(""))
result = []
def min_forks(arr):
  if len(arr) > 0:
     sum = 0
     select = []
     remain = []
     for i in arr:
        if i + sum <= k:
            select.append(i)
            sum += i
        else:
            remain.append(i)
     result.append(select)
     min_forks(remain)
  else:
      print(len(result))

cpu = sorted(cpu, reverse=True)
min_forks(cpu)

