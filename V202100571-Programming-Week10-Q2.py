n, m = map(int, input("").split())
daily = list(map(int, input("").split()))
for i in range(n):
    daily[i] *= (n-i)
daily = sorted(daily)
def max_cattles(arr, m):
     result = 0
     for i in range(0,len(arr)-1):
        arr[i+1] = arr[i] + arr[i+1]
     for i in range(0, len(arr)):
         if arr[i] < m:
             result += 1
     return result
print(max_cattles(daily, m))




