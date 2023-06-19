def calculate_makespan(times, assign):
    machines = [0] * len(assign)
    for i, time in enumerate(times):
        machine = assign[i]
        machines[machine] += time
    return max(machines)

def min_makespan(times, sigma, lamda):
    for i in range(sigma ** n):
        assign = [(i // (sigma ** j)) % sigma for j in range(n)]
        makespan = calculate_makespan(times, assign)
        if makespan < lamda:
            return True
    return False

sigma = int(input())
lamda = int(input())
n = int(input())
times = []
for i in range(0,n):
    time = int(input())
    times.append(time)
res = min_makespan(times, sigma, lamda)
if res == True:
    print('The minimum makespan is less than λ')
else:
    print('The minimum makespan is equal to or more than λ')
