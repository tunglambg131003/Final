m = int(input())
t = list(map(int, input().split()))
def load_balancing_approximation(t, m):

    sorted_t = sorted(t, reverse=True)
    machines = [0] * m

    for i in sorted_t:
        min_load = min(machines)
        min_index = machines.index(min_load)
        machines[min_index] += i

    print(max(machines))

load_balancing_approximation(t, m)




