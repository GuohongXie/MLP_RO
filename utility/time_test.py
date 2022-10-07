import time
start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print(start_time)
a = 1
for i in range(1000000):
    a *= i 
end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print(end_time)
