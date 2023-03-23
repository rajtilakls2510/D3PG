import rpyc
from time import perf_counter
conn = rpyc.connect("localhost", port=18863)
start = perf_counter()
conn.root.collect_accum_data()
end = perf_counter()
print(f"{end - start}")