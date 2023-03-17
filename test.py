import rpyc

conn = rpyc.connect("localhost", port=18864)
conn.root.some2()