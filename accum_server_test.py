import rpyc

conn = rpyc.connect("localhost", port=18863)

conn.root.collect_accum_data()