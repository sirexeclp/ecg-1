import subprocess
import pandas as pd
proc = subprocess.run(["sudo", "ipmitool", "sensor"], stdout=subprocess.PIPE)
#print(proc.stdout.decode("utf-8"))
sensor_output = proc.stdout.decode("utf-8")
df = pd.DataFrame([x.split('|') for x in sensor_output.split('\n')])
print(df)

