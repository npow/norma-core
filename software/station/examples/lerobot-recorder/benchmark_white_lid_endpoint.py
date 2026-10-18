import asyncio,json,time,statistics
from pathlib import Path
import numpy as np
from rollout_act import request_json,payload_for
from bootstrap import setup
setup()
from station_io import Station,vector
async def main():
 url='http://127.0.0.1:8765'
 m=request_json(url+'/metadata'); c=m['collection']; s=await Station.connect('localhost')
 try:
  await s.discover(); s.configure_follower(c['follower_bus'],c['motor_ids'])
  rows=[]
  for i in range(30):
   _,motors,images,_=s.snapshot(None,c['follower_bus'],c['cameras'],.2)
   start=time.perf_counter(); payload=await asyncio.to_thread(payload_for,images,vector([v.present_position for v in motors],motors))
   r=await asyncio.to_thread(request_json,url+'/predict',payload)
   rows.append({'rtt_ms':1000*(time.perf_counter()-start),**r['timings_ms']})
  report={'checkpoint':m['checkpoint'],'backend':m['backend'],'requests':len(rows),'motor_writes':0,'summary':{k:{'median':float(np.median([r[k] for r in rows])),'p95':float(np.percentile([r[k] for r in rows],95))} for k in rows[0]},'rows':rows}
  path=Path.home()/'benchmarks/white-lid-smolvla-20261017/optimized-endpoint-benchmark.json';path.write_text(json.dumps(report,indent=2));print(json.dumps(report['summary'],indent=2))
 finally:s.client.close_connection()
asyncio.run(main())
