import asyncio,json,time
from pathlib import Path
from rollout_act import request_json,payload_for
from bootstrap import setup
setup()
from station_io import Station,vector
async def main():
 url='http://127.0.0.1:8765';m=request_json(url+'/metadata');c=m['collection'];s=await Station.connect('localhost')
 try:
  await s.discover();s.configure_follower(c['follower_bus'],c['motor_ids'])
  _,motors,images,_=s.snapshot(None,c['follower_bus'],c['cameras'],.2)
  payload=payload_for(images,vector([x.present_position for x in motors],motors));payload['noise_seed']=20261017
  a=await asyncio.to_thread(request_json,url+'/predict',payload);b=await asyncio.to_thread(request_json,url+'/predict',payload)
  same=max(abs(x-y) for u,v in zip(a['actions'],b['actions']) for x,y in zip(u,v))
  payload['noise_seed']+=1;d=await asyncio.to_thread(request_json,url+'/predict',payload)
  diff=max(abs(x-y) for u,v in zip(a['actions'],d['actions']) for x,y in zip(u,v))
  report={'identical_input_and_seed_max_action_difference':same,'different_seed_max_action_difference':diff,'model_ms':b['timings_ms']['model'],'motor_writes':0};print(json.dumps(report,indent=2));(Path.home()/'benchmarks/white-lid-smolvla-20261017/seeded-endpoint-check.json').write_text(json.dumps(report,indent=2));assert same<1e-5 and diff>0
 finally:s.client.close_connection()
asyncio.run(main())
