import sys
from pathlib import Path
import csv
import ruamel.yaml

result = [['observation', 'rotation', 'translation']]
flatres = ["observation,tran1,tran2,tran3,rot1,rot2,rot3,rot4,rot5,rot6,rot7,rot8,rot9".split(',')]
yaml = ruamel.yaml.YAML()

for idx, file_name in enumerate(Path('.').glob('*.yml')):
   txt = file_name.read_text()
   if txt.startswith('%YAML:1.0'):
      txt = txt.replace('%YAML:1.0', "", 1).lstrip()
   data1 = yaml.load(txt)
   result.append([
     idx+1,
     data1['object_rotation_wrt_camera']['data'],
     data1['object_translation_wrt_camera'],
   ])
   row = [idx+1]
   row.extend(data1['object_translation_wrt_camera'])
   row.extend(data1['object_rotation_wrt_camera']['data'])
   flatres.append(row)

writer = csv.writer(sys.stdout)
writer.writerows(result)
print('---------')
writer = csv.writer(sys.stdout)
writer.writerows(flatres)