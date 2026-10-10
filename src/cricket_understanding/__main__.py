from .schema import describe
import json, sys
print(json.dumps(describe(), indent=2))

