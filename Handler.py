import json
#import clean as clean

#def comparison ():
#    clean.D435_bag(x,y,x,y,x,y,x,y)

with open('log.json') as json_file:
    data = json.load(json_file)

new_string = json.dumps(data, indent=2)
print(new_string)