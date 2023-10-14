# List of Keys
keyList = ["Paras", "Jain", "Cyware"]
valueList = ["Paras", "Jain", "Cyware"] 
# Using Dictionary comprehension
myDict = {keyList[i]: valueList[i] for i in range(len(keyList))}
print(myDict)   