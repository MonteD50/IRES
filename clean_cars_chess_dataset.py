import re
import pandas as pd

res = []
with open('Chess.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        try:
            # Line looks like: 2997. qxmsq=f skrxp=f 2924 ==> stlmt=f 2877    <conf:(0.98)> lift:(1) lev:(-0) [-4] conv:(0.9)\n
            # Extract left rule items
            left_rule = line.split(' ==> ')[0]
            ## Remove integers
            left_rule = re.sub(r'\d+', '', left_rule)
            ## Remove .
            left_rule = left_rule.replace('.', '')
            ## Remove spaces in beginning and end
            left_rule = left_rule.strip()
            ## Make proper left rule
            left_rule = "{" + ",".join(left_rule.split(" ")) + "}"

            ## Add ' to each item
            left_rule = left_rule.replace(",", "','").replace("{", "{'").replace("}", "'}")
            
            # Extract right rule items
            right_rule = line.split(' ==> ')[1].split(" ")[0]
            right_rule = "{'" + right_rule + "'}" 
            
            # Extract confidence
            confidence = line.split(' <conf:(')[1].split(')>')[0]
            
            res.append({"left": left_rule, "right": right_rule, "confidence": confidence, "support": 0})
        except Exception as e:
            print("Error", str(e))

df = pd.DataFrame(res)
df.to_csv("chess_cars.csv", index=False)

        