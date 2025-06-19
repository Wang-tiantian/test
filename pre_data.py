import json
import pandas as pd

with open('./data/data_for_llms/gemini_f2exp_1-3000_structured.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

rows = []
for idx, content in data.items():
    print(f"index:{idx}")
    if int(idx)<50:
    # if int(idx)>50 and int(idx) < 1000:
        articleInformation = content.get("articleInformation", {})
        # print(articleInformation)
        mainReactions = content.get("mainReactions", {})
        reactionDetails = content.get("reactionDetails", [])
        # print(mainReactions["reactionList"][0]["reactions"][0])
        num_reactions = mainReactions["reactionList"][0]["numReactions"]
        print(f"num_reaction = {num_reactions}")
        ac_num = len(reactionDetails)
        print(f"ac_num = {ac_num}")
        if ac_num != num_reactions:
            print(f'{idx}pass')
            continue
        else:
            for _ in range(num_reactions):
                mainReaction = mainReactions["reactionList"][0]["reactions"][_]
                del mainReaction['id']
                reactionDetail = reactionDetails[_]
                del reactionDetail['ID']
                type_reactions = mainReactions["reactionList"][0]["reactions"][_]['reactionType']
                print(type_reactions)
                if type_reactions == 'thermal catalysis':
                    dic = {"articleInformation": articleInformation,
                    "mainReactions": mainReaction,
                    "reactionDetails": reactionDetail}
                    rows.append(dic)
                else:
                    print("type no right,pass")
                print(f'num_reactions{_}done')

df = pd.DataFrame(rows)
print(df.info())
df.to_csv('./data/data_for_llms/thermal_test.csv')