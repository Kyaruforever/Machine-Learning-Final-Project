from evalWithAbc import calAigeval

def greedy(newstate):
    states=[newstate]
    vals=[calAigeval(newstate)]
    for step in range(10):
        max_index=-1
        max_val=-99999
        childs=[]
        state=states[-1]
        for child in range(7):
            childFile=state+str(child)
            childs.append(childFile)
            val=calAigeval(childFile)
            if val > max_val:
                max_val=val
                max_index=child
        states.append(childs[max_index])
        vals.append(max_val)
    
    return states[vals.index(max(vals))]

path='adder_'
print('Best Action:'+str(greedy(path)))