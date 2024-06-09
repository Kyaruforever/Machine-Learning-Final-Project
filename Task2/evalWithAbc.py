import os
import pickle
import re

libFile='./lib/7nm/7nm.lib'

dic_file='./InitialAIG/baseData.pkl'

with open(dic_file,'rb') as f:
    baseline=pickle.load(f)

def calAigeval(state,circuitPath='./InitialAIG/train',logFile='./mytask1/log',nextState='./mytask1/aig'):
    circuitName,actions=state.split('_')
    circuitPath+='/'+circuitName+'.aig'
    logFile+='/'+state+'.log'
    nextState+='/'+state+'.aig'#currentAIGfile
    if os.path.exists(nextState): 
        return geteval(circuitName=circuitName,logFile=logFile)
    synthesisOpToPosDic={
    0:"refactor",
    1:"refactor -z",
    2:"rewrite",
    3:"rewrite -z",
    4:"resub",
    5:"resub -z",
    6:"balance"
    }
    action_cmd=''
    for action in actions:
        action_cmd+=(synthesisOpToPosDic[int(action)]+';')
    abcRunCmd="./yosys/yosys-abc -c \"read "+circuitPath+";"+action_cmd+";read_lib "+libFile+"; write "+nextState+ " ; map; topo ; stime \" > " +logFile
    os.system(abcRunCmd)
    return geteval(circuitName=circuitName,logFile=logFile)

def geteval(circuitName,logFile):
    with open( logFile ) as f :
        areaInformation = re . findall ( '[a-zA-Z0-9.]+' , f . readlines () [-1])
        eval = float ( areaInformation [-9]) * float ( areaInformation [-4])
        eval=eval/baseline[circuitName]
        return eval

