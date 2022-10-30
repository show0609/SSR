import multiprocessing
import sys
import copy
sys.path.append("..")
from AlgorithmBase import AlgorithmResult
from SSR import SSR
from GreedyHopRouting import GreedyHopRouting
from QCAST import QCAST
from REPS import REPS
from topo.Topo import Topo
from topo.Node import Node
from topo.Link import Link
from random import sample
import csv
import random

def runThread(algo, requests, algoIndex, ttime, pid, resultDict):
    for i in range(ttime):
        result = algo.work(requests[i], i)
    resultDict[pid] = result

def Run(numOfRequestPerRound = 3, r = 60, q = 0.8, alpha = 0.002, alpha_sat = 0.0001, p_sat = 0.8, SocialNetworkDensity = 0.5, satNum = 5, rtime = 10, topo = None):

    times = 20   # 測試次數
    ttime = 120  # timeslot 數
    
    topo.setSatNum(satNum)
    topo.setQ(q)
    topo.setAlpha(alpha)
    topo.setAlpha_sat(alpha_sat)
    topo.setP_sat(p_sat)
    topo.setR(r)
    topo.setDensity(SocialNetworkDensity)

    # make copy
    algorithms = []
    algorithms.append(SSR(copy.deepcopy(topo)))
    algorithms.append(GreedyHopRouting(copy.deepcopy(topo)))
    algorithms.append(QCAST(copy.deepcopy(topo)))
    algorithms.append(REPS(copy.deepcopy(topo)))
    
    if ttime > topo.satCycleNum:
        print("timeslot bigger than satCycleNum")
        exit()
    
    results = [[] for _ in range(len(algorithms))]
    resultDicts = [multiprocessing.Manager().dict() for _ in algorithms] # list 裡面塞 dict (key=pid, value=result)
    jobs = []

    pid = 0
    for _ in range(times):
        
        # 產生request
        random.seed()
        allRequest = []
        for _ in range(300):
            while True:
                a = sample([i for i in range(len(topo.nodes))], 2)  # 0 ~ numOfNode-1 選2個數字
                if topo.nodes[a[0]] in topo.socialRelationship[topo.nodes[a[1]]]:
                    break
            allRequest.append((a[0], a[1]))

        for algoIndex in range(len(algorithms)):
            algo = copy.deepcopy(algorithms[algoIndex])
            requests = {i : [] for i in range(ttime)} # 在該 timeslot 有的 request
            idx = 0
            for i in range(rtime):
                for _ in range(numOfRequestPerRound):
                    src, dst = allRequest[idx]
                    requests[i].append((algo.topo.nodes[src], algo.topo.nodes[dst])) # 轉成 node
                    idx += 1

            for i in range(60,60+rtime):
                for _ in range(numOfRequestPerRound):
                    src, dst = allRequest[idx]
                    requests[i].append((algo.topo.nodes[src], algo.topo.nodes[dst])) # 轉成 node
                    idx += 1
            
            pid += 1
            job = multiprocessing.Process(target = runThread, args = (algo, requests, algoIndex, ttime, pid, resultDicts[algoIndex]))
            jobs.append(job)

    for job in jobs:
        job.start()

    for job in jobs:
        job.join()

    for algoIndex in range(len(algorithms)):
        results[algoIndex] = AlgorithmResult.Avg(resultDicts[algoIndex].values(),ttime) # 算平均, dict.values()為 list

    return results

if __name__ == '__main__':
    print("start Run and Generate data.txt")
    targetFilePath = "../../plot/data_csv/" # 檔案資料夾
    temp = AlgorithmResult()
    Ylabels = temp.Ylabels 
    
    numOfRequestPerRound = [1, 2, 3, 4, 5]
    totalRequest = [10, 20, 30, 40, 50]
    r = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    q = [0.2, 0.4, 0.6, 0.8, 1]
    alpha = [0.000, 0.002, 0.004, 0.006, 0.008, 0.01]
    alpha_sat = [0.0000, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009]
    SocialNetworkDensity = [0.25, 0.5, 0.75, 1]
    satNum = [5, 6, 7, 8, 9, 10]
    p_sat = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    testXlabel = [0] # 要測試的參數
    Xlabels = ["RequestPerRound", "totalRequest", "r", "swapProbability", "alpha", "alpha_sat", "SocialNetworkDensity", "satNum", "p_sat"]
    Xparameters = [numOfRequestPerRound, totalRequest, r, q, alpha, alpha_sat, SocialNetworkDensity, satNum, p_sat]

    topo = Topo.generate(0.9, 0.001, 0.0001, 0.7, 0.75) # (self, q, alpha, alpha_sat, p_sat, density)    
               
    for XlabelIndex in range(len(Xlabels)):
        Xlabel = Xlabels[XlabelIndex]
        Ydata = []
        if XlabelIndex not in testXlabel:
            continue
        for Xparam in Xparameters[XlabelIndex]:
            
            # check schedule
            statusFile = open("./output/status.txt", "w")
            print("run " + Xlabel + ": " + str(Xparam), file = statusFile)
            statusFile.flush()
            statusFile.close()
            # ------
            if XlabelIndex == 0: # #RequestPerRound
                result = Run(numOfRequestPerRound = Xparam, topo = copy.deepcopy(topo))
            if XlabelIndex == 1: # totalRequest
                result = Run(numOfRequestPerRound = Xparam, rtime = 1, topo = copy.deepcopy(topo))
            if XlabelIndex == 2: # r
                result = Run(r = Xparam, topo = copy.deepcopy(topo))
            if XlabelIndex == 3: # swapProbability
                result = Run(q = Xparam, topo = copy.deepcopy(topo))
            if XlabelIndex == 4: # alpha
                result = Run(alpha = Xparam, topo = copy.deepcopy(topo))
            if XlabelIndex == 5: # alpha_sat
                result = Run(alpha_sat = Xparam, topo = copy.deepcopy(topo))
            if XlabelIndex == 6: # SocialNetworkDensity
                result = Run(SocialNetworkDensity = Xparam, topo = copy.deepcopy(topo))
            if XlabelIndex == 7: # satellite number
                result = Run(satNum = Xparam, topo = copy.deepcopy(topo))
            if XlabelIndex == 8: # satellite probability p
                result = Run(p_sat = Xparam, topo = copy.deepcopy(topo))
            Ydata.append(result)


        for Ylabel in Ylabels: # 結果寫入檔案
            filename = Xlabel + "_" + Ylabel + ".csv"
            F = open(targetFilePath + filename, "w")
            writer = csv.writer(F) # create the csv writer
           
            row = []
            algoName = ["SSR", "GREEDY", "QCAST", "REPS"]
            row.append(Xlabel + " \\ " + Ylabel)
            row.extend(algoName)  
            writer.writerow(row) # write a row to the csv file
                
            for i in range(len(Xparameters[XlabelIndex])):
                row = []
                row.append(Xparameters[XlabelIndex][i])
                row.extend([algoResult.toDict()[Ylabel] for algoResult in Ydata[i]])
                writer.writerow(row)

            F.close()
    
