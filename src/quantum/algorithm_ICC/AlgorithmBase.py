from dataclasses import dataclass
from time import process_time, sleep
import sys
sys.path.append("..")
from topo.Topo import Topo  
import random

class AlgorithmResult:
    def __init__(self):
        self.algorithmRuntime = 0
        self.waitingTime = 0      # request 的平均等待時間(含還沒完成的request)
        self.idleTime = 0         # 找不到 path 的閒置時間
        self.usedQubits = 0       # request 的平均 qubit 使用數量
        self.temporaryRatio = 0
        self.successRequestNum = 0
        self.throughput = 0
        self.dropRatio = 0
        self.satSuccRatio = 0

        self.dropNum = 0
        self.totalReqNum = 0
        self.numOfTimeslot = 0    # 要處理 request 的 timeslot 數
        self.totalRuntime = 0     # 實際執行時間
        self.Ylabels = ["algorithmRuntime", "waitingTime", "idleTime", "usedQubits", "temporaryRatio", "successRequestNum", "throughput", "dropRatio", "satSuccRatio"]
        self.remainRequestPerRound = []

    def toDict(self):
        dic = {}
        dic[self.Ylabels[0]] = self.algorithmRuntime
        dic[self.Ylabels[1]] = self.waitingTime
        dic[self.Ylabels[2]] = self.idleTime
        dic[self.Ylabels[3]] = self.usedQubits
        dic[self.Ylabels[4]] = self.temporaryRatio
        dic[self.Ylabels[5]] = self.successRequestNum
        dic[self.Ylabels[6]] = self.throughput
        dic[self.Ylabels[7]] = self.dropRatio
        dic[self.Ylabels[8]] = self.satSuccRatio
        return dic
    
    def Avg(results: list, ttime):
        
        AvgResult = AlgorithmResult()

        AvgResult.remainRequestPerRound = [0 for _ in range(ttime)] 
        for result in results:  # 把 times 次的結果加總算平均
            AvgResult.algorithmRuntime += result.algorithmRuntime
            AvgResult.waitingTime += result.waitingTime
            AvgResult.idleTime += result.idleTime
            AvgResult.usedQubits += result.usedQubits
            AvgResult.temporaryRatio += result.temporaryRatio
            AvgResult.successRequestNum += result.successRequestNum
            AvgResult.throughput += result.successRequestNum / result.numOfTimeslot
            AvgResult.dropRatio += result.dropNum / result.totalReqNum
            AvgResult.satSuccRatio += result.satSuccRatio

            Len = len(result.remainRequestPerRound)
            if ttime != Len:
                print("the length of RRPR error:", Len, file = sys.stderr)
                exit(0)
            
            for i in range(ttime):
                AvgResult.remainRequestPerRound[i] += result.remainRequestPerRound[i]

        AvgResult.algorithmRuntime /= len(results)
        AvgResult.waitingTime /= len(results)
        AvgResult.idleTime /= len(results)
        AvgResult.usedQubits /= len(results)
        AvgResult.temporaryRatio /= len(results)
        AvgResult.successRequestNum /= len(results)
        AvgResult.throughput /= len(results)
        AvgResult.dropRatio /= len(results)
        AvgResult.satSuccRatio /= len(results)

        for i in range(ttime):
            AvgResult.remainRequestPerRound[i] /= len(results)
            
        return AvgResult

class AlgorithmBase:

    def __init__(self, topo):
        self.name = ""
        self.topo = topo
        self.srcDstPairs = []
        self.timeslot = 0
        self.satSuccNum = 0
        self.satLinkNum = 0
        self.result = AlgorithmResult()

    def prepare(self):
        pass
    
    def p2(self):
        pass

    def p4(self):
        pass

    def tryEntanglement(self):
        for link in self.topo.links:
            link.tryEntanglement()

    def work(self, pairs: list, time): 

        self.timeslot = time           # 目前回合
        self.srcDstPairs.extend(pairs) # 任務追加進去
        
        if self.timeslot == 0:
            self.prepare()
        
        
        self.topo.update(time)

        # start
        start = process_time()

        self.p2()  # 找path，分配資源
        
        self.tryEntanglement()

        res = self.p4() # swap

        # end   
        end = process_time()

        self.srcDstPairs.clear()

        res.totalRuntime += (end - start)
        res.algorithmRuntime = res.totalRuntime / res.numOfTimeslot

        return res

