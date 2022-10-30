import sys
sys.path.append("..")
from AlgorithmBase import AlgorithmBase
from topo.Topo import Topo 
from topo.Node import Node 
from topo.Link import Link
from random import sample
import random

class GreedyHopRouting(AlgorithmBase):

    def __init__(self, topo):
        super().__init__(topo)
        self.name = "Greedy_H"
        self.pathsSortedDynamically = [] # [(0.0, width, p, time)] (照 width 排序)
        self.requests = []
        self.totalTime = 0
        self.totalUsedQubits = 0

    def prepare(self):
        self.totalTime = 0
        self.requests.clear()
        
    def p2(self):
        print('[', self.name, '] current timeslot:', self.timeslot)
        self.pathsSortedDynamically.clear()
        
        # 消耗src資源
        for req in self.requests:
            req[0].remainingQubits -= 1
            self.totalUsedQubits += 1
        
        for req in self.srcDstPairs: # 加入新 request
            (src, dst) = req
            self.result.totalReqNum += 1
            if src.remainingQubits >= 2: # 資源夠
                self.requests.append((src, dst, self.timeslot))
                src.remainingQubits -= 1
                self.totalUsedQubits += 1
            else: # drop
                self.result.dropNum += 1
        
        if len(self.requests) > 0:  # 還有request沒做完，時間加1
            self.result.numOfTimeslot += 1

        while True:
            found = False   # record this round whether find new path

            # Find the shortest path and assing qubits for every srcDstPair
            for req in self.requests:
                (src, dst, time) = req
                p = []
                p.append(src)
                
                # Find a shortest path by greedy min hop  
                while True:
                    last = p[-1] ## 目前最後一個點
                    if last == dst: 
                        break

                    # Select avaliable neighbors of last(local)
                    selectedNeighbors = []    # type Node
                    selectedNeighbors.clear()
                    
                    for neighbor in last.neighbors:
                        if neighbor.remainingQubits > 2 or (neighbor == dst and neighbor.remainingQubits > 1): # qubit夠
                            for link in neighbor.links:
                                if link.contains(last) and (not link.assigned) : # 還有可以用的channel
                                    selectedNeighbors.append(neighbor)
                                    break
                                
                    # Choose the neighbor with smallest number of hop from it to dst
                    next = self.topo.sentinel
                    hopsCurMinNum = sys.maxsize
                    for selectedNeighbor in selectedNeighbors:
                        hopsNum = self.topo.hopsAway(selectedNeighbor, dst, 'Hop')  
                        if hopsCurMinNum > hopsNum and hopsNum != -1:
                            hopsCurMinNum = hopsNum
                            next = selectedNeighbor

                    # If have cycle, break
                    if next == self.topo.sentinel or next in p:
                        break 
                    p.append(next)
                # while end

                if p[-1] != dst:
                    continue
                
                # Caculate width for p
                width = self.topo.widthPhase2(p)
                
                if width == 0:
                    continue

                found = True
                self.pathsSortedDynamically.append((0.0, width, p, time))
                self.pathsSortedDynamically = sorted(self.pathsSortedDynamically, key=lambda x: x[1]) # 照粗度排序

                # Assign Qubits for links in path 
                for _ in range(0, width):
                    for s in range(0, len(p) - 1):
                        n1 = p[s]
                        n2 = p[s+1]
                        for link in n1.links:
                            if link.contains(n2) and (not link.assigned):
                                self.totalUsedQubits += 2
                                link.assignQubits()
                                break    
            # SDpairs end

            if not found:
                break
        # while end
        
        for req in self.requests:
            pick = False
            for path in self.pathsSortedDynamically:
                _, width, p, time = path
                if (p[0], p[-1], time) == req:
                    pick = True
                    break           
            if not pick:
                self.result.idleTime += 1 # 找不到 path
        print('[', self.name, '] p2 end')
    
    def p4(self):
        print(len(self.pathsSortedDynamically))
        for path in self.pathsSortedDynamically: # 粗度小的先?
            _, width, p, time = path
            
            oldNumOfPairs = len(self.topo.getEstablishedEntanglements(p[0], p[-1])) # =0 ?
            for i in range(1, len(p) - 1):
                prev = p[i-1]
                curr = p[i]
                next = p[i+1]
                prevLinks = []
                nextLinks = []
                
                w = width
                for link in curr.links:
                    if link.entangled and (link.n1 == prev and not link.s2 or link.n2 == prev and not link.s1) and w > 0: # ?
                        prevLinks.append(link)
                        w -= 1

                w = width
                for link in curr.links:
                    if link.entangled and (link.n1 == next and not link.s2 or link.n2 == next and not link.s1) and w > 0:
                        nextLinks.append(link)
                        w -= 1

                if len(prevLinks) == 0 or len(nextLinks) == 0:
                    break
                for (l1, l2) in zip(prevLinks, nextLinks):
                    curr.attemptSwapping(l1, l2)

            succ = len(self.topo.getEstablishedEntanglements(p[0], p[-1])) - oldNumOfPairs
        
            if succ > 0 or len(p) == 2:
                # 成功使用衛星
                for i in range(len(p) - 1):
                    n1 = p[i]
                    n2 = p[i+1]
                    tmp = self.topo.node_to_link(n1,n2)
                    if tmp[0].station_link:
                        self.satSuccNum += 1

                find = (p[0], p[-1], time)
                if find in self.requests: # request 已完成
                    self.totalTime += self.timeslot - time
                    self.requests.remove(find)
                    self.result.successRequestNum += 1

        remainTime = self.result.dropNum * self.topo.r
        tmp = self.requests[:]
        for req in tmp:
            remainTime += self.timeslot - req[2]  # 還沒完成的 request 的等待時間加總
            if self.timeslot - req[2] >= self.topo.r: # 過期
                self.requests.remove(req)
                self.result.dropNum += 1

        # station link 數量
        for link in self.topo.links:
            if link.station_link == True:
                self.satLinkNum += 1

        self.topo.clearAllEntanglements() # 全都不保留
        
        self.result.remainRequestPerRound.append(len(self.requests)/self.result.totalReqNum)     
        self.result.waitingTime = (self.totalTime + remainTime) / self.result.totalReqNum
        self.result.usedQubits = self.totalUsedQubits / self.result.totalReqNum
        if self.satLinkNum != 0:
            self.result.satSuccRatio = self.satSuccNum / self.satLinkNum

        print('[', self.name, '] p4 end')
        print('[', self.name, '] remain request:', len(self.requests))
        print('[', self.name, '] waiting time:', self.result.waitingTime) # request 的平均等待時間(含還沒完成的request)
        print('[', self.name, '] idle time:', self.result.idleTime)
        print('----------------------------------')

        return self.result
        
if __name__ == '__main__':
    
    topo = Topo.generate(0.9, 0.001, 0.0001, 0.7, 0.75) # (self, q, alpha, alpha_sat, p_sat, density)    
    algo = GreedyHopRouting(topo)

    random.seed()
    for i in range(120):
        if i < 10:
            while True:
                req = sample(topo.nodes, 2)
                if req[0] in topo.socialRelationship[req[1]]:
                    break
            algo.work([(req[0],req[1])], i)
        else:
            algo.work([], i)

