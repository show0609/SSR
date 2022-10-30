import sys
import math
import random
import gurobipy as gp
from gurobipy import quicksum
from queue import PriorityQueue
sys.path.append("..")
from AlgorithmBase import AlgorithmBase
from AlgorithmBase import AlgorithmResult
from topo.Topo import *
from topo.Topo import Topo 
from topo.Node import Node 
from topo.Link import Link
from numpy import log as ln
from random import sample

EPS = 1e-6

class REPS(AlgorithmBase):
    def __init__(self, topo):
        super().__init__(topo)
        self.name = "REPS"
        self.useless_request = []
        self.requests = []
        self.totalUsedQubits = 0
        self.totalWaitingTime = 0
        self.totalTime = 0
        self.table = {}
        
        for i in range(len(topo.nodes)):
            self.table[topo.nodes[i].id] = i

    def genNameByComma(self, varName, parName):
        return (varName + str(parName)).replace(' ', '')
    def genNameByBbracket(self, varName: str, parName: list):
        return (varName + str(parName)).replace(' ', '').replace(',', '][')
    
    def printResult(self):

        remainTime = self.result.dropNum * self.topo.r
        tmp = self.requests[:]
        for req in tmp:
            remainTime += self.timeslot - req[2]
            if self.timeslot - req[2] >= self.topo.r: # 過期
                self.requests.remove(req)
                self.result.dropNum += 1

        # station link 數量
        for link in self.topo.links:
            if link.station_link == True:
                self.satLinkNum += 1

        self.topo.clearAllEntanglements()
        self.result.waitingTime = (self.totalTime + remainTime) / self.result.totalReqNum
        self.result.usedQubits = self.totalUsedQubits / self.result.totalReqNum
        if self.satLinkNum != 0:
            self.result.satSuccRatio = self.satSuccNum / self.satLinkNum
        self.result.remainRequestPerRound.append(len(self.requests) / self.result.totalReqNum)
        
        print('[', self.name, '] remain request:', len(self.requests))
        print('[', self.name, '] waiting time:', self.result.waitingTime)
        print('[', self.name, '] idle time:', self.result.idleTime)

    def AddNewSDpairs(self):
        for req in self.requests:
            req[0].remainingQubits -= 1
            self.totalUsedQubits += 1

        for (src, dst) in self.srcDstPairs:
            self.result.totalReqNum += 1
            if src.remainingQubits >= 2: # 資源夠
                self.requests.append((src, dst, self.timeslot))
                src.remainingQubits -= 1
                self.totalUsedQubits += 1
            else:
                self.result.dropNum += 1


        self.srcDstPairs = []
        for request in self.requests:
            if request in self.useless_request:
                continue
            src = request[0]
            dst = request[1]
            if (src, dst) not in self.srcDstPairs:
                self.srcDstPairs.append((src, dst))

    def p2(self):
        print('[', self.name, '] current timeslot:', self.timeslot)
        self.AddNewSDpairs()
        self.totalWaitingTime += len(self.requests)
        self.result.idleTime += len(self.requests)
        
        self.useless_request = []
        for req in self.requests:
            if (req[0].region,req[1].region) in self.topo.not_connect_table:
                self.useless_request.append(req)
        
        if len(self.srcDstPairs) > 0:
            self.result.numOfTimeslot += 1
            self.PFT() # compute (self.ti, self.fi)
        print('[', self.name, '] p2 end')
    
    def p4(self):
        if len(self.srcDstPairs) > 0:
            self.EPS()
            self.ELS()

        print('[', self.name, '] p4 end')
        self.printResult()
        print('----------------------------------')

        return self.result

    
    # return fi(u, v)

    def LP1(self):
        print('[', self.name, '] LP1 start')
        # initialize fi(u, v) ans ti
        self.fi_LP = {SDpair : {} for SDpair in self.srcDstPairs}
        self.ti_LP = {SDpair : 0 for SDpair in self.srcDstPairs}
        
        numOfNodes = len(self.topo.nodes)
        numOfSDpairs = len(self.srcDstPairs)

        edgeIndices = []
        notEdge = []
        for edge in self.topo.edges:
            edgeIndices.append((self.table[edge[0].id], self.table[edge[1].id]))
        
        for u in range(numOfNodes):
            for v in range(numOfNodes):
                if (u, v) not in edgeIndices and (v, u) not in edgeIndices:
                    notEdge.append((u, v))
        # LP

        m = gp.Model('REPS for PFT')
        m.setParam("OutputFlag", 0)
        # lb=lower bound
        # f = m.addVars(i,u,v) edge(u,v)上被選中做entangle connection的external link數量
        # t = m.addVars(i) SD pair i中entagle connection的數量
        # x = m.addVars(u,v) 為了最大化throughput的upper bound，edge(u,v)上應該要有多少external link
        f = m.addVars(numOfSDpairs, numOfNodes, numOfNodes, lb = 0, vtype = gp.GRB.CONTINUOUS, name = "f")
        t = m.addVars(numOfSDpairs, lb = 0, vtype = gp.GRB.CONTINUOUS, name = "t")
        x = m.addVars(numOfNodes, numOfNodes, lb = 0, vtype = gp.GRB.CONTINUOUS, name = "x")
        m.update()
        
        # maximum \sum{t_i}
        m.setObjective(quicksum(t[i] for i in range(numOfSDpairs)), gp.GRB.MAXIMIZE)

        # add constraint(1a-1c)
        for i in range(numOfSDpairs):
            s = self.table[self.srcDstPairs[i][0].id]
            m.addConstr(quicksum(f[i, s, v] for v in range(numOfNodes)) - quicksum(f[i, v, s] for v in range(numOfNodes)) == t[i])

            d = self.table[self.srcDstPairs[i][1].id]
            m.addConstr(quicksum(f[i, d, v] for v in range(numOfNodes)) - quicksum(f[i, v, d] for v in range(numOfNodes)) == -t[i])

            for u in range(numOfNodes):
                if u not in [s, d]:
                    m.addConstr(quicksum(f[i, u, v] for v in range(numOfNodes)) - quicksum(f[i, v, u] for v in range(numOfNodes)) == 0)

        # 1d 1e
        for (u, v) in edgeIndices:
            dis = getDistance(self.topo.nodes[u], self.topo.nodes[v])
            probability = math.exp(-self.topo.alpha * dis)
            m.addConstr(quicksum(f[i, u, v] + f[i, v, u] for i in range(numOfSDpairs)) <= probability * x[u, v])

            capacity = self.edgeCapacity(self.topo.nodes[u], self.topo.nodes[v])
            m.addConstr(x[u, v] <= capacity)

        # (u,v)不是edge的話就把flow設成0
        for (u, v) in notEdge:
            m.addConstr(x[u, v] == 0)               
            for i in range(numOfSDpairs):
                m.addConstr(f[i, u, v] == 0)

        # 1f
        for u in range(numOfNodes):
            edgeContainu = []
            for (n1, n2) in edgeIndices:
                if u in (n1, n2):
                    edgeContainu.append((n1, n2))
                    edgeContainu.append((n2, n1))
            m.addConstr(quicksum(x[n1, n2] for (n1, n2) in edgeContainu) <= self.topo.nodes[u].remainingQubits)

        m.optimize()

        for i in range(numOfSDpairs):
            SDpair = self.srcDstPairs[i]
            for edge in self.topo.edges:
                u = edge[0]
                v = edge[1]
                varName = self.genNameByComma('f', [i, self.table[u.id], self.table[v.id]])
                self.fi_LP[SDpair][(u, v)] = m.getVarByName(varName).x

            for edge in self.topo.edges:
                u = edge[1]
                v = edge[0]
                varName = self.genNameByComma('f', [i, self.table[u.id], self.table[v.id]])
                self.fi_LP[SDpair][(u, v)] = m.getVarByName(varName).x

            for (u, v) in notEdge:
                u = self.topo.nodes[u]
                v = self.topo.nodes[v]
                self.fi_LP[SDpair][(u, v)] = 0
            
            
            varName = self.genNameByComma('t', [i])
            self.ti_LP[SDpair] = m.getVarByName(varName).x
        print('[', self.name, '] LP1 end')
    def edgeCapacity(self, u, v):
        capacity = 0
        for link in u.links:
            if link.contains(v):
                capacity += 1
        used = 0
        for SDpair in self.srcDstPairs:
            used += self.fi[SDpair][(u, v)]
            used += self.fi[SDpair][(v, u)]
        return capacity - used

    def widthForSort(self, path):
        # path[-1] is the path of weight
        return -path[-1]
    
    def PFT(self):

        # initialize fi and ti
        self.fi = {SDpair : {} for SDpair in self.srcDstPairs}
        self.ti = {SDpair : 0 for SDpair in self.srcDstPairs}

        for SDpair in self.srcDstPairs:
            for u in self.topo.nodes:
                for v in self.topo.nodes:
                    self.fi[SDpair][(u, v)] = 0
        
        # PFT
        failedFindPath = False
        while not failedFindPath:
            self.LP1()
            failedFindPath = True
            Pi = {}  ## the paths used by SD pair i
            paths = []
            for SDpair in self.srcDstPairs:
                Pi[SDpair] = self.findPathsForPFT(SDpair) # 用min(distance[u], self.fi_LP[SDpair][(u, next)])做Dijkstra

            # 照著Algorithm 1刻
            for SDpair in self.srcDstPairs:
                K = len(Pi[SDpair])
                for k in range(K):
                    width = math.floor(Pi[SDpair][k][-1])
                    Pi[SDpair][k][-1] -= width
                    paths.append(Pi[SDpair][k])
                    pathLen = len(Pi[SDpair][k]) - 1
                    self.ti[SDpair] += width
                    if width == 0:
                        continue
                    failedFindPath = False
                    for nodeIndex in range(pathLen - 1):
                        node = Pi[SDpair][k][nodeIndex]
                        next = Pi[SDpair][k][nodeIndex + 1]
                        self.fi[SDpair][(node, next)] += width

            paths = sorted(paths, key = self.widthForSort)

            for path in paths:
                pathLen = len(path) - 1
                width = path[-1]
                SDpair = (path[0], path[-2])
                isable = True
                for nodeIndex in range(pathLen - 1):
                    node = path[nodeIndex]
                    next = path[nodeIndex + 1]
                    if self.edgeCapacity(node, next) < 1:
                        isable = False
                
                if not isable:
                    for nodeIndex in range(pathLen - 1):
                        node = path[nodeIndex]
                        next = path[nodeIndex + 1]
                    continue
                
                failedFindPath = False
                self.ti[SDpair] += 1
                for nodeIndex in range(pathLen - 1):
                    node = path[nodeIndex]
                    next = path[nodeIndex + 1]
                    self.fi[SDpair][(node, next)] += 1

        print('[', self.name, '] PFT end')

        # 給edge resource ?
        for SDpair in self.srcDstPairs:
            for edge in self.topo.edges:
                u = edge[0]
                v = edge[1]
                need = self.fi[SDpair][(u, v)] + self.fi[SDpair][(v, u)]
                if need:
                    assignCount = 0
                    for link in u.links:
                        if link.contains(v) and link.assignable():
                            # link(u, v) for u, v in edgeIndices
                            link.assignQubits()
                            self.totalUsedQubits += 2
                            assignCount += 1
                            if assignCount == need:
                                break
            
    def edgeSuccessfulEntangle(self, u, v):
        if u == v:
            return 0
        capacity = 0
        for link in u.links:
            if link.contains(v) and link.entangled:
                capacity += 1
        return capacity

    def LP2(self):
        print('[', self.name, '] LP2 start')
        # initialize fi(u, v) ans ti

        numOfNodes = len(self.topo.nodes)
        numOfSDpairs = len(self.srcDstPairs)
        numOfFlow = [self.ti[self.srcDstPairs[i]] for i in range(numOfSDpairs)]
        if len(numOfFlow):
            maxK = max(numOfFlow)
        else:
            maxK = 0
        self.fki_LP = {SDpair : [{} for _ in range(maxK)] for SDpair in self.srcDstPairs}
        self.tki_LP = {SDpair : [0] * maxK for SDpair in self.srcDstPairs}
        
        edgeIndices = []
        notEdge = []
        for edge in self.topo.edges:
            edgeIndices.append((self.table[edge[0].id], self.table[edge[1].id]))
        
        for u in range(numOfNodes):
            for v in range(numOfNodes):
                if (u, v) not in edgeIndices and (v, u) not in edgeIndices:
                    notEdge.append((u, v))

        m = gp.Model('REPS for EPS')
        m.setParam("OutputFlag", 0)

        f = [0] * numOfSDpairs
        for i in range(numOfSDpairs):
            f[i] = [0] * maxK
            for k in range(maxK):
                f[i][k] = [0] * numOfNodes
                for u in range(numOfNodes):
                    f[i][k][u] = [0] * numOfNodes 
                    for v in range(numOfNodes):
                        if k < numOfFlow[i] and ((u, v) in edgeIndices or (v, u) in edgeIndices):
                            f[i][k][u][v] = m.addVar(lb = 0, ub = 1, vtype = gp.GRB.CONTINUOUS, name = "f[%d][%d][%d][%d]" % (i, k, u, v))


        t = [0] * numOfSDpairs
        for i in range(numOfSDpairs):
            t[i] = [0] * maxK
            for k in range(maxK):
                if k < numOfFlow[i]:
                    t[i][k] = m.addVar(lb = 0, ub = 1, vtype = gp.GRB.CONTINUOUS, name = "t[%d][%d]" % (i, k))
                else:
                    t[i][k] = m.addVar(lb = 0, ub = 0, vtype = gp.GRB.CONTINUOUS, name = "t[%d][%d]" % (i, k))

        m.update()
        
        m.setObjective(quicksum(t[i][k] for k in range(maxK) for i in range(numOfSDpairs)), gp.GRB.MAXIMIZE)

        ## 2a-2c
        for i in range(numOfSDpairs):
            s = self.table[self.srcDstPairs[i][0].id]
            d = self.table[self.srcDstPairs[i][1].id]
            
            for k in range(numOfFlow[i]):
                neighborOfS = []
                neighborOfD = []

                for edge in edgeIndices:
                    if edge[0] == s:
                        neighborOfS.append(edge[1])
                    elif edge[1] == s:
                        neighborOfS.append(edge[0])
                    if edge[0] == d:
                        neighborOfD.append(edge[1])
                    elif edge[1] == d:
                        neighborOfD.append(edge[0])
                    
                m.addConstr(quicksum(f[i][k][s][v] for v in neighborOfS) - quicksum(f[i][k][v][s] for v in neighborOfS) == t[i][k])
                m.addConstr(quicksum(f[i][k][d][v] for v in neighborOfD) - quicksum(f[i][k][v][d] for v in neighborOfD) == -t[i][k])

                for u in range(numOfNodes):
                    if u not in [s, d]:
                        edgeUV = []
                        for v in range(numOfNodes):
                            if v not in [s, d]:
                                edgeUV.append(v)
                        m.addConstr(quicksum(f[i][k][u][v] for v in edgeUV) - quicksum(f[i][k][v][u] for v in edgeUV) == 0)

        ## 2d
        for (u, v) in edgeIndices:
            capacity = self.edgeSuccessfulEntangle(self.topo.nodes[u], self.topo.nodes[v])
            m.addConstr(quicksum((f[i][k][u][v] + f[i][k][v][u]) for k in range(maxK) for i in range(numOfSDpairs)) <= capacity)

        m.optimize()

        for i in range(numOfSDpairs):
            SDpair = self.srcDstPairs[i]

            for k in range(numOfFlow[i]):
                for edge in self.topo.edges:
                    u = edge[0]
                    v = edge[1]
                    varName = self.genNameByBbracket('f', [i, k, self.table[u.id], self.table[v.id]])
                    self.fki_LP[SDpair][k][(u, v)] = m.getVarByName(varName).x

                for edge in self.topo.edges:
                    u = edge[1]
                    v = edge[0]
                    varName = self.genNameByBbracket('f', [i, k, self.table[u.id], self.table[v.id]])
                    self.fki_LP[SDpair][k][(u, v)] = m.getVarByName(varName).x

                # for (u, v) in notEdge:
                #     u = self.topo.nodes[u]
                #     v = self.topo.nodes[v]
                #     self.fki_LP[SDpair][k][(u, v)] = 0
            
            
                varName = self.genNameByBbracket('t', [i, k])
                self.tki_LP[SDpair][k] = m.getVarByName(varName).x
        print('[', self.name, '] LP2 end')

    def EPS(self): #Algorithm 2
        self.LP2()
        # initialize fki(u, v), tki
        numOfFlow = {SDpair : self.ti[SDpair] for SDpair in self.srcDstPairs}
        self.fki = {SDpair : [{} for k in range(numOfFlow[SDpair])] for SDpair in self.srcDstPairs}
        self.tki = {SDpair : [0 for k in range(numOfFlow[SDpair])] for SDpair in self.srcDstPairs}
        self.pathForELS = {SDpair : [] for SDpair in self.srcDstPairs}

        for SDpair in self.srcDstPairs:
            for k in range(numOfFlow[SDpair]):
                for u in self.topo.nodes:
                    for v in self.topo.nodes:
                        self.fki[SDpair][k][(u, v)] = 0

        for SDpair in self.srcDstPairs:
            for k in range(numOfFlow[SDpair]):
                self.tki[SDpair][k] = self.tki_LP[SDpair][k] >= random.random() # 如果tki_LP[SDpair][k]>=random的數字就讓tki=1 -> 機率tki_LP

                if not self.tki[SDpair][k]:
                    continue
                paths = self.findPathsForEPS(SDpair, k)

                for u in self.topo.nodes:
                    for v in self.topo.nodes:
                        self.fki[SDpair][k][(u, v)] = 0
                    
                for path in paths:
                    width = path[-1]
                    select = (width / self.tki_LP[SDpair][k]) >= random.random()
                    if not select:
                        continue
                    path = path[:-1]
                    self.pathForELS[SDpair].append(path)
                    pathLen = len(path)
                    for nodeIndex in range(pathLen - 1):
                        node = path[nodeIndex]
                        next = path[nodeIndex + 1]
                        self.fki[SDpair][k][(node, next)] = 1
                
        print('[', self.name, '] EPS end')

    def ELS(self): #Algorithm 3
        Ci = self.pathForELS
        self.y = {(u, v) : 0 for u in self.topo.nodes for v in self.topo.nodes} # dictionary{key:value}
        self.weightOfNode = {node : -ln(node.q) for node in self.topo.nodes} # set node's weight to -ln(node.q)
        needLink = {}
        nextLink = {node : [] for node in self.topo.nodes}
        Pi = {SDpair : [] for SDpair in self.srcDstPairs}
        T = [SDpair for SDpair in self.srcDstPairs]

        while len(T) > 0:
            for SDpair in self.srcDstPairs:
                removePaths = []  
                for path in Ci[SDpair]:
                    pathLen = len(path)
                    noResource = False
                    for nodeIndex in range(pathLen - 1):
                        node = path[nodeIndex]
                        next = path[nodeIndex + 1]
                        if self.y[(node, next)] >= self.edgeSuccessfulEntangle(node, next):
                            noResource = True
                    if noResource:
                        removePaths.append(path)
                for path in removePaths: 
                    Ci[SDpair].remove(path)
                if len(Ci[SDpair]) == 0 and SDpair in T:
                    T.remove(SDpair)
            
            if len(T) == 0:
                break

            # 找出最短的路
            i = -1
            minLength = math.inf
            for SDpair in T:
                for path in Ci[SDpair]:
                    if len(path) < minLength:
                        minLength = len(path)
                        i = SDpair
            
            src = i[0]
            dst = i[1]

            minR = math.inf
            for path in Ci[i]:
                r = 0
                for node in path:
                    r += self.weightOfNode[node]
                if minR > r:
                    targetPath = path
                    minR = r
            
            pathIndex = len(Pi[i])
            needLink[(i, pathIndex)] = []

            # 把多加的path組合到Pi然後y++
            Pi[i].append(targetPath)
            for nodeIndex in range(1, len(targetPath) - 2):
                prev = targetPath[nodeIndex - 1]
                node = targetPath[nodeIndex]
                next = targetPath[nodeIndex + 1]
                for link in node.links:
                    if link.contains(next) and link.entangled and link.notSwapped():
                        targetLink1 = link
                    
                    if link.contains(prev) and link.entangled and link.notSwapped():
                        targetLink2 = link
                
                self.y[((node, next))] += 1
                self.y[((next, node))] += 1
                self.y[((node, prev))] += 1
                self.y[((prev, node))] += 1

                nextLink[node].append(targetLink1)
                needLink[(i, pathIndex)].append((node, targetLink1, targetLink2))

            T.remove(i)

        # residue graph再做一次
        T = [SDpair for SDpair in self.srcDstPairs]
        while len(T) > 0:
            for SDpair in self.srcDstPairs:
                removePaths = []
                for path in Ci[SDpair]:
                    pathLen = len(path)
                    noResource = False
                    for nodeIndex in range(pathLen - 1):
                        node = path[nodeIndex]
                        next = path[nodeIndex + 1]
                        if self.y[(node, next)] >= self.edgeSuccessfulEntangle(node, next):
                            noResource = True
                    if noResource:
                        removePaths.append(path)
                for path in removePaths:
                    Ci[SDpair].remove(path)
            
            i = -1
            minLength = math.inf
            for SDpair in T:
                for path in Ci[SDpair]:
                    if len(path) - 1 < minLength:
                        minLength = len(path) - 1
                        i = SDpair
                if len(Ci[SDpair]) == 0 and i == -1:
                    i = SDpair
            
            src = i[0]
            dst = i[1]

            targetPath = self.findPathForELS(i)
            pathIndex = len(Pi[i])
            needLink[(i, pathIndex)] = []
            Pi[i].append(targetPath)
            for nodeIndex in range(1, len(targetPath) - 1):
                prev = targetPath[nodeIndex - 1]
                node = targetPath[nodeIndex]
                next = targetPath[nodeIndex + 1]
                ## if prev 和 next之間連通
                for link in node.links:
                    if link.contains(next) and link.entangled:
                        targetLink1 = link
                    
                    if link.contains(prev) and link.entangled:
                        targetLink2 = link
                
                self.y[((node, next))] += 1
                self.y[((next, node))] += 1
                self.y[((node, prev))] += 1
                self.y[((prev, node))] += 1
                nextLink[node].append(targetLink1)
                needLink[(i, pathIndex)].append((node, targetLink1, targetLink2))
            T.remove(i)
        
        print('[', self.name, '] ELS end')
        # print('[', self.name, ']' + [(src.id, dst.id) for (src, dst) in self.srcDstPairs])

        # swap
        for SDpair in self.srcDstPairs:
            src = SDpair[0]
            dst = SDpair[1]

            for pathIndex in range(len(Pi[SDpair])):
                path = Pi[SDpair][pathIndex]
                if(len(path)>0):
                    self.result.idleTime -= 1
                    break

            for pathIndex in range(len(Pi[SDpair])):
                path = Pi[SDpair][pathIndex]
                print('[', self.name, '] attempt:', [node.id for node in path])
                for (node, link1, link2) in needLink[(SDpair, pathIndex)]:
                    node.attemptSwapping(link1, link2)
                
                successPath = self.topo.getEstablishedEntanglements(src, dst)
                
                for x in successPath:
                    print('[', self.name, '] success:', [z.id for z in x])

                if len(successPath):
                    tmp = self.requests[:]
                    for request in tmp:
                        if (src, dst) == (request[0], request[1]):
                            print('[', self.name, '] finish time:', self.timeslot - request[2])
                            self.totalTime += self.timeslot - request[2]
                            self.requests.remove(request)
                            self.result.successRequestNum += 1
                            
                            # 成功使用衛星
                            for i in range(len(path) - 1):
                                n1 = path[i]
                                n2 = path[i+1]
                                tmp = self.topo.node_to_link(n1,n2)
                                if tmp[0].station_link:
                                    self.satSuccNum += 1

                            break
                            
                        
                for (node, link1, link2) in needLink[(SDpair, pathIndex)]:
                    link1.clearPhase4Swap()
                    link2.clearPhase4Swap()

    def findPathsForPFT(self, SDpair):
        src = SDpair[0]
        dst = SDpair[1]
        pathList = []

        while self.DijkstraForPFT(SDpair):
            path = []
            currentNode = dst
            while currentNode != self.topo.sentinel:
                path.append(currentNode)
                currentNode = self.parent[currentNode]

            path = path[::-1]
            width = self.widthForPFT(path, SDpair)
            
            for i in range(len(path) - 1):
                node = path[i]
                next = path[i + 1]
                self.fi_LP[SDpair][(node, next)] -= width

            path.append(width)
            pathList.append(path.copy())

        return pathList
    
    def DijkstraForPFT(self, SDpair):
        src = SDpair[0]
        dst = SDpair[1]
        self.parent = {node : self.topo.sentinel for node in self.topo.nodes}
        adjcentList = {node : set() for node in self.topo.nodes}
        for node in self.topo.nodes:
            for link in node.links:
                neighbor = link.theOtherEndOf(node)
                adjcentList[node].add(neighbor)
        
        distance = {node : 0 for node in self.topo.nodes}
        visited = {node : False for node in self.topo.nodes}
        pq = PriorityQueue()

        pq.put((-math.inf, self.table[src.id]))
        while not pq.empty():
            (dist, uid) = pq.get()
            u = self.topo.nodes[uid]
            if visited[u]:
                continue

            if u == dst:
                return True
            distance[u] = -dist
            visited[u] = True
            
            for next in adjcentList[u]:
                newDistance = min(distance[u], self.fi_LP[SDpair][(u, next)])
                if distance[next] < newDistance:
                    distance[next] = newDistance
                    self.parent[next] = u
                    pq.put((-distance[next], self.table[next.id]))

        return False

    def widthForPFT(self, path, SDpair):
        numOfnodes = len(path)
        width = math.inf
        for i in range(numOfnodes - 1):
            currentNode = path[i]
            nextNode = path[i + 1]
            width = min(width, self.fi_LP[SDpair][(currentNode, nextNode)])

        return width
    
    def findPathsForEPS(self, SDpair, k):
        src = SDpair[0]
        dst = SDpair[1]
        pathList = []

        while self.DijkstraForEPS(SDpair, k):
            path = []
            currentNode = dst
            while currentNode != self.topo.sentinel:
                path.append(currentNode)
                currentNode = self.parent[currentNode]

            path = path[::-1]
            width = self.widthForEPS(path, SDpair, k)
            for i in range(len(path) - 1):
                node = path[i]
                next = path[i + 1]
                self.fki_LP[SDpair][k][(node, next)] -= width

            path.append(width)
            pathList.append(path.copy())

        return pathList
    
    def DijkstraForEPS(self, SDpair, k):
        src = SDpair[0]
        dst = SDpair[1]
        self.parent = {node : self.topo.sentinel for node in self.topo.nodes}
        adjcentList = {node : set() for node in self.topo.nodes}
        for node1 in self.topo.nodes:
            for node2 in self.topo.nodes:
                if self.edgeSuccessfulEntangle(node1, node2) > 0:
                    adjcentList[node1].add(node2)
        
        distance = {node : 0 for node in self.topo.nodes}
        visited = {node : False for node in self.topo.nodes}
        pq = PriorityQueue()

        pq.put((-math.inf, self.table[src.id]))
        while not pq.empty():
            (dist, uid) = pq.get()
            u = self.topo.nodes[uid]
            if visited[u]:
                continue

            if u == dst:
                return True
            distance[u] = -dist
            visited[u] = True
            
            for next in adjcentList[u]:
                newDistance = min(distance[u], self.fki_LP[SDpair][k][(u, next)])
                if distance[next] < newDistance:
                    distance[next] = newDistance
                    self.parent[next] = u
                    pq.put((-distance[next], self.table[next.id]))

        return False

    def widthForEPS(self, path, SDpair, k):
        numOfnodes = len(path)
        width = math.inf
        for i in range(numOfnodes - 1):
            currentNode = path[i]
            nextNode = path[i + 1]
            width = min(width, self.fki_LP[SDpair][k][(currentNode, nextNode)])

        return width
    
    def findPathForELS(self, SDpair):
        src = SDpair[0]
        dst = SDpair[1]
        if self.DijkstraForELS(SDpair):
            path = []
            currentNode = dst
            while currentNode != self.topo.sentinel:
                path.append(currentNode)
                currentNode = self.parent[currentNode]
            path = path[::-1]
            return path
        else:
            return []
    
    def DijkstraForELS(self, SDpair):
        src = SDpair[0]
        dst = SDpair[1]
        self.parent = {node : self.topo.sentinel for node in self.topo.nodes}
        adjcentList = {node : set() for node in self.topo.nodes}
        for node1 in self.topo.nodes:
            for node2 in self.topo.nodes:
                if self.y[(node1, node2)] < self.edgeSuccessfulEntangle(node1, node2):
                    adjcentList[node1].add(node2)
        
        distance = {node : math.inf for node in self.topo.nodes}
        visited = {node : False for node in self.topo.nodes}
        pq = PriorityQueue()

        pq.put((self.weightOfNode[src], self.table[src.id]))
        while not pq.empty():
            (dist, uid) = pq.get()
            u = self.topo.nodes[uid]
            if visited[u]:
                continue

            if u == dst:
                return True
            distance[u] = dist
            visited[u] = True
            
            for next in adjcentList[u]:
                newDistance = distance[u] + self.weightOfNode[next]
                if distance[next] > newDistance:
                    distance[next] = newDistance
                    self.parent[next] = u
                    pq.put((distance[next], self.table[next.id]))

        return False
if __name__ == '__main__':
    
    topo = Topo.generate(0.9, 0.001, 0.0001, 0.7, 0.75) # (self, q, alpha, alpha_sat, p_sat, density)    
    algo = REPS(topo)

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