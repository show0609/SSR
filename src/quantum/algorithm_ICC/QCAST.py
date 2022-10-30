from dataclasses import dataclass
import sys
sys.path.append("..")
from AlgorithmBase import AlgorithmBase
from topo.Topo import Topo 
from topo.Node import Node 
from topo.Link import Link
from random import sample
import random
import copy

@dataclass
class RecoveryPath:
    path: list
    width: int
    taken: int 
    available: int

@dataclass
class PickedPath:
    weight: float # EXT
    width: int
    path: list
    time: int

    def __hash__(self):
        return hash((self.weight, self.width, self.path[0], self.path[-1]))


class QCAST(AlgorithmBase):

    def __init__(self, topo, allowRecoveryPaths = True):
        super().__init__(topo)
        self.name = "QCAST"
        self.pathsSortedDynamically = []
        self.requests = []
        self.totalTime = 0  # 完成的 request 的等待時間加總
        self.totalUsedQubits = 0
        
        self.majorPaths = []            # [PickedPath, ...]
        self.recoveryPaths = {}         # {PickedPath: [PickedPath, ...], ...}
        self.pathToRecoveryPaths = {}   # {PickedPath : [RecoveryPath, ...], ...}
        
        self.allowRecoveryPaths = allowRecoveryPaths
    
    def prepare(self):
        self.totalTime = 0
        self.requests.clear()

    def p2(self):
        print('[', self.name, '] current timeslot:', self.timeslot)
        self.majorPaths.clear()
        self.recoveryPaths.clear()
        self.pathToRecoveryPaths.clear()
            
        for req in self.requests:
            req[0].remainingQubits -= 1
            self.totalUsedQubits += 1
        
        for req in self.srcDstPairs:
            (src, dst) = req
            self.result.totalReqNum += 1
            if src.remainingQubits >= 2: # 資源夠
                self.requests.append((src, dst, self.timeslot))
                src.remainingQubits -= 1
                self.totalUsedQubits += 1
            else:
                self.result.dropNum += 1

        if len(self.requests) > 0:
            self.result.numOfTimeslot += 1

        while True: 
            candidates = self.calCandidates(self.requests) # candidates -> [PickedPath, ...]   
            candidates = sorted(candidates, key=lambda x: x.weight) # 照 weight 排序
            if len(candidates) == 0:
                break
            pick = candidates[-1]   # pick -> PickedPath # EXT 大的先

            # print('-----')
            # for c in candidates:
            #     print('[', self.name, '] Path:', [x.id for x in c.path])
            #     print('[', self.name, '] EXT:', c.weight)
            #     print('[', self.name, '] width:', c.width)
            
            # print('[', self.name, '] pick: ', [x.id for x in pick.path])
            # print('-----')
            
            if pick.weight > 0.0: 
                self.pickAndAssignPath(pick)
            else:
                break

        if self.allowRecoveryPaths:
            # print('[', self.name, '] P2Extra')
            self.P2Extra()
            print('[', self.name, '] P2Extra end')
        
        for req in self.requests:
            pick = False
            for pathWithWidth in self.majorPaths:
                p = pathWithWidth.path
                if (p[0], p[-1], pathWithWidth.time) == req:
                    pick = True
                    break       
            if not pick:
                self.result.idleTime += 1
        print('[', self.name, '] p2 end')
         
    # 對每個SD-pair找出候選路徑
    def calCandidates(self, requests: list): # pairs -> [(Node, Node), ...]
        candidates = [] 
        for req in requests:

            candidate = []
            (src, dst, time) = req
            maxM = min(src.remainingQubits, dst.remainingQubits)
            if maxM == 0:   # not enough qubit
                continue

            for w in range(maxM, 0, -1): # w = maxM, maxM-1, maxM-2, ..., 1
                failNodes = []

                # collect failnodes (they don't have enough Qubits for SDpair in width w)
                for node in self.topo.nodes:
                    if node.remainingQubits < 2 * w and node != src and node != dst:
                        failNodes.append(node)

                edges = {}  # edges -> {(Node, Node): [Link, ...], ...}

                # collect edges with links 
                for link in self.topo.links:
                    if not link.assigned and link.n1 not in failNodes and link.n2 not in failNodes:
                        if not edges.__contains__((link.n1, link.n2)):
                            edges[(link.n1, link.n2)] = []
                        edges[(link.n1, link.n2)].append(link)

                neighborsOf = {node: [] for node in self.topo.nodes} # neighborsOf -> {Node: [Node, ...], ...}

                # filter available links satisfy width w
                for edge in edges:
                    links = edges[edge]
                    if len(links) >= w:
                        neighborsOf[edge[0]].append(edge[1])
                        neighborsOf[edge[1]].append(edge[0])
                                             
                if (len(neighborsOf[src]) == 0 or len(neighborsOf[dst]) == 0):
                    continue

                prevFromSrc = {}   # prevFromSrc -> {cur: prev}

                def getPathFromSrc(n): 
                    path = []
                    cur = n
                    while (cur != self.topo.sentinel): 
                        path.insert(0, cur)
                        cur = prevFromSrc[cur]
                    return path
                
                E = {node.id : [-sys.float_info.max, [0.0 for _ in range(0,w+1)]] for node in self.topo.nodes}  # E -> {Node id: [Int, [double, ...]], ...}
                q = []  # q -> [(E, Node, Node), ...]

                E[src.id] = [sys.float_info.max, [0.0 for _ in range(0,w+1)]]
                q.append((E[src.id][0], src, self.topo.sentinel))
                q = sorted(q, key=lambda q: q[0])

                # Dijkstra by EXT
                while len(q) != 0:
                    contain = q.pop(-1) # pop the node with the highest E
                    u, prev = contain[1], contain[2]
                    if u in prevFromSrc.keys():
                        continue
                    prevFromSrc[u] = prev

                    # If find the dst add path to candidates
                    if u == dst:        
                        candidate.append(PickedPath(E[dst.id][0], w, getPathFromSrc(dst), time))
                        break
                    
                    # Update neighbors by EXT
                    for neighbor in neighborsOf[u]:
                        tmp = copy.deepcopy(E[u.id][1])
                        p = getPathFromSrc(u)
                        p.append(neighbor)
                        e = self.topo.e(p, w, tmp)
                        newE = [e, tmp]
                        oldE = E[neighbor.id]

                        if oldE[0] < newE[0]:
                            E[neighbor.id] = newE
                            q.append((E[neighbor.id][0], neighbor, u))
                            q = sorted(q, key=lambda q: q[0])
                # Dijkstra end

                # 假如此SD-pair在width w有找到path則換找下一個SD-pair
                if len(candidate) > 0:
                    candidates += candidate
                    break
            # for w end      
        # for pairs end
        return candidates

    def pickAndAssignPath(self, pick: PickedPath, majorPath: PickedPath = None):
        if majorPath != None:
            self.recoveryPaths[majorPath].append(pick)
        else:
            self.majorPaths.append(pick)
            self.recoveryPaths[pick] = list()
            
        width = pick.width

        for i in range(0, len(pick.path) - 1):
            links = []
            n1, n2 = pick.path[i], pick.path[i+1]

            for link in n1.links:
                if link.contains(n2) and not link.assigned:
                    links.append(link)
            # links = sorted(links, key=lambda q: q.id)
            
            for i in range(0, width): # 前 w 分配資源
                self.totalUsedQubits += 2
                links[i].assignQubits()

    def P2Extra(self):
        for majorPath in self.majorPaths:
            p = majorPath.path

            # for l in range(1, self.topo.k + 1): # 長度 1~k 間找 recovery path
            for l in range(1, 5 + 1): # 長度 1~k 間找 recovery path
                for i in range(0, len(p) - l):
                    (src, dst) = (p[i], p[i+l])

                    candidates = self.calCandidates([(src, dst, self.timeslot)]) # candidates -> [PickedPath, ...]   
                    candidates = sorted(candidates, key=lambda x: x.weight)
                    if len(candidates) == 0:
                        continue
                    pick = candidates[-1]   # pick -> PickedPath

                    if pick.weight > 0.0: 
                        self.pickAndAssignPath(pick, majorPath)

    def p4(self):
        for pathWithWidth in self.majorPaths:
            width = pathWithWidth.width
            majorPath = pathWithWidth.path
            time = pathWithWidth.time
            
            oldNumOfPairs = len(self.topo.getEstablishedEntanglements(majorPath[0], majorPath[-1]))

            recoveryPaths = self.recoveryPaths[pathWithWidth]   # recoveryPaths -> [pickedPath, ...]
            recoveryPaths = sorted(recoveryPaths, key=lambda x: len(x.path)*10000 + majorPath.index(x.path[0])) # sort recoveryPaths by it recoverypath length and the index of the first node in recoveryPath 

            # Construct pathToRecoveryPaths table
            for recoveryPath in recoveryPaths:
                w = recoveryPath.width
                p = recoveryPath.path
                available = sys.maxsize
                for i in range(0, len(p) - 1):
                    n1 = p[i]
                    n2 = p[i+1]
                    cnt = 0
                    for link in n1.links:
                        if link.contains(n2) and link.entangled:
                            cnt += 1
                    if cnt < available:
                        available = cnt
                
                if not self.pathToRecoveryPaths.__contains__(pathWithWidth):
                    self.pathToRecoveryPaths[pathWithWidth] = []
                
                self.pathToRecoveryPaths[pathWithWidth].append(RecoveryPath(p, w, 0, available))
            # for end 

            rpToWidth = {tuple(recoveryPath.path): recoveryPath.width for recoveryPath in recoveryPaths}  # rpToWidth -> {tuple: int, ...}

            # for w-width major path, treat it as w different paths, and repair separately
            for w in range(1, width + 1):
                brokenEdges = list()    # [(int, int), ...]

                # find all broken edges on the major path
                # 尋找在 majorPath 裡斷掉的 link，其中一條斷掉就要記錄。
                for i in range(0, len(majorPath) - 1):
                    i1 = i
                    i2 = i+1
                    n1 = majorPath[i1]
                    n2 = majorPath[i2]

                    for link in n1.links:
                        if link.contains(n2) and link.assigned and link.notSwapped() and not link.entangled:
                            brokenEdges.append((i1, i2))

                edgeToRps = {brokenEdge: [] for brokenEdge in brokenEdges}   # {tuple : [tuple, ...], ...}
                rpToEdges = {tuple(recoveryPath.path): [] for recoveryPath in recoveryPaths}    # {tuple : [tuple, ...], ...}

                # Construct edgeToRps & rpToEdges
                # 掃描所有可以用的 recoveryPath，看它斷在 majorPath 的哪裡，並標記。
                for recoveryPath in recoveryPaths:
                    rp = recoveryPath.path
                    s1, s2 = majorPath.index(rp[0]), majorPath.index(rp[-1])

                    for j in range(s1, s2):
                        if (j, j+1) in brokenEdges:
                            edgeToRps[(j, j+1)].append(tuple(rp))
                            rpToEdges[tuple(rp)].append((j, j+1))
                        # elif (j+1, j) in brokenEdges:
                        #     edgeToRps[(j+1, j)].append(tuple(rp))
                        #     rpToEdges[tuple(rp)].append((j+1, j))

                realRepairedEdges = set()
                realPickedRps= set()

                # try to cover the broken edges
                # 掃描每個斷掉的 edge
                for brokenEdge in brokenEdges:
                    # if the broken edge is repaired, go to repair the next broken edge
                    if brokenEdge in realRepairedEdges: 
                        continue
                    repaired = False
                    next = 0    # last repaired location
                    rps = edgeToRps[brokenEdge] # the rps cover the edge
                    
                    # filter the avaliable rp in rps for brokenEdge
                    for rp in rps:
                        if rpToWidth[tuple(rp)] <= 0 or tuple(rp) in realPickedRps:
                            rps.remove(rp)

                    # sort rps by the start id in majorPath
                    rps = sorted(rps, key=lambda x: majorPath.index(x[0]) * 10000 + majorPath.index(x[-1]) )

                    for rp in rps:
                        if majorPath.index(rp[0]) < next:
                            continue 

                        next = majorPath.index(rp[-1])
                        pickedRps = realPickedRps
                        repairedEdges = realRepairedEdges
                        otherCoveredEdges = set(rpToEdges[tuple(rp)]) - {brokenEdge}
                        covered = False

                        for edge in otherCoveredEdges: #delete covered rps, or abort
                            prevRp = set(tuple(edgeToRps[edge])) & pickedRps    # 這個edge 所覆蓋到的rp 假如已經有被選過 表示她被修理過了 表示目前這個rp要修的edge蓋到以前的rp
                            
                            if prevRp == set():
                                repairedEdges.add(edge)
                            else: 
                                covered = True
                                break  # the rps overlap. taking time to search recursively. just abort
                        
                        if covered:
                            continue

                        repaired = True      
                        repairedEdges.add(brokenEdge) 
                        pickedRps.add(tuple(rp))

                        for rp in realPickedRps - pickedRps:
                            rpToWidth[tuple(rp)] += 1
                        for rp in pickedRps - realPickedRps:
                            rpToWidth[tuple(rp)] -= 1
                        
                        realPickedRps = pickedRps
                        realRepairedEdges = repairedEdges
                    # for rp end

                    if not repaired:   # this major path cannot be repaired
                        break
                # for brokenEdge end

                acc = majorPath
                for rp in realPickedRps:
                    for recoveryPath in self.pathToRecoveryPaths[pathWithWidth]:
                        if recoveryPath.path == rp:
                            recoveryPath.taken += 1
                            break
                    
                    toOrigin = set()
                    toAdd = set()
                    toDelete = set()

                    for i in range(0, len(acc) - 1):
                        toOrigin.add((acc[i], acc[i+1]))
                    for i in range(0, len(rp) - 1):
                        toAdd.add((rp[i], rp[i+1]))

                    startDelete = 0
                    endDelete = len(acc) - 1

                    for i in range(0, len(acc)):
                        startDelete = i
                        if acc[i] == rp[0]:
                            break
                    for i in range(len(acc) - 1, -1, -1):
                        endDelete = i
                        if acc[i] == rp[-1]:
                            break   
                    for i in range(startDelete, endDelete):
                        toDelete.add((acc[i], acc[i+1]))

                    edgesOfNewPathAndCycles = (toOrigin - toDelete) | toAdd
                    p = self.topo.shortestPath(acc[0], acc[-1], 'Hop', edgesOfNewPathAndCycles)
                    acc = p[1]

                nodes = []
                prevLinks = []
                nextLinks = [] 
                
                # swap (select links)
                for i in range(1, len(acc) - 1):
                    prev = acc[i-1]
                    curr = acc[i]
                    next = acc[i+1]
                    prevLink = []
                    nextLink = []  
                 
                    for link in curr.links:
                        if link.entangled and (link.n1 == prev and not link.s2 or link.n2 == prev and not link.s1):
                            prevLink.append(link)
                            break

                    for link in curr.links:
                        if link.entangled and (link.n1 == next and not link.s2 or link.n2 == next and not link.s1):
                            nextLink.append(link)
                            break

                    if len(prevLink) == 0 or len(nextLink) == 0:
                        break
                    
                    nodes.append(curr)
                    prevLinks.append(prevLink[0])
                    nextLinks.append(nextLink[0])

                # swap 
                if len(nodes) == len(acc) - 2 and len(acc) > 2:
                    for (node, l1, l2) in zip(nodes, prevLinks, nextLinks):                    
                        node.attemptSwapping(l1, l2)
            # for w end


            succ = len(self.topo.getEstablishedEntanglements(acc[0], acc[-1])) - oldNumOfPairs
            
            if succ > 0 or len(acc) == 2:
                # 成功使用衛星
                for i in range(len(acc) - 1):
                    n1 = acc[i]
                    n2 = acc[i+1]
                    tmp = self.topo.node_to_link(n1,n2)
                    if tmp[0].station_link:
                        self.satSuccNum += 1

                find = (acc[0], acc[-1], time)
                if find in self.requests:
                    self.totalTime += self.timeslot - time
                    self.requests.remove(find)
                    self.result.successRequestNum += 1
        # for pathWithWidth end

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
        self.result.remainRequestPerRound.append(len(self.requests) / self.result.totalReqNum)   
        self.result.waitingTime = (self.totalTime + remainTime) / self.result.totalReqNum + 1
        self.result.usedQubits = self.totalUsedQubits / self.result.totalReqNum
        if self.satLinkNum != 0:
            self.result.satSuccRatio = self.satSuccNum / self.satLinkNum

        print('[', self.name, '] remain request:', len(self.requests))
        print('[', self.name, '] waiting time:', self.result.waitingTime) # request 的平均等待時間(含還沒完成的request)
        print('[', self.name, '] idle time:', self.result.idleTime)
        print('----------------------------------')

        return self.result

if __name__ == '__main__':

    topo = Topo.generate(0.9, 0.001, 0.0001, 0.7, 0.75) # (self, q, alpha, alpha_sat, p_sat, density)    
    algo = QCAST(topo)

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
