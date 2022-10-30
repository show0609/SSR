import sys
sys.path.append("..")
from AlgorithmBase import AlgorithmBase
from topo.Topo import *
from topo.Topo import Topo 
from topo.Node import Node 
from topo.Link import Link
import random
from random import sample
import math

class Request():
    def __init__(self, src, dst, time):
        self.src = src
        self.dst = dst
        self.pseudo_src = src
        self.time = time
        self.path = []  # [(path, width), (path,width)]
        self.path1 = []
        self.path2 = []
        self.region_path = []
        self.number = -1
        self.cannot_find_path = False
        self.takeTmp = False
        
class VirtualQueue():
    def __init__(self, region1, region2):
        self.region1 = region1
        self.region2 = region2
        self.zero_list = []

class SSR(AlgorithmBase):
    def __init__(self, topo):
        super().__init__(topo)
        self.name = "SSR"
        self.pathsSortedDynamically = [] # [(path, width, req)]
        self.requests = []
        self.totalTime = 0 
        self.totalUsedQubits = 0
        self.regionNum = 6
        self.totalTimeslot = 300
        self.idx = 0
        self.takeTemporary= 0
        self.satTryTable = []
        
        self.region_station = {} # {(region1,region2): (station1,station2)}
        
        # queue
        self.realQueues = {}
        self.virtualQueues = {} # {(region1,region2): VirtualQueue(region1,region2,queue)}
        
        
        for i in range(self.regionNum):
            for j in range(self.regionNum):
                if i == j:
                    continue
                self.virtualQueues[(i,j)] = VirtualQueue(i,j)
                for x in range(self.totalTimeslot): 
                    self.virtualQueues[(i,j)].zero_list.append(x)
                    
        for link in self.topo.station_links[0]:
            self.region_station[(link.n1.region, link.n2.region)]=(link.n1, link.n2)
            self.region_station[(link.n2.region, link.n1.region)]=(link.n2, link.n1)
            
    # large area 
    def find_region_path(self,src,dst):
        # dijkstra
        INF = sys.float_info.max
        path = []
        visited = [False for _ in range(self.regionNum)]
        dis = [INF for _ in range(self.regionNum)]
        parent = [-1 for _ in range(self.regionNum)]
        
        dis[src] = 0
        for _ in range(self.regionNum):
            tmp = INF
            t = -1
            for j in range(self.regionNum):
                if not visited[j] and dis[j] < tmp:
                    tmp = dis[j]
                    t = j
            if t == -1:
                print("no shortest path")
                return []
            visited[t] = True
            for j in range(self.regionNum):
                if visited[j]:
                    continue
                if self.topo.mu[(t,j)] == 0:
                    E_tmp = INF
                else:
                    k = math.ceil(1/self.topo.mu[(t,j)])
                    post = binary_search(self.virtualQueues[(t,j)].zero_list,self.timeslot+dis[t])
                    if post == -1:
                        path = []
                        return path
                    if post+(k-1) >= len(self.virtualQueues[(t,j)].zero_list):
                        path = []
                        return path
                    x = self.virtualQueues[(t,j)].zero_list[post+(k-1)]
                    E_tmp = x - self.timeslot + 1

                if  E_tmp < dis[j]:
                    dis[j] = E_tmp
                    parent[j] = t
        pre = dst
        path.append(dst)
        while(pre != src):
            pre = parent[pre]    
            path.append(pre)
        path.reverse()
        
        self.update_queue(path)
        
        return path
    
    # 做完 dijkstra 更新
    def update_queue(self,path):
        tmp = 0
        for i in range(len(path)-1):
           
            if self.timeslot + tmp >= self.totalTimeslot:
                if len(self.virtualQueues[(path[i],path[(i+1)])].zero_list) == 0:
                    continue
                tmp = self.virtualQueues[(path[i],path[(i+1)])].zero_list.pop()
                if tmp == self.totalTimeslot-1:
                    self.virtualQueues[(path[i+1],path[i])].zero_list.pop()
                else:
                    self.virtualQueues[(path[i],path[i+1])].zero_list.append(tmp)
                continue
            
            if self.topo.mu[(path[i],path[i+1])] == 0:
                continue
                
            k = math.ceil(1/self.topo.mu[(path[i],path[i+1])])
            post = binary_search(self.virtualQueues[(path[i],path[i+1])].zero_list,self.timeslot+tmp)
            if post == -1:
                return
            for _ in range(k):
                if len(self.virtualQueues[(path[i],path[i+1])].zero_list) == 0 or post>=len(self.virtualQueues[(path[i],path[i+1])].zero_list):
                    break
                
                y = self.virtualQueues[(path[i],path[i+1])].zero_list.pop(post)
                z = self.virtualQueues[(path[i+1],path[i])].zero_list.pop(post)  
                if y != z:
                    print("wrong")
                    
            tmp = y - self.timeslot + 1

    # 每個 timeslot 更新
    def initial_queue(self):
        for i in range(self.regionNum):
            for j in range(self.regionNum):
                if i == j:
                    continue
                self.virtualQueues[(i,j)].zero_list.clear()
                for x in range(self.timeslot,self.totalTimeslot): 
                    self.virtualQueues[(i,j)].zero_list.append(x)
        for req in self.requests:
            if len(req.region_path) > 0:
                self.update_queue(req.region_path)
        for i in range(self.regionNum):
            for j in range(self.regionNum):
                if i == j:
                    continue
    
    # small area
    def find_k(self,src,pesudo_src,station):
        trust = []           
        if pesudo_src.region == station.region:
            for node in self.topo.nodes:
                if node.region == src.region and node in self.topo.socialRelationship[src]:
                    trust.append(node)
        else:
            for node in self.topo.nodes:
                if node.region == station.region and node in self.topo.socialRelationship[src]:
                    trust.append(node)
                
        k1 = self.level(station,trust,pesudo_src) 
        
        return k1
         
    def find_local_path(self,src,dst): 
        path = []
        if dst == src:
            return path
        else:
            path = self.bfs(src,dst)
        return path

    def bfs(self,src,dst):
        queue = []
        queue.append(src)
        parent = {node:False for node in self.topo.nodes}
        visited = {node:False for node in self.topo.nodes}
        visited[src] = True
        result = []
        
        if src == dst:
            return result
        
        while(len(queue)!=0):
            now = queue.pop(0)
            if now == dst:
                break
            for neighbor in now.neighbors:
                if visited[neighbor] == False and neighbor.region == now.region:
                    links = self.topo.node_to_link(now,neighbor)
                    for link in links:
                        n1_assignable = False
                        n2_assignable = False
                        if not link.assigned:
                            if link.n1 == src or link.n1 == dst:
                                if link.n1.remainingQubits >= 1:
                                    n1_assignable = True
                            else:
                                if link.n1.remainingQubits >= 2:
                                    n1_assignable = True
                            
                            if link.n2 == src or link.n2 == dst:
                                if link.n2.remainingQubits >= 1:
                                    n2_assignable = True
                            else:
                                if link.n2.remainingQubits >= 2:
                                    n2_assignable = True
                        
                        if n1_assignable and n2_assignable:
                            queue.append(neighbor)
                            visited[neighbor] = True
                            parent[neighbor] = now
                            break
                        
        result.append(dst)
        
        if parent[dst] == False:
            return result
        
        node = parent[dst] 
        while node != src:
            result.append(node)  
            node = parent[node]  
            
        result.append(src)      
        result.reverse()
        return result
        
    # BFS找中繼點 (level最小的可信任點)
    def level(self,selected_station,trust,src):
        queue = []
        result = [] 
        queue.append(selected_station)
        visited = {node:False for node in self.topo.nodes}
        level = {node:-1 for node in self.topo.nodes}
        level[selected_station] = 0
        visited[selected_station] = True
       
        if src.region != selected_station.region:
            level[src] = 1000
        
        while(len(queue)!=0):
            now = queue.pop(0)
            result.append(now)
            for neighbor in now.neighbors:
                if visited[neighbor] == False and neighbor.region == now.region:
                    queue.append(neighbor)
                    visited[neighbor] = True
                    level[neighbor] = level[now]+1
                    
        k1 = src        
        
        for node in result:
            if node in trust and level[node] < level[src] and node.remainingQubits > 2:
                k1 = node
                node.remainingQubits -= 1
                break
        return k1
    
    def prepare(self):
        self.topo.calculate_mu()
        self.topo.create_station_table()
    
    def p2(self):
        print('[', self.name, '] current timeslot:', self.timeslot)
        
        self.initial_queue()
        
        for req in self.requests:
            req.pseudo_src.remainingQubits -= 1
            self.totalUsedQubits += 1

        for (src, dst) in self.srcDstPairs: # 加入新 request
            self.result.totalReqNum += 1
            if src.remainingQubits >= 2: # 資源夠
                self.requests.append(Request(src, dst, self.timeslot))
                src.remainingQubits -= 1
                self.totalUsedQubits += 1
            else:
                self.result.dropNum += 1

        if len(self.requests) > 0:  # 還有request沒做完，時間加1
            self.result.numOfTimeslot += 1
        
        # find region path
        for req in self.requests:
            if len(req.region_path) == 0:   
                path = self.find_region_path(req.src.region, req.dst.region)
                if len(path) > 0:
                    req.region_path = path
                    req.number = self.idx
                    self.idx += 1
            print("pseudo_src:",req.pseudo_src.id, "dst:",req.dst.id,"region_path:", req.region_path)        

        # 需要 entangle 的 station link   
        self.satTryTable = [] 
        for req in self.requests:
            if len(req.region_path) > 1:
                if (req.region_path[0],req.region_path[1]) not in self.satTryTable and (req.region_path[1],req.region_path[0]) not in self.satTryTable:
                    self.satTryTable.append((req.region_path[0],req.region_path[1]))
        
        # station link 分配資源
        if len(self.requests) > 0:
            for link in self.topo.links:
                if link.station_link == True:
                    if (link.n1.region, link.n2.region) in self.satTryTable or (link.n2.region, link.n1.region) in self.satTryTable:
                        link.assigned = True
                    
        # station link 數量
        for link in self.topo.links:
            if link.station_link == True:
                self.satLinkNum += 1
    
        self.requests = sorted(self.requests, key = lambda req : req.number)
        
        for req in self.requests:
            if len(req.region_path) > 1: # 大區
                tmp = self.region_station[(req.region_path[0],req.region_path[1])]
                # 如果衛星有來，就找k2
                if len(self.topo.node_to_link(tmp[0],tmp[1])) != 0:
                    k = self.find_k(req.src,req.pseudo_src,tmp[1])
                    path1 = self.find_local_path(req.pseudo_src,tmp[0])
                    if len(path1) <= 1:
                        continue
                    width = self.topo.widthPhase2(path1)
                    self.assign((path1,width))
                    req.path1.append((path1,width))
                    
                    path2 = self.find_local_path(tmp[1],k)
                    if len(path2) <= 1:
                        continue
                    width = self.topo.widthPhase2(path2) 
                    self.assign((path2,width))
                    req.path2.append((path2,width))
                
                # 如果沒有衛星，就找k1
                else:
                    k = self.find_k(req.src,req.pseudo_src,tmp[0])
                    path = self.find_local_path(req.pseudo_src,k)
                    if len(path) <= 1:
                        continue
                    width = self.topo.widthPhase2(path)
                    self.assign((path,width))
                    req.path.append((path,width))
            
            else: # 小區
                path = self.find_local_path(req.pseudo_src,req.dst)
                if len(path) <= 1:
                        continue
                width = self.topo.widthPhase2(path)
                self.assign((path,width))
                req.path.append((path,width))
                
        # 把資源耗完
        for req in self.requests:
            req.cannot_find_path = False
            
        while(1):
            flag = 0
            for req in self.requests:
                if not req.cannot_find_path:
                    flag = 1
            if flag == 0:
                break
            for req in self.requests:
                if req.cannot_find_path:
                    continue
                if len(req.region_path) > 1:
                    tmp = self.region_station[(req.region_path[0],req.region_path[1])]
                    if self.topo.node_to_link(tmp[0],tmp[1]) in self.topo.links:
                        k = self.find_k(req.src,req.pseudo_src,tmp[1])
                        path1 = self.find_local_path(req.pseudo_src,tmp[0])
                        if len(path1) <= 1:
                            req.cannot_find_path = True
                            continue
                        if self.topo.widthPhase2(path1) == 0:
                            req.cannot_find_path = True
                            continue
                        width = self.topo.widthPhase2(path1) 
                        self.assign((path1,width))
                        req.path1.append((path1,width))
                        
                        path2 = self.find_local_path(tmp[1],k)
                        if len(path2) <= 1:
                            req.cannot_find_path = True
                            continue
                        if self.topo.widthPhase2(path2) == 0:
                            req.cannot_find_path = True
                            continue
                        width = self.topo.widthPhase2(path2) 
                        self.assign((path2,width))
                        req.path2.append((path2,width))
                    
                    else:
                        k = self.find_k(req.src,req.pseudo_src,tmp[0])
                        path = self.find_local_path(req.pseudo_src,k)
                        if len(path)<=1:
                            req.cannot_find_path = True
                            continue
                        if self.topo.widthPhase2(path) == 0:
                            req.cannot_find_path = True
                            continue
                        width = self.topo.widthPhase2(path) 
                        self.assign((path,width))
                        req.path.append((path,width))
                else:
                    path = self.find_local_path(req.pseudo_src,req.dst)
                    if len(path) <= 1:
                        req.cannot_find_path = True
                        continue
                    if self.topo.widthPhase2(path) == 0:
                        req.cannot_find_path = True
                        continue
                    width = self.topo.widthPhase2(path)
                    self.assign((path,width))
                    req.path.append((path,width))
                
        # 計算idleTime    
        for req in self.requests:
            if len(req.path) == 0 and (len(req.path1) == 0 or len(req.path2) == 0):
                self.result.idleTime += 1
        print('[', self.name, '] p2 end')

    def p4(self):
        tmp = self.requests[:]
        for req in tmp:
            for path in req.path: # 小區 [(path,width), (path,width)]
                succLinks = self.swap(path) # [[link,link], [link,link]]
                if len(succLinks) > 0:
                    if path[0][-1] == req.dst: # 送到了真開心
                        self.totalTime += self.timeslot - req.time
                        self.result.successRequestNum += 1
                        self.requests.remove(req)
                    else:
                        req.pseudo_src = path[0][-1]
                        self.takeTemporary += 1
                    break   
            
            path1 = path2 = None
            link1 = link2 = None
            
            for path in req.path1:
                succLinks = self.swap(path)
                if len(succLinks) > 0:
                    path1 = path
                    link1 = [link[-1] for link in succLinks]
                    break
            
            for path in req.path2:
                succLinks = self.swap(path)
                if len(succLinks) > 0:
                    path2 = path
                    link2 = [link[0] for link in succLinks]
                    break

            if path1 != None and path2 != None: # 都成功
                # n1 station station n2
                station1 = path1[0][-1]
                station2 = path2[0][0]
                station_link = []
                
                links = self.topo.node_to_link(station1, station2)
                
                for link in links:
                    if link.entangled == True:
                        station_link.append(link)
                        
                for l1,l2,link in zip(link1,link2,station_link):
                    if link.notSwapped():
                        if station1.attemptSwapping(l1, link) and station2.attemptSwapping(l2, link):
                            self.satSuccNum += 1
                            if path2[0][-1] == req.dst:  # 送到了真開心
                                self.totalTime += self.timeslot - req.time
                                self.result.successRequestNum += 1
                                self.requests.remove(req)
                            else:
                                req.pseudo_src = path2[0][-1]
                                req.region_path.pop(0)
                                self.takeTemporary += 1
                            break
                
        # 結算實驗數據
        remainTime = self.result.dropNum * self.topo.r
        tmp = self.requests[:]
        for req in tmp:
            req.path = []
            req.path1 = []
            req.path2 = []
            remainTime += self.timeslot - req.time
            if self.timeslot - req.time >= self.topo.r: # 過期
                self.requests.remove(req)
                self.result.dropNum += 1

        self.topo.clearAllEntanglements()
        
        self.result.remainRequestPerRound.append(len(self.requests)/self.result.totalReqNum)     
        self.result.waitingTime = (self.totalTime + remainTime) / self.result.totalReqNum
        self.result.usedQubits = self.totalUsedQubits / self.result.totalReqNum
        
        self.result.temporaryRatio = (self.takeTemporary) / self.result.totalReqNum 
        if self.satLinkNum != 0:
            self.result.satSuccRatio = self.satSuccNum / self.satLinkNum

        print('[', self.name, '] p4 end')
        print('[', self.name, '] remain request:', len(self.requests))
        print('[', self.name, '] waiting time:', self.result.waitingTime)
        print('[', self.name, '] idle time:', self.result.idleTime)
        print('----------------------------------')
        
        return self.result
    
    def assign(self, path): # (path,width)
        p, width = path
        for _ in range(0, width):
            for s in range(0, len(p) - 1):
                n1 = p[s]
                n2 = p[s+1]
                for link in n1.links:
                    if link.contains(n2) and (not link.assigned) and link.assignable():
                        self.totalUsedQubits += 2
                        link.assignQubits()
                        break
    
    def swap(self, path):
        p, width = path
        
        if len(p) <= 1:
            return []
        
        succLinks = [[] for _ in range(width)] #[[link,link], [link,link],]
        
        idx = 0
        links = self.topo.node_to_link(p[0],p[1])
        for link in links:
            if link.entangled and (link.n1 == p[0] and not link.s2 or link.n2 == p[0] and not link.s1) and idx < width:
                succLinks[idx].append(link)
                idx += 1
        
        tmp = succLinks[:]
        for links in tmp:
            if len(links) != 1:
                succLinks.remove(links)
        
        for i in range(1,len(p)-1):
            links = self.topo.node_to_link(p[i],p[i+1])
            preLinks = [succlink[-1] for succlink in succLinks]
            nextLinks = []
            curr = p[i]
            next = p[i+1]
            
            w = width
            for link in links:
                if link.entangled and (link.n1 == next and not link.s2 or link.n2 == next and not link.s1) and w > 0:
                    nextLinks.append(link)
                    w -= 1

            if len(preLinks) == 0 or len(nextLinks) == 0:
                break
            
            idx = 0
            for (l1, l2) in zip(preLinks, nextLinks):
                if curr.attemptSwapping(l1, l2):
                    succLinks[idx].append(l2)
                    idx += 1
            
            # 拿掉失敗的
            tmp = succLinks[:]
            for links in tmp:
                if len(links) != i+1:
                    succLinks.remove(links)
        
        return succLinks


def binary_search(zero_list, key):
    if len(zero_list) == 0:
        return -1
    
    low = 0
    high = len(zero_list) - 1
    while(low <= high):
        mid = (low + high) // 2
        if zero_list[mid] == key:
            return mid
        elif zero_list[mid] < key:
            low = mid + 1
        elif zero_list[mid] > key:
            high = mid - 1
    
    if low >= len(zero_list):
        return -1
    else:
        return low  


if __name__ == '__main__':
    
    topo = Topo.generate(0.9, 0.001, 0.0001, 0.7, 0.75) # (self, q, alpha, alpha_sat, p_sat, density)    
    algo = SSR(topo)

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
