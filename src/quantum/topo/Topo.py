import sys
import random
import math
from .Node import Node
from .Link import Link
from random import sample
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from pandas import *
import csv

def topoConnectionChecker(region):  # 用bfs看有沒有落單的點
    visited = {node : False for node in region}
    queue = []
    queue.append(region[0])
    visited[region[0]] = True
    while len(queue)!=0 :
        tmp = queue.pop(0)
        for node in tmp.neighbors :
            if visited[node] == False :
                queue.append(node)
                visited[node] = True
    for node in region :
        if visited[node] == False:
            return False
    return True

class socialGenerator:
    def setTopo(self, topo):
        self.topo = topo
        self.topo.socialRelationship = {node: [] for node in self.topo.nodes} 

    def genSocialRelationship(self):
        print('Generate Social Table ...')
        userNum = 20
        node2user = {}
        self.genSocialNetwork(userNum, self.topo.density)
        users = [i for i in range(userNum)]
        for i in range(len(self.topo.nodes)): # 把node分給user
            if self.topo.nodes[i] in self.topo.stations:
                continue
            user = sample(users, 1)
            node2user[i] = user[0]
        
        # n * n
        for i in range(len(self.topo.nodes)):
            for j in range(i+1, len(self.topo.nodes)):
                if self.topo.nodes[i] in self.topo.stations or self.topo.nodes[j] in self.topo.stations:
                    continue
                user1 = node2user[i]
                user2 = node2user[j]     
                if user1 in self.topo.SN[user2]:
                    n1 = self.topo.nodes[i]
                    n2 = self.topo.nodes[j]
                    self.topo.socialRelationship[n1].append(n2)
                    self.topo.socialRelationship[n2].append(n1)

    def genSocialNetwork(self, userNum, density):
        # n * n
        community1 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 2, 2, 2, 2, 3, 2, 2, 2, 3, 2]  # 0.25
        community2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 0.50
        community3 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  # 0.75
        community4 = [0 for _ in range(20)]                                        # 1.00
        community = {0.25 : community1, 0.50 : community2, 0.75 : community3, 1.00 : community4}

        self.topo.SN = {i: [] for i in range(userNum)}  # user to user
        community = community[density]
        for i in range(userNum):
            for j in range(i, userNum):
                if community[i] == community[j]:
                    self.topo.SN[i].append(j)
                    self.topo.SN[j].append(i)

class Topo:

    def __init__(self, q, alpha, alpha_sat, p_sat, density):
        
        self.nodes = []
        self.stations = []
        self.links = []
        self.station_links = []
        self.station_table = {i:[] for i in range(120)} # {time:[(Node,Node),..],}
        self.edges = [] # (Node, Node)
        self.station_edges = []
        self.node_to_link_table = {}
        self.satNum = 5 # 衛星數量
        self.regionNum = 6
        self.satCycleNum = 120 # 衛星擷取資訊數量
        self.mu = {} # station link 平均流量 {(region1, region2): mu, ...}
        
        self.q = q
        self.alpha = alpha
        self.alpha_sat = alpha_sat
        self.p_sat = p_sat
        self.density = density
        self.r = 120
        
        self.sentinel = Node("sentinel", -1.0, -1.0, -1, False, -1, q) # (self, name, lat, lon, region, station, nQubits, q)
        
        self.SN = {}
        self.socialRelationship = {}    # {n: []}
        self.shortestPathTable = {}     # {(src, dst): (path, weight, p)}
        self.expectTable = {}           # {(path1, path2) : expectRound}
        
        self.createNode()
        self.findNeighbor()
        self.findStation()
        self.createChannel()
        self.createStationChannel()
        self.setSatellite()

        # 記錄link, edge數量 (地面的link)
        self.linkNum = len(self.links)
        self.edgeNum = len(self.edges)
        for node in self.stations:
            node.neighborNum = len(node.neighbors)
            node.linkNum = len(node.links)
        
    def generate(q, alpha, alpha_sat, p_sat, density):
        topo = Topo(q, alpha, alpha_sat, p_sat, density)
        topo.drawMap()
        
        # establish social table
        generator = socialGenerator()
        generator.setTopo(topo)
        generator.genSocialRelationship()

        return topo

    def createNode(self):
        dir = "../topo/data/country/"
        self.regions = [[] for _ in range(6)]
        idx = 0
        for fileName in ["Denmark.txt", "France.txt", "Germany.txt", "Ireland.txt", "Italy.txt", "UK.txt"]:
            file = open(dir + fileName , "r", encoding='utf-8')
            for line in file:
                city = line.split()
                node = Node(city[0], float(city[1]), float(city[2]), idx, False, random.random()*5+10, self.q)
                self.nodes.append(node) # 10-14個qubit
                self.regions[idx].append(node)
            file.close()
            idx += 1
            
    def findNeighbor(self):
        degree = 3
        # edgeDis = 0
        for region in self.regions:
            for node1 in region:
                num = len(node1.neighbors)
                for _ in range(degree-num):
                    min = sys.float_info.max # find min distance
                    minNode = None
                    for node2 in region:
                        if node2 == node1:
                            continue
                        if node2 in node1.neighbors:
                            continue
                        if len(node2.neighbors) > degree:
                            continue
                        if getDistance(node1,node2) < min:
                            min = getDistance(node1, node2)
                            minNode = node2
                    if minNode != None:
                        node1.neighbors.append(minNode)
                        minNode.neighbors.append(node1)
                        self.edges.append((node1,minNode))
                        # edgeDis += min
            if not topoConnectionChecker(region) :
                print("region " + str(self.regions.index(region)) + " not connect")
                exit()

        # neighborNum = 0
        # for node in self.nodes:
        #     neighborNum += len(node.neighbors)
        # print("avg neighborNum:", neighborNum/len(self.nodes))
        # print("avg edge distance:", edgeDis/len(self.edges))
    
    def findStation(self):
        random.seed(3)
        for region in self.regions:
            station = sample(region, 3)
            for node in station:
                node.station = True
                self.stations.append(node)

        # output station information
        # f = open("./output/station.txt", "w")
        # for node in self.stations:
        #     f.write(node.id + " " + str(node.region) + "\n")
        # f.close()
        
    def createChannel(self):
        rand = int(random.random()*5+3) # 3-7個 channel
        for i in self.nodes:
            for j in self.nodes:
                if i == j:
                    continue
                self.node_to_link_table[(i,j)] = []
        for _ in range(0, rand):
            for edge in self.edges: #(Node, Node)
                link = Link(edge[0],edge[1],getDistance(edge[0],edge[1]),self.alpha,False,self.p_sat)
                
                self.links.append(link)
                self.node_to_link_table[(edge[0],edge[1])].append(link)
                self.node_to_link_table[(edge[1],edge[0])].append(link)
                edge[0].links.append(link)
                edge[1].links.append(link)
                
    def createStationChannel(self):
        self.station_links = [[] for _ in range(self.satNum)]
        self.station_edges = []
        for i in self.stations:
            for j in self.stations:
                flag = False
                if i.id == "Vejle" and j.id == "Münster": # 0,2
                    flag = True
                if i.id == "Vejle" and j.id == "Newcastle": # 0,5
                    flag = True
                if i.id == "Saint-Étienne" and j.id == "Bochum": # 1,2
                    flag = True
                if i.id == "Bordeaux" and j.id == "Newbridge": # 1,3
                    flag = True
                if i.id == "Marseille" and j.id == "ReggioEmilia": # 1,4
                    flag = True
                if i.id == "Bordeaux" and j.id == "Liverpool": # 1,5
                    flag = True
                if i.id == "Berlin" and j.id == "Florence": # 2,4
                    flag = True
                if i.id == "Bochum" and j.id == "Sheffield": # 2,5
                    flag = True
                if i.id == "Drogheda" and j.id == "Liverpool": # 3,5
                    flag = True
                
                if flag == True:
                    self.station_edges.append((i,j))
                    for k in range(self.satNum):
                        link=Link(i,j,-1,self.alpha_sat,True,self.p_sat)
                        self.station_links[k].append(link)
                        self.node_to_link_table[(i,j)].append(link)
                        self.node_to_link_table[(j,i)].append(link)
    
    def setSatellite(self):
        dir = "../topo/data/satellite/"
        idx = 0
        for fileName in ["00900_06090842.csv", "00900_09120301.csv", "00902_08050440.csv", "00902_10100032.csv", "44057_10101949.csv", "00900_06090848.csv", "00900_09120250.csv", "00902_08050446.csv", "00902_10100038.csv", "44057_01010600.csv", "44057_10101955.csv"]:
    
            data = read_csv(dir + fileName)
        
            # converting column data to list
            latitude = list(data["lat"])
            longitude = list(data["lon"])
            elevation = list(data["elevation"])
        
            for station in self.stations:
                for lat,lon,elev in zip(latitude,longitude,elevation):
                    tmp = getDistance_stationToSatellite(station,lat,lon,elev)
                    if tmp <= 1500: # 1500內可和衛星連接
                        station.satellite[idx].append(tmp)
                    else:
                        station.satellite[idx].append(-1)
            idx+=1
            
            if idx == self.satNum:
                break
            if self.satCycleNum < len(latitude):
                print("satellite cycle num error")

        if self.satNum != idx:
            print("satellite num error")
        
        # 衛星只能服務距離相加最短的一組
        selecteds = []
        selected = None
        backs = {}
        dis_back = None
        new = []
        for time in range(self.satCycleNum):
            selecteds = []
            backs.clear()
            for i in range(self.satNum):
                min = sys.float_info.max
                for link in self.station_links[0]:
                    if link.n1.satellite[i][time] != -1 and link.n2.satellite[i][time] != -1:
                        if link.n1.satellite[i][time] + link.n2.satellite[i][time] < min:
                            min = link.n1.satellite[i][time] + link.n2.satellite[i][time]
                            selected = (link,i,link.n1.satellite[i][time],link.n2.satellite[i][time])
                if min == sys.float_info.max:
                    continue
                selecteds.append(selected)
                min = sys.float_info.max
                for link in self.station_links[0]:
                    if link.n1.satellite[i][time] != -1 and link.n2.satellite[i][time] != -1:
                        if link.n1.region == selected[0].n1.region and link.n2.region == selected[0].n2.region:
                            continue
                        if link.n1.region == selected[0].n2.region and link.n2.region == selected[0].n1.region:
                            continue
                        if link.n1.satellite[i][time] + link.n2.satellite[i][time] < min:
                            min = link.n1.satellite[i][time] + link.n2.satellite[i][time]
                            dis_back = (link,i,link.n1.satellite[i][time],link.n2.satellite[i][time])
                if min == sys.float_info.max:
                    backs[i] = None
                    continue
                if dis_back == None:
                    backs[i] = None
                    continue
                backs[i] = dis_back       
                  
            new.clear()  
            for i in range(len(selecteds)):
                flag = 0
                for j in range(i,len(selecteds)):
                    if (selecteds[i][0].n1.region == selecteds[j][0].n1.region and selecteds[i][0].n2.region == selecteds[j][0].n2.region) or (selecteds[i][0].n1.region == selecteds[j][0].n2.region and selecteds[i][0].n2.region == selecteds[j][0].n1.region):
                        if selecteds[i][1] in backs.keys():
                            new.append(backs[selecteds[i][1]])
                            flag = 1
                            break
                        elif selecteds[j][1] in backs.keys():
                            new.append(backs[selecteds[j][1]])
                            flag = 1
                            break
                if(flag == 0):
                    new.append(selecteds[i])
            
            for i in range(self.satNum): 
                for link in self.station_links[0]:
                    link.n1.satellite[i][time] = -1
                    link.n2.satellite[i][time] = -1
                    
            for link in selecteds:
                if link == None:
                    continue
                link[0].n1.satellite[link[1]][time] = link[2]
                link[0].n2.satellite[link[1]][time] = link[3]               
            
        # save the satellite information to sat.csv
        F = open("./output/sat.csv", "w")
        writer = csv.writer(F) # create the csv writer
        
        row = ["time", "station1 name", "station1 region", "station2 name", "station2 region", "satellite"]
        writer.writerow(row)
        for time in range(self.satCycleNum):
            for i in range(self.satNum):
                for n1, n2 in self.station_edges:
                    if n1.satellite[i][time] != -1 and n2.satellite[i][time] != -1:
                        row = []
                        row.append(time)
                        row.append(n1.id)
                        row.append(n1.region)
                        row.append(n2.id)
                        row.append(n2.region)
                        row.append(i)
                        writer.writerow(row)
        F.close() 
    
    def drawMap(self):
        # draw the map
        color = ["red", "green", "blue", "yellow", "purple", "black"]
        color_station = ["#651313", "#00d900", "#50bafe", "#84863c", "pink","gray"]
        plt.style.use("bmh")
        plt.title("map") 
        figure(figsize=(8,6),dpi=100)
        plt.xlabel("longitude") # 經度
        plt.ylabel("latitude")  # 緯度
        for node in self.nodes:
            plt.scatter(node.lon, node.lat, s=15, marker="o", color=color[node.region])
        for node in self.stations:
            plt.scatter(node.lon, node.lat, s=15, marker="o", color=color_station[node.region])
        
        plt.savefig("./output/map.jpg")

    
    def trans(self, p):
        return self.L*p / (self.L*p - p + 1)

    def weight(self, p):
        return -1 * math.log(self.trans(p)) 

    def widthPhase2(self, path):
        curMinWidth = min(path[0].remainingQubits, path[-1].remainingQubits)

        # Check min qubits in path
        for i in range(1, len(path) - 1):
            if path[i].remainingQubits / 2 < curMinWidth: # src,dst 除外的node，qubit 要是 min(src,dst) 的兩倍
                curMinWidth = path[i].remainingQubits // 2

        # Check min links in path
        for i in range(0, len(path) - 1):
            n1 = path[i]
            n2 = path[i+1]
            t = 0
            for link in n1.links:
                if link.contains(n2) and not link.assigned:
                    t += 1

            if t < curMinWidth: # 確保 channel 夠
                curMinWidth = t

        return curMinWidth
        
    def shortestPath(self, src, dst, Type, edges = None):
        # Construct state metric (weight) table for edges
        fStateMetric = {}   # {edge: fstate}
        fStateMetric.clear() 
        if edges != None:
            fStateMetric = {edge : getDistance(edge[0], edge[1]) for edge in edges} 
        elif Type == 'Hop' and edges == None: # hop
            fStateMetric = {edge : 1 for edge in self.edges}
        elif Type == "New" and edges == None: # new
            fStateMetric = {edge : self.weight(math.exp(-self.alpha * getDistance(edge[0], edge[1]))) for edge in self.edges}
        elif Type == "Test" and edges == None:  # test
            fStateMetric = {edge : -(math.log(self.q)) + self.weight(math.exp(-self.alpha * getDistance(edge[0], edge[1]))) for edge in self.edges}
        else:   # distance
            fStateMetric = {edge : getDistance(edge[0], edge[1]) for edge in self.edges}

        # Construct neightor & weight table for nodes
        neighborsOf = {node: {} for node in self.nodes}    # {Node: {Node: weight, ...}, ...}
        if edges == None:
            for edge in self.edges:
                n1, n2 = edge
                neighborsOf[n1][n2] = fStateMetric[edge]
                neighborsOf[n2][n1] = fStateMetric[edge]
        else:
            for edge in edges:
                n1, n2 = edge
                neighborsOf[n1][n2] = fStateMetric[edge]
                neighborsOf[n2][n1] = fStateMetric[edge]

        D = {node.id : sys.float_info.max for node in self.nodes} # {int: [int, int, ...], ...}
        q = [] # [(weight, curr, prev)]

        D[src.id] = 0.0
        prevFromSrc = {}   # {cur: prev}

        q.append((D[src.id], src, self.sentinel))
        q = sorted(q, key=lambda q: q[0])

        # Dijkstra 
        while len(q) != 0:
            q = sorted(q, key=lambda q: q[0])
            contain = q.pop(0)
            w, prev = contain[1], contain[2]
            if w in prevFromSrc.keys():
                continue
            prevFromSrc[w] = prev

            # If find the dst return D & path 
            if w == dst:
                path = []
                cur = dst
                while cur != self.sentinel:
                    path.insert(0, cur)
                    cur = prevFromSrc[cur]          
                return (D[dst.id], path)
            
            # Update neighbors of w  
            for neighbor in neighborsOf[w]:
                weight = neighborsOf[w][neighbor]
                newDist = D[w.id] + weight
                oldDist = D[neighbor.id]

                if oldDist > newDist:
                    D[neighbor.id] = newDist
                    q.append((D[neighbor.id], neighbor, w))
        return (sys.float_info.max, [])     
        
    def hopsAway(self, src, dst, Type):
        # print('enter hopsAway')
        path = self.shortestPath(src, dst, Type)
        return len(path[1]) - 1

    def genShortestPathTable(self, Type):
        # n * n
        print('Generate Path Table, Type:', Type)
        for n1 in self.nodes:
            for n2 in self.nodes:
                if n1 != n2:   
                    weight, path = self.shortestPath(n1, n2, Type)       
                    p = self.Pr(path)
                    self.shortestPathTable[(n1, n2)] = (path, weight, p)
                    # print([x.id for x in path])
                    if len(path) == 0:
                        quit()
                else:
                    self.shortestPathTable[(n1, n2)] = ([], 0, 0)

    def genExpectTable(self):
        # n * n * k
        print('Generate Expect Table ...')
        for n1 in self.nodes:
            for k in self.socialRelationship[n1]:
                for n2 in self.nodes:
                    if n1 != n2 and k != n2:
                        self.expectTable[((n1, k), (k, n2))] = self.expectedRound(self.shortestPathTable[(n1, k)][2], self.shortestPathTable[(k, n2)][2])

    def expectedRound(self, p1, p2):
        # 大數法則
        times = 145 # 0
        roundSum = 0

        for _ in range(times):
            roundSum += self.Round(p1, p2)

        # print('expect:', roundSum / times)
        return roundSum / times
    
    def Round(self, p1, p2):
        state = 0 # 0 1 2
        maxRound = 1000
        currentRound = 0
        currentMaintain = 0

        if p1 < (1 / maxRound) or p2 < (1 / maxRound):
            return maxRound

        while state != 2:
            if currentRound >= maxRound:
                break
            currentRound += 1
            if state == 0:
                if random.random() <= p1:
                    state = 1
            elif state == 1:
                currentMaintain += 1
                if currentMaintain > self.L:
                    state = 0
                    currentMaintain = 0
                elif random.random() <= p2:
                    state = 2
        return currentRound

    def Pr(self, path):
        P = 1
        for i in range(len(path) - 1):
            n1 = path[i]
            n2 = path[i+1]
            d = getDistance(n1, n2)
            p = self.trans(math.exp(-self.alpha * d))
            P *= p
   
        return P * (self.q**(len(path) - 2))

    def e(self, path: list, width: int, oldP: list):
        s = len(path) - 1
        P = [0.0 for _ in range(0,width+1)]
        p = [0 for _ in range(0, s+1)]  # Entanglement percentage
        
        for i in range(0, s):
            l = getDistance(path[i], path[i+1])
            p[i+1] = math.exp(-self.alpha * l)

        start = s
        if sum(oldP) == 0:
            for m in range(0, width+1):
                oldP[m] = math.comb(width, m) * math.pow(p[1], m) * math.pow(1-p[1], width-m)
                start = 2
        
        for k in range(start, s+1):
            for i in range(0, width+1):
                exactlyM = math.comb(width, i) *  math.pow(p[k], i) * math.pow(1-p[k], width-i)
                atLeastM = exactlyM

                for j in range(i+1, width+1):
                    atLeastM += (math.comb(width, j) * math.pow(p[k], j) * math.pow(1-p[k], width-j))

                acc = 0
                for j in range(i+1, width+1):
                    acc += oldP[j]
                
                P[i] = oldP[i] * atLeastM + exactlyM * acc
            
            for i in range(0, width+1):
                oldP[i] = P[i]
        
        acc = 0
        for m in range(1, width+1):
            acc += m * oldP[m]
        
        return acc * math.pow(self.q, s-1)
    
    def getEstablishedEntanglements(self, n1: Node, n2: Node):
        stack = []
        stack.append((None, n1)) #Pair[Link, Node]
        result = []

        while stack:
            (incoming, current) = stack.pop() # icoming -> link, current -> node
            if current == n2: # dst
                path = []
                path.append(n2)
                inc = incoming
                while inc.n1 != n1 and inc.n2 != n1:
                    if inc.n1 == path[-1]:
                        prev = inc.n2
                    elif inc.n2 == path[-1]:
                        prev = inc.n1
                        
                    for internalLinks in prev.internalLinks:
                        (l1, l2) = internalLinks
                        if l1 == inc:
                            inc = l2
                            break
                        elif l2 == inc:
                            inc = l1
                            break

                    path.append(prev)

                path.append(n1)
                path.reverse()
                result.append(path)
                break

            outgoingLinks = []
            if incoming is None: # src
                for links in current.links:
                    if links.entangled and not links.swappedAt(current):
                        outgoingLinks.append(links)
            else:
                for internalLinks in current.internalLinks:
                    (l1, l2) = internalLinks
                    if l1 == incoming:
                        outgoingLinks.append(l2)
                    elif l2 == incoming:
                        outgoingLinks.append(l1)
            
            for l in outgoingLinks:
                if l.n1 == current:
                    stack.append((l, l.n2))
                elif l.n2 == current:
                    stack.append((l, l.n1))
        return result

    def clearAllEntanglements(self):
        for link in self.links:
            link.clear()
        for node in self.nodes:
            node.clear()

    def updateLinks(self):
        for link in self.links:
            l = getDistance(link.n1, link.n2)
            link.alpha = self.alpha
            link.p = math.exp(-self.alpha * l)
        for links in self.station_links:
            for link in links:
                link.alpha = self.alpha_sat
    
    def updateNodes(self):
        for node in self.nodes:
            node.q = self.q

    def setAlpha(self, alpha):
        self.alpha = alpha
        self.updateLinks()

    def setAlpha_sat(self, alpha_sat):
        self.alpha_sat = alpha_sat
        self.updateLinks()

    def setQ(self, q):
        self.q = q
        self.updateLinks()
        self.updateNodes()
    
    def setSatNum(self,satNum):
        self.satNum = satNum
        self.createStationChannel()
        self.setSatellite()
    
    def setDensity(self, density):
        self.density = density
        generator = socialGenerator()
        generator.setTopo(self)
        generator.genSocialRelationship()

    def setR(self, r):
        self.r = r
    
    def setP_sat(self,p_sat):
        self.p_sat
        for links in self.station_links:
            for link in links:
                link.p_sat = p_sat

    def update(self,time):
        # return to original
        self.links = self.links[:self.linkNum]
        self.edges = self.edges[:self.edgeNum]
        for node in self.stations:
            node.links = node.links[:node.linkNum]
            node.neighbors = node.neighbors[:node.neighborNum]
                
        # update topo link and edge
        i = 0
        for links in self.station_links:
            for link in links:
                flag = False
                if link.connect(i,time):
                    self.links.append(link) # update link
                    link.n1.links.append(link) # update node link
                    link.n2.links.append(link)
                    flag = True
                if flag:
                    if (link.n1, link.n2) not in self.edges and (link.n2, link.n1) not in self.edges:
                        self.edges.append((link.n1, link.n2)) # update edge
                        link.n1.neighbors.append(link.n2) # update node neighbor
                        link.n2.neighbors.append(link.n1)
            i += 1
        
        # dfs
        connect = []
        for link in self.links:
          if link.station_link and (link.n1.region,link.n2.region) not in connect and (link.n2.region,link.n1.region) not in connect:
              connect.append((link.n1.region,link.n2.region))
              connect.append((link.n2.region,link.n1.region))
        self.not_connect_table = []
        for i in range(self.regionNum):
            for j in range(i,self.regionNum):
              if not self.dfs(i,j,connect):
                  self.not_connect_table.append((i,j))
                
    def node_to_link(self,node1,node2):
        tmp = []
        if node1 in self.stations and node2 in self.stations:
            for link in self.node_to_link_table[(node1,node2)]:
                if link in self.links:
                    tmp.append(link)
            return tmp
        else:
            return self.node_to_link_table[(node1,node2)]
        
    def calculate_mu(self):
        mu_tmp = {}
        # init
        for i in range(6):
            for j in range(6):
                if i == j:
                    continue
                mu_tmp[(i,j)] = 0
        
        # 計算 link 總共出現幾次
        for time in range(self.satCycleNum):
            for i in range(self.satNum):
                for link in self.station_links[0]:
                    if link.connect(i, time):
                        mu_tmp[(link.n1.region,link.n2.region)] += link.p
                        
        # 計算 link 平均每個 timeslot 出現幾次
        for i in range(6):
            for j in range(6):
                if i == j:
                    continue
                self.mu[(i,j)] = (mu_tmp[(i,j)] + mu_tmp[(j,i)]) / self.satCycleNum
        
        # print(self.mu)
        
    def create_station_table(self):
        for time in range(self.satCycleNum):
            for i in range(self.satNum):
                for link in self.station_links[0]:
                    if link.n1.satellite[i][time] != -1 and link.n2.satellite[i][time] != -1:
                        self.station_table[time].append((link.n1,link.n2))
    
    def dfs(self,src,dst,connect):
        stack = []
        visited = set()
        stack.append(src)
        visited.add(src)
        while(len(stack)>0):
          now = stack.pop(0)
          if now == dst:
              return True
          for i in range(self.regionNum):
              if (now,i) in connect and i not in visited:
                  stack.append(i)
                  visited.add(i)
        return False
    
def getDistance(node1,node2):
    latA = node1.lat
    lonA = node1.lon
    latB = node2.lat 
    lonB = node2.lon
    ra = 6378140  # 赤道半徑
    rb = 6356755  # 極半徑
    flatten = (ra - rb) / ra  # Partial rate of the earth
    # change angle to radians
    radLatA = math.radians(latA)
    radLonA = math.radians(lonA)
    radLatB = math.radians(latB)
    radLonB = math.radians(lonB)

    pA = math.atan(rb / ra * math.tan(radLatA))
    pB = math.atan(rb / ra * math.tan(radLatB))
    x = math.acos(math.sin(pA) * math.sin(pB) + math.cos(pA) * math.cos(pB) * math.cos(radLonA - radLonB))
    if(x == 0):
        print(node1.id,node2.id)
    c1 = (math.sin(x) - x) * (math.sin(pA) + math.sin(pB)) ** 2 / math.cos(x / 2) ** 2
    c2 = (math.sin(x) + x) * (math.sin(pA) - math.sin(pB)) ** 2 / math.sin(x / 2) ** 2
    dr = flatten / 8 * (c1 - c2)
    distance = ra * (x + dr)
    distance = round(distance / 1000, 4)
    return distance

def getDistance_stationToSatellite(node1,latB,lonB,elevation):
    latA = node1.lat
    lonA = node1.lon
    ra = 6378140  # 赤道半徑
    rb = 6356755  # 極半徑
    flatten = (ra - rb) / ra  # Partial rate of the earth
    # change angle to radians
    radLatA = math.radians(latA)
    radLonA = math.radians(lonA)
    radLatB = math.radians(latB)
    radLonB = math.radians(lonB)

    pA = math.atan(rb / ra * math.tan(radLatA))
    pB = math.atan(rb / ra * math.tan(radLatB))
    x = math.acos(math.sin(pA) * math.sin(pB) + math.cos(pA) * math.cos(pB) * math.cos(radLonA - radLonB))

    c1 = (math.sin(x) - x) * (math.sin(pA) + math.sin(pB)) ** 2 / math.cos(x / 2) ** 2
    c2 = (math.sin(x) + x) * (math.sin(pA) - math.sin(pB)) ** 2 / math.sin(x / 2) ** 2
    dr = flatten / 8 * (c1 - c2)
    distance = ra * (x + dr)
    distance = round(distance / 1000, 4)
    return (distance**2 + elevation**2)**0.5 
