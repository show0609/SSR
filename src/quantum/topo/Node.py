import random

class Node:

    def __init__(self, name, lat, lon, region, station, nQubits, q):
        
        self.id = name
        self.lat = lat # 緯度
        self.lon = lon # 經度
        self.region = region
        self.station = station
        
        self.q = q     # swap 成功機率
        self.remainingQubits = int(nQubits)
        self.nQubits = int(nQubits)
        self.internalLinks = []
        self.neighbors = []
        self.links = []
        self.satellite = [[] for _ in range(10)]
        self.linkNum = 0
        self.neighborNum = 0

    def attemptSwapping(self, l1, l2):  # l1 -> Link, l2 -> Link
        if l1.n1 == self: 
            l1.s1 = True
        else:       
            l1.s2 = True
        
        if l2.n1 == self:    
            l2.s1 = True
        else: 
            l2.s2 = True
        
        b = random.random() <= self.q # swap 是否成功
        if b:
            self.internalLinks.append((l1, l2))
        return b

    def assignIntermediate(self): # for intermediate 
        self.remainingQubits -= 1  

    def clearIntermediate(self):
        self.remainingQubits += 1
    
    def clear(self):
        self.remainingQubits = self.nQubits
        self.internalLinks.clear()