from .Node import Node
import random
import math

class Link:
    
    def __init__(self, node1, node2, l, alpha, station_link: bool, p_sat):
        
        self.n1 = node1
        self.n2 = node2
        self.assigned = False  # 是否分配資源
        self.entangled = False # 是否成功entangle
        self.station_link = station_link
        self.alpha = alpha
        self.lifetime = 0
        self.s1 = False
        self.s2 = False
        self.B = 1
        self.sigmaD = 0.5
        self.p_sat = 1-math.exp(-(self.B*self.B)/(2*self.sigmaD*self.sigmaD)) # 衛星機率
        
        if station_link == True:
            self.p = 0
        else:
            self.p = math.exp(-alpha * l) # entangle 成功機率，l: link長度
        
        
    def connect(self, sat, time):
        if self.station_link == True:
            if self.n1.satellite[sat][time] != -1 and self.n2.satellite[sat][time] != -1:                
                self.accessed = True
                self.p = math.exp(-(self.alpha) * (self.n1.satellite[sat][time] + self.n2.satellite[sat][time])) * self.p_sat * self.p_sat
                return True
            return False
        else:
            return True
                
    def theOtherEndOf(self, n: Node):
        if (self.n1 == n): 
            tmp = self.n2
        elif (self.n2 == n):
            tmp = self.n1
        return tmp
    
    def contains(self, n: Node):  
        return self.n1 == n or self.n2 == n
    
    def swappedAt(self, n: Node): 
        return (self.n1 == n and self.s1 or self.n2 == n and self.s2)
    
    def swappedAtTheOtherEndOf(self, n: Node):  
        return (self.n1 == n and self.s2 or self.n2 == n and self.s1)
    
    def swapped(self):  
        return self.s1 or self.s2
    
    def notSwapped(self):  
        return not self.swapped()

    def assignQubits(self):
        self.assigned = True
        self.n1.remainingQubits -= 1
        self.n2.remainingQubits -= 1   
  
    def clearEntanglement(self):
        preState = self.assigned
        self.s1 = False
        self.s2 = False
        self.assigned = False
        self.entangled = False
        self.lifetime = 0

        for internalLink in self.n1.internalLinks:
            if self in internalLink:
                self.n1.internalLinks.remove(internalLink)

        for internalLink in self.n2.internalLinks:
            if self in internalLink:
                self.n2.internalLinks.remove(internalLink)

        if preState:
            self.n1.remainingQubits += 1
            self.n2.remainingQubits += 1
    
    def clearPhase4Swap(self):
        self.s1 = False
        self.s2 = False
        self.entangled = False
        self.lifetime = 0

        for internalLink in self.n1.internalLinks:
            if self in internalLink:
                self.n1.internalLinks.remove(internalLink)

        for internalLink in self.n2.internalLinks:
            if self in internalLink:
                self.n2.internalLinks.remove(internalLink)
    
    def clear(self):
        self.s1 = False
        self.s2 = False
        self.assigned = False
        self.entangled = False
    
    def tryEntanglement(self): # 有被 assign 的做 entangle
        b = (self.assigned and self.p >= random.random()) or self.entangled
        self.entangled = b
        return b
  
    def assignable(self): 
        return not self.assigned and self.n1.remainingQubits > 0 and self.n2.remainingQubits > 0
