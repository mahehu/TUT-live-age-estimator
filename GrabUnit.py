# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 16:03:54 2016

@author: agedemo
"""

import time

class GrabUnit():
    
    def __init__(self, frame):
        
        self.timestamp = time.time()

        self.detected = False
        self.ageRecognized = False
        self.genderRecognized = False
        self.expressionRecognized = False

        # Keep track of how many processes are accessing this unit        
        self.processes = 0
                
        self.frame = frame
        
    def getTimeStamp(self):
        
        return self.timestamp
        
    def getFrame(self):
        
        return self.frame

    def acquire(self):
        """ 
        A thread starts to use this resource. Increment the
        process counter.
        """
        
        self.processes += 1
        
    def release(self):
        """ 
        A thread no longer needs this resource. Decrement the
        process counter.
        """
        
        self.processes -= 1

    def isFree(self):
        
        if self.processes == 0:
            return True
        else:
            return False
    
    def getNumProcesses(self):
        
        return self.processes
        
    def getTimeStamp(self):
        
        return self.timestamp
        
    def getAge(self):
        
        return time.time() - self.timestamp

    def setDetected(self):
        
        self.detected = True

    def setAgeRecognized(self):
        
        self.ageRecognized = True

    def setGenderRecognized(self):
        
        self.genderRecognized = True

    def setExpressionRecognized(self):
        
        self.expressionRecognized = True

    def isDetected(self):
        
        return self.detected
        
    def isAgeRecognized(self):
        
        return self.ageRecognized
        
    def isGenderRecognized(self):
        
        return self.genderRecognized
        
    def isExpressionRecognized(self):
        
        return self.expressionRecognized
    
        
