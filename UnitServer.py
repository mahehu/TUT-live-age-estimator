# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 15:36:48 2016

@author: agedemo
"""

import DetectionThread
import RecognitionThread
import threading

class UnitServer():
    
    def __init__(self, maxUnits = 4):
        
        self.maxUnits = maxUnits
        self.units = []
        self.mutex = threading.Lock()
        
    def getUnit(self, caller, timestamp = None):
        
        self.mutex.acquire()

        # Detection thread will receive the newest undetected frame

        unit = None
        
        if timestamp is not None:
            
            for f in self.units:
                if f.getTimeStamp() == timestamp:
                    unit = f

        else:

            if isinstance(caller, DetectionThread.DetectionThread):
        
                validUnits = [f for f in self.units if f.isDetected() == False]
    
                if len(validUnits) == 0:
                    unit = None
                else:
                    unit = validUnits[-1]
                    unit.acquire()
                    unit.setDetected()
                    
                    #print("Locking %.6f for %s" % (unit.getTimeStamp(), str(type(caller))))

            # Age thread will receive the newest detected frame with age rec not done
    
            if isinstance(caller, RecognitionThread.RecognitionThread):
                
                validUnits = [f for f in self.units if 
                              f.isDetected() == True and
                              f.isAgeRecognized() == False]
    
                if len(validUnits) == 0:
                    unit = None
                else:
                    unit = validUnits[-1]
                    unit.acquire()
                    unit.setDetected()
                    
                  #  print("Locking %.6f for %s" % (unit.getTimeStamp(), str(type(caller))))


        self.mutex.release()

        return unit
        
    def putUnit(self, unit):
        
        self.mutex.acquire()

        #print "Adding %.6f" % (unit.getTimeStamp())
        
        if len(self.units) >= self.maxUnits:
            # Attempt to remove oldest unit
            if self.units[0].isFree():
                self.units.pop(0)
                
        if len(self.units) < self.maxUnits:
            self.units.append(unit)
        else:
            #print("Unable to add new unit.")
            pass
            
#        for i,unit in enumerate(self.units):
#            print("Unit %.6f: numProcesses: %d" % (unit.getTimeStamp(), unit.getNumProcesses()))
#        print "=" * 5
        
        self.mutex.release()
        