'''
Created on 12 juil. 2019

@author: nboutin
'''
import unittest
from evolve import majorityFunction

class Test(unittest.TestCase):

    def testMajorityFunction(self):
        '''M(A,B,C)=AB+AC+BC'''
        
        inputs = ((False,False,False),(False,False,True),(False,True,False),(False,True,True),
                  (True,False,False),(True,False,True),(True,True,False),(True,True,True))
        
        outputs = (False,False,False,True,False,True,True,True)
        
        for i,o in zip(inputs,outputs):
            a,b,c = i
            r = majorityFunction(a,b,c)
            self.assertEqual(r,o)
            

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()