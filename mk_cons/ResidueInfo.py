# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 09:47:45 2016

@author: XuGang
"""

class Residue:
    def __init__(self, resid, resname):
        
        if resid[-1].isalpha():
            self.resid = -100
        else:
            self.resid = int(resid)       
        self.resname = resname
        self.atoms = {}
        self.hec = 'X'
        
def getResidueData(atomsData):
    
    residuesData = []
    last_resid = None
    
    for atom in atomsData:
        if(atom == atomsData[0]):           
            residue = Residue(atom.resid,atom.resname) 
            residue.atoms[atom.name1] = atom
            
        elif(atom == atomsData[-1]):
            
            if(last_resid == atom.resid):                
                residue.atoms[atom.name1] = atom  
                residuesData.append(residue)
            else:                
                residue = Residue(atom.resid,atom.resname) 
                residue.atoms[atom.name1] = atom
                residuesData.append(residue)
                
        else:
            if(last_resid == atom.resid):                
                residue.atoms[atom.name1] = atom          
            else:
                residuesData.append(residue)
                residue = Residue(atom.resid,atom.resname) 
                residue.atoms[atom.name1] = atom                  
        
        last_resid = atom.resid
    
    return residuesData

def singleResname(AA):
    if(len(AA) == 1):
        return AA
    else:
        if(AA in ['GLY','AGLY']):
            return "G"
        elif(AA in ['ALA','AALA']):
            return "A"
        elif(AA in ['SER','ASER']):
            return "S"
        elif(AA in ['CYS','ACYS']):
            return "C"
        elif(AA in ['VAL','AVAL']):
            return "V"
        elif(AA in ['ILE','AILE']):
            return "I"
        elif(AA in ['LEU','ALEU']):
            return "L"
        elif(AA in ['THR','ATHR']):
            return "T"
        elif(AA in ['ARG','AARG']):
            return "R"
        elif(AA in ['LYS','ALYS']):
            return "K"
        elif(AA in ['ASP','AASP']):
            return "D"
        elif(AA in ['GLU','AGLU']):
            return "E"
        elif(AA in ['ASN','AASN']):
            return "N"
        elif(AA in ['GLN','AGLN']):
            return "Q"
        elif(AA in ['MET','AMET']):
            return "M"
        elif(AA in ['HIS','AHIS']):
            return "H"
        elif(AA in ['PRO','APRO']):
            return "P"
        elif(AA in ['PHE','APHE']):
            return "F"
        elif(AA in ['TYR','ATYR']):
            return "Y"
        elif(AA in ['TRP','ATRP']):
            return "W"
        elif(AA[:2] == 'MS' or AA[:3] == 'AMS'):
            return "M"
        elif(AA[:2] == 'CS' or AA[:3] == 'ACS'):
            return "C"            
        else:
            return "X"