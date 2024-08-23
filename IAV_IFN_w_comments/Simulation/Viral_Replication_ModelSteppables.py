'''
Hi, my name is Pedro, I wrote this code and I will help you read through it.
My comments will be inside tripple quotes, while the hashtaged comments are dedicated
to tecnical information
'''


from cc3d.cpp.PlayerPython import * 
from cc3d import CompuCellSetup
from cc3d.core.PySteppables import *
import numpy as np
import random
import os


'''
Here I define various parameters of the simulation
'''
# define 1 virus field in the center / start with 1 infected cell in the center
# 1 pixel = 1.26 microns^2
# cell diameter = 10um, or 7.94 pixels
pixel_size = 1.26
min_mcs = 10.0  # min/mcs
hours_mcs = min_mcs / 60.0  # hours/mcs
days_mcs = min_mcs / 1440.0  # day/mcs
hours_to_simulate = 80.0  # 10 in the original model
secretion_rate_V = 71.6/6.0 # 71.6 viruses released per hour (converted to 10 minutes)
secretion_rate_IFN = 1 # IFN molecules released every MCS (every 10 minutes)
IRF7_secretion_multiplier = 10 # IFN molecules released every MCS (every 10 minutes)
IFN_stimulation_threshold = 0.1 # this is to avoid activating network with negligible IFN concentration values, saving time
endocytosis_prob = 1 # 1% chance of endocytosis every MCS
death_prob = 0.033 # 3.3% chance of death every MCS (30 steps to die, 5h of virus production before death)
MOI = 0 # either 1, high (all cells start out infected), or 0d, low (approx 1% of cells start out infected) or test (1 infected cell in the middle)
IFNn = 4
IFNhm = 1
STAT_del = 1 # 0 or 1
NS1_del = 1 # 0 or 1
'''
The MOI, STAT_del and NS1_del are binary variables that define which experiment I am performing.
Below, I define the probability of a virus to carry a defect on its genome. These proabilities
were estimated based on the ratio of IFN inducing viruses to fully infectious viruses, and then
extrapolated to other parts of the viral genome based on the size of the corresponding protein
aminoacid chain.
'''

# GDP = genome defect probabilities
GDP = {"PB1g":0.094774786,
       "PB2g":0.568645964,
       "PAg":0.547128424,
       "NS1g":0.224665344,
       "NPg":0.424879137,
       "HAg":0.464785577,
       "NAg":0.402172369,
       "M1g":0.243308949,
       "M2g":0.101758182,
       "NEPg":0.125294801}
       
'''
Below, I define the MaBoSS model. A full explanation of this network does not fit here.
'''

virus_IFN_mbs = """

// viral replication phases
// TODO : make innate immunity defenses imperfect to allow disease spread

node vRNPn
{
    logic = $internalized_virus > 0;
    rate_up = @logic ? $vRNP_import : 0.0;
    rate_down = @logic ? 0.0 : 0.0;
}

node Producing
{
    logic = vRNPn && RC;
    rate_up = @logic ? $virus_replication : 0.0;
    rate_down = @logic ? 0.0 : $virus_decay;
}

node Releasing
{
    logic = Producing && HA && NA && M1 && M2 && NEP;
    rate_up = @logic ? 1000.0 : 0.0; // instantaneous transition
    rate_down = @logic ? 0.0 : 1.0;
}

// viral proteins, replication capacity

node PB1
{
    logic = $PB1g > 0 && vRNPn && !(OAS || PKR);
    rate_up = @logic ? $protein_production : 0.0;
    rate_down = @logic ? 0.0 : $protein_degradation;
}

node PB2
{
    logic = $PB2g > 0 && vRNPn && !(OAS || PKR);
    rate_up = @logic ? $protein_production : 0.0;
    rate_down = @logic ? 0.0 : $protein_degradation;
}

node PA
{
    logic = $PAg > 0 && vRNPn && !(OAS || PKR);
    rate_up = @logic ? $protein_production : 0.0;
    rate_down = @logic ? 0.0 : $protein_degradation;
}

node NP
{
    logic = $NPg > 0 && vRNPn && !(OAS || PKR);
    rate_up = @logic ? $protein_production : 0.0;
    rate_down = @logic ? 0.0 : $protein_degradation;
}

node RC
{
    logic = PB1 && PB2 && PA && NP;
    rate_up = @logic ? $RC_formation : 0.0;
    rate_down = @logic ? 0.0 : $RC_degradation;
}

node NS1
{
    logic = $NS1g > 0 && vRNPn && !(OAS || PKR); //ns1del
    rate_up = @logic*!($NS1_del) ? $protein_production : 0.0;
    rate_down = @logic*!($NS1_del) ? 0.0 : $protein_degradation;
}

node HA
{
    logic = $HAg > 0 && vRNPn && !(OAS || PKR);
    rate_up = @logic ? $protein_production : 0.0;
    rate_down = @logic ? 0.0 : $protein_degradation;
}

node NA
{
    logic = $NAg > 0 && vRNPn && !(OAS || PKR);
    rate_up = @logic ? $protein_production : 0.0;
    rate_down = @logic ? 0.0 : $protein_degradation;
}

node M1
{
    logic = $M1g > 0 && vRNPn && !(OAS || PKR);
    rate_up = @logic ? $protein_production : 0.0;
    rate_down = @logic ? 0.0 : $protein_degradation;
}

node M2
{
    logic = $M2g > 0 && vRNPn && !(OAS || PKR);
    rate_up = @logic ? $protein_production : 0.0;
    rate_down = @logic ? 0.0 : $protein_degradation;
}

node NEP
{
    logic = $NEPg > 0 && vRNPn && !(OAS || PKR);
    rate_up = @logic ? $protein_production : 0.0;
    rate_down = @logic ? 0.0 : $protein_degradation;
}

// Sensing viruses, producing IFN

node RIGI
{
    logic = 1; // TODO : NS1 downregulates RIGI
    rate_up = @logic ? 0.0 : 0.0;
    rate_down = @logic ? 0.0 : 0.0;
}

node IRF3
{
    logic = 1; // TODO : RIGI eventually goes low when both vRNPn and RIGI are activated
    rate_up = @logic ? 0.0 : 0.0;
    rate_down = @logic ? 0.0 : 0.0;
}

node IRF7
{
    logic = STATp && !(NS1); // TODO : STATp activation should upregulate this
    rate_up = @logic ? $mRNA_upregulation : 0.0;
    rate_down = @logic ? 0.0 : $mRNA_degradation;
}

node TLR7
{
    logic = 1; // TODO : TLR7 goes to 1 immediately after internalized virus and decays with vRNP_import rate, block TLR7 pathway when endocytosis is blocked
    rate_up = @logic ? 0.0 : 0.0;
    rate_down = @logic ? 0.0 : 0.0;
}

node IFNmRNA 
{
    logic = (((IRF3 || IRF7) && (RIGI && vRNPn)) || (TLR7 && IRF7 && $internalized_virus)) and ! NS1; // TODO : NS1 should not allow this to go up, but also not pull it down
    // TODO : TLR7 pathway being active all the time could make NS1 inhibition of RIGI unimportant
    rate_up = @logic ? $mRNA_upregulation : 0.0;
    rate_down = @logic ? 0.0 : $mRNA_degradation; // TODO : maybe NS1 does bring this down by shutting down host transcription
}

// IFN stimulation and ISGs

node STATp
{
    logic = $IFNc > 0;
    rate_up = @logic ? $STATp_activation*$STATp_activation_rate : 0.0;
    rate_down = @logic ? 0.0 : 0.0;
}

node IFITM
{
    logic = STATp && !(NS1);
    rate_up = @logic ? $mRNA_upregulation : 0.0;
    rate_down = @logic ? 0.0 : $mRNA_degradation;
}

node OAS
{
    logic = STATp && !(NS1);
    rate_up = @logic ? $mRNA_upregulation : 0.0;
    rate_down = @logic ? 0.0 : $mRNA_degradation;
}

node PKR
{
    logic = STATp && !(NS1);
    rate_up = @logic ? $mRNA_upregulation : 0.0;
    rate_down = @logic ? 0.0 : $mRNA_degradation;
}

"""

'''
Below, I define the initial levels of key nodes in the network and
rate constants.
'''

virus_IFN_mbs_config = """

$PB1g = 0;
$PB2g = 0;
$PAg = 0;
$NPg = 0;
$NS1g = 0;
$HAg = 0;
$NAg = 0;
$M1g = 0;
$M2g = 0;
$NEPg = 0;

$vRNP_import = 1.0/(1.0*6.0); //1h
$virus_replication = 1.0/(5.0*6.0); //5h
$virus_decay = 1.0/(10.0*6.0); //10h
$protein_production = 1.0/(0.1*6.0); //6min
$protein_degradation = 1.0/(1.0*6.0); //1h
$RC_formation = 1.0/(0.0001*6.0); //instantaneous
$RC_degradation = 1.0/(1.0*6.0); //1h
$mRNA_upregulation = 1.0/(2.0*6.0); //1h
$mRNA_degradation = 1.0/(10.0*6.0); //10h

$STATp_activation = 0.0;
$STATp_activation_rate = 1.0/(0.001*6.0); //instantaneous // TODO : isnt this instantaneous?
$internalized_virus = 0;
$IFNc = 0.0;
$NS1_del = 0.0;
//$IFNh = 10.0;
//$IFNn = 4.0;

RIGI.istate = 1;
IRF3.istate = 1;
TLR7.istate = 1;
IRF7.istate = 0;
IFNmRNA.istate = 0;
PB1.istate = 0;
PB2.istate = 0;
PA.istate = 0;
NP.istate = 0;
NS1.istate = 0;
HA.istate = 0;
NA.istate = 0;
M1.istate = 0;
M2.istate = 0;
NEP.istate = 0;
RC.istate = 0;
STATp.istate = 0;
IFITM.istate = 0;
OAS.istate = 0;
PKR.istate = 0;
vRNPn.istate = 0;
Producing.istate = 0;
Releasing.istate = 0;

"""

class Viral_Replication_ModelSteppable(SteppableBasePy):

    def __init__(self, frequency=1):
        
        '''
        this chunk here is tecnical code to allow visualization of itracellular network
        nodes in the Player windows.
        '''
        #fields
        SteppableBasePy.__init__(self,frequency)
        self.create_scalar_field_cell_level_py("vRNPn")
        self.create_scalar_field_cell_level_py("Producing")
        self.create_scalar_field_cell_level_py("Releasing")
        self.create_scalar_field_cell_level_py("NS1")
        self.create_scalar_field_cell_level_py("IRF3")
        self.create_scalar_field_cell_level_py("RIGI")
        self.create_scalar_field_cell_level_py("IFNmRNA")
        self.create_scalar_field_cell_level_py("STATp")
        self.create_scalar_field_cell_level_py("IFITM")
        self.create_scalar_field_cell_level_py("IRF7")

    def start(self):

        '''
        this chunk here is tecnical code to allow visualization of itracellular network
        nodes in the Player windows.
        '''
        #fields
        self.vRNPn = self.field.vRNPn
        self.Producing = self.field.Producing
        self.Releasing = self.field.Releasing
        self.NS1 = self.field.NS1
        self.IRF3 = self.field.IRF3
        self.RIGI = self.field.RIGI
        self.IFNmRNA = self.field.IFNmRNA
        self.STATp = self.field.STATp
        self.IFITM = self.field.IFITM
        self.IRF7 = self.field.IRF7
        
        self.secretor_virus = self.get_field_secretor("virus")
        self.secretor_IFN = self.get_field_secretor("IFN")
        
        '''
        Here I loop over all cells, define their volume constraints, and infect them based on
        the mode of infection: high MOI, low MOI, or test. Test is just one cell infected in
        the middle of the lattice.
        '''
        for cell in self.cell_list:
            cell.targetVolume = 50
            cell.lambdaVolume = 10
            
            # pre infection, starting with all infected by PFUs
            if MOI:
                self.infect_cell_PFU(cell) #ns1del
            # GP = genome presence
            # cell.dict["GP"] = {"PB1g":0,"PB2g":0,"PAg":0,"NS1g":0,"NPg":0,"HAg":0,"NAg":0,"M1g":0,"M2g":0,"NEPg":0}
            
            # the procedure below is not working after implementation of protein integrity
            if not MOI and np.random.uniform() < 0.01:
                self.infect_cell_PFU(cell) # ns1del
                self.secretor_virus.secreteInsideCell(cell, -1./cell.volume)
                
        # one infected cell in the middle
        if MOI=="test":
            cell = self.cell_field[self.dim.x/2, self.dim.y/2, 0]
            self.infect_cell_PFU(cell)
            
        '''
        This sets output files
        '''
        rs = self.simulator.getRandomSeed()
        output_folder = r'C:\Trabalhos\APL Project\Viral_Replication_Model_ProteinDefect_MaBoSS\output'
        vknockout = 'WT' if NS1_del==0 else 'NS1del'
        cknockout = '' if STAT_del==0 else '_STATKO'
        MOI_ = 'high' if MOI else 'low'
        file_name = f'{vknockout}_{MOI_}MOI{cknockout}_rs{rs}.txt'
        self.output_path= os.path.join(output_folder, file_name)
        #with open(self.output_path, 'a') as file:
            #file.write(f"time \t totIFNc \t dead \t totVirus \t STATp \n")
        
    def step(self, mcs):
        """
        Called every frequency MCS while executing the simulation
        
        :param mcs: current Monte Carlo step
        """
        '''
        below, I am collecting data from the simulation to write in a file. The actual writing
        part is not active.
        '''
        # TODO : model medium wash
        dead_cells = len(self.cell_list_by_type(self.D))
        total_cells = len(self.cell_list)
        fieldIFN = self.field.IFN
        total_IFN = 0
        secretor_i = self.get_field_secretor("IFN")
        total_IFN = secretor_i.totalFieldIntegral()
        secretor_v = self.get_field_secretor("virus")
        total_virus = secretor_v.totalFieldIntegral()
        total_IFN_concentration = total_IFN/(self.dim.x*self.dim.y*pixel_size**3)
        # time in MCS, total IFN in micromols/microns^3, total virus per cell
        #print(f"{mcs/6.}\t {total_IFN_concentration} \t {len(self.cell_list_by_type(self.D))/len(self.cell_list)}\t {total_virus/len(self.cell_list)}")
        
        #with open(self.output_path, 'a') as file:
            #file.write(f"{mcs/6.}\t {total_IFN_concentration} \t {len(self.cell_list_by_type(self.D))/len(self.cell_list)}\t {total_virus/len(self.cell_list)}\n")
        
        self.timestep_maboss()
        
        '''
        this chunk here is tecnical code to allow visualization of itracellular network
        nodes in the Player windows.
        '''
        #fields
        self.vRNPn.clear() 
        self.Producing.clear() 
        self.Releasing.clear() 
        self.NS1.clear() 
        self.IRF3.clear()  
        self.RIGI.clear() 
        self.IFNmRNA.clear() 
        self.STATp.clear() 
        self.IFITM.clear() 
        self.IRF7.clear() 
        
        STATp_cell = 0
        
        #loop over infected cells
        for cell in self.cell_list_by_type(self.I):
            
            '''
            this chunk here is tecnical code to allow visualization of itracellular network
            nodes in the Player windows.
            '''
            #fields
            vRNPn = int(cell.maboss.VModel['vRNPn'].state) 
            Producing = int(cell.maboss.VModel['Producing'].state) 
            Releasing = int(cell.maboss.VModel['Releasing'].state)
            NS1 = int(cell.maboss.VModel['NS1'].state)
            IRF3 = int(cell.maboss.VModel['IRF3'].state)
            RIGI = int(cell.maboss.VModel['RIGI'].state)
            IFNmRNA = int(cell.maboss.VModel['IFNmRNA'].state)
            STATp = int(cell.maboss.VModel['STATp'].state)
            IFITM = int(cell.maboss.VModel['IFITM'].state)
            PKR = int(cell.maboss.VModel['PKR'].state)
            IRF7 = int(cell.maboss.VModel['IRF7'].state)
            
            #fields
            self.vRNPn[cell] = vRNPn 
            self.Producing[cell] = Producing 
            self.Releasing[cell] = Releasing 
            self.NS1[cell] = NS1 
            self.IRF3[cell] = IRF3 
            self.RIGI[cell] = RIGI 
            self.IFNmRNA[cell] = IFNmRNA 
            self.STATp[cell] = STATp 
            self.IFITM[cell] = IFITM 
            self.IRF7[cell] = IRF7 
            
            '''
            This code reads the amount of IFN on a cell and calculates the rate of STAT activation
            using a Hill function.
            '''
            IFNc = self.secretor_IFN.amountSeenByCell(cell)
            cell.maboss.VModel.network.symbol_table["IFNc"] = IFNc
            cell.maboss.VModel.network.symbol_table["STATp_activation"] = (1-STAT_del)*IFNc**IFNn / (IFNc**IFNn+IFNhm**IFNn)
            #print(cell.maboss.VModel.network.symbol_table["STATp_activation"])
            
            '''
            Below, endocytosis, virus release and IFN secretion are regulated based on the cell
            phenotype and environmental conditions. Cell death depends on PKR or a phenotype of
            virus production.
            '''
            viral_exposure = self.secretor_virus.amountSeenByCell(cell)
            # virus endocytosis
            # TODO increase chance of endocytosis depending on viral concentration or endocytose more than 1 virus
            if viral_exposure > 1 and np.random.uniform() < endocytosis_prob and not Releasing:
                self.infect_cell(cell)
                # no extra internalization because network is boolean
                self.secretor_virus.secreteInsideCell(cell, -1./cell.volume)
            # virus secretion
            if Releasing:
                secreted_virus = Producing * secretion_rate_V
                self.secretor_virus.secreteInsideCell(cell, secreted_virus/cell.volume)
                # no reduction of internal virus on secretion because network is boolean
            # IFN secretion
            if IFNmRNA:
                # TODO : increase interferon production if IRF7 is active
                # secreted_IFN = secretion_rate_IFN # OLD
                secreted_IFN = secretion_rate_IFN * (1 + IRF7 * IRF7_secretion_multiplier)
                self.secretor_IFN.secreteInsideCell(cell, secreted_IFN/cell.volume)
            # cell death
            # TODO find death probability that lead to an average of 1000 virions released per cell
            if (Producing or (PKR and not NS1)) and np.random.uniform() < death_prob:
                self.delete_maboss_from_cell(cell=cell, model_name="VModel")
                cell.type = self.D
                
            if STATp:
                STATp_cell += 1
                
        #loop over uninfected cells
        for cell in self.cell_list_by_type(self.U):
            '''
            Here, if uninfected cells endocytose a virus, they become infected. They can also become 
            exposed cells depending on the IFN concentrations
            '''
            # virus endocytosis
            # TODO increase chance of endocytosis depending on viral concentration or endocytose more than 1 virus
            viral_exposure = self.secretor_virus.amountSeenByCell(cell)
            if viral_exposure > 1 and np.random.uniform() < endocytosis_prob:
                self.infect_cell(cell)
                self.secretor_virus.secreteInsideCell(cell, -1./cell.volume)
            
            # IFN stimulation
            IFN_exposure = self.secretor_IFN.amountSeenByCell(cell)
            if IFN_exposure > IFN_stimulation_threshold:
                self.stimulate_cell(cell)
                
        for cell in self.cell_list_by_type(self.E):
            
            '''
            More technical stuff
            '''
            #fields
            vRNPn = int(cell.maboss.VModel['vRNPn'].state) 
            Producing = int(cell.maboss.VModel['Producing'].state) 
            Releasing = int(cell.maboss.VModel['Releasing'].state)
            NS1 = int(cell.maboss.VModel['NS1'].state)
            IRF3 = int(cell.maboss.VModel['IRF3'].state)
            RIGI = int(cell.maboss.VModel['RIGI'].state)
            IFNmRNA = int(cell.maboss.VModel['IFNmRNA'].state)
            STATp = int(cell.maboss.VModel['STATp'].state)
            IFITM = int(cell.maboss.VModel['IFITM'].state)
            IRF7 = int(cell.maboss.VModel['IRF7'].state)
            #fields
            self.vRNPn[cell] = vRNPn 
            self.Producing[cell] = Producing 
            self.Releasing[cell] = Releasing 
            self.NS1[cell] = NS1 
            self.IRF3[cell] = IRF3 
            self.RIGI[cell] = RIGI 
            self.IFNmRNA[cell] = IFNmRNA 
            self.STATp[cell] = STATp 
            self.IFITM[cell] = IFITM 
            self.IRF7[cell] = IRF7 
            
            '''
            Calculating the STAT activation rate
            '''
            IFNc = self.secretor_IFN.amountSeenByCell(cell)
            cell.maboss.VModel.network.symbol_table["IFNc"] = IFNc
            cell.maboss.VModel.network.symbol_table["STATp_activation"] = (1-STAT_del)*IFNc**IFNn / (IFNc**IFNn+IFNhm**IFNn)
            
            '''
            Virus endocytosis. IFITM reduces endocytosis probability by a factor of 10.
            '''
            viral_exposure = self.secretor_virus.amountSeenByCell(cell)
            # TODO : How exactly does IFITM reduce virus endocytosis?
            if viral_exposure > 1 and np.random.uniform() < endocytosis_prob * (1 - 0.9*IFITM):
                self.infect_cell(cell)
                self.secretor_virus.secreteInsideCell(cell, -1./cell.volume)
                
            if STATp:
                STATp_cell += 1
                
        '''
        Output data.
        '''
        # TODO : model medium wash
        dead_cells = len(self.cell_list_by_type(self.D))
        total_cells = len(self.cell_list)
        fieldIFN = self.field.IFN
        total_IFN = 0
        secretor_i = self.get_field_secretor("IFN")
        total_IFN = secretor_i.totalFieldIntegral()
        secretor_v = self.get_field_secretor("virus")
        total_virus = secretor_v.totalFieldIntegral()
        total_IFN_concentration = total_IFN/(self.dim.x*self.dim.y*pixel_size**3)
        # time in MCS, total IFN in micromols/microns^3, total virus per cell
        #print(f"{mcs/6.}\t {total_IFN_concentration} \t {len(self.cell_list_by_type(self.D))/len(self.cell_list)}\t {total_virus/len(self.cell_list)}")
        
        #with open(self.output_path, 'a') as file:
            #file.write(f"{mcs/6.}\t {total_IFN_concentration} \t {len(self.cell_list_by_type(self.D))/len(self.cell_list)}\t {total_virus/len(self.cell_list)}\t {STATp_cell/len(self.cell_list)}\n")
        
            
    def finish(self):
        """
        Called after the last MCS to wrap up the simulation
        """

    def on_stop(self):
        """
        Called if the simulation is stopped before the last MCS
        """
        
    def stimulate_cell(self, cell):
        if cell.type == self.U:
            cell.type = self.E
            self.add_maboss_to_cell(cell=cell,model_name="VModel",
                                    bnd_str=virus_IFN_mbs,
                                    cfg_str=virus_IFN_mbs_config,
                                    time_step=1.0,time_tick=1.0,
                                    seed=random.randint(0, int(1E9)))
    def infect_cell(self, cell):
        '''
        This function solves the viral genome integrity based on defects probabilities when
        a cell gets infected. If a cell gets infected multiple times, competent genome pieces
        add up and might combine into a fully functional viral genome.
        '''
        if cell.type == self.U:
            self.add_maboss_to_cell(cell=cell,model_name="VModel",
                                    bnd_str=virus_IFN_mbs,
                                    cfg_str=virus_IFN_mbs_config,
                                    time_step=1.0,time_tick=1.0,
                                    seed=random.randint(0, int(1E9)))
        cell.type = self.I
        cell.maboss.VModel.network.symbol_table["internalized_virus"] = 1
        cell.maboss.VModel.network.symbol_table["PB1g"] += np.random.uniform()>GDP["PB1g"]
        cell.maboss.VModel.network.symbol_table["PB2g"] += np.random.uniform()>GDP["PB2g"]
        cell.maboss.VModel.network.symbol_table["PAg"] += np.random.uniform()>GDP["PAg"]
        cell.maboss.VModel.network.symbol_table["NPg"] += np.random.uniform()>GDP["NPg"]
        cell.maboss.VModel.network.symbol_table["NS1g"] += np.random.uniform()>GDP["NS1g"]
        cell.maboss.VModel.network.symbol_table["HAg"] += np.random.uniform()>GDP["HAg"]
        cell.maboss.VModel.network.symbol_table["NAg"] += np.random.uniform()>GDP["NAg"]
        cell.maboss.VModel.network.symbol_table["M1g"] += np.random.uniform()>GDP["M1g"]
        cell.maboss.VModel.network.symbol_table["M2g"] += np.random.uniform()>GDP["M2g"]
        cell.maboss.VModel.network.symbol_table["NEPg"] += np.random.uniform()>GDP["NEPg"]
        cell.maboss.VModel.network.symbol_table["NS1_del"] = NS1_del
    
        # if not np.prod(list(cell.dict["GP"].values())):
            # # estimate viral protein defects and update viral protein integrity inside the cell
            # cell.dict["GP"] = {key : cell.dict["GP"].get(key,0)+1*(np.random.uniform()>GDP.get(key,0))
                                # for key in set(cell.dict["GP"])}
        # if np.prod(list(cell.dict["GP"].values())):
            # cell.maboss.VModel.network.symbol_table['internalized_virus'] = 1
            
    def infect_cell_PFU(self, cell):
        '''
        This function solves an infection with a fully functional virus. I used this in the
        start function to infect the initial cells. PFU stands for plaque forming unit, a virus
        capable of infecting a culture, i.e., contains all essential viral proteins encoded
        in its genome.
        '''
        self.add_maboss_to_cell(cell=cell,model_name="VModel",
                                    bnd_str=virus_IFN_mbs,
                                    cfg_str=virus_IFN_mbs_config,
                                    time_step=1.0,time_tick=1.0,
                                    seed=random.randint(0, int(1E9)))
        cell.type = self.I
        cell.maboss.VModel.network.symbol_table["internalized_virus"] = 1
        cell.maboss.VModel.network.symbol_table["PB1g"] += 1
        cell.maboss.VModel.network.symbol_table["PB2g"] += 1
        cell.maboss.VModel.network.symbol_table["PAg"] += 1
        cell.maboss.VModel.network.symbol_table["NPg"] += 1
        cell.maboss.VModel.network.symbol_table["NS1g"] += 1
        cell.maboss.VModel.network.symbol_table["HAg"] += 1
        cell.maboss.VModel.network.symbol_table["NAg"] += 1
        cell.maboss.VModel.network.symbol_table["M1g"] += 1
        cell.maboss.VModel.network.symbol_table["M2g"] += 1
        cell.maboss.VModel.network.symbol_table["NEPg"] += 1
        cell.maboss.VModel.network.symbol_table["NS1_del"] = NS1_del
        
        if NS1_del: cell.maboss.VModel.network.symbol_table["NS1g"] = 0
