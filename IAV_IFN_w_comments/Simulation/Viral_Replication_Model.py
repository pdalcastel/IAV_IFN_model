
from cc3d import CompuCellSetup
        

from Viral_Replication_ModelSteppables import Viral_Replication_ModelSteppable

CompuCellSetup.register_steppable(steppable=Viral_Replication_ModelSteppable(frequency=1))


CompuCellSetup.run()
