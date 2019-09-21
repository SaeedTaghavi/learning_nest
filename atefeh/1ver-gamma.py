
#import necessary modules__________________________________________________________________________________________________
import nest
import nest.topology as topo
import math
import pylab as pl
import nest.raster_plot
import matplotlib.pyplot as plt
import numpy as np
import time
from sys import exit
import os 
import sys
sys.argv.append("-quiet")


#nest.set_verbosity('M_WARNING')
#pylab.ion()


nest.ResetKernel()	# Make sure we start with a clean slate, even if we re-run the script
			# in the same Python session.


dt = 0.1    		# the resolution in ms
simtime = 5000.0 	# Simulation time in ms
delay = 0.3   		# synaptic delay in ms


J_ex = 1.6  		# amplitude of excitatory postsynaptic potential
J_in = -0.8	  	# amplitude of inhibitory postsynaptic potential

nest.SetKernelStatus({"resolution": dt, "print_time": True,
                      "overwrite_files": True})


#setting neurons defaults___________________________________________________________________________________________________
nest.SetDefaults('iaf_cond_alpha', 
		{'C_m': 200.0,		#membrane capacitance (pF)
                 'g_L': 12.5,		#leak conductace (ns)
                 'E_L': -80.0,		#leak reversal potential (mV)
                 'V_th': -45.0,		#spike threshold (mV)
                 'V_reset': -80.0,	#resting membrane potential (mV)
                 't_ref': 2.0,		#refractory period (ms)
                 'E_ex': 0.0,		#excitatory reversal potential (mV)
                 'E_in': -64.0,		#inhibitory reversal potential (mV)
                 'tau_syn_ex':5.0,	#Rise time of the excitatory synaptic alpha function in ms. 
                 'tau_syn_in':10.0,	#Rise time of the inhibitory synaptic alpha function in ms.
		 'I_e':250.0		#double	- Constant input current in pA.  
		})

nest.CopyModel  ("static_synapse","excitatory",
                {"weight":J_ex, "delay": delay})

nest.CopyModel  ("static_synapse","inhibitory",
                {"weight":J_in, "delay": delay})


#creating layer_____________________________________________________________________________________________________________
l=topo.CreateLayer({  'columns': 100, 
		      'rows': 100, 
		      'extent': [1.0, 1.0],
                      'elements': 'iaf_cond_alpha',
		      'edge_wrap': True
		  })


#connecting layer___________________________________________________________________________________________________________
topo.ConnectLayers(l,l,{ 'connection_type': 'divergent',
               		 'mask': {'circular':{'radius':0.5}},
                 	 'kernel': {'gamma': {'kappa' : 6.0, 'theta' : 1.0}},
			 'synapse_model': 'inhibitory',
			 'allow_autapses':False,
			 'allow_multapses':True,
			 'number_of_connections':1000
	      	   })



#creating noise, recording devices and their connections____________________________________________________________________
ispikes = nest.Create("spike_detector")
nest.SetStatus(ispikes, [{"label": "striatum",
                         "withtime": True,
                         "withgid": True
#                         "to_file": True
		        }])






noise=nest.Create('poisson_generator',1,{'rate':900.0})
nrns = nest.GetLeaves(l, local_only=True)[0]
nest.Connect(noise, nrns,syn_spec="excitatory")
#nest.Connect(nrns, ispikes)
nest.Connect(nrns[1800:2000] ,ispikes)




#simulation stuffs__________________________________________________________________________________________________________
endbuild = time.time()
print("Simulating")
nest.Simulate(simtime)
endsimulate = time.time()




events_in = nest.GetStatus(ispikes, "n_events")[0]
rate_in=events_in / simtime * 1000.0 / 10000.0
print rate_in




nest.raster_plot.from_device(ispikes, hist=True)

plt.title('Population dynamics')
pl.savefig('Population dynamics')

plt.show()




#_______________________________________________________________________________________________________________________



'''
nest.SetDefaults("sinusoidal_poisson_generator",
                {'rate': 100.0, 'amplitude': 50.0,
                 'frequency': 10.0, 'phase': 0.0,
                 'individual_spike_trains': False
		})



stim = topo.CreateLayer({'rows': 1,
                      	 'columns': 1,
                      	 'elements': 'sinusoidal_poisson_generator'
		       })



cdict_stim = {'connection_type': 'divergent',
              'mask': {'circular': {'radius': 0.1},
                       'anchor': [0.2, 0.2]}}



topo.ConnectLayers(stim, l, cdict_stim)
'''




'''
my_nodes=nest.GetNodes(l)
print len(my_nodes), "********************"
connectome = nest.GetConnections()
con= nest.GetStatus(connectome, ['source', 'target'])
print "number of connections: " , len(con)
for i in con:
   print i
#print connectome

exit(0)
'''




'''


nest_mm = nest.Create('multimeter')
nest.SetStatus(nest_mm, {'record_from': ['n_events', 'mean'],
                         'withgid': True,
                         'withtime': False,
                         'interval': 0.1})

'''






# monitor the output using a multimeter, this only records with dt_rec!
'''
nest_mm = nest.Create('multimeter')
nest.SetStatus(nest_mm, {'record_from': ['n_events', 'mean'],
                         'withgid': True,
                         'withtime': False,
                         'interval': 0.1})


voltmeter = nest.Create("voltmeter")
nest.Connect(voltmeter, neuron2)
nest.voltage_trace.from_device(voltmeter)
nest.voltage_trace.show()




events_in = nest.GetStatus(ispikes, "n_events")[0]
rate_in=events_in / simtime * 1000.0 / 10000
print rate_in
 
'''



'''
 
#print network and its stuffs_______________________________________________________________________________________________
#my_nodes=nest.GetNodes(l)
#connectome = nest.GetConnections(l)
#con= nest.GetStatus(connectome, ['source', 'target'])
#print "number of connections: " , len(con)
#print connectome


#nest.PrintNetwork=()
#nest.PrintNetwork=(2)
#nest.PrintNetwork=(2,l)
#topo.PlotLayer(l, nodesize=2)


#pylab.savefig('grid_iaf.png')
'''

