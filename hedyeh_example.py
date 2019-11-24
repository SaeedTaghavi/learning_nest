import nest
import pylab
import nest.raster_plot
import numpy

import time
import numpy as np
import matplotlib.pyplot as plt

nest.ResetKernel()
nest.SetKernelStatus({'print_time': True,'local_num_threads':1})




msd = 10

epsilonP_ff = 0.12      # inter-layer connection probability


epsilonP_fb = 0.06      # inter-layer connection probability

n_vp=nest.GetKernelStatus('total_num_virtual_procs')
msdrange1=range(msd,msd+n_vp)

pyrngs=[numpy.random.RandomState(s) for s in msdrange1]
msdrange2=range(msd+n_vp+1,msd+1+2*n_vp)
nest.SetKernelStatus({'grng_seed':msd+n_vp , 'rng_seeds':msdrange2})

a = 25                 # number of spikes in one pulse packet
sdev = 0.0             # width of pulse packet (ms)
weight = 0.33           # PP amplitude (mV), bara tanavobi se sadom
pulse_times = [800.]#, 833.0, 866.0, 899.0, 932.0]   # occurrence time (center) of pulse-packet (ms)

NP        = 70
NE        = 200             # number of excitatory neurons
NI        = 50              # number of inhibitory neurons
N_neurons = NE+NI+NP           # number of neurons in total
N_rec     = 10*N_neurons    # record from all neurons

simtime    = 2000.     # Simulation time in ms
in_delay   = 1.5       # within-layer synaptic delay in ms
bet_delay  = 13.0      # between-layer synaptic delay in ms

j_exc_exc = 0.33      # EE connection strength
j_exc_inh = 1.5        # EI connection strength
j_inh_exc = -6.2       # IE connection strength
j_inh_inh = -12.0       # II connection strength

p_rate_ex = 8000.0 
p_rate_in = p_rate_ex - 1700.0

epsilonEE   = 0.2       # EE connection probability
epsilonIE   = 0.2       # IE connection probability
epsilonEI   = 0.2       # EI connection probability
epsilonII   = 0.2       # II connection probability

bet_g_ff    = 0.33      # connection strength between layers
bet_g_fb    = 0.33      # connection strength between layers





CEE   = int(epsilonEE*NE)      # number of excitatory synapses per E neuron
CIE   = int(epsilonIE*NI)      # number of inhibitory synapses per E neuron
CEI   = int(epsilonEI*NE)      # number of excitatory synapses per I neuron
CII   = int(epsilonII*NI)      # number of inhibitory synapses per I neuron 

CEP_ff = int(epsilonP_ff*NE)    # number of feedforward connections per E neuron 
CEP_fb = int(epsilonP_fb*NE)    # number of feedback connections per E neuron 

  
exci_neuron_params= {'V_th' :-54.,
                'V_reset'   :-70.,
                'tau_syn_ex': 1.0,
                'tau_syn_in': 1.,
                'tau_minus' : 20.}

inhi_neuron_params= {'V_th' :-54.,
                'V_reset'   :-70.,
                'tau_syn_ex': 1.0,
                'tau_syn_in': 1.,
                'tau_minus' : 20.}

nest.SetKernelStatus({"overwrite_files": True})

pop1_nodes_ex = nest.Create("iaf_cond_alpha",NE,params=exci_neuron_params)
pop1_nodes_in = nest.Create("iaf_cond_alpha",NI,params=inhi_neuron_params)

pop2_nodes_ex = nest.Create("iaf_cond_alpha",NE,params=exci_neuron_params)
pop2_nodes_in = nest.Create("iaf_cond_alpha",NI,params=inhi_neuron_params)


nodes= pop1_nodes_ex + pop1_nodes_in + pop2_nodes_ex + pop2_nodes_in 

nodes_ex= pop1_nodes_ex + pop2_nodes_ex 

nodes_in= pop1_nodes_in + pop2_nodes_in 

node_info=nest.GetStatus(nodes)
local_nodes=[(ni['global_id'],ni['vp']) 
             for ni in node_info if ni ['local']]
for gid,vp in local_nodes:
  nest.SetStatus([gid],{'V_m':pyrngs[vp].uniform(-75.0,-65.0)})


#pp = nest.Create('pulsepacket_generator',params={'activity':a,'sdev':sdev,'pulse_times':pulse_times})

noise = nest.Create("poisson_generator", 2, [{"rate": p_rate_ex}, {"rate": p_rate_in}])

'''
multimeter = nest.Create("multimeter")

nest.SetStatus(multimeter,[{"label": "resonance",
                            "withtime": True,
                            "withgid": True,
                            "to_file": True}])
nest.SetStatus(multimeter, {"withtime":True, "record_from":["V_m","g_ex","g_in"]})

nest.Connect(multimeter, nodes)
'''
nest.CopyModel("static_synapse","EI",{"weight":j_exc_inh,"delay":in_delay})
nest.CopyModel("static_synapse","EE",{"weight":j_exc_exc,"delay":in_delay})
nest.CopyModel("static_synapse","IE",{"weight":j_inh_exc,"delay":in_delay})
nest.CopyModel("static_synapse","II",{"weight":j_inh_inh,"delay":in_delay})
#nest.CopyModel("static_synapse","PP",{"weight":j_pulse_packet,"delay":del_pp})

##### POPULATION 1
nest.Connect(pop1_nodes_ex,pop1_nodes_ex,{'rule': 'fixed_indegree','indegree': CEE},'EE')

nest.Connect(pop1_nodes_ex,pop1_nodes_in,{'rule': 'fixed_indegree','indegree': CEI},'EI')

nest.Connect(pop1_nodes_in,pop1_nodes_ex,{'rule': 'fixed_indegree','indegree': CIE},'IE')

nest.Connect(pop1_nodes_in,pop1_nodes_in,{'rule': 'fixed_indegree','indegree': CII},'II')

##### POPULATION 2
nest.Connect(pop2_nodes_ex,pop2_nodes_ex,{'rule': 'fixed_indegree','indegree': CEE},'EE')

nest.Connect(pop2_nodes_ex,pop2_nodes_in,{'rule': 'fixed_indegree','indegree': CEI},'EI')

nest.Connect(pop2_nodes_in,pop2_nodes_ex,{'rule': 'fixed_indegree','indegree': CIE},'IE')

nest.Connect(pop2_nodes_in,pop2_nodes_in,{'rule': 'fixed_indegree','indegree': CII},'II')



##### POP1 ------>>>> POP2
nest.CopyModel('static_synapse_hom_w','bet_excitatory_ff',{'weight':bet_g_ff,'delay':bet_delay})

nest.Connect(pop1_nodes_ex[1:NP],pop2_nodes_ex[1:NP],{'rule': 'fixed_indegree','indegree': CEP_ff},'bet_excitatory_ff')


##### POP2 ------>>>> POP1  (Resonance)
nest.CopyModel('static_synapse_hom_w','bet_excitatory_fb',{'weight':bet_g_fb,'delay':bet_delay})
nest.Connect(pop2_nodes_ex[1:NP],pop1_nodes_ex[1:NP],{'rule': 'fixed_indegree','indegree': CEP_fb},'bet_excitatory_fb')



#nest.CopyModel('static_synapse_hom_w','pp_excitatory',{'weight':weight,'delay':bet_delay})

#nest.Connect(pp, pop1_nodes_ex, {'rule': 'all_to_all'},'pp_excitatory')

nest.CopyModel('static_synapse_hom_w','NOISE_EXCI_SYN',{'weight':0.25,'delay':0.1})

nest.Connect(noise[:1], pop1_nodes_ex+pop2_nodes_ex, syn_spec='NOISE_EXCI_SYN')

nest.CopyModel('static_synapse_hom_w','NOISE_INHI_SYN',{'weight':0.4,'delay':0.1})

nest.Connect(noise[1:], pop1_nodes_in+pop2_nodes_in, syn_spec='NOISE_INHI_SYN')

conns=nest.GetConnections(target=pop1_nodes_in)
conn_vals=nest.GetStatus(conns,"source")

spikes  = nest.Create('spike_detector',2,[{'label': 'resonance-py-ex'},{'label': 'resonance-py-in'}]) 

spikes_E= spikes[:1]
spikes_I= spikes[1:]


nest.SetStatus(spikes_E,[{"label": "B",
                         "withtime": True,
                         "withgid": True,
                         "to_file": True}])

nest.Connect(pop1_nodes_ex[71:200],spikes_E)
nest.Connect(pop1_nodes_in[71:200],spikes_I)

'''
vm_pars = {
    'record_to': ['memory'],
    'withtime':  True,
    'withgid':   True,
    'interval':  1.
    }
vm = nest.Create('voltmeter', 1, vm_pars)

nest.Connect(vm, nodes_ex)
'''
nest.Simulate(simtime)
'''
dmm   = nest.GetStatus(multimeter)[0]
Vms   = dmm["events"]["V_m"]
ts    = dmm["events"]["times"]
gexs  = dmm["events"]["g_ex"]
'''
spike_events = nest.GetStatus(spikes , 'n_events')
rate_ex= (spike_events[0]/simtime)*(1000.0/(2*NE))
rate_in= (spike_events[1]/simtime)*(1000.0/(2*NI))

import numpy as np
import pylab as pl

bin_w = 5.
time_range = (500.,2000.)
act = np.loadtxt('B-503-0.gdf')
evs = act[:,0]
ts = act[:,1]

if time_range!=[]:
    idx = (ts>time_range[0]) & (ts<=time_range[1])
    spikes = ts[idx]
total_time = time_range[1] - time_range[0]

if len(spikes) == 0:
  print 'psd: spike array is empty'

ids = np.unique(evs)
nr_neurons = len(ids)
#psd, max_value, freq,h = misc2.psd_sp(spikes[:,1],nr_bins,nr_neurons)
bins = np.arange(time_range[0],time_range[1],bin_w)
a,b = np.histogram(spikes, bins)
ff = abs(np.fft.fft(a- np.mean(a)))**2
Fs = 1./(bin_w*0.001)
freq2 = np.fft.fftfreq(len(bins))[0:len(bins/2)+1]
freq = np.linspace(0,Fs/2,len(ff)/2+1) # frequency axis
px = ff[0:len(ff)/2+1] # power
max_px = np.max(px[1:])
idx = px == max_px
max_freq = freq[pl.find(idx)] # freq. with max power
max_pow = px[pl.find(idx)]
# Spectral entropy
k = len(freq)
norm_px = px/sum(px)
sum_power = 0
for ii in range(k):
    sum_power += (norm_px[ii]*np.log(norm_px[ii]))
spec_ent = -(sum_power/np.log(k))

print "maximum frequency: %.2f" % max_freq
print "maximum power    : %.2f" % spec_ent

nest.raster_plot.from_device(spikes_E , hist=True)
pylab.show()

#print "Connections to pop1_nodes_in : %.2f " % len(conn_vals)


'''
Vm = nest.GetStatus(vm, 'events')[0]['V_m']
times = nest.GetStatus(vm, 'events')[0]['times']
senders = nest.GetStatus(vm, 'events')[0]['senders']

dSD = nest.GetStatus(spikes_E,keys="events")[0]

Vm_single = [Vm[senders == ii] for ii in pop1_nodes_ex[71:200]]
simtimes = numpy.arange(1, simtime)
Vm_average = numpy.mean(Vm_single, axis=0)
Var = numpy.var(Vm_average)#[1201:1999])/numpy.var(Vm_average[400:1200])
pFF       = (Var/rate_ex)

print "Excitatory rate    : %.2f 1/sec" % rate_ex
print "Inhibitory rate    : %.2f 1/sec" % rate_in
print "Normalized variance: %.2f " % Var
print "pFF first layer    : %.2f" % pFF

nest.raster_plot.from_device(spikes_E , hist=True)
#nest.raster_plot.from_device(spikes_I , hist=True)
plt.xlabel('Time (ms)', fontsize=16)
plt.ylabel('Neuron number', fontsize=16)
pylab.show()

#Here we plot average membrane potential.
'''
'''
pylab.plot(simtimes, Vm_average, 'b')
pylab.xlabel('Time (ms)', fontsize=18)
pylab.ylabel('Average membrane potential (mV)', fontsize=18)
pylab.show()


'''
#Here we plot the fourier transform of potential.
'''

import matplotlib.pyplot as plt
t = numpy.arange(len(Vm_average[800:1000]))
sp = numpy.fft.fft(Vm_average[800:1000])
freq = numpy.fft.fftfreq(t.shape[-1])
plt.plot(1000*freq, numpy.abs(sp))#, freq, sp.imag)
pylab.xlabel('frequency (Hz)', fontsize=18)
pylab.ylabel('fraction of neurons ', fontsize=18)
plt.show()


#pylab.plot(gexs, 'b')
#pylab.xlabel('Time (ms)', fontsize=18)
#pylab.ylabel('Strength', fontsize=18)
#pylab.show()

'''



