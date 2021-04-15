#!/usr/bin/env python

import glob
import numpy as np
import h5py
import datetime
import fnmatch
from astropy.time import Time
import astropy.units as u
import matplotlib.pyplot as plt
import pdb
import progressbar as pbar
import struct

#----------------------------------------------------------------

def return_root_attributes(obsfileinfo, bctlcmds):
    
	stationlist = obsfileinfo['station_id'] #ct.table(ms+"/ANTENNA").getcol("NAME")
	dt = obsfileinfo['integration'] #ct.table(ms).getcol("EXPOSURE")[0]
	filetime = datetime.datetime.strptime(obsfileinfo['filenametime'], "%Y%m%d_%H%M%S") 
	time0 = obsfileinfo['datetime']
	tlen = obsfileinfo['duration_scan']
	time1 = time0 + datetime.timedelta(seconds=tlen)
	starttime = Time(time0,format="datetime")
	endtime = Time(time1,format="datetime")
	freqs = obsfileinfo['frequencies'] 
	bits = int(np.char.split(obsfileinfo['rcusetup_cmds'], sep="=").all()[1])

	rootattrs = dict()
	rootattrs['ANTENNA_SET'] = bctlcmds['antennaset']
	rootattrs['CLOCK_FREQUENCY'] = 200 #ct.table(ms+"/OBSERVATION").getcol("LOFAR_CLOCK_FREQUENCY")[0]
	rootattrs['CLOCK_FREQUENCY_UNIT'] = "MHz"
	rootattrs['FILEDATE'] = Time(filetime, format="datetime").iso
	rootattrs['FILENAME'] = ' '
	rootattrs['FILETYPE'] = "dynspec"
	rootattrs['FILTER_SELECTION'] = ' ' #ct.table(ms+"/OBSERVATION").getcol("LOFAR_FILTER_SELECTION")[0]
	rootattrs['GROUPTYPE'] = 'Root'
	rootattrs['ICD_NUMBER'] = 'ICD3'
	rootattrs['ICD_VERSION'] = '0.1'
	rootattrs['NOTES'] = " "
	rootattrs['OBSERVER'] = 'Richard Fallows'
	rootattrs['OBSERVATION_END_MJD'] = endtime.mjd
	rootattrs['OBSERVATION_END_TAI'] = endtime.unix_tai
	rootattrs['OBSERVATION_END_UTC'] = endtime.iso.replace(" ","T")+"00000z"
	rootattrs['OBSERVATION_FREQUENCY_MIN'] = np.min(freqs)
	rootattrs['OBSERVATION_FREQUENCY_MAX'] = np.max(freqs)
	rootattrs['OBSERVATION_FREQUENCY_CENTER'] = np.median(freqs)
	rootattrs['OBSERVATION_FREQUENCY_UNIT'] = 'Hz'
	rootattrs['OBSERVATION_ID'] = 'LOFAR4SW' #ct.table(ms+"/OBSERVATION").getcol("LOFAR_OBSERVATION_ID")[0]
	rootattrs['OBSERVATION_NOF_BITS_PER_SAMPLE'] = bits
	rootattrs['OBSERVATION_NOF_STATIONS'] = len(stationlist)
	rootattrs['OBSERVATION_START_MJD'] = starttime.mjd
	rootattrs['OBSERVATION_START_TAI'] = starttime.unix_tai
	rootattrs['OBSERVATION_START_UTC'] = starttime.iso.replace(" ","T")+"00000z"
	rootattrs['OBSERVATION_STATIONS_LIST'] = [stationlist]
	rootattrs['PIPELINE_NAME'] = 'DYNAMIC SPECTRUM'
	rootattrs['PIPELINE_VERSION'] = 'LOFAR4SW iLisa HDF5 formatter v0.0.1'
	rootattrs['PROJECT_CONTACT'] = 'Dr Mario Bisi'
	rootattrs['PROJECT_CO_I'] = 'Dr Stuart Robinson'
	rootattrs['PROJECT_ID'] = 'LOFAR4SW'
	rootattrs['PROJECT_PI'] = 'Dr Richard Fallows'
	rootattrs['PROJECT_TITLE'] = 'LOFAR4SW'
	rootattrs['SYSTEM_VERSION'] = 'v0.1'
	rootattrs['TARGETS'] = bctlcmds['digdir']
	rootattrs['TOTAL_INTEGRATION_TIME'] = obsfileinfo['duration_scan']

	''' These are not root attrs in ICD3 9374
	rootattrs['BF_FORMAT'] = 'BF'
	rootattrs['BF_VERSION'] = ' '# 'Cobalt/'+ct.table(ms+"/OBSERVATION").getcol('LOFAR_SYSTEM_VERSION')[0].split("(")[0]
	rootattrs['CHANNELS_PER_SUBANDS'] = 1 #ct.table(ms+"/SPECTRAL_WINDOW").getcol("NUM_CHAN")[0]
	rootattrs['CHANNEL_WIDTH'] = 0.1953125
	rootattrs['CHANNEL_WIDTH_UNIT'] = "MHz"
		rootattrs['CREATE_OFFLINE_ONLINE'] = 'Offline'
	rootattrs['DOC_NAME'] = 'ICD 6:Dynamic-Spectrum Data'
	rootattrs['DOC_VERSION'] = 'ICD6:v-2.03.10'
	rootattrs['DYN_SPEC_GROUPS'] = "True"
		rootattrs["NOF_DYNSPEC"] = len(stationlist)
	rootattrs["NOF_SAMPLES"] = obsfileinfo['duration_scan']*len(freqs)
	rootattrs['NOF_TILED_DYNSPEC'] = 0
		rootattrs['POINT_ALTITUDE'] = bctlcmds['digdir']  #np.array(["none" for i in range(len(stationlist))])
	rootattrs['POINT_AZIMUTH'] = bctlcmds['digdir']  #np.array(["none" for i in range(len(stationlist))])
	rootattrs['POINT_DEC'] = bctlcmds['digdir']  #np.degrees(ct.table(ms+"/POINTING").getcol("DIRECTION")[0])[0][1]
	rootattrs['POINT_RA'] = bctlcmds['digdir']  #np.degrees(ct.table(ms+"/POINTING").getcol("DIRECTION")[0])[0][0]
	rootattrs['PRIMARY_POINTING_DIAMETER'] = 0.0
		rootattrs['SAMPLING_RATE'] = 1/dt
	rootattrs['SAMPLING_RATE_UNIT'] = 'Hz'
	rootattrs['SAMPLING_TIME'] = dt
	rootattrs['SAMPLING_TIME_UNIT'] = 's'
	rootattrs['SUBBAND_WIDTH'] = 0.1953125
	rootattrs['SUBBAND_WIDTH_UNIT'] = 'MHz'
	rootattrs['SYSTEM_TEMPERATURE'] = np.zeros(len(stationlist))
	rootattrs['TELESCOPE'] = obsfileinfo['station_id']
	rootattrs['TOTAL_BAND_WIDTH'] = freqs[-1] - freqs[0]
	rootattrs['TOTAL_INTEGRATION_TIME'] = dt
	rootattrs['TOTAL_INTEGRATION_TIME_UNIT'] = 's'
	rootattrs['WEATHER_HUMIDITY'] = np.zeros(len(stationlist))
	rootattrs['WEATHER_STATIONS_LIST'] = np.array(["none" for i in range(len(stationlist))])
	rootattrs['WEATHER_TEMPERATURE'] = np.zeros(len(stationlist))
	rootattrs['BEAMCTL_COMMAND'] = obsfileinfo['beamctl_cmds']
	rootattrs['RSPCTL_COMMAND'] = obsfileinfo['rspctl_cmds']
	rootattrs['CALTABLE'] = np.zeros((192), dtype=np.complex_) # Place holder for the moment.
	'''

	return rootattrs

def return_sap_attributes(obsfileinfo, bctlcmds):

	sapttrs = dict()
	dt = obsfileinfo['integration']
	sapttrs['GROUPTYPE'] = 'SubArrayPointing'
	sapttrs['POINT_DEC'] = bctlcmds['digdir']  
	sapttrs['POINT_RA'] = bctlcmds['digdir']  
	sapttrs['POINT_ALTITUDE'] = bctlcmds['digdir']  
	sapttrs['POINT_AZIMUTH'] = bctlcmds['digdir']  
	sapttrs['NOF_SAMPLES'] = obsfileinfo['duration_scan']/dt
	sapttrs['SAMPLING_RATE'] = 1/dt
	sapttrs['SAMPLING_RATE_UNIT'] = 'Hz'
	sapttrs['SAMPLING_TIME'] = dt
	sapttrs['SAMPLING_TIME_UNIT'] = 's'
	sapttrs['SUBBAND_WIDTH'] = 0.1953125
	sapttrs['SUBBAND_WIDTH_UNIT'] = 'MHz'
	sapttrs['CHANNELS_PER_SUBANDS'] = 1 #ct.table(ms+"/SPECTRAL_WINDOW").getcol("NUM_CHAN")[0]
	sapttrs['CHANNEL_WIDTH'] = 0.1953125
	sapttrs['CHANNEL_WIDTH_UNIT'] = "MHz"
	sapttrs['NOF_BEAMS'] = 1

	return sapttrs

def return_beam_attributes(obsfileinfo, rootattrs, sapattrs, STOKES=['I']):

	beamattrs = dict()
	beamattrs['GROUPTYPE'] = 'Beam'
	beamattrs['TARGETS'] = rootattrs['TARGETS']
	beamattrs['STATION_LIST'] = rootattrs['OBSERVATION_STATIONS_LIST']
	beamattrs['TRACKING'] = 'J2000'
	beamattrs['POINT_DEC'] = sapattrs['POINT_DEC']
	beamattrs['POINT_RA'] = sapattrs['POINT_RA']
	beamattrs['POINT_OFFSET_DEC'] = 0.0
	beamattrs['POINT_OFFSET_RA'] = 0.0
	beamattrs['BEAM_DIAMETER'] = 0.0
	beamattrs['BEAM_DIAMETER_DEC'] = 0.0
	beamattrs['BEAM_DIAMETER_RA'] = 0.0
	beamattrs['BEAM_FREQUENCY_CENTER'] = rootattrs['OBSERVATION_FREQUENCY_CENTER']
	beamattrs['BEAM_FREQUENCY_UNIT'] = 'MHz'
	beamattrs['FOLDED_DATA'] = ' '
	beamattrs['FOLD_PERIOD'] = ' '
	beamattrs['FOLD_PERIOD_UNIT'] = ' '
	beamattrs['DEDISPERSION'] = 'None'
	beamattrs['DEDISPERSION_MEASURE'] = ' '
	beamattrs['DEDISPERSION_UNIT'] = ' '
	beamattrs['BARYCENTER'] = 0
	beamattrs['NOF_STOKES'] = len(STOKES)
	beamattrs['STOKES_COMPONENTS'] = STOKES
	beamattrs['COMPLEX_VOLTAGE'] = 0
	beamattrs['SIGNAL_SUM'] = 'COHERENT'

	return beamattrs


def return_time_attributes(rootattrs):
	
	timeattrs = dict()
	
	timeattrs['AXIS_NAMES'] = ['Time']
	timeattrs['AXIS_UNIT'] = ['s']
	timeattrs['COORDINATE_TYPE'] = 'Time'
	timeattrs['GROUP_TYPE'] = 'TimeCoord'
	timeattrs['INCREMENT'] = obsfileinfo['integration'] 
	timeattrs['NOF_AXES'] = 1
	timeattrs['PC'] = [1.,0.]
	timeattrs['REFERENCE_PIXEL'] = [0.]
	timeattrs['REFERENCE_VALUE'] = [0.]
	timeattrs['STORAGE_TYPE'] = ['Linear']
	timeattrs['AXIS_VALUE_PIXEL'] = [0.]
	timeattrs['AXIS_VALUE_WORLD'] = [0.]
	return timeattrs

#----------------------------------------------------------------

def return_spectral_attributes(freqs):
	
	specattrs = dict()
	
	specattrs['AXIS_NAMES'] = ['Frequency']
	specattrs['AXIS_UNIT'] = ['MHz']
	specattrs['AXIS_VALUE_PIXEL'] = range(len(freqs))
	specattrs['AXIS_VALUE_WORLD'] = freqs
	specattrs['COORDINATE_TYPE'] = 'Spectral'
	specattrs['GROUP_TYPE'] = 'SpectralCoord'
	specattrs['INCREMENT'] = [0.]
	specattrs['NOF_AXES'] = 1
	specattrs['PC'] = [1.,0.]
	specattrs['REFERENCE_PIXEL'] = [0.]
	specattrs['REFERENCE_VALUE'] = [0.]
	specattrs['STORAGE_TYPE'] = ['Linear']
	specattrs['AXIS_VALUE_PIXEL'] = [0.]
	specattrs['AXIS_VALUE_WORLD'] = [0.]
	return specattrs

def return_coordinates_attributes(rootattrs,stnxyz):
	
	coattrs = dict()

	coattrs['GROUP_TYPE'] = 'Coordinates'
	coattrs['COORDINATE_TYPES'] = ['Time','Spectral']
	coattrs['NOF_AXIS'] = 2.0
	coattrs['NOF_COORDINATES'] = 2.0
	coattrs['REF_LOCATION_FRAME'] = 'ITRF'
	coattrs['REF_LOCATION_UNIT'] = ['m','m','m']
	coattrs['REF_LOCATION_VALUE'] = stnxyz
	coattrs['REF_TIME_FRAME'] = 'MJD'
	coattrs['REF_TIME_UNIT'] = 'd'
	coattrs['REF_TIME_VALUE'] = rootattrs['OBSERVATION_START_MJD']
	return coattrs    


def return_bctlmds(obsfileinfo):
 
    cmd = obsfileinfo['beamctl_cmds'][0]
    cmdsplit = np.char.split(cmd, sep='--').all()
    bctlcmds = {}
    for i in range(1, len(cmdsplit)):
        keyval = np.char.split(cmdsplit[i], sep="=").all()
        keywd = keyval[0]
        val = keyval[1]
        bctlcmds[keywd] = val
    return bctlcmds

from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates
def mytformat(x, pos):
    # Function to remove trailing zeros on time fmt
    label = mdates.num2date(x).strftime('%H:%M:%S.%f').rstrip("0").rstrip(".")
    return label

def plotUPDfullstokes(fullstokes, times, freqs, metadata, savepath='.'):
    
    #fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(12,12))
    fig, axes = plt.subplots(4, 1, figsize=(12,12))
    #axes = [ax0, ax1, ax2, ax3]
    cols = ['Spectral_r', 'coolwarm', 'coolwarm', 'coolwarm']
    slabel = ['I', 'Q', 'U', 'V'] 
    freqs = np.linspace(freqs[0], freqs[-1], len(fullstokes[0,0,::]))/1e6
    
    for i in range(len(axes)):
        spec = fullstokes[i, ::, ::].T
        ax = axes[i]
        bst = ax.pcolormesh(times, freqs, spec,
                       vmin=np.percentile(spec, 1.5),
                       vmax=np.percentile(spec, 95),
                       cmap=plt.get_cmap(cols[i]))

        cbar = fig.colorbar(bst, ax=ax)
        cbar.set_label('Flux [arb. units]')
        ax.set_title('{}'.format('Stokes '+slabel[i]))

        ax.xaxis.set_major_formatter(FuncFormatter(mytformat))
        #ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.set_xlabel('Time [UT]')
        ax.set_ylabel('Frequency [MHz]')
        #ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    
    plt.xticks(rotation=0)
    plt.subplots_adjust(wspace=0.2, hspace=0.4)
    #fig.autofmt_xdate()
    ldat_type = metadata['ldat_type']
    pngname = 'l4sw_uk902c_'+metadata['datetime'].strftime("%Y%m%dT%H%M%S")+'_SUN_'+ldat_type+'_IQUV.png'
    plt.savefig(savepath+pngname)
    #plt.show()


def readUDPstokes(files, metadata):
    startprocess = datetime.datetime.utcnow()
    print('Start process time: %s' %(startprocess))

    files = np.sort(files) 
    nsbs = len(metadata['frequencies'])
    duration = metadata['duration_scan'] 
    freqs = metadata['frequencies'] # 
    integ = metadata['integration'] # Input raw time res. Usuall 16x5.12 microseconds.
    time0 = metadata['datetime']
    tres = 0.001  		    # Desired tres after integration
    ntimes = int(tres/integ)        # In while loop below, read ever ntimes -> integrate 
    nsteps = duration/tres
    const1 = int(ntimes*244*4)      # Numbytes to read in each block. Each element is a 4 byte float32.
				    # # Tobias metadata says 411 subbands. But data seems to be 244.	
    bar = pbar.ProgressBar(maxval=nsteps, \
    widgets=[pbar.Bar('=', '[', ']'), ' ', pbar.Percentage()])
    
    iquv = []
    for f in files:
       i=0
       spec = []  
       file_obj = open(f, 'rb')
       print('Reading and formatting %s.' %(f))
       bar.start()
       while 1:
          block = file_obj.read(const1)
          if not block: break
          format = '{:d}f'.format(len(block)//4)
          block = np.array(struct.unpack(format, block))
          data = np.sum(block.reshape(-1, 244), 0)
          spec.append(data)
          bar.update(i+1) ; i+=1	
       
       file_obj.close()
       bar.finish()
       iquv.append(spec)
    
    iquv = np.array(iquv)
    endprocess = datetime.datetime.utcnow()
    processtime = (endprocess - startprocess).total_seconds()
    print('End process time: %s' %(endprocess))
    print('Duration of files: %s. Total process time: %s.' %(duration, processtime))
    ntimes = len(iquv[0, ::, 0])
    times = [time0 + datetime.timedelta(seconds=i * tres) for i in range(ntimes)]
    metadata['integration'] = tres
    
    return iquv, times, freqs, metadata


def sb_to_freqs(sb0, sb1, mode, clock_freq=200.0):
    modes = {3: 0, 5: 100, 7: 200}  # start frequency for each mode
    total_sbs = 512.0  # total number of subbands
    freqs = modes[mode] + numpy.arange(sb0, sb1 + 1) * clock_freq / (total_sbs * 2.0)
    # freqs = freqs[::-11]
    return freqs

def udp2hdf5(data, obsfileinfo, stokes='I', savepath='.'):


	bctlcmds = return_bctlmds(obsfileinfo)
	ldat_type = obsfileinfo['ldat_type']   
	stksstr = ''.join(stokes)
	strtime0 = obsfileinfo['datetime'].strftime("%Y%m%dT%H%M%S") 
	filename = './l4sw_uk902c_'+strtime0+'_SUN_'+ldat_type+'_'+stksstr+'_icd3.h5'

	outfid = h5py.File(filename, "w")

	##########################################
	# 		Make the root attributes
	rootattrs = return_root_attributes(obsfileinfo, bctlcmds)
	for ky in rootattrs.keys():
		print(ky)
		outfid.attrs[ky] = rootattrs[ky]

	##########################################
	# 		Root attribute groups
	outfid.create_group("SYS_LOG")

	##########################################
	# 		Make SAP group and attibutes
	sap = outfid.create_group("SUB_ARRAY_POINTING_"+str(0).rjust(3,"0"))
	sapattrs = return_sap_attributes(obsfileinfo, bctlcmds)
	for ky in sapattrs.keys():
		sap.attrs[ky] = sapattrs[ky]

	phist = sap.create_group("PROCESS_HISTORY")
	beam = sap.create_group("BEAM_"+str(0).rjust(3,"0"))
	beamttrs = return_beam_attributes(obsfileinfo, rootattrs, sapattrs, STOKES=stokes)
	for ky in beamttrs.keys():
		beam.attrs[ky] = beamttrs[ky]

	##########################################
	# 		Make the beam data sets
	idata = beam.create_dataset("STOKES_0",data=data[0,:,:])
	qdata = beam.create_dataset("STOKES_1",data=data[1,:,:])
	udata = beam.create_dataset("STOKES_2",data=data[2,:,:])
	vdata = beam.create_dataset("STOKES_3",data=data[3,:,:])

	################################################
	# 		Make the COORD group and attributes
	coords = beam.create_group("PROCESS_HISTORY")

	coords = beam.create_group("COORDINATES")
	station_ITRFxyz = 'Placeholder'
	coordsattrs = return_coordinates_attributes(rootattrs, station_ITRFxyz)
	for ky in coordsattrs.keys():
		coords.attrs[ky] = coordsattrs[ky]

	spec = coords.create_group("SPECTRAL")
	specattrs = return_spectral_attributes(obsfileinfo['frequencies'])
	for ky in specattrs.keys():
		spec.attrs[ky] = specattrs[ky]

	tm = coords.create_group("TIME")
	tmattrs = return_time_attributes(rootattrs)
	for ky in tmattrs.keys():
		tm.attrs[ky] = tmattrs[ky]


if __name__=='__main__':

	obsfileinfo = np.load('obsudpinfo.npy', allow_pickle=True, encoding='latin1').all()
	data = np.zeros((4, 1000, 244))
	udp2hdf5(data, obsfileinfo, stokes=['I', 'Q', 'U', 'V'], savepath='.')





