#!/usr/bin/env python

#import casacore.tables as ct
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
basedir = "/home/raf/Lofar/progs/Baselines/"
basedir = "/home/fallows/scripts/"

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
    rootattrs['BF_FORMAT'] = 'BF'
    rootattrs['BF_VERSION'] = ' '# 'Cobalt/'+ct.table(ms+"/OBSERVATION").getcol('LOFAR_SYSTEM_VERSION')[0].split("(")[0]
    rootattrs['CHANNELS_PER_SUBANDS'] = 1 #ct.table(ms+"/SPECTRAL_WINDOW").getcol("NUM_CHAN")[0]
    rootattrs['CHANNEL_WIDTH'] = 0.1953125
    rootattrs['CHANNEL_WIDTH_UNIT'] = "MHz"
    rootattrs['CLOCK_FREQUENCY'] = 200 #ct.table(ms+"/OBSERVATION").getcol("LOFAR_CLOCK_FREQUENCY")[0]
    rootattrs['CLOCK_FREQUENCY_UNIT'] = "MHz"
    rootattrs['CREATE_OFFLINE_ONLINE'] = 'Offline'
    rootattrs['DOC_NAME'] = 'ICD 6:Dynamic-Spectrum Data'
    rootattrs['DOC_VERSION'] = 'ICD6:v-2.03.10'
    rootattrs['DYN_SPEC_GROUPS'] = "True"
    rootattrs['FILEDATE'] = Time(filetime, format="datetime")
    rootattrs['OBSERVATION_ID'] = 'LOFAR4SW' #ct.table(ms+"/OBSERVATION").getcol("LOFAR_OBSERVATION_ID")[0]
    rootattrs['FILENAME'] = ' '
    rootattrs['FILETYPE'] = "dynspec"
    rootattrs['FILTER_SELECTION'] = ' ' #ct.table(ms+"/OBSERVATION").getcol("LOFAR_FILTER_SELECTION")[0]
    rootattrs['GROUPTYPE'] = 'Root'
    rootattrs["NOF_DYNSPEC"] = len(stationlist)
    rootattrs["NOF_SAMPLES"] = obsfileinfo['duration_scan']*len(freqs)
    rootattrs['NOF_TILED_DYNSPEC'] = 0
    rootattrs['NOTES'] = " "
    rootattrs['OBSERVATION_END_MJD'] = endtime.mjd
    rootattrs['OBSERVATION_END_UTC'] = endtime.iso.replace(" ","T")+"000000Z + 0 s"
    rootattrs['OBSERVATION_FREQUENCY_MIN'] = np.min(freqs)
    rootattrs['OBSERVATION_FREQUENCY_MAX'] = np.max(freqs)
    rootattrs['OBSERVATION_FREQUENCY_CENTER'] = np.median(freqs)
    rootattrs['OBSERVATION_FREQUENCY_UNIT'] = 'Hz'
    rootattrs['OBSERVATION_NOF_BITS_PER_SAMPLE'] = bits
    rootattrs['OBSERVATION_NOF_STATIONS'] = len(stationlist)
    rootattrs['OBSERVATION_START_MJD'] = starttime.mjd
    rootattrs['OBSERVATION_START_UTC'] = starttime.iso.replace(" ","T")+"000000Z + 0 s"
    rootattrs['OBSERVATION_STATIONS_LIST'] = np.array([stationlist])
    rootattrs['PIPELINE_NAME'] = 'DYNAMIC SPECTRUM'
    rootattrs['PIPELINE_VERSION'] = 'Dynspec v.3.0'
    rootattrs['POINT_ALTITUDE'] = bctlcmds['digdir']  #np.array(["none" for i in range(len(stationlist))])
    rootattrs['POINT_AZIMUTH'] = bctlcmds['digdir']  #np.array(["none" for i in range(len(stationlist))])
    rootattrs['POINT_DEC'] = bctlcmds['digdir']  #np.degrees(ct.table(ms+"/POINTING").getcol("DIRECTION")[0])[0][1]
    rootattrs['POINT_RA'] = bctlcmds['digdir']  #np.degrees(ct.table(ms+"/POINTING").getcol("DIRECTION")[0])[0][0]
    rootattrs['PRIMARY_POINTING_DIAMETER'] = 0.0
    rootattrs['PROJECT_CONTACT'] = 'Dr Mario Bisi'
    rootattrs['PROJECT_CO_I'] = 'Dr Stuart Robinson'
    rootattrs['PROJECT_ID'] = 'LOFAR4SW'
    rootattrs['PROJECT_PI'] = 'Dr Richard Fallows'
    rootattrs['PROJECT_TITLE'] = 'LOFAR4SW'
    rootattrs['SAMPLING_RATE'] = 1/dt
    rootattrs['SAMPLING_RATE_UNIT'] = 'Hz'
    rootattrs['SAMPLING_TIME'] = dt
    rootattrs['SAMPLING_TIME_UNIT'] = 's'
    rootattrs['SUBBAND_WIDTH'] = 0.1953125
    rootattrs['SUBBAND_WIDTH_UNIT'] = 'MHz'
    rootattrs['SYSTEM_TEMPERATURE'] = np.zeros(len(stationlist))
    rootattrs['SYSTEM_VERSION'] = 'v0.1'
    rootattrs['TARGET'] = bctlcmds['digdir']
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
    return rootattrs

def return_dynspec_attributes(rootattrs):
    dsattrs = dict()
    dsattrs['BARYCENTER'] = 0
    dsattrs['BEAM_DIAMETER'] = 0.0
    dsattrs['BEAM_DIAMETER_DEC'] = 0.0
    dsattrs['BEAM_DIAMETER_RA'] = 0.0
    dsattrs['BEAM_FREQUENCY_CENTER'] = rootattrs['OBSERVATION_FREQUENCY_CENTER']
    dsattrs['BEAM_FREQUENCY_MAX'] = rootattrs['OBSERVATION_FREQUENCY_MAX']
    dsattrs['BEAM_FREQUENCY_MIN'] = rootattrs['OBSERVATION_FREQUENCY_MIN']
    dsattrs['BEAM_FREQUENCY_UNIT'] = 'MHz'
    dsattrs['BEAM_NOF_STATIONS'] = 1.0
    dsattrs['BEAM_STATIONS_LIST'] = rootattrs['OBSERVATION_STATIONS_LIST']
    dsattrs['COMPLEX_VOLTAGE'] = 0
    dsattrs['DEDISPERSION'] = 'NONE'
    dsattrs['DISPERSION_MEASURE'] = 0.0
    dsattrs['DISPERSION_MEASURE_UNIT'] = ''
    dsattrs['DYNSPEC_BANDWIDTH'] = rootattrs['TOTAL_BAND_WIDTH']
    dsattrs['DYNSPEC_START_MJD'] = rootattrs['OBSERVATION_START_MJD']
    dsattrs['DYNSPEC_START_UTC'] = rootattrs['OBSERVATION_START_UTC']
    dsattrs['DYNSPEC_STOP_MJD'] = rootattrs['OBSERVATION_END_MJD']
    dsattrs['DYNSPEC_STOP_UTC'] = rootattrs['OBSERVATION_END_UTC']
    dsattrs['GROUPTYPE'] = 'DYNSPEC'
    dsattrs['ONOFF'] = 'ON'
    dsattrs['POINT_DEC'] = rootattrs['POINT_DEC']
    dsattrs['POINT_RA'] = rootattrs['POINT_RA']
    dsattrs['POSITION_OFFSET_DEC'] = 0.0
    dsattrs['POSITION_OFFSET_RA'] = 0.0
    dsattrs['SIGNAL_SUM'] = 'COHERENT'
    dsattrs['STOKES_COMPONENT'] = ['I','Q','U','V']
    dsattrs['TARGET'] = rootattrs['TARGET']
    dsattrs['TRACKING'] = 'J2000'
    # file history attributed
    return dsattrs

#----------------------------------------------------------------

def return_coordinates_attributes(rootattrs,stnxyz):
    coattrs = dict()
    coattrs['COORDINATE_TYPES'] = ['Time','Spectral','Polarisation']
    coattrs['GROUP_TYPE'] = 'Coordinates'
    coattrs['NOF_AXIS'] = 3.0
    coattrs['NOF_COORDINATES'] = 3.0
    coattrs['REF_LOCATION_FRAME'] = 'ITRF'
    coattrs['REF_LOCATION_UNIT'] = ['m','m','m']
    coattrs['REF_LOCATION_VALUE'] = stnxyz
    coattrs['REF_TIME_FRAME'] = 'MJD'
    coattrs['REF_TIME_UNIT'] = 'd'
    coattrs['REF_TIME_VALUE'] = rootattrs['OBSERVATION_START_MJD']
    # locate cal file 
    return coattrs

#----------------------------------------------------------------

def return_data_attributes(arrshape):
    dataattrs = dict()
    dataattrs['DATASET_NOF_AXIS'] = 3
    dataattrs['DATASET_SHAPE'] = arrshape
    dataattrs['GROUP_TYPE'] = 'Data'
    dataattrs['WCSINFO'] = '/Coordinates'
    return dataattrs

#----------------------------------------------------------------

def return_time_attributes(rootattrs):
    timeattrs = dict()
    timeattrs['AXIS_NAMES'] = ['Time']
    timeattrs['AXIS_UNIT'] = ['s']
    timeattrs['COORDINATE_TYPE'] = 'Time'
    timeattrs['GROUP_TYPE'] = 'TimeCoord'
    timeattrs['INCREMENT'] = [rootattrs['SAMPLING_TIME']]
    timeattrs['NOF_AXES'] = 1
    timeattrs['PC'] = [1.,0.]
    timeattrs['REFERENCE_PIXEL'] = [0.]
    timeattrs['REFERENCE_VALUE'] = [0.]
    timeattrs['STORAGE_TYPE'] = ['Linear']
    return timeattrs

#----------------------------------------------------------------

def return_spectral_attributes(freqs):
    specattrs = dict()
    specattrs['AXIS_NAMES'] = ['Frequency']
    specattrs['AXIS_UNIT'] = ['MHz']
    specattrs['AXIS_VALUE_PIXEL'] = np.array(np.arange(len(freqs)))
    specattrs['AXIS_VALUE_WORLD'] = np.array(freqs)
    specattrs['COORDINATE_TYPE'] = 'Spectral'
    specattrs['GROUP_TYPE'] = 'SpectralCoord'
    specattrs['INCREMENT'] = [0.]
    specattrs['NOF_AXES'] = 1
    specattrs['PC'] = [1.,0.]
    specattrs['REFERENCE_PIXEL'] = [0.]
    specattrs['REFERENCE_VALUE'] = [0.]
    specattrs['STORAGE_TYPE'] = ['Linear']
    return specattrs

#----------------------------------------------------------------

def return_polarisation_attributes(stokes):
    polattrs = dict()
    polattrs['AXIS_NAMES'] = stokes
    polattrs['AXIS_UNIT'] = ['none']
    polattrs['COORDINATE_TYPE'] = 'Polarization'
    polattrs['GROUP_TYPE'] = 'PolarizationCoord'
    polattrs['NOF_AXES'] = len(stokes)
    polattrs['STORAGE_TYPE'] = ['Tabular']
    return polattrs

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
    tres = 0.1  		    # Desired tres after integration
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

def buildplothdf5(bstff, udppath):

    BSTdata, obsfileinfo = readbstfolder(bstff)
    # Plot hi-res beam-formed data
    files = glob.glob(udppath+"*.fil")
    udpdata = readUDPfil(files)
    # May need to enter keywords here to account for this being hi-tres data
    data = np.array(udpdata[0])
    pdb.set_trace()
    #icd6.fil2hdf5(data, obsfileinfo, stokes=['I','Q','U','V']) 
    plotUPDfullstokes(*udpdata)

def sb_to_freqs(sb0, sb1, mode, clock_freq=200.0):
    modes = {3: 0, 5: 100, 7: 200}  # start frequency for each mode
    total_sbs = 512.0  # total number of subbands
    freqs = modes[mode] + numpy.arange(sb0, sb1 + 1) * clock_freq / (total_sbs * 2.0)
    # freqs = freqs[::-11]
    return freqs



def udp2hdf5(data, obsfileinfo, stokes='I', savepath='.'):

    ldat_type = obsfileinfo['ldat_type']   
    stksstr = ''.join(stokes)
    strtime0 = obsfileinfo['datetime'].strftime("%Y%m%dT%H%M%S") 
    filename = 'l4sw_uk902c_'+strtime0+'_SUN_'+ldat_type+'_'+stksstr+'.h5'
    outfid = h5py.File(savepath+filename, "w")
    outfid.create_group("SYS_LOG")
    ds = outfid.create_group("DYNSPEC_000")

    bctlcmds = return_bctlmds(obsfileinfo)
    rootattrs = return_root_attributes(obsfileinfo, bctlcmds)
    dsattrs = return_dynspec_attributes(rootattrs)
    for ky in dsattrs.keys():
        ds.attrs[ky] = dsattrs[ky]

    dsdata = ds.create_dataset("DATA", data=data)
    dataattrs = return_data_attributes(data.shape)
    for ky in dataattrs.keys():
        dsdata.attrs[ky] = dataattrs[ky]
    ds.create_group("EVENT")
    ds.create_group("PROCESS_HISTORY")

    coords = ds.create_group("COORDINATES")
    coordsattrs = return_coordinates_attributes(rootattrs, 'a,b,c')
    for ky in coordsattrs.keys():
        coords.attrs[ky] = coordsattrs[ky]

    pol = coords.create_group("POLARIZATION")
    polattrs = return_polarisation_attributes(stokes)
    for ky in polattrs.keys():
        pol.attrs[ky] = polattrs[ky]

    spec = coords.create_group("SPECTRAL")
    specattrs = return_spectral_attributes(obsfileinfo['frequencies'])
    for ky in specattrs.keys():
        spec.attrs[ky] = specattrs[ky]

    tm = coords.create_group("TIME")
    tmattrs = return_time_attributes(rootattrs)
    for ky in tmattrs.keys():
        tm.attrs[ky] = tmattrs[ky]


    outfid.close()
    return []






