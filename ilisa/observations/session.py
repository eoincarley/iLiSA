"""This package provides Session class that handles observations."""
import sys
import math
import time
import datetime
import ilisa.observations.stationdriver as stationdriver
import ilisa.observations.modeparms as modeparms
import ilisa.observations.beamformedstreams.bfbackend as bfbackend
import ilisa.observations.dataIO as dataIO


class Session(object):

    def __init__(self, halt_observingstate_when_finished=True):
        self.stationdrivers = []
        stndrv = stationdriver.StationDriver()
        stndrv.halt_observingstate_when_finished = halt_observingstate_when_finished
        self.stationdrivers.append(stndrv)

    def waittoboot(self, starttimestr, pause):
        """Before booting, wait until time given by starttimestr. This includes
         a dummy beam warmup."""
        nw = datetime.datetime.utcnow()
        st = datetime.datetime.strptime(starttimestr, "%Y-%m-%dT%H:%M:%S")

        maxminsetuptime = datetime.timedelta(seconds=105 + pause)  # Longest minimal time
        # before observation
        # start to set up
        d = (st - maxminsetuptime) - nw
        timeuntilboot = d.total_seconds()
        if timeuntilboot < 0.:
            timeuntilboot = 0
        print("Will boot to observe state after " + str(timeuntilboot) + " seconds...")
        time.sleep(timeuntilboot)
        return st

    def do_bfs(self, band, duration, pointsrc, starttimestr, shutdown=True):
        """Record BeamFormed Streams (BFS)."""
        stndrv = self.stationdrivers[0]

        duration = eval(duration)  # Duration in seconds

        ###
        bits = 8  # 8
        attenuation = None
        # Subbands allocation
        if band == '10_90' or band == '30_90':
            # LBA
            lanes = (0, 1)  # (0,1)
            beamletIDs = '0:243'  # '0:243'
            subbandNrs = '164:407'  # '164:407'
        elif band == '110_190':
            # HBAlo
            lanes = (0, 1, 2, 3)  # Normally (0,1,2,3) for all 4 lanes.
            beamletIDs = '0:487'
            subbandNrs = '12:499'
        elif band == '210_250':
            # HBAhi
            lanes = (0, 1)
            beamletIDs = '0:243'
            subbandNrs = '12:255'
        else:
            raise ValueError(
                "Wrong band: should be 10_90 (LBA), 110_190 (HBAlo) or 210_250 (HBAhi).")
        pointing = modeparms.normalizebeamctldir(pointsrc)

        # Wait until it is time to start
        pause = 5  # Sufficient?
        st = self.waittoboot(starttimestr, pause)

        # From swlevel 0 it takes about 1:30min? to reach swlevel 3
        print("Booting @ {}".format(datetime.datetime.utcnow()))

        # Necessary since fork creates multiple instances of myobs and each one
        # will call it's __del__ on completion and __del__ shutdown...
        stndrv.halt_observingstate_when_finished = False
        stndrv.exit_check = False

        # BEGIN Dummy or hot beam start: (takes about 10sec)
        # TODO: This seems necessary, otherwise beamctl will not start up next time,
        #       although it should not have to necessary.)
        print("Running warmup beam... @ {}".format(datetime.datetime.utcnow()))
        stndrv.stationcontroller.run_beamctl(beamletIDs, subbandNrs, band, pointing)
        stndrv.stationcontroller.rcusetup(bits,
                                         attenuation)  # setting bits also seems necessary
        stndrv.stationcontroller.stopBeam()
        # END Dummy or hot start

        print("Pause {}s after boot.".format(pause))
        time.sleep(pause)

        # Real beam start:
        print("Now running real beam... @ {}".format(datetime.datetime.utcnow()))
        beamctl_CMD = stndrv.stationcontroller.run_beamctl(beamletIDs, subbandNrs, band,
                                                          pointing)
        rcu_setup_CMD = stndrv.stationcontroller.rcusetup(bits, attenuation)
        nw = datetime.datetime.utcnow()
        timeleft = st - nw
        if timeleft.total_seconds() < 0.:
            starttimestr = nw.strftime("%Y-%m-%dT%H:%M:%S")
        print("(Beam started) Time left before recording: {}".format(
            timeleft.total_seconds()))

        REC = False
        if REC == True:
            bf_data_dir = stndrv.bf_data_dir
            port0 = stndrv.bf_port0
            stnid = stndrv.stationcontroller.stnid
            bfbackend.rec_bf_streams(starttimestr, duration, lanes, band, bf_data_dir,
                                     port0, stnid)
        else:
            print("Not recording")
            time.sleep(duration)
        sys.stdout.flush()
        stndrv.stationcontroller.stopBeam()
        headertime = datetime.datetime.strptime(starttimestr, "%Y-%m-%dT%H:%M:%S"
                                                ).strftime("%Y%m%d_%H%M%S")
        obsinfo = dataIO.ObsInfo(stndrv.stationcontroller.stnid,
                                 stndrv.project, stndrv.observer)
        obsinfo.setobsinfo_fromparams('bfs', headertime, beamctl_CMD, rcu_setup_CMD, "")
        bsxSTobsEpoch, datapath = obsinfo.getobsdatapath(stndrv.LOFARdataArchive)
        obsinfo.create_LOFARst_header(datapath)
        stndrv.halt_observingstate_when_finished = shutdown  # Necessary due to forking

    def do_acc(self, band, duration, pointsrc):
        """Record acc data for one of the LOFAR bands over a duration on all stations.
        """
        duration = int(eval(duration))
        for stndrv in self.stationdrivers:
            accdestdir = stndrv.do_acc(band, duration, pointsrc)
            print("Saved ACC data in folder: {}".format(accdestdir))

    def do_bsxST(self, statistic, freqbnd, integration, duration, pointsrc,
                 when='NOW', allsky=False):
        """Records bst,sst,xst data in one of the LOFAR bands and creates a header file
        with observational settings on all stations.
        """
        duration = int(math.ceil(eval(duration)))
        frqbndobj = modeparms.FrequencyBand(freqbnd)
        if when != 'NOW':
            now = datetime.datetime.utcnow()
            try:
                st = datetime.datetime.strptime(when, "%Y-%m-%dT%H:%M:%S")
            except ValueError:
                raise ValueError('Start time format not understood')
            maxminsetuptime = datetime.timedelta(
                seconds=0)
            d = (st - maxminsetuptime) - now
            timeuntilboot = d.total_seconds()
            if timeuntilboot < 0.:
                timeuntilboot = 0
            print("Will start after " + str(
                timeuntilboot) + " seconds...")
            time.sleep(timeuntilboot)
        for stndrv in self.stationdrivers:
            if allsky and 'HBA' in frqbndobj.antsets[0]:
                stndrv.do_SEPTON(statistic, frqbndobj, integration, duration)
            else:
                try:
                    stndrv.bsxST(statistic, frqbndobj, integration, duration, pointsrc)
                except RuntimeError as rte:
                    print("Error: {}".format(rte))

    def do_tbb(self, duration, band):
        """Record Transient Buffer Board (TBB) data from one of the LOFAR bands for
        duration seconds on all stations.
        """
        duration = float(eval(duration))
        for stndrv in self.stationdrivers:
            stndrv.do_tbb(duration, band)