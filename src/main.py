from klampt import *
from common import *
from math import *
import sensor,random,time
import numpy as np

##################### SETTINGS ########################

diff_levels = ['easy','medium','hard']
difficulty = diff_levels[1]
omniscient_sensor = False
verbose = False
random_seed = random.seed() #12345

def plotPoint(p,name="temp",color=(0,1,0,1),robot=None):
    x,y,z = p if not robot else robot.getLink(6).getWorldPosition(p)
    kviz.update_sphere(name,x,y,z,0.03)
    kviz.set_color(name,color)
    return

################ STATE ESTIMATION #####################

class MyKalmanFilter:
    """Kalman filter for state estimation"""
    def __init__(self):
        a,dt = -9.8, 0.02
        eps_mu = [5,5,15,1,1,3]
        eps_mu0 = [3,3,5,2,2,3]
        # State
        self.mu0 = [1.6,-0.08,4.8,-2.5,-0.1,2.4]
        self.cov0 = np.diag(np.square(eps_mu0))
        self.F = np.eye(6) + np.eye(6,k=3)*dt
        self.g = np.array([[0],[0],[a/2*dt**2],[0],[0],[a*dt]])
        self.SigmaX = np.diag(np.square(eps_mu))
        # Observation
        self.H = np.eye(6)[0:3]
        self.j = np.zeros((3,1))
        self.SigmaZ = np.diag(np.square(eps_mu[0:3]))

    def createEstimate(self,name,z):
        """Creates new state estimate"""
        obj = ObjectStateEstimate(name,self.mu0,self.cov0)
        return self.updateWithObservation(obj,z)

    def updateWithObservation(self,obj,z):
        """Updates the state estimate with the new observation"""
        mu,z = np.asarray([obj.x]).T, np.asarray([z]).T
        obj.x,obj.cov = kalman_filter_update(mu,obj.cov,self.F,self.g,self.SigmaX,self.H,self.j,self.SigmaZ,z)
        obj.x = obj.x.T.tolist()[0]
        return obj

    def updateWithPrediction(self,obj):
        """Predicts the state at the next time step"""
        mu = np.asarray([obj.x]).T
        obj.x,obj.cov = kalman_filter_predict(mu,obj.cov,self.F,self.g,self.SigmaX)
        obj.x = obj.x.T.tolist()[0]
        return obj

class MyObjectStateEstimator:
    """State estimator that will provide a state estimate given
    CameraColorDetectorOutput readings."""
    def __init__(self):
        # Camera parameters
        cameraRot = [0,-1,0,0,0,-1,1,0,0]
        self.Tsensor = (so3.mul(so3.rotation([1,0,0],-0.1),so3.mul(so3.rotation([0,-1,0],radians(90)),cameraRot)),[-1.5,-0.5,0.25])
        self.fov = 90
        self.w,self.h = 320,240
        self.dmax = 5
        # State estimates
        self.objectEstimates = None
        self.filter = None
        self.dt = 0.02
        self.reset()

    def reset(self):
        """Resets state estimator"""
        self.filter = MyKalmanFilter()
        self.objectEstimates = MultiObjectStateEstimate([])

    def update(self,observation):
        """Produces an updated MultiObjectStateEstimate given CameraColorDetectorOutput
        sensor reading."""
    #   Update observed objects
        observed = []
        for b in observation.blobs:
            name,pos = b.color+(1,), self.blobToWorld(b)
            obj = self.objectEstimates.get(name)
            if obj:
                self.filter.updateWithObservation(obj,pos)
            else:
                obj = self.filter.createEstimate(name,pos)
                self.objectEstimates.objects.append(obj)
            observed.append(name)

    #   Update unobserved objects
        tracking = [o.name for o in self.objectEstimates.objects]
        unobserved = set(tracking) - set(observed)
        for n in unobserved:
            obj = self.objectEstimates.get(n)
            self.filter.updateWithPrediction(obj)

    #   Plot objects
        # for t in tracking:
        #     target = self.objectEstimates.get(t)
        #     name = "state_" + str(target.name)
        #     color = target.name
        #     plotPoint(target.x[0:3],name,color)

        return self.objectEstimates

    def blobToWorld(self,blob):
        """Converts image coordinates to world coordinates."""
        bs,xscale = 0.1, tan(radians(0.5*self.fov))*self.w/2
        ls = (blob.w+blob.h)/2. if abs(blob.w-blob.h) <= 2 else max(blob.w,blob.h)
        lz = xscale*bs/ls
        lx = lz/xscale*(blob.x-self.w/2)
        ly = lz/xscale*(blob.y-self.h/2)
        wp = se3.apply(self.Tsensor,[lx,ly,lz])
        # plotPoint(wp)
        return wp

################### CONTROLLER ########################

class MyController:
    """Attributes:
    - world: the WorldModel instance used for planning
    - objectStateEstimator: a StateEstimator instance
    - state: the state of the state machine
    """
    def __init__(self,world,robotController):
        self.world = world
        # Robot
        self.robot = world.robot(0)
        self.robotController = robotController
        self.state = None
        self.qdes = None
        # Objects
        self.stateEstimator = None
        self.objectEstimates = None
        self.targetName = None
        self.reset()

    def reset(self):
        """Resets the controller"""
        self.stateEstimator = MyObjectStateEstimator()
        self.objectEstimates = None
        self.state = 'waiting'
        self.qdes = self.robotController.getCommandedConfig()
        self.initVis()

    def updateTarget(self):
        """Updates the target object for the current time step."""
        target = self.objectEstimates.get(self.targetName)
        objs = self.objectEstimates.objects

    #   Update current target
        if target:
            p,v = target.x[0:3], target.x[3:6]
            if omniscient_sensor:
                if p == [0,0,-10] or (0.5 <= p[2] <= 1 and v[2] <= 0):
                    names = [o.name for o in objs]
                    i = names.index(target.name) + 1
                    self.targetName = names[i] if i < len(objs) else names[i-1]
                    self.state = 'waiting'
            else:
                if p[2] <= 1:
                    i = objs.index(target)
                    del objs[i]
                    self.state = 'waiting'
                self.targetName = objs[0].name if objs else None

    #   Create new target
        elif objs:
            self.targetName = objs[0].name
            self.state = 'waiting'

        return self.objectEstimates.get(self.targetName)

    def getDesiredPos(self,obj,zdes):
        """Calculates desired scoop position and orientation"""
        p,v,a = obj.x[0:3], obj.x[3:6], -9.8
        wp1,wp2 = None, None
        roots = np.roots([0.5*a,v[2],p[2]-zdes])
        real_roots = [i for i in roots if i.imag == 0 and i >= 0]
        if real_roots:
            t = max(real_roots) if v[2] >= 0 else min(real_roots)
            wp1 = vectorops.add(vectorops.madd(p,v,t),[0,0,0.5*a*t*t])  # target scoop position
            t -= 0.05
            wp2 = vectorops.add(vectorops.madd(p,v,t),[0,0,0.5*a*t*t])  # target scoop axis
        return (wp1,wp2)

    def updateDesiredConfig(self,obj):
        """Updates the desired configuration, given the target object"""
        wp1,wp2 = self.getDesiredPos(obj,1)
        if not wp1: return False

    #   Setup IK objective
        ee = self.robot.getLink(6)
        d = vectorops.distance(wp2,wp1)
        lp1 = [0.07,0,0.25]                     # local ee position
        lp2 = vectorops.add(lp1, [-d,0,0])      # local ee axis
        pcurr = ee.getWorldPosition(lp1)
        if vectorops.distance(pcurr,wp1) <= 0.1: return False

    #   Solve and return
        goal = ik.objective(ee, local=[lp1,lp2], world=[wp1,wp2])
        solved = ik.solve_nearby(goal,maxDeviation=pi/2,numRestarts=10,feasibilityCheck=lambda:True)
        if solved: self.qdes = self.robot.getConfig()
        return solved

    def myPlayerLogic(self,dt,sensorReadings,objectStateEstimate,robotController):
        """Arguments:
        - dt: the simulation time elapsed since the last call
        - sensorReadings: the sensor readings given on the current time step
        - objectStateEstimate: a MultiObjectStateEstimate instance
        - robotController: a SimRobotController instance
        """
        target = self.updateTarget()
        if self.state == 'waiting' and target and self.updateDesiredConfig(target):
            self.robotController.setMilestone(self.qdes)
        return

    def loop(self,dt,robotController,sensorReadings):
        """Called every control loop (every dt seconds)
        Input:
        - dt: the simulation time elapsed since last call
        - robotController: a SimRobotController instance
        - sensorReadings: a dictionary mapping sensor names to data
        """
    #   Update objects
        stateEstimates = None
        if 'blobdetector' in sensorReadings:
            stateEstimates = self.stateEstimator.update(sensorReadings['blobdetector'])
        elif 'omniscient' in sensorReadings:
            stateEstimates = OmniscientStateEstimator().update(sensorReadings['omniscient'])
        self.objectEstimates = stateEstimates

    #   Update robot and visualization
        self.myPlayerLogic(dt,sensorReadings,stateEstimates,robotController)
        self.updateVis()
        return

    def initVis(self):
        """Initialize visualization"""
        ghost = kviz.add_ghost()
        kviz.set_color(ghost,[0,1,0,0.5])
        kviz.set_ghost_config(self.qdes)

    def updateVis(self):
        """Update visualization"""
        kviz.set_ghost_config(self.qdes)
        objs = self.objectEstimates.objects
        for o in objs:
            label_o = "object_est" + str(o.name)
            label_t = "object_trace" + str(o.name)
            if o.name == self.targetName:
                p,v,a = o.x[0:3], o.x[3:6], -9.8
                x,y,z = p
                trace = [vectorops.add(vectorops.madd(p,v,t),[0,0,a*t*t/2]) for t in np.arange(0,1,0.05)]
                kviz.update_sphere(label_o,x,y,z,0.02)  # point
                kviz.set_color(label_o,o.name)
                kviz.update_polyline(label_t,trace)     # trace
                kviz.set_color(label_t,o.name)
            else:
                kviz.remove(label_o)
                kviz.remove(label_t)
