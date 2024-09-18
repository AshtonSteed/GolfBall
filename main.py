from copy import copy, deepcopy

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as inter
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.proj3d import persp_transformation


# Class representing the launch conditions of a golf ball
# velocity: Initial launch speed of the ball (mph)
# angle2: Angle of launch relative to the horizontal plane, positive upwards (degrees)
# alt2: Azimuth angle of launch relative to the forward direction, CCW from forward (degrees)
# rpm: Rate of spin of the ball (revolutions per minute)
# spinangle2: Angle of the spin axis relative to the horizontal plane, CCW from the perspective of the observer (degrees)
# spinalt: Angle of the spin axis relative to the Y-axis, 0° means pointing straight left (degrees)
class Launch:
    def __init__(self, velocity, angle2, alt2, rpm, spinangle2, teeh=0, spinalt=0):
        angle = angle2 * np.pi / 180  # Convert angle to radians
        alt = alt2 * np.pi / 180  # Convert altitude to radians
        spinangle = spinangle2 * np.pi / 180  # Convert spin angle to radians
        spin = rpm / 30 * np.pi  # Convert spin rate to rad/s
        self.velocity = 0.44704 * velocity * np.array([np.cos(angle) * np.cos(alt), np.cos(angle) * np.sin(alt), np.sin(angle)]) #mph to m/s
        self.spin = spin
        # Spin axis vector (normalized), based on the provided angles
        self.spinaxis = np.array(
            [-np.sin(spinalt), np.cos(spinalt) * np.cos(spinangle), np.cos(spinalt) * np.sin(spinangle)])


# Class representing the properties of the surrounding air
# wind: Wind velocity vector at a height of 10 meters above ground (m/s)
# density: Air density at sea level, 20°C (kg/m^3)
# viscosity: Air viscosity at sea level, 20°C (m^2/s)
class Air:
    def __init__(self, wind):
        self.v = wind
        self.density = 1.204  # Standard air density ( 20C)
        self.viscosity = 0.0000151  # Standard air viscosity

    def windshear(self, z):
        #return (self.v / np.log(5 / 0.03)) * np.log(z / 0.03)
        if z > 0.03:
            return (self.v / np.log(5 / 0.03)) * np.log(z / 0.03) # Assuming open country conditions
        else:
            return 0
# Class representing a golf ball, including its physical properties and current state
# launch: Launch object containing initial launch conditions
# air: Air object containing atmospheric conditions
class Ball:
    def __init__(self, launch, air):

        self.air = air
        self.radius = .0427 / 2.0  # Ball radius (m)
        self.mass = 0.045927  # Ball mass (kg)
        self.area = (self.radius ** 2) * np.pi  # Cross-sectional area of the ball (m^2)
        self.p = np.array([0.0, 0.0, 0.0])  # Initial position vector (m)
        self.v = launch.velocity - self.windv() # Initial velocity relative to air (m/s)
        self.angularvelocity = launch.spin  # Angular velocity (rad/s)
        self.spinaxis = launch.spinaxis  # Spin axis vector
        self.spin = self.radius * launch.spin / np.linalg.norm(self.v)  # Non-dimensional spin rate, unsure on 2

        #self.air = air  # Reference to air properties

        # Coefficients for drag, lift, and spin decay, fitted using cubic splines
        # Data sourced from Mizota
        self.coefs = np.array([
            inter.CubicSpline(
                [0.0335, 0.0734, 0.1355, 0.2215, 0.3050, 0.4203, 0.5380, 0.7411, 1.0462, 1.3382],
                [0.2146, 0.2277, 0.2453, 0.2781, 0.3153, 0.3657, 0.4161, 0.4599, 0.5058, 0.5365],
                bc_type='natural'),  # Drag coefficient (CD) as a function of spin rate
            inter.CubicSpline(
                [0,0.0314, 0.04878, 0.0741, 0.1102, 0.1532, 0.2130, 0.2711, 0.3451, 0.4298, 0.5846, 0.8130, 1.2346],
                [0,0.0971, 0.1146, 0.1405, 0.1708, 0.2053, 0.2505, 0.2849, 0.3236, 0.3602, 0.3947, 0.4292, 0.4701],
                bc_type='natural'),  # Lift coefficient (CL) as a function of spin rate, are zeros correct?
            inter.CubicSpline(
                [0,0.0113,0.0206,0.0404,0.0737,0.1476,0.2690, 0.3979,0.5242,0.6440],
                [0,0.00142,0.00146,0.00151,0.00167,0.0019, 0.00296,0.00437,0.00627,0.00948],
                bc_type='natural'),  # Moment coefficient (CM) as a function of spin rate

        ])

    # Calculate the Reynolds number for the ball in its current state
    def re(self):
        return self.air.density * np.linalg.norm(self.v) * 2 * self.radius / self.air.viscosity

    # Retrieve the lift coefficient based on the current spin rate
    def liftcoef(self):

        return self.coefs[1](self.spin) * 1


    # Retrieve the drag coefficient based on the current spin rate
    def dragcoef(self):

        return self.coefs[0](self.spin) * 1


    def decaycoef(self):
        #print(self.angularvelocity * self.radius/ np.linalg.norm(self.v))
        return self.coefs[2](self.spin)

    def windv(self):
        return self.air.windshear(self.p[2])
    # Spin Decay using aerodynamic torque coefficients
    def spinrate(self, dt):
        self.angularvelocity -= (5 *np.pi * np.dot(self.v, self.v) * self.air.density  * self.radius * self.decaycoef()) / (2 * self.mass) * dt
        #self.angularvelocity -=  dt * self.angularvelocity/ 20 exponential decay
        self.spin = self.radius * self.angularvelocity / np.linalg.norm(self.v) # Unsure on the 2
        #print((5 *np.pi * np.dot(self.v, self.v) * self.air.density  * self.radius * self.decaycoef()) / (2 * self.mass), self.angularvelocity/ 20)

    # Calculate the net acceleration on the golf ball
    def acceleration(self):
         # Calculate the direction of the lift force (perpendicular to both velocity and spin axis)
        liftdir = np.cross(self.v, self.spinaxis)
        liftdir /= np.linalg.norm(liftdir)

        # Calculate the square of the velocity magnitude
        v2 = np.dot(self.v, self.v)

        # Calculate the force coefficients based on current state
        reball = self.air.density * v2 * self.area / (2 * self.mass)
        liftcoef = self.liftcoef()
        lifta = liftcoef * reball * liftdir  # Lift acceleration (m/s^2)

        # Calculate the drag force acting opposite to the velocity
        backwards = -self.v / np.linalg.norm(self.v)
        draga = self.dragcoef() * reball * backwards  # Drag acceleration (m/s^2)
        #print(draga, np.linalg.norm(draga))
        # Constant gravitational acceleration (m/s^2)
        gravity = np.array([0, 0, -9.81])

        # Sum of forces acting on the ball
        a = lifta + draga + gravity
        #print(np.linalg.norm(self.v), self.angularvelocity, self.spin)
        return a

    # Advance the ball's state by one time step (dt) Using the implicit midpoint method
    def step(self, dt):

        # Save old position and velocity
        pold = copy(self.p)
        vold = copy(self.v)
        wold = copy(self.windv())
        sold = copy(self.angularvelocity)

        # Initial guess for midpoint
        self.p += (self.v + self.windv()) * 0.5 * dt
        self.v = vold + wold - self.windv() + self.acceleration() * 0.5 * dt
        self.spinrate(dt * 0.5)

        # Iterate to refine midpoint using the implicit midpoint method
        ''' for i in range(0, 5):
            tempp = copy(self.p)
            tempv = copy(self.v)

            # Implicit midpoint updates
            self.p = pold + (self.v + self.windv()) * 0.5 * dt
            self.v = vold + wold - self.windv() + self.acceleration() * 0.5 * dt

            # Check convergence using a norm for vector difference
            if np.linalg.norm(tempp - self.p) < 0.001 and np.linalg.norm(tempv - self.v) < 0.001:
                break
        '''

        # No final Euler update of p and v, since midpoint is already updated
        #( WHY DOES THIS DOUBLE UPDATE??)
        self.p = pold + (self.v + self.windv()) * dt  # Full time step update for position
        self.v = vold + wold - self.windv() + self.acceleration() * dt  # Full time step update for velocity
        self.angularvelocity = sold
        # Spin decay, decreasing the angular velocity
        self.spinrate(dt)

    # Simulate the ball's flight until it hits the ground
    # dt: time step for the simulation (s)
    def simulate(self, dt):

        tracevp = [[], [], []]  # Initialize lists to store position and velocity
        while self.p[2] >= 0:  # Continue simulation while the ball is above ground
            tracevp[0].append(copy(self.p))  # Record current position
            tracevp[1].append(copy(self.v))  # Record current velocity
            tracevp[2].append([self.dragcoef(), self.liftcoef(), self.decaycoef()]) # Record current coefficients
            self.step(dt)  # Advance to the next time step

        return np.array(tracevp)  # Return the recorded trajectory data


def main():
    air = Air(np.array([0, 0, 0])) # Initialize Wind
    launch = Launch(160, 13, 0, 2700, 10, teeh=0) # Initial Launch Parameters, strong driver shot

    ball = Ball(launch, air)

    print(ball.coefs[0](0))
    print("Initial Spin Rate: ",ball.spin)
    print("Initial Reynolds Number: ", ball.re())
    print(ball.v) # reynolds number of ball

    dt = 0.005
    trace = ball.simulate(dt)




    final = ball.p

    print(ball.p)
    time = len(trace[0]) * dt

    carry = np.sqrt(ball.p[0] ** 2 + ball.p[1] ** 2)
    peak = np.max(trace[0, :, 2])



    mty = 1.0936133
    print("Carry Distance: ", carry *mty, "Yards")
    print("Forward Distance: ", final[0] *mty, "Yards")
    print("Drift Distance: ", final[1]*mty, "Yards")
    print("Max Height: ", peak * mty, "Yards")
    print("Flight Time: ", time, "Seconds")
    print("Final Spin:", ball.angularvelocity * 30 / np.pi, "RPM")
    print("Descent Angle:", 90 - np.arcsin(np.linalg.norm(np.cross(ball.v, np.array([0, 0, 1]))/np.linalg.norm(ball.v))) * 180 / np.pi)
    # 3d Vector Vis

    fig = plt.figure()# Add a 3D subplot
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1.0, 1.0, 1.0])

    # Set labels
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Height [m]')

    ax.set_box_aspect([1.0, 1.0, 1.0])
    # Plot path
    ax.plot(trace[0, :, 0], trace[0, :, 1], trace[0, :, 2], color='red')

    # Plot Ground line
    ax.plot(trace[0, :, 0], trace[0, :, 1], 0)

    # Velocity Plot
    ax.plot(trace[1, :, 0], trace[1, :, 1], trace[1, :, 2], color='green')
    #ax.plot(trace[1, :, 0] + trace[0, :, 0], trace[1, :, 1] + trace[0, :, 1], trace[1, :, 2] + trace[0, :, 2], color='green')

    # Show vectors for initial velocity and spin axis
    ax.quiver(*[0,0,0], *launch.velocity, color='black')
    ax.quiver(*[0, 0, 0], *launch.spinaxis * 30, color='grey' )


    set_axes_equal(ax)
    plt.show()


    #ax2 = fig.add_subplot(211)
    #x = np.linspace(0, 50, 100)
    #ax2.plot(np.sqrt(trace[1, :, 0] ** 2 + trace[1, :, 2] ** 2))
    #ax2.plot(trace[1, :, 2])

    # Show the plot


   # ax2 = fig.add_subplot(211)

    #x = np.linspace(0, 0.3, 100)

    #ax2.plot(x, ball.coefs[0](x))
    #ax2.plot(x, ball.coefs[1](x))

    #ax2.plot(trace[2, :, 0], color='red')
    #ax2.plot(trace[2, :, 1], color='blue')
    #ax2.plot(trace[2, :, 2], color='green')






def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])




if __name__ == '__main__':
    main()