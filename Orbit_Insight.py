# COSC2676/COSC2752 Programming Fundamentals for Scientists

# Coding assignment 2
# Student Name: Mark Sukhov
# Student ID: s3664377

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import re
import urllib.request
from matplotlib.widgets import TextBox
import warnings

# Global variable declarations 
# Number of (x, y, z) coordinate sets - affects the animation speed
N = 500
# Earth's gravitation parameter (km^3 / s^2)
mu = 398600 
# Radius of Earth (km)
R = 6378 
# Key to store a pressed key representing chosen orbit type (pie plot textbox)
o_key = None




def save_parameters(r, v, orbitals):
    """
        Saves the calculated parameters of a user selected orbit
        to a csv file located in the local folder "Keplerian_Orbits"
        The file name is the position (r) vector rounded to int form.
        The first 2 lines in the file correspond to the two state vectors.

                Parameters:
                        r (array): position vector
                        v (array): velocity vector
                        orbitals (dictionary): dictionary containing the orbital parameters
                                               matching the state vectors

                Returns:
                        None
    """
    fname = str(int(r[0]))+'i+'+str(int(r[1]))+'j+'+str(int(r[2]))+'k'
    with open("Keplerian_Orbits/" + fname + ".csv", 'w') as f:
        print(r, file=f)
        print(v, file=f)
        for key in orbitals:
            print(orbitals[key], file=f)

def extract_elements(data):
    """
        Extracts orbital elements from an array that represents
        the third line of a TLE format with only relevant parameters included
        Then performs appropriate calculations before returning a dictionary containing
        orbital parameters needed to plot the orbit

                Parameters:
                        data (array): an array containing information extracted from the third line of a TLE

                Returns:
                        orbitals (dictionary): a dictionary containing the needed orbital parameters
                        to plot the orbit
    """
    orbitals = dict()
    try:
        # Name of satellite
        name = data[0].strip()
        # Inclination
        inc = float(data[1]) * np.pi / 180
        # Right ascension of ascending node
        RAAN = float(data[2]) * np.pi / 180
        # Eccentricity - Need to add decimal in front of e
        e = float("." + data[3])
        # Omega = argument of perigee
        omega = float(data[4]) * np.pi / 180
        # Mean motion = revolutions per day 
        revs = float(data[5])
        # Revolutions per second - 1/revs = period (T)
        revs = revs / (24 * 60 * 60)
        # Semi-major axis
        a = (mu / ((2 * np.pi * revs) ** 2)) ** (1.0 / 3)
        # Semi-minor axis
        b = a * np.sqrt(1 - e ** 2)
        # Distance from the centre of the ellipse to the centre of the Earth
        c = a * e

        orbitals.update({"name" : name.strip('0'), "e": e, "a": a, "i": inc, "RAAN": RAAN, 
                         "omega": omega, "b": b, "c" : c, "T" : 1. / revs})

        return orbitals
    except RuntimeWarning:
        print("Warning: Chosen state vectors do not describe an orbit around Earth.")
        exit(1)

def transform_coords(orbitals):
    """
        Takes the dictionary which contains the orbital parameters in Keplerian format.
        Rotates the orbit from the celestial equatorial plane to the new plane 
        of orbit that corresponds to the parameters using the principles of transformation matrices.

                Parameters:
                        orbitals (dictionary): a dictionary containing the needed orbital parameters
                        to plot the orbit

                Returns:
                        coords (array): an array containing the transformed x, y, and z coordinates of the orbit
    """
    # Now we build the rotation matrix from the three rotations of the orbital plane
    # 1. Rotating the plane RAAN degrees anticlockwise along the z-axis.
    RAAN_rotation_x = [np.cos(orbitals["RAAN"]), -np.sin(orbitals["RAAN"]), 0]
    RAAN_rotation_y = [np.sin(orbitals["RAAN"]), np.cos(orbitals["RAAN"]), 0]
    RAAN_rotation_z = [0, 0, 1]
    # 3x3 rotation matrix for the RAAN tilt
    RAAN_rotation = np.array([RAAN_rotation_x, RAAN_rotation_y, RAAN_rotation_z])

    # 2. Rotating the plane i (inc) degrees along the new x-axis out of  the celestial equatorial plane. 
    i_rotation_x = [1, 0, 0]
    i_rotation_y = [0, np.cos(orbitals["i"]), -np.sin(orbitals["i"])]
    i_rotation_z = [0, np.sin(orbitals["i"]), np.cos(orbitals["i"])]
    # 3x3 rotation matrix for the inclination tilt
    i_rotation = np.array([i_rotation_x, i_rotation_y, i_rotation_z])

    # 3. Rotating the orbitalsit along its new plane (tilt) by omega degrees.
    omega_rotation_x = [np.cos(orbitals["omega"]), -np.sin(orbitals["omega"]), 0]
    omega_rotation_y = [np.sin(orbitals["omega"]), np.cos(orbitals["omega"]), 0]
    omega_rotation_z = [0, 0, 1]
    # 3x3 rotation matrix for the angle of perigee tilt
    omega_rotaion = np.array([omega_rotation_x, omega_rotation_y, omega_rotation_z])

    # perform the matrix multiplications
    # R is our rotation matrix built from three rotations of the orbital plane
    R = np.matmul(RAAN_rotation, i_rotation)
    R = np.matmul(R, omega_rotaion)

    # Initialise variables to store new axes before plotting the orbit
    x, y, z = [], [], []
    # Loop through the parametric orbit and compute the positions' x, y, z
    for i in np.linspace(0, 2 * np.pi, N):
        # Using the trigonometric interpretation of an ellipse:     
        x_param = [orbitals["a"] * np.cos(i)]
        y_param = [orbitals["b"] * np.sin(i)]
        z_param = [0]
        # positions represents the sets of coordinates of the orbit as it were 
        # on the original, pre-tilt, plane
        positions = np.array([x_param, y_param, z_param])

        # Multiply the transformation matrix R by the original ellipse coordinates
        mul1 = np.matmul(R, positions)

        # focal represents the vector connecting the centre of the orbit and the centre of earth
        x_param = [orbitals["c"]]
        y_param = [0]
        z_param = [0]
        focal = np.array([x_param, y_param, z_param])

        # Multiply the transformation matrix R by the centre position vector to align
        # the centre with the transformed orbit
        mul2 = np.matmul(R, focal)

        # Compute the new coordinate
        P = mul1 - mul2

        # Store x, y, and z values to use in plotting
        x += [P[0]]
        y += [P[1]]                 # WTF IS THIS? modify to append and remove the concatenating $$$
        z += [P[2]]
    
    # Transform the lists of x, y, and z coordinates to one array
    x = np.concatenate(x)
    y = np.concatenate(y)
    z = np.concatenate(z)
    return np.array([x, y, z])

def update(num, dataset, point):
    """
        Utility function to animate a satellite in the selected orbit.
        FuncAnimation function in the matplotlib library requires passing this function as
        an argument to update the point coordinates at a selected interval.

                Parameters:
                        num (int): counter parameter used for animating. Loops from 1 - N
                        dataset (array): set of orbit coordinates to update at every iteration
                        point (line3d): line object representing the point being animated

                Returns:
                        None
    """
    # Update satellite (point) coordinates
    point.set_data(dataset[0, int(num):int(num+1)], dataset[1, int(num):int(num+1)])    
    point.set_3d_properties(dataset[2, int(num):int(num+1)])   

def visualise(coords, name="Selected orbit", r=None):
    """
        Takes the transformed set of coordinates and plots them.
        Creates a line object within the plot to animate an orbit following the path
        Plots a grid-based earth model in the origin point
        Saves a PNG format image if plot was produced from user inputed state vectors

                Parameters:
                        coords (array): an array containing the transformed x, y, and z coordinates of the orbit
                        name (string): a string representing the name of the chosen random orbit or the default value
                                        "Selected orbit" if plot was produced from specified state vectors
                        r (array): an array represeting the position vector of a specified state vector.
                        Used for naming generated saved files. Defaults to None if generated plot is of random orbit.

                Returns:
                        None
    """
    # Split the coordinates to 3 lists
    x = coords[0]
    y = coords[1]
    z = coords[2]

    # Draw plot 
    fig = plt.figure()
    ax = plt.axes(projection='3d', computed_zorder=False)

    # Plot the orbit
    ax.plot(x, y, z, color='orange', zorder=5)

    # Create a point in the axes representing the satellite
    point, = ax.plot(x[0:1], y[0:1], color='red', marker="o", zorder=6)

    # Define sphere parameters
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    earth_x = R * np.cos(u) * np.sin(v)
    earth_y = R * np.sin(u) * np.sin(v)
    earth_z = R * np.cos(v)
    # Plot the earth
    ax.plot_wireframe(earth_x, earth_y, earth_z, color="b", alpha=0.5, lw=0.5, zorder=0)
    
    if r is not None:
        name = str(r[0].round(1)) + 'i + ' + str(r[1].round(1)) + 'j +' + str(r[2].round(1)) + 'k'
    # Add labels
    plt.title('Orbit of: ' + name)
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.xaxis.set_tick_params(labelsize=5)
    ax.yaxis.set_tick_params(labelsize=5)
    ax.zaxis.set_tick_params(labelsize=5)
    ax.set_aspect('equal', adjustable='box')

    # Animate a satellite in orbit
    ani = animation.FuncAnimation(fig, update, interval=1, frames=N, fargs=(coords, point), blit=False)
    
    plt.show()

    # Save the 3D plot as png
    # Can also save it from the plot window itself
    if r is not None:
        fname = "Plots/" + str(int(r[0])) + 'i+' + str(int(r[1])) + 'j+' + str(int(r[2])) + 'k.png'
        fig.canvas.draw()
        image_array = np.array(fig.canvas.renderer.buffer_rgba())
        plt.imsave(fname, image_array)

def data_analysis_controller(option):
    """
        Controller function for option 1 (scraping internet orbit data based on user input)
        Purpose of this function is to handle flow of execution outside of the main function
        to keep code modular.
        Delegates functionality to other functions - keeps code cleaner

                Parameters:
                        option (string): user specified option - 'latest', 'active', or 'debris'
                        represents which dataset to read data from

                Returns:
                        None
    """

    # Read all orbits in specified location and split into 3 sets
    LEO, MEO, HEO = get_LMH_orbits(option)

    # Plot pie chart of orbit ratios
    plot_pie(len(LEO), len(MEO), len(HEO), option)

    # Simulate random orbit of chosen type
    if o_key == 'l':
        sim_random(LEO)
    elif o_key == 'm':
        sim_random(MEO)
    elif o_key == 'h':
        sim_random(HEO)
    else:
        print("Warning: Error during random simulation: wrong key.")
        exit(1)

def plot_pie(n_low, n_med, n_high, option):
    """
        Handles plotting a pie chart depicting the ratios of low, medium and high earth orbits
        in the chosen datset. 
        Modifies global variable o_key to l, m, or h based on user input
                Parameters:
                        n_low (int): number of low earth orbits in selected dataset
                        n_med (int): number of medium earth orbits in selected dataset
                        n_high (int): number of high earth orbits in selected dataset
                        option (string): represents the selected dataset for use in plot desciption

                Returns:
                        None
    """
    # Total number of orbits scanned
    n_all = n_low + n_med + n_high
    # Sizes of the pie slices
    sizes = [(n_low/n_all)*100, (n_med/n_all)*100, (n_high/n_all)*100]  
    # Labels for each slice
    labels = ['LEO', 'MEO', 'HEO']
    # Colors for each slice
    colors = ['yellowgreen', 'gold', 'lightcoral']

    # If chosen dataset has no orbits of certain type
    # filter out the zero values
    filtered_sizes = []
    filtered_labels = []
    filtered_colors = []
    for size, label, color in zip(sizes, labels, colors):
        if size > 0:
            filtered_sizes.append(size)
            filtered_labels.append(label)
            filtered_colors.append(color)
    
    # Plot the pie chart
    fig, ax = plt.subplots()
    ax.pie(filtered_sizes, labels=filtered_labels, colors=filtered_colors, autopct='%1.1f%%', startangle=140)

    # Add a title
    if option == 'latest':
        plt.title('Orbit type break-down of\nsatellites launched in\nthe last 30 days')
    elif option == 'active':
        plt.title('Orbit type break-down of\nactive satellites')
    elif option == 'debris':
        plt.title('Orbit type break-down of\ntracked debris')

    # Adjust the position of the plot, move plot to the left
    plt.subplots_adjust(left=0.05, right=0.6)

    # Add annotation
    annotation = """Satellites launched: {0}\nLow Earth Orbit: {1}\nMedium Earth Orbit: {2}\nHigh Earth Orbit: {3}"""
    annotation = annotation.format(n_all, n_low, n_med, n_high)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # place a text box in upper right in axes coords
    ax.text(0.9, 1.1, annotation, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    
    axbox = fig.add_axes([0.3, 0.05, 0.5, 0.075])
    text_box = TextBox(axbox, "Choose orbit\nl, m, or h:", textalignment="left", label_pad=0.03)
    text_box.on_submit(choice)


    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')  
    plt.show()

def get_LMH_orbits(option):
    """
        Based on the value of the passed 'option' parameter this function selects the
        address for the specified data set. 
        For 'latest' and 'active - these are online datasets accessed by the urllib library.
        For 'debris' - this dataset is accessed locally from a text file.
        Scans the dataset and categorizes all read orbits into low, medium and high and returns
        them as 3 lists
                Parameters:
                        option (string): represents the user selected dataset

                Returns:
                        LEO (list): a list representing the parameters of all low earth orbits
                        MEO (list): a list representing the parameters of all medium earth orbits
                        HEO (list): a list representing the parameters of all high earth orbits
    """
    # The three addresses of the available datasets
    url_latest = "https://celestrak.org/NORAD/elements/gp.php?GROUP=last-30-days&FORMAT=tle"
    url_active = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"
    debris_fname = "tracked_debris.txt"

    # Acess the file/url specified by user
    if option == 'latest':
        try:
            page = urllib.request.urlopen(url_latest).read().decode()
            # Save the data locally
            with open("latest.txt", 'w') as f:
                print(page.replace('\n', ''), file=f)
        except urllib.error.URLError:
            print ("Warning: Failed to retrieve latest data.")
            exit(1) 
    elif option == 'active':
        try:
            page = urllib.request.urlopen(url_active).read().decode()
            # Save the data locally
            with open("active.txt", 'w') as f:
                print(page.replace('\n', ''), file=f)
        except urllib.error.URLError:
            print ("Warning: Failed to retrieve active satellite data.")
            exit(1) 
    elif option == 'debris':
        with open(debris_fname, 'r') as f:
            page = f.read()

    # Regex pattern to extract needed values from the TLE
    tle_pattern = r'(\S+ *\S*) *.*\n.*\n2 +\S+.+ +(\S+) +(\S+) +(\d+) +(\S+) +\S+ +(\S{1,10})'
    # Create an array of all partial TLE's in the chosen file/url
    TLE_list = np.array(re.findall(tle_pattern, page))
    # Isolate the mean motion values for orbit classificatione
    revs = TLE_list[:,5].astype(float) 

    # Lowest Mean Motion (revolutions per day) values for each orbit - used for classification
    MM_leo = 12
    MM_meo = 1
    # Array containing indicies of LEO
    leo_i = np.where(revs > MM_leo)
    # Array containing indicies of MEO 
    heo_i = np.where(revs < MM_meo)

    # Combine the leo_i and heo_i indicies 
    temp = np.concatenate((leo_i[0], heo_i[0]))
    # Create a complete set of indicies 
    all = np.linspace(0, len(revs), len(revs), endpoint=False).astype(int)
    # Compute the MEO indicies as the difference between the complete set
    # and the truncation of leo_i and heo_i
    meo_i = np.setdiff1d(all, temp).tolist()

    # Construct seperate listss for each orbit type
    LEO, MEO, HEO = list(TLE_list[leo_i]), list(TLE_list[meo_i]), list(TLE_list[heo_i])
    return LEO, MEO, HEO

def choice(expr):
    """
        Function to get user input from textbox under pie chart
        Entered value is stored in the global variable 'o_key'
        This was the best way I came up with for storing the textbox input string
                Parameters:
                        expr (string): a string read from the textbox field in the pie-chart window

                Returns:
                        None
    """
    expr = expr.lower()
    global o_key
    orbits = {'l' : "Low earth orbit",
              'm' : 'Medium earth orbit',
              'h' : 'High earth orbit'}
    if expr not in orbits.keys():
        print("Warning: must choose between 'l', 'm', and 'h'.")
    else:
        print(orbits[expr] + " chosen.")
        o_key = expr
        # Close the pie chart when enter is pressed
        plt.close('all')

def process_vector(v):
    """
        Processes user input to turn a string representing a vector into a vector of the form [x y z]
        Utilises re library to find matches
                Parameters:
                        v (string): string representing user input (vector)

                Returns:
                        vector (array): an array representing the user provided vector
    """
    # Extract the i, j, and k components
    vector = re.findall(r"^([+|-]?\S*)i([+|-]?\S*)j([+|-]?\S*)k", v.replace(' ', ''))

    # If no match found (or more than one) print error and exit
    if len(vector) != 1:
        print("Warning: vector must be in a 'xi+yj+zk' format.")
        exit(1)
    vector = list(vector[0])
    for i in range(len(vector)):
        if vector[i] == '':
            vector[i] = "1"
        elif vector[i] == '-':
            vector[i] = "-1"
    
    return np.array(vector).astype(float)

def keplerify(r, v):
    """
        Takes the two state vectors specified by user and transforms them into 
        corresponding parameters in keplerian form to be used for simulating the orbit

                Parameters:
                        r (array): an array representing a position vector
                        v (array): an array representing a velocity vector

                Returns:
                        orbitals (dictionary): dictionary containing the keplerian parameters of the 
                        orbit corresponding to the state vectors provided by the user
    """
    # filterwarnings makes it possible to catch run time warnings in calculations.
    warnings.filterwarnings("error")

    # Define dictionary to return
    orbitals = dict()
    try:
        # Calculate distance from centre of Earth
        distance = np.sqrt(np.dot(r, r))
        # Calculate speed
        speed = np.sqrt(np.dot(v, v))
        # Calculate radial velocity
        v_rad = np.dot(r, v)/distance
        # Calculate specific angular momentum vector
        h_vec = np.cross(r, v)
        # Calculate the magnitude of the specific angular momentum
        h_mag = np.sqrt(np.dot(h_vec, h_vec))
        # Calculate the inclination
        inc = np.arccos(h_vec[2] / h_mag)
        # Calculate the node line vector
        N_vec = np.cross(np.array([0, 0, 1]), h_vec)
        # Calculate the magnitude of the node line vector
        N_mag = np.sqrt(np.dot(N_vec, N_vec))
        # Calculate the right ascension of the ascending node angle
        RAAN = np.arccos(N_vec[0] / N_mag)
        if N_vec[1] < 0:
            RAAN = 2 * np.pi - RAAN
        # Calculate the eccentricity vector
        e_vec = (1 / 398600) * ((speed ** 2 - mu / distance) * r - distance * v_rad * v)
        # Calculate the magnitude of the eccentricity vector
        e_mag = np.sqrt(np.dot(e_vec, e_vec))
        # Calculate the argument of perigee angle
        omega = np.arccos(np.dot(N_vec / N_mag, e_vec / e_mag))
        if e_vec[2] < 0:
            omega = 2 * np.pi - omega 
        # Calculate the true anomaly angle
        theta = np.arccos(np.dot(e_vec / e_mag, r / distance))
        if v_rad < 0:
            theta = 2 * np.pi - theta
        # Calculate radius of perigee
        r_per = (h_mag ** 2) / (mu * (1 + e_mag))
        # Calculate radius of apogee
        r_apo = (h_mag ** 2) / (mu * (1 - e_mag))
        # Calculate semi-major axis
        a = (r_apo + r_per) / 2
        # Calculate semi-minor axis
        b = np.sqrt((a ** 2) * (1 - e_mag ** 2))
        # Calculate distance from the centre of the ellipse to the centre of the Earth
        c = a * e_mag
        # Calculate period
        T = (2 * np.pi * a ** (3 / 2))/np.sqrt(mu)

        # Update dictionary
        orbitals.update({"e": e_mag, "a": a, "i": inc, "RAAN": RAAN, 
                        "omega": omega, "b": b, "c" : c, "T": T, "h" : h_mag,
                        "theta" : theta, "rp" : r_per, "ra" : r_apo})

        # Return parameters
        return orbitals
    
    except RuntimeWarning:
        # If user provided invalid state vectors (= do not correspond to an orbit)
        # it is likely to encounter negative numbers when taking square roots
        # If this happens print warning and exit
        print("Warning: Unable to transfom state vector to Keplerian elements.")
        exit(1)

def sim_random(orbits):
    """
        Selects a random orbit from the input list and extracts its parameters,
        before transforming into appropriate coordinates and then visualises the orbit
                Parameters:
                        orbits (list): a list containing the orbital parameters of either
                        LEO, MEO or HEO.

                Returns:
                        None
    """
    if o_key is None:
        print("Warning: Orbit type not selected. Relaunch application.")
        exit(1)
    elif len(orbits) < 1:
        temp = {'h':'High Earth Orbits', 'm':'Medium Earth Orbits', 'l':'Low Earth Orbits'}
        print("Warning: Currently no", temp[o_key], "present in selected dataset.")
        print("Choose another orbit or dataset.")
        exit(1)

    # Select a random index 
    index = np.random.randint(0, len(orbits))

    # Extract parameters of randomly selected orbit
    orbitals = extract_elements(orbits[index])

    # Tansform orbit coordinates 
    coords = transform_coords(orbitals)

    # Simulate orbit
    visualise(coords, name=orbitals['name'])

def print_intro():
    # Print brief introduction and introduce the choices available
    welcome_text =  \
    """
    Welcome to Orbit Insight

    Orbit Insight is a robust tool for orbit simulation and visualization. 
    It offers comprehensive capabilities, including:

    - Transforming between Keplerian and Cartesian coordinates
    - Detailed orbit visualizations
    - Performing comparative analysis on live satellite data

    Dive into the dynamic world of satellite orbits with ease and precision.
    """
    print(welcome_text)
    print("Please follow the prompts:")
    print("Option 1: Analyse satellite datasets and simulate existing orbit.")
    print("Option 2: Enter state vectors to simulate specific orbit.")

def get_user_input():
    """
        Helper function to read input from the user.
        Prompts the user to choose between 2 options - simulating an orbit from specified state vectors or scrape a
        dataset to see its statistics. Based on this option, further prompts the user to either provide the state vectors 
        or choose the prefered dataset to look at.

                Parameters:
                        None

                Returns:
                        list: a list of either len = 1 or len = 2 based on user input.
                            len = 1 corresponds to option 1 and contains a string representing the chosen dataset
                            len = 2 corresponds to option 2 and contains two strings representing the two
                                    state vectors
    """
    # Prompt user to select option 1 or 2
    try:
        option = int(input("Enter option number (1 or 2) (default value = 1):").strip() or 1)
        # if option is not either 1 or 2, print a warning and shut down
        if option not in (1,2):
            print("Warning: unrecognized option. 1 or 2 only.")
            exit(1)
    except ValueError:
        # if a non integer value is entered, print a warning and shut down
        print("Warning: option must be an integer, 1 or 2.")
        exit(1)

    # Option 1 selected - Now select which dataset to analyse
    # 'Latest' and 'Active' access an online dataset
    # 'Debris' accesses an saved file "tracked_debris.txt"
    if option == 1:
        print("\nSelect a dataset from the following options ('Debris' default):")
        print("     'Latest' - Compare orbits of all satellites launched in the recent 30 days.")
        print("     'Active' - Compare orbits of all active satellites.")
        print("     'Debris' - Compare orbits of all actively tracked debris and space objects.")
        print("> Type 'Exit' to quit <")

        # Accept input until a valid choice is selected
        while True:
            chosen_set = input().strip().lower() or 'debris'
            if chosen_set == 'exit':
                exit(1)
            elif chosen_set not in ['active', 'latest', 'debris']:
                print("Warning: Invalid dataset specified.")
                print("Choose from the following: 'Active', 'Latest' or 'Debris'.")
                print("Or 'Exit' to quit.")
            else:
                break

        return [chosen_set]

    elif option == 2:
        try:
            # ISS position vector - default value for position vector input
            ISS_r = '-5978.58i-1957.75j-2582.42k' 
            # ISS velocityvector - default value for position vector input
            ISS_v = '-0.45i-5.5j+5.256k'
            print("Enter a position vector r = xi+yj+zk, or press Enter for default vectors (ISS position vector):")
            # Check if default vector was chosen: 
            # If yes, skip asking for velocity and use both default values
            r = input() or ISS_r
            if r == ISS_r:
                v = ISS_v
                r = process_vector(r)
                v = process_vector(v)
            else:
                r = process_vector(r)
                print("Enter a velocity vector v = xi+yj+zk, or press Enter for default vector (ISS velocity vector):")
                v = input()
                v = process_vector(v)
            
            return [r, v]
        except ValueError:
            # if a non int/float value is entered, print a warning and shut down
            print("Warning: position/velocity")
            exit(1)

def verify_orbit(orbitals):
    """
        Verifies that the 2 specified state vectors represent a valid orbit.
        If orbit is found to be invalid, prints a warning and exits the program.

                Parameters:
                        orbitals (dict): dictionary representing the parameters corresponding to the state vectors

                Returns:
                        None
    """
    rp = orbitals['rp']
    # Lowest altitude must be greater than the radius of Earth
    if rp < R:
        print("Warning: Chosen state vectors do not describe an orbit around Earth.")
        exit(1)

def state_vector_controller(r, v):
    """
        Controller function for option 2 (converting given state vectors to keplerian coordinates and plotting
        the corresponding orbit).
        Purpose of this function is to handle flow of execution outside of the main function
        to keep code modular.
        Delegates functionality to other functions - keeps code cleaner
                Parameters:
                        r (array): an array of 3 elemets each corresponding to i, j, or k component
                        of the position vector

                        v (array): an array of 3 elemets each corresponding to i, j, or k component
                        of the velocity vector

                Returns:
                        None
    """
    # Prepare coordinates, plot the orbit and save the parameters
    orbitals = keplerify(r, v)
    verify_orbit(orbitals)
    save_parameters(r, v, orbitals)
    coords = transform_coords(orbitals)
    visualise(coords, r=r)

# Main
def main():
    print_intro()
    user_input = get_user_input()

    # If user_input contains 1 element, option 1 was selected
    # Returned value contains the chosen url/file 
    if len(user_input) == 1:
        chosen_set = user_input[0]
        data_analysis_controller(chosen_set)
    # If user_input contains 2 elements, option 2 was selected
    # Returned value contains the 2 state vectors
    elif len(user_input) == 2:
        r, v = user_input[0], user_input[1]
        state_vector_controller(r, v)




if __name__ == "__main__":
    main()
