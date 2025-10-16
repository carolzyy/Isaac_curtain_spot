from NatNetClient import NatNetClient
import time
import threading
import cv2
import pyrealsense2 as rs
# Configuration
GRIPPER_ID = 47                # OptiTrack rigid body ID for gripper
SPOT_ID = 48 # with z_up              # OptiTrack rigid body ID for Turtlebot_test (body control)
NATNET_SERVER = "10.0.0.229"   # Motive computer IP
LOCAL_IP = "10.0.0.224"         # Local machine IP
SERVER_PORT = 9000
UPDATE_RATE = 30  # Hz

class MarkerState:
    def __init__(self):
        self._lock = threading.Lock()
        self._data = {
            'rigid': {},  # Stores arm joint angles
            'markers': {},  # Stores body position and orientation

        }

    def update_rigid(self, rigid_data):
        with self._lock:
            self._data['rigid'] = rigid_data.copy()

    def update_marker(self, markers_data):
        with self._lock:
            self._data['markers'] = markers_data.copy()

    def get_data(self):
        with self._lock:
            return {
                'rigid': self._data['rigid'].copy(),
                'markers': self._data['markers'].copy(),

            }

    def get_filter_data(self):
        print(self._data)
        if len(self._data['rigid']) >0:
            ee_position = self._data['rigid'][0]
            position_list = self._data['markers']

            filted_marker_list = get_marker_state(ee_position,position_list)
        
        return filted_marker_list




def data_updater( data, update_rate=20):
    """Continuously updates global state with smoothed OptiTrack data"""

    def receive_labeled_marker(labeled_marker_list):
        postion_list = []
        

        for i,marker in enumerate(labeled_marker_list):
            print(f'id is {marker.model_id}: {marker.pos}')
            if marker.model_id == 0:
                postion_list.append(marker.pos)
                #
                
        data.update_marker(postion_list)

    def rigid_body_listener(rigid_id, position, rotation):
        #print(rigid_id)
        print(f' ID{rigid_id} position is {position},rotation is {rotation}')
        # rotation is

        if rigid_id == SPOT_ID:
            data.update_rigid([position, rotation])

    client = NatNetClient()
    client.rigid_body_listener = rigid_body_listener
    client.labeled_marker_listener = receive_labeled_marker
    client.server_ip_address = NATNET_SERVER
    client.local_ip_address = LOCAL_IP
    client.print_level = 0

    client.run('d')

import numpy as np
def get_marker_state(ee_postion,position_list,rod_length=1.5, radius=0.05):
    #state = self.get_latest()
    ee_pos = ee_postion
    #ee_rot = state['rigid']['rot']

    filtered_markers = []
    for i,marker_pos in enumerate(position_list):
        marker_pos = np.array(marker_pos)

        # Project vec onto rod_direction
        distance = np.linalg.norm(marker_pos - ee_pos)
        
        if distance<rod_length:
            filtered_markers.append(marker_pos)
            #print(f'ID{i},position is {marker_pos}')

    elem_pos = np.array(filtered_markers)-ee_pos
    ele_mid = np.mean(elem_pos,axis=0)

    #points_sorted = elem_pos[np.argsort(elem_pos[:, 0])]

        # Choose 4 equally spaced indices
    num_samples = 4
    indices = np.linspace(0, len(filtered_markers) - 1, num=num_samples, dtype=int)

        # Select those points
    selected_points = elem_pos[indices]
    print(elem_pos)
    #print(selected_points)

        
    return np.array(selected_points),elem_pos,ele_mid

# Modified main execution
#if __name__ == "__main__":
def test():
    # Create and start threads
    position  = MarkerState()
    data_thread = threading.Thread(
        target=data_updater,
        args=(position,),
        daemon=True
    )

    data_thread.start()
    try:
        while True:
            time.sleep(5) # seconds

            _,point_list,ele_mid = position.get_filter_data()
            #show_3d_point(point_list=point_list,ref_point=ele_mid)
            print(point_list)
    except KeyboardInterrupt:
        print("Main thread interrupted. Exiting.")



def show_3d_point(point_list,ref_point = np.array([0, 0., 0.])):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # List of 3D points
    points = point_list

    # Reference point
    

    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue', label='Points')
    ax.scatter(ref_point[0], ref_point[1], ref_point[2], c='red', s=100, label='Reference Point')

    # Add labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Point Visualization')
    ax.legend()

    plt.show()

class RealSenseReader:
    def __init__(self):
        self.image_state = {}
        self.lock = threading.Lock()
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.running = False
        self.show_red_border = False

        # Configure streams
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    def _run(self):
        #self.pipeline.start(self.config)
        # Create colorizer object
        colorizer = rs.colorizer()
        try:
            while self.running:
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                timestamp = time.time()

                if not color_frame or not depth_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())

                with self.lock:
                    self.image_state = {
                        'rgb': color_image,
                        'depth': depth_image,
                        #'timestamp': timestamp
                    }

                time.sleep(0.005)
        finally:
            self.pipeline.stop()

    def start(self):
        self.running = True
        profile = self.pipeline.start(self.config)
        device = profile.get_device()
        sensors = device.query_sensors()
        exposure_value = 30
        
        for sensor in sensors:
                if sensor.supports(rs.option.enable_auto_exposure):
                    sensor.set_option(rs.option.enable_auto_exposure, 0)
                if sensor.supports(rs.option.exposure):
                    sensor.set_option(rs.option.exposure, exposure_value) #exposure_value
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self.running = False
        self.pipeline.stop()

    def get_latest(self):
        with self.lock:
            return self.image_state.copy()

    def _display_images(self):
        if len(self.image_state)>0:
            color_display = self.image_state['rgb'].copy()
            depth_display = self.image_state['depth'].copy()

            if self.show_red_border:
                current_time = time.time()
                if current_time - self.red_border_start_time < 0.5:  # Show border for 0.5 seconds
                    border_color = (0, 0, 255)  # Red in BGR
                    border_thickness = 10
                    cv2.rectangle(color_display, (0, 0), (639, 479), border_color, border_thickness)
                    cv2.rectangle(depth_display, (0, 0), (639, 479), border_color, border_thickness)
                else:
                    self.show_red_border = False

            cv2.imshow('RealSense Color', color_display)
            cv2.imshow('RealSense Depth', depth_display)
    def start_display(self):
        while True:
            self._display_images()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
def test_rs():  
    rs_reader = RealSenseReader()
    rs_reader.start()
    # Optionally, start a display loop
    rs_reader.start_display()


def test_opt_reader():   
    from wasd import OptiTrackReader
    _natnet_client = NatNetClient()
    _opti_thred = OptiTrackReader(_natnet_client)
    _opti_thred.start()
    robot_pos,robot_rot,elem_pos,filtered_point =_opti_thred.get_state(trans=np.array([0,0,0]))
    print(robot_pos)
    
test_rs()