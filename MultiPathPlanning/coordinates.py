import numpy as np

waypoints = dict()
obstacles = dict()
specific_paths = dict()

waypoints["blocks"] = [
                np.array([0.0, 0.0, 5.0], dtype=np.float32), # starting point
                np.array([75.0, 0.0, 10.0], dtype=np.float32),
                np.array([50.0, 60.0, 8.0], dtype=np.float32),
                np.array([50.0, -60.0, 12.0], dtype=np.float32),
                np.array([20.0, 110.0, 15.0], dtype=np.float32),
                np.array([20.0, -110.0, 10.0], dtype=np.float32),
                np.array([-30.0, 70.0, 6.0], dtype=np.float32),
                np.array([-30.0, -70.0, 16.0], dtype=np.float32),
                np.array([-50.0, 0.0, 5.0], dtype=np.float32),
                np.array([-60.0, 110.0, 20.0], dtype=np.float32),
                np.array([-60.0, -110.0, 13.0], dtype=np.float32),
                np.array([-100.0, 0.0, 7.0], dtype=np.float32),
                np.array([-115.0, 80.0, 11.0], dtype=np.float32),
                np.array([-115.0, -80.0, 18.0], dtype=np.float32),
            ]

specific_paths["blocks"] = [ [], [], [] ]

obstacles["blocks"] = [
                np.array([19, -25, 0, 65, 22, 15]), # Obs 1
                np.array([27, 37, 0, 40, 40, 15]), # Obs 2
                np.array([10, 45, 0, 35, 70, 15]), # Obs 3
                np.array([40, 95, 0, 65, 120, 15]), # Obs 4
                np.array([-100, 95, 0, -75, 120, 15]), # Obs 5
                np.array([-80, 45, 0, -65, 60, 15]), # Obs 6
                np.array([-40, 12, 0, -15, 35, 15]), # Obs 7
                np.array([-40, -48, 0, -15, -25, 15]), # Obs 8
                np.array([-80, -63, 0, -65, -50, 15]), # Obs 9
                np.array([-100, -125, 0, -75, -100, 15]), # Obs 10
                np.array([10, -75, 0, 35, -48, 15]), # Obs 11
                np.array([40, -125, 0, 65, -100, 15]), # Obs 12
            ]

waypoints["neighbourhood"] = [
                np.array([0.0, 0.0, 5.0], dtype=np.float32), # starting point
                np.array([130.0, 0.0, 5.0], dtype=np.float32), # B
                np.array([127.0, -127.0, 5.0], dtype=np.float32), # C
                np.array([80.0, -128.0, 5.0], dtype=np.float32), # D
                np.array([0.0, -128.0, 5.0], dtype=np.float32), # E
                np.array([-125.0, -130.0, 5.0], dtype=np.float32), # F
                np.array([-128.0, 0.0, 5.0], dtype=np.float32), # G
                np.array([-125.0, 127.0, 5.0], dtype=np.float32), # H
                np.array([0.0, 127.0, 5.0], dtype=np.float32), # I
                np.array([127.0, 125.0, 5.0], dtype=np.float32), # J
                np.array([80.0, 0.0, 5.0], dtype=np.float32), # K
                np.array([80.0, -65.0, 5.0], dtype=np.float32), # L
                np.array([130.0, -65.0, 5.0], dtype=np.float32), # M
                np.array([130.0, 63.0, 5.0], dtype=np.float32), # N
                np.array([0.0, 63.0, 5.0], dtype=np.float32), # O
                np.array([0.0, -63.0, 5.0], dtype=np.float32), # P
                np.array([-128.0, -65.0, 5.0], dtype=np.float32), # Q
                np.array([-128.0, 63.0, 5.0], dtype=np.float32), # R
                np.array([-63.0, -130.0, 5.0], dtype=np.float32), # S
                np.array([-63.0, 0.0, 5.0], dtype=np.float32), # T
                np.array([-63.0, 128.0, 5.0], dtype=np.float32), # U
                np.array([64.0, 128.0, 5.0], dtype=np.float32), # V
            ]

obstacles["neighbourhood"] = [
    np.array([90, -120, 0, 120, -96.5, 15]), # Obs 1
    np.array([90, -91.25, 0, 120, -67.5, 15]), # Obs 2
    np.array([90, -62.5, 0, 120, -38.75, 15]), # Obs 3
    np.array([90, -33.75, 0, 120, -10, 15]), # Obs 4

    np.array([42.5, -120, 0, 70, -90, 15]), # Obs 5
    np.array([42.5, -80, 0, 70, -50, 15]), # Obs 6
    np.array([42.5, -40, 0, 70, -10, 15]), # Obs 7

    np.array([10, -120, 0, 37.5, -96.5, 15]), # Obs 8
    np.array([10, -91.25, 0, 37.5, -67.5, 15]), # Obs 9
    np.array([10, -62.5, 0, 37.5, -38.75, 15]), # Obs 10
    np.array([10, -33.75, 0, 37.5, -10, 15]), # Obs 11

    np.array([-33.75, -120, 0, -10, -96.5, 15]), # Obs 12
    np.array([-33.75, -91.25, 0, -10, -67.5, 15]), # Obs 13
    np.array([-33.75, -62.5, 0, -10, -38.75, 15]), # Obs 14
    np.array([-33.75, -33.75, 0, -10, -10, 15]), # Obs 15

    np.array([-120, -120, 0, -96.5, -96.5, 15]), # Obs 16
    np.array([-120, -91.25, 0, -96.5, -67.5, 15]), # Obs 17
    np.array([-120, -62.5, 0, -96.5, -38.75, 15]), # Obs 18
    np.array([-120, -33.75, 0, -96.5, -10, 15]), # Obs 19

    np.array([-62.5, -120, 0, -38.75, -96.5, 15]), # Obs 20
    np.array([-91.25, -120, 0, -67.5, -96.5, 15]), # Obs 21

    np.array([-86.5, -33.75, 0, -43.75, -10, 15]), # Obs 22

    np.array([-33.75, 120, 0, -10, 96.5, 15]), # Obs 26
    np.array([-33.75, 91.25, 0, -10, 67.5, 15]), # Obs 25
    np.array([-33.75, 62.5, 0, -10, 38.75, 15]), # Obs 24
    np.array([-33.75, 33.75, 0, -10, 10, 15]), # Obs 23

    np.array([-120, 120, 0, -96.5, 96.5, 15]), # Obs 30
    np.array([-120, 91.25, 0, -96.5, 67.5, 15]), # Obs 29
    np.array([-120, 62.5, 0, -96.5, 38.75, 15]), # Obs 28
    np.array([-120, 33.75, 0, -96.5, 10, 15]), # Obs 27

    np.array([-86.5, 120, 0, -43.75, 96.5, 15]), # Obs 32

    np.array([-86.5, 33.75, 0, -43.75, 10, 15]), # Obs 31

    np.array([90, 120, 0, 120, 90, 15]), # Obs 33
    np.array([90, 80, 0, 120, 50, 15]), # Obs 34
    np.array([90, 40, 0, 120, 10, 15]), # Obs 35

    np.array([50, 120, 0, 80, 90, 15]), # Obs 36
    np.array([50, 40, 0, 80, 10, 15]), # Obs 37

    np.array([10, 120, 0, 40, 90, 15]), # Obs 38
    np.array([10, 80, 0, 40, 50, 15]), # Obs 39
    np.array([10, 40, 0, 40, 10, 15]), # Obs 40
]

specific_paths["neighbourhood"] = [
    [
        np.array([0.0, 63.0, 35.0], dtype=np.float32),
        np.array([0.0, 127.0, 35.0], dtype=np.float32),
        np.array([-63.0, 128.0, 35.0], dtype=np.float32),
        np.array([-125.0, 127.0, 35.0], dtype=np.float32),
        np.array([-128.0, 63.0, 35.0], dtype=np.float32),
        np.array([-128.0, 0.0, 35.0], dtype=np.float32),
        np.array([-63.0, 0.0, 35.0], dtype=np.float32),
    ], 
        
    [
        np.array([80.0, 0.0, 35.0], dtype=np.float32),
        np.array([80.0, -65.0, 35.0], dtype=np.float32),
        np.array([105.0, -93.0, 42.0], dtype=np.float32),
        np.array([130.0, -65.0, 35.0], dtype=np.float32),
        np.array([130.0, 0.0, 35.0], dtype=np.float32),
        np.array([130.0, 63.0, 35.0], dtype=np.float32),
        np.array([127.0, 125.0, 35.0], dtype=np.float32),
        np.array([64.0, 128.0, 35.0], dtype=np.float32),
        np.array([50.0, 100.0, 45.0], dtype=np.float32),
        np.array([40.0, 20.0, 40.0], dtype=np.float32),
    ],
    
    [
        np.array([-63.0, -25.0, 45.0], dtype=np.float32),
        np.array([-110.0, -20.0, 45.0], dtype=np.float32),
        np.array([-128.0, -65.0, 35.0], dtype=np.float32),
        np.array([-125.0, -130.0, 35.0], dtype=np.float32),
        np.array([-63.0, -130.0, 35.0], dtype=np.float32),
        np.array([0.0, -128.0, 35.0], dtype=np.float32),
        np.array([80.0, -128.0, 35.0], dtype=np.float32),
        np.array([127.0, -127.0, 35.0], dtype=np.float32),
        np.array([97.0, -118.0, 40.0], dtype=np.float32),
        np.array([20.0, -95.0, 40.0], dtype=np.float32),
        np.array([0.0, -80.0, 35.0], dtype=np.float32),
        np.array([0.0, -63.0, 35.0], dtype=np.float32),
    ] 
]

waypoints["africa"] = [
                np.array([0.0, 0.0, 5.0], dtype=np.float32), # starting point
                np.array([204.0, 153.0, 5.0], dtype=np.float32),
                np.array([200.0, -101.0, 10.0], dtype=np.float32),
                np.array([120.0, 105.0, 20.0], dtype=np.float32),
                np.array([89.0, 40.0, 15.0], dtype=np.float32),
                np.array([-58.0, 155.0, 25.0], dtype=np.float32),
                np.array([34.0, 26.0, 5.0], dtype=np.float32),
                np.array([46.0, -60.0, 10.0], dtype=np.float32),
                np.array([-126.0, 222.0, 30.0], dtype=np.float32),
                np.array([-247.0, -227.0, 10.0], dtype=np.float32),
                np.array([13.0, -193.0, 5.0], dtype=np.float32),
                np.array([-41.0, -239.0, 15.0], dtype=np.float32),
                np.array([-192.0, 78.0, 30.0], dtype=np.float32),
                np.array([-115.0, -167.0, 15.0], dtype=np.float32),
                np.array([150.0, 211.0, 20.0], dtype=np.float32),
                np.array([-3.0, -89.0, 5.0], dtype=np.float32),
                np.array([-50.0, 80.0, 25.0], dtype=np.float32),
                np.array([135.0, -123.0, 10.0], dtype=np.float32),
                np.array([-133.0, -14.0, 30.0], dtype=np.float32),
                np.array([63.0, -219.0, 10.0], dtype=np.float32),
                np.array([-150.0, -94.0, 30.0], dtype=np.float32),
                np.array([113.0, -39.0, 10.0], dtype=np.float32),
                np.array([25.0, 79.0, 5.0], dtype=np.float32),
                np.array([-200.0, -87.0, 15.0], dtype=np.float32),
                np.array([-174.0, 161.0, 30.0], dtype=np.float32),
                np.array([219.0, 75.0, 15.0], dtype=np.float32),
                np.array([15.0, 15.0, 5.0], dtype=np.float32),
                np.array([103.0, -179.0, 10.0], dtype=np.float32),
                np.array([-123.0, -239.0, 15.0], dtype=np.float32),
                np.array([244.0, -165.0, 5.0], dtype=np.float32),
            ]

waypoints["mountains"] = [
                np.array([0.0, 0.0, 5.0], dtype=np.float32), # starting point
                np.array([288.0, 334.0, -30.0], dtype=np.float32),
                np.array([307.0, 126.0, 0.0], dtype=np.float32),
                np.array([120.0, 415.0, 15.0], dtype=np.float32),
                np.array([306.0, 385.0, -30.0], dtype=np.float32),
                np.array([89.0, 13.0, 20.0], dtype=np.float32),
                np.array([168.0, 490.0, 0.0], dtype=np.float32),
                np.array([38.0, 351.0, 15.0], dtype=np.float32),
                np.array([453.0, 416.0, -5.0], dtype=np.float32),
                np.array([396.0, 66.0, 15.0], dtype=np.float32),
                np.array([114.0, 280.0, 10.0], dtype=np.float32),
                np.array([122.0, 149.0, 10.0], dtype=np.float32),
                np.array([11.0, 158.0, 10.0], dtype=np.float32),
                np.array([11.0, 88.0, -5.0], dtype=np.float32),
                np.array([247.0, 70.0, 10.0], dtype=np.float32),
                np.array([397.0, 117.0, 10.0], dtype=np.float32),
                np.array([495.0, 230.0, 5.0], dtype=np.float32),
                np.array([69.0, 491.0, 15.0], dtype=np.float32),
                np.array([398.0, 266.0, 10.0], dtype=np.float32),
                np.array([184.0, 203.0, 10.0], dtype=np.float32),
                np.array([82.0, 104.0, 0.0], dtype=np.float32),
                np.array([479.0, 333.0, 5.0], dtype=np.float32),
                np.array([57.0, 271.0, 10.0], dtype=np.float32),
                np.array([356.0, 386.0, 10.0], dtype=np.float32),
                np.array([345.0, 466.0, -15.0], dtype=np.float32),
                np.array([386.0, 175.0, 10.0], dtype=np.float32),
                np.array([51.0, 213.0, 10.0], dtype=np.float32),
                np.array([218.0, 443.0, -20.0], dtype=np.float32),
                np.array([444.0, 39.0, 10.0], dtype=np.float32),
                np.array([396.0, 490.0, 0.0], dtype=np.float32),
                np.array([151.0, 343.0, 10.0], dtype=np.float32),
                np.array([112.0, 225.0, -25.0], dtype=np.float32),
                np.array([296.0, 245.0, 10.0], dtype=np.float32),
                np.array([254.0, 384.0, 5.0], dtype=np.float32),
                np.array([15.0, 238.0, 20.0], dtype=np.float32),
            ]

waypoints["adrl"] = [

            ]

waypoints["forest"] = [
    
            ]


def get_waypoints(type: str) -> list:
    return waypoints[type]

def get_obstacles(type: str) -> list:
    if type not in obstacles:
        return []
    return obstacles[type]

def get_specific_path(type: str) -> list:
    return specific_paths[type]